import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import *

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
aws_region = os.getenv("AWS_REGION", "us-east-1")
minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")

spark = SparkSession.builder \
    .appName("WeatherConsumerKafkaToBronze") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "rest") \
    .config("spark.sql.catalog.iceberg.uri", "http://iceberg-rest:8181") \
    .config("spark.sql.catalog.iceberg.warehouse", "s3://iceberg/warehouse") \
    .config("spark.sql.catalog.iceberg.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
    .config("spark.sql.catalog.iceberg.s3.endpoint", minio_endpoint) \
    .config("spark.sql.catalog.iceberg.s3.path-style-access", "true") \
    .config("spark.sql.catalog.iceberg.s3.access-key-id", aws_access_key) \
    .config("spark.sql.catalog.iceberg.s3.secret-access-key", aws_secret_key) \
    .config("spark.sql.catalog.iceberg.s3.region", aws_region) \
    .config("spark.sql.defaultCatalog", "iceberg") \
    .getOrCreate()

spark.sql("CREATE NAMESPACE IF NOT EXISTS bronze")
spark.sql("""
CREATE TABLE IF NOT EXISTS bronze.weather (
    time STRING,
    province STRING,
    city STRING,
    temperature DOUBLE,
    temp_min DOUBLE,
    temp_max DOUBLE,
    humidity DOUBLE,
    feels_like DOUBLE,
    visibility DOUBLE,
    precipitation DOUBLE,
    cloudcover DOUBLE,
    wind_speed DOUBLE,
    wind_gust DOUBLE,
    wind_direction DOUBLE,
    pressure DOUBLE,
    is_day BOOLEAN,
    weather_code INT,
    weather_main STRING,
    weather_description STRING,
    weather_icon STRING,
    kafka_time TIMESTAMP,
    load_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (province)
""")

schema = StructType([
    StructField("time", StringType()),
    StructField("province", StringType()),
    StructField("city", StringType()),
    StructField("temperature", DoubleType()),
    StructField("temp_min", DoubleType()),
    StructField("temp_max", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("feels_like", DoubleType()),
    StructField("visibility", DoubleType()),
    StructField("precipitation", DoubleType()),
    StructField("cloudcover", DoubleType()),
    StructField("wind_speed", DoubleType()),
    StructField("wind_gust", DoubleType()),
    StructField("wind_direction", DoubleType()),
    StructField("pressure", DoubleType()),
    StructField("is_day", BooleanType()),
    StructField("weather_code", IntegerType()),
    StructField("weather_main", StringType()),
    StructField("weather_description", StringType()),
    StructField("weather_icon", StringType())
])

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "biopharma-kafka:29092")) \
    .option("subscribe", "weather_raw") \
    .option("startingOffsets", "earliest") \
    .load()

bronze_df = kafka_df.select(
    from_json(col("value").cast("string"), schema).alias("data"),
    col("timestamp").alias("kafka_time")
).select("data.*", "kafka_time") \
 .withColumn("load_at", current_timestamp())

def write_and_preview(batch_df, batch_id):
    if batch_df.isEmpty():
        print(f"Batch {batch_id}: no data")
        return

    print(f"\nBatch {batch_id}: top 10 rows in this batch")
    batch_df.orderBy(col("kafka_time").asc()).limit(10).show(truncate=False)

    batch_df.writeTo("iceberg.bronze.weather").append()


bronze_df.writeStream \
    .foreachBatch(write_and_preview) \
    .option("checkpointLocation", "s3a://iceberg/checkpoints/weather_bronze") \
    .start() \
    .awaitTermination()
