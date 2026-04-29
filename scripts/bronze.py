import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
aws_region = os.getenv("AWS_REGION", "us-east-1")
minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")
spark_master = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")

input_path = os.getenv("WEATHER_RAW_PATH", "data/weather_vn_cleaned.csv")
bronze_output_path = os.getenv(
    "BRONZE_WEATHER_PATH",
    "s3a://iceberg/bronze/weather_raw_parquet/"
)

spark = SparkSession.builder \
    .appName("WeatherToBronzeParquet") \
    .master(spark_master) \
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

spark.sparkContext.setLogLevel("ERROR")

raw_df = spark.read.option("header", True).csv(input_path)

bronze_df = raw_df \
    .withColumn("source_file", input_file_name()) \
    .withColumn("bronze_ingested_at", current_timestamp())

print("Bronze preview:")
bronze_df.show(10, truncate=False)

bronze_df.write.mode("append").parquet(bronze_output_path)

print(f"Bronze Parquet written to {bronze_output_path}")

spark.stop()
