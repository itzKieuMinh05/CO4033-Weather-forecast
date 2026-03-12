import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    current_timestamp,
    round as spark_round,
    hour,
    dayofweek,
    month
)

# ==============================
# Spark Session
# ==============================
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
aws_region = os.getenv("AWS_REGION", "us-east-1")
minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")

spark = SparkSession.builder \
    .appName("BronzeToSilverWeather") \
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

# ==============================
# Create namespace
# ==============================

spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")

# ==============================
# Create silver table
# ==============================

spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg.silver.weather (
    time TIMESTAMP,
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
    hour INT,
    day_of_week INT,
    month INT,
    temp_range DOUBLE,
    update_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(time))
""")

print("Silver table ready")

# ==============================
# Read Bronze
# ==============================

bronze_df = spark.read.table("iceberg.weather.weather_bronze")

print("Bronze preview:")
bronze_df.show(5)

# ==============================
# Data Cleaning
# ==============================

df = bronze_df \
    .withColumn("temperature",
        when(col("temperature").isNull(), 0.0)
        .otherwise(spark_round(col("temperature"), 2))) \
    .withColumn("temp_min",
        when(col("temp_min").isNull(), 0.0)
        .otherwise(spark_round(col("temp_min"), 2))) \
    .withColumn("temp_max",
        when(col("temp_max").isNull(), 0.0)
        .otherwise(spark_round(col("temp_max"), 2))) \
    .withColumn("humidity",
        when(col("humidity").isNull(), 0.0)
        .otherwise(spark_round(col("humidity"), 2))) \
    .withColumn("pressure",
        when(col("pressure").isNull(), 0.0)
        .otherwise(spark_round(col("pressure"), 2))) \
    .withColumn("wind_speed",
        when(col("wind_speed").isNull(), 0.0)
        .otherwise(spark_round(col("wind_speed"), 2))) \
    .withColumn("precipitation",
        when(col("precipitation").isNull(), 0.0)
        .otherwise(spark_round(col("precipitation"), 2)))

# ==============================
# Feature Engineering
# ==============================

df = df \
    .withColumn("hour", hour("event_time")) \
    .withColumn("day_of_week", dayofweek("event_time")) \
    .withColumn("month", month("event_time")) \
    .withColumn("temp_range", col("temp_max") - col("temp_min")) \
    .withColumn("update_at", current_timestamp())

# ==============================
# Select columns
# ==============================

selected_columns = [
    "event_time",
    "province",
    "city",
    "temperature",
    "temp_min",
    "temp_max",
    "humidity",
    "feels_like",
    "visibility",
    "precipitation",
    "cloudcover",
    "wind_speed",
    "wind_gust",
    "wind_direction",
    "pressure",
    "is_day",
    "weather_code",
    "weather_main",
    "weather_description",
    "weather_icon",
    "hour",
    "day_of_week",
    "month",
    "temp_range",
    "update_at"
]

df = df.selectExpr(
    "event_time as time",
    *selected_columns[1:]
)

# ==============================
# Write to Silver
# ==============================

print("Writing to iceberg.silver.weather...")

df.writeTo("iceberg.silver.weather").append()

print("Silver transformation completed")

spark.stop()