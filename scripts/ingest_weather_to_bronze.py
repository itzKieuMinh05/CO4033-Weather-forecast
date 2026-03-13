import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_timestamp
from pyspark.sql.types import *

# ==============================
# Environment config
# ==============================

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
aws_region = os.getenv("AWS_REGION", "us-east-1")
minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")

# input_path = os.getenv("WEATHER_RAW_PATH", "data/weather-vn-5.csv")
input_path = os.getenv("WEATHER_RAW_PATH", "data/weather-vn-*.csv")


# ==============================
# Spark Session
# ==============================

spark = SparkSession.builder \
    .appName("WeatherCSVToBronze") \
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

# ==============================
# Create namespace if not exists
# ==============================

spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.weather")

# ==============================
# Create Bronze table if not exists
# ==============================

spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg.weather.weather_bronze (
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
    event_time TIMESTAMP,
    load_at TIMESTAMP
)
USING iceberg
""")

# ==============================
# Schema for CSV
# ==============================

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

# ==============================
# Read CSV from MinIO
# ==============================

df = spark.read \
    .schema(schema) \
    .option("header", True) \
    .csv(input_path)

# ==============================
# Add metadata columns
# ==============================

df = df.withColumn(
    "event_time",
    to_timestamp("time")
).withColumn(
    "load_at",
    current_timestamp()
)

print("Preview data:")
df.show(10, truncate=False)

# ==============================
# Append to Iceberg Bronze
# ==============================

df.writeTo("iceberg.weather.weather_bronze").append()

print("Ingestion completed: data appended to iceberg.weather.weather_bronze")