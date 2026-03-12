import os
from pyspark.sql import SparkSession

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
aws_region = os.getenv("AWS_REGION", "us-east-1")
minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")

spark = SparkSession.builder \
    .appName("CreateWeatherBronze") \
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

spark.stop()
