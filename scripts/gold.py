import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

# ==============================
# Environment
# ==============================

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
aws_region = os.getenv("AWS_REGION", "us-east-1")
minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")
spark_master = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")

# ==============================
# Spark Session
# ==============================

spark = SparkSession.builder \
    .appName("WeatherRiskGoldLayer") \
    .master(spark_master) \
    .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.iceberg","org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type","rest") \
    .config("spark.sql.catalog.iceberg.uri","http://iceberg-rest:8181") \
    .config("spark.sql.catalog.iceberg.warehouse","s3://iceberg/warehouse") \
    .config("spark.sql.catalog.iceberg.io-impl","org.apache.iceberg.aws.s3.S3FileIO") \
    .config("spark.sql.catalog.iceberg.s3.endpoint",minio_endpoint) \
    .config("spark.sql.catalog.iceberg.s3.path-style-access","true") \
    .config("spark.sql.catalog.iceberg.s3.access-key-id",aws_access_key) \
    .config("spark.sql.catalog.iceberg.s3.secret-access-key",aws_secret_key) \
    .config("spark.sql.catalog.iceberg.s3.region",aws_region) \
    .config("spark.sql.defaultCatalog","iceberg") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ==============================
# Create namespace
# ==============================

spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.gold")

# ==============================
# Create gold table
# ==============================

spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg.gold.weather_risk (
    time TIMESTAMP,
    province STRING,
    city STRING,

    temperature DOUBLE,
    humidity DOUBLE,
    pressure DOUBLE,
    wind_speed DOUBLE,
    precipitation DOUBLE,
    cloudcover DOUBLE,

    risk_score DOUBLE,

    update_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(time))
""")

print("Gold table ready")

# ==============================
# Read Silver
# ==============================

silver_df = spark.read.table("iceberg.silver.weather")

print("Silver preview:")
silver_df.show(5)

# ==============================
# Weather Risk Score
# ==============================

gold_df = silver_df.withColumn(
    "risk_score",
      col("precipitation") * 0.4
    + col("wind_speed") * 0.3
    + col("cloudcover") * 0.2
    + col("humidity") * 0.1
).withColumn(
    "update_at",
    current_timestamp()
)

# ==============================
# Select columns
# ==============================

gold_df = gold_df.select(
    "time",
    "province",
    "city",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "precipitation",
    "cloudcover",
    "risk_score",
    "update_at"
)

# ==============================
# Write to Gold
# ==============================

print("Writing to iceberg.gold.weather_risk...")

gold_df.writeTo("iceberg.gold.weather_risk").append()

print("Gold layer created successfully")

spark.stop()
