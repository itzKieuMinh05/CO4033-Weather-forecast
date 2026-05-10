import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    dayofmonth,
    dayofweek,
    hour,
    lit,
    lower,
    month,
    radians,
    round as spark_round,
    sin,
    cos,
    to_timestamp,
    trim,
    when,
)

# ==============================
# Environment
# ==============================

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
aws_region = os.getenv("AWS_REGION", "us-east-1")
minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")
spark_master = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
bronze_input_path = os.getenv(
    "BRONZE_WEATHER_PATH",
    "s3a://iceberg/bronze/weather_raw_parquet/"
)
from pyspark.sql.window import Window

# ==============================
# Spark Session
# ==============================

spark = SparkSession.builder \
    .appName("BronzeParquetToSilverWeather") \
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

# ==============================
# Create namespace and table
# ==============================

spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")

spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg.silver.weather (
    time TIMESTAMP,
    province STRING,
    region STRING,
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
    day INT,
    month INT,
    weekday INT,
    temp_range DOUBLE,
    wind_dir_sin DOUBLE,
    wind_dir_cos DOUBLE,
    rain INT,
    extreme STRING,
    temp_level STRING,
    humidity_level STRING,
    pressure_level STRING,
    wind_level STRING,
    temp_lag_1 DOUBLE,
    humidity_lag_1 DOUBLE,
    pressure_lag_1 DOUBLE,
    update_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(time))
""")

print("Silver table ready")

# ==============================
# Read Bronze Parquet
# ==============================

bronze_df = spark.read.parquet(bronze_input_path)

print("Bronze Parquet preview:")
bronze_df.show(5, truncate=False)

# ==============================
# Transform to tabular schema
# ==============================

df = bronze_df

if "region" not in df.columns and "province" in df.columns:
    df = df.withColumn("region", col("province"))

if "province" not in df.columns and "region" in df.columns:
    df = df.withColumn("province", col("region"))

df = df.withColumn("time", to_timestamp(col("time")))

for column_name in [
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
]:
    if column_name in df.columns:
        df = df.withColumn(column_name, spark_round(col(column_name).cast("double"), 2))

if "is_day" in df.columns:
    df = df.withColumn(
        "is_day",
        when(lower(trim(col("is_day"))).isin("1", "true", "t", "yes"), lit(True))
        .when(lower(trim(col("is_day"))).isin("0", "false", "f", "no"), lit(False))
        .otherwise(col("is_day").cast("boolean"))
    )
else:
    df = df.withColumn("is_day", lit(None).cast("boolean"))

if "weather_code" in df.columns:
    df = df.withColumn("weather_code", col("weather_code").cast("int"))
else:
    df = df.withColumn("weather_code", lit(None).cast("int"))

for column_name in ["hour", "day", "month", "weekday", "rain"]:
    if column_name in df.columns:
        df = df.withColumn(column_name, col(column_name).cast("int"))

for column_name in [
    "temp_range",
    "wind_dir_sin",
    "wind_dir_cos",
    "temp_lag_1",
    "humidity_lag_1",
    "pressure_lag_1",
]:
    if column_name in df.columns:
        df = df.withColumn(column_name, spark_round(col(column_name).cast("double"), 4))

if "hour" in df.columns:
    df = df.withColumn("hour", when(col("hour").isNull(), hour("time")).otherwise(col("hour")))
else:
    df = df.withColumn("hour", hour("time"))

if "day" in df.columns:
    df = df.withColumn("day", when(col("day").isNull(), dayofmonth("time")).otherwise(col("day")))
else:
    df = df.withColumn("day", dayofmonth("time"))

if "month" in df.columns:
    df = df.withColumn("month", when(col("month").isNull(), month("time")).otherwise(col("month")))
else:
    df = df.withColumn("month", month("time"))

if "weekday" in df.columns:
    df = df.withColumn(
        "weekday",
        when(col("weekday").isNull(), (dayofweek("time") + lit(5)) % lit(7)).otherwise(col("weekday"))
    )
else:
    df = df.withColumn("weekday", (dayofweek("time") + lit(5)) % lit(7))

if "temp_max" in df.columns and "temp_min" in df.columns:
    if "temp_range" in df.columns:
        df = df.withColumn(
            "temp_range",
            when(col("temp_range").isNull(), col("temp_max") - col("temp_min")).otherwise(col("temp_range"))
        )
    else:
        df = df.withColumn("temp_range", col("temp_max") - col("temp_min"))
else:
    df = df.withColumn("temp_range", lit(None).cast("double"))

if "wind_direction" in df.columns:
    if "wind_dir_sin" in df.columns:
        df = df.withColumn(
            "wind_dir_sin",
            when(col("wind_dir_sin").isNull(), spark_round(sin(radians(col("wind_direction"))), 4))
            .otherwise(col("wind_dir_sin"))
        )
    else:
        df = df.withColumn("wind_dir_sin", spark_round(sin(radians(col("wind_direction"))), 4))

    if "wind_dir_cos" in df.columns:
        df = df.withColumn(
            "wind_dir_cos",
            when(col("wind_dir_cos").isNull(), spark_round(cos(radians(col("wind_direction"))), 4))
            .otherwise(col("wind_dir_cos"))
        )
    else:
        df = df.withColumn("wind_dir_cos", spark_round(cos(radians(col("wind_direction"))), 4))
else:
    df = df.withColumn("wind_dir_sin", lit(None).cast("double"))
    df = df.withColumn("wind_dir_cos", lit(None).cast("double"))

if "rain" not in df.columns:
    if "precipitation" in df.columns:
        df = df.withColumn("rain", when(col("precipitation") > 0, lit(1)).otherwise(lit(0)))
    else:
        df = df.withColumn("rain", lit(0))
else:
    df = df.withColumn(
        "rain",
        when(col("rain").isNull(), when(col("precipitation") > 0, lit(1)).otherwise(lit(0))).otherwise(col("rain"))
    )

if "extreme" not in df.columns:
    df = df.withColumn(
        "extreme",
        when(col("temp_max") > 35, lit("heatwave"))
        .when(col("rain") > 0, lit("rain"))
        .when(col("wind_speed") > 40, lit("storm"))
        .otherwise(lit("normal"))
    )

if "temp_level" not in df.columns:
    df = df.withColumn(
        "temp_level",
        when(col("temp_max") < 20, lit("temp_low"))
        .when(col("temp_max") < 30, lit("temp_medium"))
        .otherwise(lit("temp_high"))
    )

if "humidity_level" not in df.columns:
    df = df.withColumn(
        "humidity_level",
        when(col("humidity") < 60, lit("humidity_low"))
        .when(col("humidity") < 80, lit("humidity_medium"))
        .otherwise(lit("humidity_high"))
    )

if "pressure_level" not in df.columns:
    df = df.withColumn(
        "pressure_level",
        when(col("pressure") < 1000, lit("pressure_low"))
        .when(col("pressure") < 1015, lit("pressure_normal"))
        .otherwise(lit("pressure_high"))
    )

if "wind_level" not in df.columns:
    df = df.withColumn(
        "wind_level",
        when(col("wind_speed") < 10, lit("wind_low"))
        .when(col("wind_speed") < 25, lit("wind_medium"))
        .otherwise(lit("wind_high"))
    )

if "temp_lag_1" not in df.columns:
    df = df.withColumn("temp_lag_1", lit(None).cast("double"))

if "humidity_lag_1" not in df.columns:
    df = df.withColumn("humidity_lag_1", lit(None).cast("double"))

if "pressure_lag_1" not in df.columns:
    df = df.withColumn("pressure_lag_1", lit(None).cast("double"))

df = df.withColumn("update_at", current_timestamp())

selected_columns = [
    "time",
    "province",
    "region",
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
    "day",
    "month",
    "weekday",
    "temp_range",
    "wind_dir_sin",
    "wind_dir_cos",
    "rain",
    "extreme",
    "temp_level",
    "humidity_level",
    "pressure_level",
    "wind_level",
    "temp_lag_1",
    "humidity_lag_1",
    "pressure_lag_1",
    "update_at",
]

df = df.select(*selected_columns)

print("Silver preview:")
df.show(10, truncate=False)

print("Writing to iceberg.silver.weather...")
df.writeTo("iceberg.silver.weather").append()

print("Silver transformation completed")

spark.stop()
