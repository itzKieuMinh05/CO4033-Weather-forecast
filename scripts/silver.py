import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    current_timestamp,
    round as spark_round,
    hour,
    dayofweek,
    dayofmonth,
    month,
    to_date,
    sin,
    cos,
    radians,
    expr,
    lit,
    lag,
    max as spark_max,
    min as spark_min
)
from pyspark.sql.window import Window

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
    day INT,
    month INT,
    weekday INT,
    day_of_week INT,
    region STRING,
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

north = [
    "Bac Giang","Bac Kan","Bac Ninh","Tu Son","Cao Bang",
    "Dien Bien Phu","Ha Giang","Ha Noi","Hai Duong","Hai Phong",
    "Hoa Binh","Hung Yen","My Hao","Lai Chau","Lang Son",
    "Lao Cai","Nam Dinh","Ninh Binh","Tam Diep","Viet Tri",
    "Cam Pha","Ha Long","Mong Cai","Uong Bi","Son La",
    "Song Cong","Thai Nguyen","Tuyen Quang","Phuc Yen",
    "Vinh Yen","Yen Bai"
]

central = [
    "Quy Nhon","Phan Thiet","Da Nang","Buon Ma Thuot","Gia Nghia",
    "Pleiku","Ha Tinh","Hong Linh","Cam Ranh","Nha Trang",
    "Kon Tum","Phan Rang - Thap Cham","Tuy Hoa","Dong Hoi",
    "Hoi An","Tam Ky","Quang Ngai","Dong Ha","Thai Hoa",
    "Vinh","Sam Son","Thanh Hoa"
]

south = [
    "Chau Doc","Long Xuyen","Ba Ria","Vung Tau","Bac Lieu",
    "Ben Tre","Di An","Thu Dau Mot","Thuan An","Dong Xoai",
    "Ca Mau","Can Tho","Bien Hoa","Long Khanh","Nga Bay",
    "Vi Thanh","Ha Tien","Rach Gia","Tan An","Soc Trang",
    "Ho Chi Minh","Tay Ninh","My Tho","Tra Vinh","Vinh Long"
]

w_city_time = Window.partitionBy("city").orderBy("event_time")
w_city_day = Window.partitionBy("city", to_date(col("event_time")))

df = df \
    .withColumn("hour", hour("event_time")) \
    .withColumn("day", dayofmonth("event_time")) \
    .withColumn("month", month("event_time")) \
    .withColumn("day_of_week", dayofweek("event_time")) \
    .withColumn("weekday", expr("pmod(dayofweek(event_time) + 5, 7)")) \
    .withColumn(
        "region",
        when(col("city").isin(north), lit("north"))
        .when(col("city").isin(central), lit("central"))
        .when(col("city").isin(south), lit("south"))
        .otherwise(lit("unknown"))
    ) \
    .withColumn("wind_dir_sin", spark_round(sin(radians(col("wind_direction"))), 4)) \
    .withColumn("wind_dir_cos", spark_round(cos(radians(col("wind_direction"))), 4)) \
    .withColumn("rain", when(col("precipitation") > 0, 1).otherwise(0)) \
    .withColumn(
        "extreme",
        when(col("temp_max") > 35, "heatwave")
        .when(col("precipitation") > 50, "heavy_rain")
        .when(col("wind_speed") > 40, "storm")
        .when(col("precipitation") > 0, "rain")
        .otherwise("normal")
    ) \
    .withColumn(
        "temp_level",
        when(col("temp_max") <= 20, "temp_low")
        .when((col("temp_max") > 20) & (col("temp_max") <= 30), "temp_medium")
        .otherwise("temp_high")
    ) \
    .withColumn(
        "humidity_level",
        when(col("humidity") <= 60, "humidity_low")
        .when((col("humidity") > 60) & (col("humidity") <= 80), "humidity_medium")
        .otherwise("humidity_high")
    ) \
    .withColumn(
        "pressure_level",
        when(col("pressure") <= 1000, "pressure_low")
        .when((col("pressure") > 1000) & (col("pressure") <= 1015), "pressure_normal")
        .otherwise("pressure_high")
    ) \
    .withColumn(
        "wind_level",
        when(col("wind_speed") <= 10, "wind_low")
        .when((col("wind_speed") > 10) & (col("wind_speed") <= 25), "wind_medium")
        .otherwise("wind_high")
    ) \
    .withColumn("temp_lag_1", lag("temp_max", 1).over(w_city_time)) \
    .withColumn("humidity_lag_1", lag("humidity", 1).over(w_city_time)) \
    .withColumn("pressure_lag_1", lag("pressure", 1).over(w_city_time)) \
    .withColumn(
        "temp_range",
        spark_max("temp_max").over(w_city_day) - spark_min("temp_min").over(w_city_day)
    ) \
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
    "day",
    "month",
    "weekday",
    "day_of_week",
    "region",
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