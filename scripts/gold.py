import os
from pyspark.sql import SparkSession, Row
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
spark.sql("""drop table if exists iceberg.gold.weather_risk; """)
spark.sql("""
CREATE TABLE IF NOT EXISTS iceberg.gold.weather_risk (
    time TIMESTAMP,
    province STRING,
    city STRING,

    latn DOUBLE,
    lotn DOUBLE,

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
# Province/City lat-lon lookup
# ==============================

location_lookup = [
    Row(province="An Giang",          city="Long Xuyen",     latn=10.3861, lotn=105.4354),
    Row(province="Ba Ria Vung Tau",   city="Vung Tau",       latn=10.3460, lotn=107.0843),
    Row(province="Bac Giang",         city="Bac Giang",      latn=21.2731, lotn=106.1946),
    Row(province="Bac Kan",           city="Bac Kan",        latn=22.1473, lotn=105.8348),
    Row(province="Bac Lieu",          city="Bac Lieu",       latn=9.2940,  lotn=105.7216),
    Row(province="Bac Ninh",          city="Bac Ninh",       latn=21.1861, lotn=106.0763),
    Row(province="Ben Tre",           city="Ben Tre",        latn=10.2434, lotn=106.3756),
    Row(province="Binh Dinh",         city="Quy Nhon",       latn=13.7765, lotn=109.2237),
    Row(province="Binh Duong",        city="Thu Dau Mot",    latn=10.9804, lotn=106.6519),
    Row(province="Binh Phuoc",        city="Dong Xoai",      latn=11.5353, lotn=106.8992),
    Row(province="Binh Thuan",        city="Phan Thiet",     latn=10.9282, lotn=108.1009),
    Row(province="Ca Mau",            city="Ca Mau",         latn=9.1769,  lotn=105.1524),
    Row(province="Can Tho",           city="Can Tho",        latn=10.0452, lotn=105.7469),
    Row(province="Cao Bang",          city="Cao Bang",       latn=22.6657, lotn=106.2522),
    Row(province="Da Nang",           city="Da Nang",        latn=16.0544, lotn=108.2022),
    Row(province="Dak Lak",           city="Buon Ma Thuot",  latn=12.6667, lotn=108.0500),
    Row(province="Dak Nong",          city="Gia Nghia",      latn=11.9772, lotn=107.6908),
    Row(province="Dien Bien",         city="Dien Bien Phu",  latn=21.3856, lotn=103.0230),
    Row(province="Dong Nai",          city="Bien Hoa",       latn=10.9574, lotn=106.8426),
    Row(province="Dong Thap",         city="Cao Lanh",       latn=10.4593, lotn=105.6340),
    Row(province="Gia Lai",           city="Pleiku",         latn=13.9833, lotn=108.0000),
    Row(province="Ha Giang",          city="Ha Giang",       latn=22.8233, lotn=104.9836),
    Row(province="Ha Nam",            city="Phu Ly",         latn=20.5395, lotn=105.9136),
    Row(province="Ha Noi",            city="Ha Noi",         latn=21.0285, lotn=105.8542),
    Row(province="Ha Tinh",           city="Ha Tinh",        latn=18.3560, lotn=105.8877),
    Row(province="Hai Duong",         city="Hai Duong",      latn=20.9373, lotn=106.3145),
    Row(province="Hai Phong",         city="Hai Phong",      latn=20.8449, lotn=106.6881),
    Row(province="Hau Giang",         city="Vi Thanh",       latn=9.7833,  lotn=105.4667),
    Row(province="Ho Chi Minh",       city="Ho Chi Minh",    latn=10.8231, lotn=106.6297),
    Row(province="Hoa Binh",          city="Hoa Binh",       latn=20.8133, lotn=105.3383),
    Row(province="Hung Yen",          city="Hung Yen",       latn=20.6464, lotn=106.0511),
    Row(province="Khanh Hoa",         city="Nha Trang",      latn=12.2388, lotn=109.1967),
    Row(province="Kien Giang",        city="Rach Gia",       latn=10.0128, lotn=105.0809),
    Row(province="Kon Tum",           city="Kon Tum",        latn=14.3497, lotn=108.0005),
    Row(province="Lai Chau",          city="Lai Chau",       latn=22.3964, lotn=103.4580),
    Row(province="Lam Dong",          city="Da Lat",         latn=11.9404, lotn=108.4583),
    Row(province="Lang Son",          city="Lang Son",       latn=21.8537, lotn=106.7615),
    Row(province="Lao Cai",           city="Lao Cai",        latn=22.4809, lotn=103.9754),
    Row(province="Long An",           city="Tan An",         latn=10.5354, lotn=106.4128),
    Row(province="Nam Dinh",          city="Nam Dinh",       latn=20.4200, lotn=106.1683),
    Row(province="Nghe An",           city="Vinh",           latn=18.6796, lotn=105.6813),
    Row(province="Ninh Binh",         city="Ninh Binh",      latn=20.2506, lotn=105.9745),
    Row(province="Ninh Thuan",        city="Phan Rang",      latn=11.5639, lotn=108.9886),
    Row(province="Phu Tho",           city="Viet Tri",       latn=21.3227, lotn=105.4023),
    Row(province="Phu Yen",           city="Tuy Hoa",        latn=13.0956, lotn=109.3183),
    Row(province="Quang Binh",        city="Dong Hoi",       latn=17.4800, lotn=106.5997),
    Row(province="Quang Nam",         city="Tam Ky",         latn=15.5736, lotn=108.4739),
    Row(province="Quang Ngai",        city="Quang Ngai",     latn=15.1194, lotn=108.8044),
    Row(province="Quang Ninh",        city="Ha Long",        latn=20.9517, lotn=107.0852),
    Row(province="Quang Tri",         city="Dong Ha",        latn=16.8163, lotn=107.0997),
    Row(province="Soc Trang",         city="Soc Trang",      latn=9.6025,  lotn=105.9739),
    Row(province="Son La",            city="Son La",         latn=21.3256, lotn=103.9188),
    Row(province="Tay Ninh",          city="Tay Ninh",       latn=11.3100, lotn=106.0980),
    Row(province="Thai Binh",         city="Thai Binh",      latn=20.4500, lotn=106.3417),
    Row(province="Thai Nguyen",       city="Thai Nguyen",    latn=21.5942, lotn=105.8480),
    Row(province="Thanh Hoa",         city="Thanh Hoa",      latn=19.8067, lotn=105.7852),
    Row(province="Thua Thien Hue",    city="Hue",            latn=16.4637, lotn=107.5909),
    Row(province="Tien Giang",        city="My Tho",         latn=10.3600, lotn=106.3600),
    Row(province="Tra Vinh",          city="Tra Vinh",       latn=9.9346,  lotn=106.3456),
    Row(province="Tuyen Quang",       city="Tuyen Quang",    latn=21.8236, lotn=105.2133),
    Row(province="Vinh Long",         city="Vinh Long",      latn=10.2397, lotn=105.9571),
    Row(province="Vinh Phuc",         city="Vinh Yen",       latn=21.3089, lotn=105.5978),
    Row(province="Yen Bai",           city="Yen Bai",        latn=21.7051, lotn=104.8753),
]

lookup_df = spark.createDataFrame(location_lookup)

# ==============================
# Read Silver
# ==============================

silver_df = spark.read.table("iceberg.silver.weather")

print("Silver preview:")
silver_df.show(5)

# ==============================
# Join lat/lon lookup
# ==============================

silver_df = silver_df.join(lookup_df, on=["province", "city"], how="left")

# ==============================
# Weather Risk Score
# ==============================

gold_df = silver_df \
.withColumn("rain_norm", col("precipitation") / 50) \
.withColumn("wind_norm", col("wind_speed") / 20) \
.withColumn("cloud_norm", col("cloudcover") / 100) \
.withColumn("humidity_norm", (col("humidity") - 40) / 60) \
.withColumn(
    "risk_score",
      col("rain_norm") * 0.35
    + col("wind_norm") * 0.30
    + col("cloud_norm") * 0.20
    + col("humidity_norm") * 0.15
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
    "latn",
    "lotn",
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
