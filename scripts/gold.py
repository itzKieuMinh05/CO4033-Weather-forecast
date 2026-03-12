from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    coalesce,
    lit,
    avg,
    min as spark_min,
    max as spark_max,
    sum as spark_sum,
    count,
    to_date,
    date_trunc,
    round as spark_round,
    current_timestamp,
)
from pyspark.storagelevel import StorageLevel

# Initialize SparkSession with Iceberg
spark = (
    SparkSession.builder
    .appName("GoldWeatherAggregation")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.iceberg.type", "rest")
    .config("spark.sql.catalog.iceberg.uri", "http://iceberg-rest:8181")
    .config("spark.sql.catalog.iceberg.table-default.format-version", "2")
    .config("spark.sql.defaultCatalog", "iceberg")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Create namespace
namespace = "gold"
spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")

# Gold table 1: hourly city-level aggregates
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS iceberg.gold.weather_hourly_city (
        hour_ts TIMESTAMP,
        province STRING,
        city STRING,
        avg_temperature DOUBLE,
        min_temperature DOUBLE,
        max_temperature DOUBLE,
        avg_humidity DOUBLE,
        avg_wind_speed DOUBLE,
        total_precipitation DOUBLE,
        record_count BIGINT,
        update_at TIMESTAMP
    ) USING iceberg
    PARTITIONED BY (days(hour_ts))
    """
)

# Gold table 2: daily province-level aggregates
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS iceberg.gold.weather_daily_province (
        day DATE,
        province STRING,
        avg_temperature DOUBLE,
        min_temperature DOUBLE,
        max_temperature DOUBLE,
        avg_humidity DOUBLE,
        avg_wind_speed DOUBLE,
        total_precipitation DOUBLE,
        record_count BIGINT,
        update_at TIMESTAMP
    ) USING iceberg
    PARTITIONED BY (days(day))
    """
)

print("✅ Gold tables are ready")

# Read stream from silver.weather
silver_df = (
    spark.readStream
    .format("iceberg")
    .table("iceberg.silver.weather")
)

def write_gold(batch_df, batch_id):
    print(f"🟡 Processing gold batch {batch_id}")
    if batch_df.isEmpty():
        print(f"ℹ️ Batch {batch_id}: empty")
        return

    batch_df.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        # Normalize keys
        base_df = (
            batch_df
            .filter(col("time").isNotNull())
            .withColumn("province", coalesce(col("province"), lit("UNKNOWN")))
            .withColumn("city", coalesce(col("city"), lit("UNKNOWN")))
        )

        # ---------------------------
        # 1) Hourly city aggregates
        # ---------------------------
        hourly_df = (
            base_df.groupBy(
                date_trunc("hour", col("time")).alias("hour_ts"),
                col("province"),
                col("city"),
            )
            .agg(
                spark_round(avg("temperature"), 2).alias("avg_temperature"),
                spark_round(spark_min("temp_min"), 2).alias("min_temperature"),
                spark_round(spark_max("temp_max"), 2).alias("max_temperature"),
                spark_round(avg("humidity"), 2).alias("avg_humidity"),
                spark_round(avg("wind_speed"), 2).alias("avg_wind_speed"),
                spark_round(spark_sum("precipitation"), 2).alias("total_precipitation"),
                count(lit(1)).alias("record_count"),
            )
            .withColumn("update_at", current_timestamp())
        )

        hourly_df.createOrReplaceTempView("v_hourly_weather")

        spark.sql(
            """
            MERGE INTO iceberg.gold.weather_hourly_city t
            USING v_hourly_weather s
            ON t.hour_ts = s.hour_ts
               AND t.province = s.province
               AND t.city = s.city
            WHEN MATCHED THEN UPDATE SET
                t.avg_temperature = s.avg_temperature,
                t.min_temperature = s.min_temperature,
                t.max_temperature = s.max_temperature,
                t.avg_humidity = s.avg_humidity,
                t.avg_wind_speed = s.avg_wind_speed,
                t.total_precipitation = s.total_precipitation,
                t.record_count = s.record_count,
                t.update_at = s.update_at
            WHEN NOT MATCHED THEN INSERT *
            """
        )

        # ------------------------------
        # 2) Daily province aggregates
        # ------------------------------
        daily_df = (
            base_df.groupBy(
                to_date(col("time")).alias("day"),
                col("province"),
            )
            .agg(
                spark_round(avg("temperature"), 2).alias("avg_temperature"),
                spark_round(spark_min("temp_min"), 2).alias("min_temperature"),
                spark_round(spark_max("temp_max"), 2).alias("max_temperature"),
                spark_round(avg("humidity"), 2).alias("avg_humidity"),
                spark_round(avg("wind_speed"), 2).alias("avg_wind_speed"),
                spark_round(spark_sum("precipitation"), 2).alias("total_precipitation"),
                count(lit(1)).alias("record_count"),
            )
            .withColumn("update_at", current_timestamp())
        )

        daily_df.createOrReplaceTempView("v_daily_weather")

        spark.sql(
            """
            MERGE INTO iceberg.gold.weather_daily_province t
            USING v_daily_weather s
            ON t.day = s.day
               AND t.province = s.province
            WHEN MATCHED THEN UPDATE SET
                t.avg_temperature = s.avg_temperature,
                t.min_temperature = s.min_temperature,
                t.max_temperature = s.max_temperature,
                t.avg_humidity = s.avg_humidity,
                t.avg_wind_speed = s.avg_wind_speed,
                t.total_precipitation = s.total_precipitation,
                t.record_count = s.record_count,
                t.update_at = s.update_at
            WHEN NOT MATCHED THEN INSERT *
            """
        )

        print(f"✅ Batch {batch_id}: gold upsert done")

    finally:
        batch_df.unpersist()

query = (
    silver_df.writeStream
    .foreachBatch(write_gold)
    .outputMode("append")
    .option("checkpointLocation", "s3a://lake/gold/weather/_checkpoints")
    .start()
)

print("🚀 Gold streaming started. Awaiting termination...")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\n⚠️ Streaming stopped by user")
    query.stop()
finally:
    spark.stop()
    print("🛑 Spark session stopped")