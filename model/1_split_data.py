import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("--- BƯỚC 1: ĐỌC DỮ LIỆU TỪ MINIO LAKEHOUSE VÀ CHIA TẬP ---")

# Config (có thể override bằng biến môi trường)
AWS_KEY     = os.getenv("AWS_ACCESS_KEY_ID",     "minioadmin")
AWS_SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
MINIO_EP    = os.getenv("S3_ENDPOINT",           "http://minio:9000")
BRONZE_PATH = os.getenv("BRONZE_WEATHER_PATH",   "s3a://iceberg/bronze/weather_raw_parquet/")
TRAIN_DEST   = os.getenv("TRAIN_SPLIT_PATH",     "s3a://iceberg/train/train_split/")
TEST_DEST    = os.getenv("TEST_SPLIT_PATH",      "s3a://iceberg/train/test_split/")


def build_spark(app_name):
	return (
		SparkSession.builder
		.appName(app_name)
		.config("spark.hadoop.fs.s3a.endpoint", MINIO_EP)
		.config("spark.hadoop.fs.s3a.access.key", AWS_KEY)
		.config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET)
		.config("spark.hadoop.fs.s3a.path.style.access", "true")
		.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
		.config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
		.config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
		.config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "4g"))
		.config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
		.config("spark.default.parallelism", os.getenv("SPARK_DEFAULT_PARALLELISM", "8"))
		.getOrCreate()
	)


def spark_stratified_split(sdf, strat_cols, train_ratio=0.8, seed=42):
	work_df = sdf
	key_col = "__split_key"
	if strat_cols:
		key_exprs = [F.coalesce(F.col(col_name).cast("string"), F.lit("__NULL__")) for col_name in strat_cols]
		work_df = work_df.withColumn(key_col, F.concat_ws("||", *key_exprs))
	else:
		work_df = work_df.withColumn(key_col, F.lit("__all__"))

	counts = work_df.groupBy(key_col).count()
	window_spec = Window.partitionBy(key_col).orderBy(F.rand(seed))
	with_rank = work_df.withColumn("__rn", F.row_number().over(window_spec)).join(counts, on=key_col, how="left")
	train_cutoff = F.when(
		F.col("count") <= 1,
		F.col("count")
	).otherwise(
		F.greatest(
			F.lit(1),
			F.least(F.col("count") - F.lit(1), F.floor(F.col("count") * F.lit(train_ratio)))
		)
	)
	with_rank = with_rank.withColumn("__train_cutoff", train_cutoff)

	train_df = (
		with_rank
		.where(F.col("__rn") <= F.col("__train_cutoff"))
		.drop("__rn", "count", "__train_cutoff", key_col)
	)
	test_df = (
		with_rank
		.where(F.col("__rn") > F.col("__train_cutoff"))
		.drop("__rn", "count", "__train_cutoff", key_col)
	)
	return train_df, test_df


def cast_model_columns(sdf):
	numeric_cast_columns = [
		"temperature", "temp_min", "temp_max", "humidity", "feels_like", "visibility",
		"precipitation", "cloudcover", "wind_speed", "wind_gust", "wind_direction",
		"pressure", "is_day", "hour", "day", "month", "weekday", "temp_range",
		"wind_dir_sin", "wind_dir_cos", "rain", "weather_code", "temp_level",
		"humidity_level", "pressure_level", "wind_level", "temp_lag_1", "humidity_lag_1",
		"pressure_lag_1",
	]
	for column_name in numeric_cast_columns:
		if column_name in sdf.columns:
			sdf = sdf.withColumn(column_name, F.col(column_name).cast("double"))
	return sdf

print(f"📦 Thử đọc từ MinIO: {BRONZE_PATH}")
try:
	spark = build_spark("WeatherVN-DataSplit")
	spark.sparkContext.setLogLevel("ERROR")
	sdf = spark.read.parquet(BRONZE_PATH)
	sdf = cast_model_columns(sdf)
	row_count = sdf.count()
	print(f"✓ Đọc thành công từ MinIO, {row_count} dòng")
	use_spark_split = True
except Exception as e:
	print(f"⚠️ Không thể đọc từ MinIO ({e}). Fallback sang CSV cục bộ.")
	# fallback paths
	candidates = [
		os.path.join("..", "data", "weather_vn_cleaned.csv"),
		os.path.join("..", "..", "data", "weather_vn_cleaned.csv"),
		os.path.join("..", "weather_vn_cleaned.csv"),
		os.path.join("..", "..", "weather_vn_cleaned.csv"),
		os.path.join("..", "..", "..", "data", "weather_vn_cleaned.csv"),
		os.path.join("..", "data", "weather_vn_cleaned.csv"),
		os.path.join(os.getcwd(), "data", "weather_vn_cleaned.csv"),
	]
	found = False
	for c in candidates:
		if os.path.exists(c):
			df = pd.read_csv(c)
			print(f"✓ Đọc CSV cục bộ: {c} — {len(df)} dòng")
			use_spark_split = False
			found = True
			break
	if not found:
		raise FileNotFoundError("Không tìm thấy file dữ liệu ở MinIO hay các đường dẫn CSV dự phòng.")

if use_spark_split:
	print("Chia train/test (80/20) trên Spark — stratify theo 'extreme' + 'rain' nếu có")
	strat_cols = []
	if "extreme" in sdf.columns:
		strat_cols.append("extreme")
	if "rain" in sdf.columns:
		strat_cols.append("rain")

	train_df, test_df = spark_stratified_split(sdf, strat_cols=strat_cols, train_ratio=0.8, seed=42)

	try:
		train_df.write.mode("overwrite").parquet(TRAIN_DEST)
		test_df.write.mode("overwrite").parquet(TEST_DEST)
		print(f"✓ Ghi parquet lên MinIO: {TRAIN_DEST}, {TEST_DEST}")
	except Exception as e:
		print(f"⚠️ Không ghi được parquet lên MinIO ({e}).")

	print("✓ Split hoàn tất bằng Spark mà không cần kéo dữ liệu về pandas.")
else:
	# Basic cleaning only for local CSV fallback
	df = df.copy()
	df = df.fillna(df.select_dtypes(include=[np.number]).mean())
	print("Chia train/test (80/20) — stratify theo 'extreme' + 'rain' nếu có")
	strat_cols = []
	if "extreme" in df.columns:
		strat_cols.append("extreme")
	if "rain" in df.columns:
		strat_cols.append("rain")

	from sklearn.model_selection import train_test_split
	if strat_cols:
		train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[strat_cols])
	else:
		train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

	os.makedirs(os.path.join(os.getcwd(), "train_test_data"), exist_ok=True)
	train_path = os.path.join("train_test_data", "train_data.csv")
	test_path = os.path.join("train_test_data", "test_data.csv")
	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)
	print(f"✓ Lưu CSV: {train_path}, {test_path}")

print("\n✅ HOÀN THÀNH BƯỚC 1")
try:
	spark.stop()
except:
	pass