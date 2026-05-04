"""Shared helpers for the WeatherVN Streamlit app."""

import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Spark / MinIO config
# ─────────────────────────────────────────────
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MINIO_EP = os.getenv("S3_ENDPOINT", "http://minio:9000")
BRONZE_PATH = os.getenv("BRONZE_WEATHER_PATH", "s3a://iceberg/bronze/weather_raw_parquet/")
SPARK_MASTER = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "forecasting-streamlit")

# ─────────────────────────────────────────────
# Feature schema used by the trained models
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "humidity",
    "visibility",
    "cloudcover",
    "wind_direction",
    "pressure",
    "is_day",
    "hour",
    "day",
    "month",
    "weekday",
    "wind_dir_sin",
    "wind_dir_cos",
    "temp_lag_1",
    "humidity_lag_1",
    "pressure_lag_1",
]

FEATURE_DEFAULTS = {
    "temperature": 28.0,
    "temp_min": 24.0,
    "temp_max": 32.0,
    "humidity": 80.0,
    "visibility": 10.0,
    "cloudcover": 50.0,
    "wind_direction": 0.0,
    "pressure": 1013.0,
    "is_day": 1,
    "hour": 12,
    "day": 15,
    "month": 6,
    "weekday": 3,
    "temp_lag_1": 28.0,
    "humidity_lag_1": 80.0,
    "pressure_lag_1": 1013.0,
}

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
GLOBAL_CSS = """
<style>
.stApp { background:#F4F6FB; }
.kpi-card {
    background:#fff;
    padding:16px;
    border-radius:12px;
    box-shadow:0 2px 6px rgba(0,0,0,0.05);
    border-top:4px solid #4F6EF7;
}
</style>
"""


def _is_numeric_spark_type(data_type):
    from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType

    return isinstance(data_type, (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType))


@st.cache_resource(show_spinner="🔌 Kết nối Spark...")
def get_spark():
    existing = SparkSession.getActiveSession()
    if existing:
        return existing

    return (
        SparkSession.builder
        .appName("WeatherVN-Streamlit")
        .master(SPARK_MASTER)
        .config("spark.driver.host", DRIVER_HOST)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg.type", "rest")
        .config("spark.sql.catalog.iceberg.uri", "http://iceberg-rest:8181")
        .config("spark.sql.catalog.iceberg.warehouse", "s3://iceberg/warehouse")
        .config("spark.sql.catalog.iceberg.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config("spark.sql.catalog.iceberg.s3.endpoint", MINIO_EP)
        .config("spark.sql.catalog.iceberg.s3.path-style-access", "true")
        .config("spark.sql.catalog.iceberg.s3.access-key-id", AWS_KEY)
        .config("spark.sql.catalog.iceberg.s3.secret-access-key", AWS_SECRET)
        .config("spark.sql.catalog.iceberg.s3.region", AWS_REGION)
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_EP)
        .config("spark.hadoop.fs.s3a.access.key", AWS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


@st.cache_data(ttl=300, show_spinner="📦 Đọc dữ liệu từ MinIO...")
def load_spark_df_cached(path: str = BRONZE_PATH):
    """Read parquet from MinIO and return a pandas-ready Spark DataFrame with metadata."""
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    sdf = spark.read.parquet(path)

    for column_name in [field.name for field in sdf.schema.fields if _is_numeric_spark_type(field.dataType)]:
        sdf = sdf.withColumn(column_name, F.col(column_name).cast("double"))

    if "time" in sdf.columns:
        sdf = sdf.withColumn("time", F.to_timestamp("time"))

    return sdf, sdf.count(), sdf.columns


def get_sdf():
    """Return Spark DataFrame + row count + columns."""
    return load_spark_df_cached(BRONZE_PATH)


def sdf_to_pandas_agg(sdf, group_cols, agg_exprs, order_cols=None, limit=500):
    """GroupBy + collect → pandas. Tối đa `limit` rows."""
    q = sdf.groupBy(*group_cols).agg(*agg_exprs)
    if order_cols:
        q = q.orderBy(*order_cols)
    return q.limit(limit).toPandas()


def _first_existing_path(base_dir: Path, patterns):
    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


@st.cache_resource(show_spinner=False)
def load_models():
    root = Path(__file__).resolve().parent
    search_dirs = [
        root / "training" / "best",
        root / "models",
        root / "model",
        root / "notebook",
        root,
    ]

    extreme_patterns = [
        "best_extreme_gb.pkl",
        "best_extreme_xgb.pkl",
        "best_extreme_rf.pkl",
        "best_extreme_*.pkl",
        "xgb_model.pkl",
        "rf_model.pkl",
    ]
    rain_patterns = [
        "best_rain_rf_rain.pkl",
        "best_rain_xgb_rain.pkl",
        "best_rain_*.pkl",
        "xgb_rain_model.pkl",
        "rf_rain_model.pkl",
    ]

    def resolve_model(patterns):
        for base_dir in search_dirs:
            found = _first_existing_path(base_dir, patterns)
            if found is not None:
                return found
        return None

    models = {}

    extreme_path = resolve_model(extreme_patterns)
    rain_path = resolve_model(rain_patterns)
    le_path = resolve_model(["label_encoder.pkl"])

    if extreme_path is not None:
        models["extreme"] = joblib.load(extreme_path)
        models["rf"] = models["extreme"]
        models["xgb"] = models["extreme"]

    if rain_path is not None:
        models["rain"] = joblib.load(rain_path)
        models["rf_rain"] = models["rain"]
        models["xgb_rain"] = models["rain"]

    if le_path is not None:
        models["le"] = joblib.load(le_path)

    return models


def build_input_frame(inputs: dict) -> pd.DataFrame:
    values = {**FEATURE_DEFAULTS, **inputs}

    humidity = float(values["humidity"])
    pressure = float(values["pressure"])
    wind_direction = float(values.get("wind_direction", 0.0))
    wind_angle = np.deg2rad(wind_direction)

    derived = {
        "humidity": humidity,
        "visibility": float(values["visibility"]),
        "cloudcover": float(values["cloudcover"]),
        "wind_direction": wind_direction,
        "pressure": pressure,
        "is_day": int(values.get("is_day", 1)),
        "hour": int(values["hour"]),
        "day": int(values["day"]),
        "month": int(values["month"]),
        "weekday": int(values["weekday"]),
        "wind_dir_sin": float(values.get("wind_dir_sin", np.sin(wind_angle))),
        "wind_dir_cos": float(values.get("wind_dir_cos", np.cos(wind_angle))),
        "temp_lag_1": float(values.get("temp_lag_1", values.get("temperature", 28.0))),
        "humidity_lag_1": float(values.get("humidity_lag_1", humidity)),
        "pressure_lag_1": float(values.get("pressure_lag_1", pressure)),
    }

    return pd.DataFrame([[derived[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


# ── UI components ──────────────────────────────────────────────────────────
def kpi_card(title, value, badge="", type="info"):
    color = {
        "info": "#4F6EF7",
        "up": "#10B981",
        "down": "#EF4444",
    }[type]

    return f"""
    <div class="kpi-card" style="border-top-color:{color}">
        <div style="font-size:12px;color:#9ca3af">{title}</div>
        <div style="font-size:26px;font-weight:700">{value}</div>
        <div style="color:{color};font-size:12px">{badge}</div>
    </div>
    """


def page_header(icon, bg, title, sub):
    st.markdown(
        f"""
    <div style="background:{bg};padding:24px;border-radius:16px;margin-bottom:20px">
        <div style="font-size:30px">{icon}</div>
        <div style="font-size:22px;font-weight:700">{title}</div>
        <div style="color:#6b7280">{sub}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )