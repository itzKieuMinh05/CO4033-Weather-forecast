import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# docker compose exec spark-master spark-submit --master local[*] /opt/spark/model/3_evaluate_test.py
print("--- BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP ---")

# Try read test data from MinIO train/test split then fallback to local CSV
TEST_SPLIT_PATH = os.getenv("TEST_SPLIT_PATH", "s3a://iceberg/train/test_split/")
LOCAL_TEST_CSV = os.path.join(os.getcwd(), "train_test_data", "test_data.csv")
MAX_EVAL_ROWS = int(os.getenv("MAX_EVAL_ROWS", "100000"))


def build_spark(app_name):
    AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    MINIO_EP = os.getenv("S3_ENDPOINT", "http://minio:9000")
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
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )


def is_numeric_spark_type(data_type):
    from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType
    return isinstance(data_type, (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType))

def load_test_df():
    try:
        spark = build_spark("WeatherVN-Eval")
        try:
            sdf = spark.read.parquet(TEST_SPLIT_PATH)
            numeric_cols = [field.name for field in sdf.schema.fields if is_numeric_spark_type(field.dataType)]
            keep_cols = [col_name for col_name in numeric_cols if col_name not in []]
            for target_col in ("extreme", "rain"):
                if target_col in sdf.columns and target_col not in keep_cols:
                    keep_cols.append(target_col)
            sdf = sdf.select(*keep_cols)

            total_rows = sdf.count()
            if total_rows > MAX_EVAL_ROWS and total_rows > 0:
                fraction = min(1.0, MAX_EVAL_ROWS / float(total_rows))
                sdf = sdf.sample(withReplacement=False, fraction=fraction, seed=42)
                sampled_rows = sdf.count()
                print(f"✓ Lấy mẫu test từ MinIO: {sampled_rows} / {total_rows} dòng")
            else:
                print(f"✓ Đọc test split từ MinIO: {total_rows} dòng")

            df = sdf.toPandas()
            spark.stop()
            return df
        except Exception:
            spark.stop()
    except Exception:
        pass

    if os.path.exists(LOCAL_TEST_CSV):
        df = pd.read_csv(LOCAL_TEST_CSV)
        print(f"✓ Đọc test CSV cục bộ: {LOCAL_TEST_CSV} — {len(df)} dòng")
        return df

    # final fallback: data/test_data.csv or data/weather_vn_cleaned.csv first rows
    if os.path.exists(os.path.join(os.getcwd(), "test_data.csv")):
        return pd.read_csv(os.path.join(os.getcwd(), "test_data.csv"))
    raise FileNotFoundError("Không tìm thấy test data ở MinIO hay local CSV.")


test_df = load_test_df()
target = 'extreme'

drop_cols = [
    target, 'rain', 'time', 'weather_code', 'weather_main', 'weather_description', 'weather_icon',
    'temperature', 'temp_min', 'temp_max', 'feels_like', 'temp_range',
    'wind_speed', 'wind_gust', 'rainfall', 'precipitation',
    'temp_level', 'humidity_level', 'pressure_level', 'wind_level'
]

X_test = test_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
X_test = X_test.fillna(X_test.mean())

y_test_extreme_raw = test_df[target]
y_test_rain = test_df['rain'] if 'rain' in test_df.columns else None

# Load best models from training/best/
BEST_DIR = os.path.join(os.getcwd(), 'training', 'best')
if not os.path.exists(BEST_DIR):
    raise FileNotFoundError(f"Không tìm thấy thư mục chứa model tốt nhất: {BEST_DIR}")

# label encoder
le = joblib.load(os.path.join(BEST_DIR, 'label_encoder.pkl'))
best_extreme = None
for f in os.listdir(BEST_DIR):
    if f.startswith('best_extreme_') and f.endswith('.pkl'):
        best_extreme = joblib.load(os.path.join(BEST_DIR, f))
        best_extreme_name = f
        break
if best_extreme is None:
    raise FileNotFoundError('Không tìm thấy mô hình extreme tốt nhất trong training/best')

# find best rain model
best_rain = None
for f in os.listdir(BEST_DIR):
    if f.startswith('best_rain_') and f.endswith('.pkl'):
        best_rain = joblib.load(os.path.join(BEST_DIR, f))
        best_rain_name = f
        break


def align_features(model, X_df):
    """Tự động gióng hàng và chọn đúng các cột feature mà mô hình mong đợi."""
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)
        # Lấy đúng các cột theo thứ tự model cần
        return X_df[expected_cols]
    return X_df


def calc_metrics(model, X, y_true):
    # Gióng hàng features trước khi dự đoán
    X_aligned = align_features(model, X)
    
    # Chuyển sang numpy array để tránh cảnh báo của sklearn
    X_array = X_aligned.to_numpy() if hasattr(X_aligned, 'to_numpy') else X_aligned.values if hasattr(X_aligned, 'values') else X_aligned
    
    pred = model.predict(X_array)
    try:
        prob = model.predict_proba(X_array)
    except Exception:
        prob = None
        
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, average='macro', zero_division=0)
    rec = recall_score(y_true, pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, pred, average='macro', zero_division=0)
    
    if prob is not None:
        try:
            if prob.ndim == 2 and prob.shape[1] == 2:
                auc = roc_auc_score(y_true, prob[:, 1])
            else:
                auc = roc_auc_score(y_true, prob, average='macro', multi_class='ovr')
        except Exception:
            auc = None
    else:
        auc = None
    return {'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


# Evaluate extreme
y_test_extreme = le.transform(y_test_extreme_raw.astype(str))
metrics_extreme = calc_metrics(best_extreme, X_test, y_test_extreme)
print('\n[EXTREME - Best model]')
print(best_extreme_name)
print(metrics_extreme)

# Confusion matrix cho Extreme
# Đảm bảo dùng X đã được gióng hàng khi gọi predict trực tiếp
X_test_ext_aligned = align_features(best_extreme, X_test)
pred_ext = best_extreme.predict(X_test_ext_aligned)
cm_ext = confusion_matrix(y_test_extreme, pred_ext)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_ext, annot=True, fmt='d', cmap='OrRd', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_title('Confusion Matrix - Extreme (best)')
ax.set_ylabel('True')
ax.set_xlabel('Pred')
plt.tight_layout()
plt.savefig('confusion_extreme_best.png')
plt.close()
print('✓ Đã xuất ảnh confusion_extreme_best.png')

# Evaluate rain if available
if best_rain is not None and y_test_rain is not None:
    metrics_rain = calc_metrics(best_rain, X_test, y_test_rain)
    print('\n[RAIN - Best model]')
    print(best_rain_name)
    print(metrics_rain)

    X_test_rain_aligned = align_features(best_rain, X_test)
    pred_rain = best_rain.predict(X_test_rain_aligned)
    cm_rain = confusion_matrix(y_test_rain, pred_rain)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_rain, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
    ax.set_title('Confusion Matrix - Rain (best)')
    ax.set_ylabel('True')
    ax.set_xlabel('Pred')
    plt.tight_layout()
    plt.savefig('confusion_rain_best.png')
    plt.close()
    print('✓ Đã xuất ảnh confusion_rain_best.png')

print('\n✅ HOÀN THÀNH ĐÁNH GIÁ')