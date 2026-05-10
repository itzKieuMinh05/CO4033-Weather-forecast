"""Training script upgraded:
 - reads train split from MinIO (best-effort) or local CSV
 - trains multiple candidate models (RandomForest, XGBoost, LightGBM, GradientBoosting)
 - uses SMOTE(k_neighbors=3) for multi-class extreme target
 - uses class_weight where applicable
 - does a small GridSearchCV per model (lightweight)
 - selects best model by F1-macro, then accuracy, then recall
 - saves best models to training/best/ 
docker compose exec spark-master spark-submit --master local[*] /opt/spark/model/2_train_model.py
 """

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
SMOTE = 3
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
    print("ℹ️ XGBoost không khả dụng trong môi trường này, sẽ bỏ qua XGB.")

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None
    print("ℹ️ imbalanced-learn không khả dụng trong môi trường này, sẽ bỏ qua SMOTE.")

print("--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH TRÊN TẬP TRAIN ---")

# Paths and MinIO options
BRONZE_TRAIN_PATH = os.getenv("TRAIN_SPLIT_PATH", "s3a://iceberg/train/train_split/")
LOCAL_TRAIN_CSV = os.path.join(os.getcwd(), "train_test_data", "train_data.csv")
MAX_TRAIN_ROWS = int(os.getenv("MAX_TRAIN_ROWS", "200000"))


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

def load_train_df():
    try:
        spark = build_spark("WeatherVN-Train")
        try:
            sdf = spark.read.parquet(BRONZE_TRAIN_PATH)
            numeric_cols = [field.name for field in sdf.schema.fields if is_numeric_spark_type(field.dataType)]
            keep_cols = [col_name for col_name in numeric_cols if col_name not in []]
            for target_col in ("extreme", "rain"):
                if target_col in sdf.columns and target_col not in keep_cols:
                    keep_cols.append(target_col)
            sdf = sdf.select(*keep_cols)

            total_rows = sdf.count()
            if total_rows > MAX_TRAIN_ROWS and total_rows > 0:
                fraction = min(1.0, MAX_TRAIN_ROWS / float(total_rows))
                sdf = sdf.sample(withReplacement=False, fraction=fraction, seed=42)
                sampled_rows = sdf.count()
                print(f"✓ Lấy mẫu train từ MinIO: {sampled_rows} / {total_rows} dòng")
            else:
                print(f"✓ Đọc train split từ MinIO: {total_rows} dòng")

            df = sdf.toPandas()
            spark.stop()
            return df
        except Exception:
            spark.stop()
            print("⚠️ Không đọc được từ MinIO train split, fallback sang CSV cục bộ.")
    except Exception:
        pass

    if os.path.exists(LOCAL_TRAIN_CSV):
        df = pd.read_csv(LOCAL_TRAIN_CSV)
        print(f"✓ Đọc train CSV cục bộ: {LOCAL_TRAIN_CSV} — {len(df)} dòng")
        return df

    raise FileNotFoundError("Không tìm thấy dữ liệu train ở MinIO hay local CSV.")


df = load_train_df()
target = "extreme"

# Drop leak / non-numeric columns
drop_cols = [
    target, 'rain', 'time', 'weather_code', 'weather_main', 'weather_description', 'weather_icon',
    'temperature', 'temp_min', 'temp_max', 'feels_like', 'temp_range',
    'wind_speed', 'wind_gust', 'rainfall', 'precipitation',
    'temp_level', 'humidity_level', 'pressure_level', 'wind_level',
    'temp_lag_1', 'humidity_lag_1', 'pressure_lag_1'  # consistent with eval script
]

X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).copy()
X = X.fillna(0)
le = LabelEncoder()
if target in df.columns:
    y_raw = df[target].astype(str)
    y = le.fit_transform(y_raw)
    # save label encoder
    os.makedirs(os.path.join(os.getcwd(), "training", "best"), exist_ok=True)
    joblib.dump(le, os.path.join(os.getcwd(), "training", "best", "label_encoder.pkl"))
else:
    raise KeyError("Cột 'extreme' không tồn tại trong dữ liệu train.")

# Balance with SMOTE (k_neighbors=3)
if SMOTE is not None:
    print("Đang áp dụng SMOTE (k_neighbors=3) — multi-class")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)
else:
    print("⚠️ Không có SMOTE, dùng dữ liệu gốc để huấn luyện.")
    X_res, y_res = X, y

# Candidate models and small param grids (kept lightweight)
models = {}
grids = {}

# RandomForest
models['rf'] = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
grids['rf'] = {'n_estimators': [100, 200], 'max_depth': [8, 12]}

# XGBoost
if XGBClassifier is not None:
    models['xgb'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    grids['xgb'] = {'n_estimators': [100, 150], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1]}

# Try LightGBM if available
try:
    from lightgbm import LGBMClassifier
    models['lgbm'] = LGBMClassifier(random_state=42, n_jobs=-1)
    grids['lgbm'] = {'n_estimators': [100, 150], 'max_depth': [-1, 6], 'learning_rate': [0.05, 0.1]}
except Exception:
    print("ℹ️ LightGBM không khả dụng, bỏ qua LGBM")

# GradientBoosting (sklearn)
models['gb'] = GradientBoostingClassifier(random_state=42)
grids['gb'] = {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring = 'f1_macro'

results = []
best_overall = {'name': None, 'score': -999, 'acc': 0.0, 'recall': 0.0, 'model': None}

for name, estimator in models.items():
    grid = grids.get(name, {})
    if grid:
        print(f"\n👉 Running GridSearch for {name} — params: {grid}")
        gs = GridSearchCV(estimator, grid, cv=cv, scoring=scoring, n_jobs=1, verbose=0)
        try:
            gs.fit(X_res, y_res)
            best = gs.best_estimator_
            score = gs.best_score_
        except Exception as e:
            print(f"⚠️ GridSearch cho {name} thất bại ({e}), fitting default estimator")
            estimator.fit(X_res, y_res)
            best = estimator
            # approximate score via cross_val_score
            from sklearn.model_selection import cross_val_score
            score = np.mean(cross_val_score(best, X_res, y_res, cv=cv, scoring=scoring))
    else:
        print(f"\n👉 Training {name} with default params")
        estimator.fit(X_res, y_res)
        best = estimator
        from sklearn.model_selection import cross_val_score
        score = np.mean(cross_val_score(best, X_res, y_res, cv=cv, scoring=scoring))

    # compute additional metrics on resampled data (quick)
    from sklearn.metrics import accuracy_score, recall_score
    y_pred = best.predict(X_res)
    acc = accuracy_score(y_res, y_pred)
    rec = recall_score(y_res, y_pred, average='macro')

    print(f"{name} => F1-macro: {score:.4f}, Acc: {acc:.4f}, Recall-macro: {rec:.4f}")
    results.append((name, score, acc, rec, best))

    # selection priority: F1-macro, then acc, then recall
    if (score > best_overall['score'] or
        (abs(score - best_overall['score']) < 1e-6 and acc > best_overall['acc']) or
        (abs(score - best_overall['score']) < 1e-6 and abs(acc - best_overall['acc']) < 1e-6 and rec > best_overall['recall'])):
        best_overall = {'name': name, 'score': score, 'acc': acc, 'recall': rec, 'model': best}

# Save best extreme model
os.makedirs(os.path.join(os.getcwd(), "training", "best"), exist_ok=True)
best_name = best_overall['name']
best_model = best_overall['model']
if best_model is not None:
    best_path = os.path.join(os.getcwd(), "training", "best", f"best_extreme_{best_name}.pkl")
    joblib.dump(best_model, best_path)
    print(f"\n✅ Lưu mô hình extreme tốt nhất: {best_name} → {best_path}")

# Train binary rain models (RF and XGB) and pick best by F1
print("\n--- Huấn luyện mô hình dự báo mưa (binary) ---")
if 'rain' not in df.columns:
    print("⚠️ Cột 'rain' không tìm thấy trong dữ liệu. Bỏ qua phần mưa.")
else:
    y_rain = df['rain'].astype(int)
    X_rain = X.fillna(0)
    # simple candidates
    rain_cands = {}
    rain_cands['rf_rain'] = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    if XGBClassifier is not None:
        rain_cands['xgb_rain'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    try:
        from lightgbm import LGBMClassifier
        rain_cands['lgbm_rain'] = LGBMClassifier(random_state=42)
    except Exception:
        pass

    from sklearn.model_selection import cross_val_score
    best_rain = (None, -999, None)
    for rn, re in rain_cands.items():
        print(f"Training {rn}...")
        try:
            re.fit(X_rain, y_rain)
            s = np.mean(cross_val_score(re, X_rain, y_rain, cv=3, scoring='f1'))
        except Exception:
            s = -999
        print(f"{rn} F1 (cv): {s:.4f}")
        if s > best_rain[1]:
            best_rain = (rn, s, re)

    if best_rain[2] is not None:
        rain_path = os.path.join(os.getcwd(), "training", "best", f"best_rain_{best_rain[0]}.pkl")
        joblib.dump(best_rain[2], rain_path)
        print(f"✅ Lưu mô hình rain tốt nhất: {best_rain[0]} → {rain_path}")

print("\nHoàn tất huấn luyện. Kết quả tóm tắt:")
for r in results:
    print(f" - {r[0]}: F1={r[1]:.4f}, Acc={r[2]:.4f}, Recall={r[3]:.4f}")

print("\n✅ ĐÃ LƯU TẤT CẢ MÔ HÌNH TỐT NHẤT VÀ LABEL ENCODER VÀO training/best/")