import os
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CITY_COLUMN_CANDIDATES = ["city", "province", "location"]
DEFAULT_BRONZE_PATH = "s3a://iceberg/bronze/weather_raw_parquet/"


# ======================
# 1. LOAD + PREPARE DATA
# ======================
def load_data(path="weather_vn_cleaned.csv"):
    print("Loading data...")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    minio_endpoint = os.getenv("S3_ENDPOINT", "http://minio:9000")
    bronze_input_path = os.getenv("BRONZE_WEATHER_PATH", DEFAULT_BRONZE_PATH)
    spark_master = os.getenv("SPARK_MASTER_URL")
    spark_packages = os.getenv("SPARK_JARS_PACKAGES", "").strip()

    spark_builder = SparkSession.builder.appName("WeatherClusteringFromMinIO")
    if spark_master:
        spark_builder = spark_builder.master(spark_master)

    if spark_packages:
        spark_builder = (
            spark_builder
            .config("spark.jars.packages", spark_packages)
            .config("spark.jars.repositories", "https://repo1.maven.org/maven2")
        )

    spark = (
        spark_builder
        .config("spark.sql.catalog.iceberg.s3.endpoint", minio_endpoint)
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )

    source_path = path if path != "weather_vn_cleaned.csv" else bronze_input_path
    df = spark.read.parquet(source_path).toPandas()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    print("Raw shape:", df.shape)
    return df


def select_city_column(df):
    for col in CITY_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("Khong tim thay cot tinh/thanh (city/province/location).")


def select_weather_features(df):
    feature_candidates = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "cloudcover",

    ]
    features = [f for f in feature_candidates if f in df.columns]
    if not features:
        raise ValueError("Khong tim thay feature so hop le de clustering.")
    print("Using features:", features)
    return features


def aggregate_by_city(df, city_col, features):
    df_use = df[[city_col] + features].copy()
    df_use = df_use.dropna(subset=[city_col])

    # Trung binh theo tinh/thanh de tim pattern cap vung.
    city_metrics = df_use.groupby(city_col, as_index=False)[features].mean()
    city_metrics = city_metrics.dropna()

    print("Cities used for clustering:", len(city_metrics))
    return city_metrics


# ======================
# 2. SCALING
# ======================
def scale_data(city_metrics, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(city_metrics[features])
    return X_scaled


# ======================
# 3. FIND BEST K (ELBOW + SILHOUETTE)
# ======================
def find_optimal_k(X_scaled):
    max_k = min(10, len(X_scaled) - 1)
    if max_k < 2:
        raise ValueError("So luong tinh/thanh khong du de clustering (can >= 3).")

    k_values = list(range(2, max_k + 1))
    inertia = []
    sil_scores = []

    print("Running Elbow + Silhouette...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Elbow plot
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, inertia, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.show()

    # Silhouette plot
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, sil_scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette by k")
    plt.tight_layout()
    plt.show()

    best_k = k_values[sil_scores.index(max(sil_scores))]
    print(f"Suggested k from silhouette: {best_k}")
    return best_k


# ======================
# 4. TRAIN KMEANS
# ======================
def train_kmeans(X_scaled, k):
    print(f"Training KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


# ======================
# 5. VISUALIZATION (PCA)
# ======================
def visualize_clusters(X_scaled, labels, city_names):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=70,
        alpha=0.85,
        edgecolors="k",
        linewidth=0.3,
    )

    for i, city in enumerate(city_names):
        plt.annotate(city, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.9)

    plt.xlabel(f"PC1 ({var[0]:.1%})")
    plt.ylabel(f"PC2 ({var[1]:.1%})")
    plt.title("KMeans Clusters of Provinces/Cities")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.tight_layout()
    plt.show()


def plot_provinces_per_cluster(labels):
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    colors = plt.cm.Pastel1(range(len(cluster_counts)))

    plt.figure(figsize=(7, 5))
    bars = plt.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Provinces")
    plt.title("Number of Provinces per Cluster")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    for bar, value in zip(bars, cluster_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


# ======================
# 6. ANALYZE CLUSTERS
# ======================
def analyze_clusters(city_metrics, labels, features, city_col):
    result = city_metrics.copy()
    result["cluster"] = labels

    print("\nCluster centroids (original scale):")
    summary = result.groupby("cluster")[features].mean().round(3)
    print(summary)

    print("\nCities per cluster:")
    cities_per_cluster = result.groupby("cluster")[city_col].apply(list)
    for cluster_id, city_list in cities_per_cluster.items():
        print(f"Cluster {cluster_id} ({len(city_list)} cities): {city_list}")

    return result, summary


# ======================
# 7. SAVE RESULT
# ======================
def save_result(result, summary):
    result.to_csv("city_weather_clusters.csv", index=False)
    summary.to_csv("cluster_weather_profiles.csv")
    print("Saved city cluster assignments to city_weather_clusters.csv")
    print("Saved cluster profiles to cluster_weather_profiles.csv")


# ======================
# 8. MAIN
# ======================
def main():
    df = load_data()
    city_col = select_city_column(df)
    features = select_weather_features(df)

    city_metrics = aggregate_by_city(df, city_col, features)
    X_scaled = scale_data(city_metrics, features)

    best_k = find_optimal_k(X_scaled)
    _, labels = train_kmeans(X_scaled, k=best_k)

    visualize_clusters(X_scaled, labels, city_metrics[city_col].tolist())
    plot_provinces_per_cluster(labels)
    result, summary = analyze_clusters(city_metrics, labels, features, city_col)
    save_result(result, summary)


if __name__ == "__main__":
    main()
