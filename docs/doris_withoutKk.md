## 1) Kiem tra IP cua Doris FE
```bash
docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" doris-fe
```

Vi du: `172.18.0.2`

Cap nhat `FE_SERVERS` trong `docker-compose.yml`:
```yaml
doris-be:
  image: apache/doris:be-2.1.9
  container_name: doris-be
  ports:
    - "8040:8040" # BE HTTP
    - "9050:9050" # BE Internal
  depends_on:
    - doris-fe
  environment:
    FE_SERVERS: fe1:172.18.0.2:9010 # thay bang IP FE thuc te
    BE_ADDR: 0.0.0.0:9050
```

Ket noi MySQL client vao Doris FE:
```bash
docker exec -it doris-fe mysql -h 127.0.0.1 -P 9030 -u root
```


## 2) Create external Iceberg catalog (REST + MinIO)
```sql
SWITCH internal;
DROP CATALOG IF EXISTS iceberg;

CREATE CATALOG iceberg PROPERTIES (
    "type" = "iceberg",
    "iceberg.catalog.type" = "rest",
    "uri" = "http://iceberg-rest:8181",
    "s3.endpoint" = "http://minio:9000",
    "warehouse" = "s3://iceberg/warehouse",
    "s3.access_key" = "minioadmin",
    "s3.secret_key" = "minioadmin123",
    "s3.region" = "us-east-1",
    "use_path_style" = "true",
    "s3.connection.ssl.enabled" = "false"
);

SHOW CREATE CATALOG iceberg;
```

## 3) Tao database va bang dich cho Silver Weather
```sql
CREATE DATABASE IF NOT EXISTS weather_db;
USE weather_db;

CREATE TABLE weather_silver_features (
    time DATETIME(6),
    province VARCHAR(50),
    city VARCHAR(50),

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
    weather_main VARCHAR(50),
    weather_description VARCHAR(255),
    weather_icon VARCHAR(50),

    hour INT,
    day INT,
    month INT,
    weekday INT,

    region VARCHAR(20),
    wind_dir_sin DOUBLE,
    wind_dir_cos DOUBLE,
    rain INT,
    extreme VARCHAR(30),

    temp_level VARCHAR(20),
    humidity_level VARCHAR(20),
    pressure_level VARCHAR(20),
    wind_level VARCHAR(20),

    temp_lag_1 DOUBLE,
    humidity_lag_1 DOUBLE,
    pressure_lag_1 DOUBLE,
    temp_range DOUBLE,

    update_at DATETIME(6)
)
DUPLICATE KEY(time, province, city)
DISTRIBUTED BY HASH(province) BUCKETS 3
PROPERTIES ("replication_num" = "1");
```

```sql
INSERT INTO weather_silver_features
SELECT
    time,
    province,
    city,

    temperature,
    temp_min,
    temp_max,
    humidity,
    feels_like,
    visibility,
    precipitation,
    cloudcover,
    wind_speed,
    wind_gust,
    wind_direction,
    pressure,

    is_day,
    weather_code,
    weather_main,
    weather_description,
    weather_icon,

    hour,
    day,
    month,
    weekday,

    region,
    wind_dir_sin,
    wind_dir_cos,
    rain,
    extreme,

    temp_level,
    humidity_level,
    pressure_level,
    wind_level,

    temp_lag_1,
    humidity_lag_1,
    pressure_lag_1,
    temp_range,

    update_at
FROM iceberg.silver.weather;
```

## 4) Kiem tra du lieu
```sql
SELECT count(*) FROM weather_silver_features;

SELECT *
FROM weather_silver_features
ORDER BY time DESC
LIMIT 20;
```
