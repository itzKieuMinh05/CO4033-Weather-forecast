1. Kiểm tra IP của Doris FE:
docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" doris-fe


Ví dụ:
172.18.0.2

Thay vào: 
doris-be:
    image: apache/doris:be-2.1.9
    container_name: doris-be
    ports:
      - "8040:8040"   # BE HTTP
      - "9050:9050"   # BE Internal
    depends_on:
      - doris-fe
    environment:
      FE_SERVERS: fe1:172.20.0.98:9010 # Thay IP ở đây
      BE_ADDR: 0.0.0.0:9050

2. Sử dụng MySQL Client (để kết nối Doris FE qua MySQL)
docker exec -it doris-fe mysql -h 127.0.0.1 -P 9030 -u root




## 3) Create database and weather table in Doris
```sql
CREATE DATABASE IF NOT EXISTS weather_db;
USE weather_db;

DROP TABLE IF EXISTS weather;
CREATE TABLE weather (
    `time` DATETIME,
    `province` VARCHAR(20),
    `city` VARCHAR(50),
    `temperature` DOUBLE,
    `humidity` DOUBLE,
    `pressure` DOUBLE,
    `wind_speed` DOUBLE,
    `precipitation` DOUBLE,
    `cloudcover` DOUBLE,
    `risk_score` DOUBLE,
    `update_at` DATETIME,
)
DUPLICATE KEY(`time`, `province`, `city`)
DISTRIBUTED BY HASH(`province`) BUCKETS 3
PROPERTIES (
    "replication_num" = "1",
    "max_error_number" = "50000",
    "max_filter_ratio" = "0.2"
);
```
```sql
INSERT INTO weather_risk
SELECT
    time,
    province,
    city,
    temperature,
    humidity,
    pressure,
    wind_speed,
    precipitation,
    cloudcover,
    risk_score,
    update_at
FROM iceberg.gold.weather_risk;
```
---

## 4) Create Routine Load from Kafka (weather topic)
```sql
CREATE ROUTINE LOAD weather_job ON weather
COLUMNS(
    time, province, city, temperature, temp_min, temp_max, humidity, feels_like,
    visibility, precipitation, cloudcover, wind_speed, wind_gust, wind_direction,
    pressure, is_day, weather_code, weather_main, weather_description, weather_icon,
    kafka_time = NOW(),
    load_at = NOW()
)
PROPERTIES
(
    "format" = "json",
    "max_batch_interval" = "10",
    "max_batch_rows" = "200000"
)
FROM KAFKA
(
    "kafka_broker_list" = "biopharma-kafka:29092",
    "kafka_topic" = "weather_raw",
    "property.group.id" = "doris_weather_consumer_group_v1",
    "property.kafka_default_offsets" = "OFFSET_BEGINNING"
);
```

Check status:
```sql
SHOW ROUTINE LOAD FOR weather_job;
```

---

## 5) Create external Iceberg catalog (REST + MinIO)
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

---

## 6) Query weather tables from Iceberg
```sql
-- Bronze table (from scripts/bronze.py / scripts/consumer.py)
SELECT * FROM iceberg.weather.weather_bronze LIMIT 10;

-- Silver layer (if created in weather namespace)
SELECT * FROM iceberg.weather.weather_silver LIMIT 10;
```