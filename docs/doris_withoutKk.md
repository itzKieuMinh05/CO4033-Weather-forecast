## 1) Kiểm tra IP của Doris FE:
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

Sử dụng MySQL Client (để kết nối Doris FE qua MySQL)
docker exec -it doris-fe mysql -h 127.0.0.1 -P 9030 -u root


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

## 3) Create database and weather table in Doris
```sql
CCREATE DATABASE IF NOT EXISTS weather_db;
USE weather_db;

CREATE TABLE weather_risk (
    time DATETIME(6),
    province VARCHAR(50),
    city VARCHAR(50),
    temperature DOUBLE,
    humidity DOUBLE,
    pressure DOUBLE,
    wind_speed DOUBLE,
    precipitation DOUBLE,
    cloudcover DOUBLE,
    risk_score DOUBLE,
    update_at DATETIME(6)
)
DUPLICATE KEY(time, province, city)
DISTRIBUTED BY HASH(province) BUCKETS 3
PROPERTIES ("replication_num" = "1");
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



---
