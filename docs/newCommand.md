docker compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/bronze.py

docker compose exec spark-master spark-submit --master local[*] /opt/spark/model/1_split_data.py

docker compose exec spark-master spark-submit --master local[*] /opt/spark/model/2_train_model.py

docker compose exec spark-master spark-submit --master local[*] /opt/spark/model/3_evaluate_test.py


docker compose exec spark-master spark-submit   --master spark://spark-master:7077   /opt/spark/scripts/silver.py



### Grafana command lines
SELECT
    DATE(time) AS date,
    AVG(temperature) AS avg_temp
FROM weather_silver_features
GROUP BY DATE(time)
ORDER BY date;

SELECT
    DATE(time) AS date,
    AVG(humidity) AS avg_humidity
FROM weather_silver_features
GROUP BY DATE(time)
ORDER BY date;

SELECT province, AVG(temperature) FROM weather_db.weather_silver_features
GROUP BY province;

SELECT province, AVG(humidity) FROM weather_db.weather_silver_features
GROUP BY province;

SELECT
    DATE(time) AS date,
    SUM(precipitation) AS total_rain
FROM weather_silver_features
GROUP BY DATE(time)
ORDER BY date;

SELECT
    DATE_TRUNC(time, 'month') AS time,
    extreme,
    COUNT(*) AS value
FROM weather_silver_features
WHERE extreme IS NOT NULL
GROUP BY DATE_TRUNC(time, 'month'), extreme
ORDER BY time, extreme;

SELECT
    DATE_SUB(DATE(time), INTERVAL (DAYOFYEAR(time) - 1) % 5 DAY) AS date_5d,
    SUM(precipitation) AS total_rain
FROM weather_silver_features
GROUP BY date_5d
ORDER BY date_5d;