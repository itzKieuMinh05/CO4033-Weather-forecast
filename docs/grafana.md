1. Kết nối với grafana

2. Thêm variable city

Vào: Dashboard Settings → Variables → Add variable

Cấu hình:

Name city

Type Query

Query
SELECT DISTINCT city
FROM iceberg.gold.weather_risk
ORDER BY city

3. Chạy các query cho

Weather condition through time
SELECT
  time,
  temperature,
  humidity,
  precipitation,
  wind_speed
FROM iceberg.gold.weather_risk
WHERE city = '$city'


Weather risk score through time
SELECT
  time,
  risk_score
FROM iceberg.gold.weather_risk
WHERE city = '$city'
ORDER BY time

Risk score over provinces
SELECT
  province,
  AVG(risk_score) AS avg_risk
FROM iceberg.gold.weather_risk
WHERE $__timeFilter(time)
GROUP BY province
ORDER BY avg_risk DESC

Correspond between risk score and precipitation
SELECT
  precipitation ,
  risk_score
FROM iceberg.gold.weather_risk
WHERE $__timeFilter(time)
AND city = '$city'
order by precipitation  asc;