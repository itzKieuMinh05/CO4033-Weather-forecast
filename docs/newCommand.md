docker compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/bronze.py

docker compose exec spark-master spark-submit   --master spark://spark-master:7077   /opt/spark/scripts/silver.py