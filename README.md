### Access services
| Service              | URL                        | Credentials                |
|----------------------|----------------------------|----------------------------|
| Spark Master UI      | http://localhost:8080      | -                          |
| Spark Worker UI      | http://localhost:8081      | -                          |
| MinIO Console        | http://localhost:9002      | minioadmin/minioadmin123   |
| Iceberg REST Catalog | http://localhost:8181      | -                          |
| JupyterLab           | http://localhost:8888      | no token                   |
| Doris FE             | http://localhost:8030      | -                          |
| Grafana              | http://localhost:3000      | admin/admin                |

### Running commands
# Run Iceberg example test
docker-compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/spark_sample.py

# Run spark example
docker exec -it biopharma-spark-master spark-submit /opt/spark/scripts/spark_sample.py 
  
# Kafka
# Create topic
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic my-topic --partitions 3 --replication-factor 1
# Run Bronze layer: raw Parquet
docker compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/bronze.py
# Run Silver layer: Iceberg table
docker compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/silver.py

# Start notebook service inside Docker
docker compose up -d --build spark-master spark-worker spark-notebook minio iceberg-rest

# Open JupyterLab
http://localhost:8888
