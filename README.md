### Access services
| Service              | URL                        | Credentials                |
|----------------------|----------------------------|----------------------------|
| Spark Master UI      | http://localhost:8080      | -                          |
| Spark Worker 1 UI    | http://localhost:8081      | -                          |
| Spark Worker 2 UI    | http://localhost:8082      | -                          |
| MinIO Console        | http://localhost:9002      | minioadmin/minioadmin123   |
| Iceberg REST Catalog | http://localhost:8181      | -                          |
| Kafka UI             | http://localhost:8888      | -                          |
| Doris FE             | http://localhost:8030      | -                          |
| Grafana              | http://localhost:3000      | admin/admin                |

### Running commands
# Run Iceberg example test
docker-compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/spark_sample.py

# Run spark example
docker exec -it biopharma-spark-master spark-submit /opt/spark/scripts/spark_sample.py 
  
# Kafka
# Create topic
docker exec -it biopharma-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server biopharma-kafka:29092 --create --if-not-exists --topic weather_raw --partitions 1 --replication-factor 1

# Optional one-time table setup (consumer.py can auto-create this)
docker compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/bronze.py

# Start Kafka -> Bronze streaming job
docker compose exec spark-master spark-submit --master spark://spark-master:7077 /opt/spark/scripts/consumer.py

# Send data to Kafka from local machine
python scripts/producer.py