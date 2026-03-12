import pandas as pd
import json
from kafka import KafkaProducer


# docker exec -it biopharma-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server biopharma-kafka:29092 --create --if-not-exists --topic weather_raw --partitions 1 --replication-factor 1
# docker exec -it biopharma-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server biopharma-kafka:29092 --list
# python scripts/producer.py
KAFKA_BOOTSTRAP = ["localhost:9092"]
TOPIC = "weather_raw"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Read multiple files
df = pd.read_csv("data/weather-vn-1.csv")

count = 0
for _, row in df.iterrows():
    producer.send(TOPIC, row.to_dict())
    count += 1
    if count % 100 == 0:
        print(f"Sent {count} messages to Kafka...")

producer.flush()
print("Weather data sent to Kafka.")