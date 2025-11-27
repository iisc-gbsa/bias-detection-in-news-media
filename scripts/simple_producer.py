"""
Simple Kafka Producer for Benchmarking
Sends messages from MongoDB to Kafka without consumer metrics
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from pymongo import MongoClient
from kafka import KafkaProducer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def produce_messages(
    mongo_uri: str,
    mongo_db: str,
    mongo_collection: str,
    kafka_servers: str,
    kafka_topic: str,
    percentage: float = None,
    batch_count: int = None,
    producer_batch_size: int = 100,
):
    """Produce messages from MongoDB to Kafka"""

    logger.info("=" * 80)
    logger.info("KAFKA PRODUCER")
    logger.info("=" * 80)
    logger.info(f"Source: {mongo_uri}/{mongo_db}.{mongo_collection}")
    logger.info(f"Target: {kafka_servers}/{kafka_topic}")
    logger.info(f"Batch Size: {producer_batch_size} messages")
    logger.info("=" * 80)

    # Connect to MongoDB
    mongo_client = MongoClient(mongo_uri)
    collection = mongo_client[mongo_db][mongo_collection]

    # Get total count
    total_records = collection.count_documents({})
    logger.info(f"Found {total_records} records in MongoDB")

    # Calculate records to send
    if batch_count is not None:
        records_to_send = min(batch_count, total_records)
        logger.info(f"Sending {records_to_send} records (batch_count={batch_count})")
    else:
        records_to_send = int(total_records * (percentage / 100.0))
        logger.info(f"Sending {records_to_send} records ({percentage}% of total)")

    # Fetch documents
    documents = list(collection.find().limit(records_to_send))
    mongo_client.close()

    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=kafka_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        compression_type="gzip",
        linger_ms=10,
        batch_size=16384,
    )

    logger.info("Kafka producer initialized")

    # Send messages in batches
    start_time = time.time()
    total_sent = 0
    failed = 0

    batch_number = 0
    for batch_start in range(0, len(documents), producer_batch_size):
        batch_end = min(batch_start + producer_batch_size, len(documents))
        batch = documents[batch_start:batch_end]
        batch_number += 1

        logger.info(
            f"Sending batch {batch_number}: {len(batch)} messages ({batch_start+1}-{batch_end}/{records_to_send})"
        )
        batch_send_start = time.time()

        # Send all messages in batch asynchronously
        futures = []
        for idx_in_batch, doc in enumerate(batch):
            global_idx = batch_start + idx_in_batch

            # Remove MongoDB _id field
            if "_id" in doc:
                doc.pop("_id")

            # Add metadata
            article = {
                **doc,
                "source": "benchmark",
                "ingestion_time": datetime.utcnow().isoformat(),
                "benchmark_send_time": time.time(),
            }

            key = article.get("url", f"article_{global_idx}")

            try:
                future = producer.send(kafka_topic, value=article, key=key)
                futures.append(future)
                total_sent += 1
            except Exception as e:
                logger.error(f"Failed to queue message {global_idx}: {e}")
                failed += 1

        # Flush the batch
        producer.flush()

        # Wait for confirmations
        for future in futures:
            try:
                future.get(timeout=10)
            except Exception as e:
                logger.error(f"Failed to confirm message: {e}")
                failed += 1
                total_sent -= 1

        batch_duration = time.time() - batch_send_start
        logger.info(
            f"Batch {batch_number} completed in {batch_duration:.2f}s ({batch_end}/{records_to_send} total)"
        )

    # Final flush
    producer.flush()
    producer.close()

    duration = time.time() - start_time

    logger.info("=" * 80)
    logger.info("PRODUCER COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Sent: {total_sent} messages")
    logger.info(f"Failed: {failed} messages")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Throughput: {total_sent / duration:.2f} records/sec")
    logger.info("=" * 80)

    return {
        "total_sent": total_sent,
        "failed": failed,
        "duration_seconds": duration,
        "throughput_records_per_sec": total_sent / duration if duration > 0 else 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Kafka Producer")

    # Data selection
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--percentage", type=float, help="Percentage of data (1-100)"
    )
    data_group.add_argument("--batch-count", type=int, help="Number of records")

    # MongoDB source
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default="mongodb://localhost:27017",
        help="MongoDB URI",
    )
    parser.add_argument("--mongo-db", type=str, default="test", help="Database name")
    parser.add_argument(
        "--mongo-collection", type=str, default="articles", help="Collection name"
    )

    # Kafka settings
    parser.add_argument(
        "--kafka-servers",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap servers",
    )
    parser.add_argument("--kafka-topic", type=str, default="news", help="Kafka topic")
    parser.add_argument(
        "--producer-batch-size",
        type=int,
        default=100,
        help="Producer batch size",
    )

    args = parser.parse_args()

    produce_messages(
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        mongo_collection=args.mongo_collection,
        kafka_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        percentage=args.percentage,
        batch_count=args.batch_count,
        producer_batch_size=args.producer_batch_size,
    )
