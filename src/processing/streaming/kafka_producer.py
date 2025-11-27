"""
Kafka Producer for News Articles
Simulates news ingestion from various sources
"""

from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import pandas as pd
from typing import Optional, Dict
import logging
from datetime import datetime
from pymongo import MongoClient

from config.config import kafka_config, mongo_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsKafkaProducer:
    """Kafka producer for publishing news articles"""

    def __init__(self, bootstrap_servers: Optional[str] = None):
        self.bootstrap_servers = bootstrap_servers or kafka_config.bootstrap_servers
        self.news_topic = kafka_config.news_topic

        # Initialize producer with JSON serialization
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=1,  # Ensure ordering
            compression_type="gzip",
        )

        logger.info(f"Kafka producer initialized: {self.bootstrap_servers}")

    def publish_article(self, article: Dict, key: Optional[str] = None) -> bool:
        """
        Publish a single article to Kafka

        Args:
            article: Dictionary containing article data
            key: Optional message key for partitioning

        Returns:
            Success boolean
        """
        try:
            # Add metadata
            message = {
                **article,
                "ingestion_time": datetime.utcnow().isoformat(),
                "source": article.get("source", "unknown"),
            }

            # Send to Kafka
            future = self.producer.send(self.news_topic, value=message, key=key)

            # Wait for confirmation
            record_metadata = future.get(timeout=10)

            logger.debug(
                f"Published to topic={record_metadata.topic} "
                f"partition={record_metadata.partition} "
                f"offset={record_metadata.offset}"
            )

            return True

        except KafkaError as e:
            logger.error(f"Failed to publish article: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing article: {e}")
            return False

    def publish_from_csv(
        self,
        csv_path: str,
        batch_size: int = 100,
        delay_seconds: float = 0.0,
        realtime_mode: bool = False,
    ) -> int:
        """
        Publish articles from CSV file to Kafka

        Args:
            csv_path: Path to CSV file
            batch_size: Number of articles to publish in each batch
            delay_seconds: Delay between batches (for rate limiting)
            realtime_mode: If True, simulates real-time streaming

        Returns:
            Number of successfully published articles
        """
        try:
            # Read CSV
            df = pd.read_csv(csv_path, escapechar="\\", quotechar='"', encoding="utf-8")
            logger.info(f"Loaded {len(df)} articles from {csv_path}")

            published_count = 0
            failed_count = 0

            # Process in batches
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]

                logger.info(
                    f"Publishing batch {start_idx // batch_size + 1}: "
                    f"articles {start_idx + 1} to {end_idx}"
                )

                for _, row in batch.iterrows():
                    article = {
                        "url": row.get("url", ""),
                        "title": row.get("title", ""),
                        "author": row.get("author", ""),
                        "published_date": row.get("published_date", ""),
                        "article_text": row.get("article_text", ""),
                        "word_count": row.get("word_count", 0),
                        "media_name": row.get("media_name", ""),
                        "source": "csv_import",
                    }

                    # Use URL as key for consistent partitioning
                    key = article.get("url", None)

                    if self.publish_article(article, key):
                        published_count += 1
                    else:
                        failed_count += 1

                    # Simulate real-time streaming
                    if realtime_mode and delay_seconds > 0:
                        time.sleep(delay_seconds)

                # Flush after each batch
                self.producer.flush()

                # Delay between batches
                if not realtime_mode and delay_seconds > 0:
                    time.sleep(delay_seconds)

            logger.info(
                f"Publishing complete: {published_count} succeeded, "
                f"{failed_count} failed"
            )

            return published_count

        except Exception as e:
            logger.error(f"Error reading CSV or publishing: {e}")
            return 0

    def publish_single_realtime(self, article_data: Dict) -> bool:
        """
        Publish a single article in real-time mode
        Used for simulating live news feed
        """
        article = {**article_data, "source": "realtime", "processing_mode": "streaming"}

        return self.publish_article(article, key=article_data.get("url"))

    def publish_from_mongodb(
        self,
        database: str = "test",
        collection: str = "articles",
        batch_size: int = 100,
        query: Optional[Dict] = None,
        mongo_uri: Optional[str] = None,
    ) -> int:
        """
        Publish articles from MongoDB collection to Kafka in batches

        Args:
            database: MongoDB database name
            collection: MongoDB collection name
            batch_size: Number of articles to publish in each batch
            query: Optional MongoDB query filter
            mongo_uri: Optional MongoDB URI (defaults to config)

        Returns:
            Number of successfully published articles
        """
        try:
            uri = mongo_uri or mongo_config.uri
            client = MongoClient(uri)
            db = client[database]
            coll = db[collection]

            # Get total count for logging
            filter_query = query or {}
            total_count = coll.count_documents(filter_query)
            logger.info(f"Found {total_count} articles in {database}.{collection}")

            published_count = 0
            failed_count = 0
            batch_num = 0

            # Fetch and publish in batches using skip/limit
            for skip in range(0, total_count, batch_size):
                batch_num += 1
                cursor = coll.find(filter_query).skip(skip).limit(batch_size)
                batch_docs = list(cursor)

                logger.info(
                    f"Publishing batch {batch_num}: "
                    f"articles {skip + 1} to {min(skip + batch_size, total_count)}"
                )

                for doc in batch_docs:
                    # Convert MongoDB document to article dict
                    article = {
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "author": doc.get("author", ""),
                        "published_date": doc.get("published_date", ""),
                        "article_text": doc.get("article_text", ""),
                        "word_count": doc.get("word_count", 0),
                        "media_name": doc.get("media_name", ""),
                        "source": "mongodb_import",
                    }

                    # Remove MongoDB _id if present (not JSON serializable)
                    if "_id" in doc:
                        article["mongo_id"] = str(doc["_id"])

                    # Use URL as key for consistent partitioning
                    key = article.get("url") or None

                    if self.publish_article(article, key):
                        published_count += 1
                    else:
                        failed_count += 1

                # Flush after each batch
                self.producer.flush()
                logger.info(f"Batch {batch_num} flushed to Kafka")

            client.close()
            logger.info(
                f"Publishing complete: {published_count} succeeded, "
                f"{failed_count} failed"
            )

            return published_count

        except Exception as e:
            logger.error(f"Error reading from MongoDB or publishing: {e}")
            return 0

    def close(self):
        """Close the producer and cleanup"""
        self.producer.flush()
        self.producer.close()
        logger.info("Kafka producer closed")


def main():
    """Test the Kafka producer"""
    import argparse

    parser = argparse.ArgumentParser(description="Kafka News Producer")
    parser.add_argument("--csv", type=str, help="Path to CSV file with articles")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for publishing"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Delay between batches in seconds"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="Simulate real-time streaming mode"
    )
    parser.add_argument("--test", action="store_true", help="Publish test message")

    args = parser.parse_args()

    producer = NewsKafkaProducer()

    try:
        if args.test:
            # Publish test article
            test_article = {
                "url": "https://example.com/test",
                "title": "Test Article",
                "author": "Test Author",
                "published_date": "2024-01-01",
                "article_text": "This is a test article for bias detection.",
                "word_count": 10,
                "media_name": "Test Media",
            }

            if producer.publish_article(test_article):
                logger.info("✓ Test article published successfully")
            else:
                logger.error("✗ Failed to publish test article")

        elif args.csv:
            # Publish from CSV
            count = producer.publish_from_csv(
                csv_path=args.csv,
                batch_size=args.batch_size,
                delay_seconds=args.delay,
                realtime_mode=args.realtime,
            )
            logger.info(f"✓ Published {count} articles from CSV")

        else:
            # Default: Publish from MongoDB (test.articles)
            count = producer.publish_from_mongodb(
                database="test",
                collection="articles",
                batch_size=args.batch_size,
            )
            logger.info(f"✓ Published {count} articles from MongoDB")

    finally:
        producer.close()


if __name__ == "__main__":
    main()
