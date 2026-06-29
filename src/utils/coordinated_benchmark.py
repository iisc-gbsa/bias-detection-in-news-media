"""
Coordinated Benchmark Script
Runs producer and monitors consumer with unified metrics collection
"""

import argparse
import logging
import time
import json
import threading
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from pathlib import Path

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from pymongo import MongoClient

from src.utils.metrics_collector import MetricsCollector, create_metrics
from config.config import kafka_config, mongo_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CoordinatedBenchmark:
    """Coordinates producer and consumer benchmarking with unified metrics"""

    def __init__(
        self,
        percentage: int,
        batch_size: int,
        source_mongo_uri: str = "mongodb://localhost:27017/",
        source_db: str = "test",
        source_collection: str = "articles",
        query_filter: Optional[Dict] = None,
        limit: Optional[int] = None,
        csv_path: Optional[str] = None,
        metrics_mongo_uri: str = "mongodb://localhost:27017/",
        metrics_db: str = "test",
        metrics_collection: str = "pipeline_metrics",
        consumer_timeout: int = 300,
    ):
        """
        Initialize coordinated benchmark

        Args:
            percentage: Percentage of data to use (e.g., 20, 50, 100)
            batch_size: Number of articles per batch for Kafka producer
            source_mongo_uri: MongoDB URI for source articles
            source_db: Source database name
            source_collection: Source collection name
            query_filter: Optional MongoDB query filter
            limit: Optional limit on articles
            csv_path: Optional CSV fallback
            metrics_mongo_uri: MongoDB URI for metrics storage
            metrics_db: Metrics database name
            metrics_collection: Metrics collection name
            consumer_timeout: Timeout in seconds to wait for consumer processing
        """
        self.percentage = percentage
        self.batch_size = batch_size
        self.consumer_timeout = consumer_timeout
        self.run_id = (
            f"benchmark_{percentage}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Load source data
        self.df = self._load_data(
            source_mongo_uri,
            source_db,
            source_collection,
            csv_path,
            query_filter,
            limit,
        )

        # Calculate data size
        total_len = len(self.df)
        size = int(total_len * percentage / 100)
        self.df = self.df.iloc[:size].copy()
        logger.info(f"Using {percentage}% of data: {len(self.df)} articles")

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            mongo_uri=metrics_mongo_uri,
            metrics_db=metrics_db,
            metrics_collection=metrics_collection,
        )

        # Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            compression_type="gzip",
        )

        # Consumer monitoring setup
        self.streaming_mongo_uri = "mongodb://localhost:27018/"
        self.streaming_db = "bias_detection"
        self.streaming_collection = "realtime_news"

        # Track articles
        self.produced_articles = set()
        self.consumed_articles = {}
        self.monitoring_active = False

        logger.info(f"✓ Coordinated benchmark initialized: {self.run_id}")

    def _load_data(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        csv_path: Optional[str],
        query_filter: Optional[Dict],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Load articles from MongoDB or CSV"""
        try:
            logger.info(f"Connecting to MongoDB: {mongo_uri}")
            client = MongoClient(mongo_uri)
            db = client[db_name]
            collection = db[collection_name]

            count = collection.count_documents(query_filter or {})
            if count == 0:
                raise ValueError(f"Collection '{collection_name}' is empty")

            logger.info(f"Found {count} articles in {db_name}.{collection_name}")

            query = query_filter or {}
            cursor = collection.find(query)
            if limit:
                cursor = cursor.limit(limit)

            articles = list(cursor)
            client.close()

            df = pd.DataFrame(articles)
            logger.info(f"✓ Loaded {len(df)} articles from MongoDB")
            return df

        except Exception as e:
            logger.warning(f"Failed to load from MongoDB: {e}")
            if csv_path:
                logger.info(f"Falling back to CSV: {csv_path}")
                df = pd.read_csv(
                    csv_path, escapechar="\\", quotechar='"', encoding="utf-8"
                )
                if limit:
                    df = df.iloc[:limit]
                logger.info(f"✓ Loaded {len(df)} articles from CSV")
                return df
            else:
                raise ValueError(f"Cannot load data: {e}")

    def _monitor_consumer(self):
        """Monitor consumer progress by checking streaming MongoDB"""
        logger.info("Starting consumer monitoring thread...")
        client = None

        try:
            client = MongoClient(self.streaming_mongo_uri)
            db = client[self.streaming_db]
            collection = db[self.streaming_collection]

            start_time = time.time()
            last_count = 0

            while self.monitoring_active:
                # Check for articles with our run_id
                consumed = collection.find({"benchmark_run_id": self.run_id})

                for doc in consumed:
                    article_url = doc.get("url")
                    if article_url and article_url not in self.consumed_articles:
                        # Calculate consumer metrics
                        consume_time = doc.get("kafka_consume_time", 0)
                        process_time = doc.get("processing_time", 0)
                        mongo_write_time = doc.get("mongodb_write_time", 0)

                        # Store for later correlation
                        self.consumed_articles[article_url] = {
                            "consume_time": consume_time,
                            "process_time": process_time,
                            "mongo_write_time": mongo_write_time,
                            "timestamp": time.time(),
                        }

                current_count = len(self.consumed_articles)

                # Check if all articles consumed
                if (
                    current_count >= len(self.produced_articles)
                    and len(self.produced_articles) > 0
                ):
                    logger.info(f"✓ All {current_count} articles consumed!")
                    break

                # Progress logging
                if current_count != last_count:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Consumer progress: {current_count}/{len(self.produced_articles)} "
                        f"articles processed ({elapsed:.1f}s elapsed)"
                    )
                    last_count = current_count

                time.sleep(2)

        except Exception as e:
            logger.error(f"Consumer monitoring error: {e}", exc_info=True)
        finally:
            if client:
                client.close()
            self.monitoring_active = False
            logger.info("Consumer monitoring thread stopped")

    def run_benchmark(self):
        """Run coordinated producer + consumer benchmark"""
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Starting Coordinated Benchmark")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Percentage: {self.percentage}%")
        logger.info(f"Total Articles: {len(self.df)}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"{'=' * 70}\n")

        # Start consumer monitoring thread
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._monitor_consumer, daemon=True)
        monitor_thread.start()

        # Give monitor time to start
        time.sleep(2)

        # Produce articles in batches
        total_articles = len(self.df)
        num_batches = (total_articles + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Starting production of {total_articles} articles in {num_batches} batches"
        )

        batch_start_time = time.time()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_articles)
            batch_df = self.df.iloc[start_idx:end_idx]

            logger.info(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
            logger.info(f"Processing articles {start_idx + 1} to {end_idx}")

            batch_metrics = []

            for idx, row in batch_df.iterrows():
                article_id = row.get("url", f"article_{idx}")

                # Create metrics object
                metrics = create_metrics(
                    run_id=self.run_id,
                    batch_id=batch_idx,
                    article_id=article_id,
                    dataset_size=f"{self.percentage}%",
                    total_articles=total_articles,
                    batch_size=self.batch_size,
                )

                try:
                    # Prepare article with benchmark metadata
                    article = {
                        "url": row.get("url", ""),
                        "title": row.get("title", ""),
                        "author": row.get("author", ""),
                        "published_date": str(row.get("published_date", "")),
                        "article_text": row.get("article_text", ""),
                        "word_count": int(row.get("word_count", 0)),
                        "media_name": row.get("media_name", ""),
                        "source": "benchmark",
                        "benchmark_run_id": self.run_id,
                        "benchmark_percentage": self.percentage,
                        "benchmark_batch_id": batch_idx,
                    }

                    # Track Kafka produce time
                    produce_start = time.time()
                    future = self.producer.send(
                        kafka_config.news_topic, value=article, key=article_id
                    )
                    record_metadata = future.get(timeout=10)
                    produce_end = time.time()

                    metrics.kafka_produce_time = produce_end - produce_start
                    metrics.kafka_partition = record_metadata.partition
                    metrics.kafka_offset = record_metadata.offset
                    metrics.status = "produced"

                    # Track produced articles
                    self.produced_articles.add(article_id)

                    # Save producer metrics
                    self.metrics_collector.save_metrics(metrics)
                    batch_metrics.append(metrics)

                except Exception as e:
                    logger.error(f"Failed to produce article {article_id}: {e}")
                    metrics.status = "failed"
                    metrics.error_message = str(e)
                    self.metrics_collector.save_metrics(metrics)

            # Batch summary
            if batch_metrics:
                avg_produce = sum(m.kafka_produce_time for m in batch_metrics) / len(
                    batch_metrics
                )
                logger.info(
                    f"Batch {batch_idx + 1} complete: {len(batch_metrics)} articles, avg produce time: {avg_produce:.3f}s"
                )

            # Small delay between batches
            time.sleep(0.5)

        batch_end_time = time.time()
        total_produce_time = batch_end_time - batch_start_time

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Production Complete!")
        logger.info(f"Total articles produced: {len(self.produced_articles)}")
        logger.info(f"Total production time: {total_produce_time:.2f}s")
        logger.info(
            f"Average throughput: {len(self.produced_articles) / total_produce_time:.2f} articles/sec"
        )
        logger.info(f"{'=' * 70}\n")

        # Wait for consumer to process
        logger.info("Waiting for consumer to process articles (Ctrl+C to stop)...")

        wait_start = time.time()
        last_reported = 0

        while monitor_thread.is_alive():
            # Check if monitoring completed
            if not self.monitoring_active:
                logger.info("Consumer monitoring completed")
                break

            # Check if all articles consumed
            if len(self.consumed_articles) >= len(self.produced_articles):
                logger.info(f"✓ All {len(self.consumed_articles)} articles consumed!")
                break

            # Progress reporting (only if changed significantly)
            current_progress = (
                len(self.consumed_articles) / len(self.produced_articles) * 100
                if len(self.produced_articles) > 0
                else 0
            )
            if (
                abs(current_progress - last_reported) >= 5 or current_progress == 100
            ):  # Report every 5% change
                logger.info(
                    f"Consumer progress: {current_progress:.1f}% ({len(self.consumed_articles)}/{len(self.produced_articles)})"
                )
                last_reported = current_progress

            time.sleep(3)

        # Stop monitoring thread if still running
        self.monitoring_active = False
        monitor_thread.join(timeout=10)

        # Calculate total time and final stats
        total_elapsed = time.time() - wait_start
        completion_rate = (
            len(self.consumed_articles) / len(self.produced_articles) * 100
            if len(self.produced_articles) > 0
            else 0
        )

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Consumer Monitoring Summary:")
        logger.info(f"  Time taken: {total_elapsed:.1f}s")
        logger.info(
            f"  Articles consumed: {len(self.consumed_articles)}/{len(self.produced_articles)} ({completion_rate:.1f}%)"
        )
        logger.info(f"{'=' * 70}\n")

        # Update metrics with consumer data
        self._update_consumer_metrics()

        # Generate summary
        self._generate_summary()

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Benchmark Complete!")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"{'=' * 70}\n")

    def _update_consumer_metrics(self):
        """Update producer metrics with consumer timing data using batch operations"""
        logger.info("Correlating producer and consumer metrics...")

        try:
            from pymongo import UpdateOne

            client = MongoClient(self.metrics_collector.mongo_uri)
            db = client[self.metrics_collector.metrics_db]
            collection = db[self.metrics_collector.metrics_collection]

            # Build batch operations
            operations = []
            for article_url, consumer_data in self.consumed_articles.items():
                operations.append(
                    UpdateOne(
                        {"run_id": self.run_id, "article_id": article_url},
                        {
                            "$set": {
                                "kafka_consume_time": consumer_data["consume_time"],
                                "processing_time": consumer_data["process_time"],
                                "mongodb_write_time": consumer_data["mongo_write_time"],
                                "consumer_timestamp": consumer_data["timestamp"],
                                "status": "completed",
                            }
                        },
                    )
                )

            # Execute batch update
            updated_count = 0
            if operations:
                result = collection.bulk_write(operations, ordered=False)
                updated_count = result.modified_count

            client.close()
            logger.info(f"✓ Updated {updated_count} metrics with consumer data")

        except Exception as e:
            logger.error(f"Failed to update consumer metrics: {e}")

    def _generate_summary(self):
        """Generate benchmark summary"""
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 70)

        try:
            client = MongoClient(self.metrics_collector.mongo_uri)
            db = client[self.metrics_collector.metrics_db]
            collection = db[self.metrics_collector.metrics_collection]

            # Get all metrics for this run
            metrics = list(collection.find({"run_id": self.run_id}))

            if not metrics:
                logger.warning("No metrics found for this run")
                return

            # Calculate statistics
            completed = [m for m in metrics if m.get("status") == "completed"]
            produced = [
                m for m in metrics if m.get("status") in ["produced", "completed"]
            ]
            failed = [m for m in metrics if m.get("status") == "failed"]

            logger.info(f"\nArticle Status:")
            logger.info(f"  Total: {len(metrics)}")
            logger.info(f"  Completed (end-to-end): {len(completed)}")
            logger.info(f"  Produced only: {len(produced) - len(completed)}")
            logger.info(f"  Failed: {len(failed)}")

            if produced:
                produce_times = [
                    m.get("kafka_produce_time", 0)
                    for m in produced
                    if m.get("kafka_produce_time")
                ]
                if produce_times:
                    logger.info(f"\nKafka Produce Times:")
                    logger.info(
                        f"  Average: {sum(produce_times) / len(produce_times):.4f}s"
                    )
                    logger.info(f"  Min: {min(produce_times):.4f}s")
                    logger.info(f"  Max: {max(produce_times):.4f}s")

            if completed:
                consume_times = [
                    m.get("kafka_consume_time", 0)
                    for m in completed
                    if m.get("kafka_consume_time")
                ]
                process_times = [
                    m.get("processing_time", 0)
                    for m in completed
                    if m.get("processing_time")
                ]
                mongo_times = [
                    m.get("mongodb_write_time", 0)
                    for m in completed
                    if m.get("mongodb_write_time")
                ]

                if consume_times:
                    logger.info(f"\nKafka Consume Times:")
                    logger.info(
                        f"  Average: {sum(consume_times) / len(consume_times):.4f}s"
                    )
                    logger.info(f"  Min: {min(consume_times):.4f}s")
                    logger.info(f"  Max: {max(consume_times):.4f}s")

                if process_times:
                    logger.info(f"\nProcessing Times:")
                    logger.info(
                        f"  Average: {sum(process_times) / len(process_times):.4f}s"
                    )
                    logger.info(f"  Min: {min(process_times):.4f}s")
                    logger.info(f"  Max: {max(process_times):.4f}s")

                if mongo_times:
                    logger.info(f"\nMongoDB Write Times:")
                    logger.info(f"  Average: {sum(mongo_times) / len(mongo_times):.4f}s")
                    logger.info(f"  Min: {min(mongo_times):.4f}s")
                    logger.info(f"  Max: {max(mongo_times):.4f}s")

                # End-to-end
                e2e_times = []
                for m in completed:
                    e2e = (
                        m.get("kafka_produce_time", 0)
                        + m.get("kafka_consume_time", 0)
                        + m.get("processing_time", 0)
                        + m.get("mongodb_write_time", 0)
                    )
                    if e2e > 0:
                        e2e_times.append(e2e)

                if e2e_times:
                    logger.info(f"\nEnd-to-End Times:")
                    logger.info(f"  Average: {sum(e2e_times) / len(e2e_times):.4f}s")
                    logger.info(f"  Min: {min(e2e_times):.4f}s")
                    logger.info(f"  Max: {max(e2e_times):.4f}s")

            client.close()
            logger.info("\n" + "=" * 70)

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

    def close(self):
        """Cleanup resources"""
        self.monitoring_active = False
        self.producer.flush()
        self.producer.close()
        self.metrics_collector.close()
        logger.info("✓ Coordinated benchmark closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Coordinated Kafka Pipeline Benchmark (Producer + Consumer)"
    )

    # Required arguments
    parser.add_argument(
        "--percentage",
        type=int,
        required=True,
        choices=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Percentage of data to use (e.g., 20 for 20%%)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Number of articles per batch for Kafka producer",
    )

    # Source data options
    parser.add_argument(
        "--source-uri",
        type=str,
        default="mongodb://localhost:27017/",
        help="MongoDB URI for source articles",
    )

    parser.add_argument(
        "--source-db",
        type=str,
        default="test",
        help="Database name for source articles",
    )

    parser.add_argument(
        "--source-collection",
        type=str,
        default="articles",
        help="Collection name for source articles",
    )

    parser.add_argument(
        "--query-filter", type=str, default=None, help="MongoDB query filter as JSON"
    )

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of articles to load"
    )

    parser.add_argument(
        "--csv", type=str, default=None, help="CSV file path as fallback"
    )

    # Metrics options
    parser.add_argument(
        "--metrics-uri",
        type=str,
        default="mongodb://localhost:27017/",
        help="MongoDB URI for metrics storage",
    )

    parser.add_argument(
        "--metrics-db", type=str, default="test", help="Database name for metrics"
    )

    parser.add_argument(
        "--consumer-timeout",
        type=int,
        default=300,
        help="(Not used) Monitor auto-exits when all articles consumed. Use Ctrl+C to stop manually.",
    )

    args = parser.parse_args()

    # Parse query filter
    query_filter = None
    if args.query_filter:
        try:
            query_filter = json.loads(args.query_filter)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --query-filter: {e}")
            return

    # Initialize and run benchmark
    benchmark = CoordinatedBenchmark(
        percentage=args.percentage,
        batch_size=args.batch_size,
        source_mongo_uri=args.source_uri,
        source_db=args.source_db,
        source_collection=args.source_collection,
        query_filter=query_filter,
        limit=args.limit,
        csv_path=args.csv,
        metrics_mongo_uri=args.metrics_uri,
        metrics_db=args.metrics_db,
        consumer_timeout=args.consumer_timeout,
    )

    try:
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        logger.info("\n>>> Benchmark interrupted by user <<<")
    except Exception as e:
        logger.error(f"\n✗ Benchmark failed: {e}", exc_info=True)
    finally:
        benchmark.close()


if __name__ == "__main__":
    main()
