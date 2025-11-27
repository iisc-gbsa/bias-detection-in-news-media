"""
Metrics Collector for Kafka Pipeline Benchmarking
Tracks timing metrics at each stage: produce, consume, process, MongoDB write
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from pymongo import MongoClient
import logging
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Data class for pipeline metrics"""

    # Identification
    run_id: str
    batch_id: int
    article_id: str

    # Volume metadata
    dataset_size: str  # e.g., "20%", "50%", "100%", "set_1"
    total_articles: int
    batch_size: int

    # Timing metrics (in seconds)
    kafka_produce_time: Optional[float] = None
    kafka_consume_time: Optional[float] = None
    processing_time: Optional[float] = None
    mongodb_write_time: Optional[float] = None
    end_to_end_time: Optional[float] = None

    # Timestamps
    produce_start_ts: Optional[str] = None
    produce_end_ts: Optional[str] = None
    consume_start_ts: Optional[str] = None
    consume_end_ts: Optional[str] = None
    process_start_ts: Optional[str] = None
    process_end_ts: Optional[str] = None
    mongodb_start_ts: Optional[str] = None
    mongodb_end_ts: Optional[str] = None

    # Status
    status: str = "initiated"  # initiated, completed, failed
    error_message: Optional[str] = None

    # Kafka metadata
    kafka_partition: Optional[int] = None
    kafka_offset: Optional[int] = None

    # Additional context
    consumer_id: Optional[str] = None
    node_name: Optional[str] = None


class MetricsCollector:
    """Collector for pipeline benchmarking metrics"""

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        metrics_db: str = "test",
        metrics_collection: str = "pipeline_metrics",
    ):
        """
        Initialize metrics collector

        Args:
            mongo_uri: MongoDB connection URI
            metrics_db: Database for storing metrics
            metrics_collection: Collection for storing metrics
        """
        self.mongo_uri = mongo_uri
        self.metrics_db = metrics_db
        self.metrics_collection = metrics_collection

        # Connect to MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[metrics_db]
        self.collection = self.db[metrics_collection]

        logger.info(
            f"✓ Metrics collector initialized: {mongo_uri}{metrics_db}.{metrics_collection}"
        )

        # Create indexes for better query performance
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            self.collection.create_index("run_id")
            self.collection.create_index("dataset_size")
            self.collection.create_index("batch_id")
            self.collection.create_index([("run_id", 1), ("dataset_size", 1)])
            self.collection.create_index("produce_start_ts")
            logger.info("✓ Indexes created for metrics collection")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    def save_metrics(self, metrics: PipelineMetrics) -> bool:
        """
        Save metrics to MongoDB

        Args:
            metrics: PipelineMetrics object

        Returns:
            Success boolean
        """
        try:
            # Convert to dict and save
            metrics_dict = asdict(metrics)
            self.collection.insert_one(metrics_dict)
            logger.debug(f"✓ Saved metrics for article {metrics.article_id}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save metrics: {e}")
            return False

    def update_metrics(self, run_id: str, article_id: str, updates: Dict) -> bool:
        """
        Update existing metrics

        Args:
            run_id: Run identifier
            article_id: Article identifier
            updates: Dictionary of fields to update

        Returns:
            Success boolean
        """
        try:
            result = self.collection.update_one(
                {"run_id": run_id, "article_id": article_id}, {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"✗ Failed to update metrics: {e}")
            return False

    @contextmanager
    def track_kafka_produce(self, metrics: PipelineMetrics):
        """
        Context manager to track Kafka produce time

        Usage:
            with metrics_collector.track_kafka_produce(metrics):
                # Kafka produce code here
                pass
        """
        start_time = time.time()
        metrics.produce_start_ts = datetime.utcnow().isoformat()

        try:
            yield
            metrics.produce_end_ts = datetime.utcnow().isoformat()
            metrics.kafka_produce_time = time.time() - start_time
        except Exception as e:
            metrics.status = "failed"
            metrics.error_message = f"Produce error: {str(e)}"
            raise

    @contextmanager
    def track_kafka_consume(self, metrics: PipelineMetrics):
        """Context manager to track Kafka consume time"""
        start_time = time.time()
        metrics.consume_start_ts = datetime.utcnow().isoformat()

        try:
            yield
            metrics.consume_end_ts = datetime.utcnow().isoformat()
            metrics.kafka_consume_time = time.time() - start_time
        except Exception as e:
            metrics.status = "failed"
            metrics.error_message = f"Consume error: {str(e)}"
            raise

    @contextmanager
    def track_processing(self, metrics: PipelineMetrics):
        """Context manager to track processing time"""
        start_time = time.time()
        metrics.process_start_ts = datetime.utcnow().isoformat()

        try:
            yield
            metrics.process_end_ts = datetime.utcnow().isoformat()
            metrics.processing_time = time.time() - start_time
        except Exception as e:
            metrics.status = "failed"
            metrics.error_message = f"Processing error: {str(e)}"
            raise

    @contextmanager
    def track_mongodb_write(self, metrics: PipelineMetrics):
        """Context manager to track MongoDB write time"""
        start_time = time.time()
        metrics.mongodb_start_ts = datetime.utcnow().isoformat()

        try:
            yield
            metrics.mongodb_end_ts = datetime.utcnow().isoformat()
            metrics.mongodb_write_time = time.time() - start_time
            metrics.status = "completed"
        except Exception as e:
            metrics.status = "failed"
            metrics.error_message = f"MongoDB write error: {str(e)}"
            raise

    @contextmanager
    def track_end_to_end(self, metrics: PipelineMetrics):
        """Context manager to track end-to-end time"""
        start_time = time.time()

        try:
            yield
            metrics.end_to_end_time = time.time() - start_time
        except Exception as e:
            metrics.status = "failed"
            metrics.error_message = f"End-to-end error: {str(e)}"
            raise

    def get_batch_summary(self, run_id: str, batch_id: int) -> Dict:
        """
        Get summary statistics for a batch

        Args:
            run_id: Run identifier
            batch_id: Batch identifier

        Returns:
            Dictionary with summary statistics
        """
        try:
            pipeline = [
                {"$match": {"run_id": run_id, "batch_id": batch_id}},
                {
                    "$group": {
                        "_id": "$batch_id",
                        "total_articles": {"$sum": 1},
                        "avg_produce_time": {"$avg": "$kafka_produce_time"},
                        "avg_consume_time": {"$avg": "$kafka_consume_time"},
                        "avg_process_time": {"$avg": "$processing_time"},
                        "avg_mongodb_write_time": {"$avg": "$mongodb_write_time"},
                        "avg_end_to_end_time": {"$avg": "$end_to_end_time"},
                        "max_end_to_end_time": {"$max": "$end_to_end_time"},
                        "min_end_to_end_time": {"$min": "$end_to_end_time"},
                        "failed_count": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        },
                    }
                },
            ]

            result = list(self.collection.aggregate(pipeline))
            return result[0] if result else {}

        except Exception as e:
            logger.error(f"✗ Failed to get batch summary: {e}")
            return {}

    def get_run_summary(self, run_id: str) -> Dict:
        """
        Get summary statistics for an entire run

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with summary statistics
        """
        try:
            pipeline = [
                {"$match": {"run_id": run_id}},
                {
                    "$group": {
                        "_id": "$dataset_size",
                        "total_articles": {"$sum": 1},
                        "avg_produce_time": {"$avg": "$kafka_produce_time"},
                        "avg_consume_time": {"$avg": "$kafka_consume_time"},
                        "avg_process_time": {"$avg": "$processing_time"},
                        "avg_mongodb_write_time": {"$avg": "$mongodb_write_time"},
                        "avg_end_to_end_time": {"$avg": "$end_to_end_time"},
                        "p50_end_to_end": {
                            "$median": {
                                "input": "$end_to_end_time",
                                "method": "approximate",
                            }
                        },
                        "p95_end_to_end": {
                            "$percentile": {
                                "input": "$end_to_end_time",
                                "p": [0.95],
                                "method": "approximate",
                            }
                        },
                        "p99_end_to_end": {
                            "$percentile": {
                                "input": "$end_to_end_time",
                                "p": [0.99],
                                "method": "approximate",
                            }
                        },
                        "failed_count": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        },
                    }
                },
                {"$sort": {"_id": 1}},
            ]

            result = list(self.collection.aggregate(pipeline))
            return result

        except Exception as e:
            logger.error(f"✗ Failed to get run summary: {e}")
            return []

    def export_metrics(self, run_id: str, output_file: str) -> bool:
        """
        Export metrics to JSON file

        Args:
            run_id: Run identifier
            output_file: Output file path

        Returns:
            Success boolean
        """
        try:
            metrics = list(self.collection.find({"run_id": run_id}, {"_id": 0}))

            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            logger.info(f"✓ Exported {len(metrics)} metrics to {output_file}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to export metrics: {e}")
            return False

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("✓ Metrics collector closed")


# Utility function for creating a metrics object
def create_metrics(
    run_id: str,
    batch_id: int,
    article_id: str,
    dataset_size: str,
    total_articles: int,
    batch_size: int,
    **kwargs,
) -> PipelineMetrics:
    """
    Create a PipelineMetrics object with default values

    Args:
        run_id: Unique identifier for this benchmark run
        batch_id: Batch number
        article_id: Article identifier
        dataset_size: Size designation (e.g., "20%", "50%", "set_1")
        total_articles: Total number of articles in dataset
        batch_size: Size of the batch
        **kwargs: Additional fields

    Returns:
        PipelineMetrics object
    """
    return PipelineMetrics(
        run_id=run_id,
        batch_id=batch_id,
        article_id=article_id,
        dataset_size=dataset_size,
        total_articles=total_articles,
        batch_size=batch_size,
        **kwargs,
    )
