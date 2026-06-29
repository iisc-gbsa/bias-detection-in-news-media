"""
Configuration for Spark ML Bias Detection System
"""

import os
from dataclasses import dataclass


@dataclass
class KafkaConfig:
    """Kafka configuration"""

    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    news_topic: str = "news"
    error_topic: str = "news_errors"
    consumer_group: str = "bias-detection-group"
    auto_offset_reset: str = "earliest"

    # Streaming settings
    max_offsets_per_trigger: int = 10000
    checkpoint_location: str = "./checkpoints"


@dataclass
class MongoConfig:
    """MongoDB configuration"""

    uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27018/")
    database: str = "bias_detection"
    collection_daily: str = "daily_news"
    collection_realtime: str = "realtime_news"
    collection_errors: str = "error_logs"

    # Write optimization settings to reduce MongoDB bottleneck
    # Currently MongoDB writes consume ~50% of batch processing time
    ordered_writes: bool = (
        False  # Unordered writes for better throughput (parallel inserts)
    )
    max_batch_size: int = 1000  # MongoDB bulk write batch size
    connection_pool_size: int = 50  # Connection pool size for concurrent writes
    write_concern_w: int = 1  # Write concern: 1 = primary acknowledgment only
    write_concern_journal: bool = False  # Don't wait for journal sync (faster)
    retry_writes: bool = True  # Retry failed writes automatically
    socket_timeout_ms: int = 60000  # Socket timeout (60 seconds)
    connect_timeout_ms: int = 10000  # Connection timeout (10 seconds)


@dataclass
class SparkConfig:
    """Spark configuration"""

    app_name: str = "BiasDetectionPipeline"
    master: str = os.getenv("SPARK_MASTER", "local[*]")

    # Memory settings
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 6

    # NEW: Distributed processing settings
    # Use multiple executors instead of single executor with many cores
    # for better parallelism when not bottlenecked by I/O
    num_executors: int = 3  # Distribute work across multiple executors
    executor_cores_distributed: int = 2  # Fewer cores per executor

    # NEW: Parallelism settings
    default_parallelism: int = 6  # Match total available cores
    sql_shuffle_partitions: int = 6  # Optimize shuffle operations

    # Parallelism
    shuffle_partitions: int = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "6"))

    # Spark packages
    packages: list = None

    def __post_init__(self):
        if self.packages is None:
            self.packages = [
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
                "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
            ]


@dataclass
class BiasDetectionConfig:
    """Bias detection model configuration"""

    # Ensemble weights
    keyword_weight: float = 0.3
    tfidf_weight: float = 0.3
    embedding_weight: float = 0.4

    # BERT model
    bert_model_name: str = "all-MiniLM-L6-v2"

    # Bias dimension weights
    bias_weights: dict = None

    def __post_init__(self):
        if self.bias_weights is None:
            self.bias_weights = {
                "political": 0.20,
                "gender": 0.15,
                "religious": 0.20,
                "caste": 0.15,
                "region": 0.15,
                "socioeconomic": 0.15,
            }

    # TF-IDF settings
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 2)


# Global configurations
kafka_config = KafkaConfig()
mongo_config = MongoConfig()
spark_config = SparkConfig()
bias_config = BiasDetectionConfig()
