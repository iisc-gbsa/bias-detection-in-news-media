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
    max_offsets_per_trigger: int = 1000
    checkpoint_location: str = "./checkpoints"


@dataclass
class MongoConfig:
    """MongoDB configuration"""

    uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    database: str = "bias_detection"
    collection_daily: str = "daily_news"
    collection_realtime: str = "realtime_news"
    collection_errors: str = "error_logs"


@dataclass
class SparkConfig:
    """Spark configuration"""

    app_name: str = "BiasDetectionPipeline"
    master: str = os.getenv("SPARK_MASTER", "local[*]")

    # Memory settings
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 2

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
