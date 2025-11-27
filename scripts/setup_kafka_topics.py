"""
Setup Script for Kafka Topics
Creates required topics for the bias detection pipeline
"""

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging
from config.config import kafka_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_topics():
    """Create Kafka topics for the pipeline"""

    # Initialize admin client
    admin_client = KafkaAdminClient(
        bootstrap_servers=kafka_config.bootstrap_servers,
        client_id="bias-detection-setup",
    )

    # Define topics to create
    topics = [
        NewTopic(
            name=kafka_config.news_topic,
            num_partitions=6,
            replication_factor=1,
            topic_configs={
                "retention.ms": "604800000",  # 7 days
                "compression.type": "gzip",
            },
        ),
        NewTopic(
            name=kafka_config.error_topic,
            num_partitions=2,
            replication_factor=1,
            topic_configs={
                "retention.ms": "2592000000",  # 30 days
                "compression.type": "gzip",
            },
        ),
    ]

    # Create topics
    try:
        admin_client.create_topics(new_topics=topics, validate_only=False)
        logger.info("✓ Successfully created Kafka topics:")
        for topic in topics:
            logger.info(f"  - {topic.name} (partitions: {topic.num_partitions})")

    except TopicAlreadyExistsError:
        logger.warning("⚠ Topics already exist")

    except Exception as e:
        logger.error(f"✗ Error creating topics: {e}")
        raise

    finally:
        admin_client.close()

    logger.info("\nKafka Topic Setup Complete!")
    logger.info("=" * 60)
    logger.info("Topics:")
    logger.info(f"  News Topic: {kafka_config.news_topic}")
    logger.info(f"  Error Topic: {kafka_config.error_topic}")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_topics()
