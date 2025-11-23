"""
Error Handler for Bias Detection Pipeline
Monitors and processes errors from the dead letter topic
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp, count, window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType,
    DoubleType,
)
import logging
from datetime import datetime

from config.config import kafka_config, mongo_config, spark_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandler:
    """Handler for processing error messages from dead letter topic"""

    def __init__(self):
        """Initialize error handler"""
        self.spark = self._create_spark_session()

        # Define error message schema
        self.error_schema = StructType(
            [
                StructField("url", StringType(), True),
                StructField("title", StringType(), True),
                StructField("article_text", StringType(), True),
                StructField("error_message", StringType(), True),
                StructField("error_timestamp", StringType(), True),
                StructField("error_type", StringType(), True),
                StructField("batch_id", IntegerType(), True),
                StructField("topic", StringType(), True),
                StructField("partition", IntegerType(), True),
                StructField("offset", IntegerType(), True),
            ]
        )

    def _create_spark_session(self) -> SparkSession:
        """Create Spark session for error handling"""
        packages = ",".join(spark_config.packages)

        spark = (
            SparkSession.builder.appName(f"{spark_config.app_name}-ErrorHandler")
            .master(spark_config.master)
            .config("spark.driver.memory", spark_config.driver_memory)
            .config("spark.jars.packages", packages)
            .config(
                "spark.mongodb.output.uri",
                f"{mongo_config.uri}{mongo_config.database}.{mongo_config.collection_errors}",
            )
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")
        logger.info("✓ Spark session created for error handling")

        return spark

    def read_error_stream(self) -> "DataFrame":
        """
        Read error messages from Kafka error topic

        Returns:
            Streaming DataFrame with error messages
        """
        logger.info(f"Reading from error topic: {kafka_config.error_topic}")

        df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_config.bootstrap_servers)
            .option("subscribe", kafka_config.error_topic)
            .option("startingOffsets", "earliest")
            .option("failOnDataLoss", "false")
            .load()
        )

        # Parse JSON
        df = df.select(
            col("key").cast("string").alias("error_key"),
            from_json(col("value").cast("string"), self.error_schema).alias(
                "error_data"
            ),
            col("timestamp").alias("kafka_timestamp"),
        )

        # Flatten
        df = df.select("error_key", "error_data.*", "kafka_timestamp")

        # Add processing timestamp
        df = df.withColumn("error_processing_time", current_timestamp())

        return df

    def log_errors_to_mongodb(self, error_df: "DataFrame") -> "StreamingQuery":
        """
        Log error messages to MongoDB for tracking and analysis

        Args:
            error_df: Streaming DataFrame with error messages

        Returns:
            StreamingQuery object
        """
        mongo_uri = f"{mongo_config.uri}{mongo_config.database}.{mongo_config.collection_errors}"

        logger.info(f"Logging errors to MongoDB: {mongo_uri}")

        def write_errors_batch(batch_df, batch_id):
            """Write error batch to MongoDB"""
            try:
                if batch_df.count() > 0:
                    batch_df.write.format("mongodb").mode("append").option(
                        "connection.uri", mongo_uri
                    ).option("database", mongo_config.database).option(
                        "collection", mongo_config.collection_errors
                    ).save()

                    logger.info(
                        f"✓ Error Batch {batch_id}: Logged {batch_df.count()} errors"
                    )

                    # Print error summary
                    error_types = batch_df.groupBy("error_type").count().collect()
                    logger.info(f"  Error types in batch {batch_id}:")
                    for row in error_types:
                        logger.info(f"    - {row['error_type']}: {row['count']}")

            except Exception as e:
                logger.error(f"✗ Error Batch {batch_id}: Failed to log errors: {e}")

        query = (
            error_df.writeStream.foreachBatch(write_errors_batch)
            .queryName("error_logging")
            .option("checkpointLocation", f"{kafka_config.checkpoint_location}/errors")
            .trigger(processingTime="30 seconds")
            .start()
        )

        return query

    def monitor_error_stats(
        self, error_df: "DataFrame", window_duration: str = "10 minutes"
    ) -> "StreamingQuery":
        """
        Monitor error statistics over time

        Args:
            error_df: Streaming DataFrame with errors
            window_duration: Window duration for aggregations
        """
        stats_df = (
            error_df.withColumn("event_time", col("error_processing_time"))
            .groupBy(window(col("event_time"), window_duration), col("error_type"))
            .agg(count("*").alias("error_count"))
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("error_type"),
                col("error_count"),
            )
        )

        query = (
            stats_df.writeStream.format("console")
            .queryName("error_stats")
            .option(
                "checkpointLocation", f"{kafka_config.checkpoint_location}/error_stats"
            )
            .outputMode("complete")
            .trigger(processingTime="60 seconds")
            .start()
        )

        return query

    def retry_failed_articles(
        self, error_df: "DataFrame", retry_topic: str = None
    ) -> "StreamingQuery":
        """
        Retry processing failed articles by sending back to news topic

        Args:
            error_df: Streaming DataFrame with errors
            retry_topic: Topic to send retry messages (defaults to news topic)
        """
        retry_topic = retry_topic or kafka_config.news_topic

        # Filter retriable errors and prepare for retry
        retriable_df = error_df.filter(
            col("error_type").isin(
                ["mongodb_write_error", "streaming_mongodb_write_error"]
            )
        ).select(col("url"), col("title"), col("article_text"))

        # Write back to Kafka for retry
        query = (
            retriable_df.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_config.bootstrap_servers)
            .option("topic", retry_topic)
            .option("checkpointLocation", f"{kafka_config.checkpoint_location}/retry")
            .trigger(processingTime="5 minutes")
            .start()
        )

        logger.info(f"✓ Error retry mechanism active (retry topic: {retry_topic})")

        return query

    def run_error_handler(self, enable_retry: bool = False):
        """
        Run error handling pipeline

        Args:
            enable_retry: Enable automatic retry of failed articles
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting Error Handler")
            logger.info(f"Time: {datetime.now().isoformat()}")
            logger.info("=" * 60)

            # Read error stream
            error_df = self.read_error_stream()

            # Log errors to MongoDB
            logging_query = self.log_errors_to_mongodb(error_df)

            # Monitor error statistics
            stats_query = self.monitor_error_stats(error_df)

            queries = [logging_query, stats_query]

            # Optional: Retry mechanism
            if enable_retry:
                retry_query = self.retry_failed_articles(error_df)
                queries.append(retry_query)
                logger.info("✓ Retry mechanism enabled")

            logger.info("✓ Error handler started")
            logger.info("Press Ctrl+C to stop...")

            # Wait for termination
            for query in queries:
                query.awaitTermination()

        except KeyboardInterrupt:
            logger.info("\nStopping error handler...")
            for query in self.spark.streams.active:
                query.stop()
            logger.info("✓ Error handler stopped")

        except Exception as e:
            logger.error(f"✗ Error handler failed: {e}")
            raise

    def stop(self):
        """Stop error handler and Spark session"""
        for query in self.spark.streams.active:
            query.stop()
        self.spark.stop()
        logger.info("✓ Error handler stopped")


def main():
    """Main entry point for error handler"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Error Handler for Bias Detection Pipeline"
    )
    parser.add_argument(
        "--retry", action="store_true", help="Enable automatic retry of failed articles"
    )

    args = parser.parse_args()

    handler = ErrorHandler()

    try:
        handler.run_error_handler(enable_retry=args.retry)
    finally:
        handler.stop()


if __name__ == "__main__":
    main()
