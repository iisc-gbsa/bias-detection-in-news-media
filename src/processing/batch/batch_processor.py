"""
Spark Batch Processor for Daily News Collection
Reads from Kafka topic 'news' and processes in batches
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json,
    col,
    to_timestamp,
    current_timestamp,
    expr,
    struct,
    to_json,
)
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
import sys

from src.models.spark_bias_detector import SparkBiasDetector
from config.config import kafka_config, mongo_config, spark_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchNewsProcessor:
    """Batch processor for daily news collection"""

    def __init__(self):
        """Initialize Spark session and components"""
        self.spark = self._create_spark_session()
        self.bias_detector = SparkBiasDetector(self.spark)

        # Define input schema for Kafka messages
        self.news_schema = StructType(
            [
                StructField("url", StringType(), True),
                StructField("title", StringType(), True),
                StructField("author", StringType(), True),
                StructField("published_date", StringType(), True),
                StructField("article_text", StringType(), True),
                StructField("word_count", IntegerType(), True),
                StructField("media_name", StringType(), True),
                StructField("source", StringType(), True),
                StructField("ingestion_time", StringType(), True),
            ]
        )

    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        packages = ",".join(spark_config.packages)

        spark = (
            SparkSession.builder.appName(f"{spark_config.app_name}-Batch")
            .master(spark_config.master)
            .config("spark.driver.memory", spark_config.driver_memory)
            .config("spark.executor.memory", spark_config.executor_memory)
            .config("spark.executor.cores", spark_config.executor_cores)
            .config("spark.jars.packages", packages)
            .config(
                "spark.mongodb.output.uri",
                f"{mongo_config.uri}{mongo_config.database}.{mongo_config.collection_daily}",
            )
            .config(
                "spark.sql.streaming.checkpointLocation",
                kafka_config.checkpoint_location,
            )
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")
        logger.info("✓ Spark session created for batch processing")

        return spark

    def read_from_kafka_batch(
        self, start_offset: str = "earliest", end_offset: str = "latest"
    ) -> "DataFrame":
        """
        Read messages from Kafka in batch mode

        Args:
            start_offset: Starting offset (earliest, latest, or specific offset)
            end_offset: Ending offset

        Returns:
            DataFrame with parsed articles
        """
        logger.info(f"Reading from Kafka topic: {kafka_config.news_topic}")

        # Read from Kafka
        df = (
            self.spark.read.format("kafka")
            .option("kafka.bootstrap.servers", kafka_config.bootstrap_servers)
            .option("subscribe", kafka_config.news_topic)
            .option("startingOffsets", start_offset)
            .option("endingOffsets", end_offset)
            .load()
        )

        # Parse JSON value
        df = df.select(
            col("key").cast("string").alias("message_key"),
            from_json(col("value").cast("string"), self.news_schema).alias("data"),
            col("topic"),
            col("partition"),
            col("offset"),
            col("timestamp").alias("kafka_timestamp"),
        )

        # Flatten the data structure
        df = df.select(
            "message_key", "data.*", "topic", "partition", "offset", "kafka_timestamp"
        )

        # Add processing timestamp
        df = df.withColumn("batch_processing_time", current_timestamp())

        logger.info(f"✓ Read {df.count()} messages from Kafka")

        return df

    def process_and_analyze(self, df: "DataFrame") -> "DataFrame":
        """
        Process articles and perform bias analysis

        Args:
            df: Input DataFrame with articles

        Returns:
            DataFrame with bias analysis results
        """
        logger.info("Starting bias analysis...")

        # Perform bias detection
        df_analyzed = self.bias_detector.analyze_articles(df)

        # Add metadata
        df_analyzed = df_analyzed.withColumn("analysis_timestamp", current_timestamp())

        analyzed_count = df_analyzed.count()
        logger.info(f"✓ Analyzed {analyzed_count} articles")

        return df_analyzed

    def write_to_mongodb(self, df: "DataFrame", collection_name: str = None):
        """
        Write results to MongoDB

        Args:
            df: DataFrame to write
            collection_name: Optional collection name override
        """
        collection = collection_name or mongo_config.collection_daily
        mongo_uri = f"{mongo_config.uri}{mongo_config.database}.{collection}"

        logger.info(f"Writing to MongoDB: {mongo_uri}")

        try:
            df.write.format("mongodb").mode("append").option(
                "connection.uri", mongo_uri
            ).option("database", mongo_config.database).option(
                "collection", collection
            ).save()

            logger.info(f"✓ Successfully wrote {df.count()} records to MongoDB")

        except Exception as e:
            logger.error(f"✗ Error writing to MongoDB: {e}")
            # Write to error topic
            self._handle_write_error(df, e)

    def _handle_write_error(self, df: "DataFrame", error: Exception):
        """Handle write errors by sending to error topic"""
        try:
            # Prepare error records
            error_df = (
                df.withColumn("error_message", expr(f"'{str(error)}'"))
                .withColumn("error_timestamp", current_timestamp())
                .withColumn("error_type", expr("'mongodb_write_error'"))
            )

            # Write to Kafka error topic
            error_df.select(to_json(struct(col("*"))).alias("value")).write.format(
                "kafka"
            ).option("kafka.bootstrap.servers", kafka_config.bootstrap_servers).option(
                "topic", kafka_config.error_topic
            ).save()

            logger.info(f"✓ Sent {df.count()} error records to error topic")

        except Exception as e2:
            logger.error(f"✗ Failed to write to error topic: {e2}")

    def run_batch_job(self, start_offset: str = "earliest", end_offset: str = "latest"):
        """
        Run complete batch processing job

        Args:
            start_offset: Starting Kafka offset
            end_offset: Ending Kafka offset
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting Batch News Processing Job")
            logger.info(f"Time: {datetime.now().isoformat()}")
            logger.info("=" * 60)

            # Step 1: Read from Kafka
            df = self.read_from_kafka_batch(start_offset, end_offset)

            if df.count() == 0:
                logger.warning("No messages to process. Exiting.")
                return

            # Step 2: Process and analyze
            df_analyzed = self.process_and_analyze(df)

            # Step 3: Write to MongoDB
            self.write_to_mongodb(df_analyzed)

            # Step 4: Print summary
            self._print_summary(df_analyzed)

            logger.info("=" * 60)
            logger.info("✓ Batch processing completed successfully")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"✗ Batch processing failed: {e}")
            raise

    def _print_summary(self, df: "DataFrame"):
        """Print summary statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)

        summary = self.bias_detector.get_summary_stats(df)

        logger.info(f"\nTotal Articles: {summary['total_articles']}")
        logger.info(f"Average Overall Bias: {summary['avg_overall_bias']:.3f}")
        logger.info(f"Average Political Bias: {summary['avg_political_bias']:.3f}")
        logger.info(f"Average Gender Bias: {summary['avg_gender_bias']:.3f}")
        logger.info(f"Average Religious Bias: {summary['avg_religious_bias']:.3f}")

        logger.info("\nPolitical Bias Distribution:")
        for row in summary["political_types"]:
            logger.info(f"  {row['political_type']}: {row['count']}")

        logger.info("\nGender Bias Distribution:")
        for row in summary["gender_types"]:
            logger.info(f"  {row['gender_type']}: {row['count']}")

    def stop(self):
        """Stop Spark session"""
        self.spark.stop()
        logger.info("✓ Spark session stopped")


def main():
    """Main entry point for batch processor"""
    import argparse

    parser = argparse.ArgumentParser(description="Spark Batch Processor for Daily News")
    parser.add_argument(
        "--start-offset",
        type=str,
        default="earliest",
        help="Starting Kafka offset (earliest, latest, or JSON offset spec)",
    )
    parser.add_argument(
        "--end-offset",
        type=str,
        default="latest",
        help="Ending Kafka offset (latest or JSON offset spec)",
    )

    args = parser.parse_args()

    processor = BatchNewsProcessor()

    try:
        processor.run_batch_job(
            start_offset=args.start_offset, end_offset=args.end_offset
        )
    except Exception as e:
        logger.error(f"Batch job failed: {e}")
        sys.exit(1)
    finally:
        processor.stop()


if __name__ == "__main__":
    main()
