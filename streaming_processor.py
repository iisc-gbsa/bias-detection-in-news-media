"""
Spark Structured Streaming Processor for Real-Time News
Continuous processing of news articles from Kafka
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
    window,
    count,
    avg,
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

from spark_bias_detector import SparkBiasDetector
from config import kafka_config, mongo_config, spark_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingNewsProcessor:
    """Real-time streaming processor for news articles"""

    def __init__(self, checkpoint_location: str = None):
        """Initialize streaming processor"""
        self.checkpoint_location = (
            checkpoint_location or kafka_config.checkpoint_location
        )
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
                StructField("processing_mode", StringType(), True),
            ]
        )

    def _create_spark_session(self) -> SparkSession:
        """Create Spark session for streaming"""
        packages = ",".join(spark_config.packages)

        spark = (
            SparkSession.builder.appName(f"{spark_config.app_name}-Streaming")
            .master(spark_config.master)
            .config("spark.driver.memory", spark_config.driver_memory)
            .config("spark.executor.memory", spark_config.executor_memory)
            .config("spark.executor.cores", spark_config.executor_cores)
            .config("spark.jars.packages", packages)
            .config(
                "spark.mongodb.output.uri",
                f"{mongo_config.uri}{mongo_config.database}.{mongo_config.collection_realtime}",
            )
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_location)
            .config("spark.sql.streaming.schemaInference", "true")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")
        logger.info("✓ Spark session created for streaming processing")

        return spark

    def read_kafka_stream(self) -> "DataFrame":
        """
        Read streaming data from Kafka

        Returns:
            Streaming DataFrame
        """
        logger.info(f"Starting stream from Kafka topic: {kafka_config.news_topic}")

        # Read stream from Kafka
        df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_config.bootstrap_servers)
            .option("subscribe", kafka_config.news_topic)
            .option("startingOffsets", kafka_config.auto_offset_reset)
            .option("maxOffsetsPerTrigger", kafka_config.max_offsets_per_trigger)
            .option("failOnDataLoss", "false")
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

        # Flatten structure
        df = df.select(
            "message_key", "data.*", "topic", "partition", "offset", "kafka_timestamp"
        )

        # Add processing timestamp
        df = df.withColumn("stream_processing_time", current_timestamp())

        logger.info("✓ Kafka stream initialized")

        return df

    def process_stream(self, input_df: "DataFrame") -> "DataFrame":
        """
        Process streaming data and perform bias analysis

        Args:
            input_df: Input streaming DataFrame

        Returns:
            Processed streaming DataFrame
        """
        logger.info("Applying bias detection to stream...")

        # Apply bias detection (this works on streaming DataFrames)
        df_analyzed = self.bias_detector.analyze_articles(input_df)

        # Add metadata
        df_analyzed = df_analyzed.withColumn("analysis_timestamp", current_timestamp())

        return df_analyzed

    def write_to_mongodb_stream(
        self, df: "DataFrame", query_name: str = "news_to_mongodb"
    ):
        """
        Write streaming data to MongoDB

        Args:
            df: Streaming DataFrame
            query_name: Name for the streaming query
        """
        mongo_uri = f"{mongo_config.uri}{mongo_config.database}.{mongo_config.collection_realtime}"

        logger.info(f"Writing stream to MongoDB: {mongo_uri}")

        def write_batch_to_mongo(batch_df, batch_id):
            """Function to write each micro-batch to MongoDB"""
            try:
                if batch_df.count() > 0:
                    batch_df.write.format("mongodb").mode("append").option(
                        "connection.uri", mongo_uri
                    ).option("database", mongo_config.database).option(
                        "collection", mongo_config.collection_realtime
                    ).save()

                    logger.info(
                        f"✓ Batch {batch_id}: Wrote {batch_df.count()} records to MongoDB"
                    )
                else:
                    logger.debug(f"Batch {batch_id}: No records to write")

            except Exception as e:
                logger.error(f"✗ Batch {batch_id}: Error writing to MongoDB: {e}")
                # Write errors to error topic
                self._write_errors_to_kafka(batch_df, e, batch_id)

        # Start streaming query with foreachBatch
        query = (
            df.writeStream.foreachBatch(write_batch_to_mongo)
            .queryName(query_name)
            .option("checkpointLocation", f"{self.checkpoint_location}/{query_name}")
            .trigger(processingTime="10 seconds")
            .start()
        )

        return query

    def _write_errors_to_kafka(
        self, batch_df: "DataFrame", error: Exception, batch_id: int
    ):
        """Write error records to Kafka error topic"""
        try:
            error_df = (
                batch_df.withColumn("error_message", expr(f"'{str(error)}'"))
                .withColumn("error_timestamp", current_timestamp())
                .withColumn("error_type", expr("'streaming_mongodb_write_error'"))
                .withColumn("batch_id", expr(f"{batch_id}"))
            )

            # Write to error topic
            error_df.select(to_json(struct(col("*"))).alias("value")).write.format(
                "kafka"
            ).option("kafka.bootstrap.servers", kafka_config.bootstrap_servers).option(
                "topic", kafka_config.error_topic
            ).save()

            logger.info(f"✓ Batch {batch_id}: Sent errors to error topic")

        except Exception as e2:
            logger.error(f"✗ Batch {batch_id}: Failed to write errors: {e2}")

    def write_to_console(
        self, df: "DataFrame", query_name: str = "console_output"
    ) -> "StreamingQuery":
        """
        Write streaming data to console (for debugging)

        Args:
            df: Streaming DataFrame
            query_name: Name for the streaming query

        Returns:
            StreamingQuery object
        """
        query = (
            df.writeStream.format("console")
            .queryName(query_name)
            .option("checkpointLocation", f"{self.checkpoint_location}/{query_name}")
            .option("truncate", "false")
            .outputMode("append")
            .trigger(processingTime="10 seconds")
            .start()
        )

        return query

    def monitor_stream_stats(
        self, df: "DataFrame", window_duration: str = "5 minutes"
    ) -> "StreamingQuery":
        """
        Monitor and display streaming statistics

        Args:
            df: Streaming DataFrame with bias analysis
            window_duration: Window duration for aggregations
        """
        # Calculate statistics over time windows
        stats_df = (
            df.withColumn("event_time", col("stream_processing_time"))
            .groupBy(window(col("event_time"), window_duration), col("political_type"))
            .agg(
                count("*").alias("article_count"),
                avg("overall_bias_score").alias("avg_overall_bias"),
                avg("political_bias").alias("avg_political_bias"),
                avg("gender_bias").alias("avg_gender_bias"),
            )
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("political_type"),
                col("article_count"),
                col("avg_overall_bias"),
                col("avg_political_bias"),
                col("avg_gender_bias"),
            )
        )

        # Write stats to console
        query = (
            stats_df.writeStream.format("console")
            .queryName("streaming_stats")
            .option("checkpointLocation", f"{self.checkpoint_location}/streaming_stats")
            .outputMode("complete")
            .trigger(processingTime="30 seconds")
            .start()
        )

        return query

    def run_streaming_job(
        self, enable_console_output: bool = False, enable_monitoring: bool = True
    ):
        """
        Run complete streaming processing job

        Args:
            enable_console_output: Enable console output for debugging
            enable_monitoring: Enable monitoring statistics
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting Streaming News Processing")
            logger.info(f"Time: {datetime.now().isoformat()}")
            logger.info("=" * 60)

            # Step 1: Read from Kafka stream
            stream_df = self.read_kafka_stream()

            # Step 2: Process and analyze
            analyzed_df = self.process_stream(stream_df)

            # Step 3: Write to MongoDB
            main_query = self.write_to_mongodb_stream(analyzed_df)

            queries = [main_query]

            # Optional: Console output for debugging
            if enable_console_output:
                console_query = self.write_to_console(
                    analyzed_df.select(
                        "title",
                        "media_name",
                        "overall_bias_score",
                        "political_type",
                        "gender_type",
                    ),
                    query_name="debug_console",
                )
                queries.append(console_query)

            # Optional: Monitoring statistics
            if enable_monitoring:
                stats_query = self.monitor_stream_stats(analyzed_df)
                queries.append(stats_query)

            logger.info("✓ All streaming queries started")
            logger.info("Press Ctrl+C to stop...")

            # Wait for termination
            for query in queries:
                query.awaitTermination()

        except KeyboardInterrupt:
            logger.info("\nStopping streaming queries...")
            for query in self.spark.streams.active:
                query.stop()
            logger.info("✓ All queries stopped")

        except Exception as e:
            logger.error(f"✗ Streaming job failed: {e}")
            raise

    def stop(self):
        """Stop all active streaming queries and Spark session"""
        for query in self.spark.streams.active:
            query.stop()
        self.spark.stop()
        logger.info("✓ Streaming processor stopped")


def main():
    """Main entry point for streaming processor"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Spark Streaming Processor for Real-Time News"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint location for streaming queries",
    )
    parser.add_argument(
        "--console", action="store_true", help="Enable console output for debugging"
    )
    parser.add_argument(
        "--no-monitoring", action="store_true", help="Disable monitoring statistics"
    )

    args = parser.parse_args()

    processor = StreamingNewsProcessor(checkpoint_location=args.checkpoint)

    try:
        processor.run_streaming_job(
            enable_console_output=args.console, enable_monitoring=not args.no_monitoring
        )
    finally:
        processor.stop()


if __name__ == "__main__":
    main()
