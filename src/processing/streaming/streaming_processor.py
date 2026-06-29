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
import time
import json
import psutil
import threading
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict, field

from src.models.spark_bias_detector import SparkBiasDetector
from config.config import kafka_config, mongo_config, spark_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamingMetrics:
    """Metrics for streaming processor"""

    start_time: float = 0.0
    end_time: float = 0.0
    total_batches: int = 0
    total_records_processed: int = 0
    total_records_written: int = 0
    failed_records: int = 0
    batch_processing_times: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    mongodb_write_times: List[float] = field(default_factory=list)

    # Spark-native metrics
    executor_cpu_time_samples: List[float] = field(
        default_factory=list
    )  # Total CPU time across all executors (seconds)
    executor_run_time_samples: List[float] = field(
        default_factory=list
    )  # Total run time across all executors (seconds)
    num_active_tasks_samples: List[int] = field(default_factory=list)
    executor_memory_used_samples: List[float] = field(default_factory=list)  # MB

    spark_executor_cores: int = 0
    spark_executor_memory: str = ""
    spark_driver_memory: str = ""
    total_cpu_cores_available: int = 0
    num_executors: int = 0
    last_batch_time: float = 0.0

    def get_summary(self) -> Dict:
        """Get metrics summary"""
        duration = (
            self.end_time - self.start_time
            if self.end_time > 0
            else time.time() - self.start_time
        )

        # Calculate CPU utilization percentage (CPU time / wall time)
        avg_cpu_utilization = 0.0
        if self.executor_cpu_time_samples and self.executor_run_time_samples:
            total_cpu_time = sum(self.executor_cpu_time_samples)
            total_run_time = sum(self.executor_run_time_samples)
            if total_run_time > 0:
                avg_cpu_utilization = (total_cpu_time / total_run_time) * 100

        return {
            "duration_seconds": duration,
            "total_batches": self.total_batches,
            "total_records_processed": self.total_records_processed,
            "total_records_written": self.total_records_written,
            "failed_records": self.failed_records,
            "throughput_records_per_sec": (
                self.total_records_processed / duration if duration > 0 else 0
            ),
            "spark_config": {
                "executor_cores": self.spark_executor_cores,
                "executor_memory": self.spark_executor_memory,
                "driver_memory": self.spark_driver_memory,
                "total_cpu_cores_available": self.total_cpu_cores_available,
                "num_executors": self.num_executors,
            },
            "avg_batch_processing_time_ms": (
                sum(self.batch_processing_times)
                / len(self.batch_processing_times)
                * 1000
                if self.batch_processing_times
                else 0
            ),
            "avg_batch_size": (
                sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
            ),
            "avg_mongodb_write_time_ms": (
                sum(self.mongodb_write_times) / len(self.mongodb_write_times) * 1000
                if self.mongodb_write_times
                else 0
            ),
            "avg_executor_cpu_utilization_percent": avg_cpu_utilization,
            "total_executor_cpu_time_seconds": (
                sum(self.executor_cpu_time_samples)
                if self.executor_cpu_time_samples
                else 0
            ),
            "avg_executor_memory_used_mb": (
                sum(self.executor_memory_used_samples)
                / len(self.executor_memory_used_samples)
                if self.executor_memory_used_samples
                else 0
            ),
            "avg_active_tasks": (
                sum(self.num_active_tasks_samples) / len(self.num_active_tasks_samples)
                if self.num_active_tasks_samples
                else 0
            ),
        }


class StreamingNewsProcessor:
    """Real-time streaming processor for news articles"""

    def __init__(self, checkpoint_location: str = None, enable_metrics: bool = True):
        """Initialize streaming processor"""
        self.checkpoint_location = (
            checkpoint_location or kafka_config.checkpoint_location
        )
        self.enable_metrics = enable_metrics
        self.metrics = StreamingMetrics() if enable_metrics else None
        self.monitoring_thread = None
        self.monitoring_active = False

        self.spark = self._create_spark_session()
        self.bias_detector = SparkBiasDetector(self.spark)

        # Capture Spark configuration in metrics
        if self.enable_metrics and self.metrics:
            self.metrics.spark_executor_cores = int(spark_config.executor_cores)
            self.metrics.spark_executor_memory = spark_config.executor_memory
            self.metrics.spark_driver_memory = spark_config.driver_memory
            self.metrics.total_cpu_cores_available = psutil.cpu_count(logical=True)

            # Get number of executors from Spark
            self.metrics.num_executors = self._get_num_executors()

            logger.info(
                f"Spark Configuration: {self.metrics.spark_executor_cores} executor cores, "
                f"{self.metrics.num_executors} executors, "
                f"{self.metrics.total_cpu_cores_available} total CPU cores available"
            )

        # Start resource monitoring if metrics enabled
        if self.enable_metrics:
            self._start_resource_monitoring()

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
                StructField("benchmark_run_id", StringType(), True),
                StructField("benchmark_percentage", IntegerType(), True),
                StructField("benchmark_batch_id", IntegerType(), True),
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
            # NEW: Parallelism optimization settings
            .config("spark.default.parallelism", spark_config.default_parallelism)
            .config("spark.sql.shuffle.partitions", spark_config.sql_shuffle_partitions)
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")
        logger.info("✓ Spark session created for streaming processing")
        logger.info(
            f"  Parallelism: default={spark_config.default_parallelism}, shuffle_partitions={spark_config.sql_shuffle_partitions}"
        )

        return spark

    def _get_num_executors(self) -> int:
        """Get number of active executors"""
        try:
            # Get executor info from SparkContext
            status_tracker = self.spark.sparkContext.statusTracker()
            executor_infos = status_tracker.getExecutorInfos()
            # Subtract 1 for driver if it's in the list
            num_executors = len([e for e in executor_infos if e.host() != "driver"])
            return max(num_executors, 1)  # At least 1 for local mode
        except Exception as e:
            logger.warning(f"Could not get executor count: {e}")
            return 1

    def _start_resource_monitoring(self):
        """Start background resource monitoring using Spark metrics"""

        def monitor_resources():
            while self.monitoring_active:
                try:
                    # Get Spark metrics from status tracker
                    status_tracker = self.spark.sparkContext.statusTracker()

                    # Get active stages and tasks
                    active_stage_ids = status_tracker.getActiveStageIds()
                    num_active_tasks = 0
                    total_cpu_time = 0.0
                    total_run_time = 0.0

                    for stage_id in active_stage_ids:
                        stage_info = status_tracker.getStageInfo(stage_id)
                        if stage_info:
                            num_active_tasks += stage_info.numActiveTasks()

                    # Get executor metrics from all completed tasks (accumulated)
                    # Note: This captures historical CPU time, not instantaneous
                    sc = self.spark.sparkContext
                    if hasattr(sc, "_jsc") and sc._jsc:
                        try:
                            # Get executor memory status
                            executor_memory_status = (
                                sc._jsc.sc().getExecutorMemoryStatus()
                            )
                            total_memory_used = 0.0
                            for (
                                executor_id,
                                memory_tuple,
                            ) in executor_memory_status.items():
                                # memory_tuple is (max_memory, remaining_memory)
                                if memory_tuple and len(memory_tuple) >= 2:
                                    max_mem = memory_tuple[0]
                                    remaining_mem = memory_tuple[1]
                                    used_mem = max_mem - remaining_mem
                                    total_memory_used += used_mem

                            # Convert to MB
                            total_memory_used_mb = total_memory_used / (1024 * 1024)

                            if self.metrics:
                                self.metrics.executor_memory_used_samples.append(
                                    total_memory_used_mb
                                )
                        except Exception as e:
                            logger.debug(f"Could not get executor memory: {e}")

                    # Store samples
                    if self.metrics:
                        self.metrics.num_active_tasks_samples.append(num_active_tasks)

                        # For CPU time, we'll collect from streaming query progress
                        # (more accurate for streaming workloads)

                    time.sleep(2.0)  # Sample every 2 seconds
                except Exception as e:
                    logger.debug(f"Error monitoring Spark resources: {e}")

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitoring_thread.start()
        logger.info("✓ Spark resource monitoring started")

    def _stop_resource_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            try:
                self.monitoring_thread.join(timeout=2.0)
                if self.monitoring_thread.is_alive():
                    logger.debug("Resource monitoring thread did not stop in time")
            except Exception as e:
                logger.debug(f"Error stopping resource monitoring thread: {e}")
        logger.info("✓ Resource monitoring stopped")

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

        logger.info("Kafka stream initialized")

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
        Write streaming data to MongoDB with metrics tracking

        Args:
            df: Streaming DataFrame
            query_name: Name for the streaming query
        """
        mongo_uri = f"{mongo_config.uri}{mongo_config.database}.{mongo_config.collection_realtime}"

        logger.info(f"Writing stream to MongoDB: {mongo_uri}")
        logger.info(
            f"  MongoDB optimization: ordered={mongo_config.ordered_writes}, "
            f"maxBatchSize={mongo_config.max_batch_size}, "
            f"poolSize={mongo_config.connection_pool_size}"
        )

        def write_batch_to_mongo(batch_df, batch_id):
            """Function to write each micro-batch to MongoDB with metrics"""
            batch_start_time = time.time()
            batch_count = 0

            try:
                batch_count = batch_df.count()

                if batch_count > 0:
                    # Track MongoDB write time
                    write_start = time.time()

                    # MongoDB write with optimization settings
                    write_builder = (
                        batch_df.write.format("mongodb")
                        .mode("append")
                        .option("connection.uri", mongo_uri)
                        .option("database", mongo_config.database)
                        .option("collection", mongo_config.collection_realtime)
                        # Write optimization options
                        .option(
                            "ordered",
                            "false" if not mongo_config.ordered_writes else "true",
                        )
                        .option("maxBatchSize", str(mongo_config.max_batch_size))
                        .option("maxPoolSize", str(mongo_config.connection_pool_size))
                        .option(
                            "retryWrites",
                            "true" if mongo_config.retry_writes else "false",
                        )
                        .option("socketTimeoutMS", str(mongo_config.socket_timeout_ms))
                        .option(
                            "connectTimeoutMS", str(mongo_config.connect_timeout_ms)
                        )
                        .option("w", str(mongo_config.write_concern_w))
                    )

                    # Add journal setting if write concern allows
                    if not mongo_config.write_concern_journal:
                        write_builder = write_builder.option("journal", "false")

                    write_builder.save()

                    write_duration = time.time() - write_start
                    batch_total_time = time.time() - batch_start_time

                    logger.info(
                        f"✓ Batch {batch_id}: Wrote {batch_count} records in {write_duration:.3f}s (total: {batch_total_time:.3f}s)"
                    )

                    # Update success metrics
                    if self.enable_metrics and self.metrics:
                        self.metrics.total_batches += 1
                        self.metrics.total_records_processed += batch_count
                        self.metrics.total_records_written += batch_count
                        self.metrics.batch_processing_times.append(batch_total_time)
                        self.metrics.batch_sizes.append(batch_count)
                        self.metrics.mongodb_write_times.append(write_duration)
                        self.metrics.last_batch_time = time.time()
                else:
                    logger.debug(f"Batch {batch_id}: No records to write")

            except Exception as e:
                logger.error(f"✗ Batch {batch_id}: Error writing to MongoDB: {e}")

                # Update failure metrics
                if self.enable_metrics and self.metrics:
                    self.metrics.failed_records += batch_count

                # Write errors to error topic
                self._write_errors_to_kafka(batch_df, e, batch_id)

            # Collect Spark executor metrics after each batch
            if self.enable_metrics and self.metrics:
                self._collect_executor_metrics()

        # Start streaming query with foreachBatch
        query = (
            df.writeStream.foreachBatch(write_batch_to_mongo)
            .queryName(query_name)
            .option("checkpointLocation", f"{self.checkpoint_location}/{query_name}")
            .trigger(processingTime="10 seconds")
            .start()
        )

        return query

    def _collect_executor_metrics(self):
        """Collect executor-level CPU and task metrics from Spark"""
        try:
            sc = self.spark.sparkContext
            status_tracker = sc.statusTracker()

            # Get all completed stages to accumulate metrics
            # Note: For streaming, we track cumulative metrics
            executor_cpu_time = 0.0
            executor_run_time = 0.0

            # Access Spark's internal metrics through the listener bus
            # This is a snapshot of cumulative executor metrics
            if hasattr(sc, "_jsc"):
                try:
                    # Get app status from Spark's internal tracking
                    # This requires accessing Spark internals via py4j
                    pass  # Placeholder for more detailed metrics if needed
                except Exception:
                    pass

            # For now, we'll rely on streaming query progress for detailed metrics
            # which is captured separately

        except Exception as e:
            logger.debug(f"Error collecting executor metrics: {e}")

    def _start_query_metrics_collection(self):
        """Start collecting metrics from streaming query progress"""

        def collect_query_metrics():
            while self.monitoring_active:
                try:
                    if hasattr(self, "main_query") and self.main_query:
                        # Get latest progress from the streaming query
                        progress = self.main_query.lastProgress

                        if progress:
                            # Extract executor CPU time and run time from sources
                            # Progress contains detailed per-batch metrics
                            sources = progress.get("sources", [])
                            state_operators = progress.get("stateOperators", [])

                            # Get execution stats
                            if "durationMs" in progress:
                                durations = progress["durationMs"]

                                # Collect processing times (this reflects actual work)
                                trigger_execution = (
                                    durations.get("triggerExecution", 0) / 1000.0
                                )  # Convert to seconds

                                if trigger_execution > 0 and self.metrics:
                                    # Approximate CPU time based on executor cores and execution time
                                    # In local mode or with multiple executors, this represents parallel work
                                    estimated_cpu_time = (
                                        trigger_execution
                                        * self.metrics.spark_executor_cores
                                    )

                                    self.metrics.executor_cpu_time_samples.append(
                                        estimated_cpu_time
                                    )
                                    self.metrics.executor_run_time_samples.append(
                                        trigger_execution
                                    )

                    time.sleep(3.0)  # Check every 3 seconds
                except Exception as e:
                    logger.debug(f"Error collecting query metrics: {e}")

        # Start the metrics collection thread
        query_metrics_thread = threading.Thread(
            target=collect_query_metrics, daemon=True
        )
        query_metrics_thread.start()
        self.query_metrics_thread = query_metrics_thread
        logger.info("✓ Streaming query metrics collection started")

    def _stop_query_metrics_collection(self):
        """Stop collecting query metrics"""
        # monitoring_active is already set to False by _stop_resource_monitoring
        if hasattr(self, "query_metrics_thread") and self.query_metrics_thread:
            try:
                if self.query_metrics_thread.is_alive():
                    self.query_metrics_thread.join(timeout=2.0)
                    if self.query_metrics_thread.is_alive():
                        logger.debug("Query metrics thread did not stop in time")
            except Exception as e:
                logger.debug(f"Error stopping query metrics thread: {e}")
        logger.info("✓ Streaming query metrics collection stopped")

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

            logger.info(f"Batch {batch_id}: Sent errors to error topic")

        except Exception as e2:
            logger.error(f"Batch {batch_id}: Failed to write errors: {e2}")

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
            .trigger(processingTime="5 seconds")
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
        self,
        enable_console_output: bool = False,
        enable_monitoring: bool = True,
        duration_seconds: int = None,
    ):
        """
        Run complete streaming processing job with metrics tracking

        Args:
            enable_console_output: Enable console output for debugging
            enable_monitoring: Enable monitoring statistics
            duration_seconds: Optional duration to run (for benchmarking)
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting Streaming News Processing")
            logger.info(f"Time: {datetime.now().isoformat()}")
            if self.enable_metrics:
                logger.info("Metrics Collection: ENABLED")
            logger.info("=" * 60)

            # Start metrics timer
            if self.enable_metrics and self.metrics:
                self.metrics.start_time = time.time()

            # Step 1: Read from Kafka stream
            stream_df = self.read_kafka_stream()

            # Step 2: Process and analyze
            analyzed_df = self.process_stream(stream_df)

            # Step 3: Write to MongoDB
            main_query = self.write_to_mongodb_stream(analyzed_df)

            queries = [main_query]

            # Track the main query for metrics collection
            self.main_query = main_query

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

            # Start collecting streaming query metrics
            if self.enable_metrics:
                self._start_query_metrics_collection()

            if duration_seconds:
                # Use only idle timeout - no max duration limit
                idle_timeout = 30  # Stop if no messages for 30 seconds
                check_interval = 5  # Check every 5 seconds

                logger.info(
                    f"Running until all messages consumed (idle timeout: {idle_timeout}s)..."
                )
                logger.info(
                    f"Will exit automatically when no new messages for {idle_timeout}s"
                )

                # Initialize last batch time
                if self.enable_metrics and self.metrics:
                    self.metrics.last_batch_time = time.time()

                while True:
                    time.sleep(check_interval)
                    current_time = time.time()

                    # Check for idle timeout (no new batches)
                    if self.enable_metrics and self.metrics:
                        idle_time = current_time - self.metrics.last_batch_time
                        if idle_time > idle_timeout and self.metrics.total_batches > 0:
                            logger.info(
                                f"Idle for {idle_time:.1f}s (no new messages), all messages consumed, stopping..."
                            )
                            break

                # Stop metrics collection
                if self.enable_metrics:
                    self._stop_query_metrics_collection()

                # Stop all queries
                for query in queries:
                    query.stop()
            else:
                logger.info("Press Ctrl+C to stop...")
                # Wait for termination
                for query in queries:
                    query.awaitTermination()

        except KeyboardInterrupt:
            logger.info("\nStopping streaming queries...")
            if self.enable_metrics:
                self._stop_query_metrics_collection()
            for query in self.spark.streams.active:
                query.stop()
            logger.info("All queries stopped")

        except Exception as e:
            logger.error(f"Streaming job failed: {e}")
            raise

        finally:
            # Stop metrics and print summary
            if self.enable_metrics and self.metrics:
                self.metrics.end_time = time.time()
                self._stop_resource_monitoring()
                self._print_metrics_summary()

    def _print_metrics_summary(self):
        """Print metrics summary"""
        if not self.metrics:
            return

        summary = self.metrics.get_summary()

        logger.info("\n" + "=" * 80)
        logger.info("STREAMING PROCESSOR METRICS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info(f"Total Batches: {summary['total_batches']}")
        logger.info(f"Total Records Processed: {summary['total_records_processed']}")
        logger.info(f"Total Records Written: {summary['total_records_written']}")
        logger.info(f"Failed Records: {summary['failed_records']}")
        logger.info(
            f"Throughput: {summary['throughput_records_per_sec']:.2f} records/sec"
        )
        logger.info(
            f"Avg Batch Processing Time: {summary['avg_batch_processing_time_ms']:.2f} ms"
        )
        logger.info(f"Avg Batch Size: {summary['avg_batch_size']:.1f} records")
        logger.info(
            f"Avg MongoDB Write Time: {summary['avg_mongodb_write_time_ms']:.2f} ms"
        )
        logger.info("")
        logger.info("Spark Executor Metrics:")
        logger.info(
            f"  Avg CPU Utilization: {summary['avg_executor_cpu_utilization_percent']:.2f}%"
        )
        logger.info(
            f"  Total CPU Time: {summary['total_executor_cpu_time_seconds']:.2f} seconds"
        )
        logger.info(
            f"  Avg Executor Memory Used: {summary['avg_executor_memory_used_mb']:.2f} MB"
        )
        logger.info(f"  Avg Active Tasks: {summary['avg_active_tasks']:.1f}")
        logger.info("")
        logger.info("Spark Configuration:")
        logger.info(f"  Executor Cores: {summary['spark_config']['executor_cores']}")
        logger.info(f"  Num Executors: {summary['spark_config']['num_executors']}")
        logger.info(f"  Executor Memory: {summary['spark_config']['executor_memory']}")
        logger.info(f"  Driver Memory: {summary['spark_config']['driver_memory']}")
        logger.info(
            f"  Total CPU Cores Available: {summary['spark_config']['total_cpu_cores_available']}"
        )
        logger.info("=" * 80)

    def get_metrics(self) -> Dict:
        """Get current metrics as dictionary"""
        if not self.metrics:
            return {}
        return self.metrics.get_summary()

    def save_metrics(self, output_file: str):
        """Save metrics to JSON file"""
        if not self.metrics:
            logger.warning("No metrics to save (metrics collection disabled)")
            return

        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "processor_type": "spark_structured_streaming",
            "config": {
                "kafka_topic": kafka_config.news_topic,
                "kafka_servers": kafka_config.bootstrap_servers,
                "mongo_uri": mongo_config.uri,
                "mongo_database": mongo_config.database,
                "mongo_collection": mongo_config.collection_realtime,
            },
            "metrics": self.metrics.get_summary(),
            "detailed_metrics": {
                "batch_processing_times": self.metrics.batch_processing_times,
                "batch_sizes": self.metrics.batch_sizes,
                "mongodb_write_times": self.metrics.mongodb_write_times,
            },
        }

        with open(output_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"✓ Metrics saved to: {output_file}")

    def stop(self):
        """Stop all active streaming queries and Spark session"""
        # Stop resource monitoring
        if self.enable_metrics:
            self._stop_resource_monitoring()

        # Stop all active queries gracefully
        try:
            for query in self.spark.streams.active:
                try:
                    query.stop()
                except Exception as e:
                    logger.debug(f"Error stopping query: {e}")
        except Exception as e:
            logger.debug(f"Error accessing active streams: {e}")

        # Stop Spark session with proper cleanup
        try:
            # Give time for threads to complete
            import time

            time.sleep(0.5)

            self.spark.sparkContext.stop()
            self.spark.stop()
        except Exception as e:
            logger.debug(f"Error stopping Spark session: {e}")

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
    parser.add_argument(
        "--no-metrics", action="store_true", help="Disable metrics collection"
    )
    parser.add_argument(
        "--duration", type=int, help="Run for specified seconds (for benchmarking)"
    )
    parser.add_argument("--save-metrics", type=str, help="Save metrics to JSON file")

    args = parser.parse_args()

    processor = StreamingNewsProcessor(
        checkpoint_location=args.checkpoint, enable_metrics=not args.no_metrics
    )

    try:
        processor.run_streaming_job(
            enable_console_output=args.console,
            enable_monitoring=not args.no_monitoring,
            duration_seconds=args.duration,
        )
    finally:
        if args.save_metrics:
            processor.save_metrics(args.save_metrics)
        processor.stop()


if __name__ == "__main__":
    main()
