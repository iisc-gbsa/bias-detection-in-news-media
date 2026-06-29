"""
Kafka Streaming Benchmark
Tests the complete pipeline: MongoDB → Kafka → Spark Streaming → MongoDB
Focuses on actual Spark Structured Streaming performance with bias detection
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import threading
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedBenchmark:
    """Coordinates producer and streaming consumer benchmark"""

    def __init__(
        self,
        source_mongo_uri: str,
        source_mongo_db: str,
        source_mongo_collection: str,
        percentage: float,
        batch_count: Optional[int],
        kafka_topic: str,
        bootstrap_servers: str,
        checkpoint_location: str,
        consumer_duration: int,
        producer_batch_size: int,
        output_dir: str,
    ):
        self.source_mongo_uri = source_mongo_uri
        self.source_mongo_db = source_mongo_db
        self.source_mongo_collection = source_mongo_collection
        self.percentage = percentage
        self.batch_count = batch_count
        self.kafka_topic = kafka_topic
        self.bootstrap_servers = bootstrap_servers
        self.checkpoint_location = checkpoint_location
        self.consumer_duration = consumer_duration
        self.producer_batch_size = producer_batch_size
        self.output_dir = output_dir

        # Metrics files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.consumer_metrics_file = (
            f"{output_dir}/consumer_metrics_{self.timestamp}.json"
        )
        self.unified_report_file = (
            f"{output_dir}/benchmark_report_{self.timestamp}.json"
        )

        self.consumer_metrics = {}

    def clean_kafka_topic(self):
        """Clean Kafka topic before benchmark (optional)"""
        logger.info(f"Attempting to clean Kafka topic: {self.kafka_topic}")

        # Try to find kafka-topics command
        kafka_cmd = None
        for cmd_name in ["kafka-topics.sh", "kafka-topics"]:
            try:
                result = subprocess.run(
                    ["which", cmd_name], capture_output=True, timeout=2
                )
                if result.returncode == 0:
                    kafka_cmd = cmd_name
                    break
            except:
                continue

        if not kafka_cmd:
            logger.info("⚠ Kafka tools not found in PATH - skipping topic cleanup")
            logger.info("  (This is optional - benchmark will continue)")
            return

        try:
            # Delete topic
            cmd = [
                kafka_cmd,
                "--bootstrap-server",
                self.bootstrap_servers,
                "--delete",
                "--topic",
                self.kafka_topic,
            ]
            subprocess.run(cmd, capture_output=True, timeout=10)
            time.sleep(2)  # Wait for deletion

            # Recreate topic
            cmd = [
                kafka_cmd,
                "--bootstrap-server",
                self.bootstrap_servers,
                "--create",
                "--topic",
                self.kafka_topic,
                "--partitions",
                "3",
                "--replication-factor",
                "1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                logger.info("✓ Kafka topic cleaned and recreated")
            else:
                logger.info("⚠ Could not clean topic (may already exist)")
        except Exception as e:
            logger.info(f"⚠ Topic cleanup skipped: {e}")
            logger.info("  (This is optional - benchmark will continue)")

    def run_producer(self) -> bool:
        """Run simple producer (sends to Kafka only)"""
        logger.info("=" * 80)
        logger.info("PRODUCER: MongoDB → Kafka")
        logger.info("=" * 80)

        try:
            cmd = [
                "python",
                "scripts/simple_producer.py",
                "--mongo-uri",
                self.source_mongo_uri,
                "--mongo-db",
                self.source_mongo_db,
                "--mongo-collection",
                self.source_mongo_collection,
                "--kafka-servers",
                self.bootstrap_servers,
                "--kafka-topic",
                self.kafka_topic,
                "--producer-batch-size",
                str(self.producer_batch_size),
            ]

            if self.batch_count:
                cmd.extend(["--batch-count", str(self.batch_count)])
            else:
                cmd.extend(["--percentage", str(self.percentage)])

            logger.info(f"Running producer: {' '.join(cmd)}")

            # Run producer with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output in real-time
            for line in process.stdout:
                if line.strip():
                    logger.info(f"[PRODUCER] {line.rstrip()}")

            # Wait for completion
            return_code = process.wait()

            if return_code == 0:
                logger.info("✓ Producer completed")
                return True
            else:
                logger.error(f"✗ Producer failed with exit code: {return_code}")
                return False

        except Exception as e:
            logger.error(f"✗ Producer error: {e}")
            return False

    def run_streaming_consumer(self) -> bool:
        """Run Spark streaming processor consumer"""
        logger.info("\n" + "=" * 80)
        logger.info("STREAMING CONSUMER: Kafka → Spark → MongoDB")
        logger.info("=" * 80)

        try:
            # Wait a bit for messages to be in Kafka
            logger.info("Waiting 5 seconds for messages to be available in Kafka...")
            time.sleep(5)

            cmd = [
                "python",
                "-m",
                "src.processing.streaming.streaming_processor",
                "--checkpoint",
                self.checkpoint_location,
                "--duration",
                str(self.consumer_duration),
                "--save-metrics",
                self.consumer_metrics_file,
                "--no-monitoring",  # Disable stats query for cleaner benchmark
            ]

            logger.info(f"Running streaming consumer: {' '.join(cmd)}")
            logger.info(
                f"Note: Consumer will exit automatically after 30s idle (all messages consumed)"
            )

            # Run streaming processor with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output in real-time until process finishes
            try:
                for line in process.stdout:
                    if line.strip():
                        logger.info(f"[CONSUMER] {line.rstrip()}")
            except KeyboardInterrupt:
                logger.info("\nUser interrupted, stopping consumer...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0 or return_code is None or return_code < 0:
                logger.info("✓ Streaming consumer benchmark completed")

                # Load consumer metrics
                if os.path.exists(self.consumer_metrics_file):
                    with open(self.consumer_metrics_file, "r") as f:
                        data = json.load(f)
                        self.consumer_metrics = data.get("metrics", {})

                return True
            else:
                logger.warning(f"⚠ Streaming consumer exit code: {return_code}")

                # Still try to load metrics if file exists
                if os.path.exists(self.consumer_metrics_file):
                    with open(self.consumer_metrics_file, "r") as f:
                        data = json.load(f)
                        self.consumer_metrics = data.get("metrics", {})
                    return True

                return False

        except subprocess.TimeoutExpired:
            logger.info("✓ Consumer duration completed")

            # Load metrics if available
            if os.path.exists(self.consumer_metrics_file):
                with open(self.consumer_metrics_file, "r") as f:
                    data = json.load(f)
                    self.consumer_metrics = data.get("metrics", {})

            return True

        except Exception as e:
            logger.error(f"✗ Streaming consumer error: {e}")
            return False

    def generate_unified_report(self):
        """Generate benchmark report"""
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK REPORT")
        logger.info("=" * 80)

        # Generate report with streaming consumer metrics
        report = {
            "benchmark_id": self.timestamp,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "source_mongo_uri": self.source_mongo_uri,
                "source_mongo_db": self.source_mongo_db,
                "source_mongo_collection": self.source_mongo_collection,
                "percentage": self.percentage,
                "batch_count": self.batch_count,
                "producer_batch_size": self.producer_batch_size,
                "kafka_topic": self.kafka_topic,
                "bootstrap_servers": self.bootstrap_servers,
                "consumer_duration": self.consumer_duration,
            },
            "streaming_consumer_metrics": self.consumer_metrics,
        }

        # Save report
        with open(self.unified_report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n✓ Unified report saved to: {self.unified_report_file}")

        # Print summary
        self._print_summary(report)

    def _print_summary(self, report: Dict):
        """Print summary to console"""
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        metrics = report.get("streaming_consumer_metrics", {})

        if metrics:
            # Streaming Consumer Performance
            logger.info("\n📊 Streaming Consumer (Spark + Bias Detection):")
            logger.info(
                f"  Records Processed: {metrics.get('total_records_processed', 0)}"
            )
            logger.info(f"  Records Written: {metrics.get('total_records_written', 0)}")
            logger.info(f"  Failed Records: {metrics.get('failed_records', 0)}")
            logger.info(f"  Duration: {metrics.get('duration_seconds', 0):.2f} seconds")

            # Throughput
            logger.info("\n⚡ Throughput:")
            logger.info(
                f"  {metrics.get('throughput_records_per_sec', 0):.2f} records/sec"
            )

            # Processing Times
            logger.info("\n⏱️  Processing Times:")
            logger.info(
                f"  Avg Batch Processing: {metrics.get('avg_batch_processing_time_ms', 0):.2f} ms"
            )
            logger.info(
                f"  Avg MongoDB Write: {metrics.get('avg_mongodb_write_time_ms', 0):.2f} ms"
            )
            logger.info(
                f"  Avg Batch Size: {metrics.get('avg_batch_size', 0):.1f} records"
            )

            # Spark Executor Resource Usage
            logger.info("\n💻 Spark Executor Metrics:")
            logger.info(
                f"  Avg CPU Utilization: {metrics.get('avg_executor_cpu_utilization_percent', 0):.2f}%"
            )
            logger.info(
                f"  Total CPU Time: {metrics.get('total_executor_cpu_time_seconds', 0):.2f} seconds"
            )
            logger.info(
                f"  Avg Executor Memory: {metrics.get('avg_executor_memory_used_mb', 0):.2f} MB"
            )
            logger.info(f"  Avg Active Tasks: {metrics.get('avg_active_tasks', 0):.1f}")

            # Spark Configuration
            spark_config = metrics.get("spark_config", {})
            if spark_config:
                logger.info("\n🔧 Spark Configuration:")
                logger.info(
                    f"  Executor Cores: {spark_config.get('executor_cores', 0)}"
                )
                logger.info(f"  Num Executors: {spark_config.get('num_executors', 0)}")
                logger.info(
                    f"  Executor Memory: {spark_config.get('executor_memory', 'N/A')}"
                )
                logger.info(
                    f"  Total CPU Cores: {spark_config.get('total_cpu_cores_available', 0)}"
                )

        logger.info("\n" + "=" * 80)

    def run(self):
        """Run complete integrated benchmark"""
        try:
            logger.info("=" * 80)
            logger.info("KAFKA STREAMING BENCHMARK")
            logger.info("MongoDB → Kafka → Spark Streaming → MongoDB")
            logger.info("=" * 80)
            logger.info(f"Dataset: {self.percentage}% or {self.batch_count} records")
            logger.info(
                f"Source: {self.source_mongo_db}.{self.source_mongo_collection}"
            )
            logger.info(f"Producer Batch Size: {self.producer_batch_size} messages")
            logger.info(f"Consumer: Idle timeout mode (exits after 30s idle)")
            logger.info("=" * 80)

            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Clean Kafka topic
            self.clean_kafka_topic()

            # Step 1: Run producer (MongoDB → Kafka)
            if not self.run_producer():
                logger.error("✗ Producer failed, aborting")
                return False

            # Step 2: Run streaming consumer (Kafka → Spark → MongoDB)
            if not self.run_streaming_consumer():
                logger.error("✗ Streaming consumer failed")
                # Continue to generate report anyway

            # Step 3: Generate benchmark report
            self.generate_unified_report()

            logger.info("\n✓ Benchmark completed successfully!")
            return True

        except Exception as e:
            logger.error(f"✗ Integrated benchmark failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Integrated Kafka Producer + Streaming Consumer Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark 20% of data with 60s consumer run
  python scripts/integrated_benchmark.py --percentage 20 --consumer-duration 60
  
  # Benchmark specific count with 120s consumer run
  python scripts/integrated_benchmark.py --batch-count 1000 --consumer-duration 120
  
  # Custom configuration
  python scripts/integrated_benchmark.py \
    --percentage 30 \
    --consumer-duration 90 \
    --source-db production \
    --source-collection articles \
    --output results/my_benchmark
        """,
    )

    # Data selection
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--percentage", type=float, help="Percentage of data to process (1-100)"
    )
    data_group.add_argument(
        "--batch-count", type=int, help="Number of records to process"
    )

    # Consumer settings
    parser.add_argument(
        "--consumer-duration",
        type=int,
        default=1,
        help="Enable idle timeout mode - exits when no messages for 30s (default: enabled)",
    )

    # MongoDB source
    parser.add_argument(
        "--source-uri",
        type=str,
        default="mongodb://localhost:27017",
        help="Source MongoDB URI (default: mongodb://localhost:27017)",
    )
    parser.add_argument(
        "--source-db", type=str, default="test", help="Source database (default: test)"
    )
    parser.add_argument(
        "--source-collection",
        type=str,
        default="articles",
        help="Source collection (default: articles)",
    )

    # Kafka settings
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--topic", type=str, default="news", help="Kafka topic (default: news)"
    )
    parser.add_argument(
        "--producer-batch-size",
        type=int,
        default=100,
        help="Number of messages to send before flushing (default: 100)",
    )

    # Other settings
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints_benchmark",
        help="Checkpoint location (default: ./checkpoints_benchmark)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory (default: benchmark_results)",
    )

    args = parser.parse_args()

    # Validate
    if args.percentage and (args.percentage < 1 or args.percentage > 100):
        parser.error("Percentage must be between 1 and 100")

    if args.batch_count and args.batch_count < 1:
        parser.error("Batch count must be positive")

    # Run benchmark
    benchmark = IntegratedBenchmark(
        source_mongo_uri=args.source_uri,
        source_mongo_db=args.source_db,
        source_mongo_collection=args.source_collection,
        percentage=args.percentage if args.percentage else 100.0,
        batch_count=args.batch_count,
        kafka_topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        checkpoint_location=args.checkpoint,
        consumer_duration=args.consumer_duration,
        producer_batch_size=args.producer_batch_size,
        output_dir=args.output,
    )

    success = benchmark.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
