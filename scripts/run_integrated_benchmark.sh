#!/bin/bash

# Integrated Benchmark Runner
# Coordinates Producer + Streaming Consumer for complete pipeline testing

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
PERCENTAGE=""
BATCH_COUNT=""
CONSUMER_DURATION=1  # Flag to enable idle timeout mode (exits after 30s idle)
PRODUCER_BATCH_SIZE=100
SOURCE_URI="mongodb://localhost:27017"
SOURCE_DB="test"
SOURCE_COLLECTION="articles"
BOOTSTRAP_SERVERS="localhost:9092"
TOPIC="news"
CHECKPOINT="./checkpoints_benchmark"
OUTPUT="benchmark_results"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Kafka Streaming Benchmark (MongoDB → Kafka → Spark → MongoDB)
Consumer automatically exits when idle for 30 seconds (all messages consumed).

OPTIONS:
    -p, --percentage PCT        Percentage of data (1-100)
    -b, --batch-count COUNT     Number of records
    --producer-batch-size SIZE  Producer batch size before flush (default: 100)
    
    (Consumer exits automatically via 30s idle timeout - no duration limit)
    
    --source-uri URI            Source MongoDB URI (default: mongodb://localhost:27017)
    --source-db DB              Source database (default: test)
    --source-collection COLL    Source collection (default: articles)
    
    --bootstrap-servers SERVERS Kafka servers (default: localhost:9092)
    --topic TOPIC               Kafka topic (default: news)
    
    --checkpoint DIR            Checkpoint directory (default: ./checkpoints_benchmark)
    --output DIR                Output directory (default: benchmark_results)
    
    --clean-checkpoint          Clean checkpoint before running
    -h, --help                  Show this help

EXAMPLES:
    # Benchmark 10% of data (exits when all consumed - idle 30s)
    $0 --percentage 10

    # Benchmark 20% with custom producer batch size
    $0 --percentage 20 --producer-batch-size 1000

    # Benchmark 1000 records
    $0 --batch-count 1000

    # Clean checkpoint and run
    $0 --percentage 20 --clean-checkpoint

EOF
    exit 0
}

CLEAN_CHECKPOINT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--percentage)
            PERCENTAGE="$2"
            shift 2
            ;;
        -b|--batch-count)
            BATCH_COUNT="$2"
            shift 2
            ;;
        -d|--consumer-duration)
            CONSUMER_DURATION="$2"
            shift 2
            ;;
        --producer-batch-size)
            PRODUCER_BATCH_SIZE="$2"
            shift 2
            ;;
        --source-uri)
            SOURCE_URI="$2"
            shift 2
            ;;
        --source-db)
            SOURCE_DB="$2"
            shift 2
            ;;
        --source-collection)
            SOURCE_COLLECTION="$2"
            shift 2
            ;;
        --bootstrap-servers)
            BOOTSTRAP_SERVERS="$2"
            shift 2
            ;;
        --topic)
            TOPIC="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --clean-checkpoint)
            CLEAN_CHECKPOINT=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate
if [ -z "$PERCENTAGE" ] && [ -z "$BATCH_COUNT" ]; then
    print_error "Either --percentage or --batch-count is required"
    usage
fi

if [ ! -z "$PERCENTAGE" ] && [ ! -z "$BATCH_COUNT" ]; then
    print_error "Cannot specify both --percentage and --batch-count"
    usage
fi

# Clean checkpoint if requested
if [ "$CLEAN_CHECKPOINT" = true ]; then
    print_info "Cleaning checkpoint directory: $CHECKPOINT"
    rm -rf "$CHECKPOINT"
    print_success "Checkpoint cleaned"
fi

# Build command
CMD="python scripts/integrated_benchmark.py"
CMD="$CMD --consumer-duration $CONSUMER_DURATION"
CMD="$CMD --producer-batch-size $PRODUCER_BATCH_SIZE"
CMD="$CMD --source-uri $SOURCE_URI"
CMD="$CMD --source-db $SOURCE_DB"
CMD="$CMD --source-collection $SOURCE_COLLECTION"
CMD="$CMD --bootstrap-servers $BOOTSTRAP_SERVERS"
CMD="$CMD --topic $TOPIC"
CMD="$CMD --checkpoint $CHECKPOINT"
CMD="$CMD --output $OUTPUT"

if [ ! -z "$PERCENTAGE" ]; then
    CMD="$CMD --percentage $PERCENTAGE"
fi

if [ ! -z "$BATCH_COUNT" ]; then
    CMD="$CMD --batch-count $BATCH_COUNT"
fi

# Print configuration
print_info "========================================="
print_info "Kafka Streaming Benchmark"
print_info "========================================="
if [ ! -z "$PERCENTAGE" ]; then
    print_info "Data Selection:    $PERCENTAGE%"
else
    print_info "Data Selection:    $BATCH_COUNT records"
fi
print_info "Producer Batch:    ${PRODUCER_BATCH_SIZE} msgs/flush"
print_info "Consumer:          Idle timeout (30s)"
print_info "Source:            $SOURCE_URI/$SOURCE_DB.$SOURCE_COLLECTION"
print_info "Kafka:             $BOOTSTRAP_SERVERS/$TOPIC"
print_info "Output:            $OUTPUT"
print_info "========================================="

# Run benchmark
print_info "Starting integrated benchmark..."
echo ""

if $CMD; then
    echo ""
    print_success "Integrated benchmark completed!"
    
    # Show latest result file
    LATEST_RESULT=$(ls -t $OUTPUT/benchmark_report_*.json 2>/dev/null | head -1)
    if [ ! -z "$LATEST_RESULT" ]; then
        print_info "Result file: $LATEST_RESULT"
        
        # Show quick summary if jq is available
        if command -v jq &> /dev/null; then
            echo ""
            print_info "Quick Summary:"
            jq -r '
                "Streaming Consumer (Spark + Bias Detection):",
                "  Processed: \(.streaming_consumer_metrics.total_records_processed // 0) records",
                "  Written: \(.streaming_consumer_metrics.total_records_written // 0) records",
                "  Duration: \((.streaming_consumer_metrics.duration_seconds // 0) | floor)s",
                "  Throughput: \((.streaming_consumer_metrics.throughput_records_per_sec // 0) | floor) rec/sec",
                "  Avg Batch Processing: \((.streaming_consumer_metrics.avg_batch_processing_time_ms // 0) | floor)ms",
                "  Spark Executor Cores: \(.streaming_consumer_metrics.spark_config.executor_cores // 0)",
                "  Total CPU Cores: \(.streaming_consumer_metrics.spark_config.total_cpu_cores_available // 0)"
            ' "$LATEST_RESULT"
        fi
    fi
else
    echo ""
    print_error "Integrated benchmark failed!"
    exit 1
fi
