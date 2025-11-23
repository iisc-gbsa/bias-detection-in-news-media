#!/bin/bash
# Bias Detection Pipeline Runner
# Usage: ./run_pipeline.sh [batch|stream|error|producer|all]

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    log_info "Docker is running"
}

# Start infrastructure
start_infrastructure() {
    log_info "Starting Kafka and MongoDB..."
    docker-compose up -d
    
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check Kafka
    if docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; then
        log_info "Kafka is ready"
    else
        log_warn "Kafka might not be ready yet"
    fi
    
    # Check MongoDB
    if docker exec mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        log_info "MongoDB is ready"
    else
        log_warn "MongoDB might not be ready yet"
    fi
}

# Setup Kafka topics
setup_topics() {
    log_info "Setting up Kafka topics..."
    python setup_kafka_topics.py
}

# Run batch processor
run_batch() {
    log_info "Running batch processor..."
    python batch_processor.py "$@"
}

# Run streaming processor
run_stream() {
    log_info "Running streaming processor..."
    python streaming_processor.py "$@"
}

# Run error handler
run_error_handler() {
    log_info "Running error handler..."
    python error_handler.py "$@"
}

# Run Kafka producer
run_producer() {
    log_info "Running Kafka producer..."
    python kafka_producer.py "$@"
}

# Stop everything
stop_all() {
    log_info "Stopping all services..."
    docker-compose down
    log_info "All services stopped"
}

# Show help
show_help() {
    cat << EOF
Bias Detection Pipeline Runner

Usage: ./run_pipeline.sh [COMMAND] [OPTIONS]

Commands:
    setup           Setup infrastructure and Kafka topics
    batch           Run batch processor
    stream          Run streaming processor
    error           Run error handler
    producer        Run Kafka producer
    all             Start infrastructure and all processors
    stop            Stop all services
    help            Show this help message

Examples:
    # Setup infrastructure
    ./run_pipeline.sh setup
    
    # Run batch processing
    ./run_pipeline.sh batch --start-offset earliest
    
    # Run streaming with console output
    ./run_pipeline.sh stream --console
    
    # Publish test data
    ./run_pipeline.sh producer --test
    
    # Publish from CSV
    ./run_pipeline.sh producer --csv data.csv --batch-size 100
    
    # Stop everything
    ./run_pipeline.sh stop

Monitoring URLs:
    Kafka UI:       http://localhost:8080
    Mongo Express:  http://localhost:8081
    Spark UI:       http://localhost:4040 (when jobs are running)

EOF
}

# Main script
main() {
    case "$1" in
        setup)
            check_docker
            start_infrastructure
            setup_topics
            log_info "Setup complete!"
            log_info "Kafka UI: http://localhost:8080"
            log_info "Mongo Express: http://localhost:8081"
            ;;
        
        batch)
            shift
            run_batch "$@"
            ;;
        
        stream)
            shift
            run_stream "$@"
            ;;
        
        error)
            shift
            run_error_handler "$@"
            ;;
        
        producer)
            shift
            run_producer "$@"
            ;;
        
        all)
            check_docker
            start_infrastructure
            setup_topics
            
            log_info "Starting all processors in background..."
            python streaming_processor.py > logs/streaming.log 2>&1 &
            python error_handler.py > logs/error_handler.log 2>&1 &
            
            log_info "All services started!"
            log_info "Check logs in ./logs/ directory"
            log_info "Press Ctrl+C to stop"
            ;;
        
        stop)
            stop_all
            ;;
        
        help|--help|-h|"")
            show_help
            ;;
        
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Run main
main "$@"
