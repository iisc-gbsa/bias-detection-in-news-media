# Bias Detection in News Media

A scalable system for detecting multi-dimensional bias in Indian news articles using ensemble machine learning methods with Apache Spark, Kafka, and MongoDB.

## Overview

This project analyzes news articles from Indian media sources (Indian Express, Economic Times) to detect bias across multiple dimensions:

- **Political** (left/right/centrist)
- **Gender** (male/female stereotypes)
- **Religious** (multi-faith bias)
- **Caste** (upper/lower caste focus)
- **Regional** (geographic bias)
- **Socioeconomic** (wealth/poverty focus)

### Ensemble Detection Method

Combines three complementary approaches for robust bias detection:

1. **Keyword Matching (30%)** - Fast, interpretable baseline
2. **TF-IDF Analysis (30%)** - Term importance weighting
3. **BERT Embeddings (40%)** - Semantic context understanding using `all-MiniLM-L6-v2`

## Architecture

### System Design

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        IE[Indian Express]
        ET[Economic Times]
        OTHER[Other News Sites]
    end
    
    subgraph Ingestion["Ingestion Layer"]
        SCRAPER[Web Scrapers]
        KAFKA[Kafka<br/>news topic]
    end
    
    subgraph Processing["Processing Layer"]
        BATCH[Batch Processor<br/>Daily Analysis]
        STREAM[Streaming Processor<br/>Real-time Analysis]
        SPARK[Apache Spark]
    end
    
    subgraph ML["ML Pipeline"]
        CLASSIFY[Topic Classification]
        BIAS[Bias Detection<br/>Ensemble Model]
    end
    
    subgraph Storage["Storage Layer"]
        MONGO[(MongoDB)]
        DAILY[daily_news]
        REALTIME[realtime_news]
        ERRORS[error_logs]
    end
    
    Sources --> SCRAPER
    SCRAPER --> KAFKA
    KAFKA --> BATCH
    KAFKA --> STREAM
    BATCH --> SPARK
    STREAM --> SPARK
    SPARK --> CLASSIFY
    CLASSIFY --> BIAS
    BIAS --> MONGO
    MONGO --> DAILY
    MONGO --> REALTIME
    MONGO --> ERRORS
```

### Data Flow Sequence

```mermaid
sequenceDiagram
    participant Scraper
    participant Kafka
    participant Spark
    participant BiasDetector
    participant MongoDB
    
    Scraper->>Kafka: Publish article
    Note over Kafka: news topic
    
    alt Batch Mode
        Spark->>Kafka: Read batch (1000 msgs)
        Spark->>BiasDetector: Analyze articles
    else Streaming Mode
        Kafka->>Spark: Stream micro-batch
        Spark->>BiasDetector: Analyze articles
    end
    
    BiasDetector->>BiasDetector: 1. Keyword Score
    BiasDetector->>BiasDetector: 2. TF-IDF Score
    BiasDetector->>BiasDetector: 3. BERT Embeddings
    BiasDetector->>BiasDetector: Compute Ensemble Score
    
    BiasDetector->>MongoDB: Write results
    Note over MongoDB: daily_news or<br/>realtime_news
```

## Project Structure

```
bias-detection-in-news-media/
├── config/                    # Configuration
│   └── config.py             # Kafka, MongoDB, Spark, Bias configs
├── src/
│   ├── scrapers/             # Web scraping modules
│   │   ├── scrape_indian_express.py
│   │   ├── scrape_article_content.py
│   │   └── scraping_news.py
│   ├── processing/           # Data processing
│   │   ├── batch/
│   │   │   └── batch_processor.py
│   │   └── streaming/
│   │       ├── streaming_processor.py
│   │       └── kafka_producer.py
│   ├── models/               # ML models
│   │   ├── bias_detection.py
│   │   ├── catogorise_the_article.py
│   │   └── spark_bias_detector.py
│   └── utils/
│       └── error_handler.py
├── scripts/                  # Utility scripts
│   ├── run_pipeline.sh
│   └── setup_kafka_topics.py
├── notebooks/                # Jupyter notebooks for experiments
├── data/
│   ├── raw/                  # Raw CSV data
│   └── processed/            # Processed data
└── docker-compose.yml        # Infrastructure setup
```

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- 8GB RAM (recommended)

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make scripts executable (if needed)
chmod +x scripts/run_pipeline.sh
```

### 2. Start Infrastructure

```bash
# Start Kafka, Zookeeper, MongoDB
docker-compose up -d

# Wait 30 seconds for services to initialize
sleep 30

# Create Kafka topics
python scripts/setup_kafka_topics.py

# OR use the convenience script
./scripts/run_pipeline.sh setup
```

### 3. Verify Setup

- **Kafka UI**: http://localhost:8080
- **Mongo Express**: http://localhost:8081 (admin/admin)

### 4. Run the System

#### Option A: Batch Processing (Daily Analysis)

```bash
# 1. Publish articles to Kafka
PYTHONPATH=. python src/processing/streaming/kafka_producer.py \
    --csv data/raw/economic_times_articles_2024.csv \
    --batch-size 100

# OR use the script (recommended)
./scripts/run_pipeline.sh producer --csv data/raw/economic_times_articles_2024.csv --batch-size 100

# 2. Process batch
PYTHONPATH=. python src/processing/batch/batch_processor.py

# OR use the script (recommended)
./scripts/run_pipeline.sh batch

# 3. Check results
mongosh mongodb://localhost:27018/bias_detection
> db.daily_news.find().limit(5)
```

#### Option B: Streaming Processing (Real-time)

```bash
# Terminal 1: Start streaming processor
PYTHONPATH=. python src/processing/streaming/streaming_processor.py --console
# OR use the script (recommended)
./scripts/run_pipeline.sh stream --console

# Terminal 2: Simulate real-time news
PYTHONPATH=. python src/processing/streaming/kafka_producer.py \
    --csv data/raw/economic_times_articles_2024.csv \
    --realtime \
    --delay 0.5
# OR use the script (recommended)
./scripts/run_pipeline.sh producer --csv data/raw/economic_times_articles_2024.csv --realtime --delay 0.5

# Terminal 3: Monitor (optional)
PYTHONPATH=. python src/utils/error_handler.py --retry
# OR use the script (recommended)
./scripts/run_pipeline.sh error --retry
```

### 5. Query Results

```bash
mongosh mongodb://localhost:27018/bias_detection

# Find high-bias articles
db.realtime_news.find({ overall_bias_score: { $gt: 0.4 } }).limit(10)

# Analyze political distribution
db.realtime_news.aggregate([
    { $group: { _id: "$political_type", count: { $sum: 1 } } }
])
```

## Example Output

**Bias Detection Result:**
```json
{
  "_id": "article_123",
  "title": "Government announces new policy",
  "url": "https://...",
  "overall_bias_score": 0.342,
  "political_bias_score": 0.456,
  "political_type": "right",
  "gender_bias_score": 0.123,
  "religious_bias_score": 0.089,
  "caste_bias_score": 0.234,
  "regional_bias_score": 0.178,
  "socioeconomic_bias_score": 0.267,
  "timestamp": "2024-11-23T10:30:00Z"
}
```

## Configuration

Edit `config/config.py` to customize:

```python
# Kafka settings
kafka_config.bootstrap_servers = "localhost:9092"
kafka_config.news_topic = "news"

# MongoDB settings
mongo_config.uri = "mongodb://localhost:27018/"
mongo_config.database = "bias_detection"

# Bias detection weights
bias_config.keyword_weight = 0.3
bias_config.tfidf_weight = 0.3
bias_config.embedding_weight = 0.4
```

### Adding New Bias Dimensions

1. Update bias keywords in `src/models/bias_detection.py`
2. Add dimension weights in `config/config.py`
3. Update output schema in processors

## Troubleshooting

**Services not starting?**
```bash
# Clean restart
docker-compose down -v
docker-compose up -d

# Check service status
docker-compose ps

# View service logs
docker-compose logs kafka
docker-compose logs mongodb
```

**Port conflicts?**  
Edit `docker-compose.yml` to change ports:
- Kafka: 9092
- MongoDB: 27018
- Kafka UI: 8080
- Mongo Express: 8081

**Import errors?**  
Ensure you're running from project root:
```bash
# Check current directory
pwd

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or run as module
python -m src.processing.batch.batch_processor
```

**Kafka connection errors?**  
Wait 30 seconds after starting Docker for Kafka initialization:
```bash
docker-compose up -d
sleep 30
python scripts/setup_kafka_topics.py
```

**Module not found errors?**
```bash
# Verify structure
ls -la src/
ls -la config/

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

## Monitoring

- **Spark UI**: http://localhost:4040 (when jobs running)
- **Kafka UI**: http://localhost:8080
- **Mongo Express**: http://localhost:8081 (admin/admin)

View logs:
```bash
# Streaming processor logs
tail -f logs/streaming.log

# Error handler logs
tail -f logs/error_handler.log

# Docker service logs
docker-compose logs -f kafka
docker-compose logs -f mongodb
```

## Stopping the System

```bash
# Stop all services
docker-compose down
# OR use the script
./scripts/run_pipeline.sh stop

# Stop and remove volumes (clears all data)
docker-compose down -v
```

## Tech Stack

- **Processing**: Apache Spark 3.5
- **Messaging**: Apache Kafka
- **Database**: MongoDB
- **ML Models**: 
  - Sentence-Transformers (BERT)
  - Scikit-learn (TF-IDF)
  - Custom ensemble methods
- **Languages**: Python 3.8+

## Future Enhancements

- Add more Indian news sources
- Deploy to AWS (EMR + MSK + DynamoDB)
- Implement web dashboard for visualization
- Add real-time alerting for high-bias content
- Expand to regional language support
