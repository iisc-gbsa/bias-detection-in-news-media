# Tests

Test suite for the bias detection system.

## Test Files

### `test_model.py`
Tests for bias detection models including:
- BERT-based emotion detection
- Sentiment analysis
- Zero-shot classification
- Toxicity detection
- Ensemble bias detection

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/
```

## Test Guidelines

- Write unit tests for all new features
- Maintain test coverage above 80%
- Add integration tests for end-to-end workflows
- Mock external dependencies (APIs, databases)
