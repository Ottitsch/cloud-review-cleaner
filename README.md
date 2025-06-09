# Cloud Review Cleaner

An event-driven serverless application for analyzing customer reviews using AWS Lambda functions. This system performs profanity checking, sentiment analysis, and user management for review moderation using AWS services with LocalStack for local development.

## 🏗️ Architecture

The application implements a serverless event-driven architecture with the following components:

- **4 Lambda Functions**: Preprocessing, Profanity Check, Sentiment Analysis, User Management
- **3 S3 Buckets**: Raw reviews, processed reviews, failed reviews
- **3 DynamoDB Tables**: Reviews, users, profanity statistics
- **SSM Parameter Store**: Configuration management
- **Event Triggers**: S3 object creation and DynamoDB streams

### Function Chain Flow

```
S3 Upload → Preprocess → Profanity Check → Sentiment Analysis → User Management
              ↓              ↓                ↓                    ↓
           DynamoDB      Flag Profane    Classify Sentiment   Track/Ban Users
```

## 📊 Features

- **Text Preprocessing**: Tokenization, stop word removal, lemmatization
- **Profanity Detection**: Bad word filtering with configurable thresholds
- **Sentiment Analysis**: Positive/neutral/negative classification with confidence scores
- **User Management**: Automatic user banning after 3+ unpolite reviews
- **Batch Processing**: Process up to 3,000 reviews per batch
- **Integration Testing**: Comprehensive test suite for the complete pipeline

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- LocalStack
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cloud-review-cleaner
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start LocalStack**, make sure docker desktop is running
   ```bash
   # Option 1: (Recommended)
   localstack start
   
   # Option 2: Using the PowerShell script for multiple localstack instances (Windows)
   .\start_localstack_instances.ps1
   ```

4. **Deploy the application**
   ```bash
   # Option 1: (Recommended)
   python deploy.py
   
   # Option 2:
   python multi_deploy.py
   ```

## 📋 Usage

### Processing Reviews

#### Single Dataset Processing (Recommended)
```bash
# Process the entire reviews_devset.json
python batch_processor.py

# Process with custom parameters
python batch_processor.py --file data/reviews_devset.json --batch-size 1000 --max-reviews 10000
```

#### Multi-Instance Processing
```bash
# Deploy multiple instances for parallel processing
python multi_deploy.py --instances 3

# Process with multiple instances
python multi_batch_processor.py --instances 3 --batch-size 3000
```

### Running Tests

```bash
# Option 1: Run all tests from project root
python -m pytest -v

# Option 2: Run from tests directory
cd tests
python -m pytest . -v

# Option 3: Use the custom test runner
python tests/run_tests.py

# Option 4: Run specific test files
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_unit.py -v
```

### Viewing Results

Results are automatically saved to `results.json` and include:
- Total reviews processed
- Sentiment analysis breakdown (positive/neutral/negative)
- Profanity check failures
- Banned users list

## 📁 Project Structure

```
cloud-review-cleaner/
├── src/
│   ├── lambdas/                    # Lambda function handlers
│   │   ├── preprocess.py          # Text preprocessing & validation
│   │   ├── profanity_check.py     # Bad word detection
│   │   ├── sentiment_analysis.py  # Sentiment classification
│   │   └── user_management.py     # User tracking & banning
│   └── shared/                     # Shared utilities
│       ├── aws_utils.py           # AWS service wrappers
│       ├── text_utils_simple.py   # NLP preprocessing
│       ├── sentiment_utils.py     # Sentiment analysis logic
│       └── constants.py           # Application constants
├── config/
│   ├── aws_config.json           # AWS resource definitions
│   └── parameters.json           # SSM parameter mappings
├── tests/
│   ├── test_integration.py       # End-to-end pipeline tests
│   ├── test_unit.py             # Unit tests for individual components
│   └── run_tests.py             # Test runner
├── data/
│   └── reviews_devset.json      # Test dataset (55MB)
├── resources/
│   ├── bad_words.txt            # Profanity word list
│   └── stopwords.txt            # Stop words for preprocessing
├── deploy.py                    # Single-instance deployment
├── multi_deploy.py             # Multi-instance deployment
├── batch_processor.py          # Single-instance batch processor
├── multi_batch_processor.py    # Multi-instance batch processor
└── requirements.txt            # Python dependencies
```

## ⚙️ Configuration

### Environment Variables

Key configuration is managed through SSM Parameter Store:

- **Buckets**: Raw reviews, processed reviews, failed reviews
- **Tables**: Reviews, users, profanity statistics
- **Thresholds**: Profanity detection, sentiment classification
- **Limits**: Max unpolite reviews before ban (default: 3)

### Lambda Function Configuration

| Function | Memory | Timeout | Purpose |
|----------|--------|---------|---------|
| Preprocess | 512MB | 300s | Text preprocessing & validation |
| Profanity Check | 256MB | 60s | Bad word detection |
| Sentiment Analysis | 512MB | 180s | Sentiment classification |
| User Management | 256MB | 60s | User tracking & banning |

## 🧪 Testing

### Integration Tests

The test suite validates the complete function chain:

1. **Preprocessing**: Text cleaning and tokenization
2. **Profanity Check**: Bad word detection accuracy
3. **Sentiment Analysis**: Positive/negative/neutral classification
4. **User Management**: Review counting and user banning logic

### Test Coverage

- End-to-end pipeline processing
- Error handling and edge cases
- Batch processing validation
- Assignment requirement compliance

## 📈 Results

Based on the `reviews_devset.json` dataset (78,829 reviews):

- **Sentiment Analysis**:
  - Positive: 61,927 reviews (78.5%)
  - Neutral: 10,874 reviews (13.8%)
  - Negative: 6,028 reviews (7.6%)

- **Profanity Check**: 9,046 failed reviews (11.5%)

- **User Management**: 40 banned users out of 8,737 total users

## 🛠️ Technology Stack

- **Runtime**: Python 3.9
- **AWS Services**: Lambda, S3, DynamoDB, SSM Parameter Store
- **Development**: LocalStack for local AWS simulation
- **NLP**: NLTK, TextBlob for text processing
- **Testing**: pytest, moto for AWS service mocking
- **Data Processing**: pandas, numpy for data manipulation

## 📚 Dependencies

Key Python packages:
- `boto3` - AWS SDK for Python
- `nltk` - Natural Language Toolkit
- `textblob` - Simple NLP library
- `better-profanity` - Profanity detection
- `pandas` - Data manipulation
- `pytest` - Testing framework
- `moto` - AWS service mocking

## 🔧 Troubleshooting

### Common Issues

1. **LocalStack Connection**: Ensure LocalStack is running on port 4566
2. **Memory Issues**: Adjust Lambda memory settings for large batches
3. **Timeout Errors**: Increase Lambda timeout for batch processing
4. **Deployment Failures**: Check IAM permissions and resource limits
5. **Analyzing Databases Overload**: Increase timeout between requests to DynamoDB

### Debugging

- Check Localstack logs for detailed error messages
- Use `pytest -v` for verbose test output
- Monitor DynamoDB streams for event processing

