"""
Application constants for AWS Review Cleaner.

This module contains all configurable constants used throughout the application.
Values are organized by functional area and follow clean code principles.
"""

# SSM Parameter Store paths
SSM_BUCKET_RAW_REVIEWS = "/cloud-review-cleaner/buckets/raw-reviews"
SSM_BUCKET_PROCESSED_REVIEWS = "/cloud-review-cleaner/buckets/processed-reviews"
SSM_BUCKET_FAILED_REVIEWS = "/cloud-review-cleaner/buckets/failed-reviews"
SSM_TABLE_REVIEWS = "/cloud-review-cleaner/tables/reviews"
SSM_TABLE_USERS = "/cloud-review-cleaner/tables/users"
SSM_TABLE_PROFANITY_STATS = "/cloud-review-cleaner/tables/profanity-stats"

# SSM Parameter Store paths for Lambda functions
SSM_LAMBDA_PREPROCESS = "/cloud-review-cleaner/lambdas/preprocess"
SSM_LAMBDA_PROFANITY_CHECK = "/cloud-review-cleaner/lambdas/profanity-check"
SSM_LAMBDA_SENTIMENT_ANALYSIS = "/cloud-review-cleaner/lambdas/sentiment-analysis"
SSM_LAMBDA_USER_MANAGEMENT = "/cloud-review-cleaner/lambdas/user-management"
SSM_LAMBDA_BATCH_PREPROCESSOR = "/cloud-review-cleaner/lambdas/batch-preprocessor"
SSM_LAMBDA_STREAM_PROCESSOR = "/cloud-review-cleaner/lambdas/stream-processor"

# Lambda function names
LAMBDA_PREPROCESS = "cloud-review-cleaner-preprocess"
LAMBDA_PROFANITY_CHECK = "cloud-review-cleaner-profanity-check"
LAMBDA_SENTIMENT_ANALYSIS = "cloud-review-cleaner-sentiment-analysis"
LAMBDA_USER_MANAGEMENT = "cloud-review-cleaner-user-management"
LAMBDA_BATCH_PREPROCESSOR = "cloud-review-cleaner-batch-preprocessor"
LAMBDA_STREAM_PROCESSOR = "cloud-review-cleaner-stream-processor"

# Text processing constants
MAX_REVIEW_LENGTH = 10000
MIN_REVIEW_LENGTH = 10
DEFAULT_LANGUAGE = "en"
STOP_WORDS_FILE = "stopwords.txt"

# Profanity checking constants
BAD_WORDS_FILE = "bad_words.txt"
PROFANITY_THRESHOLD = 0.05  # Reduced threshold for better detection (5% vs 70%)

# Sentiment analysis constants
SENTIMENT_POSITIVE_THRESHOLD = 0.1
SENTIMENT_NEGATIVE_THRESHOLD = -0.1

# User management constants
MAX_UNPOLITE_REVIEWS = 3
USER_BAN_STATUS = "banned"
USER_ACTIVE_STATUS = "active"

# Batch processing constants
DEFAULT_BATCH_SIZE = 500  # Reviews per S3 file
DYNAMODB_BATCH_SIZE = 25  # DynamoDB batch write limit
LAMBDA_BATCH_SIZE = 100   # Reviews per Lambda execution for optimal performance
MAX_PARALLEL_WORKERS = 15  # Thread pool size for parallel processing (increased for 3000 review batches)
MAX_BATCH_SIZE = 3000     # Maximum reviews per batch (increased from 1000)
MAX_LAMBDA_PAYLOAD_SIZE_MB = 6  # Lambda payload size limit in MB

# Lambda optimization constants
LAMBDA_MEMORY_SIZE_SMALL = 256   # MB - For simple processing
LAMBDA_MEMORY_SIZE_MEDIUM = 512  # MB - For batch processing
LAMBDA_MEMORY_SIZE_LARGE = 1024  # MB - For heavy text processing
LAMBDA_MEMORY_SIZE_XLARGE = 1536 # MB - For large batch processing (3000 reviews)
LAMBDA_TIMEOUT_SHORT = 60        # Seconds - For simple operations
LAMBDA_TIMEOUT_MEDIUM = 180      # Seconds - For batch operations
LAMBDA_TIMEOUT_LONG = 300        # Seconds - For large dataset processing
LAMBDA_TIMEOUT_XLARGE = 600      # Seconds - For 3000 review batch processing (10 minutes)

# LocalStack configuration
LOCALSTACK_ENDPOINT = "http://localhost:4566"
AWS_REGION = "us-east-1"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "json"

# Review processing statuses
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_PROFANITY_DETECTED = "profanity_detected"

# Error codes
ERROR_INVALID_INPUT = "INVALID_INPUT"
ERROR_PROCESSING_FAILED = "PROCESSING_FAILED"
ERROR_PROFANITY_DETECTED = "PROFANITY_DETECTED"
ERROR_USER_BANNED = "USER_BANNED"

# Performance optimization constants
ENABLE_PARALLEL_PROCESSING = True
ENABLE_BATCH_OPERATIONS = True
ENABLE_STREAM_PROCESSING = True
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

# DynamoDB optimization
DYNAMODB_READ_CAPACITY = 5
DYNAMODB_WRITE_CAPACITY = 10
ENABLE_DYNAMODB_STREAMS = True

# S3 optimization  
S3_MULTIPART_THRESHOLD = 64 * 1024 * 1024  # 64MB
S3_MULTIPART_CHUNKSIZE = 16 * 1024 * 1024  # 16MB

# Monitoring and alerting thresholds
HIGH_ERROR_RATE_THRESHOLD = 0.1  # 10% error rate
HIGH_LATENCY_THRESHOLD = 30.0    # 30 seconds
LOW_THROUGHPUT_THRESHOLD = 10    # Reviews per minute 