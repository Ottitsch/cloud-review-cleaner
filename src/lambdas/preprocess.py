"""
AWS Lambda function for preprocessing customer reviews.

This function is triggered by S3 object creation events and performs:
- Text preprocessing (tokenization, stop word removal, lemmatization)
- Data validation and cleaning
- Batch processing of up to 1000 reviews
- Forwarding batches to profanity check function
"""

import json
import logging
import re
import uuid
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, List
from urllib.parse import unquote_plus

from shared.aws_utils import (
    get_parameter_store_value,
    download_from_s3,
    upload_to_s3,
    invoke_lambda,
    put_dynamodb_item,
    batch_write_to_dynamodb,
    reinitialize_clients
)
from shared.text_utils_simple import preprocess_review
from shared.constants import (
    STATUS_PROCESSING,
    STATUS_FAILED,
    ERROR_INVALID_INPUT,
    ERROR_PROCESSING_FAILED,
    MAX_BATCH_SIZE,
    MAX_LAMBDA_PAYLOAD_SIZE_MB,
    MAX_PARALLEL_WORKERS
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Batch processing constants
MAX_LAMBDA_PAYLOAD_SIZE_BYTES = MAX_LAMBDA_PAYLOAD_SIZE_MB * 1024 * 1024


def get_ssm_parameter_path(param_type: str) -> str:
    """
    Get instance-specific SSM parameter path.
    
    Args:
        param_type: Type of parameter (e.g., 'buckets/processed-reviews', 'tables/reviews')
        
    Returns:
        Full SSM parameter path
    """
    import os
    instance_name = os.environ.get('INSTANCE_NAME')
    if instance_name:
        return f'/cloud-review-cleaner/{instance_name}/{param_type}'
    else:
        return f'/cloud-review-cleaner/{param_type}'


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for review preprocessing with batch support.
    
    Args:
        event: S3 event, direct processing event, or batch processing event
        context: Lambda context object
        
    Returns:
        Dictionary with processing status and results
    """
    try:
        # Reinitialize AWS clients to pick up environment variables
        reinitialize_clients()
        
        logger.info(f"Preprocessing Lambda triggered with event keys: {list(event.keys())}")
        
        # Check if this is batch processing
        if event.get('batch_processing'):
            return handle_batch_processing(event)
        
        # Check if this is direct processing (legacy support)
        if event.get('direct_processing'):
            return handle_direct_processing(event)
        
        # Original S3 event processing
        results = []
        for record in event.get('Records', []):
            try:
                result = process_s3_record(record)
                results.append(result)
                logger.info(f"Successfully processed record: {result['review_id']}")
                
            except Exception as e:
                logger.error(f"Failed to process S3 record: {str(e)}")
                results.append({
                    'status': 'failed',
                    'error': str(e),
                    'record': record
                })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Preprocessing completed',
                'processed_count': len([r for r in results if r.get('status') == 'success']),
                'failed_count': len([r for r in results if r.get('status') == 'failed']),
                'results': results
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }


def handle_batch_processing(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle batch processing of up to 1000 reviews.
    
    Args:
        event: Batch processing event with reviews array
        
    Returns:
        Dictionary with batch processing status and results
    """
    try:
        reviews = event.get('reviews', [])
        batch_id = event.get('batch_id', str(uuid.uuid4()))
        
        if len(reviews) > MAX_BATCH_SIZE:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': f'Batch size {len(reviews)} exceeds maximum of {MAX_BATCH_SIZE}',
                    'batch_id': batch_id
                })
            }
        
        logger.info(f"Batch processing {len(reviews)} reviews with batch_id: {batch_id}")
        
        # Process reviews in parallel chunks for better performance
        chunk_size = min(100, len(reviews))  # Process in chunks of 100
        review_chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
        
        all_results = []
        total_processed = 0
        total_failed = 0
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            chunk_futures = []
            for chunk_idx, chunk in enumerate(review_chunks):
                future = executor.submit(process_review_chunk, chunk, f"{batch_id}-chunk-{chunk_idx}")
                chunk_futures.append(future)
            
            # Collect results from all chunks
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    chunk_result = future.result()
                    all_results.extend(chunk_result['results'])
                    total_processed += chunk_result['processed_count']
                    total_failed += chunk_result['failed_count']
                except Exception as e:
                    logger.error(f"Chunk processing failed: {str(e)}")
                    total_failed += chunk_size  # Assume all in chunk failed
        
        # Extract successful review data for downstream processing
        successful_reviews = [r for r in all_results if r.get('status') == 'success']
        
        if successful_reviews:
            # Trigger profanity check for successful reviews in batches
            trigger_batch_profanity_check(successful_reviews, batch_id)
        
        logger.info(f"Batch {batch_id} complete: {total_processed} processed, {total_failed} failed")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Batch preprocessing completed',
                'batch_id': batch_id,
                'processed_count': total_processed,
                'failed_count': total_failed,
                'total_count': len(reviews),
                'review_ids': [r['review_id'] for r in successful_reviews],
                'results': all_results
            })
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Batch processing failed',
                'message': str(e),
                'batch_id': batch_id
            })
        }


def process_review_chunk(reviews_chunk: List[Dict[str, Any]], chunk_id: str) -> Dict[str, Any]:
    """
    Process a chunk of reviews in parallel.
    
    Args:
        reviews_chunk: List of reviews to process
        chunk_id: Identifier for this chunk
        
    Returns:
        Dictionary with chunk processing results
    """
    logger.info(f"Processing chunk {chunk_id} with {len(reviews_chunk)} reviews")
    
    processed_count = 0
    failed_count = 0
    results = []
    batch_records = []
    
    # Process each review in the chunk
    for idx, review_data in enumerate(reviews_chunk):
        try:
            # Validate review data
            validate_single_review(review_data, f"chunk {chunk_id}, review {idx}")
            
            # Generate unique IDs
            review_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # Preprocess the review
            processed_features = preprocess_review(
                review_data['summary'], 
                review_data['reviewText']
            )
            
            # Create review record
            review_record = create_review_record(
                review_id, timestamp, review_data, processed_features, f"batch-{chunk_id}"
            )
            
            # Add to batch for DynamoDB
            batch_records.append(review_record)
            
            processed_count += 1
            results.append({
                'status': 'success',
                'review_id': review_id,
                'index': idx,
                'chunk_id': chunk_id,
                'review_data': {
                    'review_id': review_id,
                    'timestamp': timestamp,
                    'tokens': processed_features.get('combined_tokens', []),
                    'original_summary': review_data.get('summary', ''),
                    'original_review_text': review_data.get('reviewText', ''),
                    'user_id': review_data.get('reviewerID', 'unknown')
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to process review {idx} in chunk {chunk_id}: {str(e)}")
            failed_count += 1
            results.append({
                'status': 'failed',
                'error': str(e),
                'index': idx,
                'chunk_id': chunk_id
            })
    
    # Batch save to DynamoDB
    if batch_records:
        try:
            batch_write_success = batch_write_to_dynamodb(batch_records)
            if not batch_write_success:
                logger.error(f"Failed to batch write chunk {chunk_id} to DynamoDB")
                # Mark all as failed
                for result in results:
                    if result.get('status') == 'success':
                        result['status'] = 'failed'
                        result['error'] = 'DynamoDB batch write failed'
                        failed_count += 1
                        processed_count -= 1
        except Exception as e:
            logger.error(f"Exception during batch write for chunk {chunk_id}: {str(e)}")
            # Mark all as failed
            for result in results:
                if result.get('status') == 'success':
                    result['status'] = 'failed'
                    result['error'] = f'DynamoDB batch write exception: {str(e)}'
                    failed_count += 1
                    processed_count -= 1
    
    logger.info(f"Chunk {chunk_id} complete: {processed_count} processed, {failed_count} failed")
    
    return {
        'processed_count': processed_count,
        'failed_count': failed_count,
        'results': results
    }


def trigger_batch_profanity_check(successful_reviews: List[Dict[str, Any]], batch_id: str) -> bool:
    """
    Trigger profanity check for a batch of successfully processed reviews.
    
    Args:
        successful_reviews: List of successfully processed review data
        batch_id: Batch identifier
        
    Returns:
        True if trigger was successful, False otherwise
    """
    try:
        # Prepare batch payload for profanity check
        profanity_payload = {
            'batch_processing': True,
            'batch_id': batch_id,
            'reviews': [r['review_data'] for r in successful_reviews]
        }
        
        # Check payload size and split if necessary
        payload_json = json.dumps(profanity_payload)
        payload_size = len(payload_json.encode('utf-8'))
        
        if payload_size > MAX_LAMBDA_PAYLOAD_SIZE_BYTES:
            logger.warning(f"Payload size {payload_size} exceeds limit, splitting batch")
            return trigger_split_batch_profanity_check(successful_reviews, batch_id)
        
        # Get Lambda function name from parameter store
        lambda_function_name = get_parameter_store_value(get_ssm_parameter_path('lambdas/profanity-check'))
        
        # Invoke profanity check Lambda
        success = invoke_lambda(lambda_function_name, profanity_payload)
        
        if success:
            logger.info(f"Successfully triggered profanity check for batch {batch_id} ({len(successful_reviews)} reviews)")
        else:
            logger.error(f"Failed to trigger profanity check for batch {batch_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to trigger batch profanity check: {str(e)}")
        return False


def trigger_split_batch_profanity_check(successful_reviews: List[Dict[str, Any]], batch_id: str) -> bool:
    """
    Split large batches and trigger multiple profanity check calls.
    
    Args:
        successful_reviews: List of successfully processed review data
        batch_id: Batch identifier
        
    Returns:
        True if all triggers were successful, False otherwise
    """
    try:
        # Split into smaller batches (estimate ~5KB per review for safety)
        max_reviews_per_batch = min(500, len(successful_reviews))
        
        all_success = True
        for i in range(0, len(successful_reviews), max_reviews_per_batch):
            sub_batch = successful_reviews[i:i + max_reviews_per_batch]
            sub_batch_id = f"{batch_id}-split-{i // max_reviews_per_batch}"
            
            profanity_payload = {
                'batch_processing': True,
                'batch_id': sub_batch_id,
                'reviews': [r['review_data'] for r in sub_batch]
            }
            
            lambda_function_name = get_parameter_store_value(get_ssm_parameter_path('lambdas/profanity-check'))
            success = invoke_lambda(lambda_function_name, profanity_payload)
            
            if not success:
                logger.error(f"Failed to trigger profanity check for sub-batch {sub_batch_id}")
                all_success = False
            else:
                logger.info(f"Triggered profanity check for sub-batch {sub_batch_id} ({len(sub_batch)} reviews)")
        
        return all_success
        
    except Exception as e:
        logger.error(f"Failed to trigger split batch profanity check: {str(e)}")
        return False


def handle_direct_processing(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle direct processing of reviews (legacy support for backward compatibility).
    
    Args:
        event: Direct processing event with reviews array
        
    Returns:
        Dictionary with processing status and results
    """
    # Convert to batch processing format
    batch_event = {
        'batch_processing': True,
        'reviews': event.get('reviews', []),
        'batch_id': event.get('batch_id', 'legacy-direct')
    }
    
    return handle_batch_processing(batch_event)


def process_s3_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single S3 event record.
    If the S3 object contains a list, process each review in the list.
    If it contains a dict, process as a single review.
    """
    try:
        # Extract S3 information
        s3_info = record['s3']
        bucket_name = s3_info['bucket']['name']
        object_key = unquote_plus(s3_info['object']['key'])
        logger.info(f"Processing S3 object: s3://{bucket_name}/{object_key}")
        # Download and parse the review data (could be a list or dict)
        review_data = download_and_parse_review(bucket_name, object_key)
        results = []
        if isinstance(review_data, list):
            logger.info(f"Batch detected: {len(review_data)} reviews in {object_key}")
            for idx, single_review in enumerate(review_data):
                try:
                    review_id = str(uuid.uuid4())
                    timestamp = datetime.utcnow().isoformat()
                    processed_features = preprocess_review(single_review['summary'], single_review['reviewText'])
                    review_record = create_review_record(
                        review_id, timestamp, single_review, processed_features, object_key
                    )
                    save_processed_review(review_record)
                    trigger_profanity_check(review_record)
                    results.append({'status': 'success', 'review_id': review_id, 'batch_index': idx})
                except Exception as e:
                    logger.error(f"Failed to process review in batch: {str(e)}")
                    results.append({'status': 'failed', 'error': str(e), 'batch_index': idx})
            return {'status': 'batch', 'processed_count': len([r for r in results if r['status']=='success']), 'failed_count': len([r for r in results if r['status']=='failed']), 'results': results}
        elif isinstance(review_data, dict):
            # Single review (legacy)
            review_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            processed_features = preprocess_review(review_data['summary'], review_data['reviewText'])
            review_record = create_review_record(
                review_id, timestamp, review_data, processed_features, object_key
            )
            save_processed_review(review_record)
            trigger_profanity_check(review_record)
            return {'status': 'success', 'review_id': review_id, 'bucket': bucket_name, 'key': object_key, 'processed_tokens': processed_features['total_tokens']}
        else:
            raise ValueError("S3 object is neither a dict nor a list of reviews.")
    except Exception as e:
        logger.error(f"Failed to process S3 record: {str(e)}")
        try:
            save_failed_review(record, str(e))
        except Exception as save_error:
            logger.error(f"Failed to save error record: {str(save_error)}")
        raise


def download_and_parse_review(bucket_name: str, object_key: str) -> Dict[str, Any]:
    """
    Download and parse review data from S3.
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        
    Returns:
        Parsed review data (single review dict or list of review dicts)
        
    Raises:
        ValueError: If data format is invalid
        Exception: If download fails
    """
    try:
        # Download the file content
        content = download_from_s3(bucket_name, object_key)
        if not content:
            raise ValueError(f"Failed to download content from s3://{bucket_name}/{object_key}")
        
        # Parse JSON content
        try:
            review_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        
        # Handle both single reviews and batches
        if isinstance(review_data, list):
            # Validate each review in the batch
            logger.info(f"Validating batch of {len(review_data)} reviews")
            for idx, review in enumerate(review_data):
                validate_single_review(review, f"review at index {idx}")
            logger.info(f"Successfully parsed batch of {len(review_data)} reviews")
            return review_data
        elif isinstance(review_data, dict):
            # Validate single review
            validate_single_review(review_data, "single review")
            logger.info(f"Successfully parsed single review with {len(review_data)} fields")
            return review_data
        else:
            raise ValueError("JSON data must be either a review object or an array of review objects")
        
    except Exception as e:
        logger.error(f"Failed to download and parse review: {str(e)}")
        raise


def validate_single_review(review_data: Dict[str, Any], context: str = "review") -> None:
    """
    Validate a single review object.
    
    Args:
        review_data: Review data to validate
        context: Context for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(review_data, dict):
        raise ValueError(f"{context}: must be a dictionary")
    
    # Validate required fields
    required_fields = ['summary', 'reviewText', 'overall']
    missing_fields = [field for field in required_fields if field not in review_data]
    
    if missing_fields:
        raise ValueError(f"{context}: Missing required fields: {missing_fields}")
    
    # Validate data types
    if not isinstance(review_data['summary'], str):
        raise ValueError(f"{context}: Summary must be a string")
    if not isinstance(review_data['reviewText'], str):
        raise ValueError(f"{context}: ReviewText must be a string")
    if not isinstance(review_data['overall'], (int, float)):
        raise ValueError(f"{context}: Overall rating must be a number")
    
    # Validate rating range
    if not (1 <= review_data['overall'] <= 5):
        raise ValueError(f"{context}: Overall rating must be between 1 and 5")


def _safe_string_truncate(text: Any, max_length: int) -> str:
    """
    Safely truncate string to prevent DynamoDB size issues.
    
    Args:
        text: Input text (can be None)
        max_length: Maximum allowed length
        
    Returns:
        Truncated and cleaned string
    """
    if text is None:
        return ''
    
    # Convert to string and clean
    text_str = str(text).strip()
    
    # Remove problematic characters that might cause encoding issues
    text_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text_str)  # Remove control characters
    text_str = re.sub(r'\s+', ' ', text_str)  # Normalize whitespace
    
    # Truncate if too long
    if len(text_str) > max_length:
        logger.warning(f"Truncating text from {len(text_str)} to {max_length} chars")
        text_str = text_str[:max_length].rstrip()
    
    return text_str


def create_review_record(review_id: str, timestamp: str, 
                        original_data: Dict[str, Any], 
                        processed_features: Dict[str, Any],
                        source_key: str) -> Dict[str, Any]:
    """
    Create a comprehensive review record for storage.
    
    Args:
        review_id: Unique review identifier
        timestamp: Processing timestamp
        original_data: Original review data
        processed_features: Preprocessed features
        source_key: Source S3 object key
        
    Returns:
        Complete review record
    """
    return {
        'review_id': {'S': review_id},
        'timestamp': {'S': timestamp},
        'status': {'S': STATUS_PROCESSING},
        'source_key': {'S': source_key},
        'original_data': {
            'M': {
                'summary': {'S': _safe_string_truncate(original_data['summary'], 10000)},
                'reviewText': {'S': _safe_string_truncate(original_data['reviewText'], 50000)},
                'overall': {'N': str(original_data['overall'])},
                'reviewerID': {'S': _safe_string_truncate(original_data.get('reviewerID', 'unknown'), 100)},
                'asin': {'S': _safe_string_truncate(original_data.get('asin', 'unknown'), 100)},
                'reviewerName': {'S': _safe_string_truncate(original_data.get('reviewerName', 'anonymous'), 200)},
                'helpful': {'L': [{'N': str(h)} for h in original_data.get('helpful', [0, 0])]},
                'reviewTime': {'S': _safe_string_truncate(original_data.get('reviewTime', 'unknown'), 50)},
                'unixReviewTime': {'N': str(original_data.get('unixReviewTime', 0))}
            }
        },
        'processed_features': {
            'M': {
                'total_tokens': {'N': str(processed_features.get('total_tokens', 0))},
                'summary_tokens': {'N': str(processed_features.get('summary_tokens', 0))},
                'review_text_tokens': {'N': str(processed_features.get('review_text_tokens', 0))},
                'processed_summary': {'L': [{'S': token} for token in processed_features.get('processed_summary', [])]},
                'processed_review_text': {'L': [{'S': token} for token in processed_features.get('processed_review_text', [])]},
                'combined_tokens': {'L': [{'S': token} for token in processed_features.get('combined_tokens', [])]}
            }
        },
        'processing_metadata': {
            'M': {
                'preprocessed_at': {'S': timestamp},
                'lambda_request_id': {'S': 'context.aws_request_id'},
                'processing_version': {'S': '1.0'}
            }
        }
    }


def save_processed_review(review_record: Dict[str, Any]) -> bool:
    """
    Save processed review to DynamoDB and S3.
    DynamoDB is the primary storage, S3 is for backup/analysis.
    
    Args:
        review_record: Complete review record
        
    Returns:
        True if DynamoDB insertion successful, False otherwise
    """
    review_id = review_record['review_id']['S']
    dynamodb_success = False
    s3_success = False
    
    try:
        # Primary storage: Insert into DynamoDB FIRST (this is what queries use)
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        dynamodb_success = put_dynamodb_item(table_name, review_record)
        
        if dynamodb_success:
            logger.info(f"✅ Saved review {review_id} to DynamoDB")
        else:
            logger.error(f"❌ Failed to save review {review_id} to DynamoDB")
        
    except Exception as e:
        logger.error(f"❌ DynamoDB insertion error for review {review_id}: {str(e)}")
        dynamodb_success = False
    
    try:
        # Secondary storage: Upload to S3 (for backup and analysis)
        processed_bucket = get_parameter_store_value(get_ssm_parameter_path('buckets/processed-reviews'))
        s3_key = f"processed/{review_id}.json"
        s3_data = convert_dynamodb_to_json(review_record)
        s3_success = upload_to_s3(processed_bucket, s3_key, s3_data)
        
        if s3_success:
            logger.info(f"✅ Backed up review {review_id} to S3")
        else:
            logger.warning(f"⚠️ Failed to backup review {review_id} to S3 (DynamoDB still succeeded)")
        
    except Exception as e:
        logger.warning(f"⚠️ S3 backup error for review {review_id}: {str(e)} (DynamoDB still succeeded)")
        s3_success = False
    
    # Log final status
    if dynamodb_success and s3_success:
        logger.info(f"🎉 Review {review_id} saved to both DynamoDB and S3")
    elif dynamodb_success:
        logger.info(f"✅ Review {review_id} saved to DynamoDB (S3 backup failed)")
    else:
        logger.error(f"❌ Review {review_id} failed to save to DynamoDB")
    
    # Return success based on DynamoDB (primary storage) only
    return dynamodb_success


def save_failed_review(record: Dict[str, Any], error_message: str) -> bool:
    """
    Save failed review processing information.
    
    Args:
        record: Original S3 record
        error_message: Error description
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get failed bucket name from parameter store
        failed_bucket = get_parameter_store_value(get_ssm_parameter_path('buckets/failed-reviews'))
        
        # Create error record
        error_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': ERROR_PROCESSING_FAILED,
            'error_message': error_message,
            'original_record': record,
            'processing_stage': 'preprocessing'
        }
        
        # Create S3 key
        error_id = str(uuid.uuid4())
        s3_key = f"failed/{error_id}.json"
        
        # Upload to S3
        success = upload_to_s3(failed_bucket, s3_key, error_record)
        
        if success:
            logger.info(f"Saved failed review to s3://{failed_bucket}/{s3_key}")
        else:
            logger.error(f"Failed to save error record to S3")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to save error record: {str(e)}")
        return False


def trigger_profanity_check(review_record: Dict[str, Any]) -> bool:
    """
    Trigger the profanity check Lambda function.
    
    Args:
        review_record: Processed review record
        
    Returns:
        True if invocation successful, False otherwise
    """
    try:
        # Get profanity check function name from parameter store
        profanity_function = get_parameter_store_value(get_ssm_parameter_path('lambdas/profanity-check'))
        
        # Create payload for profanity check
        payload = {
            'review_id': review_record['review_id']['S'],
            'timestamp': review_record['timestamp']['S'],
            'tokens': [item['S'] for item in review_record['processed_features']['M']['combined_tokens']['L']],
            'original_summary': review_record['original_data']['M']['summary']['S'],
            'original_review_text': review_record['original_data']['M']['reviewText']['S']
        }
        
        # Invoke profanity check function asynchronously
        success = invoke_lambda(profanity_function, payload, 'Event')
        
        if success:
            logger.info(f"Triggered profanity check for review {review_record['review_id']['S']}")
        else:
            logger.error(f"Failed to trigger profanity check")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to trigger profanity check: {str(e)}")
        return False


def convert_dynamodb_to_json(dynamodb_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert DynamoDB item format to regular JSON.
    
    Args:
        dynamodb_item: Item in DynamoDB format
        
    Returns:
        Regular JSON object
    """
    def convert_value(value):
        if isinstance(value, dict):
            if 'S' in value:
                return value['S']
            elif 'N' in value:
                return float(value['N']) if '.' in value['N'] else int(value['N'])
            elif 'M' in value:
                return {k: convert_value(v) for k, v in value['M'].items()}
            elif 'L' in value:
                return [convert_value(item) for item in value['L']]
            else:
                return value
        else:
            return value
    
    return {k: convert_value(v) for k, v in dynamodb_item.items()} 