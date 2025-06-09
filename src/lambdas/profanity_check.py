"""
AWS Lambda function for profanity checking in customer reviews.

This function is triggered by the preprocessing Lambda and performs:
- Profanity detection in review text and summary
- Scoring and classification of inappropriate content
- Batch processing of up to 1000 reviews
- Forwarding batches to sentiment analysis function
- User statistics tracking for unpolite reviews
"""

import json
import logging
import re
import concurrent.futures
from typing import Dict, Any, List, Tuple
from datetime import datetime

from shared.aws_utils import (
    get_parameter_store_value,
    download_from_s3,
    update_dynamodb_item,
    invoke_lambda,
    aws_clients,
    get_dynamodb_item,
    batch_write_to_dynamodb,
    reinitialize_clients
)
from shared.constants import (
    STATUS_PROFANITY_DETECTED,
    STATUS_PROCESSING,
    PROFANITY_THRESHOLD,
    BAD_WORDS_FILE,
    ERROR_PROFANITY_DETECTED,
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
    AWS Lambda handler for profanity checking with batch support.
    
    Args:
        event: Event containing review data from preprocessing, batch data, or direct call
        context: Lambda context object
        
    Returns:
        Dictionary with profanity check results
    """
    try:
        # Reinitialize AWS clients to pick up environment variables
        reinitialize_clients()
        
        logger.info(f"Profanity check Lambda triggered with event keys: {list(event.keys())}")
        
        # Check if this is batch processing
        if event.get('batch_processing'):
            return handle_batch_processing(event)
        
        # Handle legacy single review processing
        review_id = event['review_id']
        logger.info(f"Processing profanity check for review: {review_id}")
        
        # Check if this is a direct call (stage-based) or pipeline call
        if 'stage' in event:
            # Direct call - retrieve data from DynamoDB
            review_data = get_review_from_db(review_id)
            if not review_data:
                raise Exception(f"Review {review_id} not found in database")
            
            # Extract required fields from DB record
            timestamp = review_data.get('timestamp', datetime.utcnow().isoformat())
            
            # Extract tokens from processed_features
            processed_features = review_data.get('processed_features', {})
            if isinstance(processed_features, dict) and 'M' in processed_features:
                processed_features = processed_features['M']
            
            tokens = []
            if 'combined_tokens' in processed_features:
                tokens_data = processed_features['combined_tokens']
                if 'L' in tokens_data:
                    tokens = [item.get('S', '') for item in tokens_data['L']]
            
            # Extract original data
            original_data = review_data.get('original_data', {})
            if isinstance(original_data, dict) and 'M' in original_data:
                original_data = original_data['M']
            
            original_summary = ''
            original_review_text = ''
            
            if 'summary' in original_data:
                original_summary = original_data['summary'].get('S', '')
            if 'reviewText' in original_data:
                original_review_text = original_data['reviewText'].get('S', '')
            
        else:
            # Pipeline call - extract from event
            timestamp = event['timestamp']
            tokens = event['tokens']
            original_summary = event['original_summary']
            original_review_text = event['original_review_text']
        
        # Load bad words list
        bad_words = load_bad_words_list()
        
        # Check for profanity in tokens and original text
        profanity_result = check_profanity(tokens, original_summary, original_review_text, bad_words)
        
        # Update review record with profanity results
        update_review_profanity_status(review_id, timestamp, profanity_result)
        
        # Update user statistics if profanity detected
        if profanity_result['has_profanity']:
            user_id = get_user_id_from_review(review_id)
            if user_id:
                update_user_profanity_count(user_id, review_id)
        
        # Trigger sentiment analysis if this is a direct call
        if 'stage' in event:
            trigger_sentiment_analysis_direct(review_id)
        else:
            # Pipeline call
            trigger_sentiment_analysis(event, profanity_result)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Profanity check completed',
                'review_id': review_id,
                'has_profanity': profanity_result['has_profanity'],
                'profanity_score': profanity_result['profanity_score'],
                'flagged_words': profanity_result['flagged_words']
            })
        }
        
    except Exception as e:
        logger.error(f"Profanity check failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Profanity check failed',
                'message': str(e)
            })
        }


def handle_batch_processing(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle batch processing of up to 1000 reviews for profanity checking.
    
    Args:
        event: Batch processing event with reviews array
        
    Returns:
        Dictionary with batch profanity check status and results
    """
    try:
        reviews = event.get('reviews', [])
        batch_id = event.get('batch_id', 'unknown')
        
        if len(reviews) > MAX_BATCH_SIZE:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': f'Batch size {len(reviews)} exceeds maximum of {MAX_BATCH_SIZE}',
                    'batch_id': batch_id
                })
            }
        
        logger.info(f"Batch profanity check processing {len(reviews)} reviews with batch_id: {batch_id}")
        
        # Load bad words list once for all reviews
        bad_words = load_bad_words_list()
        
        # Process reviews in parallel chunks for better performance
        chunk_size = min(100, len(reviews))  # Process in chunks of 100
        review_chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
        
        all_results = []
        total_processed = 0
        total_failed = 0
        total_profanity_detected = 0
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            chunk_futures = []
            for chunk_idx, chunk in enumerate(review_chunks):
                future = executor.submit(process_profanity_chunk, chunk, f"{batch_id}-chunk-{chunk_idx}", bad_words)
                chunk_futures.append(future)
            
            # Collect results from all chunks
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    chunk_result = future.result()
                    all_results.extend(chunk_result['results'])
                    total_processed += chunk_result['processed_count']
                    total_failed += chunk_result['failed_count']
                    total_profanity_detected += chunk_result['profanity_count']
                except Exception as e:
                    logger.error(f"Chunk processing failed: {str(e)}")
                    total_failed += chunk_size  # Assume all in chunk failed
        
        # Extract successful review data for downstream processing
        successful_reviews = [r for r in all_results if r.get('status') == 'success']
        
        if successful_reviews:
            # Trigger sentiment analysis for successful reviews in batches
            trigger_batch_sentiment_analysis(successful_reviews, batch_id)
        
        logger.info(f"Batch {batch_id} profanity check complete: {total_processed} processed, {total_failed} failed, {total_profanity_detected} with profanity")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Batch profanity check completed',
                'batch_id': batch_id,
                'processed_count': total_processed,
                'failed_count': total_failed,
                'total_count': len(reviews),
                'profanity_detected_count': total_profanity_detected,
                'review_ids': [r['review_id'] for r in successful_reviews],
                'results': all_results
            })
        }
        
    except Exception as e:
        logger.error(f"Batch profanity check failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Batch profanity check failed',
                'message': str(e),
                'batch_id': batch_id
            })
        }


def process_profanity_chunk(reviews_chunk: List[Dict[str, Any]], chunk_id: str, bad_words: List[str]) -> Dict[str, Any]:
    """
    Process a chunk of reviews for profanity checking.
    
    Args:
        reviews_chunk: List of review data to process
        chunk_id: Identifier for this chunk
        bad_words: List of bad words for profanity checking
        
    Returns:
        Dictionary with chunk processing results
    """
    logger.info(f"Processing profanity chunk {chunk_id} with {len(reviews_chunk)} reviews")
    
    processed_count = 0
    failed_count = 0
    profanity_count = 0
    results = []
    batch_updates = []
    
    # Process each review in the chunk
    for idx, review_data in enumerate(reviews_chunk):
        try:
            review_id = review_data['review_id']
            timestamp = review_data['timestamp']
            tokens = review_data.get('tokens', [])
            original_summary = review_data.get('original_summary', '')
            original_review_text = review_data.get('original_review_text', '')
            user_id = review_data.get('user_id', 'unknown')
            
            # Check for profanity in tokens and original text
            profanity_result = check_profanity(tokens, original_summary, original_review_text, bad_words)
            
            # Prepare batch update for DynamoDB
            update_data = {
                'profanity_check': {
                    'M': {
                        'has_profanity': {'BOOL': profanity_result['has_profanity']},
                        'profanity_score': {'N': str(profanity_result['profanity_score'])},
                        'flagged_words': {'L': [{'S': word} for word in profanity_result['flagged_words']]},
                        'severity': {'S': profanity_result['severity']},
                        'checked_at': {'S': datetime.utcnow().isoformat()}
                    }
                },
                'status': {'S': STATUS_PROFANITY_DETECTED if profanity_result['has_profanity'] else STATUS_PROCESSING}
            }
            
            batch_updates.append({
                'review_id': review_id,
                'timestamp': timestamp,
                'update_data': update_data
            })
            
            # Track profanity detection for user statistics
            if profanity_result['has_profanity']:
                profanity_count += 1
                # Note: User statistics will be updated in a separate process to avoid blocking
            
            processed_count += 1
            results.append({
                'status': 'success',
                'review_id': review_id,
                'index': idx,
                'chunk_id': chunk_id,
                'has_profanity': profanity_result['has_profanity'],
                'profanity_score': profanity_result['profanity_score'],
                'review_data': {
                    'review_id': review_id,
                    'timestamp': timestamp,
                    'tokens': tokens,
                    'original_summary': original_summary,
                    'original_review_text': original_review_text,
                    'user_id': user_id,
                    'profanity_result': profanity_result
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to process review {idx} in chunk {chunk_id}: {str(e)}")
            failed_count += 1
            results.append({
                'status': 'failed',
                'error': str(e),
                'index': idx,
                'chunk_id': chunk_id,
                'review_id': review_data.get('review_id', 'unknown')
            })
    
    # Batch update DynamoDB
    if batch_updates:
        try:
            batch_update_success = batch_update_profanity_status(batch_updates)
            if not batch_update_success:
                logger.error(f"Failed to batch update chunk {chunk_id} profanity status")
                # Mark all as failed
                for result in results:
                    if result.get('status') == 'success':
                        result['status'] = 'failed'
                        result['error'] = 'DynamoDB batch update failed'
                        failed_count += 1
                        processed_count -= 1
        except Exception as e:
            logger.error(f"Exception during batch update for chunk {chunk_id}: {str(e)}")
            # Mark all as failed
            for result in results:
                if result.get('status') == 'success':
                    result['status'] = 'failed'
                    result['error'] = f'DynamoDB batch update exception: {str(e)}'
                    failed_count += 1
                    processed_count -= 1
    
    logger.info(f"Chunk {chunk_id} profanity check complete: {processed_count} processed, {failed_count} failed, {profanity_count} with profanity")
    
    return {
        'processed_count': processed_count,
        'failed_count': failed_count,
        'profanity_count': profanity_count,
        'results': results
    }


def batch_update_profanity_status(batch_updates: List[Dict[str, Any]]) -> bool:
    """
    Batch update profanity status for multiple reviews.
    
    Args:
        batch_updates: List of review updates with review_id and update_data
        
    Returns:
        True if all updates were successful, False otherwise
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        all_success = True
        
        # Process updates in smaller batches to avoid timeouts
        batch_size = 25  # DynamoDB batch limit
        for i in range(0, len(batch_updates), batch_size):
            batch_chunk = batch_updates[i:i + batch_size]
            
            for update in batch_chunk:
                try:
                    success = update_dynamodb_item(
                        table_name=table_name,
                        key={
                            'review_id': {'S': update['review_id']},
                            'timestamp': {'S': update['timestamp']}
                        },
                        update_attributes=update['update_data']
                    )
                    if not success:
                        all_success = False
                        logger.error(f"Failed to update profanity status for review {update['review_id']}")
                except Exception as e:
                    logger.error(f"Exception updating profanity status for review {update['review_id']}: {str(e)}")
                    all_success = False
        
        return all_success
        
    except Exception as e:
        logger.error(f"Failed to batch update profanity status: {str(e)}")
        return False


def trigger_batch_sentiment_analysis(successful_reviews: List[Dict[str, Any]], batch_id: str) -> bool:
    """
    Trigger sentiment analysis for a batch of successfully processed reviews.
    
    Args:
        successful_reviews: List of successfully processed review data
        batch_id: Batch identifier
        
    Returns:
        True if trigger was successful, False otherwise
    """
    try:
        # Prepare batch payload for sentiment analysis
        sentiment_payload = {
            'batch_processing': True,
            'batch_id': batch_id,
            'reviews': [r['review_data'] for r in successful_reviews]
        }
        
        # Check payload size and split if necessary
        payload_json = json.dumps(sentiment_payload)
        payload_size = len(payload_json.encode('utf-8'))
        
        if payload_size > MAX_LAMBDA_PAYLOAD_SIZE_BYTES:
            logger.warning(f"Payload size {payload_size} exceeds limit, splitting batch")
            return trigger_split_batch_sentiment_analysis(successful_reviews, batch_id)
        
        # Get Lambda function name from parameter store
        lambda_function_name = get_parameter_store_value(get_ssm_parameter_path('lambdas/sentiment-analysis'))
        
        # Invoke sentiment analysis Lambda
        success = invoke_lambda(lambda_function_name, sentiment_payload)
        
        if success:
            logger.info(f"Successfully triggered sentiment analysis for batch {batch_id} ({len(successful_reviews)} reviews)")
        else:
            logger.error(f"Failed to trigger sentiment analysis for batch {batch_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to trigger batch sentiment analysis: {str(e)}")
        return False


def trigger_split_batch_sentiment_analysis(successful_reviews: List[Dict[str, Any]], batch_id: str) -> bool:
    """
    Split large batches and trigger multiple sentiment analysis calls.
    
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
            
            sentiment_payload = {
                'batch_processing': True,
                'batch_id': sub_batch_id,
                'reviews': [r['review_data'] for r in sub_batch]
            }
            
            lambda_function_name = get_parameter_store_value(get_ssm_parameter_path('lambdas/sentiment-analysis'))
            success = invoke_lambda(lambda_function_name, sentiment_payload)
            
            if not success:
                logger.error(f"Failed to trigger sentiment analysis for sub-batch {sub_batch_id}")
                all_success = False
            else:
                logger.info(f"Triggered sentiment analysis for sub-batch {sub_batch_id} ({len(sub_batch)} reviews)")
        
        return all_success
        
    except Exception as e:
        logger.error(f"Failed to trigger split batch sentiment analysis: {str(e)}")
        return False


def load_bad_words_list() -> List[str]:
    """
    Load bad words list from S3.
    
    Returns:
        List of bad words for profanity checking
        
    Raises:
        Exception: If bad words list cannot be loaded
    """
    try:
        # Get processed reviews bucket name
        bucket_name = get_parameter_store_value(get_ssm_parameter_path('buckets/processed-reviews'))
        
        # Download bad words file
        bad_words_content = download_from_s3(bucket_name, f"resources/{BAD_WORDS_FILE}")
        
        if not bad_words_content:
            raise Exception("Failed to download bad words list")
        
        # Parse bad words (one per line, ignore comments)
        bad_words = []
        for line in bad_words_content.strip().split('\n'):
            line = line.strip().lower()
            if line and not line.startswith('#'):
                bad_words.append(line)
        
        logger.info(f"Loaded {len(bad_words)} bad words for profanity checking")
        return bad_words
        
    except Exception as e:
        logger.error(f"Failed to load bad words list: {str(e)}")
        # Return a minimal default list as fallback
        return ['fuck', 'shit', 'damn', 'bitch', 'asshole']


def check_profanity(tokens: List[str], summary: str, review_text: str, bad_words: List[str]) -> Dict[str, Any]:
    """
    Check for profanity in review content.

    Args:
        tokens: Preprocessed tokens from review
        summary: Original review summary
        review_text: Original review text
        bad_words: List of bad words to check against

    Returns:
        Dictionary with profanity check results
    """
    try:
        # Handle None or empty tokens
        if tokens is None:
            tokens = []
        
        flagged_words = []
        profanity_count = 0
        total_words = len(tokens)
        
        # Check tokens for exact matches
        bad_words_lower = [word.lower() for word in bad_words]
        for token in tokens:
            token_lower = token.lower()
            if token_lower in bad_words_lower:
                flagged_words.append(token)
                profanity_count += 1
        
        # Check original text for partial matches and variations
        combined_text = f"{summary} {review_text}".lower()
        
        for bad_word in bad_words:
            # Check for exact word boundaries to avoid false positives
            pattern = r'\b' + re.escape(bad_word) + r'\b'
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            
            for match in matches:
                if match.lower() not in [fw.lower() for fw in flagged_words]:
                    flagged_words.append(match)
                    profanity_count += 1
        
        # Calculate profanity score
        profanity_score = profanity_count / max(total_words, 1) if total_words > 0 else 0
        
        # Determine if profanity threshold is exceeded
        # Flag as profanity if any profane words are found OR score exceeds threshold
        has_profanity = profanity_count > 0 or profanity_score >= PROFANITY_THRESHOLD
        
        result = {
            'has_profanity': has_profanity,
            'profanity_score': profanity_score,
            'profanity_count': profanity_count,
            'flagged_words': list(set(flagged_words)),  # Remove duplicates
            'total_words': total_words,
            'severity': classify_profanity_severity(profanity_score, profanity_count)
        }
        
        logger.info(f"Profanity check results: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to check profanity: {str(e)}")
        return {
            'has_profanity': False,
            'profanity_score': 0.0,
            'profanity_count': 0,
            'flagged_words': [],
            'total_words': len(tokens) if tokens is not None else 0,
            'severity': 'none',
            'error': str(e)
        }


def classify_profanity_severity(profanity_score: float, profanity_count: int) -> str:
    """
    Classify the severity of profanity detected.
    
    Args:
        profanity_score: Ratio of profane words to total words
        profanity_count: Total number of profane words
        
    Returns:
        Severity classification: 'none', 'mild', 'moderate', 'severe'
    """
    if profanity_count == 0:
        return 'none'
    elif profanity_count <= 2 and profanity_score < 0.1:
        return 'mild'
    elif profanity_count <= 5 and profanity_score < 0.3:
        return 'moderate'
    else:
        return 'severe'


def update_review_profanity_status(review_id: str, timestamp: str, profanity_result: Dict[str, Any]) -> bool:
    """
    Update review record with profanity check results.
    
    Args:
        review_id: Unique review identifier
        timestamp: Review timestamp
        profanity_result: Profanity check results
        
    Returns:
        True if update successful, False otherwise
    """
    logger.info(f"[DEBUG] update_review_profanity_status called with review_id={review_id} (type={type(review_id)}), timestamp={timestamp} (type={type(timestamp)})")
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        logger.info(f"[DEBUG] Table name: {table_name}")
        
        # Prepare update attributes
        update_attributes = {
            'profanity_check': {
                'M': {
                    'has_profanity': {'BOOL': profanity_result['has_profanity']},
                    'profanity_score': {'N': str(profanity_result['profanity_score'])},
                    'profanity_count': {'N': str(profanity_result['profanity_count'])},
                    'flagged_words': {'L': [{'S': word} for word in profanity_result['flagged_words']]},
                    'severity': {'S': profanity_result['severity']},
                    'checked_at': {'S': datetime.utcnow().isoformat()}
                }
            },
            'status': {'S': STATUS_PROFANITY_DETECTED if profanity_result['has_profanity'] else STATUS_PROCESSING}
        }
        logger.info(f"[DEBUG] update_attributes: {update_attributes}")
        
        # Update the item
        success = update_dynamodb_item(
            table_name=table_name,
            key={'review_id': {'S': review_id}, 'timestamp': {'S': timestamp}},
            update_attributes=update_attributes
        )
        
        if success:
            logger.info(f"Updated profanity status for review {review_id}")
        else:
            logger.error(f"Failed to update profanity status for review {review_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to update review profanity status: {str(e)}")
        return False


def get_user_id_from_review(review_id: str) -> str:
    """
    Get user ID from review record.
    
    Args:
        review_id: Review identifier
        
    Returns:
        User ID if found, None otherwise
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        
        # Query review to get user ID
        response = aws_clients.dynamodb.query(
            TableName=table_name,
            KeyConditionExpression='review_id = :review_id',
            ExpressionAttributeValues={':review_id': {'S': review_id}},
            ProjectionExpression='original_data.reviewerID'
        )
        
        if response['Items']:
            user_id = response['Items'][0]['original_data']['M']['reviewerID']['S']
            return user_id
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get user ID from review: {str(e)}")
        return None


def update_user_profanity_count(user_id: str, review_id: str) -> bool:
    """
    Update user's unpolite review count.
    
    Args:
        user_id: User identifier
        review_id: Review identifier
        
    Returns:
        True if update successful, False otherwise
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        
        # Update user record with atomic counter increment
        response = aws_clients.dynamodb.update_item(
            TableName=table_name,
            Key={'user_id': {'S': user_id}},
            UpdateExpression='ADD unpolite_review_count :inc SET last_unpolite_review = :review_id, updated_at = :timestamp',
            ExpressionAttributeValues={
                ':inc': {'N': '1'},
                ':review_id': {'S': review_id},
                ':timestamp': {'S': datetime.utcnow().isoformat()}
            },
            ReturnValues='ALL_NEW'
        )
        
        new_count = int(response['Attributes']['unpolite_review_count']['N'])
        logger.info(f"Updated user {user_id} unpolite review count to {new_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update user profanity count: {str(e)}")
        return False


def trigger_sentiment_analysis(event: Dict[str, Any], profanity_result: Dict[str, Any]) -> bool:
    """
    Trigger sentiment analysis Lambda function.
    
    Args:
        event: Original event data
        profanity_result: Results from profanity check
        
    Returns:
        True if invocation successful, False otherwise
    """
    try:
        # Get sentiment analysis function name
        sentiment_function = get_parameter_store_value(get_ssm_parameter_path('lambdas/sentiment-analysis'))
        
        # Create enhanced payload with profanity results
        payload = {
            **event,  # Include original event data
            'profanity_result': profanity_result
        }
        
        # Invoke sentiment analysis function asynchronously
        success = invoke_lambda(sentiment_function, payload, 'Event')
        
        if success:
            logger.info(f"Triggered sentiment analysis for review {event['review_id']}")
        else:
            logger.error(f"Failed to trigger sentiment analysis")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to trigger sentiment analysis: {str(e)}")
        return False


def get_review_from_db(review_id: str) -> Dict[str, Any]:
    """
    Retrieve review data from DynamoDB.
    
    Args:
        review_id: ID of the review to retrieve
        
    Returns:
        Review data dictionary or None if not found
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        
        # Query using review_id (partition key) to get the item
        # Since we don't know the exact timestamp, we query by partition key
        response = aws_clients.dynamodb.query(
            TableName=table_name,
            KeyConditionExpression='review_id = :review_id',
            ExpressionAttributeValues={':review_id': {'S': review_id}},
            Limit=1
        )
        
        if response['Items']:
            # Convert DynamoDB format to regular dict
            item = response['Items'][0]
            converted_item = {}
            
            for key, value in item.items():
                if 'S' in value:
                    converted_item[key] = value['S']
                elif 'N' in value:
                    converted_item[key] = float(value['N'])
                elif 'L' in value:
                    converted_item[key] = [v.get('S', '') for v in value['L']]
                elif 'M' in value:
                    converted_item[key] = value['M']
                else:
                    converted_item[key] = value
            
            return converted_item
        else:
            logger.warning(f"Review {review_id} not found in database")
            return None
            
    except Exception as e:
        logger.error(f"Failed to retrieve review {review_id}: {str(e)}")
        return None


def trigger_sentiment_analysis_direct(review_id: str) -> bool:
    """
    Trigger sentiment analysis Lambda function for direct processing.
    
    Args:
        review_id: ID of the review to process
        
    Returns:
        True if successfully triggered, False otherwise
    """
    try:
        function_name = get_parameter_store_value(get_ssm_parameter_path('lambdas/sentiment-analysis'))
        
        payload = {
            'review_id': review_id,
            'stage': 'sentiment_analysis'
        }
        
        response = invoke_lambda(
            function_name=function_name,
            payload=payload,
            invocation_type='Event'  # Async invocation
        )
        
        logger.info(f"Sentiment analysis triggered for review: {review_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to trigger sentiment analysis for {review_id}: {str(e)}")
        return False 