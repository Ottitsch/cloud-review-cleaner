"""
AWS Lambda function for sentiment analysis of customer reviews.

This function is triggered by the profanity check Lambda and performs:
- Sentiment analysis of review text and summary
- Rating-sentiment consistency checking
- Batch processing of up to 1000 reviews
- Final review processing and storage
- Triggering user management for banned users
"""

import json
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List
from datetime import datetime

from shared.aws_utils import (
    get_parameter_store_value,
    update_dynamodb_item,
    invoke_lambda,
    aws_clients
,
    reinitialize_clients
)
from shared.sentiment_utils import (
    analyze_sentiment,
    check_rating_consistency
)
from shared.constants import (
    STATUS_COMPLETED,
    STATUS_PROCESSING,
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
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
    AWS Lambda handler for sentiment analysis with batch support.
    
    Args:
        event: Event containing review data from profanity check, batch data, or direct call
        context: Lambda context object
        
    Returns:
        Dictionary with sentiment analysis results
    """
    try:
        logger.info(f"Sentiment analysis Lambda triggered with event keys: {list(event.keys())}")
        
        # Check if this is batch processing
        if event.get('batch_processing'):
            return handle_batch_processing(event)
        
        # Handle legacy single review processing
        review_id = event['review_id']
        logger.info(f"Processing sentiment analysis for review: {review_id}")
        
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
            
            # Get profanity result from DB
            profanity_result = {}
            if 'profanity_check' in review_data:
                profanity_check = review_data['profanity_check']
                if isinstance(profanity_check, dict) and 'M' in profanity_check:
                    profanity_check = profanity_check['M']
                    profanity_result = {
                        'has_profanity': profanity_check.get('has_profanity', {}).get('BOOL', False),
                        'profanity_score': float(profanity_check.get('profanity_score', {}).get('N', '0')),
                        'flagged_words': [item.get('S', '') for item in profanity_check.get('flagged_words', {}).get('L', [])]
                    }
            
        else:
            # Pipeline call - extract from event
            timestamp = event['timestamp']
            tokens = event['tokens']
            original_summary = event['original_summary']
            original_review_text = event['original_review_text']
            profanity_result = event.get('profanity_result', {})
        
        logger.info(f"Processing sentiment analysis for review: {review_id}")
        
        # Prepare features for sentiment analysis
        features = {
            'combined_tokens': tokens,
            'summary_text': original_summary,
            'review_text': original_review_text,
            'total_tokens': len(tokens)
        }
        
        # Perform sentiment analysis
        sentiment_result = analyze_sentiment(features)
        
        # Get overall rating from review record
        overall_rating = get_overall_rating_from_review(review_id)
        
        # Check rating-sentiment consistency
        consistency_result = None
        if overall_rating is not None:
            consistency_result = check_rating_consistency(sentiment_result, overall_rating)
        
        # Update review record with sentiment analysis results
        update_review_sentiment_status(
            review_id, 
            timestamp, 
            sentiment_result, 
            consistency_result,
            profanity_result
        )
        
        # Trigger user management if profanity was detected
        if profanity_result.get('has_profanity', False):
            trigger_user_management(event, sentiment_result)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Sentiment analysis completed',
                'review_id': review_id,
                'sentiment_label': sentiment_result['sentiment_label'],
                'sentiment_score': sentiment_result['sentiment_score'],
                'confidence': sentiment_result['confidence'],
                'is_consistent': consistency_result['is_consistent'] if consistency_result else None
            })
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Sentiment analysis failed',
                'message': str(e)
            })
        }


def handle_batch_processing(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle batch processing of up to 1000 reviews for sentiment analysis.
    
    Args:
        event: Batch processing event with reviews array
        
    Returns:
        Dictionary with batch sentiment analysis status and results
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
        
        logger.info(f"Batch sentiment analysis processing {len(reviews)} reviews with batch_id: {batch_id}")
        
        # Handle empty reviews case
        if len(reviews) == 0:
            logger.info(f"No reviews to process for batch {batch_id}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Batch sentiment analysis completed - no reviews to process',
                    'batch_id': batch_id,
                    'processed_count': 0,
                    'failed_count': 0,
                    'total_count': 0,
                    'sentiment_stats': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'profanity_reviews_count': 0,
                    'review_ids': [],
                    'results': []
                })
            }
        
        # Process reviews in parallel chunks for better performance
        chunk_size = min(100, len(reviews))  # Process in chunks of 100
        review_chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
        
        all_results = []
        total_processed = 0
        total_failed = 0
        sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
        profanity_reviews = []
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            chunk_futures = []
            for chunk_idx, chunk in enumerate(review_chunks):
                future = executor.submit(process_sentiment_chunk, chunk, f"{batch_id}-chunk-{chunk_idx}")
                chunk_futures.append(future)
            
            # Collect results from all chunks
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    chunk_result = future.result()
                    all_results.extend(chunk_result['results'])
                    total_processed += chunk_result['processed_count']
                    total_failed += chunk_result['failed_count']
                    
                    # Aggregate sentiment statistics
                    for sentiment, count in chunk_result['sentiment_stats'].items():
                        sentiment_stats[sentiment] += count
                    
                    # Collect reviews with profanity for user management
                    profanity_reviews.extend(chunk_result['profanity_reviews'])
                    
                except Exception as e:
                    logger.error(f"Chunk processing failed: {str(e)}")
                    total_failed += chunk_size  # Assume all in chunk failed
        
        # Trigger user management for reviews with profanity
        if profanity_reviews:
            trigger_batch_user_management(profanity_reviews, batch_id)
        
        logger.info(f"Batch {batch_id} sentiment analysis complete: {total_processed} processed, {total_failed} failed")
        logger.info(f"Sentiment distribution: {sentiment_stats}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Batch sentiment analysis completed',
                'batch_id': batch_id,
                'processed_count': total_processed,
                'failed_count': total_failed,
                'total_count': len(reviews),
                'sentiment_stats': sentiment_stats,
                'profanity_reviews_count': len(profanity_reviews),
                'review_ids': [r['review_id'] for r in all_results if r.get('status') == 'success'],
                'results': all_results
            })
        }
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Batch sentiment analysis failed',
                'message': str(e),
                'batch_id': batch_id
            })
        }


def process_sentiment_chunk(reviews_chunk: List[Dict[str, Any]], chunk_id: str) -> Dict[str, Any]:
    """
    Process a chunk of reviews for sentiment analysis.
    
    Args:
        reviews_chunk: List of review data to process
        chunk_id: Identifier for this chunk
        
    Returns:
        Dictionary with chunk processing results
    """
    logger.info(f"Processing sentiment chunk {chunk_id} with {len(reviews_chunk)} reviews")
    
    processed_count = 0
    failed_count = 0
    results = []
    batch_updates = []
    sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
    profanity_reviews = []
    
    # Process each review in the chunk
    for idx, review_data in enumerate(reviews_chunk):
        try:
            review_id = review_data['review_id']
            timestamp = review_data['timestamp']
            tokens = review_data.get('tokens', [])
            original_summary = review_data.get('original_summary', '')
            original_review_text = review_data.get('original_review_text', '')
            user_id = review_data.get('user_id', 'unknown')
            profanity_result = review_data.get('profanity_result', {})
            
            # Prepare features for sentiment analysis
            features = {
                'combined_tokens': tokens,
                'summary_text': original_summary,
                'review_text': original_review_text,
                'total_tokens': len(tokens)
            }
            
            # Perform sentiment analysis
            sentiment_result = analyze_sentiment(features)
            
            # Get overall rating for consistency check
            overall_rating = get_overall_rating_from_review(review_id)
            
            # Check rating-sentiment consistency
            consistency_result = None
            if overall_rating is not None:
                consistency_result = check_rating_consistency(sentiment_result, overall_rating)
            
            # Update sentiment statistics
            sentiment_label = sentiment_result['sentiment_label']
            if sentiment_label in sentiment_stats:
                sentiment_stats[sentiment_label] += 1
            
            # Prepare batch update for DynamoDB
            update_data = create_sentiment_update_data(sentiment_result, consistency_result, profanity_result)
            
            batch_updates.append({
                'review_id': review_id,
                'timestamp': timestamp,
                'update_data': update_data
            })
            
            # Track profanity reviews for user management
            if profanity_result.get('has_profanity', False):
                profanity_reviews.append({
                    'user_id': user_id,
                    'review_id': review_id,
                    'trigger_reason': 'unpolite_review',
                    'sentiment_result': sentiment_result
                })
            
            processed_count += 1
            results.append({
                'status': 'success',
                'review_id': review_id,
                'index': idx,
                'chunk_id': chunk_id,
                'sentiment_label': sentiment_result['sentiment_label'],
                'sentiment_score': sentiment_result['sentiment_score'],
                'has_profanity': profanity_result.get('has_profanity', False)
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
            batch_update_success = batch_update_sentiment_status(batch_updates)
            if not batch_update_success:
                logger.error(f"Failed to batch update chunk {chunk_id} sentiment status")
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
    
    logger.info(f"Chunk {chunk_id} sentiment analysis complete: {processed_count} processed, {failed_count} failed")
    
    return {
        'processed_count': processed_count,
        'failed_count': failed_count,
        'sentiment_stats': sentiment_stats,
        'profanity_reviews': profanity_reviews,
        'results': results
    }


def create_sentiment_update_data(sentiment_result: Dict[str, Any], consistency_result: Optional[Dict[str, Any]], profanity_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create update data for DynamoDB with sentiment analysis results.
    
    Args:
        sentiment_result: Sentiment analysis results
        consistency_result: Rating-sentiment consistency results
        profanity_result: Profanity check results
        
    Returns:
        Dictionary with DynamoDB update data
    """
    update_data = {
        'sentiment_analysis': {
            'M': {
                'sentiment_label': {'S': sentiment_result['sentiment_label']},
                'sentiment_score': {'N': str(sentiment_result['sentiment_score'])},
                'confidence': {'N': str(sentiment_result['confidence'])},
                'analyzed_at': {'S': datetime.utcnow().isoformat()}
            }
        },
        'processing_summary': {
            'M': {
                'is_unpolite': {'BOOL': profanity_result.get('has_profanity', False)},
                'sentiment_label': {'S': sentiment_result['sentiment_label']},
                'completed_at': {'S': datetime.utcnow().isoformat()}
            }
        },
        'status': {'S': STATUS_COMPLETED}
    }
    
    # Add consistency check results if available
    if consistency_result:
        update_data['sentiment_analysis']['M']['is_consistent'] = {'BOOL': consistency_result['is_consistent']}
        update_data['sentiment_analysis']['M']['consistency_score'] = {'N': str(consistency_result.get('consistency_score', 0.0))}
    
    return update_data


def batch_update_sentiment_status(batch_updates: List[Dict[str, Any]]) -> bool:
    """
    Batch update sentiment status for multiple reviews.
    
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
                        logger.error(f"Failed to update sentiment status for review {update['review_id']}")
                except Exception as e:
                    logger.error(f"Exception updating sentiment status for review {update['review_id']}: {str(e)}")
                    all_success = False
        
        return all_success
        
    except Exception as e:
        logger.error(f"Failed to batch update sentiment status: {str(e)}")
        return False


def trigger_batch_user_management(profanity_reviews: List[Dict[str, Any]], batch_id: str) -> bool:
    """
    Trigger user management for reviews with profanity.
    
    Args:
        profanity_reviews: List of reviews with profanity
        batch_id: Batch identifier
        
    Returns:
        True if trigger was successful, False otherwise
    """
    try:
        if not profanity_reviews:
            return True
        
        # Prepare batch payload for user management
        user_mgmt_payload = {
            'batch_processing': True,
            'batch_id': batch_id,
            'profanity_reviews': profanity_reviews
        }
        
        # Check payload size and split if necessary
        payload_json = json.dumps(user_mgmt_payload)
        payload_size = len(payload_json.encode('utf-8'))
        
        if payload_size > MAX_LAMBDA_PAYLOAD_SIZE_BYTES:
            logger.warning(f"Payload size {payload_size} exceeds limit, splitting batch")
            return trigger_split_batch_user_management(profanity_reviews, batch_id)
        
        # Get Lambda function name from parameter store
        lambda_function_name = get_parameter_store_value(get_ssm_parameter_path('lambdas/user-management'))
        
        # Invoke user management Lambda
        success = invoke_lambda(lambda_function_name, user_mgmt_payload)
        
        if success:
            logger.info(f"Successfully triggered user management for batch {batch_id} ({len(profanity_reviews)} reviews)")
        else:
            logger.error(f"Failed to trigger user management for batch {batch_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to trigger batch user management: {str(e)}")
        return False


def trigger_split_batch_user_management(profanity_reviews: List[Dict[str, Any]], batch_id: str) -> bool:
    """
    Split large batches and trigger multiple user management calls.
    
    Args:
        profanity_reviews: List of reviews with profanity
        batch_id: Batch identifier
        
    Returns:
        True if all triggers were successful, False otherwise
    """
    try:
        # Split into smaller batches
        max_reviews_per_batch = min(100, len(profanity_reviews))
        
        all_success = True
        for i in range(0, len(profanity_reviews), max_reviews_per_batch):
            sub_batch = profanity_reviews[i:i + max_reviews_per_batch]
            sub_batch_id = f"{batch_id}-split-{i // max_reviews_per_batch}"
            
            user_mgmt_payload = {
                'batch_processing': True,
                'batch_id': sub_batch_id,
                'profanity_reviews': sub_batch
            }
            
            lambda_function_name = get_parameter_store_value(get_ssm_parameter_path('lambdas/user-management'))
            success = invoke_lambda(lambda_function_name, user_mgmt_payload)
            
            if not success:
                logger.error(f"Failed to trigger user management for sub-batch {sub_batch_id}")
                all_success = False
            else:
                logger.info(f"Triggered user management for sub-batch {sub_batch_id} ({len(sub_batch)} reviews)")
        
        return all_success
        
    except Exception as e:
        logger.error(f"Failed to trigger split batch user management: {str(e)}")
        return False


def get_overall_rating_from_review(review_id: str) -> Optional[float]:
    """
    Get overall rating from review record.
    
    Args:
        review_id: Review identifier
        
    Returns:
        Overall rating if found, None otherwise
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        
        # Query review to get overall rating
        response = aws_clients.dynamodb.query(
            TableName=table_name,
            KeyConditionExpression='review_id = :review_id',
            ExpressionAttributeValues={':review_id': {'S': review_id}},
            ProjectionExpression='original_data.overall'
        )
        
        if response['Items']:
            rating = float(response['Items'][0]['original_data']['M']['overall']['N'])
            return rating
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get overall rating from review: {str(e)}")
        return None


def update_review_sentiment_status(
    review_id: str, 
    timestamp: str, 
    sentiment_result: Dict[str, Any],
    consistency_result: Optional[Dict[str, Any]],
    profanity_result: Dict[str, Any]
) -> bool:
    """
    Update review record with sentiment analysis results.
    
    Args:
        review_id: Unique review identifier
        timestamp: Review timestamp
        sentiment_result: Sentiment analysis results
        consistency_result: Rating-sentiment consistency results
        profanity_result: Profanity check results
        
    Returns:
        True if update successful, False otherwise
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        
        # Prepare sentiment analysis attributes
        sentiment_attributes = {
            'sentiment_label': {'S': sentiment_result['sentiment_label']},
            'sentiment_score': {'N': str(sentiment_result['sentiment_score'])},
            'confidence': {'N': str(sentiment_result['confidence'])},
            'analyzed_at': {'S': datetime.utcnow().isoformat()}
        }
        
        # Add detailed sentiment scores if available
        if 'detailed_scores' in sentiment_result:
            sentiment_attributes['detailed_scores'] = {
                'M': {
                    'positive': {'N': str(sentiment_result['detailed_scores']['positive'])},
                    'negative': {'N': str(sentiment_result['detailed_scores']['negative'])},
                    'neutral': {'N': str(sentiment_result['detailed_scores']['neutral'])}
                }
            }
        
        # Prepare update attributes
        update_attributes = {
            'sentiment_analysis': {'M': sentiment_attributes},
            'status': {'S': STATUS_COMPLETED},
            'completed_at': {'S': datetime.utcnow().isoformat()}
        }
        
        # Add consistency analysis if available
        if consistency_result:
            update_attributes['rating_consistency'] = {
                'M': {
                    'is_consistent': {'BOOL': consistency_result['is_consistent']},
                    'discrepancy_score': {'N': str(consistency_result['discrepancy_score'])},
                    'explanation': {'S': consistency_result.get('explanation', '')},
                    'checked_at': {'S': datetime.utcnow().isoformat()}
                }
            }
        
        # Add processing summary
        processing_summary = create_processing_summary(sentiment_result, profanity_result, consistency_result)
        update_attributes['processing_summary'] = {'M': processing_summary}
        
        # Update the item
        success = update_dynamodb_item(
            table_name=table_name,
            key={'review_id': {'S': review_id}, 'timestamp': {'S': timestamp}},
            update_attributes=update_attributes
        )
        
        if success:
            logger.info(f"Updated sentiment analysis for review {review_id}")
        else:
            logger.error(f"Failed to update sentiment analysis for review {review_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to update review sentiment status: {str(e)}")
        return False


def create_processing_summary(
    sentiment_result: Dict[str, Any],
    profanity_result: Dict[str, Any],
    consistency_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create a processing summary for the review.
    
    Args:
        sentiment_result: Sentiment analysis results
        profanity_result: Profanity check results
        consistency_result: Rating-sentiment consistency results
        
    Returns:
        Processing summary in DynamoDB format
    """
    summary = {
        'sentiment_label': {'S': sentiment_result['sentiment_label']},
        'has_profanity': {'BOOL': profanity_result.get('has_profanity', False)},
        'profanity_count': {'N': str(profanity_result.get('profanity_count', 0))},
        'is_unpolite': {'BOOL': profanity_result.get('has_profanity', False)},
        'confidence_score': {'N': str(sentiment_result['confidence'])},
        'processed_at': {'S': datetime.utcnow().isoformat()}
    }
    
    # Add consistency information if available
    if consistency_result:
        summary['is_rating_consistent'] = {'BOOL': consistency_result['is_consistent']}
        summary['discrepancy_score'] = {'N': str(consistency_result['discrepancy_score'])}
    
    # Add severity classification
    if profanity_result.get('has_profanity'):
        summary['profanity_severity'] = {'S': profanity_result.get('severity', 'unknown')}
    
    # Add sentiment classification
    sentiment_score = sentiment_result['sentiment_score']
    if sentiment_score >= SENTIMENT_POSITIVE_THRESHOLD:
        sentiment_class = 'positive'
    elif sentiment_score <= SENTIMENT_NEGATIVE_THRESHOLD:
        sentiment_class = 'negative'
    else:
        sentiment_class = 'neutral'
    
    summary['sentiment_classification'] = {'S': sentiment_class}
    
    return summary


def trigger_user_management(event: Dict[str, Any], sentiment_result: Dict[str, Any]) -> bool:
    """
    Trigger user management Lambda function for unpolite reviews.
    
    Args:
        event: Original event data
        sentiment_result: Results from sentiment analysis
        
    Returns:
        True if invocation successful, False otherwise
    """
    try:
        # Get user management function name
        user_mgmt_function = get_parameter_store_value(get_ssm_parameter_path('lambdas/user-management'))
        
        # Get user ID from the event or review record
        user_id = get_user_id_from_event(event)
        if not user_id:
            logger.warning(f"No user ID found for review {event['review_id']}")
            return False
        
        # Create payload for user management
        payload = {
            'review_id': event['review_id'],
            'user_id': user_id,
            'timestamp': event['timestamp'],
            'profanity_result': event.get('profanity_result', {}),
            'sentiment_result': sentiment_result,
            'trigger_reason': 'unpolite_review'
        }
        
        # Invoke user management function asynchronously
        success = invoke_lambda(user_mgmt_function, payload, 'Event')
        
        if success:
            logger.info(f"Triggered user management for user {user_id} due to unpolite review {event['review_id']}")
        else:
            logger.error(f"Failed to trigger user management")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to trigger user management: {str(e)}")
        return False


def get_user_id_from_event(event: Dict[str, Any]) -> Optional[str]:
    """
    Get user ID from event or review record.
    
    Args:
        event: Event data
        
    Returns:
        User ID if found, None otherwise
    """
    try:
        # First check if user_id is directly in the event
        if 'user_id' in event:
            return event['user_id']
        
        # Otherwise, query the review record
        review_id = event['review_id']
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        
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
        logger.error(f"Failed to get user ID from event: {str(e)}")
        return None


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