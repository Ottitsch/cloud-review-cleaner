"""
AWS Lambda function for user management and banning logic.

This function is triggered by sentiment analysis Lambda (for unpolite reviews)
or by DynamoDB streams and performs:
- User statistics tracking
- Ban enforcement when threshold is exceeded
- Batch processing of profanity reviews
- User status management
"""

import json
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

from shared.aws_utils import (
    get_parameter_store_value,
    update_dynamodb_item,
    put_dynamodb_item,
    aws_clients
,
    reinitialize_clients
)
from shared.constants import (
    MAX_UNPOLITE_REVIEWS,
    USER_BAN_STATUS,
    USER_ACTIVE_STATUS,
    MAX_BATCH_SIZE,
    MAX_PARALLEL_WORKERS
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)




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
    AWS Lambda handler for user management with batch support.
    
    Args:
        event: Event containing user and review information or batch profanity reviews
        context: Lambda context object
        
    Returns:
        Dictionary with user management results
    """
    try:
        logger.info(f"User management Lambda triggered with event keys: {list(event.keys())}")
        
        # Check if this is batch processing
        if event.get('batch_processing'):
            return handle_batch_processing(event)
        
        # Handle different event sources
        if 'Records' in event:
            # DynamoDB stream event
            return handle_dynamodb_stream_event(event)
        elif 'stage' in event:
            # Stage-based direct call (for testing/manual pipeline)
            return handle_stage_based_call(event)
        else:
            # Direct invocation from sentiment analysis
            return handle_direct_invocation(event)
        
    except Exception as e:
        logger.error(f"User management failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'User management failed',
                'message': str(e)
            })
        }


def handle_batch_processing(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle batch processing of profanity reviews for user management.
    
    Args:
        event: Batch processing event with profanity_reviews array
        
    Returns:
        Dictionary with batch user management status and results
    """
    try:
        profanity_reviews = event.get('profanity_reviews', [])
        batch_id = event.get('batch_id', 'unknown')
        
        if len(profanity_reviews) > MAX_BATCH_SIZE:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': f'Batch size {len(profanity_reviews)} exceeds maximum of {MAX_BATCH_SIZE}',
                    'batch_id': batch_id
                })
            }
        
        logger.info(f"Batch user management processing {len(profanity_reviews)} profanity reviews with batch_id: {batch_id}")
        
        # Group reviews by user_id for efficient processing
        user_reviews = defaultdict(list)
        for review in profanity_reviews:
            user_id = review.get('user_id', 'unknown')
            if user_id != 'unknown':
                user_reviews[user_id].append(review)
        
        logger.info(f"Processing {len(user_reviews)} users with profanity reviews")
        
        # Process users in parallel chunks for better performance
        user_list = list(user_reviews.items())
        chunk_size = min(50, len(user_list))  # Process in chunks of 50 users
        user_chunks = [user_list[i:i + chunk_size] for i in range(0, len(user_list), chunk_size)]
        
        all_results = []
        total_processed = 0
        total_failed = 0
        total_banned = 0
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            chunk_futures = []
            for chunk_idx, chunk in enumerate(user_chunks):
                future = executor.submit(process_user_chunk, chunk, f"{batch_id}-chunk-{chunk_idx}")
                chunk_futures.append(future)
            
            # Collect results from all chunks
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    chunk_result = future.result()
                    all_results.extend(chunk_result['results'])
                    total_processed += chunk_result['processed_count']
                    total_failed += chunk_result['failed_count']
                    total_banned += chunk_result['banned_count']
                except Exception as e:
                    logger.error(f"Chunk processing failed: {str(e)}")
                    total_failed += chunk_size  # Assume all in chunk failed
        
        logger.info(f"Batch {batch_id} user management complete: {total_processed} processed, {total_failed} failed, {total_banned} banned")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Batch user management completed',
                'batch_id': batch_id,
                'processed_count': total_processed,
                'failed_count': total_failed,
                'banned_count': total_banned,
                'total_users': len(user_reviews),
                'total_reviews': len(profanity_reviews),
                'results': all_results
            })
        }
        
    except Exception as e:
        logger.error(f"Batch user management failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Batch user management failed',
                'message': str(e),
                'batch_id': batch_id
            })
        }


def process_user_chunk(user_chunk: List[tuple], chunk_id: str) -> Dict[str, Any]:
    """
    Process a chunk of users for profanity statistics and banning.
    
    Args:
        user_chunk: List of (user_id, reviews) tuples
        chunk_id: Identifier for this chunk
        
    Returns:
        Dictionary with chunk processing results
    """
    logger.info(f"Processing user chunk {chunk_id} with {len(user_chunk)} users")
    
    processed_count = 0
    failed_count = 0
    banned_count = 0
    results = []
    
    # Process each user in the chunk
    for user_id, user_reviews in user_chunk:
        try:
            # Get current user statistics
            user_stats = get_user_statistics(user_id)
            
            # Process all profanity reviews for this user
            review_ids = [review['review_id'] for review in user_reviews]
            
            # Update user statistics for all unpolite reviews
            updated_stats = update_user_batch_unpolite_count(user_id, review_ids, user_stats, len(user_reviews))
            
            # Check if user should be banned
            user_banned = False
            if should_ban_user(updated_stats):
                ban_result = ban_user(user_id, review_ids[-1], updated_stats)  # Use last review as trigger
                user_banned = ban_result.get('banned', False)
                if user_banned:
                    banned_count += 1
            
            processed_count += 1
            results.append({
                'status': 'success',
                'user_id': user_id,
                'chunk_id': chunk_id,
                'review_count': len(user_reviews),
                'unpolite_count': updated_stats['unpolite_review_count'],
                'banned': user_banned
            })
            
            if user_banned:
                logger.info(f"✅ User {user_id} banned due to {updated_stats['unpolite_review_count']} unpolite reviews")
            else:
                logger.info(f"✅ User {user_id} statistics updated: {updated_stats['unpolite_review_count']} unpolite reviews")
            
        except Exception as e:
            logger.error(f"Failed to process user {user_id} in chunk {chunk_id}: {str(e)}")
            failed_count += 1
            results.append({
                'status': 'failed',
                'error': str(e),
                'user_id': user_id,
                'chunk_id': chunk_id
            })
    
    logger.info(f"Chunk {chunk_id} user management complete: {processed_count} processed, {failed_count} failed, {banned_count} banned")
    
    return {
        'processed_count': processed_count,
        'failed_count': failed_count,
        'banned_count': banned_count,
        'results': results
    }


def update_user_batch_unpolite_count(user_id: str, review_ids: List[str], current_stats: Dict[str, Any], review_count: int) -> Dict[str, Any]:
    """
    Update user statistics for multiple unpolite reviews in batch.
    
    Args:
        user_id: User identifier
        review_ids: List of review IDs
        current_stats: Current user statistics
        review_count: Number of new unpolite reviews
        
    Returns:
        Updated user statistics
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/users'))
        
        # Calculate new statistics
        new_unpolite_count = current_stats.get('unpolite_review_count', 0) + review_count
        new_total_reviews = current_stats.get('total_reviews', 0) + review_count
        timestamp = datetime.utcnow().isoformat()
        
        # Create complete user record (PUT operation for new users or full update)
        user_item = {
            'user_id': {'S': user_id},
            'unpolite_review_count': {'N': str(new_unpolite_count)},
            'total_reviews': {'N': str(new_total_reviews)},
            'last_unpolite_review_ids': {'L': [{'S': review_id} for review_id in review_ids[-10:]]},  # Keep last 10 review IDs
            'last_updated': {'S': timestamp},
            'status': {'S': current_stats.get('status', USER_ACTIVE_STATUS)}
        }
        
        # Add created_at for new users
        if current_stats.get('unpolite_review_count', 0) == 0:
            user_item['created_at'] = {'S': timestamp}
        else:
            # Preserve existing created_at
            user_item['created_at'] = {'S': current_stats.get('created_at', timestamp)}
        
        # Use PUT operation to create/update user record
        success = put_dynamodb_item(table_name, user_item)
        
        if success:
            logger.info(f"Updated user {user_id} statistics: {new_unpolite_count} unpolite reviews")
            return {
                'user_id': user_id,
                'unpolite_review_count': new_unpolite_count,
                'total_reviews': new_total_reviews,
                'status': current_stats.get('status', USER_ACTIVE_STATUS),
                'created_at': user_item['created_at']['S'],
                'last_updated': timestamp
            }
        else:
            logger.error(f"Failed to update user {user_id} statistics")
            return current_stats
            
    except Exception as e:
        logger.error(f"Failed to update batch unpolite count for user {user_id}: {str(e)}")
        return current_stats


def handle_direct_invocation(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle direct invocation from sentiment analysis Lambda.
    
    Args:
        event: Event with user and review information
        
    Returns:
        Processing result
    """
    try:
        user_id = event['user_id']
        review_id = event['review_id']
        trigger_reason = event.get('trigger_reason', 'unknown')
        
        logger.info(f"Processing user management for user {user_id}, reason: {trigger_reason}")
        
        # Get current user statistics
        user_stats = get_user_statistics(user_id)
        
        # Update user statistics for unpolite review
        if trigger_reason == 'unpolite_review':
            updated_stats = update_user_unpolite_count(user_id, review_id, user_stats)
            
            # Check if user should be banned
            if should_ban_user(updated_stats):
                ban_result = ban_user(user_id, review_id, updated_stats)
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'User banned due to excessive unpolite reviews',
                        'user_id': user_id,
                        'review_id': review_id,
                        'unpolite_count': updated_stats['unpolite_review_count'],
                        'banned': True,
                        'ban_result': ban_result
                    })
                }
            else:
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'User statistics updated',
                        'user_id': user_id,
                        'review_id': review_id,
                        'unpolite_count': updated_stats['unpolite_review_count'],
                        'banned': False
                    })
                }
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'User management completed',
                'user_id': user_id,
                'trigger_reason': trigger_reason
            })
        }
        
    except Exception as e:
        logger.error(f"Failed to handle direct invocation: {str(e)}")
        raise


def handle_dynamodb_stream_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle DynamoDB stream events.
    
    Args:
        event: DynamoDB stream event
        
    Returns:
        Processing result
    """
    try:
        processed_records = []
        
        for record in event['Records']:
            try:
                event_name = record['eventName']
                
                if event_name in ['INSERT', 'MODIFY']:
                    result = process_dynamodb_record(record)
                    processed_records.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to process DynamoDB record: {str(e)}")
                processed_records.append({
                    'status': 'failed',
                    'error': str(e),
                    'record_id': record.get('dynamodb', {}).get('Keys', {}).get('review_id', {}).get('S', 'unknown')
                })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'DynamoDB stream processing completed',
                'processed_count': len([r for r in processed_records if r.get('status') == 'success']),
                'failed_count': len([r for r in processed_records if r.get('status') == 'failed']),
                'results': processed_records
            })
        }
        
    except Exception as e:
        logger.error(f"Failed to handle DynamoDB stream event: {str(e)}")
        raise


def process_dynamodb_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single DynamoDB stream record.
    
    Args:
        record: DynamoDB stream record
        
    Returns:
        Processing result
    """
    try:
        # Extract data from DynamoDB record
        dynamodb_data = record['dynamodb']
        
        if 'NewImage' in dynamodb_data:
            new_image = dynamodb_data['NewImage']
            
            # Check if this is a completed review with profanity
            if (new_image.get('status', {}).get('S') == 'completed' and
                new_image.get('processing_summary', {}).get('M', {}).get('is_unpolite', {}).get('BOOL') == True):
                
                # Extract user ID and review ID
                user_id = new_image.get('original_data', {}).get('M', {}).get('reviewerID', {}).get('S')
                review_id = new_image.get('review_id', {}).get('S')
                
                if user_id and review_id:
                    logger.info(f"Processing unpolite review {review_id} for user {user_id}")
                    
                    # Get current user statistics
                    user_stats = get_user_statistics(user_id)
                    
                    # Update user statistics
                    updated_stats = update_user_unpolite_count(user_id, review_id, user_stats)
                    
                    # Check if user should be banned
                    if should_ban_user(updated_stats):
                        ban_result = ban_user(user_id, review_id, updated_stats)
                        
                        return {
                            'status': 'success',
                            'action': 'user_banned',
                            'user_id': user_id,
                            'review_id': review_id,
                            'unpolite_count': updated_stats['unpolite_review_count']
                        }
                    else:
                        return {
                            'status': 'success',
                            'action': 'stats_updated',
                            'user_id': user_id,
                            'review_id': review_id,
                            'unpolite_count': updated_stats['unpolite_review_count']
                        }
        
        return {
            'status': 'success',
            'action': 'no_action_needed',
            'reason': 'Record does not require user management action'
        }
        
    except Exception as e:
        logger.error(f"Failed to process DynamoDB record: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def get_user_statistics(user_id: str) -> Dict[str, Any]:
    """
    Get current user statistics.
    
    Args:
        user_id: User identifier
        
    Returns:
        User statistics dictionary
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/users'))
        
        response = aws_clients.dynamodb.get_item(
            TableName=table_name,
            Key={'user_id': {'S': user_id}}
        )
        
        if 'Item' in response:
            item = response['Item']
            return {
                'user_id': user_id,
                'unpolite_review_count': int(item.get('unpolite_review_count', {}).get('N', '0')),
                'total_reviews': int(item.get('total_reviews', {}).get('N', '0')),
                'status': item.get('status', {}).get('S', USER_ACTIVE_STATUS),
                'created_at': item.get('created_at', {}).get('S', datetime.utcnow().isoformat()),
                'updated_at': item.get('updated_at', {}).get('S', datetime.utcnow().isoformat())
            }
        else:
            # Create new user record
            return {
                'user_id': user_id,
                'unpolite_review_count': 0,
                'total_reviews': 0,
                'status': USER_ACTIVE_STATUS,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to get user statistics: {str(e)}")
        return {
            'user_id': user_id,
            'unpolite_review_count': 0,
            'total_reviews': 0,
            'status': USER_ACTIVE_STATUS,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }


def update_user_unpolite_count(user_id: str, review_id: str, current_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update user's unpolite review count.
    
    Args:
        user_id: User identifier
        review_id: Review identifier that triggered the update
        current_stats: Current user statistics
        
    Returns:
        Updated user statistics
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/users'))
        
        # Prepare update expression
        update_expression = "ADD unpolite_review_count :inc, total_review_count :inc SET last_unpolite_review = :review_id, updated_at = :timestamp"
        
        # If this is a new user, also set created_at
        if current_stats['unpolite_review_count'] == 0 and current_stats['total_review_count'] == 0:
            update_expression += ", created_at = :timestamp"
        
        expression_attribute_values = {
            ':inc': {'N': '1'},
            ':review_id': {'S': review_id},
            ':timestamp': {'S': datetime.utcnow().isoformat()}
        }
        
        response = aws_clients.dynamodb.update_item(
            TableName=table_name,
            Key={'user_id': {'S': user_id}},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues='ALL_NEW'
        )
        
        # Extract updated statistics
        item = response['Attributes']
        updated_stats = {
            'user_id': user_id,
            'unpolite_review_count': int(item['unpolite_review_count']['N']),
            'total_review_count': int(item['total_review_count']['N']),
            'status': item.get('status', {}).get('S', USER_ACTIVE_STATUS),
            'created_at': item.get('created_at', {}).get('S'),
            'updated_at': item['updated_at']['S'],
            'last_unpolite_review': item['last_unpolite_review']['S']
        }
        
        logger.info(f"Updated user {user_id} unpolite count to {updated_stats['unpolite_review_count']}")
        return updated_stats
        
    except Exception as e:
        logger.error(f"Failed to update user unpolite count: {str(e)}")
        # Return current stats with incremented count as fallback
        current_stats['unpolite_review_count'] += 1
        current_stats['total_review_count'] += 1
        return current_stats


def should_ban_user(user_stats: Dict[str, Any]) -> bool:
    """
    Determine if user should be banned based on unpolite review count.
    
    Args:
        user_stats: User statistics
        
    Returns:
        True if user should be banned, False otherwise
    """
    unpolite_count = user_stats['unpolite_review_count']
    current_status = user_stats['status']
    
        # Ban if unpolite count exceeds threshold and user is not already banned
    should_ban = (unpolite_count >= MAX_UNPOLITE_REVIEWS and
                  current_status != USER_BAN_STATUS)
    
    user_id = user_stats.get('user_id', 'unknown')
    logger.info(f"User {user_id}: unpolite_count={unpolite_count}, "
                f"threshold={MAX_UNPOLITE_REVIEWS}, current_status={current_status}, "
                f"should_ban={should_ban}")
    
    return should_ban


def ban_user(user_id: str, triggering_review_id: str, user_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ban a user for excessive unpolite reviews.
    
    Args:
        user_id: User identifier
        triggering_review_id: Review that triggered the ban
        user_stats: Current user statistics
        
    Returns:
        Ban operation result
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/users'))
        
        ban_timestamp = datetime.utcnow().isoformat()
        
        # Update user status to banned
        response = aws_clients.dynamodb.update_item(
            TableName=table_name,
            Key={'user_id': {'S': user_id}},
            UpdateExpression="SET #status = :banned_status, banned_at = :ban_time, ban_reason = :reason, triggering_review = :review_id, updated_at = :timestamp",
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':banned_status': {'S': USER_BAN_STATUS},
                ':ban_time': {'S': ban_timestamp},
                ':reason': {'S': f'Exceeded maximum unpolite reviews ({MAX_UNPOLITE_REVIEWS})'},
                ':review_id': {'S': triggering_review_id},
                ':timestamp': {'S': ban_timestamp}
            },
            ReturnValues='ALL_NEW'
        )
        
        ban_result = {
            'user_id': user_id,
            'banned': True,
            'banned_at': ban_timestamp,
            'unpolite_review_count': user_stats['unpolite_review_count'],
            'triggering_review': triggering_review_id,
            'ban_reason': f'Exceeded maximum unpolite reviews ({MAX_UNPOLITE_REVIEWS})'
        }
        
        logger.warning(f"BANNED USER: {user_id} due to {user_stats['unpolite_review_count']} unpolite reviews. "
                      f"Triggering review: {triggering_review_id}")
        
        return ban_result
        
    except Exception as e:
        logger.error(f"Failed to ban user {user_id}: {str(e)}")
        return {
            'user_id': user_id,
            'banned': False,
            'error': str(e),
            'attempted_at': datetime.utcnow().isoformat()
        }


def handle_stage_based_call(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle stage-based direct call (for testing/manual pipeline).
    
    Args:
        event: Event with review_id and stage
        
    Returns:
        Processing result
    """
    try:
        review_id = event['review_id']
        logger.info(f"Processing stage-based user management for review: {review_id}")
        
        # Get review data from DynamoDB
        review_data = get_review_from_db(review_id)
        if not review_data:
            raise Exception(f"Review {review_id} not found in database")
        
        # Extract user ID from review data
        original_data = review_data.get('original_data', {})
        if isinstance(original_data, dict) and 'M' in original_data:
            original_data = original_data['M']
        
        user_id = None
        if 'reviewerID' in original_data:
            user_id = original_data['reviewerID'].get('S', '')
        
        if not user_id:
            raise Exception(f"No user ID found in review {review_id}")
        
        # Check if this review has profanity (unpolite)
        profanity_check = review_data.get('profanity_check', {})
        if isinstance(profanity_check, dict) and 'M' in profanity_check:
            profanity_check = profanity_check['M']
        
        has_profanity = profanity_check.get('has_profanity', {}).get('BOOL', False)
        
        if has_profanity:
            # Process as unpolite review
            logger.info(f"Processing unpolite review {review_id} for user {user_id}")
            
            # Get current user statistics
            user_stats = get_user_statistics(user_id)
            
            # Update user statistics
            updated_stats = update_user_unpolite_count(user_id, review_id, user_stats)
            
            # Check if user should be banned
            if should_ban_user(updated_stats):
                ban_result = ban_user(user_id, review_id, updated_stats)
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'User banned due to excessive unpolite reviews',
                        'user_id': user_id,
                        'review_id': review_id,
                        'unpolite_count': updated_stats['unpolite_review_count'],
                        'banned': True,
                        'ban_result': ban_result
                    })
                }
            else:
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'User statistics updated for unpolite review',
                        'user_id': user_id,
                        'review_id': review_id,
                        'unpolite_count': updated_stats['unpolite_review_count'],
                        'banned': False
                    })
                }
        else:
            # Review is polite, just update general stats
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'User management completed - polite review',
                    'user_id': user_id,
                    'review_id': review_id,
                    'action': 'no_action_needed'
                })
            }
        
    except Exception as e:
        logger.error(f"Failed to handle stage-based call: {str(e)}")
        raise


def get_review_from_db(review_id: str) -> Dict[str, Any]:
    """
    Retrieve review data from DynamoDB.
    
    Args:
        review_id: Review identifier
        
    Returns:
        Review data or None if not found
    """
    try:
        table_name = get_parameter_store_value(get_ssm_parameter_path('tables/reviews'))
        
        response = aws_clients.dynamodb.query(
            TableName=table_name,
            KeyConditionExpression='review_id = :review_id',
            ExpressionAttributeValues={
                ':review_id': {'S': review_id}
            }
        )
        
        if response['Items']:
            return response['Items'][0]
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve review {review_id}: {str(e)}")
        return None