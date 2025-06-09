#!/usr/bin/env python3
"""
Batch Review Processor for Cloud Review Cleaner

This script demonstrates how to process reviews in batches of up to 3000 at a time
using the updated Lambda functions with enhanced batch processing capabilities.
"""

import json
import time
import boto3
import argparse
from typing import Dict, List, Any
import uuid
from datetime import datetime

# LocalStack configuration
ENDPOINT_URL = 'http://localhost:4566'
AWS_REGION = 'us-east-1'

def create_aws_clients():
    """Create AWS clients configured for LocalStack."""
    config = {
        'endpoint_url': ENDPOINT_URL,
        'aws_access_key_id': 'test',
        'aws_secret_access_key': 'test',
        'region_name': AWS_REGION
    }
    
    return {
        'lambda': boto3.client('lambda', **config),
        'dynamodb': boto3.client('dynamodb', **config),
        's3': boto3.client('s3', **config)
    }


def load_dataset(file_path: str = 'data/reviews_devset.json', max_reviews: int = None) -> List[Dict[str, Any]]:
    """
    Load the reviews dataset.
    
    Args:
        file_path: Path to the reviews JSON file
        max_reviews: Maximum number of reviews to load
        
    Returns:
        List of review dictionaries
    """
    print(f"📄 Loading reviews from {file_path}...")
    
    try:
        reviews = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))
                    if max_reviews and len(reviews) >= max_reviews:
                        break
        
        print(f"   ✅ Loaded {len(reviews)} reviews from dataset")
        return reviews
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return []


def create_review_batches(reviews: List[Dict[str, Any]], batch_size: int = 3000) -> List[List[Dict[str, Any]]]:
    """
    Split reviews into batches for processing.
    
    Args:
        reviews: List of reviews
        batch_size: Size of each batch (default 3000)
        
    Returns:
        List of review batches
    """
    batches = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batches.append(batch)
    
    print(f"📦 Created {len(batches)} batches of up to {batch_size} reviews each")
    return batches


def process_batch(clients: Dict[str, Any], reviews_batch: List[Dict[str, Any]], batch_id: str) -> Dict[str, Any]:
    """
    Process a single batch of reviews through the Lambda function chain.
    
    Args:
        clients: AWS service clients
        reviews_batch: Batch of reviews to process
        batch_id: Unique identifier for this batch
        
    Returns:
        Processing results
    """
    lambda_client = clients['lambda']
    
    print(f"🚀 Processing batch {batch_id} ({len(reviews_batch)} reviews)...")
    
    # Prepare payload for preprocessing Lambda
    payload = {
        "batch_processing": True,
        "reviews": reviews_batch,
        "batch_id": batch_id
    }
    
    try:
        # Invoke preprocessing Lambda with batch processing
        response = lambda_client.invoke(
            FunctionName='cloud-review-cleaner-preprocess',
            InvocationType='RequestResponse',  # Synchronous for better tracking
            Payload=json.dumps(payload)
        )
        
        # Parse response
        response_payload = response['Payload'].read()
        result = json.loads(response_payload)
        
        if response['StatusCode'] == 200:
            body = json.loads(result['body'])
            processed_count = body.get('processed_count', 0)
            failed_count = body.get('failed_count', 0)
            
            print(f"   ✅ Batch {batch_id}: {processed_count} processed, {failed_count} failed")
            
            # Wait for the pipeline to complete (profanity check, sentiment analysis, user management)
            wait_for_batch_completion(clients, body.get('review_ids', []), batch_id)
            
            return {
                'success': True,
                'batch_id': batch_id,
                'processed': processed_count,
                'failed': failed_count,
                'details': body
            }
        else:
            print(f"   ❌ Batch {batch_id} failed with status {response['StatusCode']}")
            return {
                'success': False,
                'batch_id': batch_id,
                'processed': 0,
                'failed': len(reviews_batch),
                'error': result.get('body', 'Unknown error')
            }
            
    except Exception as e:
        print(f"   ❌ Error processing batch {batch_id}: {e}")
        return {
            'success': False,
            'batch_id': batch_id,
            'processed': 0,
            'failed': len(reviews_batch),
            'error': str(e)
        }


def wait_for_batch_completion(clients: Dict[str, Any], review_ids: List[str], batch_id: str, timeout: int = 300):
    """
    Wait for a batch of reviews to complete processing through the entire pipeline.
    
    Args:
        clients: AWS service clients
        review_ids: List of review IDs to monitor
        batch_id: Batch identifier for logging
        timeout: Maximum time to wait in seconds
    """
    if not review_ids:
        return
    
    dynamodb = clients['dynamodb']
    table_name = 'cloud-review-cleaner-reviews'
    
    print(f"⏳ Waiting for batch {batch_id} pipeline completion ({len(review_ids)} reviews)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            completed_count = 0
            
            # Check completion status for each review
            for review_id in review_ids:
                try:
                    response = dynamodb.query(
                        TableName=table_name,
                        KeyConditionExpression='review_id = :rid',
                        ExpressionAttributeValues={
                            ':rid': {'S': review_id}
                        },
                        ProjectionExpression='#status, sentiment_analysis, profanity_check',
                        ExpressionAttributeNames={'#status': 'status'}
                    )
                    
                    if response['Count'] > 0:
                        item = response['Items'][0]
                        status = item.get('status', {}).get('S', '')
                        has_sentiment = 'sentiment_analysis' in item
                        has_profanity = 'profanity_check' in item
                        
                        if status == 'completed' and has_sentiment and has_profanity:
                            completed_count += 1
                            
                except Exception:
                    pass  # Skip individual review errors
            
            completion_percentage = (completed_count / len(review_ids)) * 100
            
            if completed_count == len(review_ids):
                elapsed = int(time.time() - start_time)
                print(f"   ✅ Batch {batch_id} pipeline completed in {elapsed}s (100% - {completed_count}/{len(review_ids)})")
                return
            
            # Progress update every 30 seconds
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:
                print(f"   ⏳ Batch {batch_id} progress: {completion_percentage:.1f}% ({completed_count}/{len(review_ids)}) - {elapsed}s elapsed")
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            print(f"   ⚠️ Error checking batch {batch_id} completion: {e}")
            time.sleep(10)
    
    # Timeout reached
    elapsed = int(time.time() - start_time)
    print(f"   ⏰ Batch {batch_id} timeout after {elapsed}s - may still be processing")


def analyze_results(clients: Dict[str, Any], dataset_name: str = 'reviews_devset.json') -> Dict[str, Any]:
    """
    Analyze the results from the processed reviews with memory-efficient pagination.
    
    Args:
        clients: AWS service clients
        dataset_name: Name of the dataset file processed
        
    Returns:
        Analysis results
    """
    print("📊 Analyzing processing results...")
    
    dynamodb = clients['dynamodb']
    reviews_table = 'cloud-review-cleaner-reviews'
    users_table = 'cloud-review-cleaner-users'
    
    try:
        # Use smaller pages to avoid memory issues
        page_size = 100
        
        # Initialize counters
        sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
        profanity_count = 0
        total_reviews = 0
        
        # Scan reviews in small pages
        scan_kwargs = {
            'TableName': reviews_table,
            'FilterExpression': '#status = :status',
            'ExpressionAttributeNames': {'#status': 'status'},
            'ExpressionAttributeValues': {':status': {'S': 'completed'}},
            'Limit': page_size
        }
        
        print(f"   📄 Scanning reviews in pages of {page_size}...")
        page_count = 0
        base_delay = 0.5  # Base delay of 0.5 seconds
        current_delay = base_delay
        consecutive_errors = 0
        
        while True:
            try:
                response = dynamodb.scan(**scan_kwargs)
                reviews = response['Items']
                page_count += 1
                
                if reviews:
                    print(f"      Page {page_count}: {len(reviews)} reviews")
                    
                    # Process this page of reviews
                    for review in reviews:
                        total_reviews += 1
                        
                        # Check sentiment
                        if 'sentiment_analysis' in review:
                            sentiment_data = review['sentiment_analysis']['M']
                            sentiment_label = sentiment_data.get('sentiment_label', {}).get('S', 'neutral')
                            if sentiment_label in sentiment_stats:
                                sentiment_stats[sentiment_label] += 1
                        
                        # Check profanity
                        if 'profanity_check' in review:
                            profanity_data = review['profanity_check']['M']
                            has_profanity = profanity_data.get('has_profanity', {}).get('BOOL', False)
                            if has_profanity:
                                profanity_count += 1
                
                # Success - reset error count and reduce delay if it was increased
                consecutive_errors = 0
                current_delay = max(base_delay, current_delay * 0.9)  # Gradually reduce delay on success
                
                # Check if there are more pages
                if 'LastEvaluatedKey' not in response:
                    break
                    
                scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
                
                # Adaptive delay based on system health
                if current_delay > base_delay:
                    print(f"      ⏳ Using adaptive delay: {current_delay:.2f}s (system recovery)")
                time.sleep(current_delay)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"      ⚠️ Error on page {page_count}: {e}")
                
                # Exponential backoff on errors (up to 5 seconds max)
                current_delay = min(5.0, base_delay * (2 ** consecutive_errors))
                print(f"      ⏳ Backing off for {current_delay:.2f}s (error #{consecutive_errors})")
                time.sleep(current_delay)
                
                # Try to continue with next page if possible
                if 'LastEvaluatedKey' in locals() and 'response' in locals() and 'LastEvaluatedKey' in response:
                    scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
                    continue
                elif consecutive_errors < 3:
                    # Retry the same page up to 3 times
                    continue
                else:
                    print("      ❌ Too many consecutive errors, stopping scan")
                    break
        
        print(f"   ✅ Reviews analysis complete: {total_reviews} reviews processed")

        # Get user data with similar pagination approach
        try:
            print("   👥 Scanning users...")
            
            user_scan_kwargs = {
                'TableName': users_table,
                'Limit': page_size
            }
            
            all_users = []
            user_page_count = 0
            user_consecutive_errors = 0
            user_current_delay = base_delay
            
            while True:
                try:
                    user_response = dynamodb.scan(**user_scan_kwargs)
                    page_users = user_response['Items']
                    user_page_count += 1
                    
                    if page_users:
                        print(f"      User page {user_page_count}: {len(page_users)} users")
                        all_users.extend(page_users)
                    
                    # Success - reset error count and reduce delay if it was increased
                    user_consecutive_errors = 0
                    user_current_delay = max(base_delay, user_current_delay * 0.9)
                    
                    if 'LastEvaluatedKey' not in user_response:
                        break
                        
                    user_scan_kwargs['ExclusiveStartKey'] = user_response['LastEvaluatedKey']
                    
                    # Adaptive delay for user scanning
                    if user_current_delay > base_delay:
                        print(f"      ⏳ User scan adaptive delay: {user_current_delay:.2f}s")
                    time.sleep(user_current_delay)
                    
                except Exception as e:
                    user_consecutive_errors += 1
                    print(f"      ⚠️ Error on user page {user_page_count}: {e}")
                    
                    # Exponential backoff on errors
                    user_current_delay = min(5.0, base_delay * (2 ** user_consecutive_errors))
                    print(f"      ⏳ User scan backing off for {user_current_delay:.2f}s (error #{user_consecutive_errors})")
                    time.sleep(user_current_delay)
                    
                    if user_consecutive_errors >= 3:
                        print("      ❌ Too many user scan errors, stopping user analysis")
                        break
            
            # Filter banned users
            banned_users = []
            for user in all_users:
                status = user.get('status', {}).get('S', 'active')
                if status == 'banned':
                    user_id = user.get('user_id', {}).get('S', 'unknown')
                    banned_users.append(user_id)
            
            total_users = len(all_users)
            banned_user_count = len(banned_users)
            
            print(f"   ✅ User analysis complete: {total_users} users, {banned_user_count} banned")
                
        except Exception as e:
            print(f"   ⚠️ Could not retrieve user data: {e}")
            total_users = 0
            banned_users = []
            banned_user_count = 0
        
        results = {
            'dataset': dataset_name,
            'processed_count': total_reviews,
            'sentiment_analysis': {
                'total_reviews': total_reviews,
                'positive_reviews': sentiment_stats['positive'],
                'neutral_reviews': sentiment_stats['neutral'],
                'negative_reviews': sentiment_stats['negative']
            },
            'profanity_check': {
                'failed_reviews': profanity_count
            },
            'user_management': {
                'total_users': total_users,
                'banned_users_count': banned_user_count,
                'banned_users': sorted(banned_users)  # Sort for consistent output
            }
        }
        
        print(f"   📈 Analysis complete:")
        print(f"      Total reviews processed: {total_reviews}")
        print(f"      Sentiment distribution: {sentiment_stats}")
        print(f"      Reviews with profanity: {profanity_count}")
        print(f"      Total users: {total_users}")
        print(f"      Banned users: {banned_user_count}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        return {}


def save_results(results: Dict[str, Any], filename: str = 'results.json') -> None:
    """
    Save analysis results to JSON file in the specified format.
    
    Args:
        results: Analysis results dictionary
        filename: Output filename
    """
    try:
        output_data = {
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    """Main function to run batch processing."""
    parser = argparse.ArgumentParser(description='Batch process reviews using Lambda functions')
    parser.add_argument('--max-reviews', type=int, help='Maximum number of reviews to process')
    parser.add_argument('--batch-size', type=int, default=3000, help='Reviews per batch (default: 3000)')
    parser.add_argument('--file', type=str, default='data/reviews_devset.json', help='Reviews file path')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze existing results')
    parser.add_argument('--output', type=str, default='results.json', help='Output JSON file (default: results.json)')
    
    args = parser.parse_args()
    
    print("🎯 Cloud Review Cleaner - Batch Processor")
    print("=" * 50)
    
    # Create AWS clients
    clients = create_aws_clients()
    
    # Extract dataset name from file path
    dataset_name = args.file.split('/')[-1] if '/' in args.file else args.file
    
    if args.analyze_only:
        results = analyze_results(clients, dataset_name)
        if results:
            save_results(results, args.output)
        return
    
    # Load reviews
    reviews = load_dataset(args.file, args.max_reviews)
    if not reviews:
        print("❌ No reviews loaded, exiting")
        return
    
    # Create batches
    batches = create_review_batches(reviews, args.batch_size)
    
    # Process batches
    print(f"\n🔄 Processing {len(batches)} batches...")
    results = []
    
    for i, batch in enumerate(batches, 1):
        batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i:03d}"
        result = process_batch(clients, batch, batch_id)
        results.append(result)
        
        # Brief pause between batches to avoid overwhelming the system
        if i < len(batches):
            time.sleep(2)
    
    # Summary
    print(f"\n📋 Processing Summary:")
    total_processed = sum(r['processed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    successful_batches = sum(1 for r in results if r['success'])
    
    print(f"   Batches: {successful_batches}/{len(batches)} successful")
    print(f"   Reviews: {total_processed} processed, {total_failed} failed")
    
    # Analyze results
    if total_processed > 0:
        print(f"\n📊 Final Analysis:")
        results = analyze_results(clients, dataset_name)
        if results:
            save_results(results, args.output)


if __name__ == "__main__":
    main() 