#!/usr/bin/env python3
"""
Multi-Instance Batch Review Processor for Cloud Review Cleaner

This script distributes review processing across multiple LocalStack instances
for improved parallel processing and load distribution.
"""

import json
import time
import boto3
import argparse
import concurrent.futures
from typing import Dict, List, Any, Tuple
import uuid
from datetime import datetime
import math

# Configuration for multiple LocalStack instances
LOCALSTACK_INSTANCES = [
    {
        'name': 'instance-1', 
        'endpoint': 'http://localhost:4566', 
        'lambda_endpoint': 'http://host.docker.internal:4566',
        'port': 4566
    },
    {
        'name': 'instance-2', 
        'endpoint': 'http://localhost:4567', 
        'lambda_endpoint': 'http://host.docker.internal:4567',
        'port': 4567
    },
    {
        'name': 'instance-3', 
        'endpoint': 'http://localhost:4568', 
        'lambda_endpoint': 'http://host.docker.internal:4568',
        'port': 4568
    }
]

AWS_REGION = 'us-east-1'
PROJECT_NAME = 'cloud-review-cleaner'

def create_aws_clients(instance_config: Dict):
    """Create AWS clients configured for a specific LocalStack instance."""
    config = {
        'endpoint_url': instance_config['endpoint'],
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

def distribute_reviews_across_instances(reviews: List[Dict[str, Any]], num_instances: int) -> List[List[Dict[str, Any]]]:
    """
    Distribute reviews across instances for parallel processing.
    
    Args:
        reviews: List of all reviews
        num_instances: Number of LocalStack instances
        
    Returns:
        List of review chunks, one per instance
    """
    reviews_per_instance = math.ceil(len(reviews) / num_instances)
    
    distributed_reviews = []
    for i in range(num_instances):
        start_idx = i * reviews_per_instance
        end_idx = min((i + 1) * reviews_per_instance, len(reviews))
        instance_reviews = reviews[start_idx:end_idx]
        distributed_reviews.append(instance_reviews)
    
    print(f"📦 Distributed {len(reviews)} reviews across {num_instances} instances:")
    for i, instance_reviews in enumerate(distributed_reviews):
        print(f"   Instance {i+1}: {len(instance_reviews)} reviews")
    
    return distributed_reviews

def create_review_batches(reviews: List[Dict[str, Any]], batch_size: int = 3000) -> List[List[Dict[str, Any]]]:
    """
    Split reviews into smaller batches for processing.
    
    Args:
        reviews: List of reviews
        batch_size: Size of each batch
        
    Returns:
        List of review batches
    """
    batches = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batches.append(batch)
    
    return batches

def process_batch(clients: Dict[str, Any], reviews_batch: List[Dict[str, Any]], 
                 batch_id: str, instance_name: str) -> Dict[str, Any]:
    """
    Process a single batch of reviews through the Lambda function chain.
    
    Args:
        clients: AWS service clients for the instance
        reviews_batch: Batch of reviews to process
        batch_id: Unique identifier for this batch
        instance_name: Name of the LocalStack instance
        
    Returns:
        Processing results
    """
    lambda_client = clients['lambda']
    
    print(f"🚀 [{instance_name}] Processing batch {batch_id} ({len(reviews_batch)} reviews)...")
    
    # Prepare payload for preprocessing Lambda (same as batch_processor.py)
    payload = {
        "batch_processing": True,
        "reviews": reviews_batch,
        "batch_id": batch_id
    }
    
    function_name = f"{PROJECT_NAME}-{instance_name}-preprocess"
    
    try:
        # Invoke preprocessing Lambda with batch processing
        response = lambda_client.invoke(
            FunctionName=function_name,
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
            
            print(f"   ✅ [{instance_name}] Batch {batch_id}: {processed_count} processed, {failed_count} failed")
            
            # Wait for the pipeline to complete
            wait_for_batch_completion(clients, body.get('review_ids', []), batch_id, instance_name)
            
            return {
                'success': True,
                'instance': instance_name,
                'batch_id': batch_id,
                'processed': processed_count,
                'failed': failed_count,
                'details': body
            }
        else:
            print(f"   ❌ [{instance_name}] Batch {batch_id} failed with status {response['StatusCode']}")
            return {
                'success': False,
                'instance': instance_name,
                'batch_id': batch_id,
                'processed': 0,
                'failed': len(reviews_batch),
                'error': result.get('body', 'Unknown error')
            }
            
    except Exception as e:
        print(f"   ❌ [{instance_name}] Error processing batch {batch_id}: {e}")
        return {
            'success': False,
            'instance': instance_name,
            'batch_id': batch_id,
            'processed': 0,
            'failed': len(reviews_batch),
            'error': str(e)
        }

def wait_for_batch_completion(clients: Dict[str, Any], review_ids: List[str], 
                            batch_id: str, instance_name: str, timeout: int = 300):
    """
    Wait for a batch of reviews to complete processing through the entire pipeline.
    
    Args:
        clients: AWS service clients
        review_ids: List of review IDs to monitor
        batch_id: Batch identifier for logging
        instance_name: Name of the instance
        timeout: Maximum time to wait in seconds
    """
    if not review_ids:
        return
    
    dynamodb = clients['dynamodb']
    table_name = f"{PROJECT_NAME}-{instance_name}-reviews"
    
    print(f"⏳ [{instance_name}] Waiting for batch {batch_id} pipeline completion ({len(review_ids)} reviews)...")
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
                    continue  # Skip this review and continue checking others
            
            completion_rate = completed_count / len(review_ids)
            
            if completion_rate >= 0.95:  # 95% completion threshold
                print(f"   ✅ [{instance_name}] Batch {batch_id} pipeline completed ({completed_count}/{len(review_ids)} reviews)")
                return
            
            if time.time() - start_time > 30:  # Log progress every 30 seconds
                print(f"   ⏳ [{instance_name}] Batch {batch_id} progress: {completed_count}/{len(review_ids)} completed ({completion_rate:.1%})")
                start_time += 30  # Reset timer for next progress update
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            print(f"   ⚠️ [{instance_name}] Error checking batch {batch_id} completion: {e}")
            time.sleep(5)
    
    print(f"   ⚠️ [{instance_name}] Batch {batch_id} completion timeout")

def process_instance_reviews(instance_config: Dict, reviews: List[Dict[str, Any]], 
                           batch_size: int = 3000) -> Dict[str, Any]:
    """
    Process all reviews assigned to a specific instance.
    
    Args:
        instance_config: Configuration for the LocalStack instance
        reviews: Reviews to process on this instance
        batch_size: Size of processing batches
        
    Returns:
        Processing results for this instance
    """
    instance_name = instance_config['name']
    
    if not reviews:
        print(f"📭 [{instance_name}] No reviews to process")
        return {
            'instance': instance_name,
            'total_reviews': 0,
            'total_processed': 0,
            'total_failed': 0,
            'batches': []
        }
    
    print(f"\n🏭 [{instance_name}] Starting processing of {len(reviews)} reviews...")
    
    try:
        # Create clients for this instance
        clients = create_aws_clients(instance_config)
        
        # Split reviews into batches
        batches = create_review_batches(reviews, batch_size)
        print(f"   📦 [{instance_name}] Created {len(batches)} batches")
        
        # Process each batch
        batch_results = []
        total_processed = 0
        total_failed = 0
        
        for i, batch in enumerate(batches):
            batch_id = f"{instance_name}-{i+1}"
            result = process_batch(clients, batch, batch_id, instance_name)
            batch_results.append(result)
            
            total_processed += result.get('processed', 0)
            total_failed += result.get('failed', 0)
            
            # Small delay between batches to avoid overwhelming the instance
            if i < len(batches) - 1:
                time.sleep(2)
        
        print(f"✅ [{instance_name}] Completed processing: {total_processed} processed, {total_failed} failed")
        
        return {
            'instance': instance_name,
            'total_reviews': len(reviews),
            'total_processed': total_processed,
            'total_failed': total_failed,
            'batches': batch_results
        }
        
    except Exception as e:
        print(f"❌ [{instance_name}] Failed to process reviews: {e}")
        return {
            'instance': instance_name,
            'total_reviews': len(reviews),
            'total_processed': 0,
            'total_failed': len(reviews),
            'error': str(e),
            'batches': []
        }

def collect_results_from_all_instances(dataset_name: str = 'reviews_devset.json') -> Dict[str, Any]:
    """
    Collect and aggregate results from all LocalStack instances.
    Returns results in the same format as batch_processor.py
    
    Args:
        dataset_name: Name of the dataset file processed
        
    Returns:
        Aggregated results from all instances in standard format
    """
    print("\n📊 Collecting results from all instances...")
    
    # Aggregate counters
    total_reviews = 0
    sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
    profanity_count = 0
    all_banned_users = set()
    total_users = 0
    
    for instance_config in LOCALSTACK_INSTANCES:
        instance_name = instance_config['name']
        print(f"   📋 [{instance_name}] Collecting results...")
        
        try:
            clients = create_aws_clients(instance_config)
            dynamodb = clients['dynamodb']
            
            # Scan reviews table for completed reviews
            reviews_table = f"{PROJECT_NAME}-{instance_name}-reviews"
            
            response = dynamodb.scan(
                TableName=reviews_table,
                FilterExpression='#status = :status',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': {'S': 'completed'}}
            )
            
            reviews = response['Items']
            
            # Continue scanning if there are more items
            while 'LastEvaluatedKey' in response:
                response = dynamodb.scan(
                    TableName=reviews_table,
                    FilterExpression='#status = :status',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={':status': {'S': 'completed'}},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                reviews.extend(response['Items'])
            
            instance_reviews = len(reviews)
            total_reviews += instance_reviews
            
            # Process each review for sentiment and profanity
            for review in reviews:
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
            
            # Collect user data
            users_table = f"{PROJECT_NAME}-{instance_name}-users"
            try:
                users_response = dynamodb.scan(TableName=users_table)
                users = users_response['Items']
                
                # Continue scanning for all users if needed
                while 'LastEvaluatedKey' in users_response:
                    users_response = dynamodb.scan(
                        TableName=users_table,
                        ExclusiveStartKey=users_response['LastEvaluatedKey']
                    )
                    users.extend(users_response['Items'])
                
                # Count users and filter banned ones
                total_users += len(users)
                for user in users:
                    status = user.get('status', {}).get('S', 'active')
                    if status == 'banned':
                        user_id = user.get('user_id', {}).get('S', 'unknown')
                        all_banned_users.add(user_id)
                        
            except Exception as e:
                print(f"   ⚠️ [{instance_name}] Could not retrieve user data: {e}")
            
            print(f"   ✅ [{instance_name}] Results collected: {instance_reviews} reviews")
            
        except Exception as e:
            print(f"   ❌ [{instance_name}] Failed to collect results: {e}")
    
    # Create results in the same format as batch_processor.py
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
            'banned_users_count': len(all_banned_users),
            'banned_users': sorted(list(all_banned_users))  # Sort for consistent output
        }
    }
    
    print(f"   📈 Analysis complete:")
    print(f"      Total reviews processed: {total_reviews}")
    print(f"      Sentiment distribution: {sentiment_stats}")
    print(f"      Reviews with profanity: {profanity_count}")
    print(f"      Total users: {total_users}")
    print(f"      Banned users: {len(all_banned_users)}")
    
    return results

def save_results(results: Dict[str, Any], filename: str = 'multi_results.json') -> None:
    """
    Save analysis results to JSON file in the same format as batch_processor.py
    
    Args:
        results: Analysis results dictionary
        filename: Output filename
    """
    print(f"💾 Saving results to {filename}...")
    
    try:
        output_data = {
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Results saved to {filename}")
    except Exception as e:
        print(f"   ❌ Failed to save results: {e}")

def print_summary(processing_results: List[Dict[str, Any]], final_results: Dict[str, Any]):
    """Print a summary of processing results."""
    print("\n" + "=" * 80)
    print("📊 MULTI-INSTANCE PROCESSING SUMMARY")
    print("=" * 80)
    
    # Processing summary
    total_reviews = sum(r.get('total_reviews', 0) for r in processing_results)
    total_processed = sum(r.get('total_processed', 0) for r in processing_results)
    total_failed = sum(r.get('total_failed', 0) for r in processing_results)
    
    print(f"\n📈 Processing Statistics:")
    print(f"   Total Reviews: {total_reviews}")
    print(f"   Successfully Processed: {total_processed}")
    print(f"   Failed: {total_failed}")
    print(f"   Success Rate: {(total_processed/total_reviews*100):.1f}%" if total_reviews > 0 else "N/A")
    
    # Instance breakdown
    print(f"\n🏭 Instance Breakdown:")
    for result in processing_results:
        instance = result.get('instance', 'Unknown')
        processed = result.get('total_processed', 0)
        failed = result.get('total_failed', 0)
        total = result.get('total_reviews', 0)
        print(f"   {instance}: {processed}/{total} processed ({failed} failed)")
    
    # Analysis results - updated to match new format
    print(f"\n🎯 Analysis Results:")
    sentiment = final_results.get('sentiment_analysis', {})
    profanity = final_results.get('profanity_check', {})
    users = final_results.get('user_management', {})
    
    print(f"   Positive Reviews: {sentiment.get('positive_reviews', 0)}")
    print(f"   Negative Reviews: {sentiment.get('negative_reviews', 0)}")
    print(f"   Neutral Reviews: {sentiment.get('neutral_reviews', 0)}")
    print(f"   Reviews with Profanity: {profanity.get('failed_reviews', 0)}")
    print(f"   Banned Users: {users.get('banned_users_count', 0)}")
    
    banned_users = users.get('banned_users', [])
    if banned_users:
        print(f"   Banned User IDs: {', '.join(banned_users)}")

def verify_localstack_instances():
    """Verify that all LocalStack instances are running."""
    import requests
    
    print("🔍 Verifying LocalStack instances...")
    available_instances = []
    
    for instance in LOCALSTACK_INSTANCES:
        try:
            response = requests.get(f"{instance['endpoint']}/_localstack/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                services = health_data.get('services', {})
                print(f"   ✅ {instance['name']} is running on port {instance['port']} - Services: {len(services)} active")
                available_instances.append(instance)
            else:
                print(f"   ❌ {instance['name']} returned status {response.status_code}")
        except Exception as e:
            print(f"   ❌ {instance['name']} is not accessible: {e}")
    
    return available_instances

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Multi-Instance Batch Review Processor')
    parser.add_argument('--max-reviews', type=int, help='Maximum number of reviews to process')
    parser.add_argument('--batch-size', type=int, default=3000, help='Batch size for processing')
    parser.add_argument('--dataset', default='data/reviews_devset.json', help='Dataset file path')
    
    args = parser.parse_args()
    
    print("🌟 Multi-Instance Batch Review Processing Started")
    print("=" * 60)
    
    # Verify available instances
    available_instances = verify_localstack_instances()
    if not available_instances:
        print("\n❌ No LocalStack instances are available. Please start them first.")
        return
    
    print(f"\n📊 Using {len(available_instances)} available instances")
    
    # Load dataset
    reviews = load_dataset(args.dataset, args.max_reviews)
    if not reviews:
        print("❌ No reviews to process")
        return
    
    # Distribute reviews across available instances
    distributed_reviews = distribute_reviews_across_instances(reviews, len(available_instances))
    
    # Process reviews on all instances in parallel
    print(f"\n🚀 Starting parallel processing across {len(available_instances)} instances...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(available_instances)) as executor:
        # Submit processing tasks for each instance
        future_to_instance = {
            executor.submit(process_instance_reviews, instance, instance_reviews, args.batch_size): instance
            for instance, instance_reviews in zip(available_instances, distributed_reviews)
        }
        
        # Collect results as they complete
        processing_results = []
        for future in concurrent.futures.as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
                processing_results.append(result)
            except Exception as e:
                print(f"❌ Instance {instance['name']} processing failed: {e}")
                processing_results.append({
                    'instance': instance['name'],
                    'total_reviews': 0,
                    'total_processed': 0,
                    'total_failed': 0,
                    'error': str(e)
                })
    
    processing_time = time.time() - start_time
    print(f"\n⏱️ Parallel processing completed in {processing_time:.2f} seconds")
    
    # Collect and aggregate final results
    print("\n📊 Collecting final results...")
    dataset_name = args.dataset.split('/')[-1] if '/' in args.dataset else args.dataset
    final_results = collect_results_from_all_instances(dataset_name)
    
    # Save results in the same format as batch_processor.py
    save_results(final_results)
    
    # Print summary
    print_summary(processing_results, final_results)
    
    print(f"\n🎉 Multi-instance processing completed!")
    print(f"   Total processing time: {processing_time:.2f} seconds")
    print(f"   Results saved to: multi_results.json")

if __name__ == "__main__":
    main() 