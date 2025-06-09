#!/usr/bin/env python3
"""
Multi-instance deployment script for AWS Review Cleaner serverless application.
This script deploys Lambda functions across multiple LocalStack instances for parallel processing.
"""

import os
import json
import time
import boto3
import zipfile
import tempfile
import shutil
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration for multiple LocalStack instances
AWS_REGION = 'us-east-1'
PROJECT_NAME = 'cloud-review-cleaner'

# Define 3 LocalStack instances with different ports
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

def create_aws_clients(instance_config: Dict):
    """Create AWS clients configured for a specific LocalStack instance."""
    config = {
        'endpoint_url': instance_config['endpoint'],
        'aws_access_key_id': 'test',
        'aws_secret_access_key': 'test',
        'region_name': AWS_REGION
    }
    
    return {
        's3': boto3.client('s3', **config),
        'lambda': boto3.client('lambda', **config),
        'dynamodb': boto3.client('dynamodb', **config),
        'ssm': boto3.client('ssm', **config),
        'iam': boto3.client('iam', **config)
    }

def create_lambda_role(clients: Dict, instance_name: str):
    """Create IAM role for Lambda functions."""
    iam = clients['iam']
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        iam.create_role(
            RoleName='lambda-execution-role',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Path='/'
        )
        print(f"✅ [{instance_name}] Created Lambda execution role")
    except Exception as e:
        if 'EntityAlreadyExists' in str(e):
            print(f"⚠️ [{instance_name}] Lambda role already exists")
        else:
            print(f"❌ [{instance_name}] Failed to create Lambda role: {e}")

def create_s3_buckets(clients: Dict, instance_name: str):
    """Create S3 buckets for a specific instance."""
    s3 = clients['s3']
    
    buckets = [
        f"{PROJECT_NAME}-{instance_name}-raw-reviews",
        f"{PROJECT_NAME}-{instance_name}-processed-reviews", 
        f"{PROJECT_NAME}-{instance_name}-failed-reviews"
    ]
    
    print(f"📦 [{instance_name}] Creating S3 buckets...")
    for bucket in buckets:
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"   ✅ [{instance_name}] Created bucket: {bucket}")
        except Exception as e:
            if 'BucketAlreadyExists' in str(e):
                print(f"   ⚠️ [{instance_name}] Bucket already exists: {bucket}")
            else:
                print(f"   ❌ [{instance_name}] Failed to create bucket {bucket}: {e}")

def upload_resources(clients: Dict, instance_name: str):
    """Upload resource files to S3."""
    s3 = clients['s3']
    bucket = f"{PROJECT_NAME}-{instance_name}-processed-reviews"
    
    print(f"📤 [{instance_name}] Uploading resource files...")
    
    resources = [
        ('resources/bad_words.txt', 'resources/bad_words.txt'),
        ('resources/stopwords.txt', 'resources/stopwords.txt')
    ]
    
    for local_path, s3_key in resources:
        if os.path.exists(local_path):
            try:
                s3.upload_file(local_path, bucket, s3_key)
                print(f"   ✅ [{instance_name}] Uploaded {local_path}")
            except Exception as e:
                print(f"   ❌ [{instance_name}] Failed to upload {local_path}: {e}")

def create_dynamodb_tables(clients: Dict, instance_name: str):
    """Create DynamoDB tables for a specific instance."""
    dynamodb = clients['dynamodb']
    
    print(f"🗃️ [{instance_name}] Creating DynamoDB tables...")
    
    # Reviews table with streams (composite key: review_id + timestamp)
    try:
        dynamodb.create_table(
            TableName=f"{PROJECT_NAME}-{instance_name}-reviews",
            KeySchema=[
                {'AttributeName': 'review_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'review_id', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST',
            StreamSpecification={
                'StreamEnabled': True,
                'StreamViewType': 'NEW_AND_OLD_IMAGES'
            }
        )
        print(f"   ✅ [{instance_name}] Created table: {PROJECT_NAME}-{instance_name}-reviews")
    except Exception as e:
        if 'ResourceInUseException' in str(e):
            print(f"   ⚠️ [{instance_name}] Table already exists: {PROJECT_NAME}-{instance_name}-reviews")
        else:
            print(f"   ❌ [{instance_name}] Failed to create reviews table: {e}")
    
    # Users table
    try:
        dynamodb.create_table(
            TableName=f"{PROJECT_NAME}-{instance_name}-users",
            KeySchema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )
        print(f"   ✅ [{instance_name}] Created table: {PROJECT_NAME}-{instance_name}-users")
    except Exception as e:
        if 'ResourceInUseException' in str(e):
            print(f"   ⚠️ [{instance_name}] Table already exists: {PROJECT_NAME}-{instance_name}-users")
        else:
            print(f"   ❌ [{instance_name}] Failed to create users table: {e}")

def setup_ssm_parameters(clients: Dict, instance_name: str):
    """Setup SSM parameters for a specific instance."""
    ssm = clients['ssm']
    
    print(f"⚙️ [{instance_name}] Setting up SSM parameters...")
    
    parameters = {
        f"/cloud-review-cleaner/{instance_name}/buckets/raw-reviews": f"{PROJECT_NAME}-{instance_name}-raw-reviews",
        f"/cloud-review-cleaner/{instance_name}/buckets/processed-reviews": f"{PROJECT_NAME}-{instance_name}-processed-reviews",
        f"/cloud-review-cleaner/{instance_name}/buckets/failed-reviews": f"{PROJECT_NAME}-{instance_name}-failed-reviews",
        f"/cloud-review-cleaner/{instance_name}/tables/reviews": f"{PROJECT_NAME}-{instance_name}-reviews",
        f"/cloud-review-cleaner/{instance_name}/tables/users": f"{PROJECT_NAME}-{instance_name}-users",
        f"/cloud-review-cleaner/{instance_name}/lambdas/preprocess": f"{PROJECT_NAME}-{instance_name}-preprocess",
        f"/cloud-review-cleaner/{instance_name}/lambdas/profanity-check": f"{PROJECT_NAME}-{instance_name}-profanity-check",
        f"/cloud-review-cleaner/{instance_name}/lambdas/sentiment-analysis": f"{PROJECT_NAME}-{instance_name}-sentiment-analysis",
        f"/cloud-review-cleaner/{instance_name}/lambdas/user-management": f"{PROJECT_NAME}-{instance_name}-user-management"
    }
    
    for name, value in parameters.items():
        try:
            ssm.put_parameter(
                Name=name,
                Value=value,
                Type='String',
                Overwrite=True
            )
            print(f"   ✅ [{instance_name}] Set parameter: {name}")
        except Exception as e:
            print(f"   ❌ [{instance_name}] Failed to set parameter {name}: {e}")

def create_lambda_package(function_path: str, function_name: str, instance_name: str) -> str:
    """Create Lambda deployment package for a specific instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy function file as lambda_function.py
        shutil.copy2(function_path, os.path.join(temp_dir, 'lambda_function.py'))
        
        # Copy shared modules
        shared_src = Path('src/shared')
        if shared_src.exists():
            shared_dst = Path(temp_dir) / 'shared'
            shutil.copytree(shared_src, shared_dst)
            
            # Create __init__.py files
            (Path(temp_dir) / '__init__.py').touch()
            (shared_dst / '__init__.py').touch()
        
        # Create environment file with instance-specific configuration
        env_config = {
            'INSTANCE_NAME': instance_name,
            'PROJECT_NAME': PROJECT_NAME,
            'SSM_PREFIX': f'/cloud-review-cleaner/{instance_name}'
        }
        
        with open(os.path.join(temp_dir, 'environment.json'), 'w') as f:
            json.dump(env_config, f)
        
        # Create zip file
        zip_path = f"/tmp/{function_name}-{instance_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        return zip_path

def deploy_lambda_functions(clients: Dict, instance_name: str, instance_config: Dict):
    """Deploy Lambda functions for a specific instance."""
    lambda_client = clients['lambda']
    
    print(f"🚀 [{instance_name}] Deploying Lambda functions...")
    
    functions = [
        {'name': 'preprocess', 'path': 'src/lambdas/preprocess.py', 'handler': 'lambda_function.lambda_handler'},
        {'name': 'profanity-check', 'path': 'src/lambdas/profanity_check.py', 'handler': 'lambda_function.lambda_handler'},
        {'name': 'sentiment-analysis', 'path': 'src/lambdas/sentiment_analysis.py', 'handler': 'lambda_function.lambda_handler'},
        {'name': 'user-management', 'path': 'src/lambdas/user_management.py', 'handler': 'lambda_function.lambda_handler'}
    ]
    
    for func in functions:
        if not os.path.exists(func['path']):
            print(f"   ❌ [{instance_name}] Function file not found: {func['path']}")
            continue
        
        function_name = f"{PROJECT_NAME}-{instance_name}-{func['name']}"
        zip_path = create_lambda_package(func['path'], func['name'], instance_name)
        
        try:
            # Check if function exists
            try:
                lambda_client.get_function(FunctionName=function_name)
                # Update existing function
                with open(zip_path, 'rb') as f:
                    lambda_client.update_function_code(
                        FunctionName=function_name,
                        ZipFile=f.read()
                    )
                
                # Update function configuration for 3000 review batch support
                lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Timeout=600,        # 10 minutes for 3000 review batches
                    MemorySize=1536,    # 1.5GB for large batch processing
                    Environment={
                        'Variables': {
                            'AWS_ENDPOINT_URL': instance_config['lambda_endpoint'],  # Use host.docker.internal for Lambda
                            'AWS_ACCESS_KEY_ID': 'test',
                            'AWS_SECRET_ACCESS_KEY': 'test',
                            'AWS_DEFAULT_REGION': AWS_REGION,
                            'INSTANCE_NAME': instance_name,
                            'PROJECT_NAME': PROJECT_NAME,
                            'SSM_PREFIX': f'/cloud-review-cleaner/{instance_name}'
                        }
                    }
                )
                
                print(f"   ✅ [{instance_name}] Updated function: {function_name}")
            except lambda_client.exceptions.ResourceNotFoundException:
                # Create new function
                with open(zip_path, 'rb') as f:
                    lambda_client.create_function(
                        FunctionName=function_name,
                        Runtime='python3.9',
                        Role='arn:aws:iam::000000000000:role/lambda-execution-role',
                        Handler=func['handler'],
                        Code={'ZipFile': f.read()},
                        Timeout=600,        # 10 minutes for 3000 review batches
                        MemorySize=1536,    # 1.5GB for large batch processing
                        Environment={
                            'Variables': {
                                'AWS_ENDPOINT_URL': instance_config['lambda_endpoint'],  # Use host.docker.internal for Lambda
                                'AWS_ACCESS_KEY_ID': 'test',
                                'AWS_SECRET_ACCESS_KEY': 'test',
                                'AWS_DEFAULT_REGION': AWS_REGION,
                                'INSTANCE_NAME': instance_name,
                                'PROJECT_NAME': PROJECT_NAME,
                                'SSM_PREFIX': f'/cloud-review-cleaner/{instance_name}'
                            }
                        }
                    )
                print(f"   ✅ [{instance_name}] Created function: {function_name}")
                
        except Exception as e:
            print(f"   ❌ [{instance_name}] Failed to deploy {function_name}: {e}")
        finally:
            # Cleanup
            if os.path.exists(zip_path):
                os.remove(zip_path)

def deploy_to_instance(instance_config: Dict) -> Tuple[str, bool]:
    """Deploy to a single LocalStack instance."""
    instance_name = instance_config['name']
    
    try:
        print(f"\n🚀 Starting deployment to {instance_name} ({instance_config['endpoint']})...")
        
        # Create clients for this instance
        clients = create_aws_clients(instance_config)
        
        # Deploy components
        create_lambda_role(clients, instance_name)
        create_s3_buckets(clients, instance_name)
        upload_resources(clients, instance_name)
        create_dynamodb_tables(clients, instance_name)
        setup_ssm_parameters(clients, instance_name)
        deploy_lambda_functions(clients, instance_name, instance_config)
        
        print(f"✅ Successfully deployed to {instance_name}")
        return instance_name, True
        
    except Exception as e:
        print(f"❌ Failed to deploy to {instance_name}: {e}")
        return instance_name, False

def verify_localstack_instances():
    """Verify that all LocalStack instances are running."""
    import requests
    
    print("🔍 Verifying LocalStack instances...")
    for instance in LOCALSTACK_INSTANCES:
        try:
            response = requests.get(f"{instance['endpoint']}/_localstack/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                services = health_data.get('services', {})
                print(f"   ✅ {instance['name']} is running on port {instance['port']} - Services: {len(services)} active")
            else:
                print(f"   ❌ {instance['name']} returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ {instance['name']} is not accessible: {e}")
            return False
    return True

def main():
    """Main deployment function."""
    print("🌟 Multi-Instance LocalStack Deployment Started")
    print("=" * 60)
    
    # Verify all instances are running
    if not verify_localstack_instances():
        print("\n❌ Not all LocalStack instances are running. Please start them first.")
        print("\nTo start multiple LocalStack instances:")
        print("  Terminal 1: SERVICES=lambda,s3,dynamodb,ssm,iam PORT=4566 localstack start")
        print("  Terminal 2: SERVICES=lambda,s3,dynamodb,ssm,iam PORT=4567 localstack start")
        print("  Terminal 3: SERVICES=lambda,s3,dynamodb,ssm,iam PORT=4568 localstack start")
        return
    
    # Deploy to all instances in parallel
    print(f"\n🚀 Deploying to {len(LOCALSTACK_INSTANCES)} LocalStack instances in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(deploy_to_instance, instance) for instance in LOCALSTACK_INSTANCES]
        results = []
        
        for future in concurrent.futures.as_completed(futures):
            instance_name, success = future.result()
            results.append((instance_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Deployment Summary:")
    
    successful = [name for name, success in results if success]
    failed = [name for name, success in results if not success]
    
    if successful:
        print(f"✅ Successfully deployed to: {', '.join(successful)}")
    if failed:
        print(f"❌ Failed to deploy to: {', '.join(failed)}")
    
    print(f"\n🎉 Deployment completed! {len(successful)}/{len(LOCALSTACK_INSTANCES)} instances ready.")
    
    if len(successful) == len(LOCALSTACK_INSTANCES):
        print("\n🚀 All instances are ready for parallel processing!")
        print("Use multi_batch_processor.py to process reviews across all instances.")

if __name__ == "__main__":
    main() 