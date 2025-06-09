#!/usr/bin/env python3
"""
Fixed deployment script for AWS Review Cleaner serverless application.
This script properly deploys Lambda functions and configures event triggers.
"""

import os
import json
import time
import boto3
import zipfile
import tempfile
import shutil
from pathlib import Path

# Configuration
LOCALSTACK_ENDPOINT = 'http://localhost:4566'
LAMBDA_ENDPOINT = 'http://host.docker.internal:4566'  # For Lambda functions running in Docker
AWS_REGION = 'us-east-1'
PROJECT_NAME = 'cloud-review-cleaner'

def create_aws_clients():
    """Create AWS clients configured for LocalStack."""
    config = {
        'endpoint_url': LOCALSTACK_ENDPOINT,
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

def create_lambda_role(clients):
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
        print("✅ Created Lambda execution role")
    except Exception as e:
        if 'EntityAlreadyExists' in str(e):
            print("⚠️ Lambda role already exists")
        else:
            print(f"❌ Failed to create Lambda role: {e}")

def create_s3_buckets(clients):
    """Create S3 buckets."""
    s3 = clients['s3']
    
    buckets = [
        f"{PROJECT_NAME}-raw-reviews",
        f"{PROJECT_NAME}-processed-reviews", 
        f"{PROJECT_NAME}-failed-reviews"
    ]
    
    print("📦 Creating S3 buckets...")
    for bucket in buckets:
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"   ✅ Created bucket: {bucket}")
        except Exception as e:
            if 'BucketAlreadyExists' in str(e):
                print(f"   ⚠️ Bucket already exists: {bucket}")
            else:
                print(f"   ❌ Failed to create bucket {bucket}: {e}")

def upload_resources(clients):
    """Upload resource files to S3."""
    s3 = clients['s3']
    bucket = f"{PROJECT_NAME}-processed-reviews"
    
    print("📤 Uploading resource files...")
    
    resources = [
        ('resources/bad_words.txt', 'resources/bad_words.txt'),
        ('resources/stopwords.txt', 'resources/stopwords.txt')
    ]
    
    for local_path, s3_key in resources:
        if os.path.exists(local_path):
            try:
                s3.upload_file(local_path, bucket, s3_key)
                print(f"   ✅ Uploaded {local_path}")
            except Exception as e:
                print(f"   ❌ Failed to upload {local_path}: {e}")

def create_dynamodb_tables(clients):
    """Create DynamoDB tables."""
    dynamodb = clients['dynamodb']
    
    print("🗃️ Creating DynamoDB tables...")
    
    # Reviews table with streams (composite key: review_id + timestamp)
    try:
        dynamodb.create_table(
            TableName=f"{PROJECT_NAME}-reviews",
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
        print(f"   ✅ Created table: {PROJECT_NAME}-reviews (with composite key: review_id + timestamp)")
    except Exception as e:
        if 'ResourceInUseException' in str(e):
            print(f"   ⚠️ Table already exists: {PROJECT_NAME}-reviews")
        else:
            print(f"   ❌ Failed to create reviews table: {e}")
    
    # Users table
    try:
        dynamodb.create_table(
            TableName=f"{PROJECT_NAME}-users",
            KeySchema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )
        print(f"   ✅ Created table: {PROJECT_NAME}-users")
    except Exception as e:
        if 'ResourceInUseException' in str(e):
            print(f"   ⚠️ Table already exists: {PROJECT_NAME}-users")
        else:
            print(f"   ❌ Failed to create users table: {e}")

def setup_ssm_parameters(clients):
    """Setup SSM parameters."""
    ssm = clients['ssm']
    
    print("⚙️ Setting up SSM parameters...")
    
    parameters = {
        "/cloud-review-cleaner/buckets/raw-reviews": f"{PROJECT_NAME}-raw-reviews",
        "/cloud-review-cleaner/buckets/processed-reviews": f"{PROJECT_NAME}-processed-reviews",
        "/cloud-review-cleaner/buckets/failed-reviews": f"{PROJECT_NAME}-failed-reviews",
        "/cloud-review-cleaner/tables/reviews": f"{PROJECT_NAME}-reviews",
        "/cloud-review-cleaner/tables/users": f"{PROJECT_NAME}-users",
        "/cloud-review-cleaner/lambdas/preprocess": f"{PROJECT_NAME}-preprocess",
        "/cloud-review-cleaner/lambdas/profanity-check": f"{PROJECT_NAME}-profanity-check",
        "/cloud-review-cleaner/lambdas/sentiment-analysis": f"{PROJECT_NAME}-sentiment-analysis",
        "/cloud-review-cleaner/lambdas/user-management": f"{PROJECT_NAME}-user-management"
    }
    
    for name, value in parameters.items():
        try:
            ssm.put_parameter(
                Name=name,
                Value=value,
                Type='String',
                Overwrite=True
            )
            print(f"   ✅ Set parameter: {name}")
        except Exception as e:
            print(f"   ❌ Failed to set parameter {name}: {e}")

def create_lambda_package(function_path, function_name):
    """Create Lambda deployment package."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy function file as lambda_function.py (Lambda handler expects this name)
        shutil.copy2(function_path, os.path.join(temp_dir, 'lambda_function.py'))
        
        # Copy shared modules
        shared_src = Path('src/shared')
        if shared_src.exists():
            shared_dst = Path(temp_dir) / 'shared'
            shutil.copytree(shared_src, shared_dst)
            
            # Create __init__.py files
            (Path(temp_dir) / '__init__.py').touch()
            (shared_dst / '__init__.py').touch()
        
        # Create zip file
        zip_path = f"/tmp/{function_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arc_name)
        
        return zip_path

def deploy_lambda_functions(clients):
    """Deploy Lambda functions."""
    lambda_client = clients['lambda']
    
    print("🚀 Deploying Lambda functions...")
    
    functions = {
        f"{PROJECT_NAME}-preprocess": "src/lambdas/preprocess.py",
        f"{PROJECT_NAME}-profanity-check": "src/lambdas/profanity_check.py", 
        f"{PROJECT_NAME}-sentiment-analysis": "src/lambdas/sentiment_analysis.py",
        f"{PROJECT_NAME}-user-management": "src/lambdas/user_management.py"
    }
    
    for function_name, function_file in functions.items():
        if not os.path.exists(function_file):
            print(f"   ⚠️ Function file not found: {function_file}")
            continue
            
        try:
            # Create deployment package
            zip_path = create_lambda_package(function_file, function_name)
            
            # Read zip file
            with open(zip_path, 'rb') as f:
                zip_content = f.read()
            
            # Try to update existing function first
            try:
                lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_content
                )
                
                # Update function configuration for 3000 review batch support
                lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Timeout=600,        # 10 minutes for 3000 review batches
                    MemorySize=1536,     # 1.5GB for large batch processing
                    Environment={
                        'Variables': {
                            'AWS_ENDPOINT_URL': LAMBDA_ENDPOINT,
                            'AWS_ACCESS_KEY_ID': 'test',
                            'AWS_SECRET_ACCESS_KEY': 'test',
                            'AWS_DEFAULT_REGION': AWS_REGION
                        }
                    }
                )
                
                print(f"   🔄 Updated function: {function_name}")
            except Exception as update_error:
                # Function doesn't exist, create it
                # Configure memory and timeout based on function type
                memory_size = 1536  # 1.5GB for large batch processing (3000 reviews)
                timeout = 600       # 10 minutes for 3000 review batches
                
                lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role='arn:aws:iam::000000000000:role/lambda-execution-role',
                    Handler='lambda_function.lambda_handler',
                    Code={'ZipFile': zip_content},
                    Timeout=timeout,
                    MemorySize=memory_size,
                    Environment={
                        'Variables': {
                            'AWS_ENDPOINT_URL': LAMBDA_ENDPOINT,
                            'AWS_ACCESS_KEY_ID': 'test',
                            'AWS_SECRET_ACCESS_KEY': 'test',
                            'AWS_DEFAULT_REGION': AWS_REGION
                        }
                    }
                )
                print(f"   ✅ Created function: {function_name}")
            
            # Cleanup
            os.remove(zip_path)
            
        except Exception as e:
            print(f"   ❌ Failed to deploy {function_name}: {e}")

def configure_s3_triggers(clients):
    """Configure S3 event triggers for Lambda functions."""
    s3 = clients['s3']
    lambda_client = clients['lambda']
    
    print("🔗 Configuring S3 event triggers...")
    
    bucket_name = f"{PROJECT_NAME}-raw-reviews"
    function_name = f"{PROJECT_NAME}-preprocess"
    
    try:
        # Add permission for S3 to invoke Lambda
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId='s3-trigger',
            Action='lambda:InvokeFunction',
            Principal='s3.amazonaws.com',
            SourceArn=f'arn:aws:s3:::{bucket_name}'
        )
        
        # Configure S3 event notification
        notification_config = {
            'LambdaConfigurations': [
                {
                    'Id': 'ObjectCreatedEvent',
                    'LambdaFunctionArn': f'arn:aws:lambda:{AWS_REGION}:000000000000:function:{function_name}',
                    'Events': ['s3:ObjectCreated:*']
                }
            ]
        }
        
        s3.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration=notification_config
        )
        
        print(f"   ✅ Configured S3 trigger: {bucket_name} → {function_name}")
        
    except Exception as e:
        print(f"")

def verify_deployment(clients):
    """Verify that deployment was successful."""
    print("\n🔍 Verifying deployment...")
    
    lambda_client = clients['lambda']
    s3 = clients['s3']
    dynamodb = clients['dynamodb']
    
    # Check Lambda functions
    try:
        response = lambda_client.list_functions()
        functions = [f['FunctionName'] for f in response['Functions']]
        expected_functions = [
            f"{PROJECT_NAME}-preprocess",
            f"{PROJECT_NAME}-profanity-check", 
            f"{PROJECT_NAME}-sentiment-analysis",
            f"{PROJECT_NAME}-user-management"
        ]
        
        for func in expected_functions:
            if func in functions:
                print(f"   ✅ Lambda function deployed: {func}")
            else:
                print(f"   ❌ Lambda function missing: {func}")
    except Exception as e:
        print(f"   ❌ Failed to verify Lambda functions: {e}")
    
    # Check S3 buckets
    try:
        response = s3.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        expected_buckets = [
            f"{PROJECT_NAME}-raw-reviews",
            f"{PROJECT_NAME}-processed-reviews",
            f"{PROJECT_NAME}-failed-reviews"
        ]
        
        for bucket in expected_buckets:
            if bucket in buckets:
                print(f"   ✅ S3 bucket created: {bucket}")
            else:
                print(f"   ❌ S3 bucket missing: {bucket}")
    except Exception as e:
        print(f"   ❌ Failed to verify S3 buckets: {e}")
    
    # Check DynamoDB tables
    try:
        response = dynamodb.list_tables()
        tables = response['TableNames']
        expected_tables = [
            f"{PROJECT_NAME}-reviews",
            f"{PROJECT_NAME}-users"
        ]
        
        for table in expected_tables:
            if table in tables:
                print(f"   ✅ DynamoDB table created: {table}")
            else:
                print(f"   ❌ DynamoDB table missing: {table}")
    except Exception as e:
        print(f"   ❌ Failed to verify DynamoDB tables: {e}")

def main():
    """Main deployment function."""
    print("🚀 DEPLOYING AWS REVIEW CLEANER SERVERLESS APPLICATION")
    print("=" * 60)
    
    # Create AWS clients
    print("🔧 Creating AWS clients...")
    clients = create_aws_clients()
    
    # Deploy infrastructure
    create_lambda_role(clients)
    time.sleep(1)
    
    create_s3_buckets(clients)
    time.sleep(1)
    
    upload_resources(clients)
    time.sleep(1)
    
    create_dynamodb_tables(clients)
    time.sleep(2)  # DynamoDB needs more time
    
    setup_ssm_parameters(clients)
    time.sleep(1)
    
    deploy_lambda_functions(clients)
    time.sleep(2)  # Lambda functions need time to be ready
    
    configure_s3_triggers(clients)
    time.sleep(1)
    
    verify_deployment(clients)
    
    print("\n🎉 DEPLOYMENT COMPLETED!")
    print("=" * 60)
    print("📋 Next steps:")
    print("   1. Test serverless pipeline: python test_real_pipeline.py")
    print("   2. Run integration tests: python -m pytest tests/test_integration.py -v")
    print("   3. Process full dataset with actual Lambda functions")
    print("")

if __name__ == "__main__":
    main() 