"""
AWS utilities for the Review Cleaner application.

This module provides common AWS service clients and utilities including:
- SSM Parameter Store access
- S3 operations
- DynamoDB operations
- Lambda invocation utilities
"""

import os
import json
import logging
from typing import Any, Dict, Optional, List
from functools import lru_cache
import inspect

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .constants import (
    LOCALSTACK_ENDPOINT,
    AWS_REGION,
    LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class AWSClientManager:
    """Manages AWS service clients with LocalStack support."""
    
    def __init__(self, use_localstack: bool = True):
        """
        Initialize AWS client manager.
        
        Args:
            use_localstack: Whether to use LocalStack endpoint
        """
        self.use_localstack = use_localstack
        # Detect instance-specific endpoint
        if use_localstack:
            # First priority: AWS_ENDPOINT_URL environment variable 
            self.endpoint_url = os.environ.get('AWS_ENDPOINT_URL')
            
            if not self.endpoint_url:
                # Second priority: Detect from INSTANCE_NAME
                instance_name = os.environ.get('INSTANCE_NAME', 'instance-1')
                if instance_name == 'instance-1':
                    port = 4566
                elif instance_name == 'instance-2':
                    port = 4567
                elif instance_name == 'instance-3':
                    port = 4568
                else:
                    port = 4566  # default
                
                # Use host.docker.internal for Lambda execution environment
                self.endpoint_url = f'http://host.docker.internal:{port}'
                
            logger.debug(f"Using AWS_ENDPOINT_URL: {self.endpoint_url}")
            logger.debug(f"Initialized AWS clients for {os.environ.get('INSTANCE_NAME', 'single-instance')} mode")
        else:
            self.endpoint_url = None
        self.region = AWS_REGION
        
    @lru_cache(maxsize=10)
    def get_client(self, service_name: str) -> Any:
        """
        Get a cached AWS service client.
        
        Args:
            service_name: Name of the AWS service (e.g., 's3', 'dynamodb')
            
        Returns:
            Boto3 client for the specified service
            
        Raises:
            NoCredentialsError: If AWS credentials are not configured
        """
        try:
            # Build client configuration
            client_config = {
                'service_name': service_name,
                'endpoint_url': self.endpoint_url,
                'region_name': self.region
            }
            
            # Add credentials if available in environment (for LocalStack)
            if os.environ.get('AWS_ACCESS_KEY_ID'):
                client_config['aws_access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID')
            if os.environ.get('AWS_SECRET_ACCESS_KEY'):
                client_config['aws_secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY')
            
            return boto3.client(**client_config)
        except NoCredentialsError as e:
            logger.error(f"AWS credentials not configured: {e}")
            raise
            
    @property
    def s3(self) -> Any:
        """Get S3 client."""
        return self.get_client('s3')
    
    @property
    def dynamodb(self) -> Any:
        """Get DynamoDB client."""
        return self.get_client('dynamodb')
    
    @property
    def ssm(self) -> Any:
        """Get SSM client."""
        return self.get_client('ssm')
    
    @property
    def lambda_client(self) -> Any:
        """Get Lambda client."""
        return self.get_client('lambda')


# Global client manager instance
aws_clients = AWSClientManager()

def reinitialize_clients():
    """Reinitialize the global client manager to pick up environment changes."""
    global aws_clients
    aws_clients = AWSClientManager()


def get_parameter_store_value(parameter_name: str) -> str:
    """
    Retrieve a value from AWS Systems Manager Parameter Store.
    
    Args:
        parameter_name: The name/path of the parameter to retrieve
        
    Returns:
        The parameter value as a string
        
    Raises:
        ClientError: If parameter doesn't exist or access is denied
        ValueError: If parameter name is invalid
    """
    if not parameter_name:
        raise ValueError("Parameter name cannot be empty")
        
    try:
        response = aws_clients.ssm.get_parameter(
            Name=parameter_name,
            WithDecryption=True
        )
        value = response['Parameter']['Value']
        logger.info(f"Retrieved parameter: {parameter_name}")
        return value
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ParameterNotFound':
            logger.error(f"Parameter not found: {parameter_name}")
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied for parameter: {parameter_name}")
        else:
            logger.error(f"Error retrieving parameter {parameter_name}: {e}")
        raise


def get_parameter_store_values(parameter_names: List[str]) -> Dict[str, str]:
    """
    Retrieve multiple values from AWS Systems Manager Parameter Store.
    
    Args:
        parameter_names: List of parameter names/paths to retrieve
        
    Returns:
        Dictionary mapping parameter names to their values
        
    Raises:
        ClientError: If any parameter access fails
        ValueError: If parameter names list is empty
    """
    if not parameter_names:
        raise ValueError("Parameter names list cannot be empty")
        
    try:
        response = aws_clients.ssm.get_parameters(
            Names=parameter_names,
            WithDecryption=True
        )
        
        # Check for any invalid parameters
        invalid_params = response.get('InvalidParameters', [])
        if invalid_params:
            logger.warning(f"Invalid parameters: {invalid_params}")
            
        # Build result dictionary
        result = {}
        for param in response['Parameters']:
            result[param['Name']] = param['Value']
            
        logger.info(f"Retrieved {len(result)} parameters")
        return result
        
    except ClientError as e:
        logger.error(f"Error retrieving parameters: {e}")
        raise


def upload_to_s3(bucket_name: str, key: str, data: Any, content_type: str = 'application/json') -> bool:
    """
    Upload data to S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key
        data: Data to upload (will be JSON serialized if not string)
        content_type: MIME type of the content
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        if isinstance(data, (dict, list)):
            data = json.dumps(data, indent=2)
        elif not isinstance(data, (str, bytes)):
            data = str(data)
            
        aws_clients.s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type
        )
        
        logger.info(f"Uploaded to S3: s3://{bucket_name}/{key}")
        return True
        
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def download_from_s3(bucket_name: str, key: str) -> Optional[str]:
    """
    Download data from S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key
        
    Returns:
        Downloaded content as string, None if failed
    """
    try:
        response = aws_clients.s3.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read().decode('utf-8')
        logger.info(f"Downloaded from S3: s3://{bucket_name}/{key}")
        return content
        
    except ClientError as e:
        logger.error(f"Failed to download from S3: {e}")
        return None


def validate_reviews_table_key(table_name: str, key_or_item: dict):
    """Validate that the DynamoDB key has the correct structure for the reviews table."""
    if 'cloud-review-cleaner-reviews' in table_name:
        if not (isinstance(key_or_item, dict) and 'review_id' in key_or_item):
            raise ValueError(f"DynamoDB key for reviews table must include 'review_id'. Got: {key_or_item}")
        v = key_or_item['review_id']
        if not (isinstance(v, dict) and 'S' in v and isinstance(v['S'], str)):
            raise ValueError(f"DynamoDB key 'review_id' must be a string type. Got: {v}")


def put_dynamodb_item(table_name: str, item: Dict[str, Any]) -> bool:
    """
    Put an item into DynamoDB table.
    
    Args:
        table_name: Name of the DynamoDB table
        item: Item to insert (Python dict)
        
    Returns:
        True if insert successful, False otherwise
    """
    try:
        # Defensive check for reviews table
        if 'review_id' in item:
            validate_reviews_table_key(table_name, item)
        aws_clients.dynamodb.put_item(
            TableName=table_name,
            Item=item
        )
        logger.info(f"Item inserted into DynamoDB table: {table_name} | Key: {{'review_id': {item.get('review_id')}}}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert item into DynamoDB table: {table_name} | Key: {{'review_id': {item.get('review_id')}}} | Error: {e}")
        return False


def get_dynamodb_item(table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get an item from DynamoDB table.
    
    Args:
        table_name: Name of the DynamoDB table
        key: Primary key to look up
        
    Returns:
        Item if found, None otherwise
    """
    try:
        response = aws_clients.dynamodb.get_item(
            TableName=table_name,
            Key=key
        )
        
        item = response.get('Item')
        if item:
            logger.info(f"Item retrieved from DynamoDB table: {table_name}")
            return item
        else:
            logger.info(f"Item not found in DynamoDB table: {table_name}")
            return None
            
    except ClientError as e:
        logger.error(f"Failed to get item from DynamoDB: {e}")
        return None


def update_dynamodb_item(table_name: str, key: Dict[str, Any], 
                        update_expression: Optional[str] = None, 
                        expression_attribute_values: Optional[Dict[str, Any]] = None,
                        expression_attribute_names: Optional[Dict[str, str]] = None,
                        update_attributes: Optional[Dict[str, Any]] = None) -> bool:
    """
    Update an item in DynamoDB table. Accepts either a full update expression or a dict of attributes.
    
    Args:
        table_name: Name of the DynamoDB table
        key: Primary key of the item to update
        update_expression: Update expression (e.g., "SET #status = :status")
        expression_attribute_values: Values for the update expression
        expression_attribute_names: Names for the update expression (optional)
        update_attributes: If provided, a dict of attributes to set (overrides update_expression)
        
    Returns:
        True if update successful, False otherwise
    """
    try:
        # Defensive check for reviews table
        if 'cloud-review-cleaner-reviews' in table_name:
            validate_reviews_table_key(table_name, key)
        
        if update_attributes is not None:
            set_exprs = []
            expr_attr_values = {}
            expr_attr_names = {}
            for k, v in update_attributes.items():
                set_exprs.append(f"#{k} = :{k}")
                expr_attr_values[f":{k}"] = v
                expr_attr_names[f"#{k}"] = k
            update_expression = "SET " + ", ".join(set_exprs)
            expression_attribute_values = expr_attr_values
            expression_attribute_names = expr_attr_names
        
        update_params = {
            'TableName': table_name,
            'Key': key,
            'UpdateExpression': update_expression,
            'ExpressionAttributeValues': expression_attribute_values
        }
        
        if expression_attribute_names:
            update_params['ExpressionAttributeNames'] = expression_attribute_names
        
        aws_clients.dynamodb.update_item(**update_params)
        logger.info(f"Item updated in DynamoDB table: {table_name} | Key: {key} | UpdateExpression: {update_expression}")
        return True
    except Exception as e:
        logger.error(f"Failed to update item in DynamoDB table: {table_name} | Key: {key} | Error: {e}")
        return False


def batch_write_to_dynamodb(items: List[Dict[str, Any]], table_name: Optional[str] = None) -> bool:
    """
    Batch write multiple items to DynamoDB (up to 25 items per batch).
    
    Args:
        items: List of DynamoDB items to write
        table_name: Name of the table (if None, uses the reviews table from parameter store)
        
    Returns:
        True if all items were written successfully, False otherwise
    """
    try:
        if not items:
            logger.warning("No items to write to DynamoDB")
            return True
            
        # Get table name from parameter store if not provided
        if table_name is None:
            # Detect instance-specific parameter path
            instance_name = os.environ.get('INSTANCE_NAME')
            if instance_name:
                param_path = f'/cloud-review-cleaner/{instance_name}/tables/reviews'
            else:
                param_path = '/cloud-review-cleaner/tables/reviews'
            table_name = get_parameter_store_value(param_path)
        
        # DynamoDB batch_write_item has a limit of 25 items per request
        batch_size = 25
        all_success = True
        
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            
            # Prepare batch request
            request_items = {
                table_name: [
                    {'PutRequest': {'Item': item}} for item in batch_items
                ]
            }
            
            # Execute batch write with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = aws_clients.dynamodb.batch_write_item(RequestItems=request_items)
                    
                    # Check for unprocessed items
                    unprocessed_items = response.get('UnprocessedItems', {})
                    if unprocessed_items:
                        logger.warning(f"Batch write attempt {attempt + 1}: {len(unprocessed_items.get(table_name, []))} unprocessed items")
                        if attempt < max_retries - 1:
                            # Retry with unprocessed items
                            request_items = unprocessed_items
                            import time
                            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            logger.error(f"Failed to write {len(unprocessed_items.get(table_name, []))} items after {max_retries} attempts")
                            all_success = False
                    else:
                        logger.info(f"Successfully wrote batch of {len(batch_items)} items to {table_name}")
                        break
                        
                except ClientError as e:
                    logger.error(f"Batch write attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        all_success = False
        
        return all_success
        
    except Exception as e:
        logger.error(f"Failed to batch write to DynamoDB: {e}")
        return False


def invoke_lambda(function_name: str, payload: Dict[str, Any], invocation_type: str = 'Event') -> bool:
    """
    Invoke a Lambda function asynchronously.
    
    Args:
        function_name: Name of the Lambda function
        payload: Payload to send to the function
        invocation_type: 'Event' for async, 'RequestResponse' for sync
        
    Returns:
        True if invocation successful, False otherwise
    """
    try:
        aws_clients.lambda_client.invoke(
            FunctionName=function_name,
            InvocationType=invocation_type,
            Payload=json.dumps(payload)
        )
        
        logger.info(f"Lambda function invoked: {function_name}")
        return True
        
    except ClientError as e:
        logger.error(f"Failed to invoke Lambda function: {e}")
        return False


def upload_file_to_s3(bucket: str, key: str, file_path: str) -> bool:
    """
    Upload a local file to S3.
    Args:
        bucket: S3 bucket name
        key: S3 object key
        file_path: Local file path
    Returns:
        True if upload successful, False otherwise
    """
    s3 = boto3.client('s3', endpoint_url='http://localhost:4566', aws_access_key_id='test', aws_secret_access_key='test')
    try:
        s3.upload_file(file_path, bucket, key)
        logger.info(f"Uploaded {file_path} to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to s3://{bucket}/{key}: {e}")
        return False 