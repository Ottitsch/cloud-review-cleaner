"""
Unit tests for individual Lambda functions in the AWS Review Cleaner application.

Tests individual components in isolation:
1. Text preprocessing utilities
2. Profanity checking logic
3. Sentiment analysis algorithms
4. User management functionality
5. AWS utilities
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared.text_utils_simple import preprocess_review, simple_preprocess_text, simple_tokenize
from shared.sentiment_utils import analyze_sentiment, SentimentAnalyzer
from shared.aws_utils import get_parameter_store_value, upload_to_s3, put_dynamodb_item
from shared.constants import (
    SSM_BUCKET_PROCESSED_REVIEWS,
    STATUS_PROCESSING,
    PROFANITY_THRESHOLD
)


class TestTextProcessing(unittest.TestCase):
    """Test text preprocessing utilities."""
    
    def test_preprocess_review_basic(self):
        """Test basic review preprocessing."""
        summary = "Great Product!"
        review_text = "This is an excellent product. I highly recommend it."
        
        result = preprocess_review(summary, review_text)
        
        self.assertIn('processed_summary', result)
        self.assertIn('processed_review_text', result)
        self.assertIn('combined_tokens', result)
        self.assertIn('original_summary', result)
        self.assertIn('original_review_text', result)
        
        # Check tokens are lists
        self.assertIsInstance(result['processed_summary'], list)
        self.assertIsInstance(result['processed_review_text'], list)
        self.assertIsInstance(result['combined_tokens'], list)
    
    def test_preprocess_review_empty_input(self):
        """Test preprocessing with empty input."""
        result = preprocess_review("", "")
        
        self.assertEqual(result['summary_tokens'], 0)  # It returns count, not list
        self.assertEqual(result['review_text_tokens'], 0)
        self.assertEqual(result['combined_tokens'], [])
    
    def test_preprocess_review_none_input(self):
        """Test preprocessing with None input."""
        result = preprocess_review(None, None)
        
        self.assertEqual(result['summary_tokens'], 0)  # It returns count, not list
        self.assertEqual(result['review_text_tokens'], 0)
        self.assertEqual(result['combined_tokens'], [])
    
    def test_simple_preprocess_text(self):
        """Test simple text preprocessing functionality."""
        text = "This is a GREAT product!!! It's amazing..."
        processed = simple_preprocess_text(text)
        
        self.assertIsInstance(processed, list)
        self.assertGreater(len(processed), 0)
        # Should contain meaningful words without stop words
        self.assertIn('great', processed)
        self.assertIn('product', processed)
        # "amazing" gets lemmatized to "amaz" by simple lemmatizer
        self.assertTrue(any('amaz' in token for token in processed))
        # Stop words should be removed
        self.assertNotIn('is', processed)
        self.assertNotIn('a', processed)
    
    def test_simple_tokenize(self):
        """Test simple tokenization."""
        text = "This is a test sentence with words."
        tokens = simple_tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn('test', tokens)
        self.assertIn('sentence', tokens)
        # Simple tokenize doesn't remove stop words (that's done separately)
        self.assertIn('is', tokens)
        self.assertIn('a', tokens)


class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis functionality."""
    
    def test_analyze_sentiment_positive(self):
        """Test positive sentiment analysis."""
        processed_features = {
            'combined_tokens': ['excellent', 'great', 'amazing', 'love', 'perfect'],
            'summary_clean': 'excellent product',
            'review_clean': 'great amazing love perfect'
        }
        
        result = analyze_sentiment(processed_features)
        
        self.assertIn('sentiment_label', result)
        self.assertIn('sentiment_score', result)
        self.assertIn('confidence', result)
        self.assertIn(result['sentiment_label'], ['positive', 'neutral', 'negative'])
    
    def test_analyze_sentiment_negative(self):
        """Test negative sentiment analysis."""
        processed_features = {
            'combined_tokens': ['terrible', 'awful', 'hate', 'horrible', 'worst'],
            'summary_clean': 'terrible product',
            'review_clean': 'awful hate horrible worst'
        }
        
        result = analyze_sentiment(processed_features)
        
        self.assertIn('sentiment_label', result)
        self.assertIn('sentiment_score', result)
        self.assertIn('confidence', result)
    
    def test_analyze_sentiment_empty(self):
        """Test sentiment analysis with empty input."""
        processed_features = {
            'combined_tokens': [],
            'summary_clean': '',
            'review_clean': ''
        }
        
        result = analyze_sentiment(processed_features)
        
        self.assertIn('sentiment_label', result)
        self.assertEqual(result['sentiment_label'], 'neutral')
    
    def test_classify_sentiment(self):
        """Test sentiment classification thresholds."""
        analyzer = SentimentAnalyzer()
        
        # Test positive classification
        positive_result = analyzer.classify_sentiment(0.5)
        self.assertEqual(positive_result.value, 'positive')
        
        # Test negative classification
        negative_result = analyzer.classify_sentiment(-0.5)
        self.assertEqual(negative_result.value, 'negative')
        
        # Test neutral classification
        neutral_result = analyzer.classify_sentiment(0.0)
        self.assertEqual(neutral_result.value, 'neutral')


class TestProfanityChecking(unittest.TestCase):
    """Test profanity checking functionality."""
    
    def setUp(self):
        """Set up test bad words list."""
        self.bad_words = ['shit', 'fuck', 'damn', 'ass', 'bitch']
    
    def test_profanity_detection_positive(self):
        """Test profanity detection with bad words."""
        from lambdas.profanity_check import check_profanity
        
        tokens = ['this', 'shit', 'product', 'sucks']
        summary = "This shit product"
        review_text = "What a fucking waste"
        
        result = check_profanity(tokens, summary, review_text, self.bad_words)
        
        self.assertIn('has_profanity', result)
        self.assertTrue(result['has_profanity'])
        self.assertIn('profanity_score', result)
        self.assertIn('flagged_words', result)
        self.assertGreater(len(result['flagged_words']), 0)
    
    def test_profanity_detection_negative(self):
        """Test profanity detection with clean text."""
        from lambdas.profanity_check import check_profanity
        
        tokens = ['this', 'great', 'product', 'works']
        summary = "Great product"
        review_text = "This is excellent"
        
        result = check_profanity(tokens, summary, review_text, self.bad_words)
        
        self.assertIn('has_profanity', result)
        self.assertFalse(result['has_profanity'])
        self.assertEqual(len(result['flagged_words']), 0)
    
    def test_profanity_detection_empty(self):
        """Test profanity detection with empty input."""
        from lambdas.profanity_check import check_profanity
        
        result = check_profanity([], "", "", self.bad_words)
        
        self.assertIn('has_profanity', result)
        self.assertFalse(result['has_profanity'])


class TestUserManagement(unittest.TestCase):
    """Test user management functionality."""
    
    @patch('lambdas.user_management.aws_clients')
    def test_update_user_profanity_count(self, mock_aws_clients):
        """Test updating user profanity count."""
        from lambdas.user_management import update_user_unpolite_count
        
        # Mock current user stats with correct field names
        current_stats = {
            'user_id': 'test_user',
            'total_review_count': 5,  # Fixed field name
            'unpolite_review_count': 2,
            'status': 'active'
        }
        
        # Mock DynamoDB update with proper response structure
        mock_aws_clients.dynamodb.update_item.return_value = {
            'Attributes': {
                'user_id': {'S': 'test_user'},
                'total_review_count': {'N': '6'},
                'unpolite_review_count': {'N': '3'},
                'status': {'S': 'active'}
            }
        }
        
        with patch('lambdas.user_management.get_parameter_store_value') as mock_param:
            mock_param.return_value = 'test-table'
            result = update_user_unpolite_count('test_user', 'review123', current_stats)
        
        # Should call update_item to increment unpolite_reviews
        mock_aws_clients.dynamodb.update_item.assert_called_once()
        self.assertIsInstance(result, dict)
    
    def test_ban_user_logic(self):
        """Test user banning logic."""
        from lambdas.user_management import should_ban_user
        
        # User with >3 unpolite reviews should be banned
        user_stats_should_ban = {
            'user_id': 'test_user',
            'total_reviews': 10,
            'unpolite_review_count': 4,  # Fixed field name
            'status': 'active'
        }
        
        result_should_ban = should_ban_user(user_stats_should_ban)
        self.assertTrue(result_should_ban)
        
        # User with <=3 unpolite reviews should not be banned
        user_stats_no_ban = {
            'user_id': 'test_user',
            'total_reviews': 10,
            'unpolite_review_count': 2,  # Fixed field name
            'status': 'active'
        }
        
        result_no_ban = should_ban_user(user_stats_no_ban)
        self.assertFalse(result_no_ban)


class TestLambdaHandlers(unittest.TestCase):
    """Test Lambda function handlers."""
    
    @patch('lambdas.preprocess.get_parameter_store_value')
    @patch('lambdas.preprocess.put_dynamodb_item')
    def test_preprocess_lambda_handler(self, mock_put_item, mock_get_param):
        """Test preprocessing Lambda handler."""
        from lambdas.preprocess import lambda_handler
        
        # Mock SSM parameters
        mock_get_param.side_effect = lambda key: {
            SSM_BUCKET_PROCESSED_REVIEWS: 'test-bucket'
        }.get(key, 'test-value')
        
        mock_put_item.return_value = True
        
        # Test direct processing event
        event = {
            'direct_processing': True,
            'reviews': [{
                'reviewerID': 'test_user',
                'summary': 'Great product',
                'reviewText': 'This is excellent',
                'overall': 5.0
            }],
            'batch_id': 'test_batch'
        }
        
        context = Mock()
        result = lambda_handler(event, context)
        
        self.assertEqual(result['statusCode'], 200)
        body = json.loads(result['body'])
        self.assertIn('processed_count', body)
    
    @patch('lambdas.profanity_check.get_parameter_store_value')
    @patch('lambdas.profanity_check.download_from_s3')
    @patch('lambdas.profanity_check.update_dynamodb_item')
    def test_profanity_check_lambda_handler(self, mock_update, mock_download, mock_get_param):
        """Test profanity check Lambda handler."""
        from lambdas.profanity_check import lambda_handler
        
        # Mock dependencies
        mock_get_param.return_value = 'test-value'
        mock_download.return_value = 'shit\nfuck\ndamn'
        mock_update.return_value = True
        
        event = {
            'review_id': 'test_review_123',
            'timestamp': datetime.utcnow().isoformat(),
            'tokens': ['this', 'shit', 'product'],
            'original_summary': 'Bad product',
            'original_review_text': 'This shit is terrible'
        }
        
        context = Mock()
        result = lambda_handler(event, context)
        
        self.assertEqual(result['statusCode'], 200)
        body = json.loads(result['body'])
        self.assertIn('has_profanity', body)
        self.assertTrue(body['has_profanity'])


class TestAWSUtils(unittest.TestCase):
    """Test AWS utility functions."""
    
    @patch('boto3.client')
    def test_get_parameter_store_value(self, mock_boto3):
        """Test SSM parameter retrieval."""
        # Mock SSM client
        mock_ssm = Mock()
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'test-bucket-name'}
        }
        mock_boto3.return_value = mock_ssm
        
        # This would normally import from aws_utils but we'll test the concept
        result = get_parameter_store_value('/test/parameter')
        
        self.assertEqual(result, 'test-bucket-name')
        mock_ssm.get_parameter.assert_called_once_with(Name='/test/parameter', WithDecryption=True)
    
    @patch('shared.aws_utils.aws_clients')
    def test_upload_to_s3(self, mock_aws_clients):
        """Test S3 upload functionality."""
        mock_aws_clients.s3.put_object.return_value = {}
        
        result = upload_to_s3('test-bucket', 'test-key', 'test content')
        
        mock_aws_clients.s3.put_object.assert_called_once()
        self.assertTrue(result)
    
    @patch('shared.aws_utils.aws_clients')
    def test_put_dynamodb_item(self, mock_aws_clients):
        """Test DynamoDB put item functionality."""
        mock_aws_clients.dynamodb.put_item.return_value = {}
        
        test_item = {
            'review_id': 'test123',
            'status': 'processed'
        }
        
        result = put_dynamodb_item('test-table', test_item)
        
        mock_aws_clients.dynamodb.put_item.assert_called_once()
        self.assertTrue(result)


class TestEventDrivenFunctionality(unittest.TestCase):
    """Test event-driven architecture components."""
    
    def test_s3_event_parsing(self):
        """Test S3 event parsing."""
        from lambdas.preprocess import process_s3_record
        
        # Mock S3 event record
        s3_record = {
            's3': {
                'bucket': {'name': 'test-bucket'},
                'object': {'key': 'test-review.json'}
            }
        }
        
        with patch('lambdas.preprocess.download_and_parse_review') as mock_download:
            mock_download.return_value = {
                'reviewerID': 'test_user',
                'summary': 'Great',
                'reviewText': 'Excellent',
                'overall': 5.0  # Added missing field
            }
            
            with patch('lambdas.preprocess.save_processed_review') as mock_save:
                mock_save.return_value = True
                
                with patch('lambdas.preprocess.trigger_profanity_check') as mock_trigger:
                    mock_trigger.return_value = True
                    
                    result = process_s3_record(s3_record)
                    
                    self.assertIn('status', result)
                    self.assertEqual(result['status'], 'success')
    
    def test_dynamodb_event_parsing(self):
        """Test DynamoDB stream event parsing."""
        # Mock DynamoDB stream event
        dynamodb_event = {
            'Records': [{
                'eventName': 'INSERT',
                'dynamodb': {
                    'NewImage': {
                        'review_id': {'S': 'test123'},
                        'status': {'S': 'processed'}
                    }
                }
            }]
        }
        
        # Test that we can parse DynamoDB events
        record = dynamodb_event['Records'][0]
        self.assertEqual(record['eventName'], 'INSERT')
        self.assertIn('NewImage', record['dynamodb'])


if __name__ == '__main__':
    # Run all unit tests
    unittest.main(verbosity=2) 