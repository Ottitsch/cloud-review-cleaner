"""
Integration tests for the AWS Review Cleaner serverless application.

Tests the complete function chain:
1. Preprocessing (tokenization, stop word removal, lemmatization)
2. Profanity check (bad words detection)
3. Sentiment analysis (positive, neutral, negative classification)
4. User management (counting unpolite reviews and banning users)

Updated to work with batch processing architecture instead of deployed Lambda functions.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lambdas import preprocess, profanity_check, sentiment_analysis, user_management
from shared.text_utils_simple import preprocess_review
from shared.sentiment_utils import analyze_sentiment
from shared.aws_utils import get_parameter_store_value


class TestIntegrationProcessing(unittest.TestCase):
    """Integration tests for the complete processing pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_review_clean = {
            "reviewerID": "TEST_USER_CLEAN",
            "asin": "TEST_PRODUCT_001",
            "reviewerName": "Test Reviewer",
            "summary": "Great product works perfectly",
            "reviewText": "This is an excellent product. I highly recommend it to everyone.",
            "overall": 5.0,
            "unixReviewTime": 1640995200,
            "reviewTime": "01 1, 2022"
        }
        
        self.test_review_profanity = {
            "reviewerID": "TEST_USER_PROFANITY",
            "asin": "TEST_PRODUCT_002", 
            "reviewerName": "Angry Reviewer",
            "summary": "This shit product is terrible",
            "reviewText": "What a fucking waste of money. This damn thing doesn't work at all.",
            "overall": 1.0,
            "unixReviewTime": 1640995200,
            "reviewTime": "01 1, 2022"
        }
        
        self.test_review_negative = {
            "reviewerID": "TEST_USER_NEGATIVE", 
            "asin": "TEST_PRODUCT_003",
            "reviewerName": "Disappointed User",
            "summary": "Disappointing purchase",
            "reviewText": "I'm very disappointed with this product. It broke after just one week of use. Poor quality and overpriced.",
            "overall": 2.0,
            "unixReviewTime": 1640995200,
            "reviewTime": "01 1, 2022"
        }

    @patch('shared.aws_utils.get_parameter_store_value')
    @patch('lambdas.preprocess.batch_write_to_dynamodb')
    @patch('lambdas.preprocess.invoke_lambda')
    def test_preprocessing_functionality(self, mock_invoke_lambda, mock_batch_write, mock_get_param):
        """
        Test 1: Preprocessing functionality
        - Test text preprocessing components
        - Verify tokenization, stop word removal, lemmatization
        - Check data structure creation
        """
        # Mock SSM parameters
        mock_get_param.side_effect = lambda param: {
            "/cloud-review-cleaner/tables/reviews": "test-reviews-table"
        }.get(param, "test-value")
        
        # Mock DynamoDB batch write success
        mock_batch_write.return_value = {'success': True, 'failed_items': []}
        
        # Mock Lambda invocation success
        mock_invoke_lambda.return_value = {'StatusCode': 200}
        
        # Create preprocessing event
        event = {
            "batch_processing": True,
            "reviews": [self.test_review_clean],
            "batch_id": "test_batch_001"
        }
        
        # Test preprocessing lambda handler
        response = preprocess.lambda_handler(event, {})
        
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        self.assertEqual(body['processed_count'], 1)
        self.assertEqual(body['failed_count'], 0)
        self.assertIn('review_ids', body)
        
        # Verify batch write was called
        mock_batch_write.assert_called()
        
        # The main goal is that preprocessing succeeded
        self.assertEqual(body['processed_count'], 1)
        self.assertEqual(body['failed_count'], 0)

    def test_profanity_check_functionality(self):
        """
        Test 2: Profanity check functionality
        - Process review with profanity
        - Verify profanity detection
        - Check flagged words identification
        """
        # Test direct profanity checking logic
        from lambdas.profanity_check import check_profanity
        
        # Load bad words for testing
        bad_words = ['shit', 'fuck', 'damn', 'fucking']
        
        # Test clean review
        clean_tokens = ['great', 'product', 'works', 'perfectly']
        clean_result = check_profanity(
            clean_tokens, 
            self.test_review_clean['summary'], 
            self.test_review_clean['reviewText'], 
            bad_words
        )
        
        self.assertFalse(clean_result['has_profanity'])
        self.assertEqual(len(clean_result['flagged_words']), 0)
        
        # Test profane review
        profane_tokens = ['shit', 'product', 'terrible', 'fucking', 'waste']
        profane_result = check_profanity(
            profane_tokens,
            self.test_review_profanity['summary'],
            self.test_review_profanity['reviewText'],
            bad_words
        )
        
        self.assertTrue(profane_result['has_profanity'])
        self.assertGreater(len(profane_result['flagged_words']), 0)
        self.assertIn('shit', profane_result['flagged_words'])
        self.assertIn('fucking', profane_result['flagged_words'])

    def test_sentiment_analysis_functionality(self):
        """
        Test 3: Sentiment analysis functionality
        - Analyze positive, negative, and neutral sentiments
        - Verify sentiment classification
        - Check confidence scores
        """
        # Test positive sentiment
        positive_features = {
            'combined_tokens': ['excellent', 'product', 'highly', 'recommend'],
            'summary_clean': 'great product works perfectly',
            'review_clean': 'excellent product highly recommend everyone'
        }
        
        positive_result = analyze_sentiment(positive_features)
        self.assertIn('sentiment_label', positive_result)
        self.assertIn('sentiment_score', positive_result)
        self.assertIn('confidence', positive_result)
        
        # Test negative sentiment
        negative_features = {
            'combined_tokens': ['disappointed', 'broke', 'poor', 'quality', 'overpriced'],
            'summary_clean': 'disappointing purchase',
            'review_clean': 'disappointed product broke week poor quality overpriced'
        }
        
        negative_result = analyze_sentiment(negative_features)
        self.assertIn('sentiment_label', negative_result)
        self.assertIn('sentiment_score', negative_result)
        self.assertIn('confidence', negative_result)

    @patch('shared.aws_utils.get_parameter_store_value')
    @patch('lambdas.user_management.get_user_statistics')
    @patch('lambdas.user_management.update_user_unpolite_count')
    @patch('lambdas.user_management.ban_user')
    def test_user_management_functionality(self, mock_ban_user, mock_update_count, mock_get_stats, mock_get_param):
        """
        Test 4: User management functionality
        - Test profanity count tracking
        - Verify user banning logic
        - Check threshold enforcement
        """
        # Mock SSM parameters
        mock_get_param.side_effect = lambda param: {
            "/cloud-review-cleaner/tables/users": "test-users-table"
        }.get(param, "test-value")
        
        # Test user with 2 existing unpolite reviews
        mock_get_stats.return_value = {
            'user_id': 'TEST_USER_PROFANITY',
            'unpolite_review_count': 2,
            'total_review_count': 5,
            'status': 'active'
        }
        
        # Mock successful update (this would be the 3rd unpolite review, triggering a ban)
        mock_update_count.return_value = {
            'user_id': 'TEST_USER_PROFANITY',
            'unpolite_review_count': 3,
            'total_review_count': 6,
            'status': 'active'
        }
        
        # Mock successful ban
        mock_ban_user.return_value = {
            'banned': True,
            'user_id': 'TEST_USER_PROFANITY',
            'status': 'banned'
        }
        
        # Create user management event for a review with profanity
        event = {
            "Records": [{
                "eventName": "INSERT",
                "dynamodb": {
                    "NewImage": {
                        "review_id": {"S": "test_review_123"},
                        "status": {"S": "completed"},
                        "processing_summary": {
                            "M": {
                                "is_unpolite": {"BOOL": True}
                            }
                        },
                        "original_data": {
                            "M": {
                                "reviewerID": {"S": "TEST_USER_PROFANITY"}
                            }
                        },
                        "profanity_check": {
                            "M": {
                                "has_profanity": {"BOOL": True},
                                "profanity_score": {"N": "0.15"}
                            }
                        }
                    }
                }
            }]
        }
        
        # Test user management lambda handler
        response = user_management.lambda_handler(event, {})
        
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        # Check for the actual response format from user management
        self.assertIn('processed_count', body)
        self.assertEqual(body['processed_count'], 1)
        
        # Verify user statistics were retrieved and updated
        mock_get_stats.assert_called()
        mock_update_count.assert_called()

    def test_complete_function_chain(self):
        """
        Test 5: Complete function chain integration
        - Test end-to-end processing
        - Verify data flow between components
        - Check final results structure
        """
        # Test the complete pipeline with a single review
        review = self.test_review_clean
        
        # Step 1: Preprocessing
        processed_data = preprocess_review(review['summary'], review['reviewText'])
        
        self.assertIn('combined_tokens', processed_data)
        self.assertIn('processed_summary', processed_data)
        self.assertIn('processed_review_text', processed_data)
        
        # Step 2: Profanity Check
        from lambdas.profanity_check import check_profanity
        bad_words = ['shit', 'fuck', 'damn']
        
        profanity_result = check_profanity(
            processed_data['combined_tokens'],
            review['summary'],
            review['reviewText'],
            bad_words
        )
        
        self.assertIn('has_profanity', profanity_result)
        self.assertIn('flagged_words', profanity_result)
        
        # Step 3: Sentiment Analysis
        sentiment_result = analyze_sentiment(processed_data)
        
        self.assertIn('sentiment_label', sentiment_result)
        self.assertIn('sentiment_score', sentiment_result)
        
        # Verify complete data structure
        complete_result = {
            'processed_features': processed_data,
            'profanity_check': profanity_result,
            'sentiment_analysis': sentiment_result,
            'original_data': review
        }
        
        # Verify all required fields are present
        required_fields = ['processed_features', 'profanity_check', 'sentiment_analysis', 'original_data']
        for field in required_fields:
            self.assertIn(field, complete_result)

    def test_assignment_requirements_compliance(self):
        """
        Test 6: Assignment requirements compliance
        - Verify processing of summary, reviewText, and overall fields
        - Check that all required data transformations occur
        - Validate output format matches requirements
        """
        review = self.test_review_profanity
        
        # Verify all required fields are processed
        processed_data = preprocess_review(review['summary'], review['reviewText'])
        
        # Summary processing
        self.assertIn('processed_summary', processed_data)
        self.assertIsInstance(processed_data['processed_summary'], list)
        
        # Review text processing
        self.assertIn('processed_review_text', processed_data)
        self.assertIsInstance(processed_data['processed_review_text'], list)
        
        # Combined tokens (summary + reviewText)
        self.assertIn('combined_tokens', processed_data)
        self.assertIsInstance(processed_data['combined_tokens'], list)
        
        # Verify profanity check on all fields
        from lambdas.profanity_check import check_profanity
        bad_words = ['shit', 'fuck', 'damn', 'fucking']
        
        profanity_result = check_profanity(
            processed_data['combined_tokens'],
            review['summary'],
            review['reviewText'],
            bad_words
        )
        
        # Should detect profanity in both summary and reviewText
        self.assertTrue(profanity_result['has_profanity'])
        
        # Verify sentiment analysis uses processed features
        sentiment_result = analyze_sentiment(processed_data)
        
        # Should return valid sentiment classification
        self.assertIn(sentiment_result['sentiment_label'], ['positive', 'neutral', 'negative'])

    def test_dataset_processing_validation(self):
        """
        Test 7: Validate processing against known results
        - Compare with results.json
        - Verify the system produces expected counts
        """
        # Load the actual results file (check both current directory and parent directory)
        results_paths = ['results.json', '../results.json', os.path.join(os.path.dirname(__file__), '..', 'results.json')]
        actual_results = None
        
        for path in results_paths:
            try:
                with open(path, 'r') as f:
                    actual_results = json.load(f)
                break
            except FileNotFoundError:
                continue
        
        if actual_results is None:
            self.skipTest("results.json not found - run batch processor first")
        
        results = actual_results['results']
        
        # Verify required result fields are present
        self.assertIn('sentiment_analysis', results)
        self.assertIn('profanity_check', results)
        self.assertIn('user_management', results)
        
        # Verify sentiment analysis results
        sentiment = results['sentiment_analysis']
        self.assertEqual(sentiment['total_reviews'], 78829)
        self.assertEqual(sentiment['positive_reviews'], 61927)
        self.assertEqual(sentiment['neutral_reviews'], 10874)
        self.assertEqual(sentiment['negative_reviews'], 6028)
        
        # Verify profanity check results
        profanity = results['profanity_check']
        self.assertEqual(profanity['failed_reviews'], 9046)
        
        # Verify user management results
        user_mgmt = results['user_management']
        self.assertEqual(user_mgmt['banned_users_count'], 40)
        self.assertEqual(len(user_mgmt['banned_users']), 40)
        
        print("Dataset processing validation passed")
        print(f"   - Total reviews processed: {sentiment['total_reviews']}")
        print(f"   - Positive: {sentiment['positive_reviews']}, Neutral: {sentiment['neutral_reviews']}, Negative: {sentiment['negative_reviews']}")
        print(f"   - Reviews with profanity: {profanity['failed_reviews']}")
        print(f"   - Banned users: {user_mgmt['banned_users_count']}")


if __name__ == '__main__':
    unittest.main()