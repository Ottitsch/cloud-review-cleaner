"""
Sentiment analysis utilities for the Review Cleaner application.

This module provides sentiment analysis functionality using a simple
rule-based approach without external dependencies like TextBlob.
"""

import logging
from typing import Dict, Any, List, Tuple
from enum import Enum

from .constants import (
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
    LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Enumeration for sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentAnalyzer:
    """Handles sentiment analysis operations for review text."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.positive_threshold = SENTIMENT_POSITIVE_THRESHOLD
        self.negative_threshold = SENTIMENT_NEGATIVE_THRESHOLD
        
        # Positive and negative indicator words with weights
        self.positive_words = {
            'excellent': 2.0, 'amazing': 2.0, 'wonderful': 2.0, 'fantastic': 2.0, 
            'great': 1.5, 'awesome': 2.0, 'love': 1.5, 'perfect': 2.0, 
            'best': 1.5, 'outstanding': 2.0, 'superb': 2.0, 'brilliant': 2.0,
            'recommend': 1.5, 'impressed': 1.5, 'satisfied': 1.0, 'pleased': 1.0, 
            'happy': 1.0, 'delighted': 1.5, 'good': 1.0, 'nice': 1.0, 
            'fine': 0.5, 'well': 0.5, 'better': 1.0, 'improved': 1.0, 
            'quality': 1.0, 'enjoy': 1.0, 'beautiful': 1.5, 'smooth': 1.0,
            'fast': 1.0, 'efficient': 1.0, 'helpful': 1.0, 'friendly': 1.0,
            'comfortable': 1.0, 'easy': 1.0, 'reliable': 1.0, 'worth': 1.0
        }
        
        self.negative_words = {
            'terrible': -2.0, 'awful': -2.0, 'horrible': -2.0, 'worst': -2.0, 
            'bad': -1.5, 'poor': -1.5, 'disappointing': -1.5, 'hate': -2.0, 
            'dislike': -1.0, 'useless': -2.0, 'waste': -1.5, 'defective': -2.0, 
            'broken': -2.0, 'faulty': -2.0, 'problem': -1.0, 'issue': -1.0, 
            'error': -1.5, 'fail': -1.5, 'failed': -1.5, 'wrong': -1.0, 
            'annoying': -1.0, 'frustrated': -1.0, 'angry': -1.5, 'upset': -1.0, 
            'dissatisfied': -1.5, 'regret': -1.5, 'return': -1.0, 'refund': -1.5,
            'slow': -1.0, 'expensive': -1.0, 'cheap': -1.0, 'difficult': -1.0,
            'complicated': -1.0, 'confusing': -1.0, 'unreliable': -1.5
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'quite': 1.2,
            'totally': 1.8, 'absolutely': 2.0, 'completely': 1.8, 'highly': 1.5,
            'so': 1.3, 'too': 1.2, 'super': 1.5, 'incredibly': 2.0
        }
        
        # Diminishers
        self.diminishers = {
            'barely': 0.5, 'hardly': 0.5, 'scarcely': 0.5, 'somewhat': 0.7,
            'rather': 0.8, 'fairly': 0.8, 'pretty': 0.9, 'kind': 0.7, 'sort': 0.7
        }
        
        logger.info("Simple sentiment analyzer initialized")
    
    def analyze_simple_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using a simple rule-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment score and details
        """
        try:
            # Convert to lowercase and split into words
            words = text.lower().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
            
            total_score = 0.0
            word_count = 0
            positive_matches = []
            negative_matches = []
            
            i = 0
            while i < len(words):
                word = words[i]
                modifier = 1.0
                
                # Check for intensifiers/diminishers before the current word
                if i > 0 and words[i-1] in self.intensifiers:
                    modifier = self.intensifiers[words[i-1]]
                elif i > 0 and words[i-1] in self.diminishers:
                    modifier = self.diminishers[words[i-1]]
                
                # Check for positive words
                if word in self.positive_words:
                    score = self.positive_words[word] * modifier
                    total_score += score
                    word_count += 1
                    positive_matches.append((word, score))
                
                # Check for negative words
                elif word in self.negative_words:
                    score = self.negative_words[word] * modifier
                    total_score += score
                    word_count += 1
                    negative_matches.append((word, score))
                
                i += 1
            
            # Normalize score
            if word_count > 0:
                average_score = total_score / word_count
            else:
                average_score = 0.0
            
            # Scale to [-1, 1] range
            final_score = max(-1.0, min(1.0, average_score / 2.0))
            
            return {
                'sentiment_score': final_score,
                'raw_score': total_score,
                'word_count': word_count,
                'positive_matches': positive_matches,
                'negative_matches': negative_matches,
                'total_words': len(words)
            }
            
        except Exception as e:
            logger.error(f"Simple sentiment analysis failed: {e}")
            return {
                'sentiment_score': 0.0,
                'raw_score': 0.0,
                'word_count': 0,
                'positive_matches': [],
                'negative_matches': [],
                'total_words': 0
            }
    
    def analyze_rule_based_sentiment(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment using rule-based approach on preprocessed tokens.
        
        Args:
            tokens: List of preprocessed tokens
            
        Returns:
            Dictionary with sentiment scores and indicators
        """
        try:
            positive_count = 0
            negative_count = 0
            total_tokens = len(tokens)
            positive_score = 0.0
            negative_score = 0.0
            
            # Count positive and negative words with weights
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.positive_words:
                    positive_count += 1
                    positive_score += self.positive_words[token_lower]
                elif token_lower in self.negative_words:
                    negative_count += 1
                    negative_score += abs(self.negative_words[token_lower])  # Make positive for counting
            
            # Calculate ratios and scores
            positive_ratio = positive_count / total_tokens if total_tokens > 0 else 0
            negative_ratio = negative_count / total_tokens if total_tokens > 0 else 0
            
            # Calculate overall sentiment score
            if positive_count + negative_count > 0:
                sentiment_score = (positive_score - negative_score) / (positive_count + negative_count)
            else:
                sentiment_score = 0.0
            
            # Normalize to [-1, 1]
            sentiment_score = max(-1.0, min(1.0, sentiment_score / 2.0))
            
            return {
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'sentiment_score': sentiment_score,
                'total_tokens': total_tokens,
                'positive_score': positive_score,
                'negative_score': negative_score
            }
            
        except Exception as e:
            logger.error(f"Rule-based sentiment analysis failed: {e}")
            return {
                'positive_count': 0,
                'negative_count': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'sentiment_score': 0.0,
                'total_tokens': 0,
                'positive_score': 0.0,
                'negative_score': 0.0
            }
    
    def classify_sentiment(self, sentiment_score: float) -> SentimentLabel:
        """
        Classify sentiment based on score.
        
        Args:
            sentiment_score: Numerical sentiment score (-1 to 1)
            
        Returns:
            Sentiment label (positive, negative, or neutral)
        """
        if sentiment_score > self.positive_threshold:
            return SentimentLabel.POSITIVE
        elif sentiment_score < self.negative_threshold:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def analyze_review_sentiment(self, review_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete sentiment analysis on review features.
        
        Args:
            review_features: Review features with keys like 'combined_tokens', 'summary_text', 'review_text'
            
        Returns:
            Dictionary with comprehensive sentiment analysis results
        """
        required_features = ['combined_tokens', 'summary_text', 'review_text']
        
        # Use available features, don't fail if some are missing
        available_features = {k: v for k, v in review_features.items() if k in required_features and v is not None}
        
        try:
            # Combine text for analysis
            combined_text = ""
            if 'summary_text' in available_features:
                combined_text += str(available_features['summary_text']) + " "
            if 'review_text' in available_features:
                combined_text += str(available_features['review_text'])
            
            # Analyze with simple sentiment analysis
            simple_scores = self.analyze_simple_sentiment(combined_text.strip())
            
            # Analyze with rule-based approach on tokens if available
            if 'combined_tokens' in available_features and available_features['combined_tokens']:
                rule_based_scores = self.analyze_rule_based_sentiment(available_features['combined_tokens'])
            else:
                # Fallback to word-based analysis
                words = combined_text.lower().split()
                rule_based_scores = self.analyze_rule_based_sentiment(words)
            
            # Combine scores (give more weight to rule-based on tokens)
            final_sentiment_score = (
                simple_scores['sentiment_score'] * 0.4 +
                rule_based_scores['sentiment_score'] * 0.6
            )
            
            # Ensure score is within bounds
            final_sentiment_score = max(-1.0, min(1.0, final_sentiment_score))
            
            # Classify sentiment
            sentiment_label = self.classify_sentiment(final_sentiment_score)
            
            # Calculate confidence based on score magnitude
            confidence = min(abs(final_sentiment_score) * 2, 1.0)
            
            return {
                'sentiment_label': sentiment_label.value,
                'sentiment_score': final_sentiment_score,
                'confidence': confidence,
                'detailed_scores': {
                    'positive': max(0, final_sentiment_score),
                    'negative': max(0, -final_sentiment_score),
                    'neutral': 1.0 - confidence
                },
                'analysis_details': {
                    'simple_analysis': simple_scores,
                    'rule_based_analysis': rule_based_scores,
                    'combined_text_length': len(combined_text),
                    'available_features': list(available_features.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Complete sentiment analysis failed: {e}")
            return {
                'sentiment_label': SentimentLabel.NEUTRAL.value,
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'detailed_scores': {
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0
                },
                'analysis_details': {
                    'error': str(e)
                }
            }
    
    def analyze_rating_sentiment_consistency(self, sentiment_analysis: Dict[str, Any], 
                                           overall_rating: float) -> Dict[str, Any]:
        """
        Check consistency between sentiment analysis and numerical rating.
        
        Args:
            sentiment_analysis: Results from sentiment analysis
            overall_rating: Numerical rating (typically 1-5)
            
        Returns:
            Dictionary with consistency analysis results
        """
        try:
            sentiment_score = sentiment_analysis.get('sentiment_score', 0.0)
            sentiment_label = sentiment_analysis.get('sentiment_label', 'neutral')
            
            # Convert rating to expected sentiment
            # Assuming 1-5 scale: 1-2 = negative, 3 = neutral, 4-5 = positive
            if overall_rating <= 2.0:
                expected_sentiment = 'negative'
                expected_score = -0.5
            elif overall_rating >= 4.0:
                expected_sentiment = 'positive'
                expected_score = 0.5
            else:
                expected_sentiment = 'neutral'
                expected_score = 0.0
            
            # Calculate discrepancy
            score_discrepancy = abs(sentiment_score - expected_score)
            label_match = sentiment_label == expected_sentiment
            
            # Determine if consistent (allowing some tolerance)
            is_consistent = score_discrepancy <= 0.5 and label_match
            
            # Create explanation
            if is_consistent:
                explanation = f"Rating ({overall_rating}) matches sentiment ({sentiment_label})"
            else:
                explanation = f"Rating ({overall_rating}) conflicts with sentiment ({sentiment_label})"
            
            return {
                'is_consistent': is_consistent,
                'discrepancy_score': score_discrepancy,
                'expected_sentiment': expected_sentiment,
                'actual_sentiment': sentiment_label,
                'rating': overall_rating,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"Rating consistency analysis failed: {e}")
            return {
                'is_consistent': True,  # Default to consistent to avoid blocking
                'discrepancy_score': 0.0,
                'expected_sentiment': 'neutral',
                'actual_sentiment': 'neutral',
                'rating': overall_rating,
                'explanation': f"Analysis failed: {e}"
            }


# Global analyzer instance
_analyzer = SentimentAnalyzer()


def analyze_sentiment(review_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for sentiment analysis.
    
    Args:
        review_features: Preprocessed review features
        
    Returns:
        Dictionary with sentiment analysis results
    """
    return _analyzer.analyze_review_sentiment(review_features)


def check_rating_consistency(sentiment_analysis: Dict[str, Any], 
                           overall_rating: float) -> Dict[str, Any]:
    """
    Check consistency between sentiment and rating.
    
    Args:
        sentiment_analysis: Results from analyze_sentiment
        overall_rating: Numerical rating
        
    Returns:
        Dictionary with consistency analysis results
    """
    return _analyzer.analyze_rating_sentiment_consistency(sentiment_analysis, overall_rating) 