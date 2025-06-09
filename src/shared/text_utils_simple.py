"""
Simplified text processing utilities without NLTK dependencies.
For Lambda functions to avoid regex dependency issues.
"""

import re
import string
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Simple stop words list (most common English stop words)
SIMPLE_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now'
}

# Simple lemmatization dictionary (common word mappings)
LEMMA_DICTIONARY = {
    # Plurals to singular
    'books': 'book', 'cars': 'car', 'dogs': 'dog', 'cats': 'cat', 'houses': 'house',
    'boxes': 'box', 'dishes': 'dish', 'watches': 'watch', 'glasses': 'glass',
    'products': 'product', 'items': 'item', 'services': 'service', 'features': 'feature',
    'options': 'option', 'reviews': 'review', 'customers': 'customer', 'orders': 'order',
    'prices': 'price', 'qualities': 'quality', 'companies': 'company', 'businesses': 'business',
    
    # Past tense to present
    'worked': 'work', 'played': 'play', 'walked': 'walk', 'talked': 'talk',
    'looked': 'look', 'asked': 'ask', 'helped': 'help', 'wanted': 'want',
    'liked': 'like', 'loved': 'love', 'hated': 'hate', 'needed': 'need',
    'used': 'use', 'tried': 'try', 'bought': 'buy', 'sold': 'sell',
    'received': 'receive', 'delivered': 'deliver', 'shipped': 'ship', 'ordered': 'order',
    
    # Gerunds to base form
    'working': 'work', 'playing': 'play', 'walking': 'walk', 'talking': 'talk',
    'looking': 'look', 'asking': 'ask', 'helping': 'help', 'wanting': 'want',
    'liking': 'like', 'loving': 'love', 'hating': 'hate', 'needing': 'need',
    'using': 'use', 'trying': 'try', 'buying': 'buy', 'selling': 'sell',
    'receiving': 'receive', 'delivering': 'deliver', 'shipping': 'ship', 'ordering': 'order',
    
    # Common adjective forms
    'better': 'good', 'best': 'good', 'worse': 'bad', 'worst': 'bad',
    'bigger': 'big', 'biggest': 'big', 'smaller': 'small', 'smallest': 'small',
    'faster': 'fast', 'fastest': 'fast', 'slower': 'slow', 'slowest': 'slow',
    'cheaper': 'cheap', 'cheapest': 'cheap', 'expensive': 'expensive',
    'easier': 'easy', 'easiest': 'easy', 'harder': 'hard', 'hardest': 'hard',
    
    # Common irregular forms
    'children': 'child', 'people': 'person', 'men': 'man', 'women': 'woman',
    'feet': 'foot', 'teeth': 'tooth', 'mice': 'mouse', 'geese': 'goose'
}

def simple_lemmatize(word: str) -> str:
    """
    Simple rule-based lemmatization without external dependencies.
    
    Args:
        word: Input word to lemmatize
        
    Returns:
        Lemmatized word
    """
    if not word:
        return word
    
    word_lower = word.lower()
    
    # Check dictionary first
    if word_lower in LEMMA_DICTIONARY:
        return LEMMA_DICTIONARY[word_lower]
    
    # Apply simple rules for common suffixes
    # Handle plural nouns
    if word_lower.endswith('ies') and len(word_lower) > 4:
        return word_lower[:-3] + 'y'
    elif word_lower.endswith('es') and len(word_lower) > 3:
        if word_lower.endswith(('ches', 'shes', 'xes', 'zes')):
            return word_lower[:-2]
        elif word_lower.endswith('oes') and len(word_lower) > 4:
            return word_lower[:-2]
        else:
            return word_lower[:-1]
    elif word_lower.endswith('s') and len(word_lower) > 3:
        # Don't remove 's' from words that naturally end in 's'
        if not word_lower.endswith(('ss', 'us', 'is')):
            return word_lower[:-1]
    
    # Handle past tense verbs
    if word_lower.endswith('ed') and len(word_lower) > 3:
        if word_lower.endswith('ied'):
            return word_lower[:-3] + 'y'
        elif word_lower[-3] == word_lower[-4]:  # doubled consonant
            return word_lower[:-3]
        else:
            return word_lower[:-2]
    
    # Handle gerunds and present participles
    if word_lower.endswith('ing') and len(word_lower) > 4:
        if word_lower[-4] == word_lower[-5]:  # doubled consonant
            return word_lower[:-4]
        else:
            return word_lower[:-3]
    
    # Handle comparative and superlative adjectives
    if word_lower.endswith('er') and len(word_lower) > 3:
        if word_lower[-3] == word_lower[-4]:  # doubled consonant
            return word_lower[:-3]
        else:
            return word_lower[:-2]
    elif word_lower.endswith('est') and len(word_lower) > 4:
        if word_lower[-4] == word_lower[-5]:  # doubled consonant
            return word_lower[:-4]
        else:
            return word_lower[:-3]
    
    return word_lower

def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenization using regex (built-in re module).
    Handles very long texts by truncating if necessary.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Handle very long texts (DynamoDB item limit is 400KB)
    # Truncate text to reasonable length to avoid storage issues
    MAX_TEXT_LENGTH = 50000  # 50KB per text field is reasonable
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning(f"Text too long ({len(text)} chars), truncating to {MAX_TEXT_LENGTH}")
        text = text[:MAX_TEXT_LENGTH]
        
    # Convert to lowercase and clean newlines/special chars
    text = text.lower()
    
    # Replace problematic characters that might cause storage issues
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    
    # Replace punctuation with spaces and split
    # Keep only alphabetic characters
    tokens = re.findall(r'[a-zA-Z]+', text)
    
    # Limit total number of tokens to prevent oversized records
    MAX_TOKENS = 1000
    if len(tokens) > MAX_TOKENS:
        logger.warning(f"Too many tokens ({len(tokens)}), limiting to {MAX_TOKENS}")
        tokens = tokens[:MAX_TOKENS]
    
    return tokens

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stop words from token list.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of tokens without stop words
    """
    return [token for token in tokens if token not in SIMPLE_STOPWORDS and len(token) > 2]

def simple_preprocess_text(text: str) -> List[str]:
    """
    Simple text preprocessing pipeline with lemmatization.
    
    Args:
        text: Input text
        
    Returns:
        List of processed tokens
    """
    if not text:
        return []
        
    # Tokenize
    tokens = simple_tokenize(text)
    
    # Remove stop words
    tokens = remove_stopwords(tokens)
    
    # Apply lemmatization
    tokens = [simple_lemmatize(token) for token in tokens]
    
    # Remove empty tokens and duplicates while preserving order
    seen = set()
    processed_tokens = []
    for token in tokens:
        if token.strip() and token not in seen:
            processed_tokens.append(token)
            seen.add(token)
    
    return processed_tokens

def extract_review_features(review_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and preprocess features from review data.
    Simplified version without NLTK dependencies, now with lemmatization.
    
    Args:
        review_data: Dictionary containing review information
        
    Returns:
        Dictionary containing processed features
    """
    try:
        # Extract text fields
        summary = review_data.get('summary', '')
        review_text = review_data.get('reviewText', '')
        
        # Handle None values
        if summary is None:
            summary = ''
        if review_text is None:
            review_text = ''
            
        # Convert to strings if needed
        summary = str(summary).strip()
        review_text = str(review_text).strip()
        
        logger.info(f"Processing review with summary length: {len(summary)}, review text length: {len(review_text)}")
        
        # Process text with lemmatization
        processed_summary = simple_preprocess_text(summary)
        processed_review_text = simple_preprocess_text(review_text)
        
        # Combine all tokens
        combined_tokens = processed_summary + processed_review_text
        
        # Create features dictionary
        features = {
            'processed_summary': processed_summary,
            'processed_review_text': processed_review_text,
            'combined_tokens': combined_tokens,
            'original_summary': summary,
            'original_review_text': review_text
        }
        
        # Add metadata
        features['total_tokens'] = len(features['combined_tokens'])
        features['summary_tokens'] = len(features['processed_summary'])
        features['review_text_tokens'] = len(features['processed_review_text'])
        
        # For empty content, return empty features instead of raising error
        # This allows the system to handle empty reviews gracefully
        if features['total_tokens'] == 0:
            logger.warning("Review contains no meaningful content after preprocessing, returning empty features")
        
        logger.info(f"Extracted features: {features['total_tokens']} total tokens (after lemmatization)")
        return features
        
    except Exception as e:
        logger.error(f"Error in extract_review_features: {e}")
        raise

def preprocess_review(summary: str, review_text: str, **kwargs) -> Dict[str, Any]:
    """
    Preprocess a review's text content with tokenization, stop word removal, and lemmatization.
    
    Args:
        summary: Review summary text
        review_text: Main review text
        **kwargs: Additional parameters (for compatibility)
        
    Returns:
        Dictionary containing preprocessed features
    """
    review_data = {
        'summary': summary,
        'reviewText': review_text
    }
    
    return extract_review_features(review_data) 