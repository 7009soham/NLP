"""
Text Preprocessing Examples

This module demonstrates various text preprocessing techniques.
"""

import re
from typing import List

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, extra whitespace, etc.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def simple_tokenize(text: str) -> List[str]:
    """
    Simple word tokenization by splitting on whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    return text.split()


def remove_stopwords(tokens: List[str], stopwords: set) -> List[str]:
    """
    Remove common stopwords from token list.
    
    Args:
        tokens: List of word tokens
        stopwords: Set of stopwords to remove
        
    Returns:
        Filtered list of tokens
    """
    return [token for token in tokens if token not in stopwords]


# Example usage
if __name__ == "__main__":
    # Sample text
    sample_text = """
    Natural Language Processing (NLP) is a fascinating field! 
    Visit https://example.com for more info or email test@example.com.
    It involves 123 different techniques and algorithms.
    """
    
    print("Original text:")
    print(sample_text)
    
    print("\nCleaned text:")
    cleaned = clean_text(sample_text)
    print(cleaned)
    
    print("\nTokens:")
    tokens = simple_tokenize(cleaned)
    print(tokens)
    
    print("\nNumber of tokens:", len(tokens))
    
    # Common English stopwords (simplified list)
    common_stopwords = {'a', 'an', 'and', 'is', 'it', 'the', 'for', 'or', 'of', 'in'}
    
    print("\nTokens after stopword removal:")
    filtered_tokens = remove_stopwords(tokens, common_stopwords)
    print(filtered_tokens)
