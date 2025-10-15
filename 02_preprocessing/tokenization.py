"""
Tokenization Examples

Different approaches to splitting text into tokens.
"""

import re
from typing import List

def word_tokenize_simple(text: str) -> List[str]:
    """Simple word tokenization using split()"""
    return text.split()


def word_tokenize_regex(text: str) -> List[str]:
    """Word tokenization using regular expressions"""
    # Match words (sequences of alphanumeric characters)
    return re.findall(r'\b\w+\b', text)


def sentence_tokenize_simple(text: str) -> List[str]:
    """Simple sentence tokenization"""
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+', text)
    # Remove empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


def char_tokenize(text: str) -> List[str]:
    """Character-level tokenization"""
    return list(text)


def ngrams(tokens: List[str], n: int) -> List[tuple]:
    """
    Generate n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: Size of n-grams
        
    Returns:
        List of n-gram tuples
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# Example usage
if __name__ == "__main__":
    sample_text = "Hello world! This is NLP. Let's learn tokenization."
    
    print("Original text:")
    print(sample_text)
    
    print("\n1. Simple word tokenization:")
    words_simple = word_tokenize_simple(sample_text)
    print(words_simple)
    
    print("\n2. Regex word tokenization:")
    words_regex = word_tokenize_regex(sample_text)
    print(words_regex)
    
    print("\n3. Sentence tokenization:")
    sentences = sentence_tokenize_simple(sample_text)
    print(sentences)
    
    print("\n4. Character tokenization (first 20 chars):")
    chars = char_tokenize(sample_text)
    print(chars[:20])
    
    print("\n5. Bigrams (2-grams):")
    bigrams = ngrams(words_regex, 2)
    print(bigrams)
    
    print("\n6. Trigrams (3-grams):")
    trigrams = ngrams(words_regex, 3)
    print(trigrams)
