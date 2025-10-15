"""
Basic Python Setup for NLP

This script demonstrates how to set up and use common NLP libraries.
"""

# Common NLP libraries
# Install with: pip install nltk spacy textblob

# Example 1: Using NLTK
try:
    import nltk
    # Download required data (run once)
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    print("✓ NLTK is installed")
except ImportError:
    print("✗ NLTK is not installed. Run: pip install nltk")

# Example 2: Using spaCy
try:
    import spacy
    # Load English model (run once): python -m spacy download en_core_web_sm
    print("✓ spaCy is installed")
except ImportError:
    print("✗ spaCy is not installed. Run: pip install spacy")

# Example 3: Using TextBlob
try:
    from textblob import TextBlob
    print("✓ TextBlob is installed")
except ImportError:
    print("✗ TextBlob is not installed. Run: pip install textblob")

# Example 4: Basic text processing
def basic_text_processing():
    """Demonstrate basic text operations"""
    
    text = "Natural Language Processing is amazing!"
    
    # Basic operations
    print("\nOriginal text:", text)
    print("Lowercase:", text.lower())
    print("Uppercase:", text.upper())
    print("Word count:", len(text.split()))
    print("Character count:", len(text))
    print("Words:", text.split())
    
if __name__ == "__main__":
    print("\n=== NLP Environment Check ===")
    basic_text_processing()
