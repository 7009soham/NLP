"""
Bag of Words (BoW) Implementation

Demonstrates how to convert text to numerical features using BoW and TF-IDF.
"""

from typing import List, Dict
from collections import Counter
import math


class BagOfWords:
    """Simple Bag of Words implementation"""
    
    def __init__(self):
        self.vocabulary = {}
        
    def fit(self, documents: List[str]):
        """Build vocabulary from documents"""
        words = set()
        for doc in documents:
            words.update(doc.lower().split())
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(words))}
        
    def transform(self, document: str) -> List[int]:
        """Transform document to BoW vector"""
        words = document.lower().split()
        vector = [0] * len(self.vocabulary)
        
        for word in words:
            if word in self.vocabulary:
                vector[self.vocabulary[word]] += 1
                
        return vector
    
    def fit_transform(self, documents: List[str]) -> List[List[int]]:
        """Fit and transform documents"""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


class TfidfVectorizer:
    """Simple TF-IDF implementation"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        
    def fit(self, documents: List[str]):
        """Build vocabulary and calculate IDF"""
        # Build vocabulary
        words = set()
        for doc in documents:
            words.update(doc.lower().split())
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(words))}
        
        # Calculate IDF
        n_docs = len(documents)
        word_doc_count = Counter()
        
        for doc in documents:
            unique_words = set(doc.lower().split())
            word_doc_count.update(unique_words)
            
        for word in self.vocabulary:
            self.idf[word] = math.log(n_docs / (1 + word_doc_count[word]))
    
    def transform(self, document: str) -> List[float]:
        """Transform document to TF-IDF vector"""
        words = document.lower().split()
        word_count = Counter(words)
        total_words = len(words)
        
        vector = [0.0] * len(self.vocabulary)
        
        for word, count in word_count.items():
            if word in self.vocabulary:
                tf = count / total_words
                tfidf = tf * self.idf[word]
                vector[self.vocabulary[word]] = tfidf
                
        return vector


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "I love machine learning",
        "Machine learning is amazing",
        "I love deep learning",
        "Deep learning and machine learning are related"
    ]
    
    print("Sample documents:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")
    
    print("\n=== Bag of Words ===")
    bow = BagOfWords()
    bow_vectors = bow.fit_transform(documents)
    
    print("\nVocabulary:", sorted(bow.vocabulary.keys()))
    print("\nBoW vectors:")
    for i, vec in enumerate(bow_vectors, 1):
        print(f"Doc {i}: {vec}")
    
    print("\n=== TF-IDF ===")
    tfidf = TfidfVectorizer()
    tfidf.fit(documents)
    
    print("\nTF-IDF vectors:")
    for i, doc in enumerate(documents, 1):
        vec = tfidf.transform(doc)
        print(f"Doc {i}: {[f'{v:.3f}' for v in vec]}")
