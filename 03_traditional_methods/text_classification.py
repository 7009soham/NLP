"""
Text Classification with Traditional Methods

Examples of using traditional ML algorithms for text classification.
"""

from typing import List, Tuple
from collections import Counter
import math


def calculate_word_probabilities(documents: List[str], labels: List[int]) -> Tuple[dict, dict]:
    """
    Calculate word probabilities for Naive Bayes classifier.
    
    Args:
        documents: List of text documents
        labels: List of class labels (0 or 1 for binary classification)
        
    Returns:
        Tuple of (class_probs, word_probs)
    """
    # Count documents per class
    class_counts = Counter(labels)
    total_docs = len(labels)
    
    class_probs = {cls: count/total_docs for cls, count in class_counts.items()}
    
    # Count words per class
    word_counts = {}
    for doc, label in zip(documents, labels):
        if label not in word_counts:
            word_counts[label] = Counter()
        word_counts[label].update(doc.lower().split())
    
    # Calculate word probabilities with Laplace smoothing
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc.lower().split())
    
    vocab_size = len(vocabulary)
    word_probs = {}
    
    for cls in class_counts:
        word_probs[cls] = {}
        total_words = sum(word_counts[cls].values())
        
        for word in vocabulary:
            count = word_counts[cls][word]
            # Laplace smoothing
            word_probs[cls][word] = (count + 1) / (total_words + vocab_size)
    
    return class_probs, word_probs


def predict_naive_bayes(document: str, class_probs: dict, word_probs: dict) -> int:
    """
    Predict class for a document using Naive Bayes.
    
    Args:
        document: Text document to classify
        class_probs: Prior probabilities for each class
        word_probs: Word probabilities for each class
        
    Returns:
        Predicted class label
    """
    words = document.lower().split()
    scores = {}
    
    for cls in class_probs:
        # Start with log of prior probability
        score = math.log(class_probs[cls])
        
        # Add log probabilities of words
        for word in words:
            if word in word_probs[cls]:
                score += math.log(word_probs[cls][word])
        
        scores[cls] = score
    
    # Return class with highest score
    return max(scores, key=scores.get)


# Example usage
if __name__ == "__main__":
    # Sample training data (sentiment classification)
    train_documents = [
        "I love this movie it is amazing",
        "Great film wonderful acting",
        "Best movie ever so good",
        "Terrible movie waste of time",
        "Awful film do not watch",
        "Horrible acting very bad"
    ]
    
    # Labels: 1 = positive, 0 = negative
    train_labels = [1, 1, 1, 0, 0, 0]
    
    print("Training Naive Bayes classifier...")
    class_probs, word_probs = calculate_word_probabilities(train_documents, train_labels)
    
    print("\nClass probabilities:")
    for cls, prob in class_probs.items():
        label = "Positive" if cls == 1 else "Negative"
        print(f"  {label}: {prob:.2f}")
    
    # Test documents
    test_documents = [
        "I love this film",
        "Terrible waste of time",
        "Amazing movie",
        "Very bad"
    ]
    
    print("\nPredictions:")
    for doc in test_documents:
        prediction = predict_naive_bayes(doc, class_probs, word_probs)
        label = "Positive" if prediction == 1 else "Negative"
        print(f"  '{doc}' -> {label}")
