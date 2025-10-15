"""
Word Embedding Utilities

Helper functions for working with word embeddings.
"""

import numpy as np
from typing import Dict, List, Tuple


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def most_similar(word: str, embeddings: Dict[str, np.ndarray], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Find most similar words to a given word.
    
    Args:
        word: Target word
        embeddings: Dictionary mapping words to embedding vectors
        top_n: Number of similar words to return
        
    Returns:
        List of (word, similarity_score) tuples
    """
    if word not in embeddings:
        return []
    
    word_vec = embeddings[word]
    similarities = []
    
    for other_word, other_vec in embeddings.items():
        if other_word != word:
            sim = cosine_similarity(word_vec, other_vec)
            similarities.append((other_word, sim))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]


def word_analogy(word_a: str, word_b: str, word_c: str, 
                 embeddings: Dict[str, np.ndarray], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Solve word analogies: a is to b as c is to ?
    
    Example: king is to queen as man is to woman
    
    Args:
        word_a: First word in analogy
        word_b: Second word in analogy
        word_c: Third word in analogy
        embeddings: Dictionary mapping words to embedding vectors
        top_n: Number of results to return
        
    Returns:
        List of (word, similarity_score) tuples
    """
    if not all(w in embeddings for w in [word_a, word_b, word_c]):
        return []
    
    # Calculate: b - a + c
    target_vec = embeddings[word_b] - embeddings[word_a] + embeddings[word_c]
    
    similarities = []
    for word, vec in embeddings.items():
        if word not in [word_a, word_b, word_c]:
            sim = cosine_similarity(target_vec, vec)
            similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def load_pretrained_embeddings(filepath: str, limit: int = None) -> Dict[str, np.ndarray]:
    """
    Load pre-trained word embeddings from file.
    
    Expected format: word dim1 dim2 dim3 ... (space-separated)
    
    Args:
        filepath: Path to embeddings file
        limit: Maximum number of words to load (None for all)
        
    Returns:
        Dictionary mapping words to vectors
    """
    embeddings = {}
    count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
                
            parts = line.strip().split()
            if len(parts) > 2:
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embeddings[word] = vector
                count += 1
    
    return embeddings


# Example usage
if __name__ == "__main__":
    # Create simple example embeddings
    # In practice, these would be learned from data
    embeddings = {
        'king': np.array([0.5, 0.5, 0.1]),
        'queen': np.array([0.5, 0.5, -0.1]),
        'man': np.array([0.5, -0.5, 0.1]),
        'woman': np.array([0.5, -0.5, -0.1]),
        'prince': np.array([0.4, 0.4, 0.2]),
        'princess': np.array([0.4, 0.4, -0.2]),
    }
    
    print("Example Word Embeddings\n")
    
    print("1. Cosine Similarity:")
    sim = cosine_similarity(embeddings['king'], embeddings['queen'])
    print(f"   king <-> queen: {sim:.3f}")
    sim = cosine_similarity(embeddings['king'], embeddings['man'])
    print(f"   king <-> man: {sim:.3f}")
    
    print("\n2. Most Similar Words:")
    similar = most_similar('king', embeddings, top_n=3)
    print("   Most similar to 'king':")
    for word, score in similar:
        print(f"   - {word}: {score:.3f}")
    
    print("\n3. Word Analogies:")
    print("   king is to queen as man is to ?")
    results = word_analogy('king', 'queen', 'man', embeddings, top_n=3)
    for word, score in results:
        print(f"   - {word}: {score:.3f}")
