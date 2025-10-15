# Word Embeddings Overview

## What are Word Embeddings?

Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points.

## Why Word Embeddings?

Traditional representations like one-hot encoding have limitations:
- **High dimensionality**: Vocabulary size can be 10,000+ words
- **No semantic information**: "king" and "queen" are as different as "king" and "pizza"
- **Sparse vectors**: Mostly zeros, inefficient

Word embeddings solve these problems:
- **Low dimensionality**: Typically 50-300 dimensions
- **Semantic similarity**: Related words have similar vectors
- **Dense vectors**: All values are meaningful

## Properties of Word Embeddings

1. **Semantic Similarity**: Similar words are close in vector space
   - cos_sim("king", "queen") > cos_sim("king", "apple")

2. **Analogies**: Vector arithmetic captures relationships
   - king - man + woman ≈ queen
   - Paris - France + Italy ≈ Rome

3. **Transferability**: Pre-trained embeddings work across tasks

## Popular Word Embedding Methods

### Word2Vec (2013)
- Two architectures: CBOW and Skip-gram
- Efficient training on large corpora
- Captures semantic and syntactic patterns

### GloVe (2014)
- Global Vectors for Word Representation
- Uses word co-occurrence statistics
- Combines benefits of matrix factorization and local context

### FastText (2016)
- Extension of Word2Vec
- Uses subword information (character n-grams)
- Better handling of rare words and morphology

## Applications

- Text classification
- Sentiment analysis
- Named entity recognition
- Machine translation
- Document similarity
- Information retrieval

## Getting Started

Start with pre-trained embeddings, then learn to train your own!
