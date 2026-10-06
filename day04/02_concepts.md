# Day 04 — Concepts
## AI Mathematics: What You Must Know

---

## 1. Scalars, Vectors, Matrices

```
Scalar:  a single number              e.g., temperature = 0.7
Vector:  a list of numbers            e.g., embedding = [0.2, -0.5, 0.8, 0.1]
Matrix:  a 2D grid of numbers         e.g., batch of 100 embeddings (100×768)
Tensor:  N-dimensional array          e.g., image (H×W×C), attention (L×L)

IN AI:
  Scalar → temperature, learning rate, similarity score
  Vector → embedding (text→numbers), hidden state, token logits
  Matrix → attention weights, weight matrices in neural networks
  Tensor → image data, transformer activations
```

---

## 2. Vector Operations

```python
import numpy as np
import math

# Vectors
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Addition
c = a + b                          # [5, 7, 9]

# Scalar multiplication
scaled = 2.0 * a                   # [2, 4, 6]

# Dot product: captures alignment between vectors
dot = np.dot(a, b)                 # 1×4 + 2×5 + 3×6 = 32
# High dot product = vectors point in similar directions

# Magnitude (L2 norm): length of the vector
norm_a = np.linalg.norm(a)         # sqrt(1²+2²+3²) = 3.74
norm_a_manual = math.sqrt(sum(x**2 for x in a))

# Unit vector: normalize to length 1
unit_a = a / norm_a               # [0.267, 0.535, 0.802]

# Cosine similarity: angle between vectors
# = dot(a, b) / (|a| × |b|)
# = 1.0 means same direction (very similar)
# = 0.0 means perpendicular (unrelated)
# = -1.0 means opposite (antonyms)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # 0.974

# Euclidean distance: physical distance
euclidean = np.linalg.norm(a - b)  # 5.196
```

---

## 3. Why Cosine Similarity for Embeddings

The key insight:

```
"machine learning algorithms" (long document, 500 words)
"ML algorithms" (short document, 2 words)

Both are about the same topic.

Euclidean distance: LARGE (different magnitudes due to length)
Cosine similarity:  HIGH (same semantic direction)

Cosine similarity is SCALE INVARIANT.
It measures direction, not magnitude.
This is exactly what we want for semantic similarity.
```

---

## 4. Probability Fundamentals

```python
# Probability: likelihood of an event, between 0 and 1
P_spam = 0.3         # 30% of emails are spam

# Conditional probability: P(A | B) = "probability of A given B"
# P(spam | contains "FREE") = much higher than P(spam) alone

# Bayes' Theorem:
# P(spam | word) = P(word | spam) × P(spam) / P(word)
#
# In words:
# posterior = likelihood × prior / evidence
#
# AI Application: Naive Bayes classifier
# P(spam | email) ∝ P(email words | spam) × P(spam)

# Example in Python
def naive_bayes_classify(email_words: list[str],
                          spam_word_probs: dict[str, float],
                          ham_word_probs: dict[str, float],
                          prior_spam: float = 0.3) -> str:
    import math
    log_prob_spam = math.log(prior_spam)
    log_prob_ham = math.log(1 - prior_spam)
    
    for word in email_words:
        log_prob_spam += math.log(spam_word_probs.get(word, 1e-10))
        log_prob_ham  += math.log(ham_word_probs.get(word, 1e-10))
    
    return "spam" if log_prob_spam > log_prob_ham else "ham"
```

---

## 5. Statistics: Mean, Variance, Standard Deviation

```python
import numpy as np

scores = np.array([0.92, 0.78, 0.85, 0.91, 0.70, 0.88])

mean = scores.mean()                     # 0.84 — average
variance = scores.var()                  # how spread out
std = scores.std()                       # sqrt(variance) — same units as data

# Z-score: how many std devs from mean?
z_scores = (scores - mean) / std
# score of 0.92 → z = (0.92 - 0.84) / 0.07 = 1.14 std devs above mean
# Used in: StandardScaler, anomaly detection, normalization

# Why this matters for AI:
# - StandardScaler normalizes features to zero mean, unit variance
# - Neural networks train faster on normalized inputs
# - Evaluation: is a score of 0.80 good? Depends on mean and std!
```

---

## 6. Distance Metrics

```python
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])

# L1 (Manhattan) distance: sum of absolute differences
l1 = np.sum(np.abs(a - b))             # 2.0

# L2 (Euclidean) distance: square root of sum of squared differences
l2 = np.linalg.norm(a - b)             # 1.414

# Cosine distance: 1 - cosine_similarity
cos_dist = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # 1.0

# In AI:
# L2 distance: good for dense, low-dimensional features
# Cosine similarity: better for high-dimensional sparse features (text, embeddings)
# Inner product: fast approximation of cosine sim (if vectors normalized)
```

---

## 7. Matrix Operations

```python
# Weight matrix in a neural network
W = np.random.randn(768, 256)   # input_dim × output_dim

# Forward pass: matrix multiply embedding by weight matrix
embedding = np.random.randn(768)
hidden = W.T @ embedding         # shape: (256,)

# Attention score matrix (used in transformers)
# Q: (seq_len, d_k), K: (seq_len, d_k)
Q = np.random.randn(10, 64)  # 10 tokens, key_dim=64
K = np.random.randn(10, 64)
scores = Q @ K.T                  # shape: (10, 10) — each token attends to all others
d_k = 64
attention = np.exp(scores / np.sqrt(d_k))  # softmax numerator
attention /= attention.sum(axis=-1, keepdims=True)  # normalize
```
