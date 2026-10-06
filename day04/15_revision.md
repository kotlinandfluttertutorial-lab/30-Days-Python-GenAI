# Day 04 — Revision: AI Mathematics

---

## Formulas to Memorize

```
Dot product:           dot(a,b) = Σ aᵢ × bᵢ
Magnitude (L2 norm):   |a| = √(Σ aᵢ²)
Cosine similarity:     cos(a,b) = dot(a,b) / (|a| × |b|)
Euclidean distance:    d(a,b) = √(Σ (aᵢ - bᵢ)²)
Softmax:               p_i = exp(xᵢ) / Σ exp(xⱼ)
Z-score:               z = (x - μ) / σ
Attention:             scores = softmax(QK^T / √d_k) × V
```

## NumPy Quick Reference

```python
import numpy as np
# Dot product
np.dot(a, b)

# Magnitude
np.linalg.norm(a)

# Cosine similarity
np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Batch cosine similarity
a_norm = a / np.linalg.norm(a)
B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
scores = B_norm @ a_norm

# Softmax (stable)
exp = np.exp(logits - logits.max())
probs = exp / exp.sum()

# Statistics
scores.mean(), scores.std(), np.percentile(scores, 90)
```

## Interview One-Liners

- "Cosine similarity?" → "Angle between vectors; scale-invariant; range -1 to 1"
- "Why not Euclidean for text?" → "Long documents have large vectors; Euclidean penalizes magnitude"  
- "Softmax?" → "Converts logits to probabilities summing to 1"
- "Temperature?" → "Divides logits before softmax; low=deterministic, high=creative"
- "King-man+woman=queen?" → "Embeddings encode semantic relationships as geometric directions"
