# Day 04 — Deep Dive
## Senior AI Engineer: Why This Math Matters in Production

---

## Deep Dive 1: Cosine Similarity vs Inner Product vs Euclidean

In production vector search, you choose between three similarity measures:

```
1. Cosine similarity: dot(a,b) / (|a| × |b|)
   - Scale invariant
   - Best for: text embeddings, semantic search
   - ChromaDB and most vector DBs default to this

2. Inner product (dot product): dot(a,b)
   - Not scale invariant (favors longer vectors)
   - Best for: when vectors are already normalized (FAISS IVF Flat)
   - Faster than cosine (skip division)
   - OpenAI recommends inner product for their normalized embeddings

3. Euclidean distance: |a - b|
   - Measures geometric distance
   - Best for: dense numeric features, k-NN classification
   - Less common for text

PRODUCTION DECISION:
If embedding model normalizes vectors (|v| = 1.0 always):
  → Use inner product (faster, equivalent to cosine)
If not normalized:
  → Use cosine similarity

OpenAI text-embedding-3 vectors are normalized → use inner product.
```

---

## Deep Dive 2: Numerical Stability in Softmax

```python
# UNSTABLE: large logits cause overflow
def softmax_unstable(logits):
    exp_values = [math.exp(x) for x in logits]   # math.exp(1000) = inf!
    total = sum(exp_values)
    return [e/total for e in exp_values]

# STABLE: subtract max before exponentiation
def softmax_stable(logits):
    max_val = max(logits)
    exp_values = [math.exp(x - max_val) for x in logits]  # max exp is e^0 = 1
    total = sum(exp_values)
    return [e/total for e in exp_values]
    # Mathematically equivalent (max cancels out in division)
    # Numerically stable: no overflow

# NumPy handles this automatically
probs = np.exp(logits - logits.max()) 
probs /= probs.sum()
```

This matters because LLM logits can be large, and naive softmax causes NaN errors.

---

## Deep Dive 3: The Math Behind Temperature Sampling

```
Temperature τ is applied BEFORE softmax:
  p_i = exp(logit_i / τ) / Σ exp(logit_j / τ)

τ → 0: all probability mass on the highest logit (greedy decoding)
τ = 1: standard softmax (default)
τ → ∞: uniform distribution (pure random)

Why divide by τ?
  High logit ÷ low τ = very high value → steep distribution
  High logit ÷ high τ = moderate value → flat distribution

τ = 0.1: token with logit 10 vs token with logit 5
  p(10) = exp(100) / (exp(100) + exp(50)) ≈ 1.0  (almost certain)
  
τ = 2.0: same tokens
  p(10) = exp(5) / (exp(5) + exp(2.5)) ≈ 0.92  (still dominant but less)
```

---

## Deep Dive 4: Vector Space Algebra (Embeddings)

Famous example from Word2Vec:
```
king - man + woman ≈ queen

Vector arithmetic:
  embedding("king")   = [0.9, 0.8, -0.1, ...]
  embedding("man")    = [0.7, -0.2, 0.5, ...]
  embedding("woman")  = [0.6, -0.1, 0.6, ...]
  result              = [0.8, 0.9, 0.0, ...]  ← close to embedding("queen")

This works because the embedding space encodes semantic relationships
as geometric transformations. The "royalty" and "gender" dimensions
are different axes in the vector space.
```

This is why RAG works: queries and documents that talk about the same topic
will point in the same direction in embedding space, regardless of exact wording.
