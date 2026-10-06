# Day 04 — Assessment

---

## Quick Assessment (60 minutes, 100 points)

### Theory (5 × 8 = 40 pts)
1. Write the cosine similarity formula and explain why it's scale-invariant.
2. What is softmax? Write the formula and explain numerical stability.
3. What does temperature=0.1 vs temperature=1.5 do to LLM output?
4. What is standard deviation and why does it matter for model evaluation?
5. Explain the attention mechanism: what are Q, K, V and how are they used?

### Coding (3 × 15 = 45 pts)
1. Implement `cosine_similarity(a, b)` from scratch (no NumPy), then verify with NumPy.
2. Implement `stable_softmax(logits, temperature=1.0)` with numerical stability.
3. Implement `top_k_search(query_vec, doc_vecs, k)` using NumPy batch operations.

### One Answer (1 × 15 = 15 pts)
"A candidate tells you: 'I just use the default cosine similarity everywhere for vector search.' What are the cases where this is suboptimal?"

**Pass: 75/100**
