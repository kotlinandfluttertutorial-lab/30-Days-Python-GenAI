# Day 04 — Architecture
## Math in the AI System

```
WHERE EACH MATH OPERATION LIVES IN AN AI SYSTEM:

User Query: "What is RAG?"
    │
    ▼
[EMBED QUERY]
  text → model → vector (768 floats)
  Math: learned mapping via deep learning
    │
    ▼
[SIMILARITY SEARCH]
  cosine_similarity(query_vec, each_doc_vec)
  Math: dot(a,b) / (|a|×|b|)
    │
    ▼
[RANK RESULTS]
  np.argsort(scores)[::-1][:k]
  Math: sorting, indexing
    │
    ▼
[LLM GENERATION]
  forward_pass(input_tokens) → logits
  softmax(logits / temperature) → probabilities
  sample(probabilities) → next token
  Math: matrix multiplication, softmax
    │
    ▼
[EVALUATION]
  mean(eval_scores), std(eval_scores)
  Math: statistics
    │
    ▼
Response
```
