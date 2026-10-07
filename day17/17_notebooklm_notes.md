# Day 17 — NotebookLM Notes: Embeddings

## Key Facts

**Embedding = dense vector** (384-3072 floats) capturing semantic meaning.
Similar meanings → similar directions → high cosine similarity.

**Free local model**: `all-MiniLM-L6-v2` (384 dims, fast, good quality)
**Best quality**: `text-embedding-3-large` (3072 dims, ~$0.13/1M tokens)
**Critical rule**: Never mix embeddings from different models in the same index.

## Model Selection

| Model | Dims | Cost | Use When |
|-------|------|------|---------|
| all-MiniLM-L6-v2 | 384 | Free | Development, budget, speed |
| text-embedding-3-small | 1536 | $0.02/1M | Production, good balance |
| text-embedding-3-large | 3072 | $0.13/1M | Maximum quality |
| nomic-embed-text | 768 | Free (Ollama) | Privacy required |

## Batch Embedding

```python
# WRONG: embed one at a time
embeddings = [embed(text) for text in texts]  # N API calls

# RIGHT: batch in single call
response = client.embeddings.create(input=texts, model="text-embedding-3-small")
# One API call for entire batch
```

## Cosine Similarity with NumPy

```python
# Single comparison
sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Batch: query vs all documents (vectorized)
query_norm = query / np.linalg.norm(query)
docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
scores = np.dot(docs_norm, query_norm)  # All N scores at once
```

## Interview Facts

1. Embeddings have no "absolute meaning" per dimension — only relative positions matter
2. Changing embedding model = re-index everything from scratch
3. Normalized vectors: inner product == cosine similarity (skip normalization step = faster)
4. OpenAI embeddings are normalized by default → use inner product search
5. All-MiniLM-L6-v2: 22M params, fast, good for development
6. Semantic search finds "same meaning" even with different words; keyword search does not

## Common Mistakes

- Comparing embeddings from different models (meaningless results)
- Not normalizing when using inner product search
- Embedding entire documents (should chunk first, then embed chunks)
- Using l2 distance when cosine similarity is more appropriate for text
- Not batching embedding requests (wastes API quota, very slow)
