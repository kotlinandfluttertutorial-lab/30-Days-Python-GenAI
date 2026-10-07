# Day 17 — Concepts
## Embeddings

---

## 1. What Are Embeddings?

```python
# Embeddings: text → dense vector that captures semantic meaning
# Similar meanings → nearby vectors → high cosine similarity

# Real dimensions:
# text-embedding-3-small:  1536 dimensions
# text-embedding-3-large:  3072 dimensions
# all-MiniLM-L6-v2:         384 dimensions (free, local)
# nomic-embed-text:          768 dimensions (free, local)

from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Embed text using OpenAI API."""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Example: semantic equivalence despite different wording
texts = [
    "How do I reset my password?",
    "I forgot my password",           # same intent
    "Password recovery instructions",  # same topic
    "What is the weather today?",      # different topic
]

embeddings = [embed(t) for t in texts]
```

---

## 2. Cosine Similarity at Scale

```python
import numpy as np
from typing import Any

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def batch_cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute query similarity against all corpus vectors at once.
    1000× faster than looping.
    """
    query_norm = query / np.linalg.norm(query)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    return np.dot(corpus_norm, query_norm)

def semantic_search(
    query: str,
    documents: list[str],
    embeddings: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Find top-k most semantically similar documents."""
    query_emb = np.array(embed(query))
    scores = batch_cosine_similarity(query_emb, embeddings)
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices[:top_k]:
        if scores[idx] >= min_score:
            results.append({
                "document": documents[idx],
                "score": float(scores[idx]),
                "index": int(idx),
            })
    return results
```

---

## 3. Embedding Models Comparison

```python
# OpenAI (API call required)
from openai import OpenAI
client = OpenAI()
def openai_embed(text: str) -> list[float]:
    r = client.embeddings.create(input=text, model="text-embedding-3-small")
    return r.data[0].embedding
# Dims: 1536, Cost: $0.02/1M tokens, Good quality

# HuggingFace Sentence Transformers (FREE, local)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast, good
def local_embed(text: str) -> list[float]:
    return model.encode(text).tolist()
# Dims: 384, Cost: FREE, Fast inference, Good quality

# Ollama (FREE, local, GPU)
import requests
def ollama_embed(text: str, model: str = "nomic-embed-text") -> list[float]:
    r = requests.post("http://localhost:11434/api/embeddings",
                      json={"model": model, "prompt": text})
    return r.json()["embedding"]
# Dims: 768, Cost: FREE, Requires Ollama installed
```

---

## 4. Embedding Model Selection

```
Criteria          | Recommendation
Budget            | sentence-transformers (free)
Best quality      | text-embedding-3-large (OpenAI)
Lowest latency    | all-MiniLM-L6-v2 (384-dim)
Privacy required  | all-MiniLM-L6-v2 or nomic-embed-text
Multilingual      | paraphrase-multilingual-MiniLM-L12-v2
Code search       | text-embedding-3-small or CodeBERT

CRITICAL RULE:
Pick one model and NEVER change it.
Changing models requires re-embedding ALL documents.
Mixing models = meaningless similarity scores.
```

---

## 5. Batch Embedding Efficiently

```python
import asyncio
from openai import AsyncOpenAI

async def embed_batch_async(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed many texts efficiently with batching and rate limiting."""
    client = AsyncOpenAI()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # OpenAI accepts multiple texts per call
        response = await client.embeddings.create(
            input=batch,
            model="text-embedding-3-small",
        )
        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend(e.embedding for e in embeddings)

        # Rate limit delay between batches
        if i + batch_size < len(texts):
            await asyncio.sleep(0.1)

    return all_embeddings
```

---

## 6. Dimensionality Reduction for Visualization

```python
# PCA or UMAP to visualize high-dimensional embeddings in 2D
from sklearn.decomposition import PCA

embeddings_2d = PCA(n_components=2).fit_transform(embeddings_array)

# Clusters visible in 2D space correspond to semantic categories
# Use for: debugging embedding quality, finding misclassified docs
```
