# Day 20 — Concepts
## Advanced RAG

---

## 1. Why Naive RAG Fails

```
NAIVE RAG FAILURES:

1. KEYWORD MISMATCH:
   Query: "cardiovascular disease risk"
   Document: "heart attack probability"
   Vector similarity: ~0.6 (lower than expected — different vocabulary)
   BM25 keyword: 0.0 (no matching words!)
   → Solution: hybrid search (BM25 + vector)

2. LOW RECALL:
   "What are all the pricing tiers?"
   Top-3 chunks might miss chunk 4 which has the actual prices
   → Solution: retrieve more (top-10), then rerank

3. AMBIGUOUS QUERY:
   "What is the policy?" (which policy?)
   → Solution: query rewriting (expand query with context)

4. MULTI-ASPECT QUERY:
   "Compare RAG and fine-tuning for enterprise use cases"
   → Needs chunks about both RAG AND fine-tuning
   → Solution: multi-query retrieval (decompose into sub-queries)

5. PARENT-CHILD MISMATCH:
   Precise retrieval hits small chunk (128 tokens)
   But response needs larger surrounding context
   → Solution: parent-child retrieval (retrieve small, return large)
```

---

## 2. Hybrid Search: BM25 + Vector

```python
# BM25: improved TF-IDF with document length normalization
# Captures exact keyword matches
# Vector: captures semantic similarity
# Hybrid = combine both → better recall

from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, documents: list[str], embeddings: np.ndarray) -> None:
        self._documents = documents
        self._embeddings = embeddings
        # BM25 operates on tokenized documents
        tokenized = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        query_emb: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.5,  # Weight: 0=pure BM25, 1=pure vector
    ) -> list[tuple[int, float]]:
        # BM25 scores
        bm25_scores = np.array(
            self._bm25.get_scores(query.lower().split())
        )

        # Vector scores
        q_norm = query_emb / np.linalg.norm(query_emb)
        d_norm = self._embeddings / np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        vector_scores = np.dot(d_norm, q_norm)

        # Normalize both to [0, 1]
        def normalize(scores: np.ndarray) -> np.ndarray:
            mn, mx = scores.min(), scores.max()
            return (scores - mn) / (mx - mn + 1e-8)

        bm25_norm = normalize(bm25_scores)
        vec_norm = normalize(vector_scores)

        # Combine: Reciprocal Rank Fusion (RRF) is another option
        combined = (1 - alpha) * bm25_norm + alpha * vec_norm

        top_indices = np.argsort(combined)[::-1][:top_k]
        return [(int(i), float(combined[i])) for i in top_indices]
```

---

## 3. Reranking

```python
# Reranking: after retrieving top-K candidates, re-score with
# a more expensive cross-encoder model
# Cross-encoder: reads query + document together (context-aware)
# vs bi-encoder: embeds query and document separately

from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Rerank retrieved candidates using cross-encoder.
        Returns top_k most relevant after reranking.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        # Sort by cross-encoder score (higher = more relevant)
        reranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        for score, candidate in reranked[:top_k]:
            candidate["rerank_score"] = float(score)

        return [c for _, c in reranked[:top_k]]

# Pipeline: retrieve top-20 with vector search → rerank → keep top-3
# Why? First stage (vector) is fast but imprecise.
#      Second stage (cross-encoder) is slow but accurate.
#      Doing cross-encoder on all docs: too slow.
#      Two-stage: best of both worlds.
```

---

## 4. Query Rewriting

```python
def rewrite_query(original_query: str, conversation_history: list[dict]) -> str:
    """
    Rewrite a potentially ambiguous query using conversation history.

    Example:
    History: User asked about "Python RAG"
    User: "What about the evaluation metrics?"
    Rewritten: "What are the evaluation metrics for Python RAG?"
    """
    if not conversation_history:
        return original_query

    history_text = "\n".join(
        f"{m['role'].title()}: {m['content'][:100]}"
        for m in conversation_history[-3:]  # Last 3 turns
    )

    prompt = f"""Given this conversation history:
{history_text}

Rewrite the following question to be self-contained and specific.
Return ONLY the rewritten question.

Original: {original_query}
Rewritten:"""

    return llm.complete(prompt).strip()
```

---

## 5. Multi-Query Retrieval

```python
def generate_sub_queries(query: str, n: int = 3) -> list[str]:
    """
    Generate multiple search queries from one user question.
    Improves recall by covering different aspects.

    Example:
    "Compare RAG and fine-tuning performance and cost"
    → ["RAG performance benchmarks",
       "fine-tuning cost and compute requirements",
       "RAG vs fine-tuning accuracy comparison"]
    """
    prompt = f"""Generate {n} different search queries to find information that answers:
"{query}"

Each query should cover a different aspect. Return as JSON array of strings.
["query1", "query2", "query3"]"""

    raw = llm.complete(prompt)
    try:
        return json.loads(raw.strip())
    except Exception:
        return [query]  # Fallback to original

def multi_query_retrieve(
    query: str,
    retrieve_fn,  # callable that takes query and returns chunks
    top_k_per_query: int = 5,
) -> list[dict]:
    """Retrieve for multiple query variants, deduplicate."""
    sub_queries = [query] + generate_sub_queries(query)
    all_chunks: dict[str, dict] = {}  # id → chunk

    for sub_query in sub_queries:
        chunks = retrieve_fn(sub_query, top_k=top_k_per_query)
        for chunk in chunks:
            chunk_id = chunk.get("id", chunk["text"][:50])
            # Keep highest-scoring version of each unique chunk
            if chunk_id not in all_chunks or chunk["score"] > all_chunks[chunk_id]["score"]:
                all_chunks[chunk_id] = chunk

    # Sort by score and return unique chunks
    return sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
```

---

## 6. Semantic Chunking

```python
def semantic_chunk(text: str, embedder, threshold: float = 0.3) -> list[str]:
    """
    Chunk text at semantically distinct boundaries.
    Splits when consecutive sentences change topic significantly.
    """
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return [text]

    # Embed all sentences
    embeddings = embedder.embed_batch(sentences)

    # Find split points where semantic similarity drops
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        if sim < threshold:
            # Topic changed — start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```
