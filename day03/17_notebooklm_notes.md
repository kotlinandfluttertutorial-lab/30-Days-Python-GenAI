# Day 03 — NotebookLM Notes
## Python Engineering + NumPy + Pandas

---

## Key Concepts

**Abstract Base Class (ABC)** — Defines interface contract. `@abstractmethod` forces subclasses to implement. Raises `TypeError` at instantiation if methods missing. Used in every AI framework for embedders, LLMs, vector stores.

**Composition over Inheritance** — AI pipeline HAS an embedder, not IS an embedder. Lets you swap components (OpenAI→Local, ChromaDB→FAISS) without changing pipeline code. Better testability (swap real for mock).

**NumPy Array** — Contiguous C memory + BLAS operations + SIMD = 100-500x faster than Python lists for vector math. Essential for embeddings (768-3072 dims), batch operations, similarity search.

**Batch Cosine Similarity** — Normalize query (D,) and docs (N,D), then `np.dot(docs_norm, query_norm)` gives all N similarities at once. Faster than N individual cosine calls.

**Pandas DataFrame** — Tabular data structure for loading, profiling, cleaning, and feature engineering datasets before ML.

**Data Leakage** — Test set information leaks into training when you fit preprocessors on full data before splitting. Always: split first, `fit_transform` on train only, `transform` on val/test.

**Feature Engineering** — Creating useful ML features from raw data. Text features: word_count, char_count, avg_word_length, sentence_count, has_numbers.

## Code Patterns

```python
# Batch cosine similarity (NumPy)
q = query / np.linalg.norm(query)
d = docs / np.linalg.norm(docs, axis=1, keepdims=True)
scores = np.dot(d, q)
top_k_idx = np.argsort(scores)[::-1][:k]

# Train/val/test split (correct order)
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)
scaler.fit_transform(train[["x"]])  # fit on train only
scaler.transform(val[["x"]])         # transform only

# Abstract class
from abc import ABC, abstractmethod
class Base(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...
```

## Interview Facts

1. `@abstractmethod` raises `TypeError` at instantiation, not at definition
2. Composition: change embedder without changing RAG pipeline code
3. `np.dot(docs_norm, query_norm)` computes N cosine similarities at once
4. `fit_transform` on test data = data leakage = biased evaluation
5. `df.fillna(x)` without `inplace=True` or reassignment has no effect
6. `np.argsort()[::-1][:k]` = top-K indices by descending score

## Common Traps

- Forget `@abstractmethod` → subclass can skip method, fails at call time not instantiation
- Multiple inheritance for AI components → tight coupling, can't swap
- `df.fillna()` without inplace/reassignment → no effect (silent bug)
- `fit_transform` on test set → data leakage → inflated eval metrics
- Python loop for cosine similarity over 10K docs → too slow; use NumPy batch
- Not calling `.copy()` on DataFrame slice → SettingWithCopyWarning
