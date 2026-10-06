# Day 03 — Exercise Solutions

---

## C3. top_k_similar

```python
import numpy as np

def top_k_similar(
    query: np.ndarray,
    docs: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (indices, scores) of top-k most similar documents."""
    query_norm = query / np.linalg.norm(query)
    docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    scores = np.dot(docs_norm, query_norm)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return top_k_idx, scores[top_k_idx]
```

## D1. Abstract method fix
```python
from abc import ABC, abstractmethod
class BaseEmbedder(ABC):
    @abstractmethod  # Missing decorator — without it, subclasses don't HAVE to implement it
    def embed(self, text: str) -> list[float]:
        ...
# Without @abstractmethod, a subclass that forgets embed() won't raise an error
# until embed() is actually called — runtime failure instead of definition-time failure.
```

## D2. Data leakage fix
```python
# WRONG: fit scaler on all data including test
# RIGHT: fit only on train
train_df, test_df = train_test_split(df)
scaler = StandardScaler()
train_df["score_scaled"] = scaler.fit_transform(train_df[["score"]])
test_df["score_scaled"] = scaler.transform(test_df[["score"]])  # NOT fit_transform
```

## D4. Pandas fillna bug
```python
# Bug: fillna returns a NEW DataFrame, doesn't modify in place
# df.fillna(mean_score)  ← result discarded!

# Fix option 1: reassign
df = df.fillna(mean_score)

# Fix option 2: inplace
df.fillna(mean_score, inplace=True)

# Fix option 3 (best): column-specific fill
df["quality_score"] = df["quality_score"].fillna(df["quality_score"].median())
```

## D5. Multiple inheritance design error
```python
# Multiple inheritance from unrelated classes creates:
# 1. Diamond inheritance problems (MRO complexity)
# 2. Tight coupling (can't swap any component)
# 3. Can't test components independently

# CORRECT: composition
class RAGPipeline:
    def __init__(self, embedder, store, llm): ...
```
