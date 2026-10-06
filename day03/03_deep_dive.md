# Day 03 — Deep Dive
## Senior AI Engineer: OOP, NumPy, and Data Engineering

---

## Deep Dive 1: Why Composition Over Inheritance in AI Systems

In AI frameworks (LangChain, LlamaIndex, Haystack), composition is used everywhere. Here's why.

### The Inheritance Problem

```python
# Inheritance creates rigid coupling:
class GPT4RAG(OpenAIRAG):
    pass
# What if you want to switch the LLM but keep the RAG logic?
# You can't — they're tied together by inheritance.

# Composition gives you freedom:
pipeline = RAGPipeline(
    embedder=OpenAIEmbedder(),   # swap to LocalEmbedder() anytime
    llm=GPT4(),                  # swap to Claude() anytime
    store=ChromaDB(),            # swap to FAISS() anytime
)
```

The rule: **inherit for "is-a", compose for "has-a"**. A RAG pipeline HAS an LLM. It is not AN LLM.

---

## Deep Dive 2: NumPy Memory Layout and Performance

```python
# Why NumPy is so much faster than Python for vectorized math:

# Python list: each element is a Python object
# Memory: [ptr→obj, ptr→obj, ptr→obj, ...]
# Access: dereference pointer → get Python object → extract float → math

# NumPy array: C-contiguous block of raw floats
# Memory: [float64, float64, float64, ...]  (raw bytes)
# Access: direct offset calculation → raw float → math (SIMD)

# Real impact:
# Cosine similarity, 1M dimensions:
# Python: ~500ms (loop + Python objects)
# NumPy: ~1ms   (vectorized BLAS)

# For embeddings (768-3072 dimensions, millions of documents):
# Without NumPy: unusably slow
# With NumPy: production-ready
```

---

## Deep Dive 3: Data Leakage in Train/Val/Test Splits

This is one of the most common ML mistakes. Understanding it is required for AI engineering.

```python
# WRONG — data leakage:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fitting on entire dataset BEFORE splitting
df["score_normalized"] = scaler.fit_transform(df[["score"]])
train, val, test = split(df)
# PROBLEM: scaler learned from test set!
# The test set has "leaked" into your normalization.
# Your evaluation metrics are optimistically biased.

# RIGHT — fit only on training data:
train, val, test = split(df)
scaler = StandardScaler()
train["score_normalized"] = scaler.fit_transform(train[["score"]])
val["score_normalized"] = scaler.transform(val[["score"]])   # transform, not fit_transform
test["score_normalized"] = scaler.transform(test[["score"]])  # transform, not fit_transform

# Rule: NEVER let the test set influence preprocessing or feature engineering.
```

---

## Deep Dive 4: Abstract Base Classes as Contracts

ABCs enforce that every implementation provides the required interface, catching errors at class definition time rather than at runtime.

```python
from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, id: str, vector: list[float], text: str) -> None: ...
    
    @abstractmethod
    def search(self, query: list[float], top_k: int) -> list[tuple[str, float]]: ...
    
    @abstractmethod
    def delete(self, id: str) -> None: ...
    
    @abstractmethod
    def count(self) -> int: ...

# If you forget to implement 'delete':
class BadStore(BaseVectorStore):
    def add(self, id, vector, text): pass
    def search(self, query, top_k): return []
    def count(self): return 0
    # No delete()!

# Caught immediately at instantiation:
store = BadStore()
# TypeError: Can't instantiate abstract class BadStore
# with abstract method delete
```

This is how LangChain, LlamaIndex, and every AI framework enforces its plugin system.
