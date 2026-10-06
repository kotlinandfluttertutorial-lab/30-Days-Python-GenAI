# Day 03 — Revision

---

## OOP Quick Reference

```python
# Abstract base class
from abc import ABC, abstractmethod
class Base(ABC):
    @abstractmethod
    def method(self) -> str: ...    # Must implement in subclass

# Property
class Config:
    @property
    def temperature(self) -> float: return self._temp
    @temperature.setter
    def temperature(self, v: float) -> None: self._temp = v

# Classmethod (factory)
@classmethod
def from_env(cls) -> "Config": return cls(os.environ["KEY"])

# Staticmethod (utility)
@staticmethod
def is_valid(model: str) -> bool: return model in VALID_MODELS
```

## NumPy Quick Reference

```python
import numpy as np

# Create
v = np.array([0.1, 0.2, 0.3])          # 1D
m = np.zeros((100, 768))                # 2D batch

# Cosine similarity (single)
sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Batch cosine similarity
q_norm = query / np.linalg.norm(query)
d_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
scores = np.dot(d_norm, q_norm)          # shape: (N,)

# Top-K
top_k_idx = np.argsort(scores)[::-1][:k]
```

## Pandas Quick Reference

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.shape; df.dtypes; df.describe(); df.isnull().sum()

# Clean
df = df.dropna(subset=["text"])
df["score"] = df["score"].fillna(df["score"].median())

# Feature engineering
df["word_count"] = df["text"].str.split().str.len()

# Split (fit scaler on train only!)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

## Interview One-Liners

- "ABC?" → "Abstract base class: enforces interface contracts via @abstractmethod"
- "Inheritance vs composition?" → "Inherit for is-a, compose for has-a — prefer composition"
- "NumPy speed?" → "Contiguous memory + BLAS + SIMD = 100-500x faster than Python loops"
- "Data leakage?" → "Test set info leaks into training — always split BEFORE fitting preprocessors"
- "Validation set purpose?" → "Hyperparameter tuning; test set touched only once at the very end"
