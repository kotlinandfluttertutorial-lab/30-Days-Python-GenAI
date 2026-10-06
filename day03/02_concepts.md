# Day 03 — Concepts
## Python Engineering + NumPy + Pandas

---

## 1. Object-Oriented Programming for AI Systems

### Why OOP in AI Engineering?

AI systems have many components that share common interfaces:
- Multiple LLM providers (OpenAI, Anthropic, Groq) → base `LLMClient`
- Multiple vector stores (ChromaDB, FAISS, pgvector) → base `VectorStore`
- Multiple chunking strategies → base `TextChunker`
- Multiple evaluation metrics → base `Evaluator`

OOP lets you: swap implementations without changing calling code, test with mocks, and extend without modifying existing code.

### Class Patterns

```python
from abc import ABC, abstractmethod
from typing import Any

# Abstract Base Class — defines the interface
class BaseEmbedder(ABC):
    """All embedders must implement this interface."""
    
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        ...
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the output vector."""
        ...

# Concrete implementation
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model
        self._dims = 1536
    
    def embed(self, text: str) -> list[float]:
        # Call OpenAI API
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Batch API call
        ...
    
    @property
    def dimensions(self) -> int:
        return self._dims

# Another implementation — same interface
class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()
    
    @property
    def dimensions(self) -> int:
        return 384  # MiniLM output size

# Code that works with ANY embedder
def build_vector_index(documents: list[str], embedder: BaseEmbedder) -> None:
    embeddings = embedder.embed_batch(documents)
    # Works regardless of which embedder is used
```

### Inheritance vs Composition

```python
# INHERITANCE — "is-a" relationship
class SpecializedRAG(BaseRAG):
    """Specialized RAG IS A BaseRAG with extra features."""
    pass

# COMPOSITION — "has-a" relationship (usually preferred)
class RAGPipeline:
    """RAG pipeline HAS AN embedder, HAS A vector store, HAS AN LLM."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,    # composed
        vector_store: BaseVectorStore,
        llm: BaseLLM,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
    
    def query(self, question: str) -> str:
        embedding = self.embedder.embed(question)
        chunks = self.vector_store.search(embedding)
        context = "\n".join(c.text for c in chunks)
        return self.llm.complete(f"Context: {context}\nQuestion: {question}")
```

### Property, Classmethod, Staticmethod

```python
class LLMConfig:
    def __init__(self, model: str, temperature: float) -> None:
        self._model = model
        self._temperature = temperature
    
    @property
    def model(self) -> str:
        """Read-only model name."""
        return self._model
    
    @property
    def temperature(self) -> float:
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"Temperature {value} out of range [0, 2]")
        self._temperature = value
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables."""
        import os
        return cls(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        )
    
    @staticmethod
    def is_valid_model(model: str) -> bool:
        """Check if a model name is known."""
        valid = {"gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"}
        return model in valid
```

---

## 2. NumPy for AI Engineering

NumPy is the foundation of all numerical computing in Python. ML algorithms, embeddings, and similarity computations all use NumPy arrays.

### Arrays vs Lists

```python
import numpy as np

# Python list: general purpose, slow math
python_list = [0.1, 0.2, 0.3, 0.4, 0.5]

# NumPy array: homogeneous, fast math
array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# Speed comparison for 1M element dot product:
# Python list: ~500ms
# NumPy array: ~1ms  (500x faster)

# Why? NumPy uses BLAS/LAPACK (optimized C/Fortran libraries)
# and vectorized operations (SIMD instructions)
```

### Critical NumPy Operations for AI

```python
# Creating arrays
zeros = np.zeros(768)              # 768-dim zero vector
ones = np.ones((100, 1536))        # 100 embeddings of dim 1536
random_emb = np.random.randn(768)  # Random embedding

# Array properties
emb = np.array([0.1, 0.5, -0.3, 0.8])
print(emb.shape)    # (4,)
print(emb.dtype)    # float64
print(emb.ndim)     # 1

# Matrix (2D array): batch of embeddings
batch = np.random.randn(10, 768)   # 10 embeddings, 768 dims each
print(batch.shape)  # (10, 768)

# ESSENTIAL: Cosine similarity with NumPy
def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Batch cosine similarity: query vs many documents
def batch_cosine_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """
    query: shape (D,)  — one embedding
    docs:  shape (N, D) — N document embeddings
    Returns: shape (N,) — similarity score for each doc
    """
    # Normalize query
    query_norm = query / np.linalg.norm(query)
    # Normalize all docs at once (vectorized)
    docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    # Dot product = cosine similarity (since both normalized)
    return np.dot(docs_norm, query_norm)

# Statistical operations (used in evaluation)
scores = np.array([0.92, 0.78, 0.85, 0.91, 0.70])
print(f"Mean: {scores.mean():.3f}")
print(f"Std:  {scores.std():.3f}")
print(f"Min:  {scores.min():.3f}")
print(f"Max:  {scores.max():.3f}")
print(f"P90:  {np.percentile(scores, 90):.3f}")  # 90th percentile

# Array indexing and slicing
top_indices = np.argsort(scores)[::-1][:3]  # Top-3 indices
top_scores = scores[top_indices]
print(f"Top-3 scores: {top_scores}")

# Boolean indexing
high_relevance = scores[scores >= 0.85]  # Only scores ≥ 0.85
print(f"High relevance scores: {high_relevance}")
```

### NumPy for Embeddings

```python
# Store and search embeddings efficiently
class NumpyVectorStore:
    """Simple in-memory vector store using NumPy."""
    
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.embeddings: np.ndarray = np.empty((0, dimensions))
        self.texts: list[str] = []
    
    def add(self, text: str, embedding: list[float]) -> None:
        emb = np.array(embedding).reshape(1, -1)
        self.embeddings = np.vstack([self.embeddings, emb]) if len(self.embeddings) else emb
        self.texts.append(text)
    
    def search(self, query: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        query_np = np.array(query)
        scores = batch_cosine_similarity(query_np, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.texts[i], float(scores[i])) for i in top_indices]
```

---

## 3. Pandas for AI Data Engineering

Pandas is used to load, inspect, clean, and prepare datasets for ML and RAG.

### Essential Pandas Operations

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("dataset.csv")
df = pd.read_json("documents.json")

# Inspect
print(df.shape)         # (rows, columns)
print(df.dtypes)        # column types
print(df.head(5))       # first 5 rows
print(df.info())        # types + non-null counts
print(df.describe())    # statistics for numeric columns

# Missing values
print(df.isnull().sum())         # null count per column
print(df.isnull().mean() * 100)  # null percentage per column

# Fill missing values
df["text"].fillna("", inplace=True)
df["score"].fillna(df["score"].median(), inplace=True)

# Drop rows/columns
df.dropna(subset=["text"], inplace=True)      # Drop rows with null text
df.drop(columns=["unnecessary_col"], inplace=True)

# Filter rows
long_texts = df[df["text"].str.len() > 50]
recent = df[df["date"] > "2024-01-01"]

# String operations
df["text_clean"] = df["text"].str.strip().str.lower()
df["word_count"] = df["text"].str.split().str.len()
df["has_numbers"] = df["text"].str.contains(r"\d+", regex=True)

# Apply a function to each row
df["token_estimate"] = df["text"].apply(lambda t: len(t) // 4)

# GroupBy
category_stats = df.groupby("category").agg({
    "score": ["mean", "std", "count"],
    "word_count": "mean",
})

# Sorting
df_sorted = df.sort_values("score", ascending=False)
top_docs = df.nlargest(10, "score")

# Export
df.to_csv("cleaned_data.csv", index=False)
df.to_json("cleaned_data.json", orient="records")
```

### Feature Engineering for ML

```python
# Feature engineering transforms raw data into ML-ready features

# 1. Text features
df["text_length"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()
df["sentence_count"] = df["text"].str.count(r"[.!?]")
df["avg_word_length"] = df["text"].str.split().apply(
    lambda words: np.mean([len(w) for w in words]) if words else 0
)

# 2. Categorical encoding (for ML models)
# One-hot encoding
df = pd.get_dummies(df, columns=["category"], prefix="cat")

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# 3. Numerical normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()  # Zero mean, unit variance
df["score_normalized"] = scaler.fit_transform(df[["score"]])

# 4. Train/validation/test split
from sklearn.model_selection import train_test_split

# 70/15/15 split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

---

## 4. Logging for AI Applications

Structured logging is critical for debugging AI systems.

```python
import logging
import sys

def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure logging for an AI application."""
    handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        # JSON format for production (parse with log aggregators)
        import json
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({
                    "time": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    **(record.__dict__.get("extra", {})),
                })
        handler.setFormatter(JSONFormatter())
    else:
        # Human-readable for development
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
    
    logging.basicConfig(level=getattr(logging, level.upper()), handlers=[handler])

# Usage
logger = logging.getLogger(__name__)

def process_document(doc_id: str, text: str) -> None:
    logger.info("Processing document", extra={"doc_id": doc_id, "length": len(text)})
    
    chunks = chunk_text(text)
    logger.debug(f"Created {len(chunks)} chunks from document {doc_id}")
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = embed(chunk)
            logger.debug(f"Embedded chunk {i}", extra={"tokens": len(chunk.split())})
        except Exception as e:
            logger.error(f"Failed to embed chunk {i} of {doc_id}: {e}", exc_info=True)
            raise
```

---

## 5. Configuration Management

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class AppConfig:
    """Centralized application configuration."""
    # LLM
    openai_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    
    # RAG
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    
    # Database
    database_url: str
    redis_url: str
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        load_dotenv()
        
        required = ["OPENAI_API_KEY", "DATABASE_URL"]
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        return cls(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            top_k_results=int(os.getenv("TOP_K_RESULTS", "5")),
            database_url=os.environ["DATABASE_URL"],
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        )
```
