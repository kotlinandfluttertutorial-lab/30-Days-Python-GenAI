"""
Day 03 — OOP Patterns for AI Engineering
==========================================
Demonstrates the OOP patterns used in every AI framework:
abstract base classes, composition, properties, classmethods.

Run: python 01_oop_patterns.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# BASE CLASSES — THE AI COMPONENT INTERFACES
# ═══════════════════════════════════════════════════════════════

class BaseEmbedder(ABC):
    """
    Abstract interface for all embedding models.
    Concrete implementations: OpenAIEmbedder, LocalEmbedder, MockEmbedder.
    Code that depends on BaseEmbedder works with ANY implementation.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (may batch for efficiency)."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions."""
        ...

    def similarity(self, a: str, b: str) -> float:
        """Compute cosine similarity between two texts (uses embed)."""
        import math
        va, vb = self.embed(a), self.embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        mag_a = math.sqrt(sum(x**2 for x in va))
        mag_b = math.sqrt(sum(x**2 for x in vb))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


class BaseChunker(ABC):
    """Abstract interface for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks."""
        ...

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        ...


class BaseLLM(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════════
# CONCRETE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

class MockEmbedder(BaseEmbedder):
    """
    Deterministic test embedder.
    Use in tests instead of calling real APIs.
    """

    def __init__(self, dims: int = 4) -> None:
        self._dims = dims

    def embed(self, text: str) -> list[float]:
        """Deterministic embedding based on text hash."""
        seed = hash(text) % 10000
        import math
        raw = [math.sin(seed + i) for i in range(self._dims)]
        # Normalize to unit vector
        magnitude = math.sqrt(sum(x**2 for x in raw))
        return [x / magnitude for x in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunking with overlap."""

    def __init__(self, size: int = 512, overlap: int = 50) -> None:
        self._size = size
        self._overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self._size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) >= 20:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - self._overlap
        return chunks

    @property
    def chunk_size(self) -> int:
        return self._size


class SentenceChunker(BaseChunker):
    """Chunk at sentence boundaries for better semantic coherence."""

    def __init__(self, max_sentences: int = 5) -> None:
        self._max_sentences = max_sentences

    def chunk(self, text: str) -> list[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        for i in range(0, len(sentences), self._max_sentences):
            chunk = " ".join(sentences[i:i + self._max_sentences]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    @property
    def chunk_size(self) -> int:
        return self._max_sentences  # In sentences, not chars


class MockLLM(BaseLLM):
    """Deterministic mock LLM for testing."""

    def __init__(self, model: str = "mock-gpt") -> None:
        self._model = model
        self._call_count = 0

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self._call_count += 1
        return f"[Mock response #{self._call_count} to: {prompt[:40]}...]"

    def get_model_name(self) -> str:
        return self._model

    @property
    def call_count(self) -> int:
        return self._call_count


# ═══════════════════════════════════════════════════════════════
# COMPOSITION: RAG PIPELINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str = ""


class SimpleRAGPipeline:
    """
    RAG pipeline composed of:
    - An embedder (for query + document embedding)
    - A chunker (for splitting documents)
    - An LLM (for generating answers)

    This is composition: RAG pipeline HAS embedder, chunker, LLM.
    It doesn't inherit from any of them.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunker: BaseChunker,
        llm: BaseLLM,
    ) -> None:
        self._embedder = embedder
        self._chunker = chunker
        self._llm = llm
        self._document_store: list[tuple[str, list[float]]] = []  # (text, embedding)

    def ingest(self, text: str, source: str = "") -> int:
        """Add a document to the pipeline. Returns number of chunks added."""
        chunks = self._chunker.chunk(text)
        embeddings = self._embedder.embed_batch(chunks)

        for chunk, emb in zip(chunks, embeddings):
            self._document_store.append((chunk, emb))

        logger.info(f"Ingested {len(chunks)} chunks from '{source}'")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Find most relevant chunks for a query."""
        if not self._document_store:
            return []

        query_emb = self._embedder.embed(query)

        # Score all documents
        scored = []
        for text, doc_emb in self._document_store:
            score = self._cosine(query_emb, doc_emb)
            scored.append(RetrievedChunk(text=text, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def answer(self, question: str) -> str:
        """Full RAG: retrieve + generate."""
        chunks = self.retrieve(question)
        context = "\n".join(f"[{c.score:.2f}] {c.text}" for c in chunks)
        prompt = f"Answer based on context:\n{context}\n\nQuestion: {question}"
        return self._llm.complete(prompt)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

    @property
    def document_count(self) -> int:
        return len(self._document_store)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION WITH CLASSMETHOD
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_dims: int = 4  # Small for demo
    llm_model: str = "mock-gpt"
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load from environment (or use defaults)."""
        import os
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            top_k=int(os.getenv("TOP_K", "3")),
        )

    @classmethod
    def for_testing(cls) -> "PipelineConfig":
        """Small config for fast tests."""
        return cls(chunk_size=100, chunk_overlap=10, top_k=2)

    @staticmethod
    def valid_chunk_sizes() -> list[int]:
        return [128, 256, 512, 1024, 2048]


# ═══════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════

def demonstrate_oop() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 03 — OOP PATTERNS FOR AI ENGINEERING          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Show polymorphism: same code, different implementations
    print("\n── POLYMORPHISM ──")
    embedders: list[BaseEmbedder] = [
        MockEmbedder(dims=4),
        MockEmbedder(dims=8),
    ]
    for emb in embedders:
        vec = emb.embed("What is RAG?")
        print(f"  {emb.__class__.__name__} ({emb.dimensions}D): {[f'{v:.3f}' for v in vec[:3]]}...")

    # 2. Show composition: build RAG from parts
    print("\n── COMPOSITION: BUILD RAG FROM PARTS ──")
    config = PipelineConfig.for_testing()

    pipeline = SimpleRAGPipeline(
        embedder=MockEmbedder(dims=config.embedding_dims),
        chunker=FixedSizeChunker(size=config.chunk_size, overlap=config.chunk_overlap),
        llm=MockLLM(),
    )

    # Ingest documents
    docs = [
        ("RAG stands for Retrieval-Augmented Generation. It reduces hallucination by "
         "retrieving relevant documents and injecting them into the LLM prompt context.", "doc1"),
        ("Embeddings are dense vectors that capture semantic meaning. Similar texts "
         "produce similar vectors. Cosine similarity measures angle between vectors.", "doc2"),
        ("Vector databases store and index high-dimensional embedding vectors. "
         "They enable fast similarity search using ANN algorithms like HNSW.", "doc3"),
    ]

    for text, source in docs:
        n = pipeline.ingest(text, source)
        print(f"  Ingested '{source}': {n} chunks")

    print(f"\n  Total documents in store: {pipeline.document_count}")

    # Retrieve
    print("\n── RETRIEVAL ──")
    query = "How does RAG work?"
    results = pipeline.retrieve(query, top_k=2)
    print(f"  Query: '{query}'")
    for r in results:
        print(f"  [{r.score:.3f}] {r.text[:60]}...")

    # Full RAG answer
    print("\n── FULL RAG ANSWER ──")
    answer = pipeline.answer("What is the purpose of embeddings?")
    print(f"  Answer: {answer}")

    # 3. Show classmethod vs staticmethod
    print("\n── CLASSMETHOD vs STATICMETHOD ──")
    default_config = PipelineConfig()
    test_config = PipelineConfig.for_testing()
    valid_sizes = PipelineConfig.valid_chunk_sizes()

    print(f"  Default chunk size: {default_config.chunk_size}")
    print(f"  Test chunk size:    {test_config.chunk_size}")
    print(f"  Valid chunk sizes:  {valid_sizes}")

    print("\n✓ OOP patterns demo complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    demonstrate_oop()
