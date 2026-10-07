"""
Day 17 — Semantic Search Engine (Project 2)
=============================================
Complete semantic search system using sentence-transformers.
No API key required — uses free local embedding model.

Run: python 01_semantic_search_engine.py
Requires: pip install sentence-transformers numpy scikit-learn
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────
# EMBEDDER (free local model)
# ─────────────────────────────────────────────────────────

class LocalEmbedder:
    """Uses sentence-transformers for free local embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        print(f"Loading embedding model: {model_name}...")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dims = self._model.get_sentence_embedding_dimension()
            print(f"✓ Model loaded. Dimensions: {self._dims}")
        except ImportError:
            print("sentence-transformers not installed. Using mock embedder.")
            self._model = None
            self._dims = 4

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            return self._mock_embed(text)
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self._model is None:
            return np.array([self._mock_embed(t) for t in texts])
        return self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

    def _mock_embed(self, text: str) -> np.ndarray:
        seed = hash(text) % 10000
        v = np.array([math.sin(seed + i * 0.7) for i in range(self._dims)])
        return v / np.linalg.norm(v)


# ─────────────────────────────────────────────────────────
# DOCUMENT STORE
# ─────────────────────────────────────────────────────────

@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = field(default=None, repr=False)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class SemanticSearchEngine:
    """
    In-memory semantic search engine.
    Production would use ChromaDB, FAISS, or pgvector.
    """

    def __init__(self, embedder: LocalEmbedder) -> None:
        self._embedder = embedder
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def add_document(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        doc = Document(id=doc_id, text=text, metadata=metadata or {})
        doc.embedding = self._embedder.embed(text)
        self._documents.append(doc)
        self._rebuild_index()

    def add_documents_batch(self, documents: list[dict[str, Any]]) -> None:
        """Efficiently add many documents at once."""
        texts = [d["text"] for d in documents]
        embeddings = self._embedder.embed_batch(texts)

        for i, doc_data in enumerate(documents):
            doc = Document(
                id=doc_data["id"],
                text=doc_data["text"],
                metadata=doc_data.get("metadata", {}),
                embedding=embeddings[i],
            )
            self._documents.append(doc)

        self._rebuild_index()
        print(f"✓ Indexed {len(documents)} documents")

    def _rebuild_index(self) -> None:
        """Rebuild the embedding matrix from all documents."""
        if self._documents:
            self._embeddings = np.stack([d.embedding for d in self._documents])

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        if self._embeddings is None or len(self._documents) == 0:
            return []

        query_emb = self._embedder.embed(query)
        scores = np.dot(self._embeddings, query_emb)  # Already normalized
        top_indices = np.argsort(scores)[::-1]

        results = []
        rank = 1
        for idx in top_indices:
            doc = self._documents[idx]
            score = float(scores[idx])

            if score < min_score:
                break

            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(SearchResult(document=doc, score=score, rank=rank))
            rank += 1

            if rank > top_k:
                break

        return results

    def find_similar(self, doc_id: str, top_k: int = 5) -> list[SearchResult]:
        """Find documents similar to a given document."""
        doc = next((d for d in self._documents if d.id == doc_id), None)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found")
        query_emb = doc.embedding
        scores = np.dot(self._embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k + 1]  # +1 to skip self
        return [
            SearchResult(document=self._documents[idx], score=float(scores[idx]), rank=i + 1)
            for i, idx in enumerate(top_indices)
            if self._documents[idx].id != doc_id
        ][:top_k]

    @property
    def document_count(self) -> int:
        return len(self._documents)


# ─────────────────────────────────────────────────────────
# DEMO KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────

AI_DOCUMENTS = [
    {"id": "doc_001", "text": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It reduces hallucination by grounding LLM responses in retrieved facts.", "metadata": {"category": "rag", "difficulty": "intermediate"}},
    {"id": "doc_002", "text": "Embeddings are dense vector representations of text that capture semantic meaning. Similar texts produce similar vectors. Used in semantic search and RAG.", "metadata": {"category": "embeddings", "difficulty": "beginner"}},
    {"id": "doc_003", "text": "The Transformer architecture uses self-attention to process sequences in parallel. It was introduced in 'Attention Is All You Need' (2017) and powers all modern LLMs.", "metadata": {"category": "transformers", "difficulty": "advanced"}},
    {"id": "doc_004", "text": "Vector databases store high-dimensional embeddings and enable fast similarity search using approximate nearest neighbor algorithms like HNSW.", "metadata": {"category": "vector_db", "difficulty": "intermediate"}},
    {"id": "doc_005", "text": "Prompt engineering involves crafting inputs to guide LLM behavior. Key techniques: zero-shot, few-shot, chain-of-thought, and structured output prompting.", "metadata": {"category": "prompting", "difficulty": "beginner"}},
    {"id": "doc_006", "text": "AI Agents combine LLMs with tools, memory, and planning capabilities. They can execute multi-step tasks autonomously using the ReAct pattern.", "metadata": {"category": "agents", "difficulty": "advanced"}},
    {"id": "doc_007", "text": "LLM hallucination occurs when a model generates plausible-sounding but factually incorrect information. RAG and temperature control help mitigate this.", "metadata": {"category": "rag", "difficulty": "intermediate"}},
    {"id": "doc_008", "text": "FastAPI is a modern Python web framework for building AI backends. It supports async, automatic documentation, and Pydantic validation.", "metadata": {"category": "backend", "difficulty": "intermediate"}},
    {"id": "doc_009", "text": "Docker containerizes AI applications with all dependencies for reproducible deployments across environments.", "metadata": {"category": "infrastructure", "difficulty": "beginner"}},
    {"id": "doc_010", "text": "Fine-tuning adapts a pretrained LLM to specific tasks or styles using supervised training. Use for behavior changes, not knowledge injection (use RAG for that).", "metadata": {"category": "llm", "difficulty": "advanced"}},
    {"id": "doc_011", "text": "Chunking splits documents into smaller pieces for embedding. Strategies: fixed-size, sentence-aware, semantic chunking with overlap.", "metadata": {"category": "rag", "difficulty": "intermediate"}},
    {"id": "doc_012", "text": "Cosine similarity measures the angle between two vectors, not their magnitude. Used for embedding comparison because it's scale-invariant.", "metadata": {"category": "embeddings", "difficulty": "beginner"}},
]


# ─────────────────────────────────────────────────────────
# DEMONSTRATIONS
# ─────────────────────────────────────────────────────────

def demo_semantic_search(engine: SemanticSearchEngine) -> None:
    print("\n── SEMANTIC SEARCH DEMONSTRATION ──")

    queries = [
        "How does RAG reduce hallucination?",
        "What is the attention mechanism?",
        "How do I deploy a Python app?",
        "Compare different ways to create text representations",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        print(f"  {'Rank':<5} {'Score':>7} {'Category':<15} {'Preview'}")
        print("  " + "─" * 70)

        start = time.perf_counter()
        results = engine.search(query, top_k=3)
        elapsed = (time.perf_counter() - start) * 1000

        for r in results:
            preview = r.document.text[:50] + "..."
            cat = r.document.metadata.get("category", "?")
            print(f"  {r.rank:<5} {r.score:>7.4f} {cat:<15} {preview}")
        print(f"  Search time: {elapsed:.2f}ms")


def demo_filtered_search(engine: SemanticSearchEngine) -> None:
    print("\n── FILTERED SEARCH (METADATA FILTERING) ──")
    query = "How does retrieval work in AI?"

    for category in ["rag", "embeddings", "agents"]:
        results = engine.search(query, top_k=2, filter_metadata={"category": category})
        print(f"\n  Category filter: '{category}'")
        for r in results:
            print(f"  [{r.score:.4f}] {r.document.text[:70]}...")


def demo_find_similar(engine: SemanticSearchEngine) -> None:
    print("\n── FIND SIMILAR DOCUMENTS ──")
    doc_id = "doc_001"
    print(f"  Source: {doc_id} — {engine._documents[0].text[:60]}...")
    print("\n  Most similar documents:")

    similar = engine.find_similar(doc_id, top_k=3)
    for r in similar:
        print(f"  [{r.score:.4f}] {r.document.id}: {r.document.text[:60]}...")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 17 — SEMANTIC SEARCH ENGINE (PROJECT 2)       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Initialize
    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    engine = SemanticSearchEngine(embedder)

    # Index documents
    print(f"\nIndexing {len(AI_DOCUMENTS)} documents...")
    start = time.perf_counter()
    engine.add_documents_batch(AI_DOCUMENTS)
    elapsed = time.perf_counter() - start
    print(f"Indexing time: {elapsed:.2f}s")

    # Demos
    demo_semantic_search(engine)
    demo_filtered_search(engine)
    demo_find_similar(engine)

    print("\n✓ Semantic Search Engine complete!")
    print("\nKey patterns for production:")
    print("  • Batch embedding: much faster than one-by-one")
    print("  • Normalized vectors: inner product = cosine similarity (faster)")
    print("  • Metadata filtering: combine semantic + structured search")
    print("  • Lock embedding model: never change without re-indexing everything")
    print("  • In production: replace in-memory store with ChromaDB/FAISS/pgvector")


if __name__ == "__main__":
    main()
