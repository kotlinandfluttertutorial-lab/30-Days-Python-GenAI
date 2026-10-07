"""
Day 18 — Vector Databases: ChromaDB + FAISS
=============================================
Document Search Engine using both ChromaDB and FAISS.
Shows the same operations with both backends.

Run: python 01_vector_databases.py
Requires: pip install chromadb faiss-cpu sentence-transformers numpy
"""

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────
# SHARED EMBEDDER
# ─────────────────────────────────────────────────────────

class EmbedderBase:
    def __init__(self) -> None:
        self.dims = 384
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dims = self._model.get_sentence_embedding_dimension()
            print(f"✓ Real embedder loaded ({self.dims} dims)")
        except ImportError:
            self._model = None
            print("Using mock embedder (install sentence-transformers for real)")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model:
            return self._model.encode(texts, normalize_embeddings=True)
        vectors = np.array([
            [math.sin(hash(t + str(i)) % 10000 / 100) for i in range(self.dims)]
            for t in texts
        ], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


EMBEDDER = EmbedderBase()

# Knowledge base
DOCUMENTS = [
    {"id": "d01", "text": "RAG retrieves relevant documents before LLM generation", "cat": "rag"},
    {"id": "d02", "text": "Embeddings encode semantic meaning as dense vectors", "cat": "embeddings"},
    {"id": "d03", "text": "Transformers use self-attention for sequence processing", "cat": "architecture"},
    {"id": "d04", "text": "Vector databases enable fast similarity search at scale", "cat": "infrastructure"},
    {"id": "d05", "text": "Prompt engineering guides LLM behavior via instructions", "cat": "prompting"},
    {"id": "d06", "text": "AI Agents combine LLM with tools, memory, and planning", "cat": "agents"},
    {"id": "d07", "text": "Hallucination: LLM generating false but plausible content", "cat": "rag"},
    {"id": "d08", "text": "Chunking splits documents for embedding with overlap", "cat": "rag"},
    {"id": "d09", "text": "FastAPI builds async AI backends with Pydantic validation", "cat": "backend"},
    {"id": "d10", "text": "Fine-tuning adapts pretrained models; RAG injects knowledge", "cat": "llm"},
]


# ─────────────────────────────────────────────────────────
# CHROMADB BACKEND
# ─────────────────────────────────────────────────────────

def demo_chromadb() -> None:
    print("\n── CHROMADB DEMO ──")
    try:
        import chromadb
    except ImportError:
        print("  chromadb not installed: pip install chromadb")
        return

    client = chromadb.Client()  # In-memory

    # Delete if exists (for clean demo)
    try:
        client.delete_collection("ai_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="ai_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # Add documents with pre-computed embeddings
    texts = [d["text"] for d in DOCUMENTS]
    embeddings = EMBEDDER.encode(texts).tolist()

    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"category": d["cat"]} for d in DOCUMENTS],
    )
    print(f"  Indexed {collection.count()} documents")

    # Basic search
    query = "How does retrieval work in AI systems?"
    query_emb = EMBEDDER.encode([query]).tolist()

    start = time.perf_counter()
    results = collection.query(query_embeddings=query_emb, n_results=3)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\n  Query: '{query}'")
    print(f"  {'Score':>8}  Document")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        score = 1 - distance  # ChromaDB returns distance (lower=closer)
        text = results["documents"][0][i]
        print(f"  {score:>8.4f}  [{doc_id}] {text[:60]}...")
    print(f"  Time: {elapsed:.2f}ms")

    # Filtered search
    print(f"\n  Filtered search (category=rag):")
    results = collection.query(
        query_embeddings=query_emb,
        n_results=3,
        where={"category": "rag"},
    )
    for i in range(len(results["ids"][0])):
        text = results["documents"][0][i]
        cat = results["metadatas"][0][i].get("category")
        dist = results["distances"][0][i]
        print(f"    [{1-dist:.4f}] ({cat}) {text[:60]}...")

    # CRUD operations
    print(f"\n  CRUD operations:")
    collection.add(
        ids=["d11"],
        documents=["New document about deployment patterns"],
        embeddings=EMBEDDER.encode(["New document about deployment patterns"]).tolist(),
        metadatas=[{"category": "backend"}],
    )
    print(f"  After add: {collection.count()} documents")

    collection.delete(ids=["d11"])
    print(f"  After delete: {collection.count()} documents")


# ─────────────────────────────────────────────────────────
# FAISS BACKEND
# ─────────────────────────────────────────────────────────

def demo_faiss() -> None:
    print("\n── FAISS DEMO ──")
    try:
        import faiss
    except ImportError:
        print("  faiss-cpu not installed: pip install faiss-cpu")
        return

    dims = EMBEDDER.dims
    texts = [d["text"] for d in DOCUMENTS]
    embeddings = EMBEDDER.encode(texts).astype(np.float32)

    # Index: Inner Product (for normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(dims)
    index.add(embeddings)
    print(f"  Indexed {index.ntotal} vectors, {dims} dims")

    # Search
    query = "What is the retrieval mechanism in RAG?"
    query_emb = EMBEDDER.encode([query]).astype(np.float32)

    start = time.perf_counter()
    distances, indices = index.search(query_emb, k=3)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\n  Query: '{query}'")
    print(f"  {'Score':>8}  Document")
    for score, idx in zip(distances[0], indices[0]):
        text = texts[idx]
        print(f"  {score:>8.4f}  [{DOCUMENTS[idx]['id']}] {text[:60]}...")
    print(f"  Time: {elapsed:.2f}ms (FAISS is extremely fast!)")

    # Batch search
    print(f"\n  Batch search (3 queries at once):")
    queries = [
        "explain transformers architecture",
        "how to deploy AI models",
        "prompt injection attacks",
    ]
    query_embs = EMBEDDER.encode(queries).astype(np.float32)
    distances, indices = index.search(query_embs, k=2)

    for i, (q, dists, idxs) in enumerate(zip(queries, distances, indices)):
        print(f"  Q: '{q[:40]}'")
        for score, idx in zip(dists, idxs):
            print(f"    [{score:.4f}] {texts[idx][:55]}...")


# ─────────────────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────────────────

def compare_backends() -> None:
    print("\n── VECTOR DB COMPARISON ──")
    print(f"  {'Backend':<20} {'Use Case':<30} {'Scale':<15} {'Cost'}")
    print("  " + "─" * 80)
    comparison = [
        ("ChromaDB",    "Development, RAG prototyping", "<1M vectors",  "Free"),
        ("FAISS",       "High perf, in-process",        "<100M vectors", "Free"),
        ("pgvector",    "Existing Postgres stack",       "<10M vectors",  "Free"),
        ("Pinecone",    "Managed production",            "Billions",      "Paid"),
        ("Qdrant",      "Open-source production",        "Billions",      "Free/Paid"),
        ("Weaviate",    "Hybrid search, knowledge",      "Billions",      "Free/Paid"),
    ]
    for name, use, scale, cost in comparison:
        print(f"  {name:<20} {use:<30} {scale:<15} {cost}")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 18 — VECTOR DATABASES                         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_chromadb()
    demo_faiss()
    compare_backends()

    print("\n✓ Day 18 Vector Databases demo complete!")
    print("\nKey takeaways:")
    print("  • ChromaDB: easiest to use, auto-embedding, great for RAG prototyping")
    print("  • FAISS: fastest in-process search, no server overhead")
    print("  • pgvector: if you already use Postgres, add vector search without new service")
    print("  • Metadata filtering: always add document metadata (source, date, category)")
    print("  • HNSW algorithm: fast approximate search, high accuracy, high memory")


if __name__ == "__main__":
    main()
