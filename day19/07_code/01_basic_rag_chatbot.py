"""
Day 19 — Basic RAG Chatbot
============================
Complete RAG pipeline: ingest → embed → store → retrieve → generate.
Uses ChromaDB for vector storage and free local embeddings.
Works without LLM API key (shows prompts), or with one (shows answers).

Run: python 01_basic_rag_chatbot.py
Requires: pip install chromadb sentence-transformers
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────────────────

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.dims: int = self._model.get_sentence_embedding_dimension()
            self._is_real = True
        except ImportError:
            self._model = None
            self.dims = 64
            self._is_real = False

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._is_real:
            return self._model.encode(texts, normalize_embeddings=True).tolist()
        # Mock: deterministic hash-based
        result = []
        for text in texts:
            v = [math.sin(hash(text + str(i)) % 10000 / 100) for i in range(self.dims)]
            norm = math.sqrt(sum(x**2 for x in v))
            result.append([x / norm for x in v])
        return result


# ─────────────────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    return " ".join(text.split())


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    min_chunk_len: int = 50,
) -> list[str]:
    """Sentence-aware chunking with overlap."""
    text = clean_text(text)
    if len(text) < min_chunk_len:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", "! ", "? ", "\n", " "]:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos > 0:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_len:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap

    return chunks


# ─────────────────────────────────────────────────────────
# VECTOR STORE (ChromaDB)
# ─────────────────────────────────────────────────────────

class RAGVectorStore:
    """ChromaDB-backed vector store for RAG."""

    def __init__(self, collection_name: str = "rag_documents") -> None:
        try:
            import chromadb
            self._client = chromadb.Client()
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._is_real = True
        except ImportError:
            self._is_real = False
            self._store: list[dict] = []

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add pre-computed chunks with embeddings."""
        if self._is_real:
            self._collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=[c["embedding"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
        else:
            self._store.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Return top-K similar chunks."""
        if self._is_real:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            return [
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
                for i in range(len(results["ids"][0]))
            ]
        else:
            # Cosine similarity fallback
            def cosine(a: list[float], b: list[float]) -> float:
                d = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x**2 for x in a))
                nb = math.sqrt(sum(x**2 for x in b))
                return d / (na * nb) if na and nb else 0.0

            scored = sorted(
                self._store,
                key=lambda c: cosine(query_embedding, c["embedding"]),
                reverse=True
            )
            return [
                {"id": c["id"], "text": c["text"],
                 "score": cosine(query_embedding, c["embedding"]),
                 "metadata": c["metadata"]}
                for c in scored[:top_k]
            ]

    @property
    def count(self) -> int:
        if self._is_real:
            return self._collection.count()
        return len(self._store)


# ─────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self) -> None:
        self._provider = self._detect()

    def _detect(self) -> str:
        for key, provider in [("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"),
                               ("ANTHROPIC_API_KEY", "anthropic")]:
            if os.getenv(key):
                return provider
        return "mock"

    def complete(self, messages: list[dict], max_tokens: int = 600) -> str:
        if self._provider == "mock":
            ctx = next((m["content"][:200] for m in messages if m["role"] == "user"), "")
            return f"[Mock answer based on context: {ctx[:100]}...]"

        try:
            from openai import OpenAI
            if self._provider == "groq":
                client = OpenAI(
                    api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1",
                )
                model = "llama-3.1-8b-instant"
            else:
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                model = "gpt-4o-mini"

            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM Error: {e}]"


# ─────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────

class BasicRAGChatbot:
    """Complete RAG pipeline: ingest → retrieve → generate."""

    SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.
Answer questions ONLY using the provided context.
If the context doesn't contain the answer, say "I don't have that information in my knowledge base."
Always cite your sources using [Source: document_name]."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = RAGVectorStore()
        self.llm = LLMClient()
        self._chunk_count = 0

    def ingest_text(self, text: str, source: str, metadata: dict | None = None) -> int:
        """Ingest a text document into the RAG system."""
        chunks = chunk_text(text)
        if not chunks:
            return 0

        # Embed all chunks at once
        embeddings = self.embedder.embed_batch([c for c in chunks])

        chunk_dicts = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{source}_chunk_{self._chunk_count + i}"
            chunk_dicts.append({
                "id": chunk_id,
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    **(metadata or {}),
                },
            })

        self.vector_store.add_chunks(chunk_dicts)
        self._chunk_count += len(chunks)
        return len(chunks)

    def ingest_documents(self, documents: list[dict[str, Any]]) -> None:
        """Ingest multiple documents."""
        total = 0
        for doc in documents:
            n = self.ingest_text(doc["text"], doc["source"], doc.get("metadata"))
            total += n
            print(f"  ✓ '{doc['source']}': {n} chunks")
        print(f"\n  Total chunks indexed: {total}")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-K relevant chunks for a query."""
        query_emb = self.embedder.embed(query)
        return self.vector_store.search(query_emb, top_k=top_k)

    def answer(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """Full RAG pipeline: retrieve + generate."""
        start = time.perf_counter()

        # Retrieve
        chunks = self.retrieve(query, top_k)

        # Build context
        context_parts = []
        for chunk in chunks:
            source = chunk["metadata"].get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        # Generate
        answer = self.llm.complete(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "query": query,
            "answer": answer,
            "sources": [c["metadata"].get("source") for c in chunks],
            "top_scores": [round(c["score"], 4) for c in chunks],
            "chunks_retrieved": len(chunks),
            "latency_ms": round(elapsed_ms, 1),
        }


# ─────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {
        "source": "rag_guide",
        "text": """RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses
by retrieving relevant documents before generating an answer. It was introduced to solve
two major problems: hallucination and knowledge cutoffs. The RAG pipeline has two phases:
ingestion (load, clean, chunk, embed, store) and query (embed query, search, retrieve,
build prompt, generate). RAG is preferred over fine-tuning for factual Q&A because it
supports instant updates, enables source citations, and works without GPU infrastructure.""",
    },
    {
        "source": "embeddings_guide",
        "text": """Embeddings are dense vector representations that capture semantic meaning.
The embedding model converts text into a vector of 384 to 3072 floating-point numbers.
Similar texts produce similar vectors, measured by cosine similarity. Critical rule:
always use the same embedding model for both indexing and querying. Changing models
requires re-indexing all documents. Free options include all-MiniLM-L6-v2 (384 dims)
from sentence-transformers. Paid options include OpenAI text-embedding-3-small (1536 dims).""",
    },
    {
        "source": "chunking_best_practices",
        "text": """Chunking strategies determine retrieval quality. Fixed-size chunking (500 chars)
is simple but splits sentences mid-way. Recursive character splitting tries natural
boundaries: paragraphs, then sentences, then words. Always add 50-100 character overlap
between chunks to prevent information loss at boundaries. Larger chunks (1024 tokens)
provide more context but reduce retrieval precision. Smaller chunks (128 tokens) retrieve
precisely but may miss surrounding context. The recommended default is 512 tokens with
50-token overlap.""",
    },
    {
        "source": "vector_db_overview",
        "text": """Vector databases store high-dimensional embeddings and enable fast similarity search.
ChromaDB is ideal for development: embedded, no server, auto-embedding support.
FAISS (Facebook AI Similarity Search) provides the fastest in-process search.
pgvector extends PostgreSQL with vector types, perfect for teams already using Postgres.
Production deployments at scale use Pinecone (managed) or Qdrant (open-source).
All vector databases use Approximate Nearest Neighbor (ANN) algorithms like HNSW
for sub-linear search time complexity.""",
    },
    {
        "source": "company_policy",
        "text": """Company AI Usage Policy (v2.0, January 2024):
All AI-generated content must be reviewed by a human before publication.
Employees must not input confidential customer data into external LLM APIs.
The approved AI tools are: internal RAG system, GitHub Copilot for coding.
LLM API costs are charged to department budgets. Monthly spend limit: $500 per team.
All AI interactions are logged for compliance. Data retention: 90 days.
Questions about AI policy: contact ai-governance@company.com""",
    },
]


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 19 — BASIC RAG CHATBOT (PROJECT 3 BEGINS)     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Initialize RAG system
    rag = BasicRAGChatbot()
    print(f"\nEmbedding model: {rag.embedder.dims} dimensions")

    # Ingest knowledge base
    print(f"\nIngesting {len(KNOWLEDGE_BASE)} documents...")
    rag.ingest_documents(KNOWLEDGE_BASE)
    print(f"Vector store size: {rag.vector_store.count} chunks")

    # Test queries
    print("\n" + "="*60)
    print("QUESTION & ANSWER DEMO")
    print("="*60)

    test_queries = [
        "What problem does RAG solve?",
        "What is the recommended chunk size for RAG?",
        "Can I use confidential customer data in external LLM APIs?",
        "Which embedding model should I use for development?",
        "What is the company's monthly AI spending limit per team?",
        "What is the best programming language for web development?",  # Not in KB
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        result = rag.answer(query)
        print(f"A: {result['answer'][:200]}...")
        print(f"   Sources: {result['sources']} | Scores: {result['top_scores']} | {result['latency_ms']}ms")

    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Type questions to ask the RAG system (or 'quit' to exit):\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query or query.lower() == "quit":
                break
            result = rag.answer(query)
            print(f"RAG: {result['answer']}")
            print(f"     [Sources: {result['sources']}, Scores: {result['top_scores']}]\n")
        except (KeyboardInterrupt, EOFError):
            break

    print("\n✓ Day 19 Basic RAG Chatbot complete!")
    print("\nRAG pipeline summary:")
    print("  1. INGEST: clean → chunk → embed → store in vector DB")
    print("  2. QUERY: embed → search → retrieve → build prompt → generate → cite")
    print("  Key: same embedding model for indexing AND querying!")


if __name__ == "__main__":
    main()
