"""
Day 02 — Async Python for AI Engineering
=========================================
Demonstrates async/await patterns used in production AI systems.
No external dependencies required — uses asyncio only.

Run: python 02_async_python.py
"""

import asyncio
import time
import random
import logging
from dataclasses import dataclass
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# SIMULATION HELPERS
# ─────────────────────────────────────────────────────────

async def simulate_llm_call(prompt: str, latency: float = 2.0) -> str:
    """Simulate an LLM API call with realistic latency."""
    await asyncio.sleep(latency)
    return f"[LLM Response to: {prompt[:30]}...]"


async def simulate_embedding(text: str) -> list[float]:
    """Simulate an embedding API call."""
    await asyncio.sleep(0.1)  # Embeddings are faster than completions
    return [hash(text + str(i)) % 100 / 100.0 for i in range(4)]


async def simulate_vector_search(query_vector: list[float]) -> list[dict[str, Any]]:
    """Simulate a vector database search."""
    await asyncio.sleep(0.05)  # DB queries are very fast
    return [
        {"id": f"doc_{i}", "score": 0.9 - i * 0.05, "text": f"Relevant chunk {i}"}
        for i in range(3)
    ]


# ─────────────────────────────────────────────────────────
# DEMO 1: SEQUENTIAL vs CONCURRENT
# ─────────────────────────────────────────────────────────

async def process_sequential(prompts: list[str]) -> list[str]:
    """Process prompts one at a time — slow."""
    results = []
    for prompt in prompts:
        result = await simulate_llm_call(prompt, latency=0.5)
        results.append(result)
    return results


async def process_concurrent(prompts: list[str]) -> list[str]:
    """Process all prompts concurrently — fast."""
    tasks = [simulate_llm_call(prompt, latency=0.5) for prompt in prompts]
    return await asyncio.gather(*tasks)


async def demo_sequential_vs_concurrent() -> None:
    """Show the performance difference between sequential and concurrent."""
    print("\n── SEQUENTIAL vs CONCURRENT LLM CALLS ──")

    prompts = [
        "What is RAG?",
        "Explain embeddings.",
        "What is an AI agent?",
        "What is chunking?",
        "What is a vector database?",
    ]

    # Sequential
    start = time.perf_counter()
    results_seq = await process_sequential(prompts)
    seq_time = time.perf_counter() - start
    print(f"Sequential: {len(results_seq)} results in {seq_time:.2f}s")

    # Concurrent
    start = time.perf_counter()
    results_con = await process_concurrent(prompts)
    con_time = time.perf_counter() - start
    print(f"Concurrent: {len(results_con)} results in {con_time:.2f}s")

    speedup = seq_time / con_time
    print(f"Speedup: {speedup:.1f}x faster with concurrent!")
    print("(At 5 requests × 0.5s = 2.5s sequential vs ~0.5s concurrent)")


# ─────────────────────────────────────────────────────────
# DEMO 2: COMPLETE RAG PIPELINE ASYNC
# ─────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    query: str
    retrieved_chunks: list[dict[str, Any]]
    llm_response: str
    total_time_ms: float


async def rag_pipeline_async(query: str) -> RAGResponse:
    """
    Full async RAG pipeline:
    1. Embed the query (async API call)
    2. Search vector DB (async DB call)
    3. Generate response with context (async LLM call)

    Steps 1+2 can overlap; step 3 depends on step 2.
    """
    start = time.perf_counter()

    # Step 1: Embed query (can start immediately)
    query_embedding = await simulate_embedding(query)

    # Step 2: Search vector DB (needs embedding from step 1)
    chunks = await simulate_vector_search(query_embedding)

    # Step 3: Build prompt and call LLM
    context = "\n".join(f"- {c['text']}" for c in chunks)
    prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}"
    llm_response = await simulate_llm_call(prompt, latency=0.8)

    total_time = (time.perf_counter() - start) * 1000

    return RAGResponse(
        query=query,
        retrieved_chunks=chunks,
        llm_response=llm_response,
        total_time_ms=total_time,
    )


async def demo_rag_pipeline() -> None:
    """Show async RAG pipeline."""
    print("\n── ASYNC RAG PIPELINE ──")

    result = await rag_pipeline_async("What is Retrieval-Augmented Generation?")

    print(f"Query: {result.query}")
    print(f"Retrieved {len(result.retrieved_chunks)} chunks:")
    for chunk in result.retrieved_chunks:
        print(f"  [{chunk['score']:.2f}] {chunk['text']}")
    print(f"Response: {result.llm_response}")
    print(f"Total time: {result.total_time_ms:.0f}ms")


# ─────────────────────────────────────────────────────────
# DEMO 3: BATCH EMBEDDING WITH RATE LIMIT
# ─────────────────────────────────────────────────────────

async def embed_with_rate_limit(
    texts: list[str],
    batch_size: int = 3,
    rate_limit_delay: float = 0.1,
) -> list[list[float]]:
    """
    Embed many texts respecting rate limits.

    Pattern: process in batches, delay between batches.
    In production: use token bucket or leaky bucket algorithm.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

        # Embed all texts in this batch concurrently
        batch_embeddings = await asyncio.gather(
            *[simulate_embedding(text) for text in batch]
        )
        all_embeddings.extend(batch_embeddings)

        # Rate limit delay between batches (not after last)
        if i + batch_size < len(texts):
            await asyncio.sleep(rate_limit_delay)

    return all_embeddings


async def demo_batch_embedding() -> None:
    """Show batch embedding with rate limiting."""
    print("\n── BATCH EMBEDDING WITH RATE LIMITING ──")

    texts = [f"Document {i}: {['RAG', 'Embeddings', 'LLMs', 'Agents', 'VectorDB'][i%5]}" for i in range(10)]

    start = time.perf_counter()
    embeddings = await embed_with_rate_limit(texts, batch_size=3)
    elapsed = time.perf_counter() - start

    print(f"Embedded {len(embeddings)} texts in {elapsed:.2f}s")
    print(f"Each embedding has {len(embeddings[0])} dimensions")


# ─────────────────────────────────────────────────────────
# DEMO 4: ASYNC CONTEXT MANAGER
# ─────────────────────────────────────────────────────────

class AsyncVectorDB:
    """Async context manager for vector DB connection."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._connected = False

    async def __aenter__(self) -> "AsyncVectorDB":
        logger.info(f"Connecting to vector DB at {self.url}...")
        await asyncio.sleep(0.05)  # Simulate connection time
        self._connected = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        logger.info("Closing vector DB connection...")
        await asyncio.sleep(0.01)  # Simulate cleanup
        self._connected = False
        return False  # Don't suppress exceptions

    async def search(self, query: str) -> list[str]:
        if not self._connected:
            raise RuntimeError("Not connected!")
        await asyncio.sleep(0.05)
        return [f"Result for '{query[:20]}'"]


async def demo_async_context_manager() -> None:
    """Show async context manager pattern."""
    print("\n── ASYNC CONTEXT MANAGER ──")

    async with AsyncVectorDB("chroma://localhost:8000") as db:
        results = await db.search("What is RAG?")
        print(f"Search results: {results}")
    # Connection automatically closed even if exception occurs


# ─────────────────────────────────────────────────────────
# DEMO 5: TIMEOUT HANDLING
# ─────────────────────────────────────────────────────────

async def slow_llm_call(prompt: str) -> str:
    """Simulate a very slow LLM call (might timeout)."""
    await asyncio.sleep(10)  # 10 second call
    return "Very slow response"


async def demo_timeout() -> None:
    """Show timeout handling for LLM calls."""
    print("\n── TIMEOUT HANDLING ──")

    # Without timeout — would wait forever
    print("Attempting LLM call with 1s timeout...")
    try:
        result = await asyncio.wait_for(slow_llm_call("test"), timeout=1.0)
        print(f"Got result: {result}")
    except asyncio.TimeoutError:
        print("✓ Timeout caught! LLM call exceeded 1 second limit.")
        print("  In production: return cached result or error response")


# ─────────────────────────────────────────────────────────
# DEMO 6: GATHER WITH ERROR HANDLING
# ─────────────────────────────────────────────────────────

async def flaky_llm_call(call_id: int) -> str:
    """Simulate an LLM call that sometimes fails."""
    await asyncio.sleep(0.1)
    if call_id % 3 == 0:  # Every 3rd call fails
        raise RuntimeError(f"Call {call_id} failed (rate limit simulated)")
    return f"Response {call_id}"


async def demo_gather_with_errors() -> None:
    """Show gather with return_exceptions=True for resilient processing."""
    print("\n── GATHER WITH ERROR HANDLING ──")

    tasks = [flaky_llm_call(i) for i in range(6)]

    # return_exceptions=True — don't cancel others if one fails
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = []
    failures = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failures.append((i, str(result)))
        else:
            successes.append(result)

    print(f"Successes: {len(successes)}: {successes}")
    print(f"Failures:  {len(failures)}: {failures}")
    print("→ Partial success is better than total failure in production")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

async def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 02 — ASYNC PYTHON FOR AI ENGINEERING          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    await demo_sequential_vs_concurrent()
    await demo_rag_pipeline()
    await demo_batch_embedding()
    await demo_async_context_manager()
    await demo_timeout()
    await demo_gather_with_errors()

    print("\n" + "="*60)
    print("✓ Day 02 async demo complete!")
    print("="*60)
    print("\nKey async patterns for AI engineering:")
    print("  • asyncio.gather() — run multiple LLM calls concurrently")
    print("  • await asyncio.sleep() — non-blocking delay (rate limits)")
    print("  • asyncio.wait_for() — timeout guard on LLM calls")
    print("  • gather(return_exceptions=True) — resilient batch processing")
    print("  • async context manager — resource management")
    print("  • Batch + rate limit — respect API rate limits")


if __name__ == "__main__":
    asyncio.run(main())
