# Day 21 — Concepts
## Production RAG

---

## 1. Production Architecture

```
ASYNC INGESTION PIPELINE:
  File Upload → Queue (Redis) → Worker → Chunk → Embed → Store
                                         ↓
                                    Error handling
                                    Retry logic
                                    Progress tracking

ASYNC QUERY PIPELINE:
  HTTP Request → FastAPI → Cache Check → Vector Search
                                ↓               ↓
                           Cache Hit      Context Build
                                ↓               ↓
                           Return cached    LLM Call
                                ↓               ↓
                           Response        Cache Store → Response
```

---

## 2. Async Ingestion Pipeline

```python
import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class IngestionJob:
    job_id: str
    source: str
    text: str
    metadata: dict[str, Any]
    status: str = "pending"
    error: str | None = None

class AsyncIngestionPipeline:
    """
    Async document ingestion with:
    - Parallel embedding
    - Retry on failure
    - Deduplication via content hash
    - Progress tracking
    """

    def __init__(self, embedder, vector_store, batch_size: int = 50) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size
        self._processed_hashes: set[str] = set()  # Deduplication

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def ingest_batch(self, documents: list[dict]) -> dict[str, int]:
        """Ingest a batch of documents concurrently."""
        results = {"ingested": 0, "skipped_duplicate": 0, "errors": 0}

        for doc in documents:
            doc_hash = self._content_hash(doc["text"])
            if doc_hash in self._processed_hashes:
                results["skipped_duplicate"] += 1
                continue

            try:
                await self._process_document(doc)
                self._processed_hashes.add(doc_hash)
                results["ingested"] += 1
            except Exception as e:
                logger.error(f"Failed to ingest {doc.get('source')}: {e}")
                results["errors"] += 1

        return results

    async def _process_document(self, doc: dict) -> None:
        """Process a single document asynchronously."""
        chunks = chunk_text(doc["text"])

        # Embed in batches
        all_chunks = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            embeddings = await asyncio.to_thread(
                self.embedder.embed_batch, batch
            )
            for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                all_chunks.append({
                    "id": f"{doc['source']}_{i + j}",
                    "text": chunk,
                    "embedding": emb,
                    "metadata": {"source": doc["source"], **doc.get("metadata", {})},
                })

        # Store
        await asyncio.to_thread(self.vector_store.add_chunks, all_chunks)
```

---

## 3. Caching Strategy

```python
import json
import hashlib
from typing import Any

import redis

class RAGCache:
    """Redis-based cache for RAG query results."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600) -> None:
        try:
            self._redis = redis.from_url(redis_url)
            self._redis.ping()
            self._enabled = True
        except Exception:
            self._enabled = False

    def _cache_key(self, query: str) -> str:
        return f"rag:query:{hashlib.sha256(query.encode()).hexdigest()[:16]}"

    def get(self, query: str) -> dict | None:
        if not self._enabled:
            return None
        try:
            raw = self._redis.get(self._cache_key(query))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, query: str, result: dict) -> None:
        if not self._enabled:
            return
        try:
            self._redis.setex(
                self._cache_key(query),
                self.ttl,
                json.dumps(result),
            )
        except Exception:
            pass  # Cache failure is non-fatal

    def invalidate(self, pattern: str = "rag:query:*") -> int:
        """Invalidate all cached queries (e.g., after document update)."""
        if not self._enabled:
            return 0
        keys = self._redis.keys(pattern)
        return self._redis.delete(*keys) if keys else 0
```

---

## 4. Retry with Exponential Backoff

```python
import asyncio
import functools
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Async retry decorator with exponential backoff."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = base_delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} attempt {attempt+1} failed: {e}. Retry in {wait}s")
                    await asyncio.sleep(wait)
        return wrapper  # type: ignore
    return decorator

@retry_async(max_attempts=3, base_delay=1.0, exceptions=(Exception,))
async def call_llm_with_retry(messages: list[dict]) -> str:
    return await llm_client.complete_async(messages)
```

---

## 5. Observability

```python
import time
import uuid
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RAGTrace:
    """Full trace of a RAG request."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    cache_hit: bool = False
    chunks_retrieved: int = 0
    retrieval_latency_ms: float = 0
    llm_latency_ms: float = 0
    total_latency_ms: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0
    error: str | None = None

    def log(self) -> None:
        logger.info(
            "RAG request",
            extra={
                "trace_id": self.trace_id,
                "cache_hit": self.cache_hit,
                "retrieval_ms": self.retrieval_latency_ms,
                "llm_ms": self.llm_latency_ms,
                "total_ms": self.total_latency_ms,
                "input_tokens": self.input_tokens,
                "cost_usd": f"{self.cost_usd:.6f}",
                "error": self.error,
            }
        )
```

---

## 6. Rate Limiting

```python
import asyncio
import time
from collections import defaultdict

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = requests_per_minute
        self.window = 60  # seconds
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        window_start = now - self.window
        timestamps = [t for t in self._timestamps[user_id] if t > window_start]
        self._timestamps[user_id] = timestamps

        if len(timestamps) >= self.rpm:
            return False

        self._timestamps[user_id].append(now)
        return True

    def tokens_remaining(self, user_id: str) -> int:
        now = time.time()
        window_start = now - self.window
        recent = [t for t in self._timestamps[user_id] if t > window_start]
        return max(0, self.rpm - len(recent))
```
