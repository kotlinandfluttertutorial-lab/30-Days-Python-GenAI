# Day 02 — Deep Dive
## Senior AI Engineer Python Patterns

---

## Deep Dive 1: Why Async is Non-Negotiable for AI Backends

### The Problem with Synchronous LLM Code

Every LLM call takes 1–10 seconds. A synchronous FastAPI endpoint that calls an LLM blocks the entire server thread during that wait.

```
SYNCHRONOUS SERVER (1 worker thread):

Time  Request1  Request2  Request3
0s    [START]
1s    [WAIT..] 
2s    [WAIT..]   (Request 2 arrives, queued)
3s    [DONE ]    [START]   (Request 3 arrives, queued)
4s              [WAIT..]
5s              [DONE ]    [START]
6s                         [WAIT..]
7s                         [DONE ]

Total time for 3 requests: 7 seconds
Throughput: ~0.4 requests/second
```

```
ASYNC SERVER (1 event loop, many concurrent tasks):

Time  Request1  Request2  Request3
0s    [START]   [START]   [START]  (all start immediately)
1s    [WAIT..]  [WAIT..]  [WAIT..]
2s    [WAIT..]  [WAIT..]  [WAIT..]
3s    [DONE ]   [DONE ]   [DONE ]

Total time for 3 requests: ~3 seconds
Throughput: ~1 request/second
```

This is why FastAPI is async-first. This is why the OpenAI SDK has an `AsyncOpenAI` client.

### The Mental Model: Event Loop

```python
# The Python event loop is a scheduler for coroutines
# A coroutine is a function that can pause and resume

import asyncio

async def coroutine_a() -> None:
    print("A: starting")
    await asyncio.sleep(2)   # ← Pause here, let others run
    print("A: done")

async def coroutine_b() -> None:
    print("B: starting")
    await asyncio.sleep(1)   # ← Pause here, let others run
    print("B: done")

async def main() -> None:
    # Run both concurrently — each pauses at await, other runs
    await asyncio.gather(coroutine_a(), coroutine_b())

asyncio.run(main())
# Output:
# A: starting
# B: starting
# B: done      (B finishes first — only 1s wait)
# A: done      (A finishes at ~2s)
# Total time: ~2s (not 3s)
```

### What `await` Actually Does

```python
# await = "pause THIS coroutine and let the event loop run others"
# Only works with awaitable objects: coroutines, asyncio.Task, asyncio.Future

# WRONG — sync function inside async — BLOCKS the event loop!
async def bad_handler() -> str:
    result = requests.get("https://api.openai.com/...")  # BLOCKS
    return result.json()

# RIGHT — use async HTTP library
import httpx

async def good_handler() -> str:
    async with httpx.AsyncClient() as client:
        result = await client.get("https://api.openai.com/...")  # NON-BLOCKING
    return result.json()
```

### The Concurrency Patterns

```python
# Pattern 1: gather — all tasks run concurrently, wait for ALL
results = await asyncio.gather(task1(), task2(), task3())

# Pattern 2: gather with error handling
results = await asyncio.gather(
    task1(), task2(), task3(),
    return_exceptions=True  # Don't cancel others if one fails
)
for result in results:
    if isinstance(result, Exception):
        handle_error(result)

# Pattern 3: as_completed — process results as they arrive
async def stream_results(tasks):
    for coro in asyncio.as_completed(tasks):
        result = await coro
        yield result  # Process each as it completes

# Pattern 4: timeout — don't wait forever for LLM
try:
    result = await asyncio.wait_for(call_llm(prompt), timeout=30.0)
except asyncio.TimeoutError:
    raise LLMTimeoutError("LLM call exceeded 30 second timeout")
```

---

## Deep Dive 2: Generator Protocol — How Streaming Really Works

Understanding generators is critical because streaming LLM responses are generators.

### The Generator Protocol

```python
# A generator function returns a generator object
# The generator object implements the iterator protocol:
# __iter__() → returns self
# __next__() → returns next value or raises StopIteration

def my_generator():
    print("START")
    yield 1           # Pause, return 1
    print("AFTER 1")
    yield 2           # Pause, return 2
    print("AFTER 2")
    yield 3           # Pause, return 3
    print("DONE")
    # Function ends → StopIteration raised automatically

gen = my_generator()
print(next(gen))  # "START", returns 1
print(next(gen))  # "AFTER 1", returns 2
print(next(gen))  # "AFTER 2", returns 3
# next(gen)       # "DONE", raises StopIteration

# for loop handles StopIteration automatically
for value in my_generator():
    print(value)
```

### Streaming LLM Response — What Actually Happens

```python
from collections.abc import Generator

def stream_chat_response(prompt: str) -> Generator[str, None, None]:
    """
    Stream response from OpenAI API token by token.
    
    The API returns a stream object — each iteration yields
    one chunk. Each chunk may have 0-1 tokens.
    We yield only non-empty tokens.
    """
    from openai import OpenAI
    client = OpenAI()
    
    # This creates a stream — data flows in as LLM generates
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # Key parameter
    )
    
    accumulated = ""
    for chunk in stream:
        # Each chunk has delta content (the new token(s))
        delta = chunk.choices[0].delta
        
        if delta.content:        # Some chunks have no content
            token = delta.content
            accumulated += token
            yield token          # Yield to caller immediately
    
    # Generator completes — accumulated has full response
    # You could yield a final summary object here if needed

# Consumer: display tokens as they arrive
for token in stream_chat_response("Explain RAG in 3 sentences"):
    print(token, end="", flush=True)  # flush=True shows each token immediately
print()  # Final newline
```

### Generator Pipeline — Composing Data Transformations

```python
from collections.abc import Generator, Iterator
from pathlib import Path

# Build a processing pipeline with generators
# Each stage is lazy — only processes what's needed

def read_files(directory: str) -> Generator[tuple[str, str], None, None]:
    """Read files lazily — one at a time."""
    for path in Path(directory).glob("*.txt"):
        with open(path, encoding="utf-8") as f:
            yield str(path), f.read()

def clean_text(documents: Iterator[tuple[str, str]]) -> Generator[tuple[str, str], None, None]:
    """Clean each document."""
    for path, text in documents:
        cleaned = " ".join(text.split())  # normalize whitespace
        cleaned = cleaned.lower()
        if len(cleaned) > 50:  # Skip very short docs
            yield path, cleaned

def chunk_documents(
    documents: Iterator[tuple[str, str]],
    chunk_size: int = 512,
) -> Generator[dict, None, None]:
    """Chunk each document."""
    chunk_id = 0
    for path, text in documents:
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                yield {
                    "id": f"chunk_{chunk_id}",
                    "source": path,
                    "text": chunk,
                    "chunk_index": chunk_id,
                }
                chunk_id += 1

# Build the pipeline — nothing runs until we iterate
pipeline = chunk_documents(
    clean_text(
        read_files("/path/to/documents")
    )
)

# Consume the pipeline
for chunk in pipeline:
    embed_and_store(chunk)  # Only one chunk in memory at a time!
```

---

## Deep Dive 3: Dataclasses vs Pydantic vs TypedDict

You will use all three. Know when to use each.

```python
from dataclasses import dataclass, field
from typing import TypedDict
from pydantic import BaseModel, field_validator

# ── 1. Dataclass ──────────────────────────────────────────
# Use for: internal data structures, simple value objects
# NOT for: API input/output validation

@dataclass
class LLMMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=lambda: __import__("time").time())

@dataclass
class RetrievalResult:
    document_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)

msg = LLMMessage(role="user", content="What is RAG?")
print(msg.role)     # "user"
# Dataclass does NOT validate types — you can pass an int for role

# ── 2. TypedDict ──────────────────────────────────────────
# Use for: dict type hints (especially API response structures)
# NOT for: instantiation/construction (it's just a type hint)

class MessageDict(TypedDict):
    role: str
    content: str

def format_message(role: str, content: str) -> MessageDict:
    return {"role": role, "content": content}  # Type checker validates this

# ── 3. Pydantic BaseModel ─────────────────────────────────
# Use for: FastAPI request/response, data validation, JSON parsing
# Provides: runtime type validation, JSON serialization, error messages

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()

# Pydantic validates at construction
try:
    req = ChatRequest(message="", temperature=3.0)
except Exception as e:
    print(e)  # Validation errors for both fields

# Valid request
req = ChatRequest(message="What is RAG?")
print(req.model)  # "gpt-4o" (default)
print(req.model_dump())  # dict representation
print(req.model_dump_json())  # JSON string
```

---

## Deep Dive 4: Exception Hierarchy Design for AI Systems

Design your exceptions to communicate clearly what failed and how to recover.

```python
# Exception hierarchy for an AI service

class AIServiceError(Exception):
    """Base for all AI service errors."""
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMError(AIServiceError):
    """LLM provider errors."""
    pass

class LLMRateLimitError(LLMError):
    """Hit rate limit — wait and retry."""
    def __init__(self, retry_after: float = 60.0) -> None:
        super().__init__(f"Rate limited. Retry after {retry_after}s", retryable=True)
        self.retry_after = retry_after

class LLMAuthError(LLMError):
    """Invalid API key — do not retry."""
    def __init__(self) -> None:
        super().__init__("Authentication failed. Check API key.", retryable=False)

class LLMContextLengthError(LLMError):
    """Prompt exceeds context window — reduce input."""
    def __init__(self, max_tokens: int, actual_tokens: int) -> None:
        super().__init__(
            f"Context too long: {actual_tokens} tokens > {max_tokens} limit",
            retryable=False
        )
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


class RAGError(AIServiceError):
    """RAG pipeline errors."""
    pass

class RetrievalError(RAGError):
    """Vector search failed."""
    pass

class EmbeddingError(RAGError):
    """Embedding generation failed."""
    pass

class IngestionError(RAGError):
    """Document ingestion failed."""
    def __init__(self, document_id: str, reason: str) -> None:
        super().__init__(f"Failed to ingest document {document_id}: {reason}")
        self.document_id = document_id


# Usage: handle based on type, not message text
def handle_llm_error(error: AIServiceError) -> None:
    if isinstance(error, LLMRateLimitError):
        time.sleep(error.retry_after)
        retry_request()
    elif isinstance(error, LLMAuthError):
        alert_ops_team()
        raise  # Can't recover programmatically
    elif isinstance(error, LLMContextLengthError):
        reduce_context_and_retry(error.max_tokens)
    elif error.retryable:
        retry_with_backoff()
    else:
        log_and_fail(error)
```

---

## Deep Dive 5: The Python Patterns Used in Every LLM Framework

If you study LangChain, LlamaIndex, or any LLM framework, you'll see these patterns everywhere. Understanding them lets you extend any framework.

### Pattern: Callable Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Embedder(Protocol):
    """Any object that can embed text."""
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

# Now you can write functions that accept ANY embedder:
def build_index(documents: list[str], embedder: Embedder) -> None:
    embeddings = embedder.embed_batch(documents)
    # Works with OpenAI embedder, HuggingFace embedder, any embedder

# Any class with these methods satisfies the protocol
class OpenAIEmbedder:
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

isinstance(OpenAIEmbedder(), Embedder)  # True
```

### Pattern: Builder / Fluent Interface

```python
# Used heavily in LangChain-style chain construction
class PromptBuilder:
    def __init__(self) -> None:
        self._system: str = ""
        self._examples: list[tuple[str, str]] = []
        self._instructions: list[str] = []
    
    def system(self, message: str) -> "PromptBuilder":
        self._system = message
        return self  # Return self for chaining
    
    def example(self, user: str, assistant: str) -> "PromptBuilder":
        self._examples.append((user, assistant))
        return self
    
    def instruction(self, text: str) -> "PromptBuilder":
        self._instructions.append(text)
        return self
    
    def build(self) -> str:
        parts = [self._system]
        for inst in self._instructions:
            parts.append(f"Rule: {inst}")
        for user_ex, asst_ex in self._examples:
            parts.append(f"Example User: {user_ex}")
            parts.append(f"Example Assistant: {asst_ex}")
        return "\n\n".join(filter(None, parts))

# Fluent usage
prompt = (
    PromptBuilder()
    .system("You are a helpful AI assistant.")
    .instruction("Always cite your sources.")
    .instruction("Answer in 2-3 sentences.")
    .example("What is RAG?", "RAG stands for Retrieval-Augmented Generation...")
    .build()
)
```
