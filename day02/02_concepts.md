# Day 02 — Concepts
## Python for AI Engineering: Every Pattern You Need

---

## 1. Python Data Types for AI Engineering

### Strings — The Primary Data in AI Systems

```python
# Basic string operations used in AI
text = "The quick brown fox"

# Slicing — used in chunking algorithms
chunk = text[0:10]          # "The quick "
last_100 = text[-100:]      # last 100 chars

# Methods critical for text processing
text.lower()                # normalize before embedding
text.strip()                # remove whitespace from documents
text.split()                # simple tokenization
text.split("\n")            # split by paragraphs
"\n".join(["a", "b", "c"]) # join chunks

# f-strings — used in EVERY prompt template
model = "gpt-4o"
tokens = 1500
prompt = f"Model: {model}, Tokens: {tokens}"

# Multi-line strings for prompts
system_prompt = """
You are an expert AI assistant.
Answer only from the provided context.
If unsure, say "I don't know."
""".strip()  # Always strip multi-line prompts

# String formatting for prompt templates
template = "Answer the question: {question}\nContext: {context}"
filled = template.format(question="What is RAG?", context="RAG is...")
```

### Lists — Batch Processing in AI

```python
from typing import Any

# Used everywhere: document chunks, embeddings, messages
chunks: list[str] = ["chunk 1", "chunk 2", "chunk 3"]
embeddings: list[list[float]] = [[0.1, 0.2], [0.3, 0.4]]
messages: list[dict[str, str]] = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What is RAG?"},
]

# List operations critical for AI
chunks.append("new chunk")           # add document chunk
chunks.extend(["chunk 4", "chunk 5"]) # add many chunks
top_3 = chunks[:3]                    # get top-k results
reversed_history = messages[::-1]     # reverse conversation

# Sorting with key function — sort by similarity score
results = [("doc1", 0.92), ("doc2", 0.78), ("doc3", 0.95)]
results.sort(key=lambda x: x[1], reverse=True)  # sort by score desc

# Filter with comprehension
long_chunks = [c for c in chunks if len(c) > 50]
```

### Dictionaries — JSON Objects in Python

```python
# LLM API messages are dicts
message: dict[str, str] = {"role": "user", "content": "Hello"}

# Metadata for vector DB documents
metadata: dict[str, Any] = {
    "source": "document.pdf",
    "page": 3,
    "category": "finance",
    "created_at": "2024-01-15",
}

# Dict operations
metadata.get("page", 1)         # safe get with default
metadata.keys()                  # ["source", "page", ...]
metadata.values()                # ["document.pdf", 3, ...]
metadata.items()                 # [("source", "doc.pdf"), ...]
metadata.update({"updated": True})  # merge
"page" in metadata               # True — membership test

# Nested dicts — common in API responses
api_response = {
    "id": "chatcmpl-abc123",
    "choices": [
        {
            "message": {"role": "assistant", "content": "RAG stands for..."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 150, "completion_tokens": 200},
}
# Safely extract nested values
content = api_response["choices"][0]["message"]["content"]
total_tokens = api_response["usage"]["prompt_tokens"] + api_response["usage"]["completion_tokens"]
```

### Tuples — Immutable Data in AI

```python
# Use tuples for data that shouldn't change
ModelConfig = tuple[str, float, int]  # (model_name, temperature, max_tokens)
config: ModelConfig = ("gpt-4o", 0.7, 1024)

# Named tuples for structured data
from typing import NamedTuple

class SearchResult(NamedTuple):
    document_id: str
    score: float
    text: str

result = SearchResult(document_id="doc_001", score=0.92, text="RAG is...")
print(result.score)   # 0.92 — named access
print(result[1])      # 0.92 — index access
```

### Sets — Deduplication in AI Pipelines

```python
# Remove duplicate documents during ingestion
seen_ids: set[str] = set()
unique_docs = []

for doc in all_documents:
    if doc["id"] not in seen_ids:
        seen_ids.add(doc["id"])
        unique_docs.append(doc)

# Set operations for keyword analysis
query_words = {"python", "machine", "learning"}
doc_words = {"python", "deep", "learning", "neural"}

shared = query_words & doc_words    # intersection: {"python", "learning"}
all_words = query_words | doc_words  # union
only_query = query_words - doc_words # difference: {"machine"}
```

---

## 2. Functions — The Core Unit of AI Code

### Basic Function Patterns

```python
from typing import Optional

# Type hints on ALL parameters and return values
def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks for RAG ingestion.
    
    Args:
        text: The document text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Overlap between consecutive chunks
    
    Returns:
        List of text chunks
    
    Example:
        >>> chunks = chunk_text("Long document...", chunk_size=100)
        >>> len(chunks[0]) <= 100
        True
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks
```

### Default Arguments — Critical for LLM Config

```python
# Default arguments make functions usable with sensible defaults
def call_llm(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_message: str = "You are a helpful assistant.",
) -> str:
    """Call LLM with configurable parameters."""
    # Implementation...
    pass

# Call with just the required argument
response = call_llm("What is RAG?")

# Override specific defaults
response = call_llm("Write a poem", temperature=1.2, max_tokens=500)
```

### *args and **kwargs — Flexible AI Functions

```python
def build_messages(
    user_message: str,
    *examples: tuple[str, str],     # Variable positional: (user_q, assistant_ans) pairs
    system: str = "You are helpful",
    **extra_params: str,             # Variable keyword: extra metadata
) -> list[dict[str, str]]:
    """Build message list for LLM API with few-shot examples."""
    messages = [{"role": "system", "content": system}]
    
    # Add few-shot examples
    for user_ex, assistant_ex in examples:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})
    
    messages.append({"role": "user", "content": user_message})
    return messages

# Usage
msgs = build_messages(
    "What is the capital of Germany?",
    ("What is the capital of France?", "Paris"),  # example 1
    ("What is the capital of Italy?", "Rome"),    # example 2
    system="You are a geography expert.",
)
```

### Lambda Functions

```python
# Lambdas for quick key functions in sorting/filtering
sort_by_score = lambda x: x["score"]
results = sorted(search_results, key=sort_by_score, reverse=True)

# Lambdas in map/filter (use sparingly — prefer comprehensions)
token_counts = list(map(lambda t: len(t.split()), texts))
long_texts = list(filter(lambda t: len(t) > 100, texts))

# When NOT to use lambda — use a named function instead
# BAD: complex lambda
process = lambda x: x.strip().lower().replace("  ", " ") if x else ""
# GOOD: named function with docstring
def clean_text(text: str) -> str:
    """Clean text for embedding: strip, lowercase, normalize whitespace."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())
```

---

## 3. Comprehensions — Elegant Batch Processing

```python
# List comprehension — generate embeddings for many texts
texts = ["Python is great", "AI is the future", "RAG reduces hallucination"]

# Instead of:
token_counts = []
for text in texts:
    token_counts.append(len(text.split()))

# Use this:
token_counts = [len(text.split()) for text in texts]

# With filtering — only chunk long documents
long_docs = [doc for doc in documents if len(doc["text"]) > 200]

# Nested comprehension — flatten list of chunks
all_chunks = [
    chunk
    for doc in documents
    for chunk in chunk_text(doc["text"])
]

# Dict comprehension — build metadata index
doc_index: dict[str, dict] = {
    doc["id"]: {"title": doc["title"], "tokens": estimate_tokens(doc["text"])}
    for doc in documents
}

# Set comprehension — unique categories
categories: set[str] = {doc["category"] for doc in documents}

# Generator expression — lazy evaluation (memory efficient)
# Instead of building a full list, compute on demand
total_tokens = sum(len(text.split()) for text in huge_document_list)
```

---

## 4. Exception Handling — AI Systems Fail Often

LLM API calls fail more than most API calls. Always handle exceptions properly.

```python
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


# WRONG — bare except catches everything including KeyboardInterrupt
try:
    response = call_api()
except:
    print("Error")


# WRONG — catching Exception silently swallows errors
try:
    response = call_api()
except Exception:
    pass


# RIGHT — specific exceptions with meaningful handling
class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMRateLimitError(LLMError):
    """Too many requests to LLM API."""
    pass

class LLMAuthError(LLMError):
    """Invalid or missing API key."""
    pass

class LLMTimeoutError(LLMError):
    """LLM API call timed out."""
    pass


def call_llm_with_retry(
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """
    Call LLM with exponential backoff retry on rate limits.
    
    Retry strategy:
    - Rate limit: wait and retry (exponential backoff)
    - Auth error: fail fast (no point retrying)
    - Timeout: retry immediately once, then fail
    - Other errors: log and re-raise
    """
    for attempt in range(max_retries):
        try:
            return _call_llm_api(prompt)
            
        except LLMRateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Last attempt, re-raise
            wait = base_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt+1})")
            time.sleep(wait)
            
        except LLMAuthError:
            logger.error("Authentication failed — check API key")
            raise  # Don't retry auth errors
            
        except LLMTimeoutError:
            logger.warning(f"Timeout on attempt {attempt+1}")
            if attempt == max_retries - 1:
                raise
                
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}", exc_info=True)
            raise LLMError(f"Unexpected error: {e}") from e
    
    raise LLMError("All retry attempts exhausted")


def _call_llm_api(prompt: str) -> str:
    """Internal LLM API call (placeholder)."""
    # Real implementation would call OpenAI/Anthropic here
    return f"Response to: {prompt[:50]}"
```

### Context Managers — Resource Safety

```python
# Context managers ensure cleanup happens even on exceptions
# Critical for: file handles, database connections, temp files

# Built-in: file handling
with open("document.txt", "r", encoding="utf-8") as f:
    content = f.read()
# File is automatically closed — even if exception occurs

# Custom context manager using class
class VectorDBConnection:
    """Manages vector database connection lifecycle."""
    
    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
        self.client = None
    
    def __enter__(self) -> "VectorDBConnection":
        self.client = connect_to_db(self.connection_string)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.client:
            self.client.close()
        return False  # Don't suppress exceptions

# Usage
with VectorDBConnection("localhost:8000") as db:
    results = db.search("What is RAG?")
# Connection automatically closed

# Context manager using @contextmanager decorator
from contextlib import contextmanager

@contextmanager
def temporary_collection(name: str):
    """Create a temporary vector DB collection, clean up after."""
    collection = create_collection(name)
    try:
        yield collection
    finally:
        delete_collection(name)

with temporary_collection("test_search") as coll:
    coll.add_documents(test_docs)
    results = coll.search("test query")
```

---

## 5. Type Hints — Non-Negotiable in Production AI Code

```python
from typing import Optional, Union, Any, Callable
from collections.abc import Iterator, Generator, AsyncGenerator

# Basic types
name: str = "AI Engineer"
temperature: float = 0.7
max_tokens: int = 1024
is_streaming: bool = True

# Container types
messages: list[dict[str, str]] = []
scores: dict[str, float] = {}
unique_ids: set[str] = set()

# Optional — value or None
api_key: Optional[str] = None
# Same as:
api_key: str | None = None  # Python 3.10+ syntax

# Union — multiple types
token_or_text: Union[int, str] = "hello"
# Same as:
token_or_text: int | str = "hello"  # Python 3.10+

# Callable types — for functions as parameters
def retry(func: Callable[..., str], retries: int = 3) -> str:
    for i in range(retries):
        try:
            return func()
        except Exception:
            if i == retries - 1:
                raise
    return ""

# Generator types
def stream_llm_response(prompt: str) -> Generator[str, None, None]:
    """Yield response tokens one at a time."""
    for token in get_tokens(prompt):
        yield token

# Type aliases — for complex types
MessageList = list[dict[str, str]]
EmbeddingVector = list[float]
SearchResults = list[tuple[str, float, str]]  # (id, score, text)

def search(query_embedding: EmbeddingVector) -> SearchResults:
    ...
```

---

## 6. Modules and Packages — Project Structure

```python
# A Python package is a directory with __init__.py
# Your AI project structure:

# src/
# ├── __init__.py
# ├── llm/
# │   ├── __init__.py
# │   ├── client.py        ← LLMClient class
# │   └── prompts.py       ← Prompt templates
# ├── rag/
# │   ├── __init__.py
# │   ├── chunker.py       ← Text chunking
# │   ├── embedder.py      ← Embedding generation
# │   └── retriever.py     ← Vector search
# └── utils/
#     ├── __init__.py
#     └── tokens.py        ← Token counting

# Importing from your modules
from src.llm.client import LLMClient
from src.rag.chunker import chunk_text
from src.rag.embedder import embed_texts

# The __init__.py makes imports cleaner:
# src/rag/__init__.py:
from .chunker import chunk_text
from .embedder import embed_texts
from .retriever import search

# Then in application code:
from src.rag import chunk_text, embed_texts, search
```

---

## 7. JSON — The Language of LLM APIs

```python
import json
from typing import Any

# JSON is everywhere in AI:
# - LLM API requests and responses
# - Configuration files
# - Vector DB metadata
# - Structured LLM output

# Parsing JSON
json_str = '{"role": "user", "content": "What is RAG?"}'
message: dict[str, str] = json.loads(json_str)

# Serializing to JSON
message = {"role": "assistant", "content": "RAG stands for..."}
json_str = json.dumps(message)
json_pretty = json.dumps(message, indent=2)  # Human-readable

# Reading/writing JSON files
with open("config.json", "r") as f:
    config: dict[str, Any] = json.load(f)

with open("output.json", "w") as f:
    json.dump(results, f, indent=2)

# Parsing LLM structured output (critical skill)
llm_output = '''
{
  "intent": "question",
  "topic": "RAG",
  "confidence": 0.95,
  "entities": ["RAG", "retrieval", "generation"]
}
'''

# Safe JSON parsing from LLM (LLM may not always produce valid JSON)
def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output with error handling."""
    # LLMs sometimes wrap JSON in markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    if text.startswith("```"):
        text = text[3:]   # Remove ```
    if text.endswith("```"):
        text = text[:-3]  # Remove closing ```
    
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw output: {text}")
```

---

## 8. Async/Await — Essential for AI Backends

This is the most important Python pattern for AI Engineering.

### Why Async?

```
Without async:              With async:
Request 1 → [LLM: 2s wait] → Response 1    Request 1 ─────────────[LLM]──→ Response 1
                                              Request 2 ────[LLM]────────→ Response 2
                              TIME →          Request 3 ─[LLM]──────────→ Response 3
Request 2 → [LLM: 2s wait] → Response 2     All three complete in ~2s total
Request 3 → [LLM: 2s wait] → Response 3     (instead of 6s sequentially)
Total: 6 seconds                             Total: ~2 seconds
```

LLM calls spend most of their time waiting for the API. Async lets you do something else while waiting.

```python
import asyncio
import time
from typing import Any

# Sync version — blocks on each call
def sync_llm_call(prompt: str) -> str:
    time.sleep(2)  # Simulates LLM latency
    return f"Response to: {prompt}"

def process_batch_sync(prompts: list[str]) -> list[str]:
    return [sync_llm_call(p) for p in prompts]
    # 10 prompts × 2s = 20 seconds total

# Async version — non-blocking
async def async_llm_call(prompt: str) -> str:
    await asyncio.sleep(2)  # Simulates LLM latency (non-blocking)
    return f"Response to: {prompt}"

async def process_batch_async(prompts: list[str]) -> list[str]:
    # All calls run concurrently!
    tasks = [async_llm_call(p) for p in prompts]
    return await asyncio.gather(*tasks)
    # 10 prompts × 2s = ~2 seconds total (concurrent)

# Running async code
async def main() -> None:
    prompts = ["What is RAG?", "Explain embeddings", "What is an agent?"]
    
    results = await process_batch_async(prompts)
    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt} → A: {result}")

# Entry point for async code
if __name__ == "__main__":
    asyncio.run(main())
```

### Async with Real LLM APIs

```python
import asyncio
from openai import AsyncOpenAI

async def embed_batch(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Embed a list of texts concurrently.
    Respects batch_size to avoid rate limits.
    """
    client = AsyncOpenAI()
    
    async def embed_single(text: str) -> list[float]:
        response = await client.embeddings.create(
            input=text,
            model=model,
        )
        return response.data[0].embedding
    
    # Process in batches to respect rate limits
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await asyncio.gather(
            *[embed_single(text) for text in batch]
        )
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

---

## 9. Generators — Streaming LLM Responses

```python
from collections.abc import Generator, Iterator

# Generator: a function that yields values one at a time
# Critical use case: streaming LLM responses

def stream_tokens(text: str) -> Generator[str, None, None]:
    """Simulate streaming: yield one word at a time."""
    words = text.split()
    for word in words:
        yield word + " "

# Consumer
for token in stream_tokens("Hello this is a streaming response"):
    print(token, end="", flush=True)  # Print without newline, flush immediately

# Real streaming with OpenAI
def stream_llm_response(prompt: str) -> Generator[str, None, None]:
    """Stream LLM response token by token."""
    from openai import OpenAI
    
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # Enable streaming
    )
    
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content  # Yield each token as it arrives

# Usage
for token in stream_llm_response("Explain RAG"):
    print(token, end="", flush=True)

# Async generator for FastAPI streaming
from collections.abc import AsyncGenerator

async def async_stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """Async streaming — used in FastAPI Server-Sent Events."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
```

---

## 10. Decorators — Powerful Patterns in AI Code

```python
import functools
import time
import logging
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])

# Decorator 1: Retry with exponential backoff
def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """Decorator that retries a function on failure."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = delay * (2 ** attempt)
                    logging.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                    time.sleep(wait)
        return wrapper  # type: ignore
    return decorator

# Usage
@retry(max_attempts=3, delay=1.0)
def call_embedding_api(text: str) -> list[float]:
    """Will be retried up to 3 times on failure."""
    return get_embedding(text)


# Decorator 2: Cache results (critical for cost optimization)
def cache_result(func: F) -> F:
    """Simple in-memory cache (use Redis in production)."""
    cache: dict[str, Any] = {}
    
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper  # type: ignore

@cache_result
def get_embedding(text: str) -> list[float]:
    """Cached embedding — won't call API for same text twice."""
    return call_embedding_api(text)


# Decorator 3: Timer for performance monitoring
def timer(func: F) -> F:
    """Log how long a function takes."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper  # type: ignore

@timer
def ingest_documents(paths: list[str]) -> int:
    """Time the ingestion process."""
    return sum(1 for _ in paths)
```

---

## 11. Iterators — Understanding Data Pipelines

```python
from collections.abc import Iterator
from typing import Any

# Custom iterator for document streaming (memory efficient)
class DocumentIterator:
    """
    Iterate over documents without loading all into memory.
    Essential for large document corpora (10M+ documents).
    """
    
    def __init__(self, file_paths: list[str]) -> None:
        self.file_paths = file_paths
        self._index = 0
    
    def __iter__(self) -> "DocumentIterator":
        return self
    
    def __next__(self) -> dict[str, str]:
        if self._index >= len(self.file_paths):
            raise StopIteration
        
        path = self.file_paths[self._index]
        self._index += 1
        
        with open(path, "r", encoding="utf-8") as f:
            return {"path": path, "content": f.read()}

# Usage
doc_iter = DocumentIterator(["doc1.txt", "doc2.txt", "doc3.txt"])
for doc in doc_iter:
    chunks = chunk_text(doc["content"])
    # Process without loading ALL documents at once


# itertools for AI pipelines
import itertools

# Batch documents for embedding
def batch_items(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    """Split a list into batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# Embed in batches of 100
texts = ["text1", "text2", ...]  # thousands of texts
for batch in batch_items(texts, batch_size=100):
    embeddings = embed_batch(batch)  # Don't send 10,000 texts at once!
```

---

## 12. Virtual Environments and pip — Project Management

```bash
# Create a virtual environment (do this for every project)
python -m venv venv

# Activate
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

# Install packages with pinned versions (ALWAYS pin versions in production)
pip install openai==1.35.0
pip install anthropic==0.28.0
pip install fastapi==0.111.0

# Save exact versions to requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt (reproducible)
pip install -r requirements.txt

# Upgrade a package
pip install --upgrade openai

# Show installed packages
pip list

# Check for security vulnerabilities
pip install safety
safety check
```

### requirements.txt Best Practice

```
# requirements.txt — pin ALL versions for reproducibility
openai==1.35.0
anthropic==0.28.0
chromadb==0.4.24
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
python-dotenv==1.0.1
numpy==1.26.4
pandas==2.2.2

# requirements-dev.txt — dev only dependencies
pytest==8.2.2
pytest-asyncio==0.23.7
black==24.4.2
ruff==0.4.9
mypy==1.10.0
```
