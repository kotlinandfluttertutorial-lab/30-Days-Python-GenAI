"""
Day 02 — Python Fundamentals for AI Engineering
================================================
Demonstrates every Python pattern used in production AI code.
No external dependencies required.

Run: python 01_python_fundamentals.py
"""

import json
import math
import time
import logging
import functools
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, TypeVar
from collections.abc import Generator, Iterator
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATA TYPES IN AI CONTEXT
# ═══════════════════════════════════════════════════════════════

def demonstrate_strings() -> None:
    """String operations critical for AI engineering."""
    print("\n── STRINGS IN AI ──")

    # Prompt template — the most common string operation
    def build_rag_prompt(context: str, question: str, max_context: int = 2000) -> str:
        """Build a RAG prompt from context and question."""
        # Truncate context if too long (context window management)
        if len(context) > max_context:
            context = context[:max_context] + "\n[... context truncated ...]"

        return f"""You are a helpful AI assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {question}

Answer:""".strip()

    sample_context = (
        "RAG stands for Retrieval-Augmented Generation. "
        "It was introduced to solve the hallucination problem in LLMs. "
        "RAG retrieves relevant documents and injects them as context."
    )

    prompt = build_rag_prompt(sample_context, "What problem does RAG solve?")
    print(f"Prompt preview ({len(prompt)} chars):")
    print(prompt[:200] + "...")

    # Text cleaning for document ingestion
    raw_text = "  The   quick   brown fox \n\n has   lots   of  spaces  "
    cleaned = " ".join(raw_text.split())  # Normalize all whitespace
    print(f"\nRaw text: '{raw_text[:40]}...'")
    print(f"Cleaned:  '{cleaned}'")

    # Chunk detection — find natural boundaries
    document = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
    print(f"\nDocument split into {len(paragraphs)} paragraphs: {paragraphs}")


def demonstrate_lists() -> None:
    """List operations for document processing."""
    print("\n── LISTS IN AI ──")

    # Conversation history management
    conversation: list[dict[str, str]] = []

    def add_message(role: str, content: str) -> None:
        conversation.append({"role": role, "content": content})

    def get_recent_history(max_turns: int = 5) -> list[dict[str, str]]:
        """Keep only the last N turns to fit in context window."""
        return conversation[-max_turns * 2:]  # 2 messages per turn

    add_message("user", "What is RAG?")
    add_message("assistant", "RAG is Retrieval-Augmented Generation...")
    add_message("user", "How does it reduce hallucination?")
    add_message("assistant", "By grounding the LLM in retrieved facts...")

    recent = get_recent_history(max_turns=2)
    print(f"Full history: {len(conversation)} messages")
    print(f"Recent (last 2 turns): {len(recent)} messages")

    # Sorting search results by score
    search_results = [
        ("doc_001", 0.72, "RAG is Retrieval-Augmented Generation"),
        ("doc_003", 0.95, "RAG solves hallucination by retrieving context"),
        ("doc_002", 0.88, "RAG pipeline: chunk, embed, retrieve, generate"),
    ]
    search_results.sort(key=lambda x: x[1], reverse=True)
    print("\nSearch results (sorted by score):")
    for doc_id, score, text in search_results:
        print(f"  [{score:.2f}] {doc_id}: {text[:50]}...")


def demonstrate_dicts() -> None:
    """Dict operations for metadata and API responses."""
    print("\n── DICTS IN AI ──")

    # LLM API response structure
    mock_response: dict[str, Any] = {
        "id": "chatcmpl-abc123",
        "model": "gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "RAG stands for Retrieval-Augmented Generation...",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 200,
            "total_tokens": 350,
        },
    }

    # Safe extraction
    content = mock_response["choices"][0]["message"]["content"]
    finish_reason = mock_response["choices"][0].get("finish_reason", "unknown")
    total_tokens = mock_response["usage"]["total_tokens"]

    print(f"Response: {content[:60]}...")
    print(f"Finish reason: {finish_reason}")
    print(f"Total tokens: {total_tokens}")

    # Metadata for vector DB
    doc_metadata: dict[str, Any] = {
        "source": "ml_handbook.pdf",
        "page": 42,
        "category": "machine_learning",
        "embedding_model": "text-embedding-3-small",
    }

    # Dict comprehension — transform all values to strings for storage
    str_metadata = {k: str(v) for k, v in doc_metadata.items()}
    print(f"\nMetadata: {doc_metadata}")
    print(f"Stringified: {str_metadata}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2: DATACLASSES FOR AI OBJECTS
# ═══════════════════════════════════════════════════════════════

@dataclass
class LLMMessage:
    """A single message in an LLM conversation."""
    role: str        # "system", "user", or "assistant"
    content: str
    token_count: int = 0

    def __post_init__(self) -> None:
        """Validate after initialization."""
        valid_roles = {"system", "user", "assistant", "tool"}
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of {valid_roles}")
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
        # Estimate token count
        self.token_count = len(self.content.split()) * 4 // 3  # rough estimate

    def to_dict(self) -> dict[str, str]:
        """Convert to OpenAI API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class DocumentChunk:
    """A chunk of a document prepared for RAG ingestion."""
    chunk_id: str
    source_document: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = field(default=None, repr=False)

    @property
    def token_estimate(self) -> int:
        return len(self.text.split()) * 4 // 3

    @property
    def is_embedded(self) -> bool:
        return self.embedding is not None

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class SearchResult:
    """Result from a vector similarity search."""
    chunk_id: str
    score: float
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_relevant(self) -> bool:
        """High relevance threshold."""
        return self.score >= 0.8


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"


def demonstrate_dataclasses() -> None:
    """Show dataclass usage in AI context."""
    print("\n── DATACLASSES IN AI ──")

    # Create LLM messages
    system_msg = LLMMessage(role="system", content="You are a helpful AI assistant.")
    user_msg = LLMMessage(role="user", content="What is RAG?")
    messages = [system_msg, user_msg]

    print("Conversation messages:")
    for msg in messages:
        print(f"  {msg.role}: {msg.content[:50]} (~{msg.token_count} tokens)")

    # Create document chunks
    chunk = DocumentChunk(
        chunk_id="doc001_chunk_0",
        source_document="rag_guide.pdf",
        text="RAG stands for Retrieval-Augmented Generation...",
        chunk_index=0,
        metadata={"page": 1, "category": "rag"},
    )
    print(f"\nDocument chunk: {chunk.chunk_id}")
    print(f"  Length: {len(chunk)} chars, ~{chunk.token_estimate} tokens")
    print(f"  Embedded: {chunk.is_embedded}")

    # Create search results and filter
    results = [
        SearchResult("c001", 0.95, "RAG reduces hallucination...", "doc1.pdf"),
        SearchResult("c002", 0.72, "Vector databases store embeddings...", "doc2.pdf"),
        SearchResult("c003", 0.88, "Chunking strategy affects retrieval...", "doc1.pdf"),
    ]
    relevant = [r for r in results if r.is_relevant]
    print(f"\n{len(results)} results, {len(relevant)} highly relevant (score ≥ 0.8)")


# ═══════════════════════════════════════════════════════════════
# SECTION 3: COMPREHENSIONS
# ═══════════════════════════════════════════════════════════════

def demonstrate_comprehensions() -> None:
    """Comprehension patterns for batch AI processing."""
    print("\n── COMPREHENSIONS IN AI ──")

    documents = [
        {"id": "doc1", "text": "RAG is Retrieval-Augmented Generation", "tokens": 6},
        {"id": "doc2", "text": "Embeddings represent semantic meaning", "tokens": 5},
        {"id": "doc3", "text": "LLMs predict the next token", "tokens": 5},
        {"id": "doc4", "text": "A", "tokens": 1},  # Too short
        {"id": "doc5", "text": "Vector databases enable similarity search", "tokens": 6},
    ]

    # List comprehension — process all documents
    texts = [doc["text"] for doc in documents]
    print(f"Extracted {len(texts)} texts")

    # Filtered comprehension — only long documents
    long_docs = [doc for doc in documents if doc["tokens"] >= 5]
    print(f"Long documents: {len(long_docs)}")

    # Dict comprehension — index by ID
    doc_index = {doc["id"]: doc["text"] for doc in documents}
    print(f"Indexed: {list(doc_index.keys())}")

    # Nested comprehension — split all docs into words (flatten)
    all_words = [word for doc in documents for word in doc["text"].lower().split()]
    # Count unique words
    unique_words = set(all_words)
    print(f"Total words: {len(all_words)}, Unique: {len(unique_words)}")

    # Generator expression — memory efficient sum
    total_tokens = sum(doc["tokens"] for doc in documents)
    avg_tokens = total_tokens / len(documents)
    print(f"Total tokens: {total_tokens}, Average: {avg_tokens:.1f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 4: EXCEPTION HANDLING
# ═══════════════════════════════════════════════════════════════

class AIError(Exception):
    """Base exception for AI operations."""
    pass

class RateLimitError(AIError):
    """LLM API rate limit exceeded."""
    def __init__(self, retry_after: float = 60.0) -> None:
        super().__init__(f"Rate limited. Retry after {retry_after:.0f}s")
        self.retry_after = retry_after

class ContextTooLongError(AIError):
    """Input exceeds model context window."""
    def __init__(self, limit: int, actual: int) -> None:
        super().__init__(f"Context too long: {actual} > {limit} tokens")
        self.limit = limit
        self.actual = actual


def safe_llm_call(
    prompt: str,
    max_tokens_limit: int = 4096,
    max_retries: int = 3,
) -> str:
    """
    LLM call with proper exception handling and retry logic.
    Demonstrates production-quality error handling.
    """
    prompt_tokens = len(prompt.split()) * 4 // 3

    if prompt_tokens > max_tokens_limit:
        raise ContextTooLongError(limit=max_tokens_limit, actual=prompt_tokens)

    for attempt in range(max_retries):
        try:
            # Simulate LLM call (in production: actual API call)
            if attempt == 0 and "fail" in prompt.lower():
                raise RateLimitError(retry_after=1.0)

            return f"[Simulated response to: {prompt[:40]}...]"

        except RateLimitError as e:
            logger.warning(f"Rate limited on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(e.retry_after)
            else:
                raise  # Final attempt failed

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}", exc_info=True)
            raise AIError(f"LLM call failed: {e}") from e

    return ""  # Unreachable, but satisfies type checker


def demonstrate_exceptions() -> None:
    """Show proper exception handling."""
    print("\n── EXCEPTION HANDLING ──")

    # Success case
    try:
        result = safe_llm_call("What is RAG?")
        print(f"Success: {result}")
    except AIError as e:
        print(f"Error: {e}")

    # Context too long
    try:
        long_prompt = "word " * 10000  # Very long prompt
        result = safe_llm_call(long_prompt)
    except ContextTooLongError as e:
        print(f"Context error: {e}")

    # Rate limit (simulated)
    try:
        result = safe_llm_call("fail please", max_retries=1)
    except RateLimitError as e:
        print(f"Rate limit caught: {e}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5: DECORATORS
# ═══════════════════════════════════════════════════════════════

F = TypeVar("F", bound=Callable[..., Any])


def retry_on_failure(
    max_attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    delay: float = 1.0,
) -> Callable[[F], F]:
    """Production-quality retry decorator with exponential backoff."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
                    wait = delay * (2 ** attempt)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
        return wrapper  # type: ignore
    return decorator


def timed(func: F) -> F:
    """Log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed * 1000:.1f}ms")
        return result
    return wrapper  # type: ignore


_CACHE: dict[str, Any] = {}

def memoize(func: F) -> F:
    """Cache function results (use Redis in production)."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
        if key not in _CACHE:
            _CACHE[key] = func(*args, **kwargs)
            logger.debug(f"Cache MISS for {func.__name__}")
        else:
            logger.debug(f"Cache HIT for {func.__name__}")
        return _CACHE[key]
    return wrapper  # type: ignore


call_count = 0

@memoize
@timed
def expensive_embedding(text: str) -> list[float]:
    """Simulate an expensive embedding API call."""
    global call_count
    call_count += 1
    time.sleep(0.01)  # Simulate 10ms API call
    return [hash(text) % 100 / 100.0 for _ in range(4)]  # Fake embedding


def demonstrate_decorators() -> None:
    """Show decorator patterns."""
    print("\n── DECORATORS ──")
    global call_count
    call_count = 0

    text = "What is RAG?"

    # First call — cache miss, takes 10ms
    emb1 = expensive_embedding(text)
    print(f"First call (cache miss): {emb1[:3]}... (calls made: {call_count})")

    # Second call — cache hit, instant
    emb2 = expensive_embedding(text)
    print(f"Second call (cache hit): {emb2[:3]}... (calls made: {call_count})")

    assert emb1 == emb2, "Cache should return same result"
    print(f"Results identical: {emb1 == emb2}")


# ═══════════════════════════════════════════════════════════════
# SECTION 6: JSON HANDLING
# ═══════════════════════════════════════════════════════════════

def parse_llm_json_output(text: str) -> dict[str, Any]:
    """
    Parse JSON from LLM output — handles markdown code blocks.
    LLMs often wrap JSON in ```json ... ``` even when not asked.
    """
    text = text.strip()

    # Strip markdown code fence if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\nRaw: {text[:200]}") from e


def demonstrate_json() -> None:
    """JSON operations for LLM structured output."""
    print("\n── JSON HANDLING ──")

    # Simulated LLM output with structured JSON
    llm_outputs = [
        '{"intent": "question", "topic": "RAG", "confidence": 0.95}',
        '```json\n{"classification": "technical", "sentiment": "positive"}\n```',
        'Here is the JSON:\n```\n{"result": "success", "count": 42}\n```',
    ]

    for raw_output in llm_outputs:
        try:
            parsed = parse_llm_json_output(raw_output)
            print(f"  Parsed: {parsed}")
        except ValueError as e:
            print(f"  Parse error: {e}")

    # Building LLM API request
    request = {
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": "Classify this text: ..."},
        ],
    }
    print(f"\nAPI request JSON ({len(json.dumps(request))} chars):")
    print(json.dumps(request, indent=2)[:200] + "...")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      DAY 02 — PYTHON FUNDAMENTALS FOR AI ENGINEERING     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demonstrate_strings()
    demonstrate_lists()
    demonstrate_dicts()
    demonstrate_dataclasses()
    demonstrate_comprehensions()
    demonstrate_exceptions()
    demonstrate_decorators()
    demonstrate_json()

    print("\n" + "="*60)
    print("✓ Day 02 Python fundamentals demo complete!")
    print("="*60)
    print("\nKey patterns demonstrated:")
    print("  • String operations for prompt building")
    print("  • List operations for document processing")
    print("  • Dict operations for metadata and API responses")
    print("  • Dataclasses for LLMMessage, DocumentChunk, SearchResult")
    print("  • Comprehensions for batch processing")
    print("  • Exception hierarchy for LLM errors")
    print("  • Decorators: retry, timer, cache")
    print("  • JSON parsing from LLM output")


if __name__ == "__main__":
    main()
