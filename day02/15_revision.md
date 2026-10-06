# Day 02 — Revision
## Python for AI Engineering — Quick Reference

---

## Must-Know Patterns (Memorize)

```python
# 1. Load env vars safely
from dotenv import load_dotenv; import os
load_dotenv()
key = os.environ["OPENAI_API_KEY"]  # Raises KeyError if missing

# 2. Dataclass with mutable defaults
from dataclasses import dataclass, field
@dataclass
class Messages:
    history: list[dict] = field(default_factory=list)

# 3. Concurrent LLM calls
results = await asyncio.gather(*[call_llm(p) for p in prompts])

# 4. Retry decorator
@retry(max_attempts=3, delay=1.0, exceptions=(RateLimitError,))
def call_api(): ...

# 5. Parse LLM JSON
text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
data = json.loads(text)

# 6. Generator for large datasets
def stream_chunks(path: str) -> Generator[str, None, None]:
    with open(path) as f:
        for line in f:
            yield line.strip()

# 7. Context manager
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 8. Timeout async call
result = await asyncio.wait_for(call_llm(prompt), timeout=30.0)
```

---

## Data Type Quick Reference

| Type | Use In AI | Example |
|------|-----------|---------|
| `list[dict]` | Message history | `[{"role":"user","content":"..."}]` |
| `list[float]` | Embedding vector | `[0.1, 0.2, ..., 0.9]` |
| `dict[str, Any]` | Metadata | `{"source": "doc.pdf", "page": 3}` |
| `str` | Text, prompts | `"What is RAG?"` |
| `tuple[str, float]` | Search result pair | `("doc_001", 0.95)` |
| `set[str]` | Seen document IDs | `{"id1", "id2"}` |

---

## Async Rules

1. `await` inside async functions only
2. `asyncio.sleep()` in async, `time.sleep()` in sync
3. `asyncio.gather()` for concurrent I/O
4. `asyncio.wait_for()` for timeouts
5. `return_exceptions=True` for resilient batch processing

---

## Interview One-Liners

- "Why async?" → "LLM calls are I/O-bound; async handles thousands of concurrent waits with one thread."
- "Generator vs list?" → "Generator is lazy — computes one item at a time; essential for large datasets."
- "Dataclass vs Pydantic?" → "Dataclass for internal objects; Pydantic for API validation and serialization."
- "Why pin versions?" → "Prevent breaking changes from silently deploying to production."
