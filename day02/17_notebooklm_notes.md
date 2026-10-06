# Day 02 — NotebookLM Notes
## Python for AI Engineering

---

## Key Concepts

**Async/Await** — Non-blocking I/O for concurrent LLM calls. `asyncio.gather()` runs multiple coroutines simultaneously. Critical for FastAPI AI backends. `await asyncio.sleep()` (not `time.sleep()`) in async code.

**Generator** — Lazy sequence. `yield` instead of `return`. Memory-efficient for large document corpora. Used for streaming LLM responses.

**Dataclass** — Lightweight data container. Use `field(default_factory=list)` for mutable defaults. No runtime validation. Use for internal objects.

**Type Hints** — Annotations on parameters and returns. Enable IDE autocomplete, catch bugs early. Required in all production AI code.

**Exception Hierarchy** — Design specific exceptions: `LLMRateLimitError`, `LLMAuthError`, `ContextTooLongError`. Each has a different recovery strategy.

**JSON Handling** — LLMs often wrap JSON in markdown. Always strip code fences before `json.loads()`. Never silently return `{}` on parse failure.

**Decorator** — Wrap a function to add behavior. Production patterns: `@retry`, `@memoize`, `@timer`. Always use `@functools.wraps`.

## Code Patterns

```python
# Concurrent LLM calls
results = await asyncio.gather(*[call_llm(p) for p in prompts])

# Safe env loading
from dotenv import load_dotenv; import os
load_dotenv(); key = os.environ["KEY"]

# Dataclass with mutable default
@dataclass
class Store:
    items: list = field(default_factory=list)

# Generator for large data
def chunks(text): 
    for i in range(0, len(text), 512): yield text[i:i+512]
```

## Interview Facts

1. `asyncio.gather()` runs coroutines concurrently in one thread
2. `time.sleep()` in async code blocks the event loop — use `asyncio.sleep()`
3. Mutable default args in dataclasses must use `field(default_factory=...)`
4. `return_exceptions=True` prevents one failure from cancelling all gather tasks
5. Generators use `O(1)` memory vs `O(n)` for lists
6. `@functools.wraps` preserves the wrapped function's `__name__` and `__doc__`
7. Pin package versions in `requirements.txt` to prevent breaking changes
8. Pydantic validates at runtime; dataclass does not

## Common Traps

- Missing `await` → coroutine object returned, not the result
- `time.sleep()` in async → blocks event loop
- `except Exception: pass` → silently hides bugs
- Mutable default `[]` in dataclass → shared across all instances
- `os.getenv()` returns `None` silently; `os.environ[]` raises `KeyError`
