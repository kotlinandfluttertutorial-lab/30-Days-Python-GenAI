# Day 02 — Exercise Solutions

---

## C1. ConversationHistory

```python
from dataclasses import dataclass, field

@dataclass
class ConversationHistory:
    messages: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = "You are a helpful assistant."

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def to_api_format(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def truncate(self, max_messages: int) -> None:
        """Keep only the most recent messages (always keep pairs)."""
        max_messages = max(2, max_messages - (max_messages % 2))
        self.messages = self.messages[-max_messages:]

    @property
    def token_estimate(self) -> int:
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4
```

## C2. chunk_text Generator

```python
from collections.abc import Generator

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> Generator[str, None, None]:
    if not text or len(text) < 20:
        return

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= 20:
            yield chunk
        if end >= len(text):
            break
        start = end - overlap
```

## C3. Async Batch Embedding

```python
import asyncio

async def simulate_embed(text: str) -> list[float]:
    await asyncio.sleep(0.1)
    return [hash(text) % 100 / 100.0 for _ in range(4)]

async def embed_many(texts: list[str]) -> list[list[float]]:
    results = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = await asyncio.gather(*[simulate_embed(t) for t in batch])
        results.extend(batch_results)
    return results
```

## C4. Safe JSON Parse

```python
import json
from typing import Any

def safe_json_parse(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON: {e}\nRaw output (first 200 chars): {text[:200]}"
        ) from e
```

## C5. Retry Decorator

```python
import functools, time, logging
from typing import Callable, TypeVar, Any
F = TypeVar("F", bound=Callable[..., Any])

def retry(attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == attempts - 1:
                        raise
                    wait = delay * (2 ** attempt)
                    logging.warning(f"{func.__name__} attempt {attempt+1} failed: {e}. Retry in {wait:.1f}s")
                    time.sleep(wait)
        return wrapper  # type: ignore
    return decorator
```

---

## Debugging Solutions

**D1.** Make it concurrent:
```python
async def process_documents(docs: list[str]) -> list[str]:
    tasks = [embed_document(doc) for doc in docs]
    return await asyncio.gather(*tasks)
```

**D2.** Mutable default arguments bug:
```python
from dataclasses import dataclass, field

@dataclass
class Config:
    api_keys: list[str] = field(default_factory=list)      # Fixed
    metadata: dict[str, str] = field(default_factory=dict) # Fixed
# Never use mutable default args directly — all instances share the same object!
```

**D3.** Thread-safety issue:
The `documents` list is not thread-safe. In an async app with concurrent requests, multiple coroutines calling `add()` simultaneously can cause race conditions. Fix: use `asyncio.Lock()` or make it immutable between requests.

**D4.** Silent failure:
```python
def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config not found at {path}, using defaults")
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e
    # Don't catch everything — let unexpected errors propagate
```

**D5.** The annotation `list[int]` is wrong for the usage site — should be `list[float]`. The function signature is correct; the caller's type annotation is wrong.
