# Day 02 — Exercises

---

## Theory (10 Questions)

**T1.** What is the difference between a list and a tuple in Python? Give a concrete example where you would use each in an AI application.

**T2.** Explain why async/await is important for LLM-powered applications. What happens to throughput if you use synchronous code for LLM calls?

**T3.** What is a generator? How does it differ from a function that returns a list? Why would you use a generator for a 1-million-document corpus?

**T4.** What is the difference between `os.getenv("KEY")` and `os.environ["KEY"]`? Which should you use for required configuration values?

**T5.** Why should you always pin package versions in `requirements.txt`? What can go wrong if you don't?

**T6.** What does `@functools.wraps(func)` do in a decorator? What breaks if you omit it?

**T7.** Explain the difference between a dataclass and a Pydantic BaseModel. When would you use each?

**T8.** What is the purpose of `__post_init__` in a Python dataclass?

**T9.** Why is `except Exception: pass` a dangerous anti-pattern in AI applications?

**T10.** Explain what `asyncio.gather(*tasks, return_exceptions=True)` does differently from `asyncio.gather(*tasks)`.

---

## Coding (5 Questions)

**C1.** Write a `ConversationHistory` class that:
- Stores messages as `list[dict[str, str]]`
- Has `add_user(text)` and `add_assistant(text)` methods
- Has a `to_api_format()` method returning the OpenAI message format
- Has a `truncate(max_messages)` method keeping the most recent N messages
- Has a `token_estimate` property

**C2.** Write a `chunk_text(text, chunk_size=512, overlap=50)` generator that yields chunks. Must: handle empty strings, not yield chunks shorter than 20 chars, and overlap properly.

**C3.** Write an async function `embed_many(texts: list[str]) -> list[list[float]]` that simulates embedding calls with `asyncio.gather`, adds a 100ms delay per call (simulating API latency), and processes in batches of 10.

**C4.** Write a `safe_json_parse(text: str) -> dict` function that:
- Strips markdown code fences (```json ... ```)
- Handles `json.JSONDecodeError` with a clear error message
- Raises `ValueError` with the raw text in the error message for debugging

**C5.** Write a `@retry(attempts=3, delay=1.0, exceptions=(Exception,))` decorator that retries a function with exponential backoff. Include logging on each retry.

---

## Debugging (5 Questions)

**D1.** Find and fix all bugs:
```python
async def process_documents(docs: list[str]) -> list[str]:
    results = []
    for doc in docs:
        result = await embed_document(doc)  # assume this works
        results.append(result)
    return results

# Problem: this is sequential. Fix it to be concurrent.
```

**D2.** Fix the bug:
```python
@dataclass
class Config:
    api_keys: list[str] = []      # Bug!
    metadata: dict[str, str] = {} # Bug!
```

**D3.** Find the issue:
```python
class DocumentStore:
    def __init__(self):
        self.documents = []
    
    def add(self, doc: str):
        self.documents.append(doc)
    
    def search(self, query: str) -> list[str]:
        return [d for d in self.documents if query in d]
    
# This works. But what happens when used in an async FastAPI app
# with 100 concurrent requests all calling add() simultaneously?
```

**D4.** Find and fix the exception handling mistake:
```python
def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}  # Silently return empty dict
```

**D5.** Find the type annotation bug:
```python
from typing import Optional

def get_embedding(
    text: str,
    model: str = "text-embedding-3-small",
    dimensions: Optional[int] = None,
) -> list[float]:
    # ... implementation ...
    return [0.1, 0.2, 0.3]

# Usage that the type checker should catch:
result: list[int] = get_embedding("hello")  # Bug in the annotation
```

---

## Interview (10 Questions)

**I1.** "What Python data structure would you use to store conversation history, and why?"

**I2.** "Explain async/await in Python. Why do AI backends use it?"

**I3.** "How do you handle LLM API rate limits in production Python code?"

**I4.** "What is the difference between `Exception` and `BaseException` in Python?"

**I5.** "How would you cache embedding results to avoid paying for the same text twice?"

**I6.** "What is a Python generator and when would you use one in an AI pipeline?"

**I7.** "How do you write a decorator that adds retry logic to a function?"

**I8.** "What is the difference between `json.loads()` and `json.load()`?"

**I9.** "How do type hints help in AI engineering beyond documentation?"

**I10.** "A function returns `None` when it should return a list. Walk me through how you debug this."

---

## Architecture (1 Question)

**A1.** Design a Python class hierarchy for an LLM abstraction layer that:
- Supports OpenAI, Anthropic, and local Ollama
- Has a common interface (`complete`, `chat`, `stream`)
- Handles rate limits, retries, and timeouts
- Tracks token usage and cost
- Is testable (supports mock implementations)

Draw the class diagram and write the interface (abstract methods).

---

## Practical Challenge

**P1.** Extend `03_ai_data_processor.py`:
1. Add a `filter_by_length(chunks, min_tokens=20, max_tokens=600)` function
2. Add a `deduplicate_chunks(chunks)` function using a set of hashes
3. Add a `stats_by_source(chunks)` function returning per-file statistics
4. Add a `--verbose` CLI flag that shows each chunk as it's processed
5. Export a `processing_summary.json` with the full report
