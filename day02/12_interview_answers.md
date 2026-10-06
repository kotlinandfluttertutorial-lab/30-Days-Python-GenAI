# Day 02 — Interview Answers

---

**Q1. Why async over threading for AI backends?**

LLM calls are I/O-bound (waiting for network). Async handles thousands of concurrent I/O waits with one thread via the event loop — no thread creation overhead, no GIL contention, no thread-safety issues. Threading is better for CPU-bound work (PyTorch inference). For an LLM API backend: use async. For model serving: use threading or multiprocessing.

---

**Q2. The GIL and AI applications:**

The GIL (Global Interpreter Lock) prevents true parallel execution of Python threads. For I/O-bound work (LLM API calls), async is the solution — no threads needed. For CPU-bound work (NumPy, PyTorch), the GIL doesn't matter because NumPy/PyTorch release it. For pure Python CPU work, use `multiprocessing` to bypass the GIL.

---

**Q3. N concurrent LLM calls:**

```python
import asyncio
from openai import AsyncOpenAI

async def batch_complete(prompts: list[str]) -> list[str]:
    client = AsyncOpenAI()
    tasks = [
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": p}]
        )
        for p in prompts
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r.choices[0].message.content if not isinstance(r, Exception) else ""
        for r in responses
    ]
```

---

**Q4. Generator vs list:**

A list computes all values at once and holds them in memory. A generator computes values lazily — one at a time. Use a generator when: the dataset is too large to fit in memory, you only need to iterate once, or you want to pipeline transformations. Example: streaming 10M document chunks through an embedding pipeline — a list would use GBs of RAM; a generator uses bytes.

---

**Q5. Dataclass vs Pydantic:**

`@dataclass`: lightweight, no runtime validation, no JSON serialization. Use for internal objects.
`pydantic.BaseModel`: runtime validation, JSON serialization, FastAPI integration. Use for API models and external data.

Key difference: dataclass won't raise if you pass an int where a str is expected. Pydantic will.

---

**Q6. Retry with exponential backoff:**

```python
import functools, time

def retry(max_attempts=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))  # 1s, 2s, 4s
        return wrapper
    return decorator
```

---

**Q10. Streaming LLM response:**

```python
from openai import OpenAI

def stream_response(prompt: str):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Usage: for token in stream_response("Explain RAG"): print(token, end="", flush=True)
```

In FastAPI: use `StreamingResponse` with an async generator.

---

**Q13. time.sleep vs asyncio.sleep:**

`time.sleep(n)` blocks the entire thread — during that second, no other async task can run. `await asyncio.sleep(n)` yields control to the event loop — other coroutines can run while this one waits. In async code, ALWAYS use `asyncio.sleep`. Using `time.sleep` in an async function defeats the purpose of async.

---

**Q14. Handling JSONDecodeError from LLM:**

LLMs don't always produce valid JSON even when asked. Handle it:
1. Strip markdown code fences (LLMs wrap JSON in ```json ... ```)
2. Try `json.loads()` in a try/except
3. On failure: log the raw output, raise a descriptive ValueError
4. For robustness: try a second LLM call asking it to "fix this JSON: {raw_output}"

Never silently return `{}` — you'll get hard-to-debug downstream errors.
