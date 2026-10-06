# Day 02 — Debugging Guide

---

## Debugging Async Python

### Problem: Coroutine never runs / "RuntimeWarning: coroutine was never awaited"
```python
# BUG — forgot await
result = async_function()  # Returns coroutine object, doesn't run it!

# FIX
result = await async_function()
```

### Problem: "RuntimeError: This event loop is already running"
```python
# BUG — calling asyncio.run() inside an existing event loop (e.g., Jupyter)
asyncio.run(main())  # Fails in Jupyter

# FIX in Jupyter
import nest_asyncio
nest_asyncio.apply()
asyncio.run(main())
# OR: just use `await main()` in a Jupyter cell
```

### Problem: Async function runs slower than sync
```python
# BUG — sequential awaits inside gather
async def slow():
    a = await call_1()  # waits
    b = await call_2()  # waits after a
    return a, b

# FIX — concurrent
async def fast():
    a, b = await asyncio.gather(call_1(), call_2())
    return a, b
```

---

## Debugging Type Errors

### Check with mypy
```bash
pip install mypy
mypy your_file.py
```

### Common type errors
```python
# Error: Argument 1 to "embed" has incompatible type "str | None"
text: str | None = get_text()
embed(text)  # text could be None!

# Fix
if text is not None:
    embed(text)
# OR
embed(text or "")
```

---

## Debugging JSON Parse Failures

```python
# Always log the raw output before parsing
def debug_parse(raw: str) -> dict:
    print(f"[DEBUG] Raw LLM output:\n{raw}\n")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] Parse error at position {e.pos}: {e.msg}")
        print(f"[DEBUG] Near: {raw[max(0, e.pos-20):e.pos+20]}")
        raise
```

---

## Common Error Messages and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `coroutine was never awaited` | Missing `await` | Add `await` |
| `object is not awaitable` | Awaiting non-async function | Remove `await` |
| `JSONDecodeError` | LLM returned non-JSON | Strip code fences first |
| `KeyError: 'OPENAI_API_KEY'` | Missing env var | Add key to `.env`, call `load_dotenv()` |
| `TypeError: unhashable type: 'list'` | Using list as dict key | Use tuple instead |
| `RecursionError` | Infinite loop in generator | Add termination condition |
