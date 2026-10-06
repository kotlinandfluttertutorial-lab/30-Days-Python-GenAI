# Day 02 — Common Mistakes

---

## Mistake 1: Mutable Default Arguments in Dataclasses
```python
# WRONG — all instances share the same list!
@dataclass
class Config:
    items: list = []

c1 = Config()
c2 = Config()
c1.items.append("x")
print(c2.items)  # ["x"] — unintended!

# RIGHT
from dataclasses import field
@dataclass
class Config:
    items: list = field(default_factory=list)
```

## Mistake 2: Using time.sleep in Async Code
```python
# WRONG — blocks the entire event loop!
async def with_delay():
    time.sleep(1)  # Blocks everything

# RIGHT
async def with_delay():
    await asyncio.sleep(1)  # Non-blocking
```

## Mistake 3: Sequential Async Code
```python
# WRONG — defeats the purpose of async
async def embed_all(texts):
    results = []
    for text in texts:
        result = await embed(text)  # Sequential!
        results.append(result)
    return results

# RIGHT — truly concurrent
async def embed_all(texts):
    return await asyncio.gather(*[embed(t) for t in texts])
```

## Mistake 4: Bare Except
```python
# WRONG — hides all errors including bugs
try:
    result = call_llm()
except:  # Catches KeyboardInterrupt, SystemExit, everything!
    return None

# RIGHT
try:
    result = call_llm()
except RateLimitError:
    wait_and_retry()
except AuthError:
    raise  # Can't recover
except Exception as e:
    logger.error(f"Unexpected: {e}", exc_info=True)
    raise
```

## Mistake 5: Not Stripping LLM JSON Output
```python
# WRONG — fails when LLM wraps in ```json ... ```
result = json.loads(llm_output)  # JSONDecodeError!

# RIGHT
def parse_llm_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
```

## Mistake 6: Hardcoding Model Names as Strings
```python
# WRONG — typos silently break things
model = "gpt4o"  # Should be "gpt-4o"

# RIGHT — use an Enum
class Model(str, Enum):
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
```

## Mistake 7: Not Pinning Package Versions
```
# WRONG — could install anything
openai

# RIGHT — pin exact versions
openai==1.35.0
```
A breaking change in `openai` v2 broke all existing code. Pinned versions prevent this.
