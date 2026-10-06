# Day 02 — Infographics
## Python for AI Engineering: Visual Reference

---

## Infographic 1: Python Data Types Decision Tree

```
What do you need to store?
         │
    ┌────┴────┐
    │         │
 Ordered?   Unordered?
    │            │
 ┌──┴──┐     ┌──┴──┐
 │     │     │     │
Mut? Immut? Key?  No Key?
 │      │    │      │
List  Tuple  Dict   Set

LIST  [1, 2, 3]        — ordered, mutable, duplicates OK
                         Use: document chunks, message history, embeddings

TUPLE (1, 2, 3)        — ordered, IMMUTABLE, duplicates OK
                         Use: function returns, named tuples, config constants

DICT  {"key": "val"}   — key-value, mutable, keys unique
                         Use: metadata, API requests, JSON data

SET   {1, 2, 3}        — unordered, mutable, NO duplicates
                         Use: deduplication, membership testing

AI USAGE FREQUENCY:
Dict   ████████████████████ 35% (API messages, metadata)
List   ████████████████     30% (chunks, embeddings, results)
String ████████████         20% (text, prompts)
Tuple  ████                  8% (immutable pairs, configs)
Set    ███                   7% (deduplication, keywords)
```

---

## Infographic 2: Sync vs Async Execution Model

```
SYNCHRONOUS — One thing at a time:
────────────────────────────────────────────────────────
Thread 1: [LLM Call 1: 2s] [LLM Call 2: 2s] [LLM Call 3: 2s]
          ←────────────────── 6 seconds ──────────────────→

ASYNCHRONOUS — Concurrent with one thread:
────────────────────────────────────────────────────────
Event Loop: 
  Task 1: [START]─────[WAIT 2s]─────[DONE]
  Task 2:      [START]──[WAIT 2s]──[DONE]
  Task 3:           [START][WAIT 2s][DONE]
             ←──── ~2 seconds ────→

KEY: `await` = "pause me, run something else while I wait"
     Only works for I/O bound tasks (API calls, DB, file I/O)
     NOT for CPU bound tasks (matrix multiply, image processing)

WHEN TO USE ASYNC:
  ✓ LLM API calls
  ✓ Embedding API calls
  ✓ Database queries
  ✓ HTTP requests
  ✗ NumPy computations (use multiprocessing)
  ✗ PyTorch inference (use threading or separate process)
```

---

## Infographic 3: Generator Flow

```
REGULAR FUNCTION:           GENERATOR FUNCTION:

def get_chunks(text):       def stream_chunks(text):
    result = []                 for chunk in split(text):
    for chunk in split(text):       yield chunk       ← pause here
        result.append(chunk)
    return result           
                            
call: chunks = get_chunks() call: gen = stream_chunks()
      ↓                           ↓
ALL chunks computed         NOTHING computed yet!
ALL in memory               
                            next(gen) → first chunk (computed now)
                            next(gen) → second chunk (computed now)
                            ...

MEMORY USAGE:
Regular: O(n) — all chunks in RAM at once
Generator: O(1) — one chunk at a time

FOR AI: 1 million document chunks
  Regular: 1M × 500 bytes = 500MB RAM
  Generator: 500 bytes RAM (one at a time!)
```

---

## Infographic 4: Exception Handling Hierarchy

```
BaseException
    ├── KeyboardInterrupt  ← Ctrl+C (never catch this!)
    ├── SystemExit         ← sys.exit() (never catch this!)
    └── Exception          ← Everything you should handle
            ├── ValueError     ← Bad input value
            ├── TypeError      ← Wrong type
            ├── KeyError       ← Missing dict key
            ├── IOError        ← File/network error
            │   └── FileNotFoundError
            └── Your Custom Exceptions
                    └── AIServiceError
                            ├── LLMError
                            │   ├── LLMRateLimitError  → wait + retry
                            │   ├── LLMAuthError       → fail fast
                            │   └── LLMTimeoutError    → retry
                            └── RAGError
                                ├── RetrievalError     → fallback
                                └── EmbeddingError     → retry

RULE: Catch the MOST SPECIFIC exception you can handle.
      Let unhandled exceptions propagate up.
      NEVER: except: (bare) or except Exception: pass
```

---

## Infographic 5: Type Hints Cheat Sheet

```
BASIC TYPES:
  str          "hello"
  int          42
  float        3.14
  bool         True/False
  None         None (absence of value)

CONTAINER TYPES:
  list[str]          ["a", "b", "c"]
  dict[str, float]   {"score": 0.9}
  set[str]           {"a", "b"}
  tuple[str, int]    ("name", 42)

OPTIONAL (value or None):
  str | None         → parameter might not be provided
  Optional[str]      → same thing (older syntax)

UNION (multiple types):
  int | str          → could be either
  Union[int, str]    → same (older syntax)

CALLABLE:
  Callable[[str], float]    → function(str) → float
  Callable[..., str]        → any args → str

GENERICS:
  list[list[float]]         → list of vectors
  dict[str, list[str]]      → dict of string lists

AI-SPECIFIC TYPE ALIASES:
  Vector = list[float]           # embedding vector
  MessageList = list[dict[str,str]]  # chat messages
  SearchResults = list[tuple[str, float]]  # (id, score)
```
