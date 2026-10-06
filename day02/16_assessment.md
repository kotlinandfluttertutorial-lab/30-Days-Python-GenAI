# Day 02 — Assessment

---

## Quick Assessment (60 minutes)

### Theory (5 × 10 = 50 pts)
1. Why does `asyncio.gather()` outperform sequential `await` calls for LLM batches?
2. What is the difference between a generator and a list comprehension?
3. Name three Python patterns specific to AI engineering and explain each.
4. What goes wrong with `@dataclass` if you use `items: list = []` as a default?
5. Explain the exception handling strategy for LLM rate limits.

### Coding (3 × 20 = 60 pts)
1. Write a `ConversationHistory` class with `add_user`, `add_assistant`, `truncate(n)`, and `token_estimate`.
2. Write an async function that embeds 20 texts concurrently in batches of 5.
3. Write a `safe_load_config(path: str) -> dict` that handles `FileNotFoundError` and `JSONDecodeError` with appropriate messages.

### Architecture (1 × 40 = 40 pts)
Design a Python module structure for an AI document Q&A system. Show the directory tree, key classes in each module, and the data flow from user question to final answer.

**Total: 150 points**

**Pass: 112/150 (75%)**
