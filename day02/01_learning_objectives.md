# Day 02 — Learning Objectives
## Python for AI Engineering

---

## What You Will Accomplish Today

By the end of Day 2, you will be able to write the Python that powers every AI system in this program.

### Knowledge Objectives

1. **Master Python data types** used in AI: lists, dicts, tuples, sets, and when each applies
2. **Write type-hinted functions** — mandatory in production AI code
3. **Use comprehensions** to process data cleanly and efficiently
4. **Handle exceptions** the right way — not just catching everything with bare `except`
5. **Work with JSON** — the data format of every LLM API
6. **Use async/await** — required for non-blocking LLM API calls
7. **Write generators** — used for streaming LLM responses
8. **Use modules and packages** correctly in a project structure
9. **Handle files** — loading documents for RAG ingestion
10. **Understand iterators** deeply — essential for data pipelines

### Skill Objectives

1. **Write a complete Python module** with proper structure
2. **Process a dataset** using comprehensions and functional patterns
3. **Build an async LLM caller** that handles multiple concurrent requests
4. **Write a streaming response handler** using generators
5. **Implement proper exception hierarchy** for an AI service

### Project Objective

Build the **AI Data Processing CLI** — a command-line tool that:
- Loads and parses document files (text, JSON, CSV)
- Processes and cleans the data
- Estimates token counts and costs
- Prepares documents for RAG ingestion
- Demonstrates every Python pattern from today

---

## Why This Day Matters

Day 1 gave you the mental model. Day 2 gives you the tools to build it.

Every single pattern you learn today appears in production AI code:

| Python Pattern | Where It Appears in AI |
|---------------|----------------------|
| Type hints | Every function signature in FastAPI, Pydantic |
| Dataclasses | LLM message objects, config objects |
| async/await | FastAPI endpoints, concurrent LLM calls |
| Generators | Streaming LLM responses |
| Exception handling | LLM API error recovery |
| JSON | LLM API request/response format |
| List comprehensions | Batch embedding, document processing |
| Dict comprehensions | Metadata aggregation |
| Context managers | Database connections, file handles |
| Decorators | FastAPI routes, caching, retry logic |

---

## Today's 12-Hour Schedule

| Time | Activity | Duration |
|------|----------|----------|
| Hour 1 | Review Day 1 summary + Today's objectives | 60 min |
| Hours 2–3 | Study concepts: data types, functions, OOP basics | 120 min |
| Hour 4 | Study deep dive: async, generators, decorators | 60 min |
| Hour 5 | Code: `01_python_fundamentals.py` | 60 min |
| Hour 6 | Code: `02_async_python.py` | 60 min |
| Hour 7 | Code: `03_json_and_files.py` | 60 min |
| Hours 8–9 | Build the project: AI Data Processing CLI | 120 min |
| Hour 10 | Architecture + infographics | 60 min |
| Hour 11 | Interview questions | 60 min |
| Hour 12 | Revision + Assessment | 60 min |

---

## Connection to Day 1

```
Day 1: You learned WHAT an AI system is
Day 2: You learn HOW to write the Python that builds it

Day 1's LLM Client (02_llm_client.py) used:
- Type hints          ← Today: master these
- Dataclasses         ← Today: master these
- Enum                ← Today: covers this
- Exception handling  ← Today: master these
- async/await         ← Today: master these
```

---

## Prerequisites Check

From Day 1 you need:
- [ ] Python 3.11+ installed
- [ ] Virtual environment working
- [ ] Concept of type hints (seen in Day 1 code)
- [ ] Basic programming experience (functions, loops, conditionals)
