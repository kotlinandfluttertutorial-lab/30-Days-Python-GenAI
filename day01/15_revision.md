# Day 01 — Revision
## Quick Reference for Everything You Learned Today

---

## 5-Minute Revision

Read this before every interview. Read it every morning for the first week.

---

## Core Definitions (Memorize These)

| Term | Definition |
|------|-----------|
| **AI** | Systems that perform tasks requiring human intelligence |
| **ML** | Learning patterns from data without explicit rules |
| **Deep Learning** | ML using multi-layer neural networks |
| **LLM** | Neural network trained on text data to predict next tokens |
| **GenAI** | AI that generates new content (text, image, code) |
| **Transformer** | Architecture using self-attention for sequence processing |
| **Embedding** | Dense vector representing semantic meaning |
| **Vector DB** | Database optimized for similarity search on vectors |
| **RAG** | Retrieve relevant docs, inject as context before LLM generates |
| **Agent** | LLM + tools + memory + planning in a reasoning loop |
| **Context Window** | Maximum tokens an LLM can process in one call |
| **Temperature** | Controls randomness of LLM output sampling |
| **Tokenization** | Splitting text into subword units (tokens) |
| **Hallucination** | LLM generating confident but false information |
| **Fine-tuning** | Additional training to change model behavior/style |
| **LLMOps** | Operations practices for LLM systems in production |
| **Prompt Engineering** | Crafting prompts to guide LLM toward desired outputs |

---

## Critical Comparisons

### RAG vs Fine-Tuning
- RAG: for factual knowledge, instant updates, keeps docs local
- Fine-tuning: for behavior/style changes, slow and expensive to update

### LLM vs Agent
- LLM: single request-response turn, no tools, no memory
- Agent: multi-step loop, tools, memory, planning

### Traditional Software vs AI Software
- Traditional: deterministic, unit testable, fails with exceptions
- AI: probabilistic, needs evaluation metrics, fails silently

### AI Engineer vs ML Engineer
- AI Engineer: builds applications using pre-trained models
- ML Engineer: trains and optimizes models

### SQL DB vs Vector DB
- SQL: exact queries, structured data
- Vector DB: similarity search, high-dimensional vectors

---

## The Complete AI Stack

```
UI (Web/Mobile/CLI)
↓
API Gateway (rate limiting, auth)
↓
FastAPI Backend
↓
AI Orchestrator (RAG + Agents)
↓
LLM (OpenAI / Anthropic / Local)
↓
Storage (PostgreSQL + Redis + Vector DB)
↓
Observability (logs + metrics + evals)
```

---

## RAG Pipeline (KNOW THIS COLD)

**Ingestion (offline):**
```
Documents → Load → Clean → Chunk → Embed → Store in Vector DB
```

**Query (online):**
```
Query → Embed → Similarity Search → Rerank → Context → LLM → Answer
```

---

## Agent Loop (KNOW THIS COLD)

```
Goal → Think (LLM) → Act (tool call) → Observe (result) → Think → Act → ... → Answer
```

---

## Token Economics

```
1 token ≈ 0.75 words ≈ 4 characters
128K context ≈ 90,000 words ≈ 300 pages

GPT-4o:     $5/1M input, $15/1M output
Claude 3.5: $3/1M input, $15/1M output
Groq Llama: ~$0.1/1M (near free)
Ollama:     Free (local)

Typical RAG request: 2,000 tokens = $0.01
At 10,000 req/day = $100/day = $3,000/month
```

---

## Cosine Similarity Formula

```
cos(θ) = (A · B) / (|A| × |B|)

Where:
A · B = dot product = Σ(a_i × b_i)
|A| = magnitude = √(Σ a_i²)

Result: -1 (opposite) to 1 (identical)
```

---

## Must-Know Code Patterns

### Load Environment Variables
```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]  # Raises if missing
```

### Basic LLM Call Pattern
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
    ],
    temperature=0.7,
    max_tokens=500,
)
answer = response.choices[0].message.content
```

### Cosine Similarity
```python
import math
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0
```

---

## Interview One-Liners (Memorize)

1. **"What is RAG?"** → "Semantic retrieval of relevant document chunks injected as context before LLM generation."

2. **"What is hallucination?"** → "LLM generating statistically likely but factually wrong content because there's no truth verification mechanism."

3. **"LLM vs Agent?"** → "LLM is a single turn. Agent loops: think → tool call → observe → repeat."

4. **"What is an embedding?"** → "Dense vector mapping text to a space where semantic similarity equals geometric proximity."

5. **"RAG vs fine-tuning?"** → "RAG for factual retrieval (instant updates). Fine-tuning for behavioral changes (expensive, slow)."

---

## Self-Test Questions

Answer these without notes. If you can't, re-read the relevant section.

1. What is a token?
2. What does temperature control?
3. Why does RAG reduce hallucination?
4. What is the difference between a 128K and a 1M context window practically?
5. Why can't you mix embeddings from different models?
6. What happens in the agent loop?
7. Why use cosine similarity instead of Euclidean distance?
8. What is the difference between pretraining and fine-tuning?
9. Name three reasons to use Docker in AI applications.
10. What is prompt injection?

---

## Concepts That Appear in Every Interview

These topics came up in 90%+ of AI Engineer interviews surveyed:
- RAG architecture and implementation
- LLM context window limits and strategies
- Hallucination and mitigation
- Prompt engineering techniques
- Embeddings and vector similarity
- Agent vs RAG decision criteria
- Production observability (logs, metrics, evals)
- Cost optimization

If you can answer these confidently, you're ready for most AI Engineer interviews.
