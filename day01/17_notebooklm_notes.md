# Day 01 — NotebookLM Notes
## AI Engineer Foundation

*Optimized for Google NotebookLM import. Clean structure for AI-powered study.*

---

## Day 1 Overview

**Topic:** AI Engineer Foundation  
**Phase:** Phase 1 — Foundation  
**Key Theme:** Understanding the complete AI engineering landscape before building anything

---

## Key Concepts

### 1. Artificial Intelligence
- Systems that perform tasks requiring human intelligence
- Three approaches: symbolic (rules), statistical (ML), neural (deep learning)
- Modern AI Engineer primarily works with deep learning and LLMs

### 2. Machine Learning
- Learning from data instead of explicit rules
- Types: supervised (labeled data), unsupervised (no labels), reinforcement (rewards)
- Output: a trained model that generalizes to new inputs

### 3. Deep Learning
- ML using multi-layer neural networks
- Learns features automatically from raw data
- Foundation: CNN (vision), RNN/LSTM (sequences), Transformer (everything modern)

### 4. Large Language Model (LLM)
- Transformer-based neural network with billions of parameters
- Trained to predict next token given previous tokens
- Trained phases: pretraining → SFT → RLHF/DPO
- Key properties: emergent capabilities, context window, temperature-controlled sampling

### 5. Transformer Architecture
- Introduced: "Attention Is All You Need" (Google, 2017)
- Core mechanism: self-attention (each token attends to all others)
- Types: encoder (BERT), decoder (GPT), encoder-decoder (T5)
- Solved: RNN sequential bottleneck, long-range dependency loss

### 6. Embedding
- Dense vector mapping text to semantic space
- Similar meanings → similar vectors (small cosine distance)
- Dimensions: 384 (small) to 3072 (large)
- Critical rule: cannot mix embeddings from different models

### 7. Vector Database
- Purpose: efficient similarity search on high-dimensional vectors
- Algorithm: HNSW (Hierarchical Navigable Small World) — approximate nearest neighbor
- Operations: insert, query (top-K), filter by metadata
- Examples: ChromaDB, FAISS, pgvector, Pinecone, Qdrant

### 8. RAG (Retrieval-Augmented Generation)
- Solves: hallucination + knowledge cutoff
- Ingestion: document → chunk → embed → store in vector DB
- Query: embed query → similarity search → retrieve chunks → prompt → LLM → answer
- Key advantage over fine-tuning: instant updates, lower cost, cites sources

### 9. AI Agent
- Components: LLM + tools + memory + planning
- Pattern: Think → Act (tool call) → Observe → repeat
- Use when: multi-step tasks, unknown path, requires external data
- Failure modes: loops, prompt injection, context overflow

### 10. Context Window
- Maximum tokens in one LLM call (system + history + context + query + response)
- GPT-4o: 128K, Claude: 200K, Gemini: 1M
- "Lost in the middle" problem: model recalls beginning/end better than middle
- Larger window ≠ better — still use retrieval for precision and cost

---

## Important Definitions

**Token:** Smallest unit LLM processes. ~0.75 words, ~4 chars. Determines cost.

**Temperature:** Divides logits before softmax. Low=deterministic, High=creative.

**Hallucination:** LLM generates likely but false content. No verification mechanism.

**Fine-tuning:** Additional training to change model behavior/style. Not for knowledge.

**Cosine Similarity:** cos(θ) = (A·B)/(|A|·|B|). Angle between vectors. Range: -1 to 1.

**Prompt Engineering:** Crafting inputs that reliably guide LLM to desired outputs.

**LLMOps:** Operations for LLM systems: prompt versioning, evaluation, cost monitoring.

---

## Mental Models

### Mental Model 1: LLM as Pattern Completer
```
LLM doesn't "know" things — it "continues" text based on patterns.
"The capital of France is ___" → high probability for "Paris" because it appeared
millions of times in training data.
```

### Mental Model 2: Embeddings as Semantic GPS
```
Every piece of text has coordinates in semantic space.
Similar meanings → close coordinates.
Vector DB = GPS navigation system for semantic space.
```

### Mental Model 3: RAG as External Memory
```
LLM has no long-term memory (just training weights).
RAG gives it external memory: retrieve relevant "memories" and inject into context.
```

### Mental Model 4: Agent as Autonomous Problem Solver
```
Give an agent a goal, not a script.
Agent decides the steps: which tools, in what order, for how many iterations.
This is power and danger.
```

### Mental Model 5: Token Economy
```
Every LLM interaction has a price: tokens × cost_per_token.
Architect systems to minimize unnecessary tokens.
Cache, compress, route to cheaper models where possible.
```

---

## Architecture

### Minimal AI Application
```
User → HTTP → FastAPI → LLM API → Response
```

### RAG Architecture
```
[Offline] Documents → Chunk → Embed → Vector DB
[Online]  Query → Embed → Search → Chunks → Prompt → LLM → Answer
```

### Agent Architecture
```
Goal → [LLM: Think] → [Tool: Act] → [Observe] → Loop until done
```

### Production AI Stack
```
UI → API Gateway → FastAPI → Orchestrator → LLM
                              ↓         ↑
                       Vector DB   PostgreSQL   Redis
                              ↓
                        Observability (logs + metrics + evals)
```

---

## Code Patterns

### Cosine Similarity
```python
import math
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))
```

### Safe API Key Loading
```python
from dotenv import load_dotenv; import os
load_dotenv()
key = os.environ["OPENAI_API_KEY"]  # Raises KeyError if missing
```

### Basic LLM Call
```python
from openai import OpenAI
client = OpenAI()
r = client.chat.completions.create(
    model="gpt-4o", temperature=0.7,
    messages=[{"role":"user","content":"What is RAG?"}]
)
```

---

## Interview Facts

1. LLM = transformer predicting next token over vocabulary of 50K+ subwords
2. RAG reduces hallucination by grounding responses in retrieved facts
3. Context window = max tokens in one call (prompt + response combined)
4. Embeddings cannot be mixed across different models
5. Cosine similarity is magnitude-invariant (unlike Euclidean distance)
6. Temperature=0 → deterministic (same logit selection always)
7. Agent failure modes: loops, injection, overflow, non-determinism compounds
8. RAG preferred over fine-tuning for knowledge: instant update, cheaper, cites sources
9. AI Engineer builds applications; ML Engineer trains models
10. Hallucination rate is a metric to measure, not just a risk to mention

---

## Common Traps

- "RAG is just a database" → No: it's semantic vector similarity, not keyword lookup
- "Temperature=0 prevents hallucination" → No: temperature ≠ factual accuracy
- "Fine-tune for company knowledge" → Wrong: use RAG; fine-tune for behavior changes
- "Mix embeddings from different models" → Impossible: incompatible vector spaces
- Using `shell=True` with subprocess and user input → Command injection vulnerability
- No API key validation before use → Cryptic AuthenticationError at runtime
- No rate limiting on LLM endpoints → $10K bill from a bot attack

---

## Important Comparisons

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Use case | Factual knowledge | Behavior/style |
| Update speed | Instant | Hours-days |
| Cost | Low | High |
| Hallucination | Reduced | Still possible |
| Privacy | Local | Shared with provider |

| Dimension | LLM | Agent |
|-----------|-----|-------|
| Steps | One | Multiple |
| Tools | None | Many |
| Memory | Context only | Persistent |
| Cost | One call | Multiple calls |
| Reliability | High | Lower |

---

## Questions to Review

1. Can you explain the full RAG pipeline in under 2 minutes?
2. What is the formula for cosine similarity?
3. Why can't you compare embeddings from different models?
4. What three things does an AI Agent have that an LLM chatbot doesn't?
5. What is the difference between temperature and top-p sampling?
6. What is "lost in the middle" and how does it affect RAG design?
7. Why is hallucination rate a metric, not just a risk?
8. What are the four phases of AI model training?

---

## Summary

Day 1 established the complete mental model for the 30-day program.

**The key insight:** AI Engineers build applications *using* LLMs, not *training* them. The core skill is knowing how to combine: LLM APIs + RAG + Agents + FastAPI + Vector DBs + Docker + Evaluation into production AI systems.

**The core architecture:** Every AI application is a combination of: user interface → API backend → AI orchestration (RAG/Agents) → LLM → storage → observability.

**The hardest insight:** AI systems fail silently. Traditional debugging finds exceptions. AI debugging requires evaluation metrics, logging, and continuous monitoring.

**What comes next:** Day 2 deepens Python fundamentals. Day 3 adds NumPy and Pandas. By Day 15, you'll implement RAG from scratch. By Day 30, you'll design enterprise AI systems.
