# Mock Interview — AI Engineer
## Complete 12-Round Interview Simulation

---

## How to Use This Document

1. Read each question out loud
2. Answer out loud (not in your head — verbalize everything)
3. Time yourself: recruiter questions 30s-1min, technical 2-5min, design 10-15min
4. Record yourself — you'll catch filler words and unclear explanations
5. Check your answer against the guide below

---

## Round 1: Recruiter Screen (15 minutes)

**Q1.** Tell me about yourself and why you want to be an AI Engineer.

**Q2.** What AI/ML projects have you built?

**Q3.** Have you used any LLM APIs (OpenAI, Anthropic)?

**Q4.** What is RAG and why is it important?

**Q5.** What does an AI Engineer do differently from a data scientist?

**Guide for R1:**
- Q1: Use the 60-second introduction from `AI_ENGINEER_INTRODUCTION.md`
- Q2: Mention 2-3 projects from the program with specific technologies
- Q4: "RAG: retrieves relevant documents, injects as context before LLM generation — reduces hallucination and handles private knowledge bases"

---

## Round 2: Python Round (30 minutes)

**Q1.** Walk me through why we use async for AI backends.

**Q2.** Show me a retry decorator for LLM API calls.
```python
# Write this from memory:
def retry_with_backoff(max_attempts=3, delay=1.0):
    ...
```

**Q3.** This code has bugs — find them all:
```python
@dataclass
class EmbeddingCache:
    cache: dict = {}
    
async def embed_all(texts):
    return [get_embedding(t) for t in texts]
    
def parse_llm_output(text):
    return json.loads(text)
```

**Q4.** How would you make this code production-ready:
```python
api_key = "sk-hardcoded-key"
result = call_llm(api_key, prompt)
```

**Q5.** What is `asyncio.to_thread()` and when do you need it?

**Guide for R2:**
- Q3 bugs: (1) mutable default `{}` in dataclass — use `field(default_factory=dict)`, (2) missing `await` in async function — not truly concurrent, (3) JSON parsing can fail if LLM wraps in markdown fences
- Q4: load from env, validate not None, raise descriptive error

---

## Round 3: ML Round (30 minutes)

**Q1.** Explain bias-variance tradeoff. Where does Random Forest fit?

**Q2.** I have an imbalanced dataset (95% class A, 5% class B). What metric do I use and why not accuracy?

**Q3.** Walk me through preventing data leakage in a feature engineering pipeline.

**Q4.** Our model has 98% training accuracy but 62% test accuracy. What happened and how do you fix it?

**Q5.** Code challenge: implement train/val/test split with stratification.

**Guide for R3:**
- Q2: Use F1, PR-AUC, or ROC-AUC. Accuracy is misleading — always predicting A = 95% accuracy with zero predictive value.
- Q4: Severe overfitting. Fixes: regularization, dropout, more data, simpler model, feature selection, cross-validation to detect earlier.

---

## Round 4: Deep Learning Round (30 minutes)

**Q1.** Write the PyTorch training loop from memory (5 steps).

**Q2.** Why is `model.eval()` required during inference? What breaks if you forget it?

**Q3.** Explain backpropagation intuitively. No math.

**Q4.** Why does zero weight initialization fail for neural networks?

**Q5.** You're training a model and loss is NaN after epoch 2. What do you check?

**Guide for R4:**
- Q2: Dropout is active in train mode → random neurons zeroed → different outputs each call. BatchNorm uses batch statistics vs running statistics. Forgetting eval() = non-deterministic inference.
- Q5: Exploding gradients (check grad norm), too high LR, NaN in input data, log of negative value in loss.

---

## Round 5: LLM Round (30 minutes)

**Q1.** Explain LLM inference step by step from input tokens to output.

**Q2.** What is temperature=0.1 vs temperature=1.5? When do you use each?

**Q3.** A user complains our AI assistant made up a statistic. What caused it and how do you fix it?

**Q4.** Calculate the cost: 1,000 requests/day to GPT-4o with avg 2,000 input tokens and 300 output tokens.

**Q5.** How do you implement streaming with the OpenAI SDK?

**Guide for R5:**
- Q4: Input: (1000 × 2000 / 1M) × $5 = $10/day. Output: (1000 × 300 / 1M) × $15 = $4.50/day. Total: ~$14.50/day = ~$435/month.
- Q5: `stream=True` parameter → iterate `for chunk in stream: yield chunk.choices[0].delta.content`

---

## Round 6: RAG Round (45 minutes)

**Q1.** Walk me through a complete RAG system end-to-end. Don't skip any steps.

**Q2.** What is hybrid search? Code the score combination.

**Q3.** Our RAG system returns irrelevant answers. Walk me through debugging it.

**Q4.** What is faithfulness in RAG evaluation? How do you measure it?

**Q5.** Design question: We have 5 million documents and need query latency < 2 seconds. What's your architecture?

**Q6.** What is the difference between a cross-encoder and a bi-encoder?

**Guide for R6:**
- Q3: (1) Print chunks — are they relevant? (2) Check similarity scores — all low? wrong model. (3) Try known-good query. (4) Is metadata filter too restrictive? (5) If chunks relevant but answer wrong → system prompt issue.
- Q5: pgvector with IVF index, Redis cache (target >30% hit rate), reranker, 3 chunks max. LLM: gpt-4o-mini for speed and cost.

---

## Round 7: Agents Round (30 minutes)

**Q1.** Explain the ReAct pattern. Why is it important?

**Q2.** Show me the structure of an agent system with tool calling.

**Q3.** How do you prevent infinite loops in an agent?

**Q4.** What is prompt injection in agents? How do you defend against it?

**Q5.** When would you use an agent vs a RAG pipeline?

**Guide for R7:**
- Q2: Tool schemas → LLM call → check `finish_reason=="tool_calls"` → execute → append as "tool" role → continue loop → final answer
- Q5: RAG for Q&A over documents (fast, cheap, reliable). Agent for tasks where the path is unknown, multiple tools needed, or multi-step reasoning required.

---

## Round 8: Backend Round (30 minutes)

**Q1.** Write a FastAPI endpoint for RAG queries with auth and rate limiting.

**Q2.** How does `Depends()` work and why is it useful?

**Q3.** How do you implement streaming in FastAPI?

**Q4.** A user reports our API is slow. Walk me through diagnosing the bottleneck.

**Q5.** What is `@app.on_event("startup")` / `lifespan` used for in an AI app?

**Guide for R8:**
- Q4: Check p95 latency breakdown: auth overhead? DB query? embedding call? LLM call? Cache miss rate? Add request timing middleware to measure each phase.
- Q5: Load ML model, connect to database, initialize LLM client, warm up vector index — operations that should happen once at startup, not per request.

---

## Round 9: System Design Round (45 minutes)

**Design Challenge:** "Design a Q&A system for a law firm with 200,000 case documents. Requirements: <3s response time, document-level access control, citations required, 500 concurrent users."

**Interviewer follow-ups:**
- "How do you handle very long documents?"
- "What if two users have access to different documents?"
- "The LLM starts making up cases that don't exist — how do you detect it?"
- "How do you reduce costs as usage grows?"
- "How would you scale to 5,000 users?"

**Guide for R9:**
- Draw the architecture first: clients → API gateway → FastAPI → RAG → LLM
- Access control: user_id in chunk metadata, filter all vector searches by user's document IDs
- Hallucination detection: faithfulness eval — scan answer for claims not in retrieved chunks
- Cost reduction: caching, smaller model for classification/routing, reduce chunk count
- Scaling: more FastAPI replicas, Redis cluster, migrate to Qdrant if vector search bottlenecks

---

## Round 10: AI Security Round (20 minutes)

**Q1.** What is prompt injection? Give me an example attack and defense.

**Q2.** How do you handle PII in a RAG system?

**Q3.** One of our tools is a file reader. What security checks do you add?

**Q4.** How do you audit AI system actions for compliance?

**Guide for R10:**
- Q3: Validate path (no `..`, no `/etc`, no system dirs), limit file types, limit file size, check user has permission for that path.

---

## Round 11: LLMOps Round (20 minutes)

**Q1.** What should be in every log entry for an LLM request?

**Q2.** How do you detect when your prompt change degraded quality?

**Q3.** We're spending $50K/month on LLM APIs. Walk me through reducing costs.

**Q4.** How do you version prompts like code?

**Guide for R11:**
- Q1: trace_id, timestamp, user_id, model, input_tokens, output_tokens, cost_usd, latency_ms, cache_hit, finish_reason, eval_score
- Q3: (1) Measure by endpoint/model. (2) Route simple queries to mini models. (3) Cache: target 30%+ hit rate. (4) Reduce context: fewer/smaller chunks. (5) Compress prompts.

---

## Round 12: Behavioral Round (20 minutes)

**Q1 (STAR):** Tell me about a time you debugged a production system failure.

**Q2 (STAR):** Describe a technical decision where you had to balance quality and speed.

**Q3 (STAR):** Tell me about a time you had to learn a new technology quickly.

**Q4:** Where do you see AI engineering going in the next 2 years?

**Q5:** Why do you want to work specifically in AI engineering?

**Guide for R12:**
- Q4 (2026 perspective): Multimodal agents become mainstream, LLM costs drop 10x enabling wider adoption, evaluation and observability become as standardized as application monitoring, agentic systems handle complex multi-day workflows, on-device models reduce cloud dependency.
- Q5: Connect to your genuine motivation — most important answer of the interview.

---

## Scoring Your Mock Interview

Rate each round 1-5 after practicing:

| Round | Topic | Self-Score | Notes |
|-------|-------|-----------|-------|
| 1 | Recruiter | /5 | |
| 2 | Python | /5 | |
| 3 | ML | /5 | |
| 4 | Deep Learning | /5 | |
| 5 | LLM | /5 | |
| 6 | RAG | /5 | |
| 7 | Agents | /5 | |
| 8 | Backend | /5 | |
| 9 | System Design | /5 | |
| 10 | Security | /5 | |
| 11 | LLMOps | /5 | |
| 12 | Behavioral | /5 | |
| **Total** | | /60 | |

**Score interpretation:**
- 50-60: Ready. Apply now.
- 40-49: Almost ready. Practice weak rounds.
- Below 40: Review relevant days, practice daily for 1 more week.
