# AI Engineer Master Notes
## Single-Document Final Revision Guide

---

## PART 1: FOUNDATIONS

### The AI Stack
```
User → Frontend → API Gateway → FastAPI → AI Orchestrator
                                              ↓
                                     RAG Pipeline | Agent System
                                              ↓
                                      LLM (OpenAI/Anthropic/Local)
                                              ↓
                              PostgreSQL | Redis | Vector DB
                                              ↓
                                  Observability (Logs/Metrics/Evals)
```

### Core Formulas (Memorize)
```
Cosine similarity:    dot(a,b) / (|a| × |b|)
Softmax:              exp(xᵢ) / Σexp(xⱼ)
Temperature scaling:  p_i = exp(logit_i / τ)
Attention:            softmax(QK^T / √d_k) × V
MSE:                  mean((y - ŷ)²)
Cross-entropy:        -mean(y × log(ŷ))
Gradient descent:     θ = θ - lr × dL/dθ
```

---

## PART 2: MACHINE LEARNING

### Algorithm Selection
```
Continuous target  → Linear Regression (baseline) → Random Forest
Binary/multi-class → Logistic Regression (baseline) → Random Forest
Find groups        → K-Means (choose K with elbow method)
High-dimensional   → PCA first, then algorithm
```

### Evaluation Metrics
```
Accuracy:      (TP+TN)/total            — balanced classes only
Precision:     TP/(TP+FP)              — false positive cost high
Recall:        TP/(TP+FN)              — false negative cost high
F1:            2×P×R/(P+R)            — balance P and R
ROC-AUC:       area under curve        — model comparison
R²:            1 - SS_res/SS_tot       — regression quality
RMSE:          √MSE                    — same units as target
```

### Critical Rules
- ALWAYS split before fitting preprocessors (data leakage)
- Use `Pipeline(scaler, model)` for CV to prevent leakage
- `fit_transform` on train only; `transform` on val/test
- `random_state=42` everywhere for reproducibility

---

## PART 3: DEEP LEARNING

### PyTorch Training Loop (5 steps — memorize)
```python
for X, y in loader:
    optimizer.zero_grad()          # 1. Clear gradients
    logits = model(X)              # 2. Forward pass
    loss = criterion(logits, y)    # 3. Compute loss
    loss.backward()                # 4. Backprop
    optimizer.step()               # 5. Update weights
```

### Activation Functions
```
Hidden layers: ReLU (default), LeakyReLU (if dying ReLU)
Binary output: Sigmoid
Multi-class output: Softmax
Transformers: GELU
```

### Key Rules
- `model.train()` for training (enables Dropout, BN train mode)
- `model.eval()` + `torch.no_grad()` for inference
- He init: `std = √(2/fan_in)` — designed for ReLU
- AdamW: Adam + weight decay; standard for transformer fine-tuning

---

## PART 4: TRANSFORMERS

### Self-Attention (The Core Formula)
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q = "what am I looking for?"
K = "what do I have?"
V = "the actual information"
√d_k = prevents saturation of softmax for large d_k
```

### Transformer Block Components (in order)
```
Input → LayerNorm → Multi-Head Attention → Add (residual)
      → LayerNorm → Feed-Forward (×4 width, GELU) → Add (residual)
      → Output
```

### Encoder vs Decoder
```
Encoder (BERT): bidirectional attention → embeddings, classification
Decoder (GPT):  causal mask → text generation, LLM
Enc-Dec (T5):   encoder reads, decoder generates → translation
```

---

## PART 5: LLMs

### Inference Pipeline
```
Input tokens → Transformer → Logits(50K vocab) → /temperature →
top-k filter → top-p filter → softmax → sample → output token → repeat
```

### Key Parameters
```
temperature=0:   greedy, deterministic
temperature=0.7: balanced (default)
temperature>1.5: very random, avoid
top_k=50:        consider top 50 tokens
top_p=0.9:       cover 90% probability mass
```

### LLM Cost Estimate (numbers)
```
GPT-4o:           $5/1M input, $15/1M output
GPT-4o-mini:      $0.15/1M input, $0.60/1M output
Claude 3.5 Sonnet: $3/1M input, $15/1M output
Groq Llama 70B:   ~$0.59/1M (fast, cheap)
```

### Hallucination Mitigations (priority order)
1. RAG — ground in retrieved facts
2. Temperature=0 for factual tasks
3. System prompt: "only answer from context"
4. Structured output with citation fields
5. Second LLM call to verify

---

## PART 6: PROMPT ENGINEERING

### Pattern Hierarchy
```
1. Zero-shot (just instructions)
2. Few-shot (add 2-5 examples)
3. Chain-of-thought (step by step)
4. Structured output (JSON schema)
5. System prompt (persistent rules)
```

### JSON from LLM (always do this)
```python
raw = raw.strip()
if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"): raw = raw[4:]
if raw.endswith("```"): raw = raw[:-3]
data = json.loads(raw.strip())
```

### Injection Defense
```python
INJECTION_PATTERNS = ["ignore previous", "you are now", "forget your"]
for pattern in INJECTION_PATTERNS:
    if pattern in user_input.lower():
        return "I cannot process this request."
```

---

## PART 7: EMBEDDINGS

### Key Facts
```
all-MiniLM-L6-v2:      384 dims, FREE, fast
text-embedding-3-small: 1536 dims, $0.02/1M
text-embedding-3-large: 3072 dims, $0.13/1M

CRITICAL: Never mix embedding models in the same index
```

### Batch Cosine Similarity (vectorized)
```python
q_norm = query / np.linalg.norm(query)
d_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
scores = np.dot(d_norm, q_norm)  # All N at once
top_k = np.argsort(scores)[::-1][:k]
```

---

## PART 8: RAG

### Complete Pipeline
```
INGESTION: Documents → Load → Clean → Chunk(512, 50) → Embed → Store

QUERY:     Query → Embed → Hybrid Search(BM25+vector)
           → Rerank(top-20→top-3) → Context → Prompt → LLM → Answer+Citations
```

### Hybrid Search Combination
```python
hybrid = (1-α) × normalize(bm25) + α × normalize(vector)  # α=0.5
# OR: Reciprocal Rank Fusion (rank-based, more robust)
```

### RAG Metrics
```
Faithfulness:       answer grounded in context?
Answer Relevance:   answer addresses question?
Context Precision:  retrieved chunks relevant?
Context Recall:     all relevant chunks retrieved?
```

### Key Rules
```
- Same embedding model for indexing AND querying
- Always add overlap (50-100 tokens)
- Store metadata: source, page, date, category
- Content hash for deduplication
- Cache frequent queries in Redis
```

---

## PART 9: AGENTS

### ReAct Loop
```python
for i in range(MAX_ITERATIONS):
    response = llm.complete(messages)
    parsed = json.loads(response)
    if parsed["action"] == "final_answer":
        return parsed["answer"]
    result = execute_tool(parsed["action"], parsed["args"])
    messages.append({"role": "tool", "content": result})
```

### OpenAI Tool Calling
```python
response = client.chat.completions.create(
    model="gpt-4o", messages=messages, tools=schemas, tool_choice="auto"
)
if response.choices[0].finish_reason == "tool_calls":
    for tc in response.choices[0].message.tool_calls:
        result = execute(tc.function.name, json.loads(tc.function.arguments))
        messages.append({"role":"tool","tool_call_id":tc.id,"content":result})
```

### Agent Failure Modes + Fixes
```
Infinite loop         → MAX_ITERATIONS guard
Prompt injection      → scan tool results for injection patterns
Context overflow      → summarize history after N turns
Tool selection wrong  → improve tool descriptions
JSON parse error      → re-prompt with error message
```

---

## PART 10: FASTAPI

### Production Endpoint Pattern
```python
@app.post("/endpoint", response_model=ResponseModel)
async def endpoint(
    request: RequestModel,           # Validated by Pydantic
    api_key: str = Depends(auth),    # Dependency injection
) -> ResponseModel:
    req_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        result = await do_async_work(request)
        return ResponseModel(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] {e}", exc_info=True)
        raise HTTPException(500, "Internal server error")
```

### Authentication
```python
async def auth(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key not in VALID_KEYS:
        raise HTTPException(401, "Invalid API key")
    return x_api_key
```

### Streaming
```python
return StreamingResponse(
    async_gen(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache"},
)
```

---

## PART 11: DOCKER

### Production Dockerfile
```dockerfile
FROM python:3.11-slim
RUN useradd --create-home appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # layer cache
COPY . .
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Services (AI Stack)
```yaml
services:
  api: {build: ., depends_on: {postgres: {condition: service_healthy}}}
  postgres: {image: pgvector/pgvector:pg16, healthcheck: pg_isready}
  redis: {image: redis:7-alpine, healthcheck: redis-cli ping}
  chroma: {image: chromadb/chroma:latest}
```

---

## PART 12: LLMOPS + SECURITY

### Every LLM Log Entry Must Have
```python
{
    "trace_id": uuid,
    "timestamp": ISO8601,
    "user_id": anonymized,
    "model": "gpt-4o",
    "input_tokens": 1500,
    "output_tokens": 300,
    "cost_usd": 0.012,
    "latency_ms": 2100,
    "cache_hit": false,
    "finish_reason": "stop",
    "eval_score": 0.88,
}
```

### Security Checklist
```
Input:   PII scan, injection scan, length limit, rate limit
Process: structural separation <system>/<user>
Output:  PII check, toxicity check, length check, prompt leak check
Tools:   validate args, check permissions, scan results
```

### Cost Optimization Priority
1. Cache frequent queries (30%+ hit rate = 30% cost reduction)
2. Route simple queries to cheaper model
3. Reduce context: fewer/smaller chunks
4. Compress system prompts
5. Batch embedding requests

---

## PART 13: SYSTEM DESIGN

### Enterprise AI System Database Schema
```sql
users(id, email, role, daily_token_budget)
documents(id, user_id, status, content_hash, chunk_count)
conversations(id, user_id, title)
messages(id, conversation_id, role, content, sources, cost_usd, eval_score)
document_chunks(id, document_id, content, embedding vector(1536), metadata)
-- Index: CREATE INDEX USING ivfflat(embedding vector_cosine_ops)
```

### Complete Request Flow
```
Request → Auth → Rate Limit → PII Scan →
Cache Hit → return
Cache Miss → Embed → Hybrid Search → Rerank →
Token Budget Check → LLM → Guardrails →
Cache → Save DB → Log → Return
```

### Trade-off Decisions
```
pgvector vs Qdrant:   pgvector if <5M chunks; Qdrant for scale/hybrid
Cache by hash vs semantic: hash (fast, simple) first
Chunking: recursive char split (better than fixed)
LLM routing: gpt-4o-mini for simple, gpt-4o for complex
Sync vs async ingestion: always async (don't block HTTP)
```

---

## QUICK-FIRE ANSWERS (30 seconds each)

1. RAG = retrieval-augmented generation, grounds LLM in retrieved facts
2. Hallucination = statistical completion, no truth verification mechanism
3. Cosine similarity = scale-invariant angle between vectors, range -1 to 1
4. Transformer attention = softmax(QK^T/√d_k) × V
5. BERT = encoder-only, bidirectional, embeddings/classification
6. GPT = decoder-only, causal mask, generation
7. Temperature → sharpness of token probability distribution
8. Top-p = nucleus sampling, covers p% of probability mass
9. Hybrid search = BM25 (keywords) + vector (semantics) combined
10. Reranker = cross-encoder rescores top-K bi-encoder results
11. ReAct = reasoning + acting, interleaves thoughts and tool calls
12. MAX_ITERATIONS = prevent agent infinite loops
13. Prompt injection = user input overrides system instructions
14. Indirect injection = attack hidden in retrieved documents/tool results
15. PII = personally identifiable information, detect before external LLMs
16. Faithfulness = answer grounded in context, no hallucinated claims
17. p95 latency = 95% of requests complete within this time
18. LLM-as-judge = strong LLM evaluates weaker LLM's output
19. Content hash = SHA256 of document, prevents re-ingesting duplicates
20. `asyncio.gather()` = run N coroutines concurrently, wait for all

---

*This document is your 30-minute pre-interview review. Read it the morning of every interview.*
