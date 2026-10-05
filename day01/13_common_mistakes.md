# Day 01 — Common Mistakes
## Mistakes Every AI Engineering Beginner Makes

---

## Category 1: Conceptual Mistakes

### Mistake 1: "RAG is just a database lookup"

**What people think:** RAG just stores documents and retrieves them by keyword.

**What's actually happening:** RAG uses embedding models to convert text to dense vectors. Retrieval is semantic similarity (cosine similarity on 1536-dimensional vectors), not keyword search. The retrieved chunks are then injected into an LLM's prompt as context.

**Why it matters:** If you think RAG is a database lookup, you'll build a bad RAG system — one that fails on synonym queries, paraphrases, and semantic questions.

---

### Mistake 2: Confusing LLM Parameters with API Parameters

**What people confuse:**
- "Parameters" of the model (175B parameters of GPT-3) = weights learned during training
- "Parameters" of an API call (temperature, top-k) = inference configuration

Both are called "parameters" in different contexts.

**The fix:** Say "model weights" for training parameters and "inference parameters" or "sampling parameters" for the API configuration.

---

### Mistake 3: Thinking Fine-Tuning Solves Everything

**What beginners think:** "Our LLM doesn't know our product catalog. Let's fine-tune it on our product data."

**Why this is wrong:**
- Fine-tuning teaches style/behavior, not factual retrieval
- Fine-tuned models still hallucinate
- Fine-tuned knowledge can't be updated without retraining
- Cost: fine-tuning is expensive

**The fix:** Use RAG for factual knowledge retrieval. Use fine-tuning for behavior/style changes.

---

### Mistake 4: Thinking Temperature = 0 Means No Hallucination

**What people think:** "If I set temperature to 0, the model won't make things up."

**Why this is wrong:** Temperature controls sampling randomness, not factual accuracy. A temperature-0 model will deterministically generate the same hallucinated answer every time if it doesn't know the truth.

**The fix:** Use RAG to provide facts. Use evaluation to measure hallucination rates regardless of temperature.

---

### Mistake 5: Ignoring Token Costs Until Bills Arrive

**What happens:** Engineer builds a RAG system, launches it, sees a $10,000 bill two weeks later.

**How it happens:**
- System prompt is 2,000 tokens (way too long)
- Retrieves 10 chunks × 500 tokens = 5,000 tokens
- GPT-4o at $5/1M tokens × 7,000 tokens × 10,000 requests/day = $350/day = $10,500/month

**The fix:** Calculate costs before launching. Build a cost estimator. Set budget alerts in OpenAI dashboard.

---

## Category 2: Implementation Mistakes

### Mistake 6: Hardcoding API Keys

```python
# WRONG — never do this:
client = OpenAI(api_key="sk-proj-abc123def456...")  # In your code!

# RIGHT:
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

**Why it matters:** If you push this to GitHub, your key gets scraped by bots within minutes. This has happened to countless engineers. Result: unexpected charges or compromised account.

---

### Mistake 7: Not Checking If API Key Exists

```python
# WRONG — will give cryptic error later:
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RIGHT — fail fast with clear message:
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Create a .env file with: OPENAI_API_KEY=sk-..."
    )
client = OpenAI(api_key=api_key)
```

---

### Mistake 8: No Error Handling for LLM Calls

```python
# WRONG:
response = client.chat.completions.create(...)
content = response.choices[0].message.content

# RIGHT — LLM calls fail: rate limits, timeouts, content filters:
try:
    response = client.chat.completions.create(...)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")
except RateLimitError:
    # Wait and retry
    time.sleep(60)
    # retry...
except APIConnectionError:
    # Try fallback provider
    ...
except Exception as e:
    logger.error("LLM call failed", error=str(e))
    raise
```

---

### Mistake 9: Comparing Embeddings From Different Models

```python
# WRONG — these are incompatible:
embedding_1 = openai_client.embeddings.create(
    input="Hello world",
    model="text-embedding-3-small"  # OpenAI model
)

embedding_2 = sentence_transformer.encode("Hello world")  # HuggingFace model

# Cosine similarity of these two = meaningless noise
# They're in completely different vector spaces!
```

**Rule:** All embeddings in a system must use the same model. Lock the model version.

---

### Mistake 10: Using Virtual Environments Inconsistently

```bash
# Common mistake: forgetting to activate the venv

# Wrong workflow:
pip install openai  # Installs globally, not in project
python script.py    # May use wrong Python, missing packages

# Right workflow:
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
(venv) $ pip install openai
(venv) $ python script.py
```

---

## Category 3: Architecture Mistakes

### Mistake 11: Building an Agent When RAG Would Do

**Symptom:** "We built an agent to answer questions about our documentation."

**Problem:** Agents are complex, slow, expensive, and unreliable. For document Q&A, RAG is:
- Faster (one retrieval pass)
- Cheaper (fewer LLM calls)
- More reliable (deterministic retrieval)
- Easier to debug

**Rule:** Use the simplest system that solves the problem.

```
Simple Q&A → RAG
Multi-step research → Agent with RAG tool
Autonomous task → Agent
```

---

### Mistake 12: Chunking Without Considering Semantic Boundaries

```python
# WRONG — fixed character chunking:
chunks = [text[i:i+500] for i in range(0, len(text), 500)]
# Splits sentences, breaks context, degrades retrieval quality

# BETTER — sentence-aware chunking:
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)
```

---

### Mistake 13: Not Adding Metadata to Vector DB Entries

```python
# WRONG — stores only the text:
collection.add(
    documents=["Our return policy allows 30-day returns."],
    ids=["chunk_1"]
)

# RIGHT — stores text + metadata for filtering:
collection.add(
    documents=["Our return policy allows 30-day returns."],
    metadatas=[{
        "source": "policy_document.pdf",
        "page": 3,
        "category": "returns",
        "updated": "2024-01-15",
    }],
    ids=["chunk_1"]
)
# Now you can filter: only search chunks from category="returns"
```

---

## Category 4: Production Mistakes

### Mistake 14: No Logging in AI Applications

**Symptom:** "Our AI gave a user a bad answer and we have no idea why."

**Why logging is harder in AI systems:** The input is the prompt (long), the output is text (variable), and the failure is usually silent (no exception thrown).

**Fix:**
```python
import logging

def call_llm(prompt: str) -> str:
    logging.info("LLM call", extra={
        "prompt_tokens": estimate_tokens(prompt),
        "prompt_preview": prompt[:100],
    })
    response = llm.complete(prompt)
    logging.info("LLM response", extra={
        "response_tokens": estimate_tokens(response),
        "response_preview": response[:100],
    })
    return response
```

---

### Mistake 15: No Rate Limiting on AI Endpoints

**Symptom:** A single user (or bot) sends 10,000 requests, costing you $1,000.

**Fix:** Always add rate limiting to LLM-powered endpoints.

```python
from fastapi import FastAPI, HTTPException
from datetime import datetime

# Use Redis-based rate limiting in production
rate_limit_store: dict[str, list[datetime]] = {}

def check_rate_limit(user_id: str, limit: int = 100) -> None:
    """Simple in-memory rate limiter (use Redis in production)."""
    now = datetime.utcnow()
    user_calls = rate_limit_store.get(user_id, [])
    
    # Keep only calls in the last minute
    recent_calls = [t for t in user_calls if (now - t).seconds < 60]
    
    if len(recent_calls) >= limit:
        raise HTTPException(429, "Rate limit exceeded. Try again in 60 seconds.")
    
    rate_limit_store[user_id] = recent_calls + [now]
```

---

## Category 5: Interview Mistakes

### Mistake 16: Defining RAG as "it uses a database"

**Wrong answer:** "RAG uses a database to retrieve information."  
**Correct answer:** "RAG retrieves semantically similar document chunks using vector similarity search and injects them as context into the LLM prompt."

---

### Mistake 17: Saying "It uses AI" When Describing Your System

**Wrong:** "The system uses AI to understand the query."  
**Correct:** "The query is embedded using `text-embedding-3-small`, then we do cosine similarity search against 50K document chunks in ChromaDB, retrieve the top-5, and pass them as context to Claude 3.5 Sonnet."

Specificity signals real experience. Vagueness signals tutorials.

---

### Mistake 18: Not Knowing the Costs

**Red flag for interviewers:** Not knowing approximate costs of your systems.

**You should know:**
- GPT-4o: ~$5/1M input, ~$15/1M output tokens
- Claude 3.5 Sonnet: ~$3/1M input, ~$15/1M output
- text-embedding-3-small: ~$0.02/1M tokens
- Groq (Llama 3.1 70B): ~$0.59/1M tokens
- Self-hosted: GPU cost only

Know these numbers. Calculate the cost of your project at 10K requests/day.

---

### Mistake 19: Not Having a Position on Trade-Offs

**Weak answer:** "It depends." (without elaborating)  
**Strong answer:** "It depends on X and Y. If X is the constraint, I'd choose A because of [specific reason]. If Y is the constraint, I'd choose B because of [specific reason]."

Every trade-off question tests whether you can reason from first principles, not recite a rule.

---

## Summary: The 5 Most Dangerous Mistakes

| Rank | Mistake | Consequence |
|------|---------|-------------|
| 1 | Hardcoding API keys | Account compromised, unexpected charges |
| 2 | No rate limiting | $10,000 bill from a bot |
| 3 | Wrong model for the job | Overpaying by 10-100x |
| 4 | No evaluation system | Silent degradation in production |
| 5 | Agent when RAG would do | Complex, slow, expensive, unreliable |
