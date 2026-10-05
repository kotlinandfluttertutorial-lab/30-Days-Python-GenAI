# Day 01 — Debugging Guide
## How to Debug AI Applications

---

## The Debugging Philosophy for AI Systems

Traditional debugging: find the line that threw the exception.

AI debugging: find why a correct-looking system produces incorrect-feeling results.

The key difference: **AI systems fail silently.** There's no stack trace when an LLM hallucinates. There's no error when a retrieval returns the wrong document. The system runs fine — it just gives wrong answers.

This makes AI debugging harder and more important.

---

## Debugging Framework: TRACE

Use this framework for any AI system issue:

```
T — Trace the full request pipeline
R — Reproduce the issue reliably
A — Analyze each component independently
C — Compare against known good outputs
E — Evaluate the fix before deploying
```

---

## Common Day 1 Errors

### Error 1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'dotenv'
```

**Root cause:** Package not installed, or virtual environment not activated.

**Debugging process:**
```bash
# Step 1: Check if venv is active
# You should see (venv) in your terminal prompt

# Step 2: If not, activate it
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Step 3: Install the missing package
pip install python-dotenv

# Step 4: Verify
python -c "from dotenv import load_dotenv; print('OK')"
```

**Prevention:** Always activate venv before coding. Keep requirements.txt updated.

---

### Error 2: API Key Not Found

```
openai.AuthenticationError: No API key provided.
```

**Root cause:** API key not set in environment.

**Debugging process:**
```python
import os
from dotenv import load_dotenv

# Step 1: Load .env
load_dotenv()

# Step 2: Check what's in the environment
api_key = os.getenv("OPENAI_API_KEY")
print(f"Key found: {api_key is not None}")
print(f"Key starts with: {api_key[:8] if api_key else 'None'}")

# Step 3: If not found, check .env file exists
import os.path
print(f".env file exists: {os.path.exists('.env')}")
```

**Common causes:**
1. `.env` file doesn't exist → create it from `.env.example`
2. `.env` is in wrong directory → must be in same directory as script
3. Environment variable has wrong name → check spelling exactly
4. Key value has extra spaces → `OPENAI_API_KEY = sk-...` (space around =) won't work

---

### Error 3: Connection Error to Ollama

```
ConnectionError: Cannot connect to Ollama. Is it running?
```

**Root cause:** Ollama server not running.

**Debugging process:**
```bash
# Step 1: Start Ollama
ollama serve

# Step 2: In another terminal, test connection
curl http://localhost:11434/api/tags

# Step 3: If no models found, pull one
ollama pull llama3.2

# Step 4: Verify model is available
ollama list
```

---

### Error 4: Wrong Cosine Similarity Values

```
Expected similarity between similar texts to be high, but got 0.02
```

**Root cause:** Using toy/incorrect embeddings, or comparing embeddings from different models.

**Debugging process:**
```python
# Step 1: Check vector dimensions
v1 = get_embedding("Hello world")
v2 = get_embedding("Hi there")
print(f"v1 dimensions: {len(v1)}")  # Should be 768 or 1536
print(f"v2 dimensions: {len(v2)}")
print(f"Dimensions match: {len(v1) == len(v2)}")

# Step 2: Check vector magnitudes (should not be 0)
import math
mag_v1 = math.sqrt(sum(x**2 for x in v1))
print(f"v1 magnitude: {mag_v1}")  # Should be ~1.0 for normalized embeddings

# Step 3: Verify similarity makes intuitive sense
from your_code import cosine_similarity
sim_similar = cosine_similarity(
    get_embedding("The dog ran"),
    get_embedding("A dog was running")
)
sim_different = cosine_similarity(
    get_embedding("The dog ran"),
    get_embedding("Quantum physics equations")
)
print(f"Similar texts: {sim_similar:.3f}")    # Should be > 0.8
print(f"Different texts: {sim_different:.3f}")  # Should be < 0.3
```

---

### Error 5: LLM Returns Empty or Cut-Off Response

```
LLM returned empty response or response appears truncated
```

**Root cause:** `max_tokens` too low, or response filtered.

**Debugging process:**
```python
# Step 1: Increase max_tokens
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    max_tokens=4096,  # Increase this
)

# Step 2: Check finish_reason
print(f"Finish reason: {response.choices[0].finish_reason}")
# "stop" = normal
# "length" = hit token limit → increase max_tokens
# "content_filter" = content policy → change your prompt

# Step 3: Check if response is None
content = response.choices[0].message.content
if content is None:
    print("Content is None — possible content filter")
elif len(content) < 10:
    print(f"Suspiciously short response: '{content}'")
```

---

## Debugging LLM Quality Issues

These are harder because there's no error — just bad output.

### Issue: LLM gives irrelevant answers

**Debugging steps:**
```
1. Print the full prompt (not just the question — everything sent to LLM)
2. Check if retrieved context is actually relevant
3. Check if system prompt is clear
4. Test with temperature=0 to isolate variability
5. Try the same prompt in the LLM provider's playground
```

### Issue: LLM ignores instructions

**Debugging steps:**
```
1. Simplify system prompt to one clear instruction
2. Put critical instructions at the END of the prompt (recency bias)
3. Use explicit format instructions: "ALWAYS respond with..."
4. Try a different model — some follow instructions better
5. Add few-shot examples of correct behavior
```

### Issue: LLM response quality varies wildly

**Debugging steps:**
```
1. Log every request and response
2. Identify patterns in bad responses (certain query types? lengths?)
3. Check temperature — high temperature increases variance
4. Measure consistency: run same query 10 times, compare
5. Consider a classifier that routes complex queries to a stronger model
```

---

## Debugging RAG-Specific Issues

### Issue: Retrieved documents are irrelevant

**Debugging steps:**
```python
# Step 1: Print retrieved chunks
results = vector_db.query(
    query_embeddings=[query_embedding],
    n_results=10,  # Retrieve more to see what's available
    include=["documents", "distances", "metadatas"]
)

for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"Rank {i+1} | Distance: {dist:.3f} | Preview: {doc[:100]}")

# Step 2: Check if distance scores are reasonable
# ChromaDB uses L2 distance by default (lower = better)
# If all distances are > 1.5, embeddings may not match document domain

# Step 3: Test with a known query that should match
test_query = "exact text from one of your documents"
test_result = vector_db.query(...)
# Should return that exact document as rank 1
```

### Issue: Embedding search misses exact keyword matches

**Root cause:** Vector search is semantic, not exact. "GPT-4" and "OpenAI's model" are semantically similar but a keyword search for "GPT-4" would miss "OpenAI's model".

**Fix:**
```python
# Use hybrid search: combine BM25 (keyword) + vector (semantic)
# Days 20+ covers this in detail
# For now: be aware the problem exists
```

---

## Debugging Methodology for AI Systems

### The Scientific Method for AI Debugging

```
1. OBSERVE: What exact behavior is wrong?
   - Wrong answer? Empty answer? Hallucination?
   - What was the input? What was the output?
   - When did it start? Always or sometimes?

2. HYPOTHESIZE: What could cause this?
   - Bad retrieval? (RAG problem)
   - Bad prompt? (LLM instruction problem)
   - Bad data? (ingestion problem)
   - Model issue? (capability limit)

3. EXPERIMENT: Test one hypothesis at a time
   - Don't change multiple things at once
   - Log everything before and after change
   - Use version control for prompts

4. MEASURE: Did the fix work?
   - Don't trust your gut — measure
   - Run 10+ test cases, not 1
   - Check for regressions in other cases

5. DOCUMENT: What caused it and how you fixed it?
   - Add to this file
   - Prevents same bug next time
```

---

## Debugging Toolbox

### Print Debugging (always start here)

```python
def debug_llm_call(prompt: str) -> str:
    print(f"[DEBUG] Prompt ({len(prompt)} chars):")
    print(f"[DEBUG] {prompt[:500]}...")  # First 500 chars
    
    response = call_llm(prompt)
    
    print(f"[DEBUG] Response ({len(response)} chars):")
    print(f"[DEBUG] {response[:200]}...")
    
    return response
```

### Logging (use in production)

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

logger = logging.getLogger(__name__)
logger.debug("LLM call", extra={"prompt_tokens": 150, "model": "gpt-4o"})
```

### Testing LLM behavior

```python
def test_llm_classification(model, test_cases: list[tuple[str, str]]) -> float:
    """
    Test LLM classification accuracy on known cases.
    Returns accuracy score.
    """
    correct = 0
    for input_text, expected in test_cases:
        result = model.classify(input_text)
        if result.lower() == expected.lower():
            correct += 1
        else:
            print(f"WRONG: '{input_text[:50]}' → got '{result}', expected '{expected}'")
    return correct / len(test_cases)
```

---

## Red Flags to Watch For

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Empty response | max_tokens too low or content filter | Increase max_tokens, check content |
| Always same answer | Temperature=0 or cached response | Check temperature and cache |
| Very slow responses | Large context or slow model | Reduce tokens, use faster model |
| Inconsistent quality | Temperature too high | Lower temperature to 0.3 |
| Ignores instructions | System prompt buried or competing instructions | Restructure prompt |
| Hallucinates facts | No RAG, model doesn't know | Add RAG or tool for fact lookup |
| High costs | No caching, expensive model, bloated prompts | Audit token usage per request |
