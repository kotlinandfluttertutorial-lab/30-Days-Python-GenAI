# Day 15 — Concepts
## LLM Fundamentals

---

## 1. LLM Architecture and Inference

```
TRAINING (happens once, done by researchers):
  Trillions of text tokens → Self-supervised next-token prediction
  → Billions of parameters adjusted via gradient descent
  → Result: a model that knows language, facts, code, reasoning

INFERENCE (what you do as an AI engineer):
  Input tokens → Transformer layers → Logits over vocabulary
  → Sampling → Output token → Append → Repeat
  
  This loop continues until:
  - End-of-sequence token generated
  - max_tokens limit reached
  - Stop sequence encountered
```

---

## 2. Context Window

```python
# Context window = maximum tokens per API call
# = system prompt + conversation history + retrieved context + user query + response

def estimate_context_usage(
    system_prompt: str,
    conversation_history: list[dict],
    retrieved_context: str,
    user_query: str,
) -> dict[str, int]:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)  # ~4 chars per token

    system_tokens = count_tokens(system_prompt)
    history_tokens = sum(count_tokens(m["content"]) for m in conversation_history)
    context_tokens = count_tokens(retrieved_context)
    query_tokens = count_tokens(user_query)
    total = system_tokens + history_tokens + context_tokens + query_tokens

    return {
        "system": system_tokens,
        "history": history_tokens,
        "context": context_tokens,
        "query": query_tokens,
        "total_input": total,
        "remaining": 128000 - total,  # GPT-4o limit
    }
```

---

## 3. Sampling Strategies

```python
import numpy as np

def sample_next_token(logits: np.ndarray, temperature: float = 1.0,
                       top_k: int = 0, top_p: float = 1.0) -> int:
    """
    Sample next token from logits using temperature + top-k + top-p.
    """
    # Temperature scaling
    logits = logits / max(temperature, 1e-8)

    # Top-k: only consider k most likely tokens
    if top_k > 0:
        top_k_values = np.sort(logits)[::-1][:top_k]
        threshold = top_k_values[-1]
        logits = np.where(logits >= threshold, logits, -np.inf)

    # Top-p (nucleus sampling): consider tokens covering p probability mass
    if top_p < 1.0:
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        # Remove tokens beyond p
        remove = sorted_idx[cumsum > top_p]
        logits[remove] = -np.inf

    # Softmax
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    # Sample
    return int(np.random.choice(len(probs), p=probs))

# Rules of thumb:
# temperature=0: greedy (always pick highest probability)
# temperature=0.3-0.7: deterministic enough for factual tasks
# temperature=0.8-1.2: creative writing
# top_k=50: consider top 50 tokens
# top_p=0.9: cover 90% probability mass (nucleus sampling)
# In practice: often use temp=0.7 + top_p=0.9 together
```

---

## 4. LLM APIs

```python
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is RAG?"},
    ],
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
)

content = response.choices[0].message.content
finish_reason = response.choices[0].finish_reason  # "stop", "length", "content_filter"
input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens

# ── Anthropic ─────────────────────────────────────────────
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=500,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "What is RAG?"}],
)
content = message.content[0].text

# ── Groq (free, fast) ─────────────────────────────────────
from openai import OpenAI  # Groq uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "What is RAG?"}],
    temperature=0.7,
)
```

---

## 5. Streaming Responses

```python
from openai import OpenAI

client = OpenAI()

# Stream tokens as they're generated
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain attention in 3 sentences."}],
    stream=True,
)

print("Streaming: ", end="", flush=True)
full_response = ""
for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="", flush=True)  # Display immediately
        full_response += token
print()  # Final newline

# Async streaming (for FastAPI)
async def stream_response(prompt: str):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    async with client.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
```

---

## 6. Hallucinations — Production Mitigation

```python
# 1. Retrieval-Augmented Generation (RAG)
system_prompt = """Answer ONLY based on the provided context.
If the context doesn't contain the answer, say "I don't have information about that."
Never make up information."""

# 2. Structured output with citations
schema = {
    "answer": "string",
    "source_citations": ["list of source IDs used"],
    "confidence": "high|medium|low",
    "unsupported_claims": ["list any claims not in context"]
}

# 3. Temperature = 0 for factual tasks
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0,  # Deterministic, less hallucination
    messages=[...],
)

# 4. Prompt instruction
instruction = """Important: If you are not certain about a fact,
explicitly say "I'm not sure about this" rather than guessing."""
```

---

## 7. Open-Source LLMs with Ollama

```python
import requests
import json

def ollama_chat(model: str, messages: list[dict], temperature: float = 0.7) -> str:
    """Call a locally running Ollama LLM."""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

# Pull a model: ollama pull llama3.2
# Run server:   ollama serve
# Usage:
result = ollama_chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is RAG?"}],
)
print(result)
```
