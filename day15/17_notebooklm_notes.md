# Day 15 — NotebookLM Notes: LLM Fundamentals

## Inference Process

```
Input tokens → Transformer blocks → Logits (50K vocab)
→ Scale by 1/temperature → Top-k filter → Top-p filter
→ Softmax → Sample next token → Append → Repeat
```

## Sampling Parameters

| Parameter | Effect | Use Case |
|-----------|--------|---------|
| temperature=0 | Greedy (deterministic) | Facts, math, code |
| temperature=0.3-0.7 | Balanced | General QA |
| temperature=0.8-1.2 | Creative | Writing, brainstorming |
| top_k=50 | Consider top 50 tokens only | Reduces incoherence |
| top_p=0.9 | Cover 90% probability mass | Nucleus sampling (common default) |

## Context Window Facts

```
GPT-4o:           128K tokens ≈ 90,000 words ≈ 300 pages
Claude 3.5 Sonnet: 200K tokens
Gemini 1.5 Pro:   1M tokens
Llama 3.1:        128K tokens

Typical RAG call breakdown:
  System: 200 tokens
  Context: 1500 tokens (5 chunks × 300)
  History: 800 tokens
  Query: 50 tokens
  Response: 300 tokens
  Total: ~2850 tokens
```

## OpenAI API Pattern

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
    ],
    temperature=0.7,
    max_tokens=500,
    stream=False,
)
content = response.choices[0].message.content
tokens = response.usage.total_tokens
```

## Interview Facts

1. LLM inference = autoregressive: generates one token at a time
2. `finish_reason="length"` means hit max_tokens limit (response truncated!)
3. Temperature=0 is NOT the same as deterministic (quantization/rounding differences)
4. Context window = input+output combined (not separate limits)
5. Groq: free API, very fast Llama/Mistral inference (great for learning)
6. Open-source LLMs via Ollama: completely free, private, runs locally
7. System prompt tokens count toward context limit every single call

## Hallucination Mitigation Priority

1. RAG (ground in retrieved facts) — most effective
2. Temperature=0 for factual tasks
3. System prompt: "only answer from context"
4. Structured output with citation fields
5. Verification pass: second LLM call to fact-check
