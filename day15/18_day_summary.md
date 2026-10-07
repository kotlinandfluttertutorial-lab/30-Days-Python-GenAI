# Day 15 — Day Summary: LLM Fundamentals

## What You Built
LLM CLI Assistant with provider auto-detection, streaming responses, conversation history management with context window limits, token tracking, and cost estimation.

## Key Takeaways
1. LLM inference = autoregressive: one token at a time, repeat until done
2. Temperature + top-k + top-p together control creativity vs determinism
3. Context window includes EVERYTHING: system + history + context + response
4. Streaming with generators enables real-time token display
5. Cost = (input_tokens × input_price + output_tokens × output_price) / 1000

## Phase 5 Complete (Days 13-15)
NLP → Transformers → LLMs. The conceptual stack is complete.

## Tomorrow: Day 16 — Prompt Engineering
Zero-shot, few-shot, chain-of-thought, structured output, JSON extraction, system prompt design, prompt versioning, and prompt injection defense.
