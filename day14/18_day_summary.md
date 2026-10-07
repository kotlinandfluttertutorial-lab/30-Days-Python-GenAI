# Day 14 — Day Summary: Transformers

## What You Built
Complete Transformer implementation: scaled dot-product attention, multi-head attention, positional encoding, transformer block with residual connections and LayerNorm, causal mask, full encoder.

## Key Takeaways
1. Self-attention: `softmax(QK^T / √d_k) × V` — the single most important formula in AI
2. Multiple heads: parallel attention computations, each specializing in different patterns
3. Residual connections + LayerNorm: enable training of very deep networks
4. Encoder (BERT) ≠ Decoder (GPT): different masking, different use cases
5. Causal mask: decoder prevents future token visibility during autoregressive training

## Phase 5 Half-Complete
Transformers are the architecture behind everything from Day 15 onward: LLMs, embeddings, RAG, agents.

## Tomorrow: Day 15 — LLM Fundamentals
Inference, context windows, sampling strategies (temperature, top-k, top-p), LLM APIs, open-source LLMs.
