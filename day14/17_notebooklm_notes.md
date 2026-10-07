# Day 14 — NotebookLM Notes: Transformers

## The Core Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q = queries: "what am I looking for?"
K = keys:    "what do I have?"
V = values:  "the actual information"
√d_k: scale factor (prevents vanishing gradient in softmax)
```

## Architecture Components

| Component | Purpose | Key Detail |
|-----------|---------|-----------|
| Self-attention | Token-to-token relationships | Every token attends to every other |
| Multi-head attention | Multiple relationship types | h parallel attention computations |
| Positional encoding | Position information | Sine/cosine (or learned) |
| Feed-forward | Per-token transformation | 4× d_model width, GELU activation |
| Residual connection | Gradient highway | x + sublayer(x) |
| LayerNorm | Stabilize training | After each sublayer (pre-norm modern) |

## Encoder vs Decoder

```
Encoder (BERT): bidirectional attention → understanding, embeddings
Decoder (GPT):  causal attention → generation, text completion
Encoder-Decoder (T5): translation, summarization
```

## Causal Mask
Lower triangular matrix. Token i can only attend to tokens 0..i.
Prevents model from "seeing the future" during autoregressive generation.

## Interview Facts

1. Why scale by √d_k? Large dot products push softmax into near-zero gradient regions
2. Why multiple heads? Each head can specialize: one for syntax, one for semantics, etc.
3. Residual connections: allow gradients to flow through deep networks without vanishing
4. Pre-norm (norm before sublayer) vs Post-norm (after): pre-norm more stable for deep models
5. GELU: smoother than ReLU; standard in modern transformers (BERT, GPT)
6. Context window = max sequence length the model was trained on
7. KV cache: in inference, cache K and V from previous tokens to avoid recomputation

## Why Transformers Won

```
RNN:         Sequential → can't parallelize → slow training
Transformer: Parallel → train on 1000s of GPUs simultaneously
RNN:         Distant context lost (vanishing gradient)
Transformer: Direct attention → "The bank on the [river]" ← direct link
```

## Common Mistakes

- Forgetting mask in decoder → model sees future tokens → data leakage during training
- Wrong d_k check: d_model must be divisible by n_heads
- Computing attention without scale → extreme logits → softmax saturates → near-zero gradients
- Comparing encoder (BERT) to decoder (GPT) outputs without understanding the difference
