# Day 12 — NotebookLM Notes: Deep Learning Engineering

## Architecture Comparison

| Architecture | Input | Captures | Best For |
|-------------|-------|----------|---------|
| MLP | Fixed vectors | Global patterns | Tabular, embeddings |
| CNN | Sequences/images | Local patterns (n-grams, edges) | Text classification, images |
| RNN/LSTM | Sequences | Sequential dependencies | Time series, old NLP |
| Transformer | Sequences | All pairwise dependencies | All modern NLP, LLMs |

## Transfer Learning Strategy

```
STEP 1: Freeze pretrained model, train only classification head
        (few epochs, high LR ~1e-3)
        → Model learns to map BERT representations to labels

STEP 2: Unfreeze BERT, train everything with tiny LR ~1e-5
        → Fine-tune BERT for your specific domain
        → Risk: catastrophic forgetting if LR too high
```

## Interview Facts

1. LSTM solves vanishing gradient via gates (input, forget, output)
2. Bidirectional LSTM: read left→right AND right→left → context from both directions
3. BatchNorm: normalizes within a batch; position = after linear, before activation
4. LayerNorm: normalizes across features; used in Transformers (works with batch_size=1)
5. Transfer learning: fine-tuning is almost always better than training from scratch
6. Warmup scheduler: gradually increase LR to avoid destroying pretrained weights
7. Dropout at training = implicit ensemble of 2^n sub-networks

## CNN vs LSTM for Text (as of 2024)

Both are OBSOLETE for most NLP tasks. Transformers dominate.
But they appear in interviews and some production systems still use them.
Key to know: why Transformers won (parallel training, global attention, scale).
