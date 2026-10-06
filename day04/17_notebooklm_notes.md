# Day 04 — NotebookLM Notes: AI Mathematics

## Key Formulas
- Cosine similarity: `dot(a,b) / (|a|×|b|)` — range -1 to 1
- Softmax: `exp(xᵢ) / Σexp(xⱼ)` — sums to 1
- Attention: `softmax(QK^T / √d_k) × V`
- Z-score: `(x - μ) / σ`

## Why Cosine for Embeddings
Scale-invariant = same angle regardless of vector length. Text documents vary wildly in length → varying embedding magnitudes → cosine similarity handles this correctly, Euclidean does not.

## Temperature Math
`p_i = exp(logit_i / τ)`. τ→0: greedy (deterministic). τ=1: standard. τ>1: uniform (creative). Never use τ>2 in production — outputs become incoherent.

## Interview Facts
1. Cosine similarity measures direction not length
2. `np.dot(a,b)` is inner product; divide by norms for cosine
3. Softmax unstable without max subtraction (numerical stability)
4. Temperature divides logits BEFORE softmax
5. Unit vector: magnitude = 1.0; enables inner product = cosine similarity

## Common Traps
- Euclidean vs cosine: use cosine for text embeddings
- No max subtraction in softmax: overflow at large logits
- Comparing embeddings of different dimensions: always validate shapes
