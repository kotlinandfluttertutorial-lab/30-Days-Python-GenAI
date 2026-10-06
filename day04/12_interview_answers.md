# Day 04 — Interview Answers

**Q1. Cosine similarity:**
Cosine similarity measures the angle between two vectors: `cos(θ) = dot(a,b) / (|a|×|b|)`. Range -1 to 1. Used for embeddings because it's scale-invariant — a short and long document about the same topic produce vectors pointing in the same direction. Euclidean distance would separate them due to magnitude difference.

**Q2. Softmax:**
Softmax converts a vector of real numbers (logits) into a probability distribution that sums to 1: `p_i = exp(logit_i) / Σ exp(logit_j)`. In LLMs, it's applied to the output logits over the vocabulary (~50K tokens) to get token probabilities before sampling.

**Q3. Temperature:**
Temperature divides logits before softmax: `p_i = exp(logit_i / τ)`. Low τ → sharper distribution (deterministic). High τ → flatter distribution (creative/random). τ→0 is greedy decoding (always pick the most likely token). τ=1 is standard. τ>1 makes rare tokens more likely, enabling creative generation.

**Q5. Attention Q, K, V:**
Q (queries), K (keys), V (values) are linear projections of the input. Attention scores = `softmax(QK^T / √d_k)` × V. Each token's query is compared to all keys (dot product) to produce attention weights, then those weights sum over the values. The `√d_k` scaling prevents very large dot products that would make gradients vanish after softmax.

**Q8. Unit vector:**
A unit vector has magnitude 1.0. Normalizing: `v / |v|`. When all embedding vectors are unit vectors, cosine similarity = inner product (faster). OpenAI's embedding models normalize their outputs, so inner product search (FAISS IndexFlatIP) is faster than computing full cosine similarity.

**Q10. King - man + woman ≈ queen:**
Word2Vec embeddings encode semantic relationships as geometric directions. The vector from "man" to "woman" represents the "gender change" direction. The vector from "king" to "queen" is nearly identical. So `king + (woman - man) ≈ queen`. This shows embeddings have captured semantic structure, not just word co-occurrence.
