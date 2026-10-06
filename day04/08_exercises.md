# Day 04 — Exercises

---

## Theory (10)
**T1.** What is the dot product and what does a high dot product value indicate?
**T2.** Why is cosine similarity scale-invariant? Explain with an example.
**T3.** What is Bayes' theorem? Give a spam detection example.
**T4.** What does standard deviation tell you about a set of evaluation scores?
**T5.** What is the difference between L1 and L2 distance? When would you use each?
**T6.** Why does softmax sum to 1? Why is it used for token probabilities?
**T7.** What is a unit vector? Why normalize embeddings before comparison?
**T8.** Explain the attention score formula: `scores = QK^T / √d_k`. Why divide by √d_k?
**T9.** What does it mean for two vectors to be orthogonal? Cosine similarity = ?
**T10.** Why use log probabilities in Naive Bayes instead of raw probabilities?

---

## Coding (5)
**C1.** Implement from scratch: `dot_product`, `magnitude`, `cosine_similarity`, `euclidean_distance`. Verify with NumPy.
**C2.** Implement `softmax(logits, temperature=1.0)` with numerical stability.
**C3.** Implement `top_k_search(query, documents, k)` returning `[(doc, score)]`.
**C4.** Implement `normalize_vectors(vectors)` that normalizes each row of a matrix.
**C5.** Implement `bayes_classify(word_list, spam_probs, ham_probs, prior_spam=0.3)`.

---

## Interview (5)
**I1.** "Explain cosine similarity and why it's used for embeddings."
**I2.** "What is softmax and when is it used in an LLM?"
**I3.** "Why does temperature control creativity in LLM output?"
**I4.** "What is the difference between dot product similarity and cosine similarity?"
**I5.** "How does vector algebra explain why RAG retrieval works?"
