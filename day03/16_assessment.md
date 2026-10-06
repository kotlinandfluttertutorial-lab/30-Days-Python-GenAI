# Day 03 — Assessment

---

## Quick Assessment (60 minutes, 150 points)

### Theory (5 × 10 = 50 pts)
1. What does `@abstractmethod` enforce and when does it raise an error?
2. Why is composition generally preferred over inheritance in AI systems?
3. Explain batch cosine similarity with NumPy — why is it faster than a loop?
4. What is data leakage? Give a concrete example with a StandardScaler.
5. List 5 useful features you can engineer from a raw text column.

### Coding (3 × 20 = 60 pts)
1. Write a `BaseVectorStore` ABC and a `DictVectorStore` concrete implementation.
2. Write `top_k_similar(query, docs, k)` using NumPy batch operations.
3. Write a `clean_dataframe(df)` function that handles: drop null text rows, fill numeric nulls with median, drop duplicate text rows, add word_count column.

### Architecture (1 × 40 = 40 pts)
Design the class hierarchy for an AI document pipeline that supports:
- Multiple embedding providers (OpenAI, local, mock)
- Multiple chunking strategies (fixed, sentence, semantic)
- Multiple vector stores (ChromaDB, FAISS, in-memory)
Show the abstract base classes, concrete implementations, and how a pipeline composes them.

**Pass: 112/150 (75%)**
