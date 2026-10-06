# Day 03 — Common Mistakes

**Mistake 1: Forgetting @abstractmethod**
Without `@abstractmethod`, subclasses can skip implementing methods silently. Always decorate abstract methods.

**Mistake 2: Multiple inheritance for components**
Using `class RAG(OpenAIEmbedder, ChromaDB, GPT4)` creates brittle coupling. Use composition instead.

**Mistake 3: Data leakage**
Fitting scalers/encoders before splitting is the #1 ML mistake. Split first, then fit on train only.

**Mistake 4: Modifying DataFrame without inplace or reassignment**
`df.fillna(0)` does nothing. Use `df = df.fillna(0)` or `df.fillna(0, inplace=True)`.

**Mistake 5: Using Python loops for NumPy operations**
```python
# Slow
scores = [np.dot(query, doc) for doc in docs]
# Fast
scores = np.dot(docs, query)
```

**Mistake 6: Forgetting to normalize vectors before cosine similarity**
If vectors aren't normalized, `np.dot(a, b)` is NOT cosine similarity — it's a dot product that depends on magnitude.

**Mistake 7: Not profiling data before modeling**
Always run `df.describe()`, `df.isnull().sum()`, and `df.dtypes` before any ML work. Unknown distribution = unknown failure modes.
