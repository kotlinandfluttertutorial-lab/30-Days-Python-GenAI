# Day 04 — Common Mistakes

**Mistake 1: Using Euclidean distance for text similarity**
Euclidean distance depends on vector magnitude. Long documents have larger vectors. Use cosine similarity (scale-invariant) for semantic text comparison.

**Mistake 2: Unstable softmax**
`exp(1000)` overflows to `inf`. Always subtract `max(logits)` before `exp()`.

**Mistake 3: Comparing non-normalized vectors as if they were cosine-normalized**
If your system stores non-unit vectors and you use inner product (instead of full cosine), results are wrong. Either normalize at insert time OR compute full cosine similarity at query time.

**Mistake 4: Forgetting dimension mismatch**
Adding vectors of different sizes silently produces wrong results in Python (loops) but raises errors in NumPy. Always validate `a.shape == b.shape` before similarity computations.

**Mistake 5: Using cosine similarity where inner product is expected**
Some vector DB configurations use inner product. If you set the index to inner product but store non-normalized vectors, you'll get wrong rankings. Know what your vector DB expects.
