# Day 03 — Interview Answers

**Q1. Abstract base class:**
An ABC uses `@abstractmethod` to define methods that ALL subclasses MUST implement. If a subclass doesn't implement an abstract method, instantiating it raises `TypeError` immediately — catching errors at class definition rather than runtime. In AI systems, ABCs define the interface for embedders, vector stores, and LLMs, enabling polymorphism and testing.

**Q3. NumPy speed:**
Python lists store pointers to Python objects scattered in memory. NumPy arrays store raw C floats in a contiguous memory block. Operations use BLAS/LAPACK (optimized C/Fortran) and SIMD CPU instructions that operate on 4-16 floats simultaneously. A 768-dim dot product: Python takes ~0.5ms per call; NumPy takes ~0.001ms. For 1M document embeddings, this is the difference between seconds and milliseconds.

**Q4. Data leakage:**
Data leakage occurs when information from the test set influences the training process, producing optimistically biased evaluation metrics. Classic example: fitting a StandardScaler on the full dataset before splitting — the scaler has "seen" the test set's distribution. Fix: always split first, then fit all preprocessors on training data only. Apply `fit_transform` to train, `transform` (never `fit_transform`) to val and test.

**Q5. Text features for ML:**
- `word_count`: text length proxy
- `char_count`: character count
- `avg_word_length`: vocabulary complexity
- `sentence_count`: document structure
- `has_numbers`: numeric content indicator
- TF-IDF vectors: term frequency features (Day 13)
- Embedding vectors: semantic features (Day 17)

**Q8. fit_transform vs transform:**
`fit_transform`: learns parameters from data (mean, std for StandardScaler) AND applies the transformation. Only use on training data. `transform`: applies the already-learned parameters without re-learning. Use on val/test to ensure they use the same scale as training. Using `fit_transform` on test data is data leakage — the test set's statistics would influence normalization.

**Q9. Validation set:**
You need three sets: train (learn model parameters), validation (tune hyperparameters and select model), test (final honest evaluation). If you use the test set for model selection, you've implicitly optimized for it and your final metrics are biased. The test set should be touched EXACTLY ONCE — at the very end. The validation set is your sandbox for experimentation.
