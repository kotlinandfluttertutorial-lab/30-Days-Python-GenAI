# Day 03 — Infographics

---

## Infographic 1: OOP Hierarchy for AI Systems

```
                    ABC (Abstract Base Class)
                           │
             ┌─────────────┼──────────────┐
             │             │              │
       BaseEmbedder   BaseChunker     BaseLLM
             │             │              │
      ┌──────┴──────┐ ┌────┴────┐   ┌────┴────┐
      │             │ │         │   │         │
 OpenAIEmb   LocalEmb  Fixed  Sentence  OpenAI  Anthropic

USAGE:
def build_rag(embedder: BaseEmbedder) -> RAGPipeline:
    # Works with ANY embedder — polymorphism
    return RAGPipeline(embedder=embedder, ...)
```

---

## Infographic 2: NumPy Array Memory vs Python List

```
PYTHON LIST: [0.1, 0.5, -0.3, 0.8]
  Memory:  [ptr] [ptr] [ptr] [ptr]
              ↓     ↓     ↓     ↓
           [PyFloat] [PyFloat] ...  (objects, scattered in RAM)
  Math:    dereference + unbox each value → slow

NUMPY ARRAY: np.array([0.1, 0.5, -0.3, 0.8])
  Memory:  [0.1][0.5][-0.3][0.8]  (raw float64 bytes, contiguous)
  Math:    direct SIMD operations → 100-500x faster

FOR 768-DIM EMBEDDINGS:
  Python list dot product:  ~0.5ms
  NumPy dot product:        ~0.001ms (500x faster)
```

---

## Infographic 3: Train / Val / Test Split

```
FULL DATASET (1000 rows)
├──────────────────────────────────────┤
│         TRAIN (700, 70%)             │ ← Model learns from this
├──────────┬───────────────────────────┤
│ VAL(150) │     TEST (150, 15%)       │
│  15%     │                           │
└──────────┴───────────────────────────┘

VAL:  Tune hyperparameters, select model
TEST: Final honest evaluation — touch ONLY ONCE

CRITICAL: Never fit preprocessors on TEST data (data leakage)
```

---

## Infographic 4: Pandas DataFrame Operations

```
LOAD → PROFILE → CLEAN → ENGINEER → SPLIT → EXPORT

LOAD:     pd.read_csv(), pd.read_json()
PROFILE:  df.shape, df.dtypes, df.describe(), df.isnull().sum()
CLEAN:    dropna(), fillna(), drop_duplicates(), str operations
ENGINEER: new columns from existing (word_count, normalized_score)
SPLIT:    train_test_split() from sklearn
EXPORT:   df.to_csv(), df.to_json()
```
