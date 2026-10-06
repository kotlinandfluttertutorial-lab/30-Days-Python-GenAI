# Day 03 — Exercises

---

## Theory (10 Questions)

**T1.** What is an abstract base class and why is it useful for AI systems?
**T2.** What is the difference between inheritance and composition? When should you use each?
**T3.** What does `@property` do? Give a use case in an AI configuration class.
**T4.** What is the difference between `@classmethod` and `@staticmethod`?
**T5.** Why is NumPy significantly faster than Python lists for vector math?
**T6.** What is "data leakage" in a train/val/test split? Give a concrete example.
**T7.** What does `df.describe()` show and why is it useful before ML?
**T8.** What is feature engineering? Give three features you could engineer from a text column.
**T9.** What is the purpose of `StandardScaler` and when do you apply it?
**T10.** What does `np.argsort(scores)[::-1][:5]` compute? Why is this pattern used in RAG?

---

## Coding (5 Questions)

**C1.** Write a `BaseVectorStore` abstract class with abstract methods: `add`, `search`, `delete`, `count`. Then write a `DictVectorStore` implementation using a Python dict.

**C2.** Write a `DatasetProfile` class with: `load_csv(path)`, `profile()` (print shape, dtypes, nulls), `clean()` (drop nulls, remove duplicates), `feature_stats()` (mean/std/percentiles of numeric cols).

**C3.** Write a NumPy function `top_k_similar(query: np.ndarray, docs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]` that returns `(indices, scores)` of top-k most similar documents.

**C4.** Write a function `train_val_test_split(df: pd.DataFrame, train=0.7, val=0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` that splits without using sklearn.

**C5.** Write a `TextFeatureEngineer` class that takes a Pandas Series of texts and adds: `word_count`, `char_count`, `avg_word_length`, `has_numbers`, `sentence_count` columns to a DataFrame.

---

## Debugging (5 Questions)

**D1.** Fix the abstract class:
```python
from abc import ABC
class BaseEmbedder(ABC):
    def embed(self, text):  # Missing @abstractmethod
        pass
# Why is this a problem? What can go wrong?
```

**D2.** Fix the data leakage:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df["score_scaled"] = scaler.fit_transform(df[["score"]])
train_df, test_df = train_test_split(df)
# Bug: what is wrong with this order?
```

**D3.** Fix the NumPy bug:
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# Trying to compute element-wise product then sum (dot product)
result = np.sum(a * b)  # Is this correct? What's the NumPy way?
```

**D4.** Fix the Pandas bug:
```python
df["text_clean"] = df["text"].str.lower()
mean_score = df["quality_score"].mean()
df.fillna(mean_score)  # Bug!
```

**D5.** Find the OOP design error:
```python
class RAGPipeline(OpenAIEmbedder, ChromaVectorStore, GPT4LLM):
    """RAG pipeline using multiple inheritance."""
    pass
# What is architecturally wrong with this design?
```

---

## Interview (10 Questions)
**I1.** What is the difference between `@abstractmethod` and a regular method in a base class?
**I2.** How does the Liskov Substitution Principle apply to AI system design?
**I3.** Why do AI frameworks use abstract base classes for embedders and LLMs?
**I4.** What is "duck typing" in Python and how does it relate to ABCs?
**I5.** When would you use NumPy vs plain Python for similarity computations?
**I6.** Explain the difference between `fit_transform` and `transform` in sklearn.
**I7.** What is the purpose of the validation set (separate from test)?
**I8.** How would you handle a dataset with 30% missing values in a key column?
**I9.** What are the risks of one-hot encoding a high-cardinality categorical column?
**I10.** How do you know if a feature you engineered is actually useful for a model?

---

## Practical Challenge
**P1.** Build a `DataPipeline` class that:
1. Loads a CSV (you create synthetic data)
2. Profiles it (missing, types, shape)
3. Cleans it (fill nulls, remove short texts)
4. Engineers 5 features
5. Splits train/val/test
6. Exports each split as CSV
7. Prints a summary report
