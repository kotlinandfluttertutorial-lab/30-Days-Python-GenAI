"""
Day 03 — NumPy + Pandas for AI Engineering
============================================
Demonstrates the numerical and data engineering foundations
used in every ML and AI system.

Run: python 02_numpy_pandas.py
Requires: pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from typing import Any
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# NUMPY: VECTOR AND MATRIX OPERATIONS
# ═══════════════════════════════════════════════════════════════

def numpy_fundamentals() -> None:
    print("\n── NUMPY: FUNDAMENTALS ──")

    # Arrays
    v1 = np.array([0.2, 0.8, -0.3, 0.5])          # 1D: embedding
    matrix = np.random.randn(5, 4)                  # 2D: 5 embeddings of dim 4
    batch = np.zeros((100, 768))                    # 2D: 100 embeddings of dim 768

    print(f"  Embedding shape: {v1.shape}, dtype: {v1.dtype}")
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Batch shape: {batch.shape}")

    # Vectorized math — much faster than Python loops
    scores = np.array([0.92, 0.78, 0.85, 0.61, 0.95])
    above_threshold = scores[scores >= 0.8]
    top_k_idx = np.argsort(scores)[::-1][:3]  # top-3 indices

    print(f"\n  Scores: {scores}")
    print(f"  Above 0.8: {above_threshold}")
    print(f"  Top-3 indices: {top_k_idx}")
    print(f"  Top-3 scores: {scores[top_k_idx]}")


def numpy_cosine_similarity() -> None:
    print("\n── NUMPY: COSINE SIMILARITY ──")

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def batch_cosine_sim(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity of one query against many documents.
        query: (D,)    docs: (N, D)   returns: (N,)
        Much faster than looping!
        """
        query_norm = query / np.linalg.norm(query)
        docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
        return np.dot(docs_norm, query_norm)

    # Simulate embeddings
    np.random.seed(42)
    query = np.random.randn(768)
    documents = np.random.randn(1000, 768)

    # Single comparison
    sim = cosine_sim(query, documents[0])
    print(f"  Single similarity: {sim:.4f}")

    # Batch comparison — find top-5 most similar
    import time
    start = time.perf_counter()
    all_scores = batch_cosine_sim(query, documents)
    elapsed = (time.perf_counter() - start) * 1000

    top5_idx = np.argsort(all_scores)[::-1][:5]
    print(f"  Batch similarity (1000 docs, 768 dims): {elapsed:.2f}ms")
    print(f"  Top-5 scores: {all_scores[top5_idx]}")

    # Compare: Python loop vs NumPy batch
    start = time.perf_counter()
    loop_scores = [cosine_sim(query, documents[i]) for i in range(1000)]
    loop_elapsed = (time.perf_counter() - start) * 1000
    print(f"  Python loop equivalent: {loop_elapsed:.2f}ms (NumPy is {loop_elapsed/elapsed:.1f}x faster)")


def numpy_statistics() -> None:
    print("\n── NUMPY: STATISTICS FOR AI EVALUATION ──")

    # Simulate evaluation scores
    np.random.seed(0)
    faithfulness_scores = np.clip(np.random.normal(0.82, 0.12, 200), 0, 1)
    relevance_scores = np.clip(np.random.normal(0.75, 0.15, 200), 0, 1)

    for name, scores in [("Faithfulness", faithfulness_scores), ("Relevance", relevance_scores)]:
        print(f"\n  {name}:")
        print(f"    Mean:   {scores.mean():.3f}")
        print(f"    Std:    {scores.std():.3f}")
        print(f"    Median: {np.median(scores):.3f}")
        print(f"    P10:    {np.percentile(scores, 10):.3f}")
        print(f"    P90:    {np.percentile(scores, 90):.3f}")
        print(f"    Below 0.5: {(scores < 0.5).sum()} ({(scores < 0.5).mean() * 100:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# PANDAS: DATA ENGINEERING
# ═══════════════════════════════════════════════════════════════

def create_sample_dataset() -> pd.DataFrame:
    """Create a realistic sample dataset for ML."""
    np.random.seed(42)
    n = 500

    texts = [
        f"Document about {topic}: " + " ".join(
            np.random.choice(
                ["machine learning", "neural network", "AI", "data", "model",
                 "training", "inference", "embedding", "vector", "retrieval"],
                size=np.random.randint(10, 30)
            ).tolist()
        )
        for topic in np.random.choice(
            ["RAG", "LLM", "embeddings", "agents", "fine-tuning"], size=n
        )
    ]

    return pd.DataFrame({
        "id": [f"doc_{i:04d}" for i in range(n)],
        "text": texts,
        "category": np.random.choice(["technical", "general", "advanced"], size=n, p=[0.5, 0.3, 0.2]),
        "quality_score": np.clip(np.random.normal(0.75, 0.15, n), 0.1, 1.0),
        "word_count": [len(t.split()) for t in texts],
        "source": np.random.choice(["web", "book", "paper"], size=n),
        "has_missing": np.random.choice([True, False, None], size=n, p=[0.1, 0.8, 0.1]),
    })


def pandas_profiling(df: pd.DataFrame) -> None:
    print("\n── PANDAS: DATA PROFILING ──")

    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Column types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<20} {dtype}")

    print(f"\n  Missing values:")
    for col, count in df.isnull().sum().items():
        pct = df.isnull().mean()[col] * 100
        if count > 0:
            print(f"    {col:<20} {count:>4} ({pct:.1f}%)")

    print(f"\n  Numeric statistics:")
    print(df[["quality_score", "word_count"]].describe().round(3).to_string(
        indent=4
    ))

    print(f"\n  Category distribution:")
    print(df["category"].value_counts().to_string(index=True))


def pandas_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── PANDAS: DATA CLEANING ──")
    original_size = len(df)

    # 1. Drop rows with missing text (can't embed nothing)
    df = df.dropna(subset=["text"]).copy()

    # 2. Fill missing quality scores with median
    median_score = df["quality_score"].median()
    df["quality_score"] = df["quality_score"].fillna(median_score)

    # 3. Remove very short texts
    df = df[df["word_count"] >= 5]

    # 4. Normalize text
    df["text_clean"] = df["text"].str.strip().str.lower()
    df["text_clean"] = df["text_clean"].str.replace(r"\s+", " ", regex=True)

    # 5. Remove duplicates
    df = df.drop_duplicates(subset=["text_clean"])

    print(f"  Original rows: {original_size}")
    print(f"  After cleaning: {len(df)}")
    print(f"  Removed: {original_size - len(df)}")
    return df


def pandas_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── PANDAS: FEATURE ENGINEERING ──")

    # Text features
    df["char_count"] = df["text"].str.len()
    df["sentence_count"] = df["text"].str.count(r"[.!?]") + 1
    df["avg_word_length"] = df["text"].apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
    )
    df["has_numbers"] = df["text"].str.contains(r"\d+", regex=True).astype(int)

    # Score buckets (categorical feature from continuous)
    df["quality_tier"] = pd.cut(
        df["quality_score"],
        bins=[0, 0.5, 0.7, 0.85, 1.0],
        labels=["low", "medium", "high", "excellent"],
    )

    # Normalized score (zero mean, unit std)
    df["score_normalized"] = (
        (df["quality_score"] - df["quality_score"].mean())
        / df["quality_score"].std()
    )

    # One-hot encode category
    df = pd.get_dummies(df, columns=["category"], prefix="cat")

    print(f"  Features added: char_count, sentence_count, avg_word_length, ...")
    print(f"  Quality tier distribution:")
    print(df["quality_tier"].value_counts().to_string())
    print(f"\n  New shape: {df.shape}")
    return df


def pandas_train_val_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n── PANDAS: TRAIN / VAL / TEST SPLIT ──")

    from sklearn.model_selection import train_test_split

    # Stratify by quality_tier to ensure balanced splits
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    print(f"  Total: {len(df)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.0f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.0f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.0f}%)")

    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 03 — NUMPY + PANDAS FOR AI ENGINEERING        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # NumPy section
    numpy_fundamentals()
    numpy_cosine_similarity()
    numpy_statistics()

    # Pandas section
    df = create_sample_dataset()
    pandas_profiling(df)
    df = pandas_cleaning(df)
    df = pandas_feature_engineering(df)
    train_df, val_df, test_df = pandas_train_val_test_split(df)

    print("\n" + "="*60)
    print("✓ Day 03 NumPy + Pandas demo complete!")
    print("="*60)
    print("\nKey patterns for AI engineering:")
    print("  • NumPy arrays: fast vectorized operations on embeddings")
    print("  • Batch cosine similarity: query vs 1000 docs in <1ms")
    print("  • Pandas profiling: understand your data before modeling")
    print("  • Feature engineering: transform raw text into ML features")
    print("  • Train/val/test split: always before any model training")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
