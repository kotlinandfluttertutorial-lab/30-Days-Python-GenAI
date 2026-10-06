"""
Day 05 — Statistics + Data Analysis
=====================================
Complete data analysis pipeline for AI evaluation data.

Run: python 01_statistics_analysis.py
Requires: pip install numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
from typing import Any


def create_evaluation_dataset() -> pd.DataFrame:
    """Create a realistic RAG evaluation dataset."""
    np.random.seed(42)
    n = 300

    word_counts = np.random.lognormal(mean=4.5, sigma=0.6, size=n).astype(int)

    return pd.DataFrame({
        "query_id": [f"q_{i:04d}" for i in range(n)],
        "faithfulness": np.clip(np.random.normal(0.82, 0.12, n), 0, 1),
        "relevance": np.clip(np.random.normal(0.75, 0.15, n), 0, 1),
        "word_count": np.clip(word_counts, 10, 800),
        "retrieval_score": np.clip(np.random.normal(0.78, 0.14, n), 0, 1),
        "latency_ms": np.clip(np.random.lognormal(6.5, 0.4, n), 100, 8000).astype(int),
        "category": np.random.choice(["factual", "analytical", "creative"], n, p=[0.5, 0.3, 0.2]),
    })


def distribution_analysis(df: pd.DataFrame) -> None:
    print("\n── DISTRIBUTION ANALYSIS ──")

    for col in ["faithfulness", "relevance", "retrieval_score"]:
        data = df[col]
        q25, q75 = data.quantile(0.25), data.quantile(0.75)
        iqr = q75 - q25

        print(f"\n  {col}:")
        print(f"    Mean ± Std: {data.mean():.3f} ± {data.std():.3f}")
        print(f"    Median:     {data.median():.3f}")
        print(f"    IQR:        [{q25:.3f}, {q75:.3f}] = {iqr:.3f}")
        print(f"    Range:      [{data.min():.3f}, {data.max():.3f}]")

        # Outliers using IQR
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"    Outliers:   {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")


def correlation_analysis(df: pd.DataFrame) -> None:
    print("\n── CORRELATION ANALYSIS ──")

    numeric_cols = ["faithfulness", "relevance", "retrieval_score", "word_count"]
    corr = df[numeric_cols].corr()

    print("\n  Correlation matrix:")
    print(corr.round(3).to_string())

    print("\n  Strong correlations (|r| > 0.3):")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.3:
                direction = "positive" if r > 0 else "negative"
                print(f"    {corr.columns[i]} ↔ {corr.columns[j]}: {r:.3f} ({direction})")


def outlier_investigation(df: pd.DataFrame) -> None:
    print("\n── OUTLIER INVESTIGATION ──")

    # In RAG evaluation, outliers are the most important!
    # Low scores = system failures
    # High scores = what worked perfectly?

    low_faith = df[df["faithfulness"] < df["faithfulness"].quantile(0.05)]
    high_faith = df[df["faithfulness"] > df["faithfulness"].quantile(0.95)]

    print(f"  Bottom 5% faithfulness ({len(low_faith)} queries):")
    print(f"    Mean relevance: {low_faith['relevance'].mean():.3f}")
    print(f"    Mean word count: {low_faith['word_count'].mean():.0f}")
    print(f"    Categories: {low_faith['category'].value_counts().to_dict()}")

    print(f"\n  Top 5% faithfulness ({len(high_faith)} queries):")
    print(f"    Mean relevance: {high_faith['relevance'].mean():.3f}")
    print(f"    Mean word count: {high_faith['word_count'].mean():.0f}")


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── FEATURE ENGINEERING ──")

    df = df.copy()

    # Derived features
    df["composite_score"] = (df["faithfulness"] * 0.5 + df["relevance"] * 0.3 + df["retrieval_score"] * 0.2)
    df["log_latency"] = np.log1p(df["latency_ms"])
    df["log_word_count"] = np.log1p(df["word_count"])
    df["score_percentile"] = df["composite_score"].rank(pct=True)

    print(f"  Composite score range: [{df['composite_score'].min():.3f}, {df['composite_score'].max():.3f}]")
    print(f"  Mean composite score: {df['composite_score'].mean():.3f}")

    return df


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 05 — STATISTICS + DATA ANALYSIS               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    df = create_evaluation_dataset()
    print(f"\nDataset: {df.shape[0]} evaluation results")

    distribution_analysis(df)
    correlation_analysis(df)
    outlier_investigation(df)
    df = feature_engineering(df)

    print("\n✓ Statistics + data analysis complete!")
    print("\nKey insights for AI evaluation:")
    print("  • Distribution analysis reveals quality baseline")
    print("  • Outliers reveal system failure modes (investigate these!)")
    print("  • Correlation shows what predicts quality")
    print("  • Feature engineering creates ML-ready evaluation features")


if __name__ == "__main__":
    main()
