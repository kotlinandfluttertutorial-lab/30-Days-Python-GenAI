# Day 05 — NotebookLM Notes: Statistics + Data Analysis

## Key Concepts

**Normal Distribution** — Bell curve. Mean ± 1 std = 68%, ± 2 std = 95.4%. Model evaluation scores often follow this. Use to set quality thresholds.

**Outlier Detection** — IQR method: outlier if < Q1 - 1.5×IQR or > Q3 + 1.5×IQR. In RAG evaluation: low-score outliers = system failures to investigate; high-score = what to replicate.

**Pearson Correlation** — Range -1 to 1. Measures linear relationship. Use to find which features predict quality. Correlation ≠ causation.

**Skewed Distributions** — Latency, word count are right-skewed. Apply `log1p()` transform to normalize.

**Sampling Strategy** — For evaluation: stratified sampling ensures all categories/difficulty levels are represented. Random sampling might miss rare failure modes.

## Code Patterns

```python
# Outlier detection (IQR)
Q1, Q3 = df["score"].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df["score"] < Q1 - 1.5*IQR) | (df["score"] > Q3 + 1.5*IQR)]

# Correlation matrix
df[numeric_cols].corr()

# Log transform skewed data
df["log_latency"] = np.log1p(df["latency_ms"])

# Composite score from multiple metrics
df["composite"] = df["faithfulness"]*0.5 + df["relevance"]*0.3 + df["retrieval"]*0.2
```

## Interview Facts

1. IQR outlier rule: < Q1-1.5×IQR or > Q3+1.5×IQR
2. Pearson correlation: -1 (negative) to 0 (none) to 1 (positive)
3. Log transform reduces right skew (latency, word counts)
4. Stratified sampling preserves class proportions
5. In RAG eval: outliers reveal failure modes, not just noise
