# Day 05 — Concepts
## Statistics + Data Analysis

---

## 1. Distributions

```python
import numpy as np
from scipy import stats  # pip install scipy

# Normal (Gaussian) distribution
# Most natural phenomena follow this
# Mean ± 2std contains 95.4% of data
np.random.seed(42)
scores = np.random.normal(loc=0.82, scale=0.1, size=1000)
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")

# Check if data is normally distributed
stat, p_value = stats.normaltest(scores)
print(f"Normal test p-value: {p_value:.4f}")  # >0.05 = likely normal

# Uniform distribution
# All values equally likely
uniform = np.random.uniform(0, 1, 1000)  # e.g., temperature values

# Bernoulli / Binomial
# Binary events (spam/not-spam, relevant/irrelevant)
binary = np.random.binomial(n=1, p=0.3, size=1000)  # 30% spam
```

---

## 2. Sampling and Population

```python
# Population: ALL possible examples
# Sample: a subset we observe

# Why sampling matters in AI evaluation:
# - We can't label millions of LLM responses
# - We sample 500, evaluate quality, infer about all outputs

# Representative sampling: sample must reflect the population
# Stratified sampling: sample proportionally from each group
from sklearn.model_selection import StratifiedShuffleSplit

# When splitting a dataset with class imbalance:
# Class A: 90%, Class B: 10%
# Random split might give test set with 100% Class A
# Stratified split preserves proportions in each split
```

---

## 3. Correlation and Covariance

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "word_count": np.random.randint(50, 500, 200),
    "quality_score": np.random.uniform(0.5, 1.0, 200),
    "relevance": np.random.uniform(0.3, 1.0, 200),
})

# Pearson correlation: -1 to 1
# 1 = perfect positive correlation
# 0 = no correlation
# -1 = perfect negative correlation
correlation = df["word_count"].corr(df["quality_score"])
print(f"Word count vs quality: {correlation:.3f}")

# Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Covariance: raw (not normalized) measure of joint variation
cov = df["word_count"].cov(df["quality_score"])
# Correlation = covariance / (std_a × std_b)
```

---

## 4. Outlier Detection

```python
# Method 1: Z-score (for normally distributed data)
z_scores = np.abs(stats.zscore(df["quality_score"]))
outliers_z = df[z_scores > 3]  # >3 std devs from mean

# Method 2: IQR (Interquartile Range) — more robust
Q1 = df["quality_score"].quantile(0.25)
Q3 = df["quality_score"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = df[(df["quality_score"] < lower) | (df["quality_score"] > upper)]

print(f"Z-score outliers: {len(outliers_z)}")
print(f"IQR outliers: {len(outliers_iqr)}")

# In RAG evaluation: outliers are often the most important cases!
# Very low scores = system failures to investigate
# Very high scores = what worked perfectly? Replicate it.
```

---

## 5. Feature Engineering Patterns

```python
# For RAG evaluation dataset
df["log_word_count"] = np.log1p(df["word_count"])  # Log transform skewed data
df["score_percentile"] = df["quality_score"].rank(pct=True)  # Rank percentile
df["score_bucket"] = pd.qcut(df["quality_score"], q=4, labels=["low", "med", "high", "top"])

# Interaction features
df["length_quality"] = df["word_count"] * df["quality_score"]

# Rolling statistics (for time-series evaluation)
df_sorted = df.sort_values("timestamp") if "timestamp" in df else df
# df["rolling_avg"] = df["quality_score"].rolling(window=10).mean()
```

---

## 6. Data Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# A pipeline ensures consistent preprocessing for train/val/test
preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Fill nulls with median
    ("scaler", StandardScaler()),                    # Normalize
])

# FIT on train, TRANSFORM on val/test
X_train_processed = preprocessing.fit_transform(X_train)
X_val_processed = preprocessing.transform(X_val)   # Note: transform only!
X_test_processed = preprocessing.transform(X_test) # Note: transform only!
```
