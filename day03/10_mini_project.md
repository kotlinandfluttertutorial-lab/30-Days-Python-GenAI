# Day 03 — Mini Project
## AI Dataset Analyzer

**Code:** `07_code/02_numpy_pandas.py` (includes full pipeline)

---

## What You Build

A CLI data pipeline that takes any CSV and prepares it for ML/RAG:
1. Load and profile the dataset
2. Clean (handle nulls, duplicates, short texts)
3. Engineer features (word count, text length, etc.)
4. Split into train/val/test
5. Export ready-to-use splits

## Run It

```bash
pip install numpy pandas scikit-learn
python 02_numpy_pandas.py
```

## Expected Output

```
── NUMPY: FUNDAMENTALS ──
  Embedding shape: (4,), dtype: float64
  Batch similarity (1000 docs, 768 dims): 0.5ms
  Python loop equivalent: 280ms (NumPy is 500x faster)

── PANDAS: DATA PROFILING ──
  Shape: 500 rows × 7 columns
  Missing values:
    has_missing         23 (4.6%)

── PANDAS: FEATURE ENGINEERING ──
  Features added: char_count, sentence_count, avg_word_length, ...
  New shape: (487, 15)

── PANDAS: TRAIN / VAL / TEST SPLIT ──
  Train: 340 (70%), Val: 73 (15%), Test: 74 (15%)
```

## Extension Tasks

1. Add outlier detection (IQR method) for numeric columns
2. Add correlation analysis (which features correlate with quality_score?)
3. Export a full HTML report using `df.describe().to_html()`
4. Add support for JSON input files
