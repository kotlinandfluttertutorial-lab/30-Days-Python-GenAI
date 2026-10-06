# Day 03 — Debugging Guide

---

## NumPy Debugging

**Shape mismatch errors:**
```python
# Error: shapes (768,) and (1536,) not aligned
a = np.array([...])  # shape (768,)
b = np.array([...])  # shape (1536,)
np.dot(a, b)         # ValueError!

# Debug: always print shapes
print(f"a.shape: {a.shape}, b.shape: {b.shape}")
```

**Division by zero in normalization:**
```python
# Can happen with zero vectors (empty text embedded)
norm = np.linalg.norm(v)
if norm == 0:
    return np.zeros_like(v)  # Handle gracefully
normalized = v / norm
```

**Unexpected NaN values:**
```python
scores = batch_cosine_sim(query, docs)
print(f"NaN count: {np.isnan(scores).sum()}")
# NaN often means a zero-magnitude vector was embedded
```

---

## Pandas Debugging

**SettingWithCopyWarning:**
```python
# Warning when modifying a slice
filtered = df[df["score"] > 0.5]
filtered["new_col"] = "value"  # Warning!

# Fix: use .copy()
filtered = df[df["score"] > 0.5].copy()
filtered["new_col"] = "value"  # Clean
```

**Type errors in operations:**
```python
# df["word_count"] is object dtype (strings), not int
df["word_count"] = df["text"].str.split().str.len()
# Now it's int — operations work
avg = df["word_count"].mean()
```

**KeyError on column:**
```python
# Always check column names before accessing
print(df.columns.tolist())
# Column names might have trailing spaces: "score " not "score"
df.columns = df.columns.str.strip()
```

---

## OOP Debugging

**TypeError: Can't instantiate abstract class:**
```
TypeError: Can't instantiate abstract class MyEmbedder 
with abstract methods embed, embed_batch

Fix: implement ALL abstract methods in your subclass.
Run: print(MyEmbedder.__abstractmethods__) to see what's missing.
```
