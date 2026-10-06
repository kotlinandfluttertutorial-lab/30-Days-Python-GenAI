# Day 08 — Concepts
## Machine Learning Evaluation

---

## 1. Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
import numpy as np

y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0])
y_prob = np.array([0.9, 0.3, 0.1, 0.8, 0.2, 0.7, 0.85, 0.15, 0.4, 0.05])

# Accuracy: fraction correct
# MISLEADING for imbalanced classes (99% accuracy by always predicting majority)
acc = accuracy_score(y_true, y_pred)  # 0.7

# Confusion Matrix: TP, FP, FN, TN
cm = confusion_matrix(y_true, y_pred)
# [[TN, FP],
#  [FN, TP]]

# Precision: of predicted positive, how many were actually positive?
# "Don't cry wolf" — high precision = few false alarms
prec = precision_score(y_true, y_pred)  # TP / (TP + FP)

# Recall (Sensitivity): of actual positive, how many did we catch?
# "Don't miss cancer" — high recall = few misses
rec = recall_score(y_true, y_pred)  # TP / (TP + FN)

# F1: harmonic mean of precision and recall
# Balances precision and recall
f1 = f1_score(y_true, y_pred)  # 2 × (P × R) / (P + R)

# ROC-AUC: area under ROC curve (probability scores)
# 0.5 = random, 1.0 = perfect
auc = roc_auc_score(y_true, y_prob)

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1:        {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}")
print(classification_report(y_true, y_pred))
```

---

## 2. When to Use Which Metric

```
ACCURACY    → balanced classes, simple overview
PRECISION   → cost of false positives is high (spam filter: don't block legit email)
RECALL      → cost of false negatives is high (cancer detection: don't miss)
F1          → when you need to balance precision and recall
ROC-AUC     → evaluating ranking quality, comparing models
PR-AUC      → highly imbalanced classes (fraud detection)

EXAMPLE DECISIONS:
Spam filter:     High precision (don't block legit email)
Cancer screening: High recall (don't miss cancer)
Fraud detection: High recall + good precision → F1 or PR-AUC
LLM classifier: F1 (usually balanced)
RAG relevance:  Precision@K (top-K retrieved are relevant)
```

---

## 3. Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.5])
y_pred = np.array([2.5, 0.0, 2.1, 7.8, 4.0])

mse = mean_squared_error(y_true, y_pred)        # Penalizes large errors
rmse = np.sqrt(mse)                             # Same units as target
mae = mean_absolute_error(y_true, y_pred)       # Robust to outliers
r2 = r2_score(y_true, y_pred)                   # 1.0=perfect, 0=predicts mean

print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
```

---

## 4. Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold CV: train on 80%, test on 20%, repeat 5 times
# Gives more reliable estimate than single train/test split
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# StratifiedKFold: preserves class proportions in each fold
# Critical for imbalanced datasets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

---

## 5. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search: try ALL combinations
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}

# 3 × 3 × 3 = 27 combinations × 5 folds = 135 model fits!
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,  # Use all CPU cores
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score:  {grid_search.best_score_:.3f}")

# RandomizedSearch: more efficient for large param spaces
from scipy.stats import randint
param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": [None, 5, 10, 20, 50],
}
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,   # Only try 20 random combinations
    cv=5,
    scoring="f1",
    random_state=42,
)
rand_search.fit(X_train, y_train)
```

---

## 6. Model Selection Framework

```python
from sklearn.model_selection import cross_validate

def evaluate_model(model, X, y, cv=5):
    """Comprehensive model evaluation."""
    results = cross_validate(
        model, X, y, cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
        return_train_score=True,
    )
    return {
        "test_accuracy": results["test_accuracy"].mean(),
        "test_f1": results["test_f1"].mean(),
        "test_roc_auc": results["test_roc_auc"].mean(),
        "train_f1": results["train_f1"].mean(),
        "overfit_gap": results["train_f1"].mean() - results["test_f1"].mean(),
    }
```
