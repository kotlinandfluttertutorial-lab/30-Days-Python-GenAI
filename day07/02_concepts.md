# Day 07 — Concepts
## Machine Learning Fundamentals

---

## 1. Supervised vs Unsupervised Learning

```
SUPERVISED: You have labels (correct answers)
  Input: features → Model → Output: prediction
  Examples: spam classification, house price prediction, sentiment analysis
  Algorithms: Linear Regression, Logistic Regression, Decision Tree, Random Forest, SVM

UNSUPERVISED: No labels, find patterns
  Input: features → Model → Output: structure/clusters
  Examples: customer segmentation, document clustering, anomaly detection
  Algorithms: K-Means, DBSCAN, PCA, autoencoders

SEMI-SUPERVISED: Some labels, mostly unlabeled
  LLM pretraining is semi-supervised (predicting next token)
  Useful when labeling is expensive
```

---

## 2. Key Algorithms

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)  # R²

# When to use:
# - Predict a continuous value
# - Relationship between features and target is roughly linear
# - Need interpretable model (coefficients tell you importance)
# - Baselines before trying complex models
```

### Logistic Regression (Classification)
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)           # class labels
probabilities = model.predict_proba(X_test)   # probability per class

# When to use:
# - Binary or multi-class classification
# - Need probability estimates (not just labels)
# - Need interpretable model
# - Good baseline before neural networks
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Pros: Interpretable, handles non-linear relationships, no feature scaling needed
# Cons: Prone to overfitting (fix: limit max_depth or use Random Forest)
# When to use: when interpretability matters, mixed feature types
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_

# Why it works:
# - Many decision trees, each trained on random subset of data + features
# - Average predictions = reduces variance (overfitting)
# - "Ensemble" method

# When to use:
# - When decision tree overfits
# - When you need feature importance scores
# - When you don't have time to tune a complex model
# - One of the best "off-the-shelf" algorithms
```

### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
model.fit(X_train, y_train)

# How it works:
# For each test point, find K nearest training points
# Majority class wins (classification) or mean (regression)

# When to use:
# - Small datasets
# - Need simple interpretability
# - For embedding-based classification (cosine distance)
# When NOT to use: large datasets (O(n) prediction time)
```

### K-Means Clustering
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5, random_state=42)
labels = model.fit_predict(X)

# How to choose K: elbow method
# - Run KMeans for K=1 to 15
# - Plot inertia (sum of squared distances to cluster center)
# - Pick the "elbow" where adding more clusters gives diminishing returns

inertias = []
for k in range(1, 15):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
```

---

## 3. Feature Engineering for ML

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Numeric features
df["score_log"] = np.log1p(df["score"])
df["score_squared"] = df["score"] ** 2
df["score_interaction"] = df["score_a"] * df["score_b"]

# Text features (pre-embeddings, classical NLP)
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
text_features = tfidf.fit_transform(df["text"])

# Categorical encoding
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["category"])
df_encoded = pd.get_dummies(df, columns=["category"])

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

## 4. Model Selection Framework

```
1. Start with a simple baseline (mean prediction, logistic regression)
2. Check: is the problem solvable? Is data quality good?
3. Try Random Forest (usually good without tuning)
4. If more performance needed: gradient boosting (XGBoost/LightGBM)
5. If text/sequence: neural networks (Day 10-12)
6. If images: CNN (Day 12)
7. Evaluate ALL models on the same val set
8. Pick the best model, evaluate ONCE on test set

NEVER skip the baseline. A "good" ML model is only good
if it beats a simple baseline.
```
