"""
Day 07 — Machine Learning Fundamentals
=======================================
All core ML algorithms with sklearn. Compare performance.
Build your first ML prediction API scaffold.

Run: python 01_ml_fundamentals.py
Requires: pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score,
)
from typing import Any


def classification_comparison() -> None:
    print("\n── CLASSIFICATION ALGORITHM COMPARISON ──")

    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=6,
        n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000),
        "Decision Tree (d=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Decision Tree (d=20)": DecisionTreeClassifier(max_depth=20, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    print(f"  {'Model':<30} {'Train Acc':>10} {'Test Acc':>10} {'Note'}")
    print("  " + "─" * 70)

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        train_acc = model.score(X_train_s, y_train)
        test_acc = model.score(X_test_s, y_test)
        gap = train_acc - test_acc
        note = "⚠️ Overfit" if gap > 0.05 else "✓" if test_acc > 0.8 else "Underfit"
        print(f"  {name:<30} {train_acc:>10.3f} {test_acc:>10.3f} {note}")

    # Best model detail
    best = RandomForestClassifier(n_estimators=100, random_state=42)
    best.fit(X_train_s, y_train)
    y_pred = best.predict(X_test_s)
    print(f"\n  Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

    # Feature importance
    importances = best.feature_importances_
    top3 = np.argsort(importances)[::-1][:3]
    print(f"  Top 3 important features: {top3}")
    print(f"  Their importances: {importances[top3]}")


def clustering_demo() -> None:
    print("\n── K-MEANS CLUSTERING ──")

    # Create clustered data
    from sklearn.datasets import make_blobs
    X, true_labels = make_blobs(n_samples=300, centers=4, random_state=42)

    # Find optimal K using inertia (elbow method)
    inertias = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    print("  Inertia by K (look for 'elbow'):")
    for k, inertia in enumerate(inertias, 1):
        bar = "█" * int(inertia / inertias[0] * 20)
        print(f"    K={k}: {bar} {inertia:.0f}")

    # Fit best K
    best_k = 4
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(true_labels, labels)
    print(f"\n  K={best_k} ARI (vs true clusters): {ari:.3f}  (1.0=perfect)")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 07 — MACHINE LEARNING FUNDAMENTALS            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    classification_comparison()
    clustering_demo()

    print("\n✓ ML Fundamentals demo complete!")
    print("\nAlgorithm Selection Guide:")
    print("  Continuous target  → Linear Regression (start), RandomForest")
    print("  Binary/Multi-class → Logistic Regression (start), RandomForest")
    print("  Find groups        → K-Means")
    print("  Always compare:    Baseline → LogReg/LinReg → RandomForest → Tuned")


if __name__ == "__main__":
    main()
