"""
Day 08 — ML Evaluation System
================================
Comprehensive evaluation: metrics, cross-validation, hyperparameter tuning.

Run: python 01_ml_evaluation.py
Requires: pip install numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Any


def demonstrate_metrics() -> None:
    print("\n── CLASSIFICATION METRICS ──")

    np.random.seed(42)
    y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.9, 0.3, 0.1, 0.8, 0.2, 0.7, 0.9, 0.15, 0.4, 0.1,
                       0.85, 0.95, 0.6, 0.05, 0.8])

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Confusion Matrix:")
    print(f"           Pred 0  Pred 1")
    print(f"  True 0:  TN={tn:<4}  FP={fp}")
    print(f"  True 1:  FN={fn:<4}  TP={tp}")
    print()
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.3f}  (correct predictions)")
    print(f"  Precision: {precision_score(y_true, y_pred):.3f}  (of predicted pos, how many correct)")
    print(f"  Recall:    {recall_score(y_true, y_pred):.3f}  (of actual pos, how many found)")
    print(f"  F1:        {f1_score(y_true, y_pred):.3f}  (harmonic mean of P and R)")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_prob):.3f}  (ranking quality, 1.0=perfect)")

    print("\n  When to use which metric:")
    print("  • Accuracy   → balanced classes")
    print("  • Precision  → false positives are costly (spam filter)")
    print("  • Recall     → false negatives are costly (cancer detection)")
    print("  • F1         → balance between P and R")
    print("  • ROC-AUC    → compare models on ranking quality")


def demonstrate_cross_validation() -> None:
    print("\n── CROSS-VALIDATION ──")

    X, y = make_classification(n_samples=500, n_features=8, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ])

    # Single train/test split — unreliable (depends on random seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    single_score = f1_score(y_test, model.predict(X_test))
    print(f"  Single split F1: {single_score:.3f}  (could be lucky or unlucky)")

    # 5-fold CV — more reliable
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"  5-fold CV F1:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Per-fold:        {cv_scores.round(3)}")
    print(f"\n  Rule: CV mean ± 2×std gives 95% confidence interval")
    print(f"  This model: F1 = [{cv_scores.mean() - 2*cv_scores.std():.3f}, "
          f"{cv_scores.mean() + 2*cv_scores.std():.3f}]")


def demonstrate_hyperparameter_tuning() -> None:
    print("\n── HYPERPARAMETER TUNING ──")

    X, y = make_classification(n_samples=400, n_features=6, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 5],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train_s, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.3f}")

    best_model = grid.best_estimator_
    test_f1 = f1_score(y_test, best_model.predict(X_test_s))
    print(f"  Test F1:     {test_f1:.3f}")


def full_evaluation_pipeline() -> None:
    print("\n── FULL EVALUATION PIPELINE ──")

    X, y = make_classification(
        n_samples=800, n_features=10, n_informative=6,
        n_classes=2, class_weight={0: 1, 1: 3},  # Imbalanced
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print(f"  {'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("  " + "─" * 60)

    best_f1 = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  {name:<25} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    print(f"\n  Best model: {best_name} (F1={best_f1:.3f})")
    print(f"  → Evaluate ONCE on test set, then deploy")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 08 — ML EVALUATION SYSTEM                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demonstrate_metrics()
    demonstrate_cross_validation()
    demonstrate_hyperparameter_tuning()
    full_evaluation_pipeline()

    print("\n✓ ML Evaluation demo complete!")
    print("\nKey evaluation rules:")
    print("  1. Choose metric BEFORE training (not after)")
    print("  2. Use cross-validation for reliable estimates")
    print("  3. Tune hyperparameters on val set, report on test set")
    print("  4. For imbalanced data: F1/AUC, not accuracy")
    print("  5. Report confidence intervals, not just point estimates")


if __name__ == "__main__":
    main()
