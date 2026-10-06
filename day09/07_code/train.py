"""
Day 09 — ML Training Pipeline
================================
Trains a classification model, evaluates it, and saves the pipeline.

Run: python train.py
Creates: models/classifier_v1.pkl
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def create_dataset() -> pd.DataFrame:
    """Create a realistic synthetic dataset."""
    np.random.seed(42)
    n = 2000

    X_base, y = make_classification(
        n_samples=n, n_features=6, n_informative=4,
        n_classes=2, class_weight={0: 1, 1: 1.5},
        random_state=42,
    )
    feature_names = ["feature_a", "feature_b", "feature_c",
                     "feature_d", "feature_e", "feature_f"]
    df = pd.DataFrame(X_base, columns=feature_names)
    df["target"] = y

    # Add some noise
    df.loc[np.random.choice(df.index, 50), "feature_a"] = np.nan
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """Clean and split data."""
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")

    # Fill missing values
    for col in df.columns:
        if col != "target" and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def select_best_model(X_train, y_train, X_val, y_val) -> tuple:
    """Compare models on validation set."""
    logger.info("Comparing models...")

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        f1 = f1_score(y_val, model.predict(X_val))
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        results[name] = {"f1": f1, "auc": auc, "model": model}
        logger.info(f"  {name}: F1={f1:.3f}, AUC={auc:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    logger.info(f"Best model: {best_name}")
    return results[best_name]["model"], best_name


def final_evaluation(model, X_test, y_test) -> dict:
    """Evaluate on held-out test set (only once)."""
    logger.info("Final test set evaluation...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "test_samples": len(y_test),
    }
    logger.info(f"Test F1: {metrics['f1']:.3f}, AUC: {metrics['roc_auc']:.3f}")
    return metrics


def save_model(model, feature_cols: list, metrics: dict, model_name: str) -> Path:
    """Save model with metadata."""
    Path("models").mkdir(exist_ok=True)

    model_path = Path("models/classifier_v1.pkl")
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "version": "1.0.0",
        "trained_at": datetime.utcnow().isoformat(),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "metrics": metrics,
    }

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to models/model_metadata.json")
    return model_path


def main() -> None:
    logger.info("="*50)
    logger.info("DAY 09 — ML TRAINING PIPELINE")
    logger.info("="*50)

    df = create_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_data(df)
    best_model, best_name = select_best_model(X_train, y_train, X_val, y_val)
    metrics = final_evaluation(best_model, X_test, y_test)
    model_path = save_model(best_model, feature_cols, metrics, best_name)

    logger.info("Training complete!")
    logger.info(f"Run: uvicorn api:app --reload")


if __name__ == "__main__":
    main()
