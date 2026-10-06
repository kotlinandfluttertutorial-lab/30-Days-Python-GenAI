"""
Day 07 — ML Prediction API
============================
First FastAPI + ML model integration.
This pattern is used in Day 9 (full production ML service).

Run: uvicorn 02_ml_prediction_api:app --reload
Test: http://localhost:8000/docs
"""

import pickle
import numpy as np
from pathlib import Path
from pydantic import BaseModel, field_validator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from fastapi import FastAPI, HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ─────────────────────────────────────────────────────────
# MODEL TRAINING (runs once at startup)
# ─────────────────────────────────────────────────────────

def train_model() -> Pipeline:
    """Train a simple classification model."""
    X, y = make_classification(
        n_samples=1000, n_features=4, n_informative=3, random_state=42
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X, y)
    print(f"Model trained. Feature count: 4")
    return pipeline


# ─────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError(f"Expected 4 features, got {len(v)}")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str = "1.0.0"


# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────

if HAS_FASTAPI:
    app = FastAPI(
        title="ML Prediction API",
        description="Day 07 — Simple ML model served via FastAPI",
        version="1.0.0",
    )

    # Load model at startup
    model = train_model()

    @app.get("/health")
    def health_check() -> dict:
        return {"status": "healthy", "model": "RandomForest v1.0"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Make a prediction for the given features."""
        try:
            X = np.array(request.features).reshape(1, -1)
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0, prediction])
            return PredictionResponse(
                prediction=prediction,
                probability=probability,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch")
    def predict_batch(requests: list[PredictionRequest]) -> list[PredictionResponse]:
        """Batch predictions."""
        results = []
        X = np.array([r.features for r in requests])
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        for pred, probs in zip(predictions, probabilities):
            results.append(PredictionResponse(
                prediction=int(pred),
                probability=float(probs[int(pred)]),
            ))
        return results
else:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    print("Then: uvicorn 02_ml_prediction_api:app --reload")


# ─────────────────────────────────────────────────────────
# STANDALONE TEST (no FastAPI required)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training model and testing predictions...")
    pipeline = train_model()

    test_samples = [
        [0.5, -1.2, 0.8, 1.5],
        [-0.3, 0.7, -0.5, 0.2],
        [1.0, 1.5, 0.3, -0.8],
    ]

    for features in test_samples:
        X = np.array(features).reshape(1, -1)
        pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0, pred]
        print(f"  Features: {features} → Class: {pred}, Confidence: {proba:.3f}")

    if HAS_FASTAPI:
        print("\nTo run the API:")
        print("  uvicorn 02_ml_prediction_api:app --reload")
        print("  Then visit http://localhost:8000/docs")
