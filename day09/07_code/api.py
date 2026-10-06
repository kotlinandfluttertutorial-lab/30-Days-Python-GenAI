"""
Day 09 — Production ML Prediction Service
==========================================
FastAPI service that serves the trained ML model.

Setup:
  1. python train.py         ← train and save model
  2. uvicorn api:app --reload  ← start API
  3. http://localhost:8000/docs  ← interactive docs

This is PROJECT 1: ML Prediction System.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, field_validator

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    features: list[float]
    request_id: str | None = None

    @field_validator("features")
    @classmethod
    def must_have_features(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("features cannot be empty")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str  # "high", "medium", "low"
    model_version: str
    latency_ms: float
    request_id: str | None = None


class BatchRequest(BaseModel):
    items: list[PredictionRequest]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


# ─────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────

class ModelManager:
    """Loads and manages the ML model."""

    def __init__(self, model_path: str = "models/classifier_v1.pkl",
                 metadata_path: str = "models/model_metadata.json") -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model: Any = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        """Load model at startup."""
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. Run train.py first.")
            return

        logger.info(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)

        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)

        logger.info(f"Model loaded: {self.metadata.get('model_name', 'unknown')} v{self.metadata.get('version', '?')}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unknown")

    @property
    def expected_features(self) -> int:
        return self.metadata.get("n_features", 0)

    def predict(self, features: list[float]) -> tuple[int, float]:
        """Returns (prediction, probability)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        X = np.array(features).reshape(1, -1)
        if self.expected_features > 0 and X.shape[1] != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features, got {X.shape[1]}"
            )

        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0, prediction])
        return prediction, probability


# ─────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────

def create_app() -> Any:
    if not HAS_FASTAPI:
        return None

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="ML Prediction Service",
        description="Production ML model serving API — Day 09",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = ModelManager()

    @app.on_event("startup")
    async def startup() -> None:
        manager.load()

    @app.middleware("http")
    async def add_timing(request: Request, call_next: Any) -> Any:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy" if manager.is_loaded else "degraded",
            model_loaded=manager.is_loaded,
            model_version=manager.version if manager.is_loaded else None,
        )

    @app.get("/model/info")
    async def model_info() -> dict:
        if not manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "version": manager.version,
            "n_features": manager.expected_features,
            "metrics": manager.metadata.get("metrics", {}),
            "trained_at": manager.metadata.get("trained_at"),
            "model_type": manager.metadata.get("model_name"),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        if not manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

        start = time.perf_counter()
        try:
            prediction, probability = manager.predict(request.features)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Prediction failed")

        latency_ms = (time.perf_counter() - start) * 1000

        confidence = "high" if probability >= 0.8 else "medium" if probability >= 0.6 else "low"

        logger.info(
            "prediction",
            extra={
                "prediction": prediction,
                "probability": probability,
                "latency_ms": latency_ms,
                "request_id": request.request_id,
            }
        )

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            confidence=confidence,
            model_version=manager.version,
            latency_ms=round(latency_ms, 2),
            request_id=request.request_id,
        )

    @app.post("/predict/batch")
    async def predict_batch(request: BatchRequest) -> list[PredictionResponse]:
        if not manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        results = []
        for item in request.items:
            start = time.perf_counter()
            try:
                prediction, probability = manager.predict(item.features)
                latency_ms = (time.perf_counter() - start) * 1000
                confidence = "high" if probability >= 0.8 else "medium" if probability >= 0.6 else "low"
                results.append(PredictionResponse(
                    prediction=prediction,
                    probability=round(probability, 4),
                    confidence=confidence,
                    model_version=manager.version,
                    latency_ms=round(latency_ms, 2),
                    request_id=item.request_id,
                ))
            except Exception as e:
                logger.error(f"Batch item error: {e}")
                results.append(PredictionResponse(
                    prediction=-1,
                    probability=0.0,
                    confidence="low",
                    model_version=manager.version,
                    latency_ms=0.0,
                ))

        return results

    return app


# Create app instance (loaded by uvicorn)
app = create_app()


if __name__ == "__main__":
    # Quick test without FastAPI
    manager = ModelManager()
    manager.load()

    if manager.is_loaded:
        test_features = [0.5, -1.2, 0.8, 1.5, 0.3, -0.7]
        prediction, probability = manager.predict(test_features)
        print(f"Test prediction: {prediction} (confidence: {probability:.3f})")
        print("\nTo run the API:")
        print("  uvicorn api:app --reload --port 8000")
        print("  Then: http://localhost:8000/docs")
    else:
        print("Model not found. Run train.py first:")
        print("  python train.py")
