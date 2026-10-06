# Day 09 — NotebookLM Notes: End-to-End ML

## Production ML Pipeline Steps

1. **Load + Validate** — check types, shapes, missing values
2. **EDA** — understand distributions, correlations, outliers
3. **Preprocess** — fill nulls, encode, scale (in Pipeline)
4. **Train** — multiple algorithms, cross-validation
5. **Evaluate** — val set for model selection, test set once
6. **Serialize** — joblib.dump(pipeline, "model.pkl")
7. **Serve** — FastAPI with /predict, /health, /model/info
8. **Deploy** — Docker container

## Key Patterns

```python
# Pipeline prevents leakage
pipe = Pipeline([("scaler", StandardScaler()), ("model", RF())])
pipe.fit(X_train, y_train)        # fits scaler + model
pipe.predict(X_test)              # transforms + predicts

# Save and load
joblib.dump(pipe, "model.pkl")
pipe = joblib.load("model.pkl")

# FastAPI prediction endpoint
@app.post("/predict")
async def predict(req: PredictionRequest) -> PredictionResponse:
    X = np.array(req.features).reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, pred]
    return PredictionResponse(prediction=pred, probability=prob)
```

## Production Checklist

- [ ] Model loaded at startup (not per-request)
- [ ] Input validation with Pydantic
- [ ] Feature count validation
- [ ] /health endpoint
- [ ] /model/info with version and metrics
- [ ] Logging every prediction
- [ ] Error handling (422 for bad input, 503 if model not loaded)
- [ ] Docker + docker-compose

## Interview Facts

1. `Pipeline` wraps preprocessing + model: `fit` calls both, `predict` calls both
2. `joblib` is faster than `pickle` for numpy arrays (used in sklearn models)
3. Load model at startup: `@app.on_event("startup")` in FastAPI
4. Always return probability with prediction (confidence matters)
5. Version your models: `model_v1.0.pkl`, `model_v1.1.pkl`
6. Docker copies code + trains at build OR mounts pre-trained model
