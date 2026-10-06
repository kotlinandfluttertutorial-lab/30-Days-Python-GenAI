# Day 09 — Concepts
## End-to-End Machine Learning

---

## The Production ML Pipeline

```
DATA SOURCE (CSV/DB/API)
    ↓
DATA LOADING + VALIDATION
    ↓
EXPLORATORY DATA ANALYSIS
    ↓
PREPROCESSING PIPELINE
  • Handle missing values
  • Encode categoricals
  • Scale numerics
    ↓
FEATURE ENGINEERING
    ↓
MODEL TRAINING
  • Multiple algorithms
  • Cross-validation
  • Hyperparameter tuning
    ↓
MODEL EVALUATION
  • Final test set metrics
  • Confusion matrix
  • Feature importance
    ↓
MODEL SERIALIZATION
  • joblib.dump(pipeline, "model.pkl")
    ↓
FASTAPI SERVICE
  • POST /predict
  • POST /predict/batch
  • GET /health
  • GET /model/info
    ↓
DOCKER CONTAINER
  • Dockerfile
  • docker-compose.yml
    ↓
DEPLOYMENT
```

---

## Model Serialization

```python
import joblib
from sklearn.pipeline import Pipeline

# Save model + preprocessing pipeline together
pipeline = Pipeline([...])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "models/sentiment_classifier_v1.0.pkl")

# Load for inference
loaded_pipeline = joblib.load("models/sentiment_classifier_v1.0.pkl")
predictions = loaded_pipeline.predict(X_new)

# Version your models!
# models/
# ├── sentiment_v1.0.pkl   ← current production
# ├── sentiment_v1.1.pkl   ← A/B test challenger
# └── sentiment_v0.9.pkl   ← rollback option
```

---

## Production ML Service Checklist

```python
# ✓ Input validation (Pydantic)
# ✓ Type conversion before model.predict()
# ✓ Model loaded at startup, not per-request
# ✓ Prediction + probability returned
# ✓ /health endpoint
# ✓ /model/info endpoint (version, features, training date)
# ✓ Error handling (invalid input → 422, model error → 500)
# ✓ Logging every prediction
# ✓ Response time in headers
# ✓ Rate limiting
# ✓ Dockerfile
# ✓ docker-compose.yml for local testing
```
