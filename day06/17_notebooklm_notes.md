# Day 06 — NotebookLM Notes: ML Mathematics

## Key Concepts

**Loss Function** — Measures how wrong model predictions are. MSE for regression; cross-entropy for classification. Training = minimize loss.

**Gradient Descent** — Update rule: `θ = θ - lr × gradient`. Gradient = direction of steepest ascent. Subtract to go downhill. Iterative optimization.

**Learning Rate** — Step size. Too high: oscillates/diverges. Too low: slow convergence. Adam default: 0.001. LLM fine-tuning: 1e-5 to 5e-5.

**Overfitting** — High train accuracy, low test accuracy. Model memorized training data. Signs: large gap between train/val loss.

**Underfitting** — High train AND test loss. Model too simple. Signs: both losses plateau at high value.

**Regularization** — L1 (Lasso): sparse weights, feature selection. L2 (Ridge): all weights small. Dropout: randomly zero neurons during training.

**Bias-Variance Tradeoff** — Total error = Bias² + Variance. High bias = underfitting. High variance = overfitting. Goal: find the sweet spot.

## Formulas

```
MSE = mean((y_true - y_pred)²)
RMSE = √MSE
MAE = mean(|y_true - y_pred|)
Cross-entropy = -mean(y_true × log(y_pred))
Gradient descent: θ = θ - lr × dL/dθ
L2 loss: L + λ × Σwᵢ²
R² = 1 - SS_res / SS_tot
```

## Code Patterns

```python
# From scratch linear regression
for epoch in range(epochs):
    y_pred = X @ weights + bias
    error = y_pred - y
    weights -= lr * (2/n) * (X.T @ error)
    bias -= lr * (2/n) * error.sum()

# Sigmoid for logistic regression
def sigmoid(z): return 1 / (1 + np.exp(-z))
```

## Interview Facts

1. MSE penalizes large errors more (squared); MAE does not
2. Cross-entropy is the loss for classification (LLMs use it for next token)
3. LLM pretraining loss = categorical cross-entropy over 50K vocab tokens
4. Train loss < Val loss = overfitting; both high = underfitting
5. L1 → sparsity; L2 → small weights; Dropout → robust neurons
6. LLM fine-tuning uses very small learning rate (1e-5) to not destroy pretrained knowledge
