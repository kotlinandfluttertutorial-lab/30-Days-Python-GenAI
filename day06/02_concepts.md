# Day 06 — Concepts
## ML Mathematics: Loss Functions, Gradient Descent, Regularization

---

## 1. Loss Functions

A loss function measures how wrong the model's predictions are.
The goal of training: minimize the loss.

```python
import numpy as np

# Mean Squared Error (MSE) — for regression
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

# Root MSE — same units as target
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))

# Mean Absolute Error — robust to outliers
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

# Binary Cross-Entropy — for binary classification (spam/not-spam)
def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-15) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Categorical Cross-Entropy — for multi-class (which token? which category?)
def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-15) -> float:
    y_pred = np.clip(y_pred, eps, 1.0)
    return -np.sum(y_true * np.log(y_pred))

# Example: training LLM
# y_true = one-hot for correct next token (e.g., "Paris")
# y_pred = softmax output probabilities over 50K vocab
# loss = categorical_crossentropy(y_true, y_pred)
# Training: minimize this loss over trillions of examples
```

---

## 2. Gradient Descent

Gradient descent is how every ML model learns. It iteratively moves parameters in the direction that reduces the loss.

```python
# Intuition: you're on a hill in a foggy landscape
# You feel the slope under your feet
# You take a small step downhill
# Repeat until you reach a valley (minimum loss)

# Math:
# θ_new = θ_old - learning_rate × gradient

# Gradient = derivative of loss with respect to parameter
# = "how much does the loss change if I increase this parameter slightly?"
# Positive gradient → increasing parameter increases loss → decrease it
# Negative gradient → increasing parameter decreases loss → increase it

# Simple linear regression example:
def gradient_descent_linear_regression(
    X: np.ndarray,         # features: (n_samples, n_features)
    y: np.ndarray,         # targets: (n_samples,)
    learning_rate: float = 0.01,
    epochs: int = 1000,
) -> tuple[np.ndarray, float]:
    """
    Fit linear regression using gradient descent.
    Returns (weights, bias).
    """
    n_samples = len(X)
    weights = np.zeros(X.shape[1])
    bias = 0.0
    loss_history = []

    for epoch in range(epochs):
        # Forward pass: predict
        y_pred = X @ weights + bias

        # Compute loss
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)

        # Backward pass: compute gradients
        # dL/dw = (2/n) × X^T × (y_pred - y)
        # dL/db = (2/n) × sum(y_pred - y)
        error = y_pred - y
        dw = (2 / n_samples) * X.T @ error
        db = (2 / n_samples) * error.sum()

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}: Loss = {loss:.6f}")

    return weights, bias
```

---

## 3. Learning Rate

```
Too high learning rate → overshooting, divergence
Too low learning rate → very slow convergence, might get stuck

                    Loss
                     │
                     │      too high: bounces around, diverges
                     │
     good LR ────────┼──────────────────────────────────────── good
                     │
     too low ────────┼──────────────────────────────────────── barely moves
                     │
                     └────────────────────────────────── epochs

Common starting values:
  - Adam optimizer: 0.001
  - SGD: 0.01 - 0.1
  - Fine-tuning LLMs: 1e-5 to 5e-5 (very small, don't destroy learned knowledge)
```

---

## 4. Overfitting and Underfitting

```
UNDERFITTING (high bias):
  Training loss: HIGH
  Validation loss: HIGH
  Model too simple, hasn't learned enough
  Fix: bigger model, more features, less regularization, train longer

OVERFITTING (high variance):
  Training loss: LOW
  Validation loss: HIGH
  Model memorized training data, doesn't generalize
  Fix: regularization, more data, simpler model, dropout, early stopping

GOOD FIT:
  Training loss: LOW
  Validation loss: LOW (and close to training loss)

The goal: minimize validation loss, not training loss!
```

---

## 5. Regularization

Regularization penalizes model complexity to prevent overfitting.

```python
# L1 Regularization (Lasso): adds sum(|weights|) to loss
# Effect: pushes some weights to exactly 0 → feature selection

# L2 Regularization (Ridge): adds sum(weights²) to loss
# Effect: keeps all weights small, none exactly 0

# L1 + L2 (Elastic Net): combines both

# Example: logistic regression with L2 regularization
from sklearn.linear_model import LogisticRegression

# C = 1/lambda (inverse regularization strength)
# Small C = strong regularization
# Large C = weak regularization
model = LogisticRegression(C=0.1, penalty='l2')

# Dropout (neural networks):
# Randomly zero out neurons during training
# Forces network to be robust (can't rely on specific neurons)
# torch.nn.Dropout(p=0.3)  # 30% dropout rate
```

---

## 6. Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

Bias: error from wrong assumptions in the model
  High bias = model too simple (linear for curved data)
  → underfits, wrong on both train and test

Variance: error from sensitivity to training data fluctuations
  High variance = model too complex
  → overfits, great on train, poor on test

Finding the sweet spot is model selection.
```

---

## 7. Linear Regression from Scratch

```python
class LinearRegression:
    """Linear regression implemented from scratch with gradient descent."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray = np.array([])
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            self.weights -= self.lr * (2/n) * X.T @ error
            self.bias -= self.lr * (2/n) * error.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score: 1.0 = perfect, 0 = predicts mean."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot
```
