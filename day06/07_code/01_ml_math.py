"""
Day 06 — ML Mathematics: Loss Functions + Gradient Descent
===========================================================
Implements ML math from scratch to build deep understanding.

Run: python 01_ml_math.py
Requires: pip install numpy scikit-learn
"""

import numpy as np
from typing import Callable


# ─────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────
# LINEAR REGRESSION FROM SCRATCH
# ─────────────────────────────────────────────────────────

class LinearRegressionScratch:
    """Linear regression via gradient descent."""

    def __init__(self, lr: float = 0.01, epochs: int = 500) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights_: np.ndarray = np.array([])
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        n, d = X.shape
        self.weights_ = np.zeros(d)
        self.bias_ = 0.0
        self.loss_history_ = []

        for epoch in range(self.epochs):
            y_pred = X @ self.weights_ + self.bias_
            error = y_pred - y
            loss = float(np.mean(error ** 2))
            self.loss_history_.append(loss)

            self.weights_ -= self.lr * (2 / n) * (X.T @ error)
            self.bias_ -= self.lr * (2 / n) * error.sum()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights_ + self.bias_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r_squared(y, self.predict(X))


# ─────────────────────────────────────────────────────────
# LOGISTIC REGRESSION FROM SCRATCH
# ─────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))


class LogisticRegressionScratch:
    """Binary logistic regression via gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 500) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights_: np.ndarray = np.array([])
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        n, d = X.shape
        self.weights_ = np.zeros(d)
        self.bias_ = 0.0

        for epoch in range(self.epochs):
            y_pred = sigmoid(X @ self.weights_ + self.bias_)
            error = y_pred - y
            self.weights_ -= self.lr * (1 / n) * (X.T @ error)
            self.bias_ -= self.lr * (1 / n) * error.sum()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ─────────────────────────────────────────────────────────
# OVERFITTING DEMONSTRATION
# ─────────────────────────────────────────────────────────

def demonstrate_overfitting() -> None:
    print("\n── OVERFITTING DEMONSTRATION ──")
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X.ravel() + np.sin(X.ravel()) + np.random.randn(100) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"  {'Model':<30} {'Train R²':>10} {'Test R²':>10} {'Diagnosis'}")
    print("  " + "─" * 65)

    for degree, alpha in [(1, 0.0), (3, 0.0), (15, 0.0), (15, 1.0)]:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_poly, y_train)

        train_r2 = model.score(X_train_poly, y_train)
        test_r2 = model.score(X_test_poly, y_test)

        if train_r2 - test_r2 > 0.3:
            diagnosis = "Overfitting ⚠️"
        elif test_r2 < 0.5:
            diagnosis = "Underfitting"
        else:
            diagnosis = "Good fit ✓"

        label = f"degree={degree}, ridge={alpha}"
        print(f"  {label:<30} {train_r2:>10.3f} {test_r2:>10.3f} {diagnosis}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 06 — ML MATHEMATICS                           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Test loss functions
    print("\n── LOSS FUNCTIONS ──")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    print(f"  MSE:  {mse(y_true, y_pred):.4f}")
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  MAE:  {mae(y_true, y_pred):.4f}")
    print(f"  R²:   {r_squared(y_true, y_pred):.4f}")

    # Linear regression
    print("\n── LINEAR REGRESSION FROM SCRATCH ──")
    np.random.seed(42)
    X = np.column_stack([np.ones(100), np.random.randn(100)])
    y = 3 * X[:, 1] + 1.5 + np.random.randn(100) * 0.5

    model = LinearRegressionScratch(lr=0.1, epochs=200)
    model.fit(X, y)
    print(f"  R² score: {model.score(X, y):.4f}")
    print(f"  Loss (initial→final): {model.loss_history_[0]:.3f} → {model.loss_history_[-1]:.4f}")

    # Logistic regression
    print("\n── LOGISTIC REGRESSION FROM SCRATCH ──")
    from sklearn.datasets import make_classification
    X_cls, y_cls = make_classification(n_samples=200, n_features=4, random_state=42)

    clf = LogisticRegressionScratch(lr=0.1, epochs=500)
    clf.fit(X_cls, y_cls)
    print(f"  Accuracy: {clf.score(X_cls, y_cls):.3f}")

    # Overfitting
    demonstrate_overfitting()

    print("\n✓ ML Mathematics demo complete!")
    print("\nKey concepts:")
    print("  • Loss function = how wrong the model is")
    print("  • Gradient descent = iteratively reduce loss")
    print("  • Learning rate = step size (too big → diverge, too small → slow)")
    print("  • Overfitting = great on train, poor on test")
    print("  • Regularization = penalize complexity to prevent overfitting")


if __name__ == "__main__":
    main()
