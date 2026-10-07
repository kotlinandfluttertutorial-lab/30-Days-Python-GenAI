"""
Day 10 — Neural Network From Scratch
=======================================
Implements a complete neural network with backpropagation.
No deep learning frameworks — pure NumPy.

Run: python 01_neural_network_scratch.py
Requires: pip install numpy scikit-learn
"""

import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - z.max(axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────
# NEURAL NETWORK
# ─────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully-connected neural network from scratch.
    Architecture: input → [hidden layers] → output
    """

    def __init__(
        self,
        layer_sizes: list[int],    # e.g., [4, 16, 8, 1]
        learning_rate: float = 0.01,
        hidden_activation: str = "relu",
    ) -> None:
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.n_layers = len(layer_sizes) - 1
        self.hidden_activation = hidden_activation
        self.loss_history: list[float] = []

        # Initialize weights and biases (He initialization)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in)  # He initialization
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))

    def _activate(self, z: np.ndarray, layer_idx: int) -> np.ndarray:
        """Apply activation function for a given layer."""
        is_output = (layer_idx == self.n_layers - 1)
        if is_output:
            if self.layer_sizes[-1] == 1:
                return sigmoid(z)   # Binary classification output
            return softmax(z)       # Multi-class output
        if self.hidden_activation == "relu":
            return relu(z)
        return sigmoid(z)

    def _activate_grad(self, z: np.ndarray, layer_idx: int) -> np.ndarray:
        """Gradient of activation function."""
        is_output = (layer_idx == self.n_layers - 1)
        if is_output:
            return np.ones_like(z)  # Gradient absorbed into loss gradient
        if self.hidden_activation == "relu":
            return relu_grad(z)
        return sigmoid_grad(z)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. Saves intermediate values for backprop."""
        self.zs: list[np.ndarray] = []      # Pre-activation
        self.activations: list[np.ndarray] = [X]  # Post-activation (layer 0 = input)

        current = X
        for i in range(self.n_layers):
            z = current @ self.weights[i] + self.biases[i]
            a = self._activate(z, i)
            self.zs.append(z)
            self.activations.append(a)
            current = a

        return current  # Final output

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Backpropagation — compute gradients and update weights."""
        n = len(X)
        output = self.activations[-1]

        # Output layer gradient (binary cross-entropy loss)
        y_reshaped = y.reshape(-1, 1) if output.ndim > 1 else y
        delta = output - y_reshaped  # Combined loss + sigmoid gradient

        for i in range(self.n_layers - 1, -1, -1):
            a_prev = self.activations[i]  # Activation from previous layer

            # Gradients
            dW = (1 / n) * a_prev.T @ delta
            db = (1 / n) * delta.sum(axis=0)

            # Gradient for previous layer
            if i > 0:
                delta_prev = delta @ self.weights[i].T
                delta = delta_prev * self._activate_grad(self.zs[i - 1], i - 1)

            # Update weights
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    def compute_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-15
        output = np.clip(output, eps, 1 - eps)
        y_r = y.reshape(-1, 1) if output.ndim > 1 else y
        return float(-np.mean(y_r * np.log(output) + (1 - y_r) * np.log(1 - output)))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        verbose: bool = True,
    ) -> "NeuralNetwork":
        self.loss_history = []

        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(output, y)
            self.loss_history.append(loss)
            self.backward(X, y)

            if verbose and epoch % 100 == 0:
                preds = (output.ravel() >= 0.5).astype(int)
                acc = float(np.mean(preds == y))
                print(f"  Epoch {epoch:4d}: loss={loss:.4f}, acc={acc:.3f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X).ravel()
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ─────────────────────────────────────────────────────────
# DEMONSTRATIONS
# ─────────────────────────────────────────────────────────

def demo_activations() -> None:
    print("\n── ACTIVATION FUNCTIONS ──")
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(f"z values:  {z}")
    print(f"sigmoid:   {sigmoid(z).round(3)}")
    print(f"relu:      {relu(z)}")
    print(f"tanh:      {np.tanh(z).round(3)}")

    # Show vanishing gradient issue with sigmoid
    deep_z = np.array([-10.0, 10.0])
    print(f"\nVanishing gradient (sigmoid at extreme z):")
    print(f"  sigmoid(-10)={sigmoid(deep_z[0]):.8f}  gradient={sigmoid_grad(deep_z[0]):.8f} ← nearly 0!")
    print(f"  relu(-10)={relu(deep_z[0])}          gradient={relu_grad(deep_z[0])}     ← 0 for negative")
    print(f"  relu(10)={relu(deep_z[1])}           gradient={relu_grad(deep_z[1])}     ← always 1 for positive!")


def demo_neural_network() -> None:
    print("\n── NEURAL NETWORK FROM SCRATCH ──")

    # Non-linearly separable data (XOR-like)
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Architecture: 2 → 16 → 8 → 1
    nn = NeuralNetwork(
        layer_sizes=[2, 16, 8, 1],
        learning_rate=0.05,
        hidden_activation="relu",
    )

    print(f"Architecture: {nn.layer_sizes}")
    print(f"Parameters: {sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))}")

    print("\nTraining:")
    nn.fit(X_train, y_train, epochs=500, verbose=True)

    train_acc = nn.score(X_train, y_train)
    test_acc = nn.score(X_test, y_test)
    print(f"\nFinal: Train={train_acc:.3f}, Test={test_acc:.3f}")
    print(f"Loss trajectory: {nn.loss_history[0]:.3f} → {nn.loss_history[-1]:.4f}")


def demo_weight_initialization() -> None:
    print("\n── WEIGHT INITIALIZATION IMPORTANCE ──")

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    configs = [
        ("He init (good)",  NeuralNetwork([10, 32, 16, 1], lr=0.05)),
        ("Zero init (bad)", None),  # Special case
    ]

    for name, nn in configs:
        if nn is None:
            nn = NeuralNetwork([10, 32, 16, 1], lr=0.05)
            for i in range(nn.n_layers):
                nn.weights[i] = np.zeros_like(nn.weights[i])

        print(f"\n  {name}:")
        nn.fit(X_train, y_train, epochs=200, verbose=False)
        acc = nn.score(X_test, y_test)
        final_loss = nn.loss_history[-1]
        print(f"  Test accuracy: {acc:.3f}, Final loss: {final_loss:.4f}")
        print(f"  → {'Good' if acc > 0.7 else 'Failed to learn!'}")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 10 — NEURAL NETWORK FROM SCRATCH              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_activations()
    demo_neural_network()
    demo_weight_initialization()

    print("\n✓ Day 10 Neural Network demo complete!")
    print("\nKey concepts:")
    print("  • Activation functions add non-linearity (without = just linear regression)")
    print("  • ReLU: default for hidden layers (fast, no vanishing gradient for positive)")
    print("  • He init: weights = randn × √(2/fan_in) — critical for deep networks")
    print("  • Backprop: chain rule backwards through every layer")
    print("  • More layers = can learn more complex patterns (but harder to train)")


if __name__ == "__main__":
    np.random.seed(42)
    main()
