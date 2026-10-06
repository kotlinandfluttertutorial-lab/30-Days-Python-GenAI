# Day 10 — Concepts
## Neural Networks

---

## 1. The Neuron

```
Inputs: x₁, x₂, x₃
Weights: w₁, w₂, w₃
Bias: b

z = w₁x₁ + w₂x₂ + w₃x₃ + b    ← linear combination
a = activation(z)                ← non-linear transformation

Without activation functions, stacking layers is equivalent to one linear layer.
Non-linearity = the ability to learn complex patterns.
```

---

## 2. Activation Functions

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Output: 0 to 1. Vanishing gradient problem for deep networks.
# Use: output layer for binary classification

def tanh(z):
    return np.tanh(z)
# Output: -1 to 1. Better than sigmoid (zero-centered).
# Vanishing gradient still occurs.

def relu(z):
    return np.maximum(0, z)
# Output: 0 to ∞. Fast, no vanishing gradient for positive inputs.
# Dying ReLU problem: neurons can "die" (output 0 always).
# Use: hidden layers in most networks

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
# Fixes dying ReLU: small gradient for negative inputs

def softmax(z):
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()
# Output: probability distribution summing to 1
# Use: output layer for multi-class classification

# WHEN TO USE:
# Hidden layers: ReLU (default), LeakyReLU (if dying ReLU)
# Output - regression: none (linear)
# Output - binary: sigmoid
# Output - multi-class: softmax
```

---

## 3. Forward Propagation

```python
class NeuralNetworkLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        # He initialization: better for ReLU
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)
        self.activation_fn = activation

    def forward(self, X):
        self.input = X
        self.z = X @ self.W + self.b      # linear: (batch, in) × (in, out) = (batch, out)
        self.output = self.activate(self.z)
        return self.output

    def activate(self, z):
        if self.activation_fn == 'relu':
            return np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_fn == 'softmax':
            exp_z = np.exp(z - z.max(axis=1, keepdims=True))
            return exp_z / exp_z.sum(axis=1, keepdims=True)
        return z  # linear
```

---

## 4. Backpropagation

```
The chain rule applied recursively through the network.

LOSS → OUTPUT LAYER → HIDDEN LAYERS → INPUT

For each layer, compute:
  dL/dW = input.T × dL/dz    (gradient w.r.t. weights)
  dL/db = sum(dL/dz, axis=0) (gradient w.r.t. biases)
  dL/dX = dL/dz × W.T        (gradient w.r.t. input, for next layer)

Then update:
  W -= learning_rate × dL/dW
  b -= learning_rate × dL/db

The key insight: gradient flows backward through the same weights
used in the forward pass. This is why weight initialization matters
(He init ensures gradients don't vanish or explode).
```

---

## 5. Neural Network From Scratch

```python
class TwoLayerNN:
    """Binary classifier with one hidden layer."""
    
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        self.lr = learning_rate
        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)          # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))     # Sigmoid output
        return self.a2

    def backward(self, X, y):
        n = len(X)
        # Output layer gradient
        dz2 = self.a2 - y.reshape(-1, 1)          # BCE gradient
        dW2 = (1/n) * self.a1.T @ dz2
        db2 = (1/n) * dz2.sum(axis=0)
        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float)   # ReLU gradient
        dW1 = (1/n) * X.T @ dz1
        db1 = (1/n) * dz1.sum(axis=0)
        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-15) + (1-y) * np.log(1-output + 1e-15))
                print(f"Epoch {epoch}: Loss={loss:.4f}")

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int).ravel()
```
