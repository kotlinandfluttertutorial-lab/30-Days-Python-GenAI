# Day 10 — NotebookLM Notes: Neural Networks

## Core Components

**Neuron** — Takes inputs, multiplies by weights, adds bias, applies activation: `a = activation(Wx + b)`

**Activation Functions:**
- ReLU: `max(0, z)` — default hidden layer; no vanishing gradient (positive)
- Sigmoid: `1/(1+e^-z)` — output for binary; vanishing gradient for deep
- Softmax: `exp(z)/Σexp(z)` — output for multi-class; sums to 1
- Tanh: `tanh(z)` — zero-centered; better than sigmoid for hidden

**Weight Initialization:**
- Zero init: ALL neurons compute the same thing → symmetry problem → fails
- He init: `randn × √(2/fan_in)` — designed for ReLU, prevents vanishing/exploding

**Backpropagation** — Chain rule through the network. Gradient flows backward through the same weights. Each layer computes dL/dW, dL/db, then updates.

**Gradient Descent Update:** `W -= lr × dL/dW`

## Architecture Design

```
INPUT → HIDDEN1 → HIDDEN2 → OUTPUT
ReLU     ReLU      Sigmoid/Softmax

For binary classification: output = 1 neuron, sigmoid, BCE loss
For multi-class: output = n neurons, softmax, CE loss
For regression: output = 1 neuron, no activation, MSE loss
```

## Interview Facts

1. Without activation functions, stacking layers = one linear transformation
2. ReLU dying: neuron outputs 0 always → fix with Leaky ReLU
3. Vanishing gradient: sigmoid/tanh derivatives < 0.25 → gradients shrink exponentially
4. He initialization: designed for ReLU; Glorot/Xavier for sigmoid/tanh
5. More parameters → more expressive → more likely to overfit → need regularization
6. Batch size: small = noisy gradients (good regularization); large = stable but may overfit

## Common Mistakes

- Not normalizing inputs → slow or failed training
- All-zero weight initialization → symmetry, all neurons learn same thing
- Too high learning rate → loss oscillates or diverges
- Too small network for complex data → underfitting
- Too large network for small dataset → overfitting (add dropout)
