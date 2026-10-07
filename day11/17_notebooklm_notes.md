# Day 11 — NotebookLM Notes: PyTorch

## Core PyTorch Pattern (memorize this loop)

```python
# The canonical PyTorch training loop:
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()      # 1. Clear old gradients
        logits = model(X_batch)    # 2. Forward pass
        loss = criterion(logits, y_batch)  # 3. Compute loss
        loss.backward()            # 4. Backprop (autograd)
        optimizer.step()           # 5. Update weights

    model.eval()
    with torch.no_grad():          # No grad for evaluation
        for X_val, y_val in val_loader:
            logits = model(X_val)
            ...
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `nn.Module` | Base class for all models |
| `nn.Sequential` | Chain layers in order |
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()` | Activation |
| `nn.Dropout(p)` | Regularization |
| `nn.BatchNorm1d(dim)` | Normalize within batch |
| `Dataset` | Wraps data, defines `__len__` and `__getitem__` |
| `DataLoader` | Batches + shuffles a Dataset |

## Interview Facts

1. `model.train()` enables Dropout + BN train mode; `model.eval()` disables
2. `with torch.no_grad()`: skips gradient computation → faster, less memory
3. `optimizer.zero_grad()` MUST be called before each `loss.backward()` (gradients accumulate)
4. Adam: adaptive learning rates per parameter; default lr=1e-3
5. AdamW: Adam + weight decay; preferred for LLM fine-tuning
6. Gradient clipping (`clip_grad_norm_`): prevents exploding gradients in RNNs/LSTMs
7. `model.state_dict()` saves weights; `load_state_dict()` restores them

## Common Mistakes

- Forgetting `optimizer.zero_grad()` → gradients accumulate across batches
- Forgetting `model.eval()` during inference → Dropout active → different outputs each call
- Forgetting `torch.no_grad()` during eval → wastes memory computing unused gradients
- Wrong loss function: use `CrossEntropyLoss` for multi-class (takes logits, not softmax output)
- Putting data on wrong device: model on GPU but data on CPU → error
