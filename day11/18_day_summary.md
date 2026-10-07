# Day 11 — Day Summary: PyTorch

## What You Built
A complete PyTorch classification system: custom Dataset, DataLoader, nn.Module with BatchNorm + Dropout, full training loop with early stopping, checkpointing, and comparison with sklearn.

## Key Takeaways
1. PyTorch autograd handles backpropagation automatically — you never compute gradients manually
2. The 5-step training pattern: zero_grad → forward → loss → backward → step
3. `model.train()` / `model.eval()` switches matter for Dropout and BatchNorm
4. AdamW with weight decay is the modern standard optimizer
5. Always checkpoint the best model (by val loss), not the last epoch

## Tomorrow: Day 12 — Deep Learning Engineering
CNN for images, RNN/LSTM for sequences, attention basics, transfer learning — the architectures that power modern AI.
