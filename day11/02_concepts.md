# Day 11 — Concepts
## PyTorch

---

## 1. Tensors

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Tensors: PyTorch's ndarray
x = torch.tensor([1.0, 2.0, 3.0])
m = torch.zeros(3, 4)                  # 3×4 matrix
r = torch.randn(100, 768)              # 100 random 768-dim vectors

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

# From/to NumPy
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(np_arr)
back_to_np = t.numpy()

# Gradients: this is PyTorch's superpower
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x                    # y = x² + 3x
y.backward()                           # Compute dy/dx automatically!
print(x.grad)                          # dy/dx = 2x + 3 = 7 at x=2
```

---

## 2. nn.Module — Building Models

```python
class ClassificationNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        # Define layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),              # Regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Create model
model = ClassificationNet(input_dim=10, hidden_dim=64, n_classes=2)
print(model)  # Shows architecture
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 3. Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import torch

class TabularDataset(Dataset):
    """Custom dataset for tabular data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)  # Long for classification labels

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# Wrap in DataLoader for batching, shuffling, parallel loading
train_dataset = TabularDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,    # Shuffle every epoch
    num_workers=0,   # Parallel loading workers (0 = main thread)
)
```

---

## 4. Training Loop

```python
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """One epoch of training. Returns (loss, accuracy)."""
    model.train()  # Enables dropout, batch norm train mode
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()           # Clear gradients
        logits = model(X_batch)         # Forward pass
        loss = criterion(logits, y_batch)  # Compute loss
        loss.backward()                 # Compute gradients
        optimizer.step()                # Update weights

        total_loss += loss.item() * len(X_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on val/test set."""
    model.eval()   # Disables dropout, switches batch norm to eval mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Don't compute gradients (saves memory + speed)
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(X_batch)

    return total_loss / total, correct / total
```

---

## 5. Checkpointing

```python
# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)

# Load checkpoint
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["val_loss"]

# Best practice: save best model (by val loss)
best_val_loss = float("inf")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, "best_model.pt")
```

---

## 6. Key Optimizers

```python
# Adam — almost always the right choice
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# AdamW — Adam with weight decay (better for transformers/LLMs)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# SGD with momentum — good for CV tasks
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)
# Call each epoch: scheduler.step(val_loss)
# Reduces LR when val loss stops improving
```
