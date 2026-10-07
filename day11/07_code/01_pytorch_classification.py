"""
Day 11 — PyTorch Classification Model
========================================
Complete PyTorch training pipeline:
- Custom Dataset + DataLoader
- nn.Module with proper architecture
- Training loop with validation
- Checkpointing
- Comparison with sklearn baseline

Run: python 01_pytorch_classification.py
Requires: pip install torch numpy scikit-learn
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────

class ClassifierNet(nn.Module):
    """Feedforward neural network for tabular classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


# ─────────────────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────────────────

def train_model(
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "best_model.pt",
) -> nn.Module:

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    n_classes = len(np.unique(y_train))
    model = ClassifierNet(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        n_classes=n_classes,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    print("─" * 55)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"{epoch + 1:>6} {train_loss:>11.4f} {train_acc:>10.3f} "
                  f"{val_loss:>10.4f} {val_acc:>9.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nBest val loss: {best_val_loss:.4f} (epoch {checkpoint['epoch'] + 1})")
    return model


@torch.no_grad()
def get_predictions(model: nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    logits = model(X_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    return preds, probs


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 11 — PYTORCH CLASSIFICATION MODEL             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Create dataset
    X, y = make_classification(
        n_samples=2000, n_features=15, n_informative=10,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train
    model = train_model(
        input_dim=15,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        epochs=50, batch_size=32,
    )

    # Final evaluation
    preds, probs = get_predictions(model, X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print(f"\nTest Accuracy: {acc:.3f}")
    print(f"Test F1:       {f1:.3f}")

    # Compare with sklearn
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"\nComparison:")
    print(f"  PyTorch NN:    {acc:.3f}")
    print(f"  Random Forest: {rf_acc:.3f}")
    print(f"  → {'Neural Net wins' if acc > rf_acc else 'Random Forest wins'}")

    # Cleanup
    if Path("best_model.pt").exists():
        os.remove("best_model.pt")

    print("\n✓ Day 11 PyTorch demo complete!")
    print("\nKey PyTorch patterns:")
    print("  • Dataset + DataLoader: structured data batching")
    print("  • model.train() / model.eval(): affects Dropout, BatchNorm")
    print("  • optimizer.zero_grad() → loss.backward() → optimizer.step()")
    print("  • torch.no_grad(): evaluation mode, no gradient computation")
    print("  • Early stopping: stop when val loss stops improving")
    print("  • Checkpointing: save best model, not last epoch")


if __name__ == "__main__":
    main()
