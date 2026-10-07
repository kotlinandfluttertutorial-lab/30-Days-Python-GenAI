# Day 12 — Concepts
## Deep Learning Engineering: CNN, RNN, LSTM, Attention

---

## 1. Convolutional Neural Networks (CNN)

```python
import torch.nn as nn

class TextCNN(nn.Module):
    """
    CNN for text classification.
    Uses 1D convolutions to detect local n-gram patterns.
    """
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, n_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes), n_classes)

    def forward(self, x):
        # x: (batch, seq_len) → embedding → (batch, seq_len, embed_dim)
        emb = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        # Apply convolutions with different filter sizes (unigram, bigram, trigram)
        pooled = [conv(emb).relu().max(dim=2)[0] for conv in self.convs]
        out = torch.cat(pooled, dim=1)  # (batch, n_filters * len(filter_sizes))
        return self.fc(self.dropout(out))
```

---

## 2. RNN and LSTM

```python
class LSTMClassifier(nn.Module):
    """
    LSTM for sequence classification.
    Uses bidirectional LSTM to read sequence both ways.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,        # (batch, seq, features) format
            bidirectional=True,      # Read both left-to-right AND right-to-left
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)  # × 2 for bidirectional

    def forward(self, x):
        emb = self.dropout(self.embedding(x))       # (batch, seq, embed_dim)
        output, (hidden, _) = self.lstm(emb)
        # Use last hidden state from both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(last_hidden))

# Why LSTM over vanilla RNN?
# RNN: gradient vanishing over long sequences (>20 tokens)
# LSTM: gates (input, forget, output) control information flow
#       Can "remember" information from 100+ steps ago
```

---

## 3. Transfer Learning

```python
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class BertClassifier(nn.Module):
    """
    Transfer learning: pretrained BERT + custom classification head.
    BERT has learned language from billions of text documents.
    We just add a classification head and fine-tune.
    """
    def __init__(self, model_name: str, n_classes: int, freeze_bert: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for BERT-base
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        # BERT outputs: last_hidden_state (batch, seq, 768) + pooler_output (batch, 768)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# Fine-tuning strategy:
# 1. Freeze BERT, train classifier only (few epochs)
# 2. Unfreeze BERT, train everything with tiny LR (1e-5 to 5e-5)
# This avoids catastrophic forgetting
```

---

## 4. Batch Normalization and Dropout

```python
# Batch Normalization: normalize activations within a batch
# Effect: stabilizes training, allows higher learning rates
# Position: after linear, before activation
nn.Sequential(
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),  # Normalize
    nn.ReLU(),
    nn.Dropout(0.3),      # Regularize
)

# Layer Normalization: normalize across features (not batch)
# Used in Transformers and when batch size = 1
nn.LayerNorm(hidden_size)

# Dropout:
# Training: randomly zero out neurons (p=probability of zeroing)
# Inference: model.eval() disables dropout
# Effect: prevents co-adaptation, acts as ensemble

# When to use:
# Dropout: after any layer with large parameter count
# BatchNorm: after each dense/conv layer in feedforward networks
# LayerNorm: in transformers (allows batch_size=1 inference)
```

---

## 5. Early Stopping + Learning Rate Scheduling

```python
# Early stopping: stop training when val loss doesn't improve
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# Learning rate warmup (standard for transformer fine-tuning):
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,      # Slowly increase LR for first 100 steps
    num_training_steps=total_steps,  # Then linearly decay to 0
)
```
