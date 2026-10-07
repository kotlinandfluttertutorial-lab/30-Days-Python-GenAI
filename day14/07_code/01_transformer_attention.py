"""
Day 14 — Transformer Attention Implementation
===============================================
Implements scaled dot-product attention, multi-head attention,
positional encoding, and a full transformer block from scratch.

Run: python 01_transformer_attention.py
Requires: pip install torch numpy
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# SCALED DOT-PRODUCT ATTENTION
# ─────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


# ─────────────────────────────────────────────────────────
# MULTI-HEAD ATTENTION
# ─────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_weights: torch.Tensor | None = None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_output, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_o(attn_output)


# ─────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────
# MINI TRANSFORMER ENCODER
# ─────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Stack of transformer blocks for classification (BERT-style).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(token_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        # Use [CLS] token (position 0) for classification
        return self.classifier(x[:, 0, :])


# ─────────────────────────────────────────────────────────
# DEMONSTRATIONS
# ─────────────────────────────────────────────────────────

def demo_attention_mechanism() -> None:
    print("\n── SCALED DOT-PRODUCT ATTENTION ──")
    torch.manual_seed(42)

    batch, seq_len, d_k = 2, 6, 64
    Q = torch.randn(batch, 1, seq_len, d_k)  # 1 head
    K = torch.randn(batch, 1, seq_len, d_k)
    V = torch.randn(batch, 1, seq_len, d_k)

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  Q, K, V shape: {Q.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    print(f"  Weights sum to 1: {weights.sum(dim=-1).mean().item():.4f}")
    print(f"\n  Attention weight matrix (first batch, first query):")
    print(f"  {weights[0, 0, 0].detach().numpy().round(3)}")


def demo_multi_head_attention() -> None:
    print("\n── MULTI-HEAD ATTENTION ──")
    torch.manual_seed(42)

    batch, seq_len, d_model = 2, 8, 256
    n_heads = 8

    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch, seq_len, d_model)

    output = mha(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}  (same as input — that's the point)")
    print(f"  n_heads={n_heads}, d_k per head={d_model // n_heads}")
    print(f"  Total attention params: {sum(p.numel() for p in mha.parameters()):,}")


def demo_positional_encoding() -> None:
    print("\n── POSITIONAL ENCODING ──")
    torch.manual_seed(42)

    d_model, seq_len = 64, 20
    pe = PositionalEncoding(d_model, max_seq_len=100)

    dummy = torch.zeros(1, seq_len, d_model)
    encoded = pe(dummy)

    # Show that adjacent positions have similar encodings
    pos_enc = encoded[0].detach()
    sim_adj = F.cosine_similarity(pos_enc[0].unsqueeze(0), pos_enc[1].unsqueeze(0)).item()
    sim_far = F.cosine_similarity(pos_enc[0].unsqueeze(0), pos_enc[19].unsqueeze(0)).item()
    print(f"  Adjacent position similarity (0,1): {sim_adj:.4f}  ← should be high")
    print(f"  Distant position similarity (0,19): {sim_far:.4f}  ← should be lower")


def demo_causal_mask() -> None:
    print("\n── CAUSAL MASK (DECODER) ──")
    seq_len = 5
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"  Causal mask ({seq_len}×{seq_len}):")
    print(f"  (1=can attend, 0=masked as -inf)")
    for row in mask:
        print(f"    {[int(v.item()) for v in row]}")
    print(f"\n  Token 0 can see: only itself")
    print(f"  Token 3 can see: tokens 0,1,2,3")
    print(f"  Token 4 can see: all tokens")


def demo_full_transformer() -> None:
    print("\n── FULL TRANSFORMER ENCODER ──")
    torch.manual_seed(42)

    model = TransformerEncoder(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        n_classes=3,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: 2 layers, d_model=128, 4 heads")
    print(f"  Total parameters: {total_params:,}")

    # Forward pass
    batch_size, seq_len = 4, 32
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(token_ids)
    probs = F.softmax(logits, dim=-1)

    print(f"  Input: token_ids {token_ids.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Predicted class probabilities (first item): {probs[0].detach().numpy().round(3)}")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 14 — TRANSFORMER ATTENTION IMPLEMENTATION     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_attention_mechanism()
    demo_multi_head_attention()
    demo_positional_encoding()
    demo_causal_mask()
    demo_full_transformer()

    print("\n✓ Day 14 Transformer demo complete!")
    print("\nKey Transformer insights:")
    print("  • Self-attention: Q × K^T / √d_k → weights → weighted sum of V")
    print("  • Multiple heads: each learns different relationship types")
    print("  • Residual connections: gradient highway through deep networks")
    print("  • LayerNorm: stabilizes training (works with any batch size)")
    print("  • Causal mask: decoder-only models can't see future tokens")
    print("  • GELU activation: smoother than ReLU, standard in transformers")


if __name__ == "__main__":
    main()
