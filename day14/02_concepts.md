# Day 14 — Concepts
## Transformers: The Architecture That Changed Everything

---

## 1. The Problem Transformers Solved

```
RNN/LSTM limitations:
1. Sequential processing: can't parallelize (slow training on modern hardware)
2. Vanishing gradient: context from 100+ steps back gets diluted
3. Fixed context: information compression through hidden state

Transformer solution:
1. Parallel: all tokens processed simultaneously
2. Direct attention: every token can directly attend to every other token
3. Full context: attention over the entire sequence at once
```

---

## 2. Self-Attention — The Core Mechanism

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,   # Queries: (batch, n_heads, seq_len, d_k)
    K: torch.Tensor,   # Keys:    (batch, n_heads, seq_len, d_k)
    V: torch.Tensor,   # Values:  (batch, n_heads, seq_len, d_v)
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Intuition:
    - Q (query): "what information am I looking for?"
    - K (key):   "what information do I have?"
    - V (value): "the actual information"

    Process:
    1. Compute similarity: Q × K^T → attention scores
    2. Scale by √d_k (prevents extremely large values → vanishing softmax gradient)
    3. Optionally mask (for decoder: prevent attending to future tokens)
    4. Softmax → attention weights (sum to 1 per query)
    5. Weighted sum of values → output
    """
    d_k = Q.size(-1)

    # Step 1 + 2: Scaled dot-product
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, n_heads, seq_len, seq_len)

    # Step 3: Optional mask (causal mask for decoder)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: Softmax
    weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = torch.matmul(weights, V)
    # output shape: (batch, n_heads, seq_len, d_v)

    return output, weights
```

---

## 3. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Run self-attention h times in parallel with different projections.

    Why multiple heads?
    - Each head can attend to different types of relationships
    - Head 1 might focus on syntactic dependencies
    - Head 2 might focus on semantic similarity
    - Head 3 might focus on positional patterns
    - Concat and project → richer representations
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Each head has smaller dimension

        # Projection layers (learned)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        # Project to Q, K, V (each head gets its own projection)
        Q = self.split_heads(self.W_q(x))  # (batch, n_heads, seq, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Attention
        attn_output, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch, n_heads, seq, d_k)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_o(attn_output)
```

---

## 4. Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    Add position information to embeddings.
    Without this, the Transformer is permutation-invariant
    (doesn't know token order).

    Sine/cosine encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
    - Unique encoding for each position
    - Nearby positions have similar encodings
    - Can generalize to unseen sequence lengths
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dims: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dims: cosine
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_seq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
```

---

## 5. Feed-Forward Network + Transformer Block

```python
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Applied independently to each token's representation.
    Typically 4× wider than d_model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),           # Gaussian Error Linear Unit — smoother than ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer block:
    1. Multi-head self-attention
    2. Add & Norm (residual connection)
    3. Feed-forward network
    4. Add & Norm

    Residual connections: x → x + sublayer(x)
    Critical for training deep networks (gradient highway)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)  # Pre-norm (modern variant)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x
```

---

## 6. Encoder vs Decoder

```
ENCODER (BERT-style):
- Bidirectional attention (every token sees all others)
- Good for: understanding, classification, embeddings
- Example: "The cat sat" → each word attends to all words
- Used for: sentence embeddings, NLI, named entity recognition

DECODER (GPT-style):
- Causal attention (token only sees previous tokens)
- Good for: generation (predict next token)
- Example: "The cat" → can attend to "The" and "cat", but not future tokens
- Used for: text generation, code completion, chat

ENCODER-DECODER (T5, BART):
- Encoder processes input, decoder generates output
- Used for: translation, summarization, Q&A

Modern LLMs (GPT-4, Claude, Llama) = Decoder-only
BERT and sentence embedding models = Encoder-only
```

---

## 7. Causal Mask (for Decoders)

```python
def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Upper triangular mask: token i can only attend to tokens j <= i.
    This prevents the model from "seeing the future" during generation.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims

# seq_len=4 mask:
# [[1, 0, 0, 0],   ← token 0 only sees itself
#  [1, 1, 0, 0],   ← token 1 sees 0,1
#  [1, 1, 1, 0],   ← token 2 sees 0,1,2
#  [1, 1, 1, 1]]   ← token 3 sees all
```
