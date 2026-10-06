"""
Day 04 — AI Mathematics Implementation
=======================================
Implements all core AI math operations from scratch,
then shows the NumPy equivalents.

Run: python 01_math_fundamentals.py
"""

import math
import numpy as np
from typing import Any


# ═══════════════════════════════════════════════════════════════
# FROM SCRATCH (no NumPy) — understand the math
# ═══════════════════════════════════════════════════════════════

def dot_product(a: list[float], b: list[float]) -> float:
    """Dot product: sum of element-wise products."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def magnitude(v: list[float]) -> float:
    """L2 norm (Euclidean length) of a vector."""
    return math.sqrt(sum(x**2 for x in v))


def normalize(v: list[float]) -> list[float]:
    """Scale vector to unit length."""
    mag = magnitude(v)
    if mag == 0:
        return [0.0] * len(v)
    return [x / mag for x in v]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.
    = dot(a, b) / (|a| × |b|)
    Range: -1 (opposite) to 1 (identical direction)
    """
    dot = dot_product(a, b)
    mag_a, mag_b = magnitude(a), magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """L2 distance between two points."""
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """L1 distance (sum of absolute differences)."""
    return sum(abs(x - y) for x, y in zip(a, b))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    m = mean(values)
    return sum((x - m)**2 for x in values) / len(values)


def std_dev(values: list[float]) -> float:
    return math.sqrt(variance(values))


def z_score(value: float, values: list[float]) -> float:
    m, s = mean(values), std_dev(values)
    return (value - m) / s if s > 0 else 0.0


def softmax(logits: list[float]) -> list[float]:
    """Convert logits to probabilities. Used in classification heads."""
    max_logit = max(logits)  # Subtract for numerical stability
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ═══════════════════════════════════════════════════════════════
# SIMILARITY CALCULATOR (MINI PROJECT)
# ═══════════════════════════════════════════════════════════════

class SimilarityCalculator:
    """
    Demonstrates embedding similarity with toy vectors.
    In production, embeddings come from OpenAI/HuggingFace.
    Here we use hand-crafted vectors to show the math clearly.
    """

    # Toy 6D embeddings (dimensions: tech, biology, geography, food, sports, abstract)
    VOCABULARY: dict[str, list[float]] = {
        "Python":        [0.9, 0.0, 0.0, 0.0, 0.0, 0.3],
        "JavaScript":    [0.8, 0.0, 0.0, 0.0, 0.0, 0.2],
        "Transformer":   [0.8, 0.0, 0.0, 0.0, 0.0, 0.7],
        "Paris":         [0.0, 0.0, 0.9, 0.0, 0.0, 0.2],
        "France":        [0.0, 0.0, 0.9, 0.1, 0.0, 0.1],
        "Berlin":        [0.0, 0.0, 0.8, 0.0, 0.0, 0.2],
        "Dog":           [0.0, 0.9, 0.1, 0.0, 0.0, 0.0],
        "Cat":           [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
        "Pizza":         [0.0, 0.0, 0.1, 0.9, 0.0, 0.0],
        "Sushi":         [0.0, 0.0, 0.2, 0.8, 0.0, 0.0],
        "Football":      [0.0, 0.1, 0.0, 0.0, 0.9, 0.0],
        "Basketball":    [0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
        "RAG":           [0.9, 0.0, 0.0, 0.0, 0.0, 0.8],
        "Embedding":     [0.8, 0.0, 0.0, 0.0, 0.0, 0.9],
    }

    def get_embedding(self, word: str) -> list[float]:
        if word not in self.VOCABULARY:
            raise KeyError(f"'{word}' not in vocabulary: {list(self.VOCABULARY.keys())}")
        return self.VOCABULARY[word]

    def compute_similarity(self, word1: str, word2: str) -> dict[str, Any]:
        v1 = self.get_embedding(word1)
        v2 = self.get_embedding(word2)

        cos_sim = cosine_similarity(v1, v2)
        euc_dist = euclidean_distance(v1, v2)
        dot = dot_product(v1, v2)
        mag1, mag2 = magnitude(v1), magnitude(v2)

        return {
            "word1": word1,
            "word2": word2,
            "cosine_similarity": round(cos_sim, 4),
            "euclidean_distance": round(euc_dist, 4),
            "dot_product": round(dot, 4),
            "magnitude_1": round(mag1, 4),
            "magnitude_2": round(mag2, 4),
            "interpretation": (
                "Very similar" if cos_sim > 0.9 else
                "Similar" if cos_sim > 0.7 else
                "Somewhat related" if cos_sim > 0.4 else
                "Unrelated"
            ),
        }

    def find_most_similar(self, word: str, top_k: int = 3) -> list[tuple[str, float]]:
        target = self.get_embedding(word)
        scores = []
        for w, vec in self.VOCABULARY.items():
            if w != word:
                sim = cosine_similarity(target, vec)
                scores.append((w, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════

def demo_vector_math() -> None:
    print("\n── VECTOR MATH ──")

    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"dot(a, b)           = {dot_product(a, b):.4f}  (= 1×4 + 2×5 + 3×6 = 32)")
    print(f"|a| (magnitude)     = {magnitude(a):.4f}  (= √(1²+2²+3²))")
    print(f"cosine_similarity   = {cosine_similarity(a, b):.4f}  (very similar direction)")
    print(f"euclidean_distance  = {euclidean_distance(a, b):.4f}  (physical distance)")

    # Compare with NumPy
    a_np, b_np = np.array(a), np.array(b)
    np_cos = float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    print(f"\nNumPy cosine (verify): {np_cos:.4f}  ← same result, faster computation")


def demo_softmax() -> None:
    print("\n── SOFTMAX (LLM TOKEN PROBABILITIES) ──")

    # Raw model outputs (logits) for next token prediction
    logits = [2.5, 0.1, -1.2, 1.8, 0.7]
    probs = softmax(logits)

    print("Logits → Probabilities:")
    for i, (logit, prob) in enumerate(zip(logits, probs)):
        bar = "█" * int(prob * 30)
        print(f"  token_{i}: {logit:5.1f} → {prob:.4f} {bar}")
    print(f"  Sum: {sum(probs):.6f}  (always 1.0)")


def demo_statistics() -> None:
    print("\n── STATISTICS FOR AI EVALUATION ──")

    # RAG retrieval scores
    scores = [0.92, 0.78, 0.85, 0.91, 0.70, 0.88, 0.95, 0.62, 0.83, 0.79]
    m = mean(scores)
    s = std_dev(scores)

    print(f"Scores: {[round(s, 2) for s in scores]}")
    print(f"Mean:   {m:.3f}")
    print(f"Std:    {s:.3f}")
    print(f"\nZ-scores (how many std devs from mean):")
    for score in [0.95, 0.85, 0.62]:
        z = z_score(score, scores)
        interpretation = "above avg" if z > 0 else "below avg"
        print(f"  score={score:.2f} → z={z:.2f} ({interpretation})")


def demo_similarity_calculator() -> None:
    print("\n── SIMILARITY CALCULATOR ──")

    calc = SimilarityCalculator()

    comparisons = [
        ("Python", "JavaScript"),
        ("Paris", "France"),
        ("Python", "Paris"),
        ("RAG", "Embedding"),
        ("Dog", "Cat"),
        ("Python", "Football"),
    ]

    print(f"{'Pair':<35} {'Cosine':>8} {'Interpretation'}")
    print("─" * 65)
    for w1, w2 in comparisons:
        result = calc.compute_similarity(w1, w2)
        print(f"({w1}, {w2}){'':<{30 - len(w1) - len(w2) - 4}} "
              f"{result['cosine_similarity']:>8.4f}  {result['interpretation']}")

    print("\nMost similar to 'RAG':")
    for word, score in calc.find_most_similar("RAG", top_k=3):
        print(f"  {word:<15} {score:.4f}")


def demo_attention_math() -> None:
    print("\n── ATTENTION MATH (TRANSFORMER PREVIEW) ──")

    np.random.seed(42)
    seq_len, d_k = 4, 8

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    # Scaled dot-product attention
    scores = (Q @ K.T) / np.sqrt(d_k)     # shape: (4, 4)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)  # softmax
    output = weights @ V                   # shape: (4, 8)

    print(f"Q shape: {Q.shape} (4 tokens, 8-dim queries)")
    print(f"Attention weights shape: {weights.shape} (token × token)")
    print(f"Output shape: {output.shape} (weighted combination of values)")
    print(f"\nAttention weights (row = attending token, col = attended token):")
    print(np.round(weights, 3))
    print("\n→ Day 14 (Transformers) will implement this fully")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 04 — AI MATHEMATICS                           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_vector_math()
    demo_softmax()
    demo_statistics()
    demo_similarity_calculator()
    demo_attention_math()

    print("\n" + "="*60)
    print("✓ Day 04 AI mathematics demo complete!")
    print("="*60)
    print("\nKey math for AI:")
    print("  • Dot product → alignment / similarity")
    print("  • Cosine similarity → scale-invariant semantic similarity")
    print("  • Softmax → probability distribution from logits")
    print("  • Mean/Std → normalization and evaluation")
    print("  • Matrix multiplication → neural network forward pass")


if __name__ == "__main__":
    main()
