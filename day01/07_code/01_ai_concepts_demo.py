"""
Day 01 — AI Concepts Demo
=========================
This script demonstrates AI concepts programmatically.
No LLM API key required — pure Python demonstrations.

Run: python 01_ai_concepts_demo.py
"""

import math
from typing import Any


# ─────────────────────────────────────────────────────────
# 1. What Is an Embedding? (Simplified)
# ─────────────────────────────────────────────────────────

def demonstrate_embeddings() -> None:
    """
    Show how embeddings represent meaning as numbers.
    Using toy 4-dimensional vectors to illustrate the concept.
    Real embeddings have 768–3072 dimensions.
    """
    print("\n" + "="*60)
    print("CONCEPT 1: EMBEDDINGS")
    print("="*60)
    print("Embeddings map text/concepts to number vectors.")
    print("Similar concepts produce similar vectors.\n")

    # Toy 4D embeddings (in reality: 768-3072 dimensions)
    # Dimensions represent (roughly): [tech, biology, place, abstract]
    toy_embeddings: dict[str, list[float]] = {
        "Python":        [0.9, 0.1, 0.0, 0.3],
        "JavaScript":    [0.8, 0.1, 0.0, 0.3],
        "Paris":         [0.1, 0.0, 0.9, 0.2],
        "France":        [0.1, 0.0, 0.9, 0.3],
        "Dog":           [0.1, 0.9, 0.1, 0.1],
        "Cat":           [0.1, 0.8, 0.1, 0.1],
        "Transformer":   [0.8, 0.1, 0.0, 0.7],
    }

    print(f"{'Word':<15} {'Vector (4D toy)'}")
    print("-" * 50)
    for word, vec in toy_embeddings.items():
        vec_str = [f"{v:.1f}" for v in vec]
        print(f"{word:<15} {vec_str}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity = dot(a, b) / (|a| * |b|)
    Range: -1 (opposite) to 1 (identical direction)
    
    Used extensively in:
    - Finding similar documents in RAG
    - Semantic search
    - Recommendation systems
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(x**2 for x in b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def demonstrate_similarity() -> None:
    """Show that similar concepts produce similar vectors."""
    print("\n" + "="*60)
    print("CONCEPT 2: COSINE SIMILARITY")
    print("="*60)
    print("Similar things → high similarity score (close to 1.0)")
    print("Unrelated things → low similarity score (close to 0.0)\n")

    toy_embeddings: dict[str, list[float]] = {
        "Python":     [0.9, 0.1, 0.0, 0.3],
        "JavaScript": [0.8, 0.1, 0.0, 0.3],
        "Paris":      [0.1, 0.0, 0.9, 0.2],
        "France":     [0.1, 0.0, 0.9, 0.3],
        "Dog":        [0.1, 0.9, 0.1, 0.1],
    }

    comparisons = [
        ("Python", "JavaScript"),   # Should be high (both programming)
        ("Paris", "France"),        # Should be high (both French geography)
        ("Python", "Paris"),        # Should be low (unrelated)
        ("Python", "Dog"),          # Should be very low
    ]

    print(f"{'Pair':<30} {'Similarity':>12} {'Interpretation'}")
    print("-" * 65)
    for word1, word2 in comparisons:
        sim = cosine_similarity(toy_embeddings[word1], toy_embeddings[word2])
        if sim > 0.9:
            interpretation = "Very similar ✓✓"
        elif sim > 0.7:
            interpretation = "Similar ✓"
        elif sim > 0.4:
            interpretation = "Somewhat related"
        else:
            interpretation = "Unrelated ✗"
        print(f"({word1}, {word2}){'':>{25 - len(word1) - len(word2) - 4}} {sim:>12.4f}  {interpretation}")


# ─────────────────────────────────────────────────────────
# 2. What Is Tokenization?
# ─────────────────────────────────────────────────────────

def demonstrate_tokenization() -> None:
    """
    Show how text is split into tokens.
    
    LLMs don't see words — they see tokens (subword units).
    'Tokenization' → ['Token', 'ization'] = 2 tokens
    'Python' → ['Python'] = 1 token
    
    This affects:
    - Cost (you pay per token)
    - Context limits (128K context = 128K tokens, not words)
    - Behavior (model may treat rare words differently)
    """
    print("\n" + "="*60)
    print("CONCEPT 3: TOKENIZATION (Simplified)")
    print("="*60)
    print("LLMs don't process words — they process tokens.")
    print("Tokens are subword units.\n")

    # Simplified word-level tokenizer (real tokenizers use BPE)
    examples = [
        "Hello, how are you?",
        "Retrieval-Augmented Generation",
        "AI Engineer",
        "The quick brown fox jumps over the lazy dog",
        "supercalifragilisticexpialidocious",
    ]

    for text in examples:
        # Simple split (real tokenizers are more complex)
        simple_tokens = text.split()
        # Approximate: count punctuation as extra tokens
        approx_token_count = len(simple_tokens) + text.count(",") + text.count(".")
        
        print(f"Text: '{text}'")
        print(f"  Simple tokens: {simple_tokens}")
        print(f"  Approx token count: ~{approx_token_count}")
        
        # Cost estimate
        cost_per_1k_tokens = 0.005  # GPT-4o input price
        cost = (approx_token_count / 1000) * cost_per_1k_tokens
        print(f"  Cost (GPT-4o): ~${cost:.6f}")
        print()


# ─────────────────────────────────────────────────────────
# 3. What Is a Context Window?
# ─────────────────────────────────────────────────────────

def demonstrate_context_window() -> None:
    """
    Show the concept of context windows and token limits.
    """
    print("\n" + "="*60)
    print("CONCEPT 4: CONTEXT WINDOWS")
    print("="*60)
    print("The context window is the maximum amount of text")
    print("an LLM can process in a single call.\n")

    models = [
        ("GPT-4o",          128_000, 5.00, 15.00),
        ("Claude 3.5 Sonnet", 200_000, 3.00, 15.00),
        ("Gemini 1.5 Pro",  1_000_000, 7.00, 21.00),
        ("Llama 3.1 8B",    128_000, 0.00, 0.00),   # Free local
        ("Llama 3.1 70B",   128_000, 0.00, 0.00),   # Free local
    ]

    words_per_token = 0.75  # Approximate ratio

    print(f"{'Model':<25} {'Context':>10} {'Words':>10} {'$/1K in':>10} {'$/1K out':>10}")
    print("-" * 70)
    for model, ctx, cost_in, cost_out in models:
        words = int(ctx * words_per_token)
        cost_str = f"${cost_in:.2f}" if cost_in > 0 else "Free"
        cost_out_str = f"${cost_out:.2f}" if cost_out > 0 else "Free"
        print(f"{model:<25} {ctx:>10,} {words:>10,} {cost_str:>10} {cost_out_str:>10}")

    print("\nWhat fits in 128K tokens?")
    print("  - 4-5 average novels")
    print("  - ~350 pages of text")
    print("  - ~500 typical email chains")
    print("  - The entire Python documentation (twice)")


# ─────────────────────────────────────────────────────────
# 4. LLM Temperature Demo
# ─────────────────────────────────────────────────────────

def demonstrate_temperature() -> None:
    """
    Conceptually demonstrate what temperature does.
    Temperature controls randomness in token selection.
    """
    print("\n" + "="*60)
    print("CONCEPT 5: TEMPERATURE & SAMPLING")
    print("="*60)
    print("Temperature controls how random the output is.\n")

    # Simulate probability distribution at different temperatures
    import math
    
    # Raw logits for "The capital of France is ___"
    candidates = {
        "Paris":    10.0,   # Highest logit
        "Lyon":     6.0,
        "Berlin":   4.0,
        "London":   3.0,
        "a city":   2.0,
    }

    def softmax_with_temperature(logits: dict[str, float], temp: float) -> dict[str, float]:
        """Apply temperature scaling and softmax to get probabilities."""
        scaled = {k: v / temp for k, v in logits.items()}
        max_val = max(scaled.values())
        exp_vals = {k: math.exp(v - max_val) for k, v in scaled.items()}
        total = sum(exp_vals.values())
        return {k: v / total for k, v in exp_vals.items()}

    for temperature in [0.1, 0.7, 1.0, 1.5]:
        probs = softmax_with_temperature(candidates, temperature)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Temperature = {temperature}:")
        for token, prob in sorted_probs:
            bar = "█" * int(prob * 40)
            print(f"  {token:<10} {prob:.3f} {bar}")
        print(f"  → Effect: {'Very deterministic' if temperature < 0.3 else 'Balanced' if temperature < 0.9 else 'Creative' if temperature <= 1.2 else 'Very random'}\n")

    print("Rules of thumb:")
    print("  temp=0.0: Deterministic (good for facts, math)")
    print("  temp=0.3-0.7: Balanced (good for most tasks)")
    print("  temp=0.8-1.2: Creative (good for writing)")
    print("  temp>1.5: Very random (rarely useful)")


# ─────────────────────────────────────────────────────────
# 5. RAG Concept Demo
# ─────────────────────────────────────────────────────────

def demonstrate_rag_concept() -> None:
    """
    Show the RAG concept with a minimal simulation.
    No LLM required — shows the retrieval logic clearly.
    """
    print("\n" + "="*60)
    print("CONCEPT 6: RAG (Retrieval-Augmented Generation)")
    print("="*60)

    # Simulate a knowledge base (in real RAG, these are embeddings)
    knowledge_base = [
        {
            "id": 1,
            "text": "Our return policy allows returns within 30 days of purchase.",
            "keywords": ["return", "policy", "30 days", "purchase"],
        },
        {
            "id": 2,
            "text": "We offer free shipping on orders over $50.",
            "keywords": ["shipping", "free", "50", "order"],
        },
        {
            "id": 3,
            "text": "Customer support is available 24/7 via chat and email.",
            "keywords": ["support", "24/7", "chat", "email", "customer"],
        },
        {
            "id": 4,
            "text": "All electronics come with a 1-year manufacturer warranty.",
            "keywords": ["warranty", "electronics", "1 year", "manufacturer"],
        },
    ]

    def simple_retrieve(query: str, knowledge_base: list[dict], top_k: int = 2) -> list[dict]:
        """
        Simplified retrieval using keyword matching.
        Real RAG uses vector similarity (cosine similarity on embeddings).
        """
        query_words = set(query.lower().split())
        scores = []
        
        for doc in knowledge_base:
            doc_keywords = set(doc["keywords"])
            doc_words = set(doc["text"].lower().split())
            # Count matching keywords + words
            overlap = len(query_words & doc_keywords) + len(query_words & doc_words)
            scores.append((overlap, doc))
        
        # Sort by relevance score
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]

    def rag_answer(query: str) -> None:
        """Simulate the full RAG pipeline."""
        print(f"\nQuery: '{query}'")
        print("\nStep 1: RETRIEVE relevant documents")
        retrieved = simple_retrieve(query, knowledge_base)
        for doc in retrieved:
            print(f"  → [{doc['id']}] {doc['text']}")
        
        print("\nStep 2: BUILD context for LLM")
        context = "\n".join([f"- {doc['text']}" for doc in retrieved])
        prompt = f"""Answer the question based on the context below.
        
Context:
{context}

Question: {query}

Answer:"""
        print(f"  → Prompt constructed ({len(prompt)} chars)")
        
        print("\nStep 3: LLM GENERATES answer (simulated)")
        # In real RAG, this calls the LLM
        print(f"  → [In production: LLM generates answer citing the context]")
        print(f"  → The answer would reference: '{retrieved[0]['text'][:50]}...'")

    queries = [
        "What is your return policy?",
        "Do you have free shipping?",
    ]
    
    for query in queries:
        rag_answer(query)
        print()


# ─────────────────────────────────────────────────────────
# 6. AI System Comparison
# ─────────────────────────────────────────────────────────

def demonstrate_system_types() -> None:
    """Compare traditional software vs AI software behavior."""
    print("\n" + "="*60)
    print("CONCEPT 7: TRADITIONAL SOFTWARE vs AI SOFTWARE")
    print("="*60)

    print("\n--- TRADITIONAL SOFTWARE ---")
    print("Same input ALWAYS produces same output:\n")
    
    def get_tax(amount: float, rate: float) -> float:
        return round(amount * rate, 2)
    
    for _ in range(3):
        result = get_tax(100.0, 0.20)
        print(f"  get_tax(100, 0.20) = {result}")
    
    print("\n--- AI SOFTWARE ---")
    print("Same input produces DIFFERENT outputs (due to temperature):")
    print("(Simulated - showing the concept)\n")
    
    import random
    random.seed(None)  # True randomness
    
    example_responses = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "France's capital city is Paris, a major European hub.",
        "Paris serves as the capital of France.",
    ]
    
    for i in range(3):
        # Simulate non-deterministic LLM response
        response = random.choice(example_responses)
        print(f"  Call {i+1}: '{response}'")
    
    print("\nKey implication:")
    print("  Traditional: write unit tests → done")
    print("  AI: write evaluation functions + measure quality scores")


# ─────────────────────────────────────────────────────────
# 7. Token Cost Calculator
# ─────────────────────────────────────────────────────────

def token_cost_calculator() -> None:
    """
    Calculate costs for different LLM usage patterns.
    Critical for production planning.
    """
    print("\n" + "="*60)
    print("CONCEPT 8: TOKEN COST CALCULATOR")
    print("="*60)

    scenarios = [
        {
            "name": "Simple chatbot (no RAG)",
            "system_tokens": 100,
            "context_tokens": 0,
            "query_tokens": 50,
            "response_tokens": 200,
            "requests_per_day": 1000,
        },
        {
            "name": "RAG assistant (3 chunks)",
            "system_tokens": 200,
            "context_tokens": 900,
            "query_tokens": 50,
            "response_tokens": 300,
            "requests_per_day": 1000,
        },
        {
            "name": "Complex agent (5 tool calls)",
            "system_tokens": 500,
            "context_tokens": 2000,
            "query_tokens": 100,
            "response_tokens": 500,
            "requests_per_day": 500,
        },
    ]

    models = [
        ("GPT-4o",      0.005, 0.015),
        ("Claude Haiku", 0.00025, 0.00125),
        ("Llama 3.1",   0.0,   0.0),
    ]

    for scenario in scenarios:
        input_tokens = (
            scenario["system_tokens"]
            + scenario["context_tokens"]
            + scenario["query_tokens"]
        )
        output_tokens = scenario["response_tokens"]
        daily_requests = scenario["requests_per_day"]
        
        print(f"\n{scenario['name']}:")
        print(f"  Input tokens/request: {input_tokens:,}")
        print(f"  Output tokens/request: {output_tokens:,}")
        print(f"  Requests/day: {daily_requests:,}")
        print(f"\n  {'Model':<20} {'Cost/request':>15} {'Daily cost':>12} {'Monthly cost':>14}")
        print(f"  {'-'*65}")
        
        for model_name, input_cost, output_cost in models:
            cost_per_request = (
                (input_tokens / 1000) * input_cost +
                (output_tokens / 1000) * output_cost
            )
            daily_cost = cost_per_request * daily_requests
            monthly_cost = daily_cost * 30
            
            if cost_per_request == 0:
                cost_str = "Free"
                daily_str = "Free"
                monthly_str = "Free"
            else:
                cost_str = f"${cost_per_request:.4f}"
                daily_str = f"${daily_cost:.2f}"
                monthly_str = f"${monthly_cost:,.2f}"
            
            print(f"  {model_name:<20} {cost_str:>15} {daily_str:>12} {monthly_str:>14}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    """Run all AI concept demonstrations."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 01 — AI CONCEPTS DEMONSTRATION                ║")
    print("║        No API key required — pure Python                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demonstrate_embeddings()
    demonstrate_similarity()
    demonstrate_tokenization()
    demonstrate_context_window()
    demonstrate_temperature()
    demonstrate_rag_concept()
    demonstrate_system_types()
    token_cost_calculator()

    print("\n" + "="*60)
    print("✓ Day 01 concept demo complete!")
    print("="*60)
    print("\nKey takeaways:")
    print("1. Embeddings = text as numbers, capturing meaning")
    print("2. Cosine similarity = how similar two vectors are")
    print("3. Tokens = LLM's atomic unit (not words)")
    print("4. Context window = max tokens per LLM call")
    print("5. Temperature = output randomness control")
    print("6. RAG = retrieve context before generating")
    print("7. AI output is probabilistic, not deterministic")
    print("8. Tokens cost money — always calculate costs")


if __name__ == "__main__":
    main()
