"""
Day 01 — AI Engineer CLI Profile (Mini Project)
================================================
A professional CLI tool that demonstrates:
1. Knowledge of the AI engineering landscape
2. Clean Python code structure
3. Type hints throughout
4. Proper error handling

This IS your mini project for Day 1.

Run: python 03_ai_engineer_profile.py
"""

import sys
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────

@dataclass
class AIConcept:
    """Represents an AI/ML concept with explanation."""
    name: str
    category: str
    one_liner: str
    details: str
    use_case: str
    related: list[str] = field(default_factory=list)


@dataclass
class TechStack:
    """Represents a technology in the AI stack."""
    name: str
    layer: str
    purpose: str
    when_to_use: str


# ─────────────────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────────────────

AI_CONCEPTS: list[AIConcept] = [
    AIConcept(
        name="LLM",
        category="Foundation",
        one_liner="A large neural network that predicts the next token in a sequence.",
        details=(
            "Trained on trillions of tokens of internet text. "
            "Learns grammar, facts, reasoning, and code through self-supervised prediction. "
            "Emergent capabilities appear at scale that weren't explicitly trained."
        ),
        use_case="Text generation, Q&A, code, summarization, classification",
        related=["Transformer", "Tokenization", "Context Window", "Temperature"],
    ),
    AIConcept(
        name="RAG",
        category="Architecture Pattern",
        one_liner="Enhance LLM answers by retrieving relevant context before generating.",
        details=(
            "Addresses two core LLM problems: knowledge cutoff and hallucination. "
            "Documents are chunked, embedded, and stored in a vector database. "
            "At query time, relevant chunks are retrieved and injected into the prompt."
        ),
        use_case="Document Q&A, knowledge assistants, enterprise AI",
        related=["Embeddings", "Vector Database", "Chunking", "Reranking"],
    ),
    AIConcept(
        name="Embedding",
        category="Representation",
        one_liner="A dense vector representing the semantic meaning of text.",
        details=(
            "Text → model → 768-3072 dimensional vector. "
            "Similar meaning → similar vectors. "
            "Enables mathematical operations on semantics: cosine similarity, clustering."
        ),
        use_case="Semantic search, RAG retrieval, clustering, recommendation",
        related=["Vector Database", "Cosine Similarity", "RAG", "Semantic Search"],
    ),
    AIConcept(
        name="Agent",
        category="System Pattern",
        one_liner="LLM + Tools + Memory + Planning that can act autonomously.",
        details=(
            "The agent loop: Think → Act (tool call) → Observe result → Think again. "
            "Goes beyond single Q&A to multi-step reasoning and tool use. "
            "Failure modes: infinite loops, tool errors, context exhaustion."
        ),
        use_case="Research automation, coding assistants, workflow automation",
        related=["Tool Calling", "Memory", "ReAct", "MCP"],
    ),
    AIConcept(
        name="Transformer",
        category="Architecture",
        one_liner="Neural architecture using self-attention to process sequences in parallel.",
        details=(
            "Introduced in 'Attention Is All You Need' (2017). "
            "Self-attention: each token attends to all other tokens. "
            "Foundation of BERT (encoder), GPT (decoder), T5 (encoder-decoder)."
        ),
        use_case="All modern NLP, LLMs, vision transformers",
        related=["Attention", "BERT", "GPT", "Positional Encoding"],
    ),
    AIConcept(
        name="Vector Database",
        category="Infrastructure",
        one_liner="Database optimized for storing and querying high-dimensional vectors.",
        details=(
            "Traditional SQL cannot efficiently do 'find top-K most similar vectors'. "
            "Uses ANN algorithms: HNSW, IVF. "
            "Key operations: insert, similarity search, metadata filtering."
        ),
        use_case="RAG storage, semantic search, recommendation",
        related=["Embeddings", "HNSW", "FAISS", "ChromaDB", "pgvector"],
    ),
    AIConcept(
        name="Prompt Engineering",
        category="Technique",
        one_liner="Crafting inputs that guide LLM behavior toward desired outputs.",
        details=(
            "Zero-shot: direct instruction. "
            "Few-shot: examples in the prompt. "
            "Chain-of-thought: ask model to reason step by step. "
            "The prompt is code — version it, test it, monitor it."
        ),
        use_case="All LLM applications",
        related=["Zero-shot", "Few-shot", "CoT", "Structured Output"],
    ),
    AIConcept(
        name="Fine-tuning",
        category="Training",
        one_liner="Additional training on specific data to change model behavior or style.",
        details=(
            "Supervised Fine-Tuning (SFT): train on instruction-response pairs. "
            "RLHF: train on human preference data. "
            "DPO: train on preference pairs without explicit reward model. "
            "When to use: behavior change, style, domain-specific format."
        ),
        use_case="Custom style, behavior change, domain-specific models",
        related=["RLHF", "DPO", "LoRA", "SFT"],
    ),
    AIConcept(
        name="Hallucination",
        category="Failure Mode",
        one_liner="LLM generating plausible-sounding but factually incorrect content.",
        details=(
            "Not a bug — an inherent property of how LLMs work. "
            "Types: factual errors, citation fabrication, reasoning errors. "
            "Mitigations: RAG (ground in facts), evaluation, structured output, guardrails."
        ),
        use_case="Understanding and mitigating LLM failure modes",
        related=["RAG", "Evaluation", "Guardrails", "Grounding"],
    ),
    AIConcept(
        name="LLMOps",
        category="Operations",
        one_liner="Operational practices for maintaining LLM applications in production.",
        details=(
            "Prompt versioning, output evaluation, cost monitoring, latency tracking. "
            "Key difference from MLOps: non-deterministic outputs require evaluation, not just metrics. "
            "Production concerns: prompt injection, PII leakage, cost overruns."
        ),
        use_case="Operating LLM systems at scale",
        related=["Observability", "Evaluation", "Guardrails", "Cost Optimization"],
    ),
]

TECH_STACK: list[TechStack] = [
    TechStack("Python", "Language", "Primary AI engineering language", "Always"),
    TechStack("FastAPI", "Backend", "High-performance async AI API server", "All production AI backends"),
    TechStack("PyTorch", "ML Framework", "Deep learning, custom models", "Day 11+ when training models"),
    TechStack("HuggingFace Transformers", "NLP", "Pre-trained models, tokenizers", "NLP tasks, embedding models"),
    TechStack("OpenAI SDK", "LLM", "GPT-4o, embeddings, fine-tuning", "When using OpenAI models"),
    TechStack("Anthropic SDK", "LLM", "Claude models", "When using Anthropic models"),
    TechStack("ChromaDB", "Vector DB", "Embedded vector store for development", "Development, small-medium scale"),
    TechStack("FAISS", "Vector Search", "Fast in-process similarity search", "High-performance search, research"),
    TechStack("pgvector", "Vector DB", "PostgreSQL vector extension", "Production with existing Postgres"),
    TechStack("PostgreSQL", "Database", "Relational storage for users, metadata", "All production applications"),
    TechStack("Redis", "Cache/Queue", "Response caching, rate limiting, sessions", "Production caching, queues"),
    TechStack("Docker", "Containerization", "Reproducible deployment", "All production deployments"),
    TechStack("Pydantic", "Validation", "Data validation, LLM structured output", "All FastAPI applications"),
    TechStack("pytest", "Testing", "Unit and integration testing", "All code"),
]


# ─────────────────────────────────────────────────────────
# Display Functions
# ─────────────────────────────────────────────────────────

def print_header(title: str, width: int = 65) -> None:
    """Print a formatted header."""
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_concept(concept: AIConcept, verbose: bool = False) -> None:
    """Print a formatted concept entry."""
    print(f"\n  [{concept.category}] {concept.name}")
    print(f"  {'─' * 55}")
    print(f"  What: {concept.one_liner}")
    
    if verbose:
        print(f"  Why:  {concept.details}")
        print(f"  When: {concept.use_case}")
        if concept.related:
            print(f"  See also: {', '.join(concept.related)}")


def show_concept_map() -> None:
    """Display the AI engineering concept map."""
    print_header("AI ENGINEERING CONCEPT MAP")
    
    categories: dict[str, list[AIConcept]] = {}
    for concept in AI_CONCEPTS:
        categories.setdefault(concept.category, []).append(concept)
    
    for category, concepts in categories.items():
        print(f"\n  ── {category.upper()} ──")
        for concept in concepts:
            print(f"    • {concept.name:<25} {concept.one_liner[:45]}...")


def show_tech_stack() -> None:
    """Display the AI engineering tech stack."""
    print_header("AI ENGINEERING TECH STACK")
    
    layers: dict[str, list[TechStack]] = {}
    for tech in TECH_STACK:
        layers.setdefault(tech.layer, []).append(tech)
    
    for layer, techs in layers.items():
        print(f"\n  ── {layer.upper()} ──")
        for tech in techs:
            print(f"    • {tech.name:<30} {tech.purpose}")


def show_concept_detail(concept_name: str) -> None:
    """Show detailed information about a specific concept."""
    concept = next(
        (c for c in AI_CONCEPTS if c.name.lower() == concept_name.lower()),
        None
    )
    
    if not concept:
        print(f"  Concept '{concept_name}' not found.")
        print(f"  Available: {', '.join(c.name for c in AI_CONCEPTS)}")
        return
    
    print_header(f"CONCEPT: {concept.name}")
    print(f"\n  Category: {concept.category}")
    print(f"\n  Definition:")
    print(f"    {concept.one_liner}")
    print(f"\n  Deep Explanation:")
    print(f"    {concept.details}")
    print(f"\n  Use Case:")
    print(f"    {concept.use_case}")
    print(f"\n  Related Concepts:")
    for related in concept.related:
        print(f"    → {related}")


def show_learning_path() -> None:
    """Display the 30-day learning path."""
    print_header("30-DAY AI ENGINEER LEARNING PATH")
    
    phases = [
        ("Phase 1: Foundation", "Days 1–3",
         "Python, NumPy, Pandas, AI landscape"),
        ("Phase 2: Mathematics", "Days 4–6",
         "Linear algebra, probability, gradient descent"),
        ("Phase 3: Machine Learning", "Days 7–9",
         "scikit-learn, evaluation, end-to-end ML pipeline"),
        ("Phase 4: Deep Learning", "Days 10–12",
         "PyTorch, CNN, RNN, LSTM, attention"),
        ("Phase 5: NLP + Transformers + LLM", "Days 13–15",
         "Tokenization, attention, LLM APIs, sampling"),
        ("Phase 6: GenAI Engineering", "Days 16–18",
         "Prompt engineering, embeddings, vector databases"),
        ("Phase 7: RAG", "Days 19–22",
         "Basic RAG, advanced RAG, production RAG, evaluation"),
        ("Phase 8: AI Agents", "Days 23–25",
         "Agents, tool calling, MCP, multi-agent systems"),
        ("Phase 9: Production", "Days 26–27",
         "FastAPI, Docker, cloud deployment"),
        ("Phase 10: LLMOps + Job Ready", "Days 28–30",
         "Observability, security, system design, interviews"),
    ]
    
    for i, (phase, days, topics) in enumerate(phases, 1):
        print(f"\n  {i:2}. {phase}")
        print(f"     Duration: {days}")
        print(f"     Topics: {topics}")


def show_comparison(topic: str) -> None:
    """Show key comparisons."""
    comparisons = {
        "rag-vs-finetuning": {
            "title": "RAG vs Fine-Tuning",
            "rows": [
                ("Use when", "Facts change, need citations", "Behavior/style change"),
                ("Cost", "Low (API calls only)", "High (training compute)"),
                ("Update speed", "Instant (add docs)", "Slow (retrain)"),
                ("Hallucination", "Reduced", "Can still occur"),
                ("Privacy", "Docs stay local", "Training data shared"),
                ("Best for", "Q&A, knowledge retrieval", "Custom formats, personas"),
            ]
        },
        "llm-vs-agent": {
            "title": "LLM vs AI Agent",
            "rows": [
                ("Turns", "Single request-response", "Multi-step loop"),
                ("Memory", "Context window only", "Persistent memory"),
                ("Tools", "None", "Multiple tools"),
                ("Planning", "None", "Decompose complex tasks"),
                ("Cost", "One LLM call", "Multiple LLM calls"),
                ("Reliability", "High", "Lower (more failure modes)"),
                ("Use when", "Simple Q&A, generation", "Complex multi-step tasks"),
            ]
        },
        "sql-vs-vector": {
            "title": "SQL DB vs Vector DB",
            "rows": [
                ("Query type", "Exact match, range", "Similarity search"),
                ("Data type", "Structured tables", "High-dim vectors"),
                ("Performance", "Fast for exact queries", "Fast for similarity"),
                ("Use case", "Users, orders, metadata", "Embeddings, RAG"),
                ("Examples", "PostgreSQL, MySQL", "ChromaDB, FAISS, Pinecone"),
                ("Combined", "—", "pgvector: both in one DB"),
            ]
        },
    }

    topic_key = topic.lower().replace(" ", "-")
    if topic_key not in comparisons:
        print(f"  Available comparisons: {', '.join(comparisons.keys())}")
        return

    comp = comparisons[topic_key]
    print_header(comp["title"])
    
    col_width = 30
    header = f"  {'Aspect':<20} {'Left':<{col_width}} {'Right':<{col_width}}"
    print(header)
    print("  " + "─" * (20 + col_width * 2))
    
    left_name, right_name = comp["title"].split(" vs ")
    print(f"  {'':20} {left_name:<{col_width}} {right_name:<{col_width}}")
    print("  " + "─" * (20 + col_width * 2))
    
    for aspect, left_val, right_val in comp["rows"]:
        print(f"  {aspect:<20} {left_val:<{col_width}} {right_val:<{col_width}}")


def show_interview_cheatsheet() -> None:
    """Quick interview cheatsheet for Day 1 concepts."""
    print_header("DAY 01 INTERVIEW CHEATSHEET")
    
    qa_pairs = [
        ("What is an LLM?",
         "A large neural network predicting the next token, trained on massive text data."),
        ("What is RAG?",
         "Retrieval-Augmented Generation: retrieve relevant docs before generating to reduce hallucination."),
        ("What is an AI Agent?",
         "LLM + tools + memory + planning that loops: Think → Act → Observe → repeat."),
        ("What is an embedding?",
         "A dense vector representing semantic meaning. Similar meanings → similar vectors."),
        ("What is a vector database?",
         "A DB optimized for similarity search on high-dimensional vectors using ANN algorithms."),
        ("LLM vs AI Engineer?",
         "ML Engineer trains models. AI Engineer builds applications USING pre-trained models."),
        ("Why does RAG beat fine-tuning for Q&A?",
         "Instant updates, lower cost, citations, no hallucination risk from training data."),
        ("What causes hallucination?",
         "LLMs generate statistically likely tokens — no mechanism to verify truth."),
    ]
    
    for i, (question, answer) in enumerate(qa_pairs, 1):
        print(f"\n  Q{i}: {question}")
        print(f"  A:  {answer}")


def interactive_menu() -> None:
    """Run interactive CLI menu."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║        AI ENGINEER KNOWLEDGE PROFILE — DAY 01            ║")
    print("║        30-Day AI Engineer Program                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    menu_options = {
        "1": ("Concept Map", show_concept_map),
        "2": ("Tech Stack", show_tech_stack),
        "3": ("Learning Path", show_learning_path),
        "4": ("Compare: RAG vs Fine-Tuning", lambda: show_comparison("rag-vs-finetuning")),
        "5": ("Compare: LLM vs Agent", lambda: show_comparison("llm-vs-agent")),
        "6": ("Compare: SQL vs Vector DB", lambda: show_comparison("sql-vs-vector")),
        "7": ("Interview Cheatsheet", show_interview_cheatsheet),
        "8": ("Concept Detail (LLM)", lambda: show_concept_detail("LLM")),
        "9": ("Concept Detail (RAG)", lambda: show_concept_detail("RAG")),
        "0": ("Exit", None),
    }
    
    while True:
        print("\n" + "─" * 50)
        print("  MENU:")
        for key, (label, _) in menu_options.items():
            print(f"    {key}. {label}")
        
        choice = input("\n  Enter choice (0-9): ").strip()
        
        if choice == "0":
            print("\n  ✓ Session complete. Keep building!\n")
            break
        elif choice in menu_options:
            label, func = menu_options[choice]
            if func:
                func()
        else:
            print("  Invalid choice. Enter 0-9.")


def run_full_demo() -> None:
    """Run non-interactive demo of all sections."""
    show_concept_map()
    show_tech_stack()
    show_learning_path()
    show_comparison("rag-vs-finetuning")
    show_comparison("llm-vs-agent")
    show_interview_cheatsheet()


if __name__ == "__main__":
    # If running with --demo flag, show all sections non-interactively
    if "--demo" in sys.argv:
        run_full_demo()
    else:
        interactive_menu()
