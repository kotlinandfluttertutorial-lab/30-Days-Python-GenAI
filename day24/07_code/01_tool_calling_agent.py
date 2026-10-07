"""
Day 24 — Tool-Using AI Agent (Native Function Calling)
=======================================================
Uses OpenAI/Groq native tool calling API.
Falls back to manual JSON parsing when no API key.

Run: python 01_tool_calling_agent.py
"""

import json
import os
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Safely evaluate math expression."""
    allowed = set("0123456789+-*/().,% ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    try:
        result = eval(expression, {"__builtins__": {}, "math": math}, {})
        return str(round(float(result), 6))
    except Exception as e:
        return f"Math error: {e}"


def search_knowledge_base(query: str) -> str:
    """Search AI engineering knowledge base."""
    kb = {
        "rag": "RAG: Retrieval-Augmented Generation. Chunks docs, embeds, stores in vector DB. Reduces hallucination.",
        "embedding": "Embeddings: dense vectors (384-3072 dims) encoding semantic meaning. Cosine similarity for comparison.",
        "agent": "Agents: LLM + tools + memory. ReAct loop: think → act → observe. Use max_iterations to prevent loops.",
        "transformer": "Transformers: self-attention QKV mechanism. Encoder (BERT) or Decoder (GPT). Base of all LLMs.",
        "fastapi": "FastAPI: async Python web framework. Auto-docs, Pydantic validation, dependency injection.",
        "docker": "Docker: containerizes apps with all deps. Dockerfile → Image → Container. Docker Compose for multi-service.",
        "vector db": "Vector DBs: store embeddings, ANN search. ChromaDB (dev), FAISS (perf), pgvector (Postgres).",
        "fine-tuning": "Fine-tuning: additional training on task-specific data. Use for behavior/style, not knowledge. Use RAG for knowledge.",
    }
    q = query.lower()
    matches = [v for k, v in kb.items() if k in q or any(w in q for w in k.split())]
    return " | ".join(matches) if matches else "No relevant information found."


def count_words(text: str) -> str:
    """Count words, characters, sentences in text."""
    words = len(text.split())
    chars = len(text)
    sentences = sum(1 for c in text if c in ".!?")
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


def get_current_utc_time() -> str:
    """Get current UTC date and time."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def format_as_markdown_table(data: str) -> str:
    """Convert CSV-formatted data to a markdown table."""
    lines = [l.strip() for l in data.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return data
    header = "| " + " | ".join(lines[0].split(",")) + " |"
    separator = "| " + " | ".join(["---"] * len(lines[0].split(","))) + " |"
    rows = ["| " + " | ".join(l.split(",")) + " |" for l in lines[1:]]
    return "\n".join([header, separator] + rows)


# ─────────────────────────────────────────────────────────
# TOOL REGISTRY (maps name → callable)
# ─────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base,
    "count_words": count_words,
    "get_current_utc_time": lambda: get_current_utc_time(),
    "format_as_markdown_table": format_as_markdown_table,
}

# Tool schemas for OpenAI API
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate math expressions. Use for any arithmetic, percentages, or calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression, e.g. '100 * 0.15' or '(50 + 30) / 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search AI engineering knowledge base for information about RAG, embeddings, agents, transformers, FastAPI, Docker, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count words, characters, and sentences in a text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_utc_time",
            "description": "Get the current UTC date and time. Use when user asks about time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ─────────────────────────────────────────────────────────
# TOOL CALLING AGENT
# ─────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool safely."""
    func = TOOL_REGISTRY.get(name)
    if not func:
        return f"Error: tool '{name}' not found"
    try:
        return str(func(**args)) if args else str(func())
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


def run_agent_with_tools(goal: str, verbose: bool = True) -> str:
    """Run the OpenAI native tool-calling agent loop."""
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return run_mock_agent(goal, verbose)

    from openai import OpenAI

    if os.getenv("GROQ_API_KEY"):
        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        model = "llama-3.1-70b-versatile"
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = "gpt-4o-mini"

    if verbose:
        print(f"\n{'='*60}")
        print(f"GOAL: {goal}")
        print(f"{'='*60}")

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant with access to tools. "
                "Use tools to get accurate information. "
                "Never guess — always use a tool if you need data."
            ),
        },
        {"role": "user", "content": goal},
    ]

    for step in range(8):  # Max 8 iterations
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            tools=TOOL_SCHEMAS,  # type: ignore
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        # No tool calls → final answer
        if not msg.tool_calls:
            answer = msg.content or ""
            if verbose:
                print(f"\n✓ Answer: {answer}")
            return answer

        # Execute tool calls
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if verbose:
                print(f"\n[Step {step+1}] Tool: {name}({args})")

            result = execute_tool(name, args)

            if verbose:
                print(f"  Result: {result[:100]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Agent did not converge."


def run_mock_agent(goal: str, verbose: bool = True) -> str:
    """Demonstrate tool calling structure without API key."""
    if verbose:
        print(f"\n[MOCK MODE — No API key] Goal: {goal}")
        print("  In production, the LLM would:")
        print("  1. Decide which tool to call based on the goal")
        print("  2. Structured tool call: {name, args} — no parsing needed")
        print("  3. Receive tool result as 'tool' role message")
        print("  4. Generate final answer using tool results")

        # Show what would happen
        if "calculat" in goal.lower() or "%" in goal or any(c.isdigit() for c in goal):
            print(f"\n  → Would call: calculator(expression='...')")
            print(f"  → Tool returns: calculated result")
        elif "time" in goal.lower() or "date" in goal.lower():
            actual_time = get_current_utc_time()
            print(f"\n  → Would call: get_current_utc_time()")
            print(f"  → Tool returns: {actual_time}")
        else:
            print(f"\n  → Would call: search_knowledge_base(query='{goal[:30]}')")
            result = search_knowledge_base(goal)
            print(f"  → Tool returns: {result[:80]}...")

    return "[Mock answer — add API key to .env for real response]"


# ─────────────────────────────────────────────────────────
# DEMONSTRATIONS
# ─────────────────────────────────────────────────────────

def demo_tool_schemas() -> None:
    print("\n── TOOL SCHEMAS ──")
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        print(f"\n  Tool: {fn['name']}")
        print(f"  Description: {fn['description'][:70]}...")
        print(f"  Parameters: {list(params.keys())} (required: {required})")


def demo_tool_execution() -> None:
    print("\n── DIRECT TOOL EXECUTION ──")
    tests = [
        ("calculator", {"expression": "85.50 * 0.15"}),
        ("calculator", {"expression": "1000 * (1 + 0.05) ** 10"}),
        ("search_knowledge_base", {"query": "how does RAG work"}),
        ("search_knowledge_base", {"query": "vector database options"}),
        ("get_current_utc_time", {}),
        ("count_words", {"text": "The quick brown fox jumps over the lazy dog"}),
    ]
    for name, args in tests:
        result = execute_tool(name, args)
        print(f"  {name}({args}) → {result}")


def demo_agent_goals() -> None:
    print("\n── AGENT WITH TOOL CALLING ──")
    goals = [
        "What is 15% of $1,250.75?",
        "What is RAG and how many words are in that description?",
        "What time is it right now?",
    ]
    for goal in goals:
        run_agent_with_tools(goal)


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 24 — TOOL CALLING AI AGENT                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_tool_schemas()
    demo_tool_execution()
    demo_agent_goals()

    print("\n✓ Day 24 Tool Calling demo complete!")
    print("\nKey tool calling advantages over manual JSON parsing:")
    print("  • LLM API guarantees valid tool call structure")
    print("  • Parallel tool calls in one step")
    print("  • finish_reason='tool_calls' vs 'stop' signals clearly")
    print("  • Tool result goes back as 'tool' role — LLM knows it's a result")
    print("  • Better at deciding WHEN to use tools vs answer directly")


if __name__ == "__main__":
    main()
