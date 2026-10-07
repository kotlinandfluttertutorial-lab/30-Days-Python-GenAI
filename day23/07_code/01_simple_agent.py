"""
Day 23 — Simple AI Agent
==========================
Implements ReAct agent from scratch.
Works with MockLLM (no API key) or real LLM.

Run: python 01_simple_agent.py
"""

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────
# TOOL DEFINITION
# ─────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., str]
    parameters: dict[str, str]

    def call(self, **kwargs: Any) -> str:
        try:
            if self.parameters:
                return str(self.function(**kwargs))
            return str(self.function())
        except Exception as e:
            return f"Tool error: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        return "Error: expression contains invalid characters"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # No builtins
        return str(round(float(result), 6))
    except Exception as e:
        return f"Math error: {e}"


def get_current_datetime() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def knowledge_search(query: str) -> str:
    """Search a small local knowledge base."""
    knowledge = {
        "rag": "RAG combines retrieval with generation. Reduces hallucination by grounding LLM in retrieved facts.",
        "embedding": "Embeddings convert text to vectors. Similar texts have similar vectors.",
        "transformer": "Transformers use self-attention. Enables parallel processing of sequences.",
        "python": "Python is a high-level interpreted language. Main language for AI/ML.",
        "agent": "AI agents combine LLM + tools + memory + planning to solve multi-step tasks.",
        "chromadb": "ChromaDB is an embedded vector database. Great for development and prototyping.",
        "fastapi": "FastAPI is an async Python framework. Auto-generates OpenAPI docs.",
    }
    query_lower = query.lower()
    matches = []
    for key, value in knowledge.items():
        if key in query_lower or any(w in query_lower for w in key.split()):
            matches.append(value)
    return " | ".join(matches) if matches else f"No information found for: '{query}'"


def word_counter(text: str) -> str:
    words = len(text.split())
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


# ─────────────────────────────────────────────────────────
# LLM CLIENTS
# ─────────────────────────────────────────────────────────

class MockLLM:
    """Scripted responses for demonstration without API key."""

    def __init__(self) -> None:
        self._call_count = 0
        self._scripts: dict[int, str] = {}

    def set_script(self, script: list[str]) -> None:
        self._scripts = {i: s for i, s in enumerate(script)}
        self._call_count = 0

    def complete(self, messages: list[dict]) -> str:
        response = self._scripts.get(self._call_count, '{"thought": "done", "action": "final_answer", "answer": "Task complete."}')
        self._call_count += 1
        return response


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model

    def complete(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""


class GroqLLM:
    def __init__(self, model: str = "llama-3.1-70b-versatile") -> None:
        from openai import OpenAI
        self._client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = model

    def complete(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""


def get_llm():
    if os.getenv("GROQ_API_KEY"):
        print("Using Groq LLM")
        return GroqLLM()
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI LLM")
        return OpenAILLM()
    else:
        print("Using Mock LLM (no API key found)")
        return MockLLM()


# ─────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────

class SimpleAgent:
    """ReAct agent: Think → Act → Observe → repeat."""

    MAX_ITERATIONS = 8

    def __init__(self, llm, tools: list[Tool]) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    def _system_prompt(self) -> str:
        tool_lines = "\n".join(
            f"  - {t.name}({', '.join(t.parameters)}): {t.description}"
            for t in self.tools.values()
        )
        return f"""You are a helpful AI agent. Use tools to answer questions.

TOOLS:
{tool_lines}
  - final_answer(answer): Provide the final answer

RESPONSE FORMAT (always valid JSON):
{{
  "thought": "your reasoning about what to do",
  "action": "tool_name",
  "args": {{"arg_name": "value"}}
}}

Or for final answer:
{{
  "thought": "I have everything I need",
  "action": "final_answer",
  "answer": "complete answer here"
}}

Rules:
- ALWAYS use JSON format
- Use tools to get real information, never guess
- Reach a final_answer within {self.MAX_ITERATIONS} iterations"""

    def run(self, goal: str, verbose: bool = True) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": goal},
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"AGENT GOAL: {goal}")
            print(f"{'='*60}")

        for iteration in range(self.MAX_ITERATIONS):
            raw = self.llm.complete(messages)
            messages.append({"role": "assistant", "content": raw})

            # Parse
            try:
                parsed = self._parse(raw)
            except Exception as e:
                observation = f"Parse error: {e}. Respond with valid JSON."
                messages.append({"role": "user", "content": observation})
                continue

            thought = parsed.get("thought", "")
            action = parsed.get("action", "")

            if verbose:
                print(f"\n[Step {iteration + 1}]")
                print(f"  Thought: {thought}")

            if action == "final_answer":
                answer = parsed.get("answer", "")
                if verbose:
                    print(f"  ✓ Final Answer: {answer}")
                return answer

            if action in self.tools:
                args = parsed.get("args", {})
                if verbose:
                    print(f"  → Tool: {action}({args})")

                observation = self.tools[action].call(**args)

                if verbose:
                    print(f"  ← Result: {observation[:120]}")

                messages.append({
                    "role": "user",
                    "content": f"Tool '{action}' returned: {observation}",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Error: tool '{action}' not found. Available: {list(self.tools.keys())}",
                })

        return "Agent did not reach a conclusion."

    @staticmethod
    def _parse(text: str) -> dict:
        text = text.strip()
        for prefix in ["```json", "```"]:
            if text.startswith(prefix):
                text = text[len(prefix):]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())


# ─────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────

def demo_with_mock_llm() -> None:
    """Demo using scripted mock LLM."""
    print("\n── DEMO WITH MOCK LLM (no API key needed) ──")
    mock_llm = MockLLM()

    # Script: solve a math problem
    mock_llm.set_script([
        '{"thought": "I need to calculate the compound interest", "action": "calculator", "args": {"expression": "1000 * (1 + 0.05) ** 10"}}',
        '{"thought": "I have the result", "action": "final_answer", "answer": "After 10 years at 5% annual compound interest, $1000 grows to approximately $1628.89."}',
    ])

    tools = [
        Tool("calculator", "Evaluate math expressions", calculator, {"expression": "math expression"}),
        Tool("search", "Search knowledge base", knowledge_search, {"query": "search term"}),
        Tool("get_time", "Get current UTC time", get_current_datetime, {}),
    ]

    agent = SimpleAgent(mock_llm, tools)
    result = agent.run("What is $1000 at 5% compound interest for 10 years?")
    print(f"\nFinal: {result}")


def demo_with_real_llm() -> None:
    """Demo using real LLM if API key available."""
    llm = get_llm()
    if isinstance(llm, MockLLM):
        print("\nSkipping real LLM demo (no API key)")
        return

    tools = [
        Tool("calculator", "Evaluate math expressions", calculator, {"expression": "math expression"}),
        Tool("search", "Search AI knowledge base", knowledge_search, {"query": "search query"}),
        Tool("get_time", "Get current UTC date and time", get_current_datetime, {}),
        Tool("word_counter", "Count words/chars in text", word_counter, {"text": "text to analyze"}),
    ]

    agent = SimpleAgent(llm, tools)

    goals = [
        "What time is it and how many days have passed since January 1, 2024?",
        "Search for information about RAG and tell me how many words are in that description.",
    ]

    for goal in goals:
        agent.run(goal)


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 23 — SIMPLE AI AGENT (ReAct Pattern)          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_with_mock_llm()
    demo_with_real_llm()

    print("\n✓ Day 23 Agent demo complete!")
    print("\nAgent loop summary:")
    print("  1. THINK: LLM reasons about what to do next")
    print("  2. ACT: Execute the selected tool")
    print("  3. OBSERVE: Feed tool result back to LLM")
    print("  4. REPEAT until final_answer is reached")
    print("\n  Key constraints:")
    print("  - MAX_ITERATIONS: prevents infinite loops")
    print("  - Error handling: unknown tool → inform LLM → retry")
    print("  - Parse error: ask LLM to respond with valid JSON")


if __name__ == "__main__":
    main()
