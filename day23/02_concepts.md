# Day 23 — Concepts
## AI Agents

---

## 1. What Is an Agent?

```
LLM = one question → one answer (stateless)
Agent = goal → plan → tools → observe → loop until done (stateful)

Agent = LLM + Tools + Memory + Planning

KEY DIFFERENCE FROM CHATBOT:
  Chatbot: "What is the weather?" → "I don't have access to weather data."
  Agent:   "What is the weather?" → calls weather_api() → "It's 72°F and sunny."

KEY DIFFERENCE FROM RAG:
  RAG:   retrieves docs → one LLM call → answer (fixed pipeline)
  Agent: plans → calls any tool → observes → re-plans → multiple LLM calls
```

---

## 2. The Agent Loop (ReAct Pattern)

```
ReAct = Reasoning + Acting
Pattern: Think → Act → Observe → (repeat)

LOOP:
  1. THOUGHT: "I need to find the population of Paris to answer this"
  2. ACTION: search(query="Paris population 2024")
  3. OBSERVATION: "Paris population: 2.1 million (city) 12 million (metro)"
  4. THOUGHT: "I have the data, now I can answer"
  5. FINAL ANSWER: "Paris has a population of 2.1 million in the city..."
```

---

## 3. Implementing an Agent from Scratch

```python
import json
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., str]
    parameters: dict[str, str]  # param_name → description

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": "string", "description": v}
                    for k, v in self.parameters.items()
                },
                "required": list(self.parameters.keys()),
            },
        }

class SimpleAgent:
    """
    ReAct agent implemented from scratch.
    No frameworks — understand the mechanics.
    """

    MAX_ITERATIONS = 10

    def __init__(self, llm_client, tools: list[Tool]) -> None:
        self.llm = llm_client
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self._history: list[dict[str, str]] = []

    def _build_system_prompt(self) -> str:
        tool_desc = "\n".join(
            f"- {t.name}: {t.description} | Args: {t.parameters}"
            for t in self.tools.values()
        )
        return f"""You are an AI agent with access to tools.
To use a tool, respond with JSON:
{{"thought": "why I need this tool", "action": "tool_name", "args": {{"arg": "value"}}}}

When you have the final answer, respond with:
{{"thought": "I have the answer", "action": "final_answer", "answer": "complete answer"}}

Available tools:
{tool_desc}

Important:
- Always use a tool if you need external information
- Never make up information, use tools
- Keep going until you reach a final answer"""

    def run(self, user_goal: str) -> str:
        """Execute the agent loop."""
        self._history = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_goal},
        ]

        print(f"\n[Agent] Goal: {user_goal}")

        for iteration in range(self.MAX_ITERATIONS):
            response = self.llm.complete(self._history)
            self._history.append({"role": "assistant", "content": response})

            # Parse response
            try:
                parsed = self._parse_response(response)
            except Exception as e:
                print(f"[Agent] Parse error: {e}")
                break

            action = parsed.get("action", "")
            thought = parsed.get("thought", "")

            print(f"\n[Iteration {iteration + 1}]")
            print(f"  Thought: {thought[:100]}")

            # Final answer
            if action == "final_answer":
                answer = parsed.get("answer", "")
                print(f"  Final Answer: {answer[:100]}")
                return answer

            # Execute tool
            if action in self.tools:
                args = parsed.get("args", {})
                print(f"  Action: {action}({args})")

                try:
                    result = self.tools[action].function(**args)
                    print(f"  Observation: {str(result)[:100]}")
                except Exception as e:
                    result = f"Tool error: {e}"
                    print(f"  Error: {result}")

                self._history.append({
                    "role": "user",
                    "content": f"Observation from {action}: {result}",
                })
            else:
                print(f"  Unknown action: {action}")
                self._history.append({
                    "role": "user",
                    "content": f"Error: Unknown tool '{action}'. Available: {list(self.tools.keys())}",
                })

        return "Agent did not reach a conclusion within the iteration limit."

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from agent response."""
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        return json.loads(response)
```

---

## 4. Tool Implementations

```python
import math
import json
from datetime import datetime

def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Restrict to safe operations
        allowed = set("0123456789+-*/().,% ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # Safe after character check
        return str(round(result, 4))
    except Exception as e:
        return f"Math error: {e}"

def get_current_time() -> str:
    """Get current date and time."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def text_search(query: str) -> str:
    """Simple keyword search over a local knowledge base."""
    knowledge = {
        "python": "Python is a high-level programming language known for readability.",
        "rag": "RAG (Retrieval-Augmented Generation) reduces LLM hallucination.",
        "transformer": "Transformers use self-attention. Introduced in 'Attention Is All You Need'.",
    }
    query_lower = query.lower()
    results = [v for k, v in knowledge.items() if k in query_lower]
    return results[0] if results else "No information found for that query."

# Create tool objects
DEMO_TOOLS = [
    Tool("calculator", "Evaluate math expressions", calculator,
         {"expression": "math expression to evaluate"}),
    Tool("get_time", "Get current UTC time", lambda: get_current_time(), {}),
    Tool("search", "Search local knowledge base", text_search,
         {"query": "search query"}),
]
```

---

## 5. Agent Memory

```python
from dataclasses import dataclass, field

@dataclass
class AgentMemory:
    """Short-term and long-term memory for agents."""

    # Short-term: current conversation
    conversation_history: list[dict] = field(default_factory=list)
    max_history: int = 20

    # Working memory: key facts from this session
    working_memory: dict[str, str] = field(default_factory=dict)

    def add_turn(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        # Trim to max
        if len(self.conversation_history) > self.max_history:
            # Keep system prompt, remove oldest turns
            self.conversation_history = (
                self.conversation_history[:1] +
                self.conversation_history[-self.max_history + 1:]
            )

    def remember(self, key: str, value: str) -> None:
        """Store a fact in working memory."""
        self.working_memory[key] = value

    def recall(self, key: str) -> str | None:
        return self.working_memory.get(key)

    def as_context_string(self) -> str:
        if not self.working_memory:
            return ""
        items = "\n".join(f"- {k}: {v}" for k, v in self.working_memory.items())
        return f"Known facts:\n{items}"
```

---

## 6. Agent vs Workflow

```
WORKFLOW: Predefined steps, always the same path
  Document → Extract → Classify → Store
  Fast, predictable, testable, debuggable

AGENT: Dynamic steps, chooses its own path
  Goal → (searches? calculates? looks up? combines?)
  Flexible, handles unexpected cases

USE WORKFLOW WHEN:
  - Steps are known in advance
  - Reliability is critical
  - Cost and latency must be predictable
  - Process is repeatable

USE AGENT WHEN:
  - Path to answer is unknown
  - Different queries need different tools
  - Multi-step reasoning required
  - Flexibility > reliability

ANTI-PATTERN: Using an agent when a RAG pipeline would do.
A RAG pipeline with hybrid search handles 80% of Q&A use cases.
Agents add complexity, latency, cost, and failure modes.
```
