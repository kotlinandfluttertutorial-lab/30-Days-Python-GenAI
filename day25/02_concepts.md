# Day 25 — Concepts
## Advanced Agents + MCP

---

## 1. Multi-Agent Systems

```
SINGLE AGENT:
  One LLM → decides everything → one context window → one failure point

MULTI-AGENT:
  Orchestrator agent → delegates to specialist agents
  Benefits: parallelism, specialization, larger effective context
  Risks: more failure points, harder to debug, higher latency + cost

PATTERNS:
  1. Sequential: A → B → C (pipeline)
  2. Parallel: A calls B and C simultaneously
  3. Hierarchical: Manager → Worker agents
  4. Peer: Agents coordinate as equals
```

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    message_type: str = "request"  # request | response | error

class OrchestratorAgent:
    """
    Manages specialist sub-agents.
    Decomposes tasks, delegates, aggregates results.
    """

    def __init__(self, llm, specialists: dict[str, Any]) -> None:
        self.llm = llm
        self.specialists = specialists

    def decompose_task(self, goal: str) -> list[dict]:
        """Break complex goal into sub-tasks for specialists."""
        prompt = f"""Break this task into sub-tasks. Return JSON array.
Each item: {{"task": "...", "agent": "researcher|analyst|writer", "depends_on": []}}

Task: {goal}"""
        raw = self.llm.complete([{"role": "user", "content": prompt}])
        import json
        return json.loads(raw.strip())

    def run(self, goal: str) -> str:
        """Orchestrate multiple specialists to complete a goal."""
        sub_tasks = self.decompose_task(goal)
        results: dict[str, str] = {}

        for task in sub_tasks:
            agent_name = task["agent"]
            if agent_name in self.specialists:
                result = self.specialists[agent_name].run(task["task"])
                results[task["task"]] = result

        # Aggregate
        context = "\n\n".join(f"[{k}]: {v}" for k, v in results.items())
        return self.llm.complete([
            {"role": "user", "content": f"Synthesize these findings:\n{context}\n\nGoal: {goal}"}
        ])
```

---

## 2. Model Context Protocol (MCP)

```
MCP = Model Context Protocol
Introduced by Anthropic (2024)

Purpose: Standardize how LLMs connect to external tools and data sources.

BEFORE MCP:
  Every AI app had custom tool implementations
  OpenAI format ≠ Anthropic format ≠ custom agents
  Tool implementations: copy-paste across projects

WITH MCP:
  Standardized protocol: LLM ↔ MCP Server
  MCP servers expose: Tools, Resources, Prompts
  Any MCP-compatible client can use any MCP server
```

---

## 3. MCP Architecture

```
MCP CLIENT (your AI app)
    ↕ JSON-RPC 2.0 over stdio/HTTP/SSE
MCP SERVER (tool provider)
    ├── TOOLS:     Functions the LLM can call
    ├── RESOURCES: Data the LLM can read (files, DB rows, APIs)
    └── PROMPTS:   Templated prompt sequences

EXAMPLE MCP SERVERS:
  - filesystem: read/write local files
  - postgres:   run SQL queries
  - web search: search the web
  - github:     interact with repositories
  - custom:     anything you build
```

```python
# MCP Server implementation (conceptual)
from typing import Any

class MCPServer:
    """Simple MCP server exposing tools and resources."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, dict] = {}
        self._tool_functions: dict[str, Any] = {}

    def tool(self, name: str, description: str, parameters: dict):
        """Decorator to register a tool."""
        def decorator(func):
            self._tools[name] = {
                "name": name,
                "description": description,
                "inputSchema": parameters,
            }
            self._tool_functions[name] = func
            return func
        return decorator

    def list_tools(self) -> list[dict]:
        return list(self._tools.values())

    def call_tool(self, name: str, arguments: dict) -> Any:
        if name not in self._tool_functions:
            raise ValueError(f"Tool '{name}' not found")
        return self._tool_functions[name](**arguments)


# Create an MCP server
server = MCPServer("ai-tools")

@server.tool(
    name="calculate",
    description="Evaluate math expressions",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }
)
def calculate(expression: str) -> str:
    return str(eval(expression))

@server.tool(
    name="get_file_content",
    description="Read content of a text file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"],
    }
)
def get_file_content(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

# MCP server serves these via JSON-RPC
# Any MCP-compatible client (Claude Desktop, custom app) can use them
```

---

## 4. Agent Security

```python
import re
from typing import Any

class AgentSecurityLayer:
    """
    Security controls for AI agents.
    Applied before tool execution and after LLM generation.
    """

    # Injection patterns in tool results
    INJECTION_PATTERNS = [
        r"ignore.{0,20}(previous|all|system|instructions)",
        r"you are now",
        r"new instructions",
        r"disregard",
        r"<\/?system>",
        r"###\s*(instruction|system|override)",
    ]

    # Tools that require human confirmation
    SENSITIVE_TOOLS = {
        "delete_file", "send_email", "execute_code",
        "make_payment", "modify_database", "deploy_app",
    }

    def scan_tool_result(self, tool_name: str, result: str) -> str:
        """Scan tool results for injection attempts."""
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, result.lower()):
                return f"[SECURITY: Suspicious content in {tool_name} result was filtered]"
        return result

    def check_tool_permission(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check if a tool call is permitted."""
        # Sensitive tools require confirmation
        if tool_name in self.SENSITIVE_TOOLS:
            return False, f"Tool '{tool_name}' requires human approval"

        # Check for path traversal in file tools
        if "path" in args:
            path = args["path"]
            if ".." in path or path.startswith("/etc") or path.startswith("/root"):
                return False, "Path traversal attempt blocked"

        return True, "Permitted"

    def sanitize_tool_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Sanitize arguments before tool execution."""
        safe_args = {}
        for k, v in args.items():
            if isinstance(v, str):
                # Remove null bytes and control characters
                v = v.replace("\x00", "").replace("\r", "")
                # Limit length
                v = v[:10000]
            safe_args[k] = v
        return safe_args
```

---

## 5. Agent Failure Modes and Defenses

```
1. INFINITE LOOP
   Cause: agent keeps calling tools without converging
   Defense: max_iterations, force final_answer after N steps

2. CONTEXT OVERFLOW
   Cause: many tool calls fill up context window
   Defense: summarize history periodically, limit tool result length

3. PROMPT INJECTION
   Cause: malicious content in tool results hijacks agent
   Defense: scan all tool results, structural separation

4. RESOURCE EXHAUSTION
   Cause: agent calls expensive tools repeatedly
   Defense: tool call budget, rate limiting per agent run

5. HALLUCINATED TOOL CALLS
   Cause: agent calls tools with wrong/invalid args
   Defense: Pydantic validation of tool args before execution

6. SENSITIVE DATA EXPOSURE
   Cause: agent passes PII to external tools
   Defense: PII detection before external tool calls
```
