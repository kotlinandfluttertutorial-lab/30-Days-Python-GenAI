# Day 25 — NotebookLM Notes: Advanced Agents + MCP

## Multi-Agent Patterns

```
Sequential:    A → B → C  (pipeline, each depends on previous)
Parallel:      A calls B and C simultaneously (asyncio.gather)
Hierarchical:  Orchestrator → Specialist agents
Peer:          Agents communicate directly (uncommon, complex)
```

## MCP (Model Context Protocol)

**What**: Standard protocol for LLM ↔ tool/data connections.
**By**: Anthropic (2024), now open standard.
**Transports**: stdio (local), HTTP+SSE (remote).
**Components**: Tools (callable), Resources (readable data), Prompts (templates).

```
Your App → MCP Client → [stdio/HTTP] → MCP Server
                                         ├── Tools
                                         ├── Resources
                                         └── Prompts
```

**Why it matters**: Any MCP server works with any MCP-compatible client.
Reuse community MCP servers (filesystem, postgres, github, etc.) in your AI app.

## Agent Security Checklist

```
Input:
  [ ] Scan user input for injection patterns
  [ ] Validate input length and format

Tool args:
  [ ] Pydantic validation before execution
  [ ] Path traversal check for file tools
  [ ] SQL injection check for DB tools
  [ ] PII detection before external tools

Tool results:
  [ ] Scan for injection patterns in results
  [ ] Truncate excessively long results
  [ ] Log all tool calls for audit

Agent loop:
  [ ] MAX_ITERATIONS limit
  [ ] Tool call budget per run
  [ ] Sensitive tools require human-in-the-loop
  [ ] Timeout per tool call
```

## Interview Facts

1. MCP separates tool DEFINITION (MCP server) from tool USE (AI app)
2. Orchestrator pattern: manager LLM delegates to specialist LLMs
3. Human-in-the-loop: agents pause and request approval for sensitive actions
4. Indirect prompt injection: malicious content in retrieved data hijacks agent
5. Context overflow mitigation: summarize conversation history after N turns
6. Tool call budget: limit total tool calls per agent run to control cost

## Common Mistakes

- No rate limiting on agent tool calls → cost explosion
- Not scanning tool results for injection → indirect injection attack
- Sensitive operations (delete, deploy, email) without human confirmation
- No timeout on individual tool calls → agent hangs
- Multi-agent without clear ownership of state → race conditions
