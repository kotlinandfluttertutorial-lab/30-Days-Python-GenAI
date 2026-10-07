# Day 24 — Day Summary: Tool Calling

## What You Built
Tool-using agent with native OpenAI/Groq function calling: tool schemas, parallel execution, error handling, mock fallback for no-API-key testing.

## Key Takeaways
1. Native tool calling is more reliable than parsing JSON from free text
2. `finish_reason="tool_calls"` tells you the LLM wants to call a tool
3. Tool results go back as `"role": "tool"` with `tool_call_id`
4. Error handling: send error message back so LLM can recover
5. Tool descriptions are the most important factor in correct tool selection

## Tomorrow: Day 25 — Advanced Agents + MCP
Multi-agent systems, agent security, Model Context Protocol (MCP) — the emerging standard for tool/resource communication.
