# Day 23 — NotebookLM Notes: AI Agents

## Agent = LLM + Tools + Memory + Planning

**ReAct Loop:**
```
Think (LLM picks action) → Act (execute tool) → Observe (result)
→ Think (plan next) → Act → Observe → ... → Final Answer
```

## Implementation Essentials

```python
# System prompt includes:
# 1. Available tools with descriptions
# 2. Required JSON response format
# 3. MAX_ITERATIONS constraint

# Response format:
{"thought": "reasoning", "action": "tool_name", "args": {"key": "value"}}
# Or final answer:
{"thought": "done", "action": "final_answer", "answer": "..."}
```

## Agent Failure Modes

1. **Infinite loop**: keeps searching without concluding → MAX_ITERATIONS + force final_answer
2. **Prompt injection**: tool result contains "ignore previous instructions" → sanitize tool results
3. **Context overflow**: many tool calls fill context → summarize history periodically
4. **JSON parse error**: LLM doesn't format correctly → re-prompt with parse error message
5. **Tool selection error**: picks wrong tool → clear tool descriptions + few-shot examples
6. **Hallucinated tool calls**: invents a tool that doesn't exist → handle unknown tool gracefully

## Agent vs RAG vs Workflow

```
RAG:      fixed pipeline, one retrieval, one generation — simple, fast, reliable
Workflow: predefined steps — deterministic, testable, debuggable
Agent:    dynamic path, chooses tools — flexible, handles unknown cases

Use Agent ONLY when:
- Path to answer is genuinely unknown
- Multiple different tools might be needed
- Multi-step reasoning with external data required
```

## Interview Facts

1. ReAct: Reasoning + Acting — interleaves thought and action in the same prompt
2. `max_iterations` prevents loops but choosing the right number is tricky (too low = fails, too high = expensive)
3. Agents multiply failure modes: each tool call adds a potential failure point
4. Memory types: short-term (conversation history), long-term (vector store of past sessions)
5. Tool description quality: most important factor in tool selection accuracy
6. Structured output (JSON): essential for reliable action parsing — plain text is fragile

## Common Mistakes

- No MAX_ITERATIONS limit → infinite loops, huge costs
- Tools with vague descriptions → LLM picks wrong tool
- No error handling on tool failures → agent gets stuck
- Using agents for simple Q&A that RAG handles better
- Forgetting to append tool results to conversation history
