# Day 24 — NotebookLM Notes: Tool Calling

## Native Tool Calling vs Manual JSON Parsing

| Aspect | Native Tool Calling | Manual JSON |
|--------|-------------------|-------------|
| Structure | API guarantees | Fragile parsing |
| Parallel calls | Native support | Complex to implement |
| Error signaling | finish_reason field | Parse errors |
| Tool result role | "tool" role | Stuffed in "user" |
| Reliability | High | Lower |

## OpenAI Tool Call Flow

```python
# Step 1: Send with tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tool_schemas,
    tool_choice="auto",
)

# Step 2: Check for tool calls
if response.choices[0].finish_reason == "tool_calls":
    for tc in response.choices[0].message.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        result = execute_tool(name, args)
        # Append as "tool" role
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# Step 3: Get final answer
final = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tool_schemas)
```

## Tool Schema Best Practices

1. `description`: explain exactly when to use it, and when NOT to
2. `required`: only truly required params; use defaults for optional
3. `enum`: for constrained string values (categories, modes)
4. Parameter descriptions: include examples ("e.g., 'Paris, France'")
5. Separate tools with overlapping purpose → clear descriptions on which to use

## Interview Facts

1. `tool_choice="auto"`: LLM decides whether to use tools
2. `tool_choice="required"`: must call at least one tool
3. `tool_choice={"type": "function", "function": {"name": "x"}}`: force specific tool
4. Parallel tool calls: one API call → multiple tool calls → execute all → continue
5. Tool errors should be sent back as tool result — LLM can recover
6. `finish_reason="stop"`: direct answer, no tool call needed

## Common Mistakes

- Returning exception stack trace as tool result → confuses LLM
- Not appending assistant message before tool results
- Using `tool_choice="required"` when the LLM should be able to answer directly
- Forgetting `tool_call_id` in tool result → API error
- Tool descriptions too vague → wrong tool selected
