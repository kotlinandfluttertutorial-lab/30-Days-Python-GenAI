# Day 24 — Concepts
## Tool Calling (Function Calling)

---

## 1. What Is Tool Calling?

```
BEFORE tool calling (Day 23 approach):
  LLM outputs JSON in free text → you parse it → risky

WITH tool calling (native API feature):
  You define tool schemas → LLM decides when to call → API guarantees structure
  → Much more reliable, no fragile JSON parsing

Supported by:
  OpenAI:    "tools" parameter in chat.completions.create()
  Anthropic: "tools" parameter in messages.create()
  Groq:      Same as OpenAI API
```

---

## 2. OpenAI Tool Calling

```python
from openai import OpenAI
import json

client = OpenAI()

# Define tools with JSON Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris, France'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression like '2 + 2' or '100 * 0.15'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

messages = [{"role": "user", "content": "What is 15% of 85.50?"}]

# Step 1: LLM decides to call a tool
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # "auto", "required", "none", or specific tool
)

choice = response.choices[0]
print(f"Finish reason: {choice.finish_reason}")  # "tool_calls" when tool needed

# Step 2: Execute the tool
if choice.finish_reason == "tool_calls":
    tool_calls = choice.message.tool_calls

    # Append assistant's decision to messages
    messages.append(choice.message)

    for tool_call in tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        print(f"Calling: {func_name}({func_args})")

        # Execute
        if func_name == "calculator":
            result = str(eval(func_args["expression"]))
        elif func_name == "get_weather":
            result = f"72°F, sunny in {func_args['location']}"

        # Append tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

# Step 3: LLM generates final response using tool results
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)
print(final_response.choices[0].message.content)
# → "15% of $85.50 is $12.83"
```

---

## 3. Anthropic Tool Calling

```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [
    {
        "name": "search_database",
        "description": "Search the company database for customer information",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer ID"},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to retrieve",
                },
            },
            "required": ["customer_id"],
        },
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Get name and email for customer C123"}],
)

# Handle tool use
for content in response.content:
    if content.type == "tool_use":
        tool_name = content.name
        tool_input = content.input
        print(f"Claude wants to call: {tool_name}({tool_input})")

        # Execute tool
        result = execute_tool(tool_name, tool_input)

        # Continue conversation with tool result
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": "Get name and email for customer C123"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    }],
                },
            ],
        )
```

---

## 4. Parallel Tool Calls

```python
# OpenAI supports calling multiple tools in one step
# Example: "What's the weather in Paris AND London?"
# → LLM makes two simultaneous tool calls

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris and London?"}],
    tools=tools,
)

# Multiple tool calls in one response
for tool_call in response.choices[0].message.tool_calls:
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"Tool: {func_name}, Args: {args}")
    # Execute each tool, append all results
```

---

## 5. Tool Schema Best Practices

```python
# GOOD tool schema:
{
    "name": "search_products",
    "description": (
        "Search the product catalog by keyword. "
        "Use when user asks about product availability, specs, or pricing. "
        "Do NOT use for order status — use get_order_status instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Product search terms (e.g., 'wireless headphones blue')",
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "furniture", "all"],
                "description": "Product category to filter by. Use 'all' if unsure.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-20)",
                "default": 5,
            },
        },
        "required": ["query"],  # Only query is required; others have defaults
    },
}

# BAD: too vague
{
    "name": "search",
    "description": "Search for things",
    "parameters": {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
}
```

---

## 6. Error Handling in Tool Execution

```python
from pydantic import BaseModel, ValidationError
from typing import Any

class ToolResult(BaseModel):
    success: bool
    output: str
    error: str | None = None

def safe_execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_registry: dict[str, callable],
) -> ToolResult:
    """Execute a tool safely with full error handling."""
    if tool_name not in tool_registry:
        return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

    func = tool_registry[tool_name]

    try:
        result = func(**tool_args)
        return ToolResult(success=True, output=str(result))

    except TypeError as e:
        return ToolResult(
            success=False, output="",
            error=f"Wrong arguments for {tool_name}: {e}"
        )
    except TimeoutError:
        return ToolResult(
            success=False, output="",
            error=f"Tool {tool_name} timed out"
        )
    except Exception as e:
        return ToolResult(
            success=False, output="",
            error=f"Tool {tool_name} failed: {type(e).__name__}: {e}"
        )

# When a tool fails: send the error back to the LLM
# The LLM can then decide to retry, use a different tool, or apologize
tool_result = safe_execute_tool(tool_name, args, tools)
if not tool_result.success:
    # Tell LLM the tool failed so it can recover
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": f"ERROR: {tool_result.error}. Please try a different approach.",
    })
```
