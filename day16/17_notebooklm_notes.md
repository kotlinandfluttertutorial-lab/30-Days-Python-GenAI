# Day 16 — NotebookLM Notes: Prompt Engineering

## Prompting Hierarchy (least to most effort)

1. Zero-shot → just instructions
2. Few-shot → add 2-5 examples
3. Chain-of-thought → "think step by step"
4. Structured output → JSON schema
5. System prompt → persistent behavior rules

## Key Patterns

```python
# Zero-shot
"Classify as POSITIVE/NEGATIVE/NEUTRAL: {text}"

# Few-shot  
"Examples: ... → POSITIVE | ... → NEGATIVE | Text: {text}"

# CoT
"Solve step by step. Problem: {problem} Let's think step by step:"

# JSON output
"Return ONLY JSON: {schema}. Text: {text}"

# RAG
"Answer ONLY from context. Context: {context}. Question: {question}"
```

## System Prompt Structure

```
1. ROLE: "You are a [expert/assistant/analyst]"
2. CAPABILITIES: "You can [list what it should do]"
3. CONSTRAINTS: "Never [what it must not do]"
4. FORMAT: "Always respond with [format rules]"
```

## Prompt Injection

```
Attack: user input that overrides system prompt
Direct: "Ignore all previous instructions..."
Indirect: malicious text in retrieved documents

Defenses:
1. Input validation (regex for known patterns)
2. Structural separation (<system>...</system> <user>...</user>)
3. Separate classification call: is this input legitimate?
4. Output validation: does output match expected format?
```

## Interview Facts

1. "The prompt is code" — version it, test it, review it
2. Few-shot examples must match the desired output format exactly
3. CoT improves accuracy on reasoning but uses more tokens
4. `response_format={"type": "json_object"}` = guaranteed valid JSON (OpenAI)
5. Indirect injection via RAG context is harder to defend than direct injection
6. System prompt tokens count every single request → keep it short!
7. Prompt regression: changing a prompt can break previously working queries

## Common Mistakes

- Not stripping markdown fences before JSON.loads()
- Not testing prompts on edge cases before production
- Overly long system prompts (cost + context issues)
- Not versioning prompts — no rollback when something breaks
- Trusting all retrieved context without injection defense
