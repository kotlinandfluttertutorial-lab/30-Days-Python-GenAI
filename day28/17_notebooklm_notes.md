# Day 28 — NotebookLM Notes: LLMOps + Security + Evaluation

## LLMOps Core Concerns

| Concern | What to Track | Why |
|---------|--------------|-----|
| Latency | p50/p95/p99 ms | SLA compliance |
| Cost | tokens × price per day | Budget control |
| Quality | eval scores | Detect regressions |
| Errors | rate by type | Reliability |
| Cache | hit rate | Cost optimization |
| Hallucinations | rate from evals | Trust + safety |

## Security Checklist

```
INPUT:
  [ ] Scan for injection patterns (regex + blocklist)
  [ ] PII detection before sending to external LLMs
  [ ] Rate limiting per user/API key
  [ ] Input length limits

PROCESSING:
  [ ] Structural separation: <system> vs <user> tags
  [ ] Tool result scanning (indirect injection)
  [ ] No secrets in prompts

OUTPUT:
  [ ] Guardrails: PII, toxicity, prompt leakage
  [ ] Length constraints
  [ ] Format validation (JSON if expected)
```

## PII Types to Detect

- Email addresses (regex: `\b..@..\b`)
- Phone numbers (US: `\d{3}-\d{3}-\d{4}`)
- SSN (`\d{3}-\d{2}-\d{4}`)
- Credit card numbers
- IP addresses
- Names + DOB combinations

## Prompt Injection Defense

```
1. Regex scanning: block known patterns
2. Structural separation: <system>...</system><user>...</user>
3. Output classification: post-process with classifier
4. Rate limit injection attempts
5. Log and alert on detected attempts
```

## Interview Facts

1. Log prompt PREVIEW (100 chars), never full prompt — avoids PII in logs
2. p95 latency more useful than average (captures tail latency)
3. Daily cost alert at 80% budget → gives time to investigate before hitting limit
4. Guardrails should be BOTH on input AND output
5. Indirect injection: malicious content in retrieved docs hijacks system prompt
6. Never log user messages that might contain PII
7. LLM eval regression: track eval score over time; alert on drops

## Common Mistakes

- Logging full prompts (PII exposure)
- No daily cost budget → surprise $10K bill
- Guardrails only on input, not output
- No p95/p99 tracking → don't know about slow tail
- Same model for everything → overpaying for simple tasks
- No eval baseline → don't know if prompt changes broke things
