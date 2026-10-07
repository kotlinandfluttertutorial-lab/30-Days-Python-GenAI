# Day 28 — Day Summary: LLMOps + Security + Evaluation

## What You Covered
Structured logging (LLMRequestLog), metrics with percentile tracking, prompt injection defense, PII detection/redaction, output guardrails, cost monitoring with budget alerts.

## Key Takeaways
1. Structured JSON logs: every LLM call logged with trace_id, latency, tokens, cost
2. p95/p99 latency: more important than average for SLA monitoring
3. PII: detect before sending to LLM, redact from logs
4. Guardrails: both input (injection defense) and output (PII, toxicity)
5. Daily cost budget: alert at 80% — don't wait for 100% (or 200%)

## Phase 10 Half-Complete (Days 28-30)
Tomorrow: the biggest day — Enterprise AI System Design + full major project.
