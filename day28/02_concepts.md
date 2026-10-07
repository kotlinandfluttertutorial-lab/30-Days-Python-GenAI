# Day 28 — Concepts
## LLMOps + Security + Evaluation

---

## 1. Structured Logging for LLM Applications

```python
import logging
import json
import uuid
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

@dataclass
class LLMRequestLog:
    """Structured log entry for every LLM interaction."""
    trace_id: str
    timestamp: str
    user_id: str | None
    model: str
    provider: str
    prompt_preview: str      # First 100 chars only (no full prompt in logs!)
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str
    cache_hit: bool = False
    eval_score: float | None = None
    hallucination_detected: bool = False
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

def log_llm_request(
    model: str,
    prompt: str,
    response: str,
    latency_ms: float,
    usage: dict,
    user_id: str | None = None,
) -> LLMRequestLog:
    """Create and log a structured LLM request record."""
    entry = LLMRequestLog(
        trace_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat(),
        user_id=user_id,
        model=model,
        provider="openai" if "gpt" in model else "anthropic",
        prompt_preview=prompt[:100],  # Never log full prompt
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        latency_ms=latency_ms,
        cost_usd=estimate_cost(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
        finish_reason=usage.get("finish_reason", "stop"),
    )
    logging.getLogger("llm").info(entry.to_json())
    return entry
```

---

## 2. Metrics and Monitoring

```python
# Key metrics for LLM applications:
# p50/p95/p99 latency
# Token usage per day/hour
# Cost per day/user
# Cache hit rate
# Error rate by provider
# Hallucination rate (from evals)
# User satisfaction (thumbs up/down)

# Simple in-memory metrics (use Prometheus in production)
from collections import defaultdict
from typing import Any
import time

class LLMMetrics:
    def __init__(self) -> None:
        self._latencies: list[float] = []
        self._costs: list[float] = []
        self._token_counts: list[int] = []
        self._errors: int = 0
        self._cache_hits: int = 0
        self._total_requests: int = 0

    def record(self, latency_ms: float, cost_usd: float, tokens: int, cached: bool = False) -> None:
        self._latencies.append(latency_ms)
        self._costs.append(cost_usd)
        self._token_counts.append(tokens)
        self._total_requests += 1
        if cached:
            self._cache_hits += 1

    def record_error(self) -> None:
        self._errors += 1

    def summary(self) -> dict[str, Any]:
        if not self._latencies:
            return {"requests": 0}
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        return {
            "total_requests": self._total_requests,
            "error_rate": self._errors / max(self._total_requests, 1),
            "cache_hit_rate": self._cache_hits / max(self._total_requests, 1),
            "p50_latency_ms": sorted_lat[n // 2],
            "p95_latency_ms": sorted_lat[int(n * 0.95)],
            "p99_latency_ms": sorted_lat[int(n * 0.99)],
            "total_cost_usd": sum(self._costs),
            "avg_cost_usd": sum(self._costs) / n,
            "total_tokens": sum(self._token_counts),
        }
```

---

## 3. Prompt Injection Defense (Production)

```python
import re
from enum import Enum

class ThreatLevel(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"

INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", ThreatLevel.BLOCKED),
    (r"you\s+are\s+now\s+a", ThreatLevel.BLOCKED),
    (r"forget\s+your\s+instructions", ThreatLevel.BLOCKED),
    (r"new\s+system\s+prompt", ThreatLevel.BLOCKED),
    (r"disregard\s+all", ThreatLevel.BLOCKED),
    (r"act\s+as\s+if\s+you", ThreatLevel.SUSPICIOUS),
    (r"pretend\s+you\s+are", ThreatLevel.SUSPICIOUS),
    (r"jailbreak", ThreatLevel.SUSPICIOUS),
]

def scan_input(text: str) -> tuple[ThreatLevel, str | None]:
    """Scan user input for injection patterns."""
    text_lower = text.lower()
    for pattern, level in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return level, pattern
    return ThreatLevel.SAFE, None

def safe_prompt(system_prompt: str, user_input: str) -> str:
    """Structurally separate system instructions from user input."""
    return f"""<system_instructions>
{system_prompt}
</system_instructions>

<user_message>
{user_input}
</user_message>

Respond to the user message following the system instructions above."""
```

---

## 4. PII Detection and Handling

```python
import re

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}

def detect_pii(text: str) -> dict[str, list[str]]:
    """Detect PII in text. Returns dict of type → [matches]."""
    found: dict[str, list[str]] = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[pii_type] = matches
    return found

def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Redact PII from text. Returns (redacted_text, counts)."""
    counts: dict[str, int] = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            replacement = f"[{pii_type.upper()}_REDACTED]"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            counts[pii_type] = len(matches)
    return text, counts
```

---

## 5. Guardrails

```python
from pydantic import BaseModel

class GuardrailResult(BaseModel):
    passed: bool
    violations: list[str]
    action: str  # "allow", "warn", "block"

def check_output_guardrails(response: str) -> GuardrailResult:
    """Check LLM output against safety rules."""
    violations = []

    # Check 1: Response shouldn't contain system prompt fragments
    if "system_instructions" in response.lower() or "you are an AI" in response.lower():
        violations.append("potential_prompt_leak")

    # Check 2: Length constraints
    if len(response) < 5:
        violations.append("response_too_short")
    if len(response) > 10000:
        violations.append("response_too_long")

    # Check 3: PII in output
    pii_found = detect_pii(response)
    if pii_found:
        violations.append(f"pii_in_output: {list(pii_found.keys())}")

    # Check 4: Toxic content (simplified)
    toxic_phrases = ["kill", "murder", "hack this", "illegal instructions"]
    for phrase in toxic_phrases:
        if phrase in response.lower():
            violations.append(f"toxic_content: {phrase}")

    if not violations:
        return GuardrailResult(passed=True, violations=[], action="allow")
    
    critical = ["pii_in_output", "toxic_content"]
    is_critical = any(any(c in v for c in critical) for v in violations)
    
    return GuardrailResult(
        passed=False,
        violations=violations,
        action="block" if is_critical else "warn",
    )
```

---

## 6. Cost Monitoring

```python
from datetime import date

class CostTracker:
    """Track LLM costs with budget alerts."""

    PRICING = {
        "gpt-4o":       {"input": 5.0,  "output": 15.0},
        "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "llama-3.1-70b": {"input": 0.59, "output": 0.79},
    }

    def __init__(self, daily_budget_usd: float = 100.0) -> None:
        self.daily_budget = daily_budget_usd
        self._daily_costs: dict[str, float] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(model, {"input": 5.0, "output": 15.0})
        cost = (input_tokens / 1e6) * pricing["input"] + (output_tokens / 1e6) * pricing["output"]
        today = str(date.today())
        self._daily_costs[today] = self._daily_costs.get(today, 0) + cost

        today_total = self._daily_costs[today]
        if today_total > self.daily_budget * 0.8:
            logging.getLogger("cost").warning(
                f"Cost alert: ${today_total:.2f} / ${self.daily_budget:.2f} daily budget"
            )
        return cost

    def today_cost(self) -> float:
        return self._daily_costs.get(str(date.today()), 0.0)
```
