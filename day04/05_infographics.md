# Day 04 — Infographics

---

## Infographic 1: Cosine Similarity Geometry

```
2D VECTOR SPACE:

Y
│         b=[0.5, 0.9]
│        /
│       /  ← small angle (cos ≈ 0.98 = very similar)
│      /
│     /
│    / ← a=[0.6, 0.8]
│   /
└──────────────── X

c=[−0.8, 0.2] ← large angle with a (cos ≈ −0.4 = different direction)

COSINE SIMILARITY = cos(θ) = dot(a,b) / (|a| × |b|)
                  = measure of ANGLE, not LENGTH

WHY THIS MATTERS:
  short doc: "Python is cool" → small magnitude vector
  long doc: "Python is cool, Python has libraries..." → large magnitude
  Both point in same DIRECTION → high cosine similarity ✓
  Euclidean distance would say they're far apart (wrong!)
```

---

## Infographic 2: Probability as Area

```
P(SPAM) = 0.30  ──────────────────────────────
                ████████████████████████████████████████████████████████████
                100% of emails
                ←──30%──→←─────────70%─────────→
                  SPAM              HAM

P(contains "FREE" | SPAM) = 0.80   (80% of spam contains "FREE")
P(contains "FREE" | HAM)  = 0.05   (5% of ham contains "FREE")

P(SPAM | contains "FREE") = P("FREE"|SPAM) × P(SPAM) / P("FREE")
                           = 0.80 × 0.30 / (0.80×0.30 + 0.05×0.70)
                           = 0.24 / (0.24 + 0.035)
                           = 0.87  ← 87% probability it's spam!
```

---

## Infographic 3: Softmax Temperature Effect

```
LOGITS: token_a=5.0, token_b=3.0, token_c=1.0

Temperature = 0.1 (cold/deterministic):
  token_a: ████████████████████████████████████████ 99.99%
  token_b: ░ 0.01%
  token_c: ░ 0.00%

Temperature = 1.0 (balanced):
  token_a: ████████████████████░░░░░░░░░░░░░░░░░░░░ 84.4%
  token_b: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░ 11.4%
  token_c: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  4.2%

Temperature = 2.0 (hot/creative):
  token_a: ████████████████████████░░░░░░░░░░░░░░░░ 59%
  token_b: █████████████████████░░░░░░░░░░░░░░░░░░░ 31%
  token_c: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 10%
```
