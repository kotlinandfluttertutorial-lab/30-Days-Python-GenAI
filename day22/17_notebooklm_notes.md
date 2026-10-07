# Day 22 — NotebookLM Notes: RAG Evaluation

## Core Metrics

| Metric | What It Measures | Range | How Computed |
|--------|-----------------|-------|-------------|
| Faithfulness | Answer grounded in context? | 0-1 | LLM-as-judge |
| Answer Relevance | Does answer address question? | 0-1 | Embedding similarity |
| Context Precision | Fraction of retrieved chunks relevant | 0-1 | LLM or embedding |
| Context Recall | Fraction of relevant chunks retrieved | 0-1 | LLM with reference |
| Answer Correctness | Factually accurate? | 0-1 | LLM with reference answer |

## RAGAS

Open-source framework for RAG evaluation. Uses LLMs to judge outputs.
Key input: question, answer, contexts, ground_truth (optional for some metrics).

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

## LLM-as-Judge

Use a strong LLM (e.g., GPT-4o) to evaluate outputs from a weaker LLM.
Prompt includes: question, context, answer → judge returns score + reasoning.
Advantage: flexible, doesn't need reference answers.
Risk: judge LLM can also be wrong (needs calibration against human labels).

## Evaluation Dataset

- Minimum: 50-100 questions for meaningful results
- Include: easy, medium, hard questions
- Include: questions NOT in the knowledge base (to test proper "I don't know" behavior)
- Auto-generate with LLM from source documents (if no labeled data)

## Interview Facts

1. Faithfulness > answer relevance in priority: wrong answers are worse than off-topic
2. Context precision: low = retrieved irrelevant chunks = wastes context window
3. Context recall: low = missed relevant docs = answer can't be complete
4. LLM-as-judge needs calibration: compare to human annotations on a sample
5. Evaluation should run in CI/CD: any model/prompt change triggers eval run
6. Track scores over time: detect regressions before users do

## Production Thresholds (Typical)

```
Faithfulness:      > 0.85 (below = hallucination problem)
Answer Relevance:  > 0.75 (below = off-topic answers)
Context Precision: > 0.70 (below = bad retrieval)
Context Recall:    > 0.80 (below = missing relevant docs)
```

## Common Mistakes

- Evaluating only on "easy" questions (inflated scores)
- Not including "unanswerable" questions in eval set
- Using RAGAS without human calibration on your domain
- Running eval only once (need continuous monitoring)
- Ignoring individual failed cases (they reveal specific failure modes)
