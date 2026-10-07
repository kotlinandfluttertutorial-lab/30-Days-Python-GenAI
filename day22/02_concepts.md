# Day 22 — Concepts
## RAG Evaluation

---

## 1. Why RAG Evaluation Is Hard

```
Traditional ML: Compare prediction to ground truth label
  accuracy = (correct predictions) / (total predictions)

RAG: Text output has no single correct answer
  "What is RAG?" → many valid answers of different quality

We need to measure:
1. Did we retrieve the RIGHT documents? (retrieval quality)
2. Did the LLM answer FAITHFULLY from those documents? (faithfulness)
3. Is the answer RELEVANT to the question? (answer relevance)
4. Is the answer factually CORRECT? (answer correctness)
```

---

## 2. Key RAG Metrics

```
RETRIEVAL METRICS:
  Context Precision:   % of retrieved chunks that are relevant
                       = relevant_retrieved / total_retrieved
  Context Recall:      % of relevant chunks that were retrieved
                       = relevant_retrieved / total_relevant

GENERATION METRICS:
  Faithfulness:        Is the answer grounded in retrieved context?
                       (No hallucinated claims)
  Answer Relevance:    Does the answer address the question?
  Answer Correctness:  Is the answer factually correct?
                       (Requires reference answer)
  Context Relevance:   Is the retrieved context relevant to the question?

COMPOSITE METRICS:
  RAGAS Score = (faithfulness + answer_relevance + context_precision + context_recall) / 4
```

---

## 3. RAGAS Framework

```python
# RAGAS: Retrieval Augmented Generation Assessment
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is RAG?", "How does chunking work?"],
    "answer": [generated_answers[0], generated_answers[1]],
    "contexts": [[chunks_used[0]], [chunks_used[1]]],
    "ground_truth": ["RAG combines retrieval...", "Chunking splits..."],
}

dataset = Dataset.from_dict(eval_data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)  # DataFrame with per-question scores
```

---

## 4. Custom Evaluation (No RAGAS)

```python
from pydantic import BaseModel

class FaithfulnessResult(BaseModel):
    is_faithful: bool
    score: float           # 0.0-1.0
    unsupported_claims: list[str]  # Claims not in context
    reasoning: str

def evaluate_faithfulness(
    answer: str,
    context: str,
    llm_client,
) -> FaithfulnessResult:
    """
    Use LLM to evaluate if the answer is faithful to the context.
    LLM-as-judge: using a strong LLM to evaluate another LLM's output.
    """
    prompt = f"""Evaluate if the following answer is faithful to the context.
An answer is faithful if every claim in it can be directly supported by the context.

Context:
{context}

Answer:
{answer}

Return JSON:
{{
  "is_faithful": true/false,
  "score": 0.0-1.0,
  "unsupported_claims": ["list claims not in context"],
  "reasoning": "brief explanation"
}}"""

    raw = llm_client.complete(prompt)
    data = parse_json(raw)
    return FaithfulnessResult(**data)


def evaluate_answer_relevance(
    question: str,
    answer: str,
    embedder,
) -> float:
    """
    Embedding-based answer relevance.
    If the answer is on-topic, it should be semantically similar to the question.
    """
    import numpy as np
    q_emb = np.array(embedder.embed(question))
    a_emb = np.array(embedder.embed(answer))
    return float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb)))


def evaluate_context_precision(
    question: str,
    contexts: list[str],
    ground_truth: str,
    embedder,
) -> float:
    """
    What fraction of retrieved contexts are relevant?
    Proxy: cosine similarity between context and ground truth.
    """
    import numpy as np
    if not contexts:
        return 0.0

    gt_emb = np.array(embedder.embed(ground_truth))
    relevant = 0
    for ctx in contexts:
        ctx_emb = np.array(embedder.embed(ctx))
        sim = np.dot(gt_emb, ctx_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(ctx_emb))
        if sim > 0.7:  # Threshold for "relevant"
            relevant += 1

    return relevant / len(contexts)
```

---

## 5. Evaluation Dataset Creation

```python
# Good evaluation dataset properties:
# 1. Diverse question types (factual, multi-hop, comparison)
# 2. Multiple difficulty levels
# 3. Known ground truth answers
# 4. Questions with answers in docs AND questions without

def create_eval_dataset_from_docs(
    documents: list[str],
    llm_client,
    n_questions: int = 20,
) -> list[dict]:
    """
    Auto-generate QA pairs from documents.
    Used when you don't have labeled test data.
    """
    questions = []
    for doc in documents[:n_questions // 2]:
        prompt = f"""Based on this text, generate one factual question and its answer.
Return JSON: {{"question": "...", "answer": "...", "source": "first 50 chars of text"}}

Text: {doc[:500]}"""
        raw = llm_client.complete(prompt)
        try:
            qa = json.loads(clean_json(raw))
            questions.append(qa)
        except Exception:
            pass
    return questions
```

---

## 6. Evaluation Pipeline

```python
def run_evaluation_pipeline(
    rag_system,
    eval_dataset: list[dict],
    embedder,
    llm_client,
) -> dict:
    """Full evaluation run."""
    faithfulness_scores = []
    relevance_scores = []
    context_precision_scores = []

    for item in eval_dataset:
        result = rag_system.answer(item["question"])
        answer = result["answer"]
        context = " ".join(result.get("contexts", []))

        # Metrics
        faith = evaluate_faithfulness(answer, context, llm_client)
        relevance = evaluate_answer_relevance(item["question"], answer, embedder)
        precision = evaluate_context_precision(
            item["question"], result.get("contexts", []),
            item.get("ground_truth", ""), embedder
        )

        faithfulness_scores.append(faith.score)
        relevance_scores.append(relevance)
        context_precision_scores.append(precision)

    return {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "answer_relevance": sum(relevance_scores) / len(relevance_scores),
        "context_precision": sum(context_precision_scores) / len(context_precision_scores),
        "n_evaluated": len(eval_dataset),
    }
```
