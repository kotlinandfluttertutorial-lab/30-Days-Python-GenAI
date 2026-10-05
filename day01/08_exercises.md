# Day 01 — Exercises

---

## Theory Exercises (10 Questions)

**Instructions:** Answer from memory first, then check `02_concepts.md`.

**T1.** What is the difference between AI, Machine Learning, and Deep Learning? Give one example for each that the others cannot do.

**T2.** Explain what an LLM actually does at a technical level. Do not say "it's like ChatGPT." Explain the prediction mechanism.

**T3.** List three reasons why an LLM might hallucinate. For each reason, name one mitigation technique.

**T4.** Explain the RAG pipeline in sequence. What are the 5 steps from document to answer?

**T5.** What is the difference between an AI Agent and a RAG system? When would you use each?

**T6.** What is an embedding? Why is cosine similarity used instead of Euclidean distance?

**T7.** Explain what a "context window" is and why it matters for production AI systems. Give a concrete example of a system that would hit context window limits.

**T8.** Compare an AI Engineer and an ML Engineer. What does each primarily build? What's the key skill difference?

**T9.** List five ways AI software is fundamentally different from traditional deterministic software. For each difference, explain the practical implication for engineering.

**T10.** Explain what temperature does in LLM sampling. When would you use temperature=0.1? When would you use temperature=1.2?

---

## Coding Exercises (5 Questions)

**C1.** Implement `cosine_similarity(a: list[float], b: list[float]) -> float` from scratch using only Python (no NumPy). Test it with at least 3 pairs of vectors and verify the results make intuitive sense.

**C2.** Write a function `estimate_tokens(text: str) -> int` that estimates the token count of a string. Use the heuristic: 1 token ≈ 4 characters. Add a `cost_usd(tokens: int, price_per_1k: float) -> float` function. Test with a 1000-word text.

**C3.** Create a `ConceptCard` dataclass with fields: `name`, `category`, `definition`, `use_case`. Create 5 instances for AI concepts from today's material. Write a function that filters cards by category.

**C4.** Write a `SimpleRetriever` class that:
- Has a `documents: list[str]` attribute
- Has an `add_document(doc: str)` method
- Has a `search(query: str, top_k: int) -> list[str]` method using keyword overlap scoring
- Returns top-k most relevant documents

**C5.** Write a function `format_prompt(template: str, **kwargs: str) -> str` that safely formats a prompt template. It must:
- Replace `{variable}` placeholders
- Raise `ValueError` if a required variable is missing
- Return the formatted prompt

Test with:
```python
template = "Answer based on context:\n{context}\n\nQuestion: {question}"
```

---

## Debugging Exercises (5 Questions)

**D1.** Find and fix all bugs in this code:
```python
def cosine_similarity(a, b):
    dot = sum(a * b)  # Bug 1
    mag_a = sum(x**2 for x in a) ** 0.5
    mag_b = sum(x**2 for x in b) ** 0.5
    return dot / mag_a * mag_b  # Bug 2
```

**D2.** This code crashes. Find the root cause and fix it:
```python
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_client():
    api_key = os.getenv("OPENAI_API_KEY")
    from openai import OpenAI
    return OpenAI(api_key=api_key)  # Bug: what if key is None?

client = get_api_client()
```

**D3.** This function has a logic bug — it returns wrong results for some inputs. Find and fix it:
```python
def find_most_similar(query: list[float], 
                       documents: list[list[float]]) -> int:
    """Returns index of most similar document."""
    best_score = 0  # Bug: what if all similarities are negative?
    best_idx = 0
    
    for i, doc in enumerate(documents):
        score = cosine_similarity(query, doc)
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx
```

**D4.** This code has a dangerous security issue. Identify it and fix it:
```python
import subprocess

def get_model_info(model_name: str) -> str:
    result = subprocess.run(
        f"ollama show {model_name}",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout
```

**D5.** This code has a performance issue that would be catastrophic in production. Find it:
```python
def search_documents(query: str, all_documents: list[str]) -> list[str]:
    """Find documents containing any query word."""
    results = []
    query_words = query.lower().split()
    
    for doc in all_documents:
        for word in query_words:
            if word in doc.lower():
                results.append(doc)
                break
    
    # The performance bug is NOT in the search loop above...
    # Look at how results are used in the caller:
    
    return results

# In production code that calls this:
def process_all_queries(queries: list[str], documents: list[str]) -> dict:
    all_results = {}
    for query in queries:
        results = search_documents(query, documents)
        all_results[query] = results
    return all_results
    # What's wrong if len(documents) = 1,000,000 and len(queries) = 10,000?
```

---

## Interview Exercises (10 Questions)

Practice answering out loud. Time yourself: 30 seconds for short answer, 2 minutes for detailed answer.

**I1.** "Walk me through what happens when I type a message into ChatGPT and press Enter. Be specific about the LLM's role."

**I2.** "What is RAG and when would you use it instead of fine-tuning? Give a concrete business example."

**I3.** "What causes LLM hallucination and how do you mitigate it in production?"

**I4.** "Explain what an embedding is to a software engineer who has never heard of it."

**I5.** "What's the difference between a vector database and a SQL database? When would you use each?"

**I6.** "A user complains our AI assistant is giving wrong answers. Walk me through how you would debug this."

**I7.** "What is the context window and why does it matter for designing AI systems?"

**I8.** "How do LLM costs scale? A startup is asking if they can afford to use GPT-4o for their application. What questions would you ask?"

**I9.** "What is an AI Agent? How is it different from a simple chatbot?"

**I10.** "You have 1 million documents and need to build a Q&A system over them. Describe the architecture at a high level."

---

## Architecture Exercise (1 Question)

**A1.** Design a customer support AI system for an e-commerce company with 100,000 product pages.

Requirements:
- Users ask questions about products
- System should cite specific product pages
- Must handle 10,000 questions per day
- Must respond in under 2 seconds

Design:
1. Draw the architecture (ASCII or Mermaid)
2. Identify every component and its purpose
3. Explain the data flow
4. Identify the top 3 failure modes
5. Explain how you would evaluate quality

---

## Practical Challenge (1 Challenge)

**P1.** Build a "Concept Flashcard CLI":

Requirements:
- Load AI concepts from a Python data structure
- Present one concept at a time
- Show the name, ask user to explain it
- Then show the definition
- Track which concepts you got right/wrong
- At the end, show a score and which concepts need review
- Use type hints throughout
- Handle keyboard interrupts gracefully

Bonus: Add a "quiz mode" that asks multiple-choice questions.

---

## Scoring Guide

| Section | Questions | Points Each | Max Points |
|---------|-----------|-------------|------------|
| Theory | 10 | 5 | 50 |
| Coding | 5 | 6 | 30 |
| Debugging | 5 | 4 | 20 |
| Interview | 10 | — | practice only |
| Architecture | 1 | 10 | 10 |
| Practical | 1 | — | bonus |
| **Total** | | | **110** |

**Assessment:** See `16_assessment.md` for scoring.
