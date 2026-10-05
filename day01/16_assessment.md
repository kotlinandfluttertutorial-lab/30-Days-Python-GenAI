# Day 01 — Assessment
## Daily Assessment: AI Engineer Foundation

---

## Instructions

Complete all sections without notes or reference materials.  
Time limit: 90 minutes.  
Score yourself using the rubric at the end.

---

## Section 1: Theory Questions (10 points each = 100 points)

**1.** Define an LLM in exactly 2 sentences. Your definition must include: what it's trained on, what it predicts, and why this produces general intelligence.

**2.** List the 5 steps in the RAG pipeline in order. For each step, state what the input is and what the output is.

**3.** An LLM is asked "Who won the 2024 Nobel Peace Prize?" It answers confidently but incorrectly. Explain exactly why this happens and name two production solutions.

**4.** What is the context window? Give an example of a real system that would hit context window limits and explain how you would handle it.

**5.** What is cosine similarity? Write the formula. Why is it preferred over Euclidean distance for embeddings?

**6.** Compare AI software and traditional software on these five dimensions: determinism, failure modes, testing approach, versioning, debugging methodology.

**7.** What is the difference between an AI Engineer and an ML Engineer? State the primary output of each role.

**8.** An agent is given the goal: "Book the cheapest flight to Tokyo next month." Describe the agent loop in steps (minimum 5 steps) showing Think → Act → Observe for each.

**9.** What is temperature in LLM sampling? When would you set temperature=0? When would you set temperature=1.0? What happens at temperature=2.0?

**10.** Explain three reasons why a company would choose RAG over fine-tuning for adding company knowledge to an LLM assistant.

---

## Section 2: Code Questions (20 points each = 100 points)

**11.** Write a complete, working Python function `cosine_similarity(a: list[float], b: list[float]) -> float` that:
- Uses type hints
- Handles the zero-magnitude edge case
- Includes a docstring
- Is correct (will pass unit tests)

**12.** Write a Python function `estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float` that returns the estimated cost in USD. Support: "gpt-4o" ($5/$15 per 1M), "gpt-4o-mini" ($0.15/$0.60), "claude-3-5-sonnet" ($3/$15).

**13.** Write a `LLMMessage` dataclass and a `format_chat_messages(system: str, conversation: list[LLMMessage]) -> list[dict]` function that returns the messages in OpenAI API format.

**14.** Write a `SimpleVectorStore` class with methods: `add(id: str, text: str, vector: list[float])`, `search(query_vector: list[float], top_k: int) -> list[tuple[str, float]]`. Use cosine similarity. Include type hints and docstrings.

**15.** Write a function `safe_load_env(required_keys: list[str]) -> dict[str, str]` that:
- Loads the .env file
- Checks all required keys exist
- Raises ValueError with a clear message listing missing keys
- Returns a dict of all found keys

---

## Section 3: Debugging Questions (10 points each = 50 points)

Find and fix all bugs in each snippet. Explain each bug.

**16.** 
```python
def calculate_similarity(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        emb = get_embedding(text)
        embeddings.append(emb)
    
    similarities = []
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row.append(sim)
        similarities.append(row)  # This part is correct
    
    return similarities
    # Bug: This computes ALL pairwise similarities. 
    # What's the performance issue at scale? How would you fix it?
```

**17.**
```python
from dotenv import load_dotenv
import os

def create_ai_client():
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    
    from openai import OpenAI
    client = OpenAI(key)  # Bug 1
    return client

client = create_ai_client()
response = client.chat.completions.create(
    model="gpt4o",  # Bug 2
    messages={"role": "user", "content": "Hi"}  # Bug 3
)
```

**18.**
```python
def chunk_text(text: str, chunk_size: int) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:chunk_size]  # Bug
        chunks.append(chunk)
    return chunks
```

**19.**
```python
class VectorStore:
    vectors = []  # Bug: class-level vs instance-level
    
    def add(self, vector: list[float]) -> None:
        self.vectors.append(vector)
    
    def count(self) -> int:
        return len(self.vectors)

# What happens here?
store1 = VectorStore()
store1.add([1.0, 2.0])

store2 = VectorStore()
print(store2.count())  # What does this print? Why is it a bug?
```

**20.**
```python
import os

API_KEY = os.getenv("OPENAI_API_KEY")

def call_llm(prompt: str) -> str:
    from openai import OpenAI
    # Bug: API key is captured at module load time
    # What's the issue if .env is loaded AFTER this module?
    client = OpenAI(api_key=API_KEY)
    ...
```

---

## Section 4: Interview Questions (20 points each = 100 points)

Write your answers as if you're speaking in an interview. Be concise but complete.

**21.** "What is RAG? Walk me through how it works."  
(Target: 2-minute answer. Must include: embeddings, vector search, context injection, generation)

**22.** "Why do LLMs hallucinate and how do you prevent it in production?"  
(Target: 90-second answer. Must include: mechanism, 2+ mitigations, evaluation)

**23.** "Design a Q&A system over a company's internal documentation."  
(Target: 3-minute answer. Must include: architecture diagram, components, evaluation)

**24.** "What is the difference between temperature=0 and temperature=1?"  
(Target: 60-second answer with example)

**25.** "A user reports our AI is giving outdated answers. How do you investigate and fix this?"  
(Target: 90-second answer. Must include: diagnostic steps, root cause, fix)

---

## Section 5: Architecture Question (50 points)

**26.** Design a "Smart Document Search" system:

**Requirements:**
- Users upload PDF documents
- Users can search documents by natural language query
- Results show relevant passages with source document + page number
- Must support 100,000 documents
- Must respond in under 3 seconds for 95% of queries

**Deliverables (all required for full marks):**
- ASCII or Mermaid architecture diagram
- Data flow description (step by step)
- Component list with technology choices and justification
- List of failure modes and mitigations
- How you would evaluate quality
- Cost estimate for 1,000 searches/day

---

## Scoring Rubric

| Section | Max Points | Your Score |
|---------|-----------|-----------|
| Theory (10 × 10) | 100 | ___ |
| Code (5 × 20) | 100 | ___ |
| Debugging (5 × 10) | 50 | ___ |
| Interview (5 × 20) | 100 | ___ |
| Architecture | 50 | ___ |
| **Total** | **400** | ___ |
| **Percentage** | | ___ % |

---

## Score Interpretation

| Score | Percentage | Interpretation | Action |
|-------|-----------|----------------|--------|
| 360-400 | 90-100% | Excellent — Day 1 mastered | Proceed to Day 2 |
| 300-359 | 75-89% | Good — solid foundation | Proceed to Day 2, review weak areas |
| 240-299 | 60-74% | Needs revision | Review concepts, redo exercises |
| Below 240 | < 60% | Repeat Day 1 | Full repeat required before Day 2 |

---

## Scoring Guide for Code Questions

**Full marks (20/20):** Code runs, correct output, type hints, handles edge cases, clear docstring.

**Partial marks (10-15/20):** Code runs but missing type hints OR edge case handling.

**Low marks (5-10/20):** Code has bugs but approach is correct.

**No marks (0/20):** Cannot write the code or completely wrong approach.

---

## After the Assessment

1. Grade yourself honestly — no cheating
2. Note which questions you struggled with
3. For each wrong answer, find the section in today's materials
4. Re-read those sections
5. Update `progress.md` with your score

If you scored below 75%, do NOT proceed to Day 2 until you revise.  
The foundation must be solid — everything else builds on today.
