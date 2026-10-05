# Day 01 — Exercise Solutions

---

## Theory Solutions

**T1.** AI vs ML vs Deep Learning:
- **AI**: Any system mimicking intelligence. Example: chess engine with hard-coded rules (AI but NOT ML)
- **ML**: Learning from data. Example: spam filter trained on labeled emails (ML but NOT necessarily DL)
- **Deep Learning**: ML with neural networks. Example: image classification with CNN (DL, can't be done well with traditional ML at scale)

Key distinction: Rule-based AI systems are not ML. Shallow ML (SVM, decision trees) are not DL. DL requires neural network layers.

**T2.** LLM Technical Mechanism:
An LLM is a function `f(tokens_1...n) → probability_distribution_over_vocabulary`. Given all previous tokens as input, it outputs a probability distribution over the entire vocabulary (50,000+ words/subwords). The next token is sampled from this distribution (controlled by temperature, top-k, top-p). This process is called autoregressive generation — it repeats until a stop token or max length is reached.

**T3.** Three Hallucination Causes + Mitigations:
1. **Statistical completion**: LLM trained to produce likely next tokens, not verified facts → Mitigation: RAG (provide facts in prompt)
2. **Training data limits**: Model trained on data up to date X, doesn't know later events → Mitigation: retrieval-augmented generation, tool use (web search)
3. **Confidence calibration**: Model doesn't know what it doesn't know → Mitigation: instruct "if unsure, say so", evaluate calibration metrics

**T4.** RAG Pipeline Steps:
1. **INGEST**: Documents → Load → Clean → Chunk (512 tokens, 50% overlap)
2. **EMBED**: Each chunk → Embedding model → Dense vector
3. **STORE**: Vectors + text + metadata → Vector database
4. **RETRIEVE**: Query → embed → similarity search → top-K chunks
5. **GENERATE**: System prompt + retrieved chunks + query → LLM → answer with citations

**T5.** Agent vs RAG:
- **RAG**: For factual Q&A over known documents. Fixed pipeline. No planning. Fast, cheap, reliable.
- **Agent**: For tasks requiring multiple steps, unknown path, tool use. Flexible but slower, costlier, more failure modes.
- Use RAG when: "What does our policy say about X?" (answer exists in docs)
- Use Agent when: "Research X topic and write a report" (requires multiple searches, synthesis, decisions)

**T6.** Embeddings + Cosine Similarity:
An embedding is a dense vector in high-dimensional space where semantic similarity corresponds to geometric proximity. We use **cosine similarity** (angle between vectors) instead of Euclidean distance because: document length affects vector magnitude. A 100-word and 1000-word document about the same topic would have very different Euclidean distances despite similar meaning. Cosine similarity is magnitude-invariant — it measures direction, not length.

**T7.** Context Window:
The maximum number of tokens an LLM can process in a single call. Tokens include: system prompt + history + retrieved context + user query + response. At 128K tokens ≈ 90,000 words. Production example that hits limits: a customer support agent that has a 2-hour conversation (all history) + retrieves 10 documents (RAG) + long system prompt. Solution: sliding window, summarization of old history, smart truncation.

**T8.** AI Engineer vs ML Engineer:
- **AI Engineer**: Builds applications using pre-trained models. Key skills: LLM APIs, RAG, agents, FastAPI, vector DBs. Output: AI-powered product features.
- **ML Engineer**: Trains and optimizes models. Key skills: PyTorch, experiment tracking, training infrastructure, feature engineering. Output: trained model artifacts.
- AI Engineer writes more backend code; ML Engineer writes more training pipeline code.

**T9.** AI vs Traditional Software Differences:
1. **Non-determinism** → Cannot use simple assert-based unit tests; need evaluation metrics
2. **Silent failures** → No exception thrown when LLM hallucinates; need monitoring and evals
3. **Prompt is code** → Business logic in prompts must be versioned, tested, reviewed
4. **Cost per call** → Each request costs money; must optimize token usage
5. **External dependency** → LLM provider outage = system outage; need fallbacks and circuit breakers

**T10.** Temperature:
Temperature divides logits before softmax, making distribution sharper (low temp) or flatter (high temp).
- **Temperature=0.1**: Nearly deterministic. Best for: facts, math, classification, structured output. Interviewer wants consistent answers about your codebase.
- **Temperature=1.2**: Creative, unpredictable. Best for: brainstorming, creative writing, diverse options. Risky for factual tasks.
- Rule: temperature=0.0 for tests/evals, 0.3-0.7 for production, 0.8+ only for creative tasks.

---

## Coding Solutions

**C1.** Cosine Similarity:
```python
import math
from typing import Union

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns value in [-1, 1]. Higher = more similar.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: {len(a)} != {len(b)}")
    
    if not a or not b:
        raise ValueError("Vectors cannot be empty")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x ** 2 for x in a))
    magnitude_b = math.sqrt(sum(x ** 2 for x in b))
    
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    
    # Clamp to [-1, 1] to handle floating point imprecision
    return max(-1.0, min(1.0, dot_product / (magnitude_a * magnitude_b)))


# Tests
if __name__ == "__main__":
    # Identical vectors → 1.0
    v1 = [1.0, 0.0, 1.0]
    assert abs(cosine_similarity(v1, v1) - 1.0) < 1e-9, "Identical should be 1.0"
    
    # Orthogonal vectors → 0.0
    v2 = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(v1, v2)) < 1e-9, "Orthogonal should be 0.0"
    
    # Opposite direction → -1.0
    v3 = [-1.0, 0.0, -1.0]
    assert abs(cosine_similarity(v1, v3) - (-1.0)) < 1e-9, "Opposite should be -1.0"
    
    print("All tests passed!")
```

**C2.** Token Estimator:
```python
def estimate_tokens(text: str) -> int:
    """Estimate token count: 1 token ≈ 4 characters (OpenAI heuristic)."""
    return max(1, len(text) // 4)


def cost_usd(tokens: int, price_per_1k: float) -> float:
    """Calculate cost in USD."""
    if tokens < 0:
        raise ValueError("Token count cannot be negative")
    if price_per_1k < 0:
        raise ValueError("Price cannot be negative")
    return (tokens / 1000) * price_per_1k


# Test
sample_text = "Python is a high-level programming language. " * 22  # ~1000 words
tokens = estimate_tokens(sample_text)
print(f"Estimated tokens: {tokens}")
print(f"Cost (GPT-4o input $5/1K): ${cost_usd(tokens, 5.0):.4f}")
print(f"Cost (Claude $3/1K): ${cost_usd(tokens, 3.0):.4f}")
print(f"Cost (Groq $0.1/1K): ${cost_usd(tokens, 0.1):.4f}")
```

**C3.** ConceptCard Dataclass:
```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConceptCard:
    name: str
    category: str
    definition: str
    use_case: str
    difficulty: str = "intermediate"  # beginner, intermediate, advanced


def filter_by_category(cards: list[ConceptCard], category: str) -> list[ConceptCard]:
    return [c for c in cards if c.category.lower() == category.lower()]


# Create 5 concept cards
cards = [
    ConceptCard(
        name="RAG",
        category="Architecture",
        definition="Retrieval-Augmented Generation: retrieve relevant docs before LLM generation",
        use_case="Document Q&A, knowledge assistants",
    ),
    ConceptCard(
        name="Embedding",
        category="Representation",
        definition="Dense vector representing semantic meaning of text",
        use_case="Similarity search, RAG retrieval",
    ),
    ConceptCard(
        name="Transformer",
        category="Architecture",
        definition="Neural architecture using self-attention for sequence processing",
        use_case="Foundation of all modern LLMs",
        difficulty="advanced",
    ),
    ConceptCard(
        name="Temperature",
        category="LLM Parameter",
        definition="Controls randomness: low=deterministic, high=creative",
        use_case="Tuning LLM output style",
        difficulty="beginner",
    ),
    ConceptCard(
        name="Vector Database",
        category="Infrastructure",
        definition="Database optimized for high-dimensional vector similarity search",
        use_case="Storing and querying embeddings in RAG",
    ),
]

architecture_cards = filter_by_category(cards, "Architecture")
print(f"Architecture cards: {[c.name for c in architecture_cards]}")
```

**C4.** SimpleRetriever:
```python
from dataclasses import dataclass, field


@dataclass
class SimpleRetriever:
    """Keyword-based document retriever (concept demo)."""
    documents: list[str] = field(default_factory=list)

    def add_document(self, doc: str) -> None:
        """Add a document to the retrieval corpus."""
        if not doc.strip():
            raise ValueError("Document cannot be empty")
        self.documents.append(doc)

    def _score(self, query: str, document: str) -> int:
        """Keyword overlap score (simplified BM25 concept)."""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        return len(query_words & doc_words)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Return top-k most relevant documents."""
        if not self.documents:
            return []
        
        if not query.strip():
            raise ValueError("Query cannot be empty")

        scored = [
            (self._score(query, doc), doc)
            for doc in self.documents
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k] if _ > 0]


# Test
retriever = SimpleRetriever()
retriever.add_document("Our return policy allows 30-day returns.")
retriever.add_document("Free shipping on orders over $50.")
retriever.add_document("Customer support available 24/7.")
retriever.add_document("Electronics come with 1-year warranty.")

results = retriever.search("return policy refund")
print(f"Search results: {results}")
```

**C5.** Format Prompt:
```python
def format_prompt(template: str, **kwargs: str) -> str:
    """
    Safely format a prompt template with variables.
    
    Raises ValueError if required variable is missing.
    Does NOT raise if extra kwargs are provided (ignores them).
    """
    import re
    
    # Find all variables in template
    required_vars = set(re.findall(r'\{(\w+)\}', template))
    
    # Check all required vars are provided
    missing = required_vars - set(kwargs.keys())
    if missing:
        raise ValueError(
            f"Missing required template variables: {missing}. "
            f"Template requires: {required_vars}"
        )
    
    # Format the template
    return template.format(**{k: v for k, v in kwargs.items() if k in required_vars})


# Test
template = "Answer based on context:\n{context}\n\nQuestion: {question}"

# Should work
prompt = format_prompt(
    template,
    context="Our return policy: 30 days no questions asked.",
    question="Can I return this item?",
)
print(prompt)

# Should raise ValueError
try:
    format_prompt(template, context="Some context")  # Missing 'question'
except ValueError as e:
    print(f"Correctly caught error: {e}")
```

---

## Debugging Solutions

**D1.** Cosine Similarity Bugs:
```python
# Bug 1: sum(a * b) doesn't work — a and b are lists, can't multiply directly
# Fix: sum(x * y for x, y in zip(a, b))

# Bug 2: Operator precedence — dot / mag_a * mag_b divides then multiplies
# Fix: dot / (mag_a * mag_b)  — parentheses!

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))  # Fixed Bug 1
    mag_a = sum(x**2 for x in a) ** 0.5
    mag_b = sum(x**2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)  # Fixed Bug 2
```

**D2.** Missing API Key Handling:
```python
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_client():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. "
            "Add it to your .env file: OPENAI_API_KEY=sk-..."
        )
    
    from openai import OpenAI
    return OpenAI(api_key=api_key)
```

**D3.** Best Score Initialization Bug:
```python
def find_most_similar(query: list[float], 
                       documents: list[list[float]]) -> int:
    """Returns index of most similar document."""
    if not documents:
        raise ValueError("Documents list cannot be empty")
    
    # Bug was: best_score = 0 means any negative similarity is ignored
    # Fix: initialize to -infinity so ANY score beats it
    best_score = float('-inf')
    best_idx = 0
    
    for i, doc in enumerate(documents):
        score = cosine_similarity(query, doc)
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx
```

**D4.** Command Injection:
```python
import subprocess

def get_model_info(model_name: str) -> str:
    """
    BUG: shell=True with user input is a command injection vulnerability.
    If model_name = "llama3 && rm -rf /", the rm command would execute!
    
    Fix: Use list form (no shell) and validate input.
    """
    # Validate input — only allow alphanumeric, dash, dot, colon
    import re
    if not re.match(r'^[a-zA-Z0-9\-\.:]+$', model_name):
        raise ValueError(f"Invalid model name: {model_name}")
    
    result = subprocess.run(
        ["ollama", "show", model_name],  # List form, no shell=True
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout
```

**D5.** O(n*m) Performance Issue:
```python
# The bug: search_documents scans ALL documents for EVERY query.
# With 1M documents and 10K queries: 10 BILLION document scans!
# This would take hours.

# Fix: Build an index ONCE, not per-query.

from collections import defaultdict

class DocumentSearchIndex:
    """Pre-built inverted index for efficient search."""
    
    def __init__(self, documents: list[str]) -> None:
        # Build inverted index: word → set of document indices
        self.documents = documents
        self.index: dict[str, set[int]] = defaultdict(set)
        
        for i, doc in enumerate(documents):
            for word in doc.lower().split():
                self.index[word].add(i)
    
    def search(self, query: str) -> list[str]:
        """O(query_words * avg_docs_per_word) instead of O(queries * docs)."""
        query_words = query.lower().split()
        if not query_words:
            return []
        
        # Start with docs matching first word
        result_indices = self.index.get(query_words[0], set())
        
        # Union with other words (OR search)
        for word in query_words[1:]:
            result_indices |= self.index.get(word, set())
        
        return [self.documents[i] for i in sorted(result_indices)]
```

---

## Interview Solutions

**I1.** ChatGPT Step-by-Step:
"You type a message → tokenizer splits it into token IDs (numbers) → these tokens are sent to the GPT model along with the conversation history and a system prompt. The transformer model processes all tokens with self-attention layers, learning relationships between every token. It produces logits (unnormalized scores) for every possible next token in the vocabulary. Temperature is applied, softmax converts to probabilities, and a token is sampled. That token is appended and the process repeats until an end token is produced. The tokens are decoded back to text and streamed to you."

**I10.** 1M Document Q&A Architecture:
"I'd build a RAG system: Ingest pipeline → clean → chunk documents at ~512 tokens with 50-token overlap → embed each chunk using a model like `text-embedding-3-small` → store vectors + chunk text + document metadata in a vector database (Pinecone or pgvector at this scale). For each user query: embed the query → vector similarity search returns top-K relevant chunks → construct prompt with context + query → LLM generates grounded answer. To handle scale: async ingestion, Redis caching for repeated queries, reranking for precision, hybrid search (BM25 + vector) for recall."

---

## Architecture Solution (A1)

```
E-COMMERCE PRODUCT Q&A SYSTEM

User Query: "Does the XPhone 15 have 5G?"
         │
         ▼
[API Gateway] — Rate limit: 100 req/min/user
         │
         ▼
[FastAPI Backend] — Validate, authenticate
         │
         ▼
[Check Redis Cache] — Same question? Return cached answer
         │ (cache miss)
         ▼
[Embed Query] — text-embedding-3-small
         │
         ▼
[Vector DB Search] — ChromaDB/pgvector
   Filter: category='mobile', brand='apple'
   Return top-5 product page chunks
         │
         ▼
[Reranker] — Cross-encoder reranker, keep top-3
         │
         ▼
[Build Prompt]
  System: "Answer from product specs below. Cite product IDs."
  Context: [chunk 1, chunk 2, chunk 3]
  User: [query]
         │
         ▼
[LLM] — GPT-4o-mini (cheap, fast enough)
         │
         ▼
[Cache Result] — Redis, TTL: 1 hour
         │
         ▼
[Response] — Answer + citations

FAILURE MODES:
1. LLM answers about wrong product — Fix: better metadata filtering + chunk citation
2. Context window overflow — Fix: max 3 chunks, truncate if needed
3. LLM provider outage — Fix: fallback to Anthropic Claude

EVALUATION:
- Answer relevance: does answer address the question?
- Groundedness: is every claim in the retrieved context?
- Human eval: weekly sample of 50 QA pairs rated by human
```
