# Day 19 — Concepts
## RAG Fundamentals

---

## 1. Why RAG Exists

```
PROBLEM 1: Knowledge Cutoff
  LLMs are trained on data up to date X.
  They know nothing about events, documents, or data after that.
  → RAG provides current information at query time

PROBLEM 2: Hallucination
  LLMs generate statistically likely text, not verified facts.
  → RAG grounds responses in retrieved documents

PROBLEM 3: Private Knowledge
  Company documents, product manuals, support tickets = not in training.
  → RAG injects this private knowledge at query time

PROBLEM 4: Source Attribution
  Users need to know where information came from.
  → RAG enables citation of source documents
```

---

## 2. The Complete RAG Architecture

```
INGESTION PIPELINE (offline, run once or periodically):
─────────────────────────────────────────────────────────
                    ┌─────────┐
  Documents         │  LOAD   │  PyPDF, docx, HTML, web
  (PDF/Word/Web) →  │   +     │  unstructured library
                    │  CLEAN  │  whitespace, encoding
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  CHUNK  │  512 tokens, 50 overlap
                    │         │  sentence boundaries
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  EMBED  │  text-embedding-3-small
                    │         │  all-MiniLM-L6-v2
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  STORE  │  ChromaDB / pgvector
                    │ Vector  │  + metadata (source, page)
                    │   DB    │
                    └─────────┘

QUERY PIPELINE (online, runs per user request):
─────────────────────────────────────────────────────────
  User Query →  ┌─────────────┐
                │ EMBED QUERY │  same embedding model!
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │   RETRIEVE  │  top-K by cosine similarity
                │  (Vector DB)│  optional: metadata filter
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  RERANK     │  optional: cross-encoder
                │  (optional) │  improves precision
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │ BUILD PROMPT│  system + context + query
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  LLM CALL   │  with retrieved context
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │   ANSWER    │  with source citations
                └─────────────┘
```

---

## 3. Chunking Strategies

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Strategy 1: Fixed-size chunking (simple, fast)
# BAD: splits sentences mid-way
def fixed_chunk(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Strategy 2: Recursive character splitting (recommended)
# Tries to split at: \n\n → \n → . → space
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document_text)

# Strategy 3: Sentence-aware chunking
# Split at sentence boundaries, group N sentences per chunk

# Chunk size selection:
# 128 tokens:  precise retrieval, low context, more chunks needed
# 512 tokens:  balanced (RECOMMENDED default)
# 1024 tokens: more context per chunk, less precise
# 2048 tokens: use for long-context answers

# Always add overlap (50-100 tokens):
# Prevents important info from being split across chunks
```

---

## 4. Context Construction

```python
def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict],
    system_prompt: str,
    max_context_tokens: int = 3000,
) -> list[dict]:
    """Build RAG prompt from retrieved chunks."""

    # Build context with source citations
    context_parts = []
    total_tokens = 0

    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.get("source", f"Document {i+1}")
        text = chunk["text"]
        chunk_tokens = len(text) // 4  # Approximate

        if total_tokens + chunk_tokens > max_context_tokens:
            break

        context_parts.append(f"[Source {i+1}: {source}]\n{text}")
        total_tokens += chunk_tokens

    context = "\n\n---\n\n".join(context_parts)

    return [
        {"role": "system", "content": f"""{system_prompt}

Answer the user's question ONLY based on the provided context.
If the context doesn't contain the answer, say "I don't have that information."
Always cite which source(s) you used, e.g., "[Source 1]".
Never use knowledge outside the provided context."""},
        {"role": "user", "content": f"""Context:
{context}

Question: {query}"""},
    ]
```

---

## 5. RAG vs Fine-Tuning: Decision Framework

```
USE RAG WHEN:
  ✓ Knowledge changes frequently (docs updated regularly)
  ✓ Need to cite sources / attribution required
  ✓ Privacy: documents must stay on-premise
  ✓ Limited budget (no GPU for fine-tuning)
  ✓ Factual Q&A over specific documents
  ✓ Want to update knowledge instantly (add docs, no retraining)

USE FINE-TUNING WHEN:
  ✓ Specific response style or tone required
  ✓ Teaching new capabilities or behavior
  ✓ Domain-specific vocabulary or format
  ✓ High volume of similar queries (amortize training cost)
  ✓ Reduce prompt length (bake instructions into weights)

COMBINE BOTH:
  ✓ Fine-tune for style/behavior + RAG for factual knowledge
  ✓ Example: fine-tuned support bot + RAG over product docs
```
