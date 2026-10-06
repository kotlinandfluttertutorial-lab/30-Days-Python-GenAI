# Day 03 — Architecture
## OOP Component Architecture for AI Systems

---

## AI System Component Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   AI PIPELINE ARCHITECTURE                  │
│                   (Composition Pattern)                     │
│                                                            │
│  RAGPipeline                                               │
│  ├── embedder: BaseEmbedder   ← any implementation         │
│  │   ├── OpenAIEmbedder                                    │
│  │   ├── LocalEmbedder (sentence-transformers)             │
│  │   └── MockEmbedder (for tests)                         │
│  │                                                         │
│  ├── chunker: BaseChunker     ← any chunking strategy      │
│  │   ├── FixedSizeChunker                                  │
│  │   ├── SentenceChunker                                   │
│  │   └── SemanticChunker                                   │
│  │                                                         │
│  ├── store: BaseVectorStore   ← any vector store           │
│  │   ├── ChromaDB                                          │
│  │   ├── FAISS                                             │
│  │   └── pgvector                                          │
│  │                                                         │
│  └── llm: BaseLLM             ← any LLM                   │
│      ├── OpenAI                                            │
│      ├── Anthropic                                         │
│      └── MockLLM (for tests)                               │
│                                                            │
└────────────────────────────────────────────────────────────┘

DATA FLOW:
Ingest:  text → chunk → embed → store
Query:   question → embed → search → rerank → context → llm → answer

SWAP COMPONENTS WITHOUT CHANGING PIPELINE CODE:
pipeline = RAGPipeline(
    embedder=LocalEmbedder(),  # Changed from OpenAI to local
    chunker=FixedSizeChunker(),
    store=FAISS(),             # Changed from ChromaDB to FAISS
    llm=Claude(),
)
```

---

## Data Pipeline Architecture

```
RAW DATA (CSV/JSON/text)
         │
         ▼
    [PANDAS LOAD]
         │
         ▼
    [PROFILING]     ← understand missing values, types, distributions
         │
         ▼
    [CLEANING]      ← drop nulls, fill gaps, normalize text
         │
         ▼
    [FEATURE ENG]   ← create useful features for ML
         │
         ▼
    [SPLIT]         ← train/val/test (fit preprocessing on TRAIN only!)
         │
         ▼
    [EXPORT]        ← save as CSV/Parquet for ML training
         │
         ▼
   ML TRAINING (Day 7)
```
