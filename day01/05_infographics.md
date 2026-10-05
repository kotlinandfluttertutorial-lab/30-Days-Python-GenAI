# Day 01 — Infographics
## Visual Maps of the AI Engineering Landscape

---

## Infographic 1: The AI Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    ARTIFICIAL INTELLIGENCE                    │
│                  (Any intelligent behavior)                   │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │               MACHINE LEARNING                       │   │
│   │          (Learning patterns from data)               │   │
│   │                                                       │   │
│   │   ┌─────────────────────────────────────────────┐   │   │
│   │   │              DEEP LEARNING                   │   │   │
│   │   │        (Neural networks with layers)         │   │   │
│   │   │                                               │   │   │
│   │   │   ┌─────────────────────────────────────┐   │   │   │
│   │   │   │         TRANSFORMERS                 │   │   │   │
│   │   │   │    (Attention-based architecture)    │   │   │   │
│   │   │   │                                       │   │   │   │
│   │   │   │   ┌───────────────────────────────┐  │   │   │   │
│   │   │   │   │          LLMs                  │  │   │   │   │
│   │   │   │   │   (Large Language Models)      │  │   │   │   │
│   │   │   │   │                                │  │   │   │   │
│   │   │   │   │  GPT-4, Claude, Gemini, Llama  │  │   │   │   │
│   │   │   │   └───────────────────────────────┘  │   │   │   │
│   │   │   └─────────────────────────────────────┘   │   │   │
│   │   └─────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** LLMs are a specific type of Deep Learning, which is a specific type of ML, which is a specific type of AI. When someone says "AI", they often mean "LLM-powered application."

---

## Infographic 2: The Modern AI Engineering Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: USER INTERFACE                   │
│         Web App / Mobile App / CLI / API Client              │
└───────────────────────────┬─────────────────────────────────┘
                             │ HTTP Request
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 2: API GATEWAY                        │
│          Rate Limiting | Authentication | Routing            │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3: APPLICATION BACKEND                    │
│                    FastAPI Service                           │
│         Business Logic | Validation | Error Handling         │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 4: AI ORCHESTRATION                       │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ RAG PIPELINE│  │AGENT SYSTEM │  │   PROMPT MANAGER    │ │
│  │             │  │             │  │                     │ │
│  │ Retrieve →  │  │ Plan →      │  │ Templates          │ │
│  │ Rank →      │  │ Tool Call → │  │ Versioning         │ │
│  │ Context     │  │ Execute →   │  │ Injection defense  │ │
│  │             │  │ Observe     │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └───────────────┬┘                     │            │
│                         └─────────────────────┘             │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 5: LLM LAYER                              │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │  OpenAI    │  │ Anthropic  │  │  Local (Ollama/vLLM)   │ │
│  │  GPT-4o    │  │  Claude    │  │  Llama / Mistral       │ │
│  └────────────┘  └────────────┘  └────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              LAYER 6: DATA & STORAGE                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  PostgreSQL  │  │   Redis      │  │   Vector DB        │ │
│  │  (Relational)│  │  (Cache)     │  │  (ChromaDB/FAISS)  │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              LAYER 7: OBSERVABILITY                          │
│                                                              │
│  Logs | Metrics | Traces | Evals | Alerts | Cost Tracking    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Infographic 3: RAG Pipeline Flow

```
DOCUMENTS (PDF, Word, Web, Database)
         │
         ▼
┌─────────────────┐
│   INGESTION     │
│   PIPELINE      │   ← Runs once or periodically
│                 │
│  Load Document  │
│       ↓         │
│  Clean Text     │
│       ↓         │
│  Chunk Text     │   ← Split into 256-512 token pieces
│       ↓         │
│  Embed Chunks   │   ← Convert to vectors
│       ↓         │
│  Store in       │
│  Vector DB      │
└─────────────────┘
         │
         │ (Vector Database now contains all document embeddings)
         │
USER QUERY: "What is our refund policy?"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   RETRIEVAL PIPELINE                          │
│                                                              │
│  Query → [Embed Query] → Query Vector                        │
│                                │                             │
│                                ▼                             │
│                    [Similarity Search in Vector DB]          │
│                                │                             │
│                                ▼                             │
│            Top-K Similar Chunks Returned                     │
│                                │                             │
│                                ▼                             │
│                    [Optional: Rerank]                        │
│                                │                             │
│                                ▼                             │
│                    Context = Relevant Chunks                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION PIPELINE                         │
│                                                              │
│  PROMPT = System Prompt + Context + User Query               │
│                                │                             │
│                                ▼                             │
│                       [LLM generates]                        │
│                                │                             │
│                                ▼                             │
│                    Answer + Citations                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Infographic 4: LLM Tokens and Context Window

```
TEXT INPUT:
"The quick brown fox jumps over the lazy dog"

TOKENIZATION:
┌────────┬───────┬───────┬───────┬───────┬──────┬──────┬──────┬──────┐
│  The   │ quick │ brown │  fox  │ jumps │ over │  the │ lazy │  dog │
│  [1]   │  [2]  │  [3]  │  [4]  │  [5]  │  [6]  │  [7] │  [8] │  [9] │
└────────┴───────┴───────┴───────┴───────┴──────┴──────┴──────┴──────┘

9 TOKENS for this sentence


CONTEXT WINDOW VISUALIZATION:
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW (128K tokens)                      │
│                                                                       │
│ ┌──────────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────┐ │
│ │System Prompt │ │  Retrieved    │ │Conversation  │ │   User     │ │
│ │  (~200 tok)  │ │   Context     │ │  History     │ │   Query    │ │
│ │              │ │ (~1500 tok)   │ │  (~1000 tok) │ │  (~50 tok) │ │
│ └──────────────┘ └───────────────┘ └──────────────┘ └────────────┘ │
│                                                                       │
│ ←────────────────── 128,000 tokens maximum ──────────────────────→  │
└─────────────────────────────────────────────────────────────────────┘

WHAT FITS IN A CONTEXT WINDOW:
128K tokens ≈ 90,000 words ≈ 300 pages of text

GPT-4o:    128K tokens
Claude:    200K tokens
Gemini:    1,000K tokens (1M!)
```

---

## Infographic 5: Embedding Similarity

```
SEMANTIC SPACE (simplified to 2D):

                     Technology
                         ↑
                         │         ● Python
                         │       ● JavaScript
                         │     ● Programming
                         │
    Medicine ────────────┼──────────────────► Language
    ● Doctor             │
    ● Hospital           │
    ● Surgery            │
                         │
                     Science
                         │
                         │   ● Paris ←──── ● France
                         │
                         │   ● London ←─── ● England
                         │
                         ▼
                    Geography


COSINE SIMILARITY EXAMPLES:
("Paris", "France")   → 0.92  (very similar)
("Python", "Java")    → 0.78  (similar domain)
("Python", "Doctor")  → 0.12  (unrelated)
("Hot", "Cold")       → 0.15  (antonyms)

DISTANCE ≠ SIMILARITY:
Paris  → [0.8, -0.2, 0.6, ..., 0.3]   ←── 1536 dimensions
France → [0.7, -0.1, 0.5, ..., 0.4]   ←── angle between = small = similar
Dog    → [-0.3, 0.8, -0.1, ..., -0.2]  ←── angle between = large = different
```

---

## Infographic 6: The Agent Loop

```
     USER GOAL: "Research AI trends and write a summary"
              │
              ▼
     ┌─────────────────┐
     │   LLM THINKS    │ ← "I need to search for AI trends first"
     │   (Reasoning)   │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │  SELECTS TOOL   │ ← search("latest AI trends 2024")
     │   (Planning)    │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │  EXECUTES TOOL  │ ← Tool runs, returns results
     │   (Action)      │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │   OBSERVES      │ ← "Found: GPT-4o, Claude 3.5, Gemini 1.5..."
     │   (Perception)  │
     └────────┬────────┘
              │
        More steps?
         /        \
       Yes         No
        │           │
        ▼           ▼
  [Loop back]   ┌─────────────────┐
  [to Think]    │  FINAL ANSWER   │
                │  (Generation)   │
                └─────────────────┘

KEY: LLM drives the loop. Tools execute the actions. Memory stores history.
```

---

## Infographic 7: AI Engineer vs ML Engineer

```
                    SOFTWARE ENGINEERING
                           │
              ┌────────────┴────────────┐
              │                         │
       AI ENGINEER               ML ENGINEER
              │                         │
    ┌─────────┴──────────┐   ┌──────────┴────────────┐
    │ PRIMARY FOCUS:     │   │ PRIMARY FOCUS:        │
    │ Building AI apps   │   │ Training AI models    │
    │                    │   │                       │
    │ Key Skills:        │   │ Key Skills:           │
    │ - LLM APIs         │   │ - PyTorch / TF        │
    │ - RAG systems      │   │ - Training pipelines  │
    │ - Agents           │   │ - Model optimization  │
    │ - FastAPI          │   │ - Feature engineering │
    │ - Vector DBs       │   │ - Experiment tracking │
    │ - Prompt eng.      │   │ - Model serving       │
    │ - LLMOps           │   │ - MLflow / DVC        │
    │                    │   │                       │
    │ Typical Output:    │   │ Typical Output:       │
    │ - Chat assistant   │   │ - Trained model       │
    │ - RAG system       │   │ - Recommendation sys  │
    │ - AI agent         │   │ - Computer vision sys │
    │ - AI API service   │   │ - NLP pipeline        │
    └────────────────────┘   └───────────────────────┘

              ↓                          ↓
    "Build the AI product"     "Build the AI brain"


    OVERLAP:
    Both need Python, both need evaluation,
    both need production deployment knowledge
```

---

## Infographic 8: Token Cost Calculator Mental Model

```
REQUEST ANATOMY:
┌──────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT                                            │
│ "You are an AI assistant. Answer based on context."      │
│ ≈ 200 tokens                                             │
├──────────────────────────────────────────────────────────┤
│ RETRIEVED CONTEXT (RAG)                                  │
│ [chunk 1: 300 tokens]                                    │
│ [chunk 2: 300 tokens]                                    │
│ [chunk 3: 300 tokens]                                    │
│ ≈ 900 tokens                                             │
├──────────────────────────────────────────────────────────┤
│ USER QUERY                                               │
│ "What is the return policy for damaged items?"           │
│ ≈ 50 tokens                                              │
├──────────────────────────────────────────────────────────┤
│ TOTAL INPUT: ≈ 1,150 tokens                              │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
                   LLM PROCESSES
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ RESPONSE                                                 │
│ "Based on the policy, damaged items can be returned..."  │
│ ≈ 150 tokens (output)                                    │
└──────────────────────────────────────────────────────────┘

COST CALCULATION (GPT-4o):
Input:  1,150 tokens × $0.005/1K = $0.00575
Output: 150 tokens × $0.015/1K  = $0.00225
Total per request: ≈ $0.008

At 10,000 requests/day: $80/day = $2,400/month

OPTIMIZATION LEVERS:
- Shorter system prompts
- Fewer/smaller chunks
- Smaller output model
- Cache frequent queries
- Use cheaper model for classification
```

---

## Infographic 9: Traditional Software vs AI Software

```
TRADITIONAL SOFTWARE:
Input ──→ [Deterministic Logic] ──→ Output
 "2+2"        exact rules           "4"
              ↑ always same         ↑ always same

AI SOFTWARE:
Input ──→ [Probabilistic Model] ──→ Output
"What is  temperature=0.7      "The capital of France is
 France's                       Paris, a city known for..."
 capital?"
                                Different words each time,
                                same correct answer (usually)

TESTING DIFFERENCE:
Traditional: assert get_tax(100, 0.2) == 20.0
AI:          assert evaluate_answer(response) >= 0.8  (quality score)

FAILURE MODES:
Traditional:  ValueError, TypeError, None, exceptions
AI:           Hallucinations, off-topic, wrong tone, jailbreaks

DEBUGGING:
Traditional:  Stack trace → line number → fix
AI:           Prompt analysis → eval → A/B test → fix
```

---

## Infographic 10: The 30-Day Learning Arc

```
      KNOWLEDGE
         │
 EXPERT  │                                          ●  Day 30
         │                                    ●
         │                              ●
 SENIOR  │                        ●
         │                   ●
 MID     │              ●
         │         ●
 JUNIOR  │    ●
         │
 NOVICE  ●───────────────────────────────────────────────────
         1   3   6   9   12  15  18  21  24  27  30
                                                     DAYS

KEY MILESTONES:
Day 1  → Understand the landscape
Day 9  → First ML production system
Day 15 → Understand LLMs deeply
Day 22 → Build production RAG
Day 25 → Build AI agents
Day 29 → Design enterprise systems
Day 30 → INTERVIEW READY
```
