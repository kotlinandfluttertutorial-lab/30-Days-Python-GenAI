# Day 01 — Concepts
## AI Engineer Foundation: Every Term You Need to Know

---

## 1. Artificial Intelligence (AI)

### Definition
AI is the field of computer science focused on building systems that can perform tasks that typically require human intelligence: understanding language, recognizing patterns, making decisions, solving problems.

### Why It Was Created
Humans wanted to automate cognitive tasks, not just physical ones. Early automation replaced physical labor. AI replaces (or augments) mental labor.

### Problem It Solves
- Automating decisions that are too complex to hard-code with rules
- Processing unstructured data (text, images, audio) at scale
- Adapting to new situations without explicit reprogramming

### How It Works
AI systems learn from data instead of following hand-written rules. Given enough examples, they find patterns that generalize to new situations.

### Three Levels of Understanding

**Beginner:**
AI is software that can do human-like tasks — recognize speech, understand text, make recommendations.

**Mid-Level:**
AI systems use statistical models trained on large datasets to approximate functions that map inputs to outputs. The "intelligence" is really sophisticated pattern recognition.

**Senior AI Engineer:**
AI is a spectrum of techniques — from rule-based systems to probabilistic models to deep neural networks. Different problems require different approaches. The key decisions are: What architecture? What training data? What evaluation metrics? What are the failure modes? How do we handle uncertainty?

### Analogy
Teaching a child to recognize dogs. You don't give them a rule book. You show them thousands of dogs. Eventually they generalize and recognize dogs they've never seen. AI works the same way — show examples, find patterns, generalize.

---

## 2. Machine Learning (ML)

### Definition
A subset of AI where systems learn patterns from data automatically, without being explicitly programmed for every case.

### Why It Was Created
Writing explicit rules for every scenario is impossible at scale. A spam filter with hand-written rules breaks instantly when spammers change tactics. A model trained on spam learns the underlying patterns.

### Problem It Solves
- Pattern recognition at scale
- Prediction from historical data
- Classification of new inputs
- Anomaly detection

### How It Works
```
Data → Algorithm → Model → Predictions
         ↑
    (Optimization)
```

1. Collect labeled data (inputs + correct outputs)
2. Choose an algorithm (linear regression, decision tree, etc.)
3. Algorithm learns a function that maps inputs to outputs
4. Evaluate on held-out data
5. Deploy model for predictions

### Key Types

| Type | Description | Example |
|------|-------------|---------|
| Supervised | Learn from labeled examples | Email → spam/not-spam |
| Unsupervised | Find patterns without labels | Customer segmentation |
| Reinforcement | Learn from rewards/penalties | Game playing |

### Senior Engineer Perspective
ML models are statistical functions. They generalize imperfectly. Understanding the failure modes is more important than understanding the mathematics. In production, the hardest problems are data quality, distribution shift, and evaluation — not the algorithm itself.

---

## 3. Deep Learning (DL)

### Definition
A subset of ML using neural networks with many layers to learn hierarchical representations of data.

### Why It Was Created
Traditional ML required hand-crafted features (feature engineering). Deep learning learns features automatically from raw data — images, audio, text.

### Problem It Solves
- Processing raw unstructured data (images, audio, text)
- Learning representations automatically
- Tasks where manual feature engineering is impossible at scale

### How It Works
```
Input → Layer 1 → Layer 2 → ... → Layer N → Output
        (simple   (complex    (abstract
        features)  features)  features)
```

Each layer learns increasingly abstract representations:
- Image: pixels → edges → shapes → objects → scenes
- Text: characters → words → phrases → meaning → intent

### Why "Deep"?
"Deep" refers to the depth of the network — the number of layers. Modern networks have hundreds or thousands of layers.

### Key Architectures (You Will Build These)
- **CNN** — Convolutional Neural Networks (images, spatial data)
- **RNN/LSTM** — Recurrent networks (sequences, time series)
- **Transformer** — Attention-based (text, everything modern)

---

## 4. Natural Language Processing (NLP)

### Definition
A field of AI focused on enabling computers to understand, interpret, and generate human language.

### Problem It Solves
- Text classification (spam, sentiment, category)
- Information extraction (named entities, relationships)
- Machine translation
- Question answering
- Text generation

### Evolution
```
Rule-based (regex, parse trees)
    ↓
Statistical ML (TF-IDF, SVM, Naive Bayes)
    ↓
Word Embeddings (Word2Vec, GloVe)
    ↓
Deep Learning (LSTM, attention)
    ↓
Transformers (BERT, GPT)
    ↓
Large Language Models (GPT-4, Claude, Gemini)
```

### Key NLP Tasks
- Tokenization — split text into tokens
- Part-of-speech tagging — noun, verb, adjective
- Named entity recognition — "Apple" is a company
- Sentiment analysis — positive, negative, neutral
- Text summarization
- Machine translation
- Question answering

---

## 5. Generative AI (GenAI)

### Definition
AI systems that can generate new content — text, images, audio, video, code — that did not exist before.

### Why It's Different from Traditional ML
Traditional ML: classifies, predicts, detects
Generative AI: creates, generates, synthesizes

### Key Generative Models

| Model Type | What It Generates | Examples |
|-----------|------------------|---------|
| LLM | Text, code | GPT-4, Claude, Gemini |
| Diffusion | Images | DALL-E, Stable Diffusion |
| Speech models | Audio | Whisper, ElevenLabs |
| Video models | Video | Sora, Runway |
| Multimodal | Multiple types | GPT-4V, Gemini |

### Why This Matters for AI Engineers
Most AI Engineer roles focus on building applications **on top of** generative AI models — not training them. You will call LLM APIs, build RAG systems, design agents, and ship production applications.

---

## 6. Large Language Model (LLM)

### Definition
A large neural network trained on massive text datasets that can understand and generate human language. "Large" refers to the number of parameters (billions to trillions).

### Why It Was Created
Previous NLP models were task-specific. LLMs are general-purpose. One model can translate, summarize, answer questions, write code, and reason — without task-specific training.

### Problem It Solves
- General language understanding and generation
- Zero-shot and few-shot task performance
- Knowledge synthesis from training data
- Code generation and explanation

### How It Works (Simplified)

```
"The cat sat on the ___"
                        ↓
                  [LLM processes]
                        ↓
               Probability distribution:
               "mat" → 45%
               "floor" → 20%
               "chair" → 15%
               ...
                        ↓
               Sample from distribution
                        ↓
               Output: "mat"
```

LLMs predict the most likely next token given all previous tokens. This simple objective, at scale, produces emergent capabilities.

### Key Concepts

| Term | Meaning |
|------|---------|
| Parameters | Numbers the model learned during training (weights) |
| Context window | Maximum tokens the model can process at once |
| Token | Smallest unit of text the model processes |
| Temperature | Controls randomness of output |
| Top-k / Top-p | Sampling strategies |
| Inference | Running the model to generate output |
| Pretraining | Initial training on massive datasets |
| Fine-tuning | Additional training on specific data |

### Popular LLMs (October 2026)
- **OpenAI:** GPT-4o, o1, o3
- **Anthropic:** Claude 3.5, Claude 4
- **Google:** Gemini 1.5, 2.0 Pro
- **Meta:** Llama 3.1, 3.2 (open-source)
- **Mistral:** Mistral Large, Mixtral
- **Cohere:** Command R+

### LLM vs Traditional ML

| Aspect | Traditional ML | LLM |
|--------|---------------|-----|
| Training | Supervised, labeled data | Self-supervised, internet text |
| Task | Single specific task | General purpose |
| Output | Label, score, prediction | Generated text |
| Prompt | Not applicable | Critical |
| Deterministic | Yes | No (probabilistic) |
| Size | KB to MB | GB to TB |

---

## 7. Transformer Architecture

### Definition
A neural network architecture that uses self-attention mechanisms to process sequences. The foundation of all modern LLMs.

### Why It Was Created
Introduced in the paper "Attention Is All You Need" (2017, Google). Solved the limitations of RNNs: parallelization, long-range dependencies.

### Problem It Solved
RNNs processed sequences one step at a time (slow, lost context over long sequences). Transformers process the entire sequence at once with attention.

### Core Components

```
Input → [Tokenizer] → [Embedding] → [Positional Encoding]
    ↓
[Multi-Head Attention] × N layers
    ↓
[Feed-Forward Network] × N layers
    ↓
[Output Head]
    ↓
Output
```

| Component | Purpose |
|-----------|---------|
| Tokenizer | Splits text into tokens (subword units) |
| Embedding | Converts tokens to dense vectors |
| Positional encoding | Adds position information |
| Self-attention | Lets each token attend to all other tokens |
| Multi-head attention | Multiple parallel attention computations |
| Feed-forward network | Per-token transformation |
| Layer normalization | Stabilizes training |
| Residual connections | Allows gradients to flow |

### Two Types

**Encoder-only (BERT-style):**
- Reads entire sequence bidirectionally
- Good for: classification, embedding, understanding
- Example: BERT, RoBERTa

**Decoder-only (GPT-style):**
- Generates text left-to-right
- Good for: generation, completion, chat
- Example: GPT-4, Claude, Llama

**Encoder-Decoder (T5-style):**
- Encoder reads, decoder generates
- Good for: translation, summarization
- Example: T5, BART

---

## 8. Embeddings

### Definition
A numerical vector representation of data (text, image, etc.) that captures semantic meaning. Similar things have similar vectors.

### Why They Exist
Computers cannot work with raw text. Embeddings convert text to numbers in a way that preserves meaning — so semantic similarity can be computed mathematically.

### Problem They Solve
- "Paris" and "France" should be similar vectors
- "Dog" and "Cat" should be closer than "Dog" and "Car"
- Traditional encodings (one-hot) don't capture this

### How They Work

```
"Paris" → [0.2, 0.8, -0.3, ..., 0.1]  (768 or 1536 numbers)
"France" → [0.3, 0.7, -0.2, ..., 0.2]
"Dog" → [-0.5, 0.1, 0.9, ..., -0.4]
```

The model is trained so that semantically similar text produces numerically similar vectors.

### Measuring Similarity

**Cosine similarity** — angle between vectors:
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

- Score of 1.0 = identical direction (very similar)
- Score of 0.0 = orthogonal (unrelated)
- Score of -1.0 = opposite (antonyms)

### Types of Embeddings

| Type | What It Embeds | Use Case |
|------|---------------|---------|
| Word embeddings | Individual words | NLP basics |
| Sentence embeddings | Full sentences | Semantic search |
| Document embeddings | Full documents | RAG |
| Image embeddings | Images | Image search |
| Code embeddings | Code | Code search |

### Key Embedding Models
- OpenAI `text-embedding-3-large` — 3072 dimensions
- OpenAI `text-embedding-3-small` — 1536 dimensions
- `all-MiniLM-L6-v2` — 384 dimensions (fast, local)
- `nomic-embed-text` — 768 dimensions (open-source)
- Cohere `embed-english-v3.0`

---

## 9. Vector Database

### Definition
A database optimized for storing and querying high-dimensional vectors (embeddings) using similarity search.

### Why It Exists
Traditional databases (SQL) can't efficiently search "find all vectors most similar to this query vector." Vector databases are purpose-built for this.

### Problem It Solves
- Store millions of document embeddings
- Find the top-K most similar documents to a query
- Filter by metadata (date, category, author)
- Scale similarity search to billions of vectors

### How It Works

```
Documents → Embed → Store vectors in Vector DB
                          ↓
Query → Embed → Search Vector DB → Top-K similar docs
```

Internally uses Approximate Nearest Neighbor (ANN) algorithms like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File Index).

### Popular Vector Databases

| Database | Type | Best For |
|----------|------|---------|
| ChromaDB | Embedded | Development, small-medium scale |
| FAISS | Library | In-process, research |
| Pinecone | Managed cloud | Production at scale |
| Weaviate | Open-source cloud | Complex queries |
| pgvector | PostgreSQL extension | Existing Postgres users |
| Qdrant | Open-source | Production, high performance |

---

## 10. Retrieval-Augmented Generation (RAG)

### Definition
A technique that enhances LLM responses by retrieving relevant context from an external knowledge base before generating an answer.

### Why It Was Created
LLMs have two fundamental problems:
1. **Knowledge cutoff** — they don't know recent events
2. **Hallucination** — they make up facts confidently

RAG solves both: instead of relying on memorized training data, the LLM is given the relevant documents at query time.

### Problem It Solves
- LLMs answering questions about your company's documents
- Keeping AI responses up-to-date without retraining
- Reducing hallucinations by grounding responses in facts
- Citing sources for generated answers

### How It Works

```
User: "What is our refund policy?"
         ↓
1. EMBED: Convert query to vector
         ↓
2. RETRIEVE: Search vector DB for similar chunks
         ↓
3. CONTEXT: Found: "Refund Policy: 30 days no questions asked..."
         ↓
4. PROMPT: "Answer based on context: [policy text] | Question: [query]"
         ↓
5. GENERATE: LLM generates answer citing the policy
         ↓
Answer: "According to the policy, you can return within 30 days..."
```

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|------------|
| When to use | Factual Q&A, docs | Style, behavior, format |
| Cost | Low (retrieval only) | High (training compute) |
| Update speed | Instant (add docs) | Slow (retrain) |
| Hallucination | Reduced | Can still occur |
| Data privacy | Keep docs local | Share data with provider |
| Best for | Knowledge-intensive | Behavior-change |

---

## 11. AI Agent

### Definition
An AI system that can perceive its environment, make decisions, use tools, remember past interactions, and take actions to achieve goals — going beyond single-turn responses.

### Why It Was Created
Simple LLM chatbots are limited to a single question-answer turn. Agents can break down complex tasks, use tools (search, code execution, APIs), and iteratively work toward goals.

### Problem It Solves
- Tasks too complex for a single LLM call
- Tasks requiring external information (web search, database)
- Tasks requiring computation (running code, calculations)
- Tasks requiring memory across multiple steps

### Core Components

```
Agent = LLM + Tools + Memory + Planning
```

| Component | Purpose | Example |
|-----------|---------|---------|
| LLM | Reasoning, decision-making | GPT-4, Claude |
| Tools | Actions in the world | Search, code execution, DB |
| Memory | Remember past interactions | Conversation history, vector store |
| Planning | Break down complex tasks | Chain-of-thought, ReAct |

### The Agent Loop

```
User Goal
    ↓
Think (LLM reasons about next step)
    ↓
Act (Execute a tool or action)
    ↓
Observe (See the result)
    ↓
Think → Act → Observe (repeat)
    ↓
Final Answer when goal achieved
```

### Agent vs Chatbot vs RAG

| System | Input | Memory | Tools | Planning |
|--------|-------|--------|-------|---------|
| Chatbot | Text | Limited | None | None |
| RAG | Text | None | Retrieval | None |
| Agent | Text/Data | Yes | Multiple | Yes |

---

## 12. API (Application Programming Interface)

### Definition
A contract that defines how software components communicate. An interface for calling functions in other systems over a network.

### Why AI Engineers Care
Every AI capability is exposed through an API:
- LLM APIs (OpenAI, Anthropic)
- Embedding APIs
- Image generation APIs
- Speech-to-text APIs

You will build FastAPI backends that serve AI capabilities to frontends and other services.

### REST API Basics

```
HTTP Method | Action
GET         | Read
POST        | Create / Execute
PUT         | Replace
PATCH       | Partial update
DELETE      | Delete

Request:
POST /api/chat
Headers: { "Authorization": "Bearer sk-..." }
Body: { "message": "Hello", "model": "gpt-4" }

Response:
200 OK
Body: { "response": "Hello! How can I help?" }
```

---

## 13. Docker

### Definition
A platform for packaging applications and their dependencies into portable containers that run consistently anywhere.

### Why AI Engineers Need It
AI applications have complex dependencies:
- Python 3.11
- PyTorch with CUDA
- Multiple Python packages with specific versions
- Environment variables for API keys

Without Docker: "It works on my machine"
With Docker: Works everywhere identically.

### Key Concepts

```
Dockerfile → Docker Image → Docker Container

Dockerfile: Instructions to build the image
Image: Immutable snapshot of the application
Container: Running instance of an image
```

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
```

### Docker Compose
For multi-service AI applications:
```yaml
services:
  api:      # FastAPI backend
  db:       # PostgreSQL
  redis:    # Cache
  chroma:   # Vector database
```

---

## 14. Cloud Fundamentals

### Definition
On-demand access to compute, storage, and networking resources over the internet.

### Why AI Engineers Use Cloud
- GPU compute for model inference
- Managed databases and vector stores
- Serverless functions for scalability
- Managed Kubernetes for orchestration
- CDN for global low-latency serving

### Key Cloud Providers

| Provider | AI Services |
|----------|------------|
| AWS | SageMaker, Bedrock, Lambda |
| Google Cloud | Vertex AI, Cloud Run, BigQuery |
| Azure | Azure OpenAI, ML Studio |
| Replicate | Model hosting |
| Modal | GPU serverless functions |

### What You Will Learn
- Deploying Docker containers to cloud
- Environment variables and secrets management
- Cloud storage for documents and models
- Basic CI/CD pipelines

---

## 15. LLMOps

### Definition
The operational practices for building, deploying, monitoring, and maintaining LLM-powered applications in production.

### Why It Exists
LLM applications have unique operational challenges:
- Non-deterministic outputs (different every run)
- Prompt changes can break behavior subtly
- Token costs accumulate quickly
- Latency is high (seconds per request)
- Hallucinations can go undetected

### Key LLMOps Concerns

| Concern | What | Why |
|---------|------|-----|
| Prompt versioning | Track prompt changes | Rollback bad prompts |
| Evaluation | Measure output quality | Catch regressions |
| Observability | Log inputs/outputs/traces | Debug issues |
| Cost monitoring | Track token usage | Prevent budget overruns |
| Latency monitoring | Track response times | SLA compliance |
| Safety | Detect prompt injection, PII | Security |

---

## 16. AI Engineer vs ML Engineer

This comparison is critical for understanding what roles expect.

| Aspect | AI Engineer | ML Engineer |
|--------|------------|------------|
| Primary focus | Applications, systems | Models, training |
| Key skill | Building AI products | Training/optimizing models |
| Tools | LLM APIs, RAG, Agents, FastAPI | PyTorch, scikit-learn, MLflow |
| Output | Production AI apps | Trained models |
| Math depth | Understand concepts | Deep mathematical knowledge |
| Infra focus | APIs, backends, Docker | Training infrastructure |
| Data focus | Retrieval, context | Feature engineering, pipelines |
| Typical job | Build an AI assistant | Train a recommendation model |

### What You Are Becoming
An **AI Engineer** — someone who builds production-quality AI applications using LLMs, RAG, agents, and supporting infrastructure.

You are NOT primarily training models from scratch. You are building systems that use pre-trained models intelligently.

---

## 17. The Modern AI Stack

```
┌─────────────────────────────────────────────┐
│                USER INTERFACE                │
│           (Web, Mobile, CLI, API)            │
├─────────────────────────────────────────────┤
│              AI APPLICATION LAYER            │
│          (FastAPI, Business Logic)           │
├─────────────────────────────────────────────┤
│            AI ORCHESTRATION LAYER            │
│          (RAG Pipeline, Agent Loop)          │
├──────────────┬──────────────┬───────────────┤
│   RETRIEVAL  │    MEMORY    │    TOOLS       │
│ (Vector DB)  │  (Redis/DB)  │  (APIs/Code)  │
├──────────────┴──────────────┴───────────────┤
│                  LLM LAYER                   │
│          (OpenAI, Anthropic, Local)          │
├─────────────────────────────────────────────┤
│            INFRASTRUCTURE LAYER              │
│     (PostgreSQL, Redis, Docker, Cloud)       │
├─────────────────────────────────────────────┤
│            OBSERVABILITY LAYER               │
│        (Logs, Metrics, Traces, Evals)        │
└─────────────────────────────────────────────┘
```

---

## 18. Traditional Software vs AI Software

| Aspect | Traditional Software | AI Software |
|--------|---------------------|------------|
| Behavior | Deterministic | Probabilistic |
| Testing | Unit tests, integration | Evals, benchmarks |
| Bugs | Logic errors | Hallucinations, bias |
| Debugging | Stack trace, logs | Prompt analysis, evals |
| Versioning | Code versions | Model + prompt versions |
| Deployment | Build once | Continuous monitoring |
| Failures | Exceptions, errors | Silent wrong answers |

**The hardest part of AI Engineering:** AI systems can fail silently. A wrong SQL query throws an error. A hallucinating LLM returns a confident, plausible-sounding wrong answer.

This is why evaluation and observability are so critical in this program.
