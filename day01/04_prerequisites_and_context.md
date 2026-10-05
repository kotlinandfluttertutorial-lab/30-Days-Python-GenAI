# Day 01 — Prerequisites and Context

---

## What You Need Before Day 1

### Programming Experience
You need basic programming experience. You do NOT need prior AI/ML experience.

**Required:**
- Understand functions, loops, conditionals
- Know what an API is
- Comfortable with the command line

**Helpful but not required:**
- Python experience (you'll learn AI-specific Python in Day 2)
- SQL experience (useful for Day 7+)
- Web development background (useful for Day 26+)

---

## Environment Setup

### Step 1: Install Python 3.11+

```bash
# Check if Python is installed
python --version
# Should show: Python 3.11.x or higher

# If not installed:
# Windows: Download from python.org
# Mac: brew install python@3.11
# Linux: sudo apt install python3.11
```

### Step 2: Verify pip

```bash
pip --version
# Should show pip version
```

### Step 3: Install Git

```bash
git --version
# If not installed: https://git-scm.com/downloads
```

### Step 4: Install VS Code (Recommended)

Download from: https://code.visualstudio.com/

Recommended extensions:
- Python (Microsoft)
- Pylance
- Python Debugger
- Even Better TOML
- GitLens

### Step 5: Create Your Project Workspace

```bash
# Create a working directory for your daily code
mkdir ai-engineer-practice
cd ai-engineer-practice

# Initialize git
git init
git config user.name "Your Name"
git config user.email "your@email.com"
```

---

## Understanding Virtual Environments

This is critical — learn it on Day 1.

### Why Virtual Environments?

```
Problem without venv:
Project A needs numpy==1.24
Project B needs numpy==2.0
→ They conflict! Installing one breaks the other

Solution with venv:
Project A has its own Python + packages
Project B has its own Python + packages
→ Complete isolation
```

### Creating a Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# You'll see (venv) in your terminal prompt
(venv) $

# Install packages
pip install requests

# Save dependencies
pip freeze > requirements.txt

# Deactivate when done
deactivate

# Another developer installs your deps
pip install -r requirements.txt
```

### Best Practice

```
your-project/
├── venv/              ← never commit this
├── .gitignore         ← include venv/ in here
├── requirements.txt   ← commit this
└── .env               ← never commit this (contains secrets)
```

**.gitignore contents:**
```
venv/
__pycache__/
*.pyc
.env
.DS_Store
```

---

## Understanding .env Files

Critical for AI engineering — you will use these every day.

### Why .env Files?

Your code will call LLM APIs. These require API keys.

```python
# WRONG — never do this:
openai_client = OpenAI(api_key="sk-proj-abc123...")  # key in code!

# RIGHT — use environment variables:
import os
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

### .env File Pattern

```bash
# .env (never commit this)
OPENAI_API_KEY=sk-proj-your-actual-key
ANTHROPIC_API_KEY=sk-ant-your-actual-key
DATABASE_URL=postgresql://user:pass@localhost/db

# .env.example (DO commit this — shows structure without secrets)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### Loading .env in Python

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file

api_key = os.getenv("OPENAI_API_KEY")
```

---

## Getting API Keys (You'll Need These)

### OpenAI API Key
1. Go to platform.openai.com
2. Create account → API → Create new key
3. Store in .env as OPENAI_API_KEY
4. Note: Has per-request costs (very cheap for learning)

### Anthropic API Key
1. Go to console.anthropic.com
2. Create account → API Keys → Create Key
3. Store in .env as ANTHROPIC_API_KEY

### Free Alternatives (No Cost)
- **Groq** — Free tier, fast Llama/Mistral access
- **Google AI Studio** — Free Gemini access
- **Ollama** — Run models locally, completely free

```bash
# Install Ollama (local LLM runner)
# Windows: Download from ollama.com
# Mac: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Run interactively
ollama run llama3.2
```

---

## The Tech Stack You'll Use in This Program

### Day 1–9 (Foundation)
```
python-dotenv    # Environment variables
requests         # HTTP calls
numpy            # Array math
pandas           # Data manipulation
scikit-learn     # ML algorithms
matplotlib       # Plotting
```

### Day 10–15 (Deep Learning)
```
torch            # PyTorch deep learning
torchvision      # Vision datasets
transformers     # HuggingFace models
tokenizers       # Fast tokenization
```

### Day 16–22 (GenAI + RAG)
```
openai           # OpenAI API client
anthropic        # Anthropic API client
chromadb         # Vector database
faiss-cpu        # Vector similarity search
sentence-transformers  # Embedding models
langchain        # LLM framework (for reference)
```

### Day 23–25 (Agents)
```
# Tools built from scratch using:
openai           # Function calling
anthropic        # Tool use
pydantic         # Tool schemas
```

### Day 26–30 (Production)
```
fastapi          # API framework
uvicorn          # ASGI server
pydantic         # Data validation
sqlalchemy       # ORM
psycopg2         # PostgreSQL
redis            # Cache
pytest           # Testing
structlog        # Structured logging
docker           # Containerization (external tool)
```

---

## Context: The AI Engineering Job Market

### Why Now Is the Right Time

The AI Engineer role is one of the fastest-growing in tech. Companies that built traditional software are now being forced to add AI capabilities. They need engineers who can:

1. Build AI features on existing backends
2. Integrate LLM APIs into products
3. Build RAG systems over company knowledge bases
4. Ship AI agents that automate workflows
5. Monitor and evaluate AI systems in production

### What Companies Are Hiring For

Based on current job postings (2024-2025):

**Must-have skills:**
- Python (advanced)
- LLM API integration (OpenAI, Anthropic)
- RAG system design and implementation
- Vector databases (ChromaDB, Pinecone, pgvector)
- FastAPI or equivalent backend framework
- Docker

**Nice-to-have skills:**
- LangChain / LlamaIndex
- Agents and tool calling
- Evaluation frameworks (RAGAS)
- Cloud deployment (AWS/GCP/Azure)
- ML fundamentals (scikit-learn)

**What differentiates senior candidates:**
- System design at scale
- Evaluation and monitoring
- Security (prompt injection, PII)
- Cost optimization
- Production debugging

### Realistic Timeline for Job Readiness

With 12 hours/day for 30 days:
- Days 1-15: Foundation + theory
- Days 16-22: Core AI engineering skills (RAG — the #1 job skill)
- Days 23-30: Production + interview prep

Most AI Engineer jobs require strong RAG knowledge. By Day 22, you will be ready to interview for entry-level AI Engineer roles. By Day 30, you will be ready for mid-to-senior roles.

---

## Historical Context: How We Got Here

Understanding the progression helps you understand why each technology exists.

```
1950s — Alan Turing proposes the Turing Test
1950s-80s — Symbolic AI (rules, logic, expert systems)
    Problem: Brittle, doesn't scale to real-world complexity

1980s-2000s — Machine Learning (statistical models)
    Naive Bayes, SVM, Decision Trees
    Problem: Requires hand-crafted features

2012 — AlexNet wins ImageNet with deep CNN
    Deep learning revolution begins

2014 — Word2Vec (word embeddings)
    Semantic representations

2017 — "Attention Is All You Need" (Transformers)
    Parallelizable, captures long-range dependencies

2018 — BERT (bidirectional transformer encoder)
    Pre-train then fine-tune paradigm

2019 — GPT-2 (decoder transformer)
    Generative language model

2020 — GPT-3 (175B parameters)
    Few-shot learning, emergent capabilities

2022 — ChatGPT (RLHF aligned GPT-3.5)
    Mass adoption of AI applications

2023 — GPT-4, Claude, LLaMA
    The AI Engineer role becomes standard

2024 — Multimodal models, 1M context windows, Agents
    RAG and Agents become core infrastructure

2025-2026 — Reasoning models, Agentic systems
    AI Engineers building complex multi-agent systems
```

---

## What Changes Each Phase

As you progress through the 30 days, the nature of the problems you solve changes:

| Phase | Type of Problem | Key Skill |
|-------|----------------|-----------|
| Days 1-6 | Understanding + fundamentals | Comprehension |
| Days 7-12 | Classical ML + Deep Learning | Implementation |
| Days 13-18 | NLP + LLMs + Embeddings | API integration |
| Days 19-22 | RAG systems | System design |
| Days 23-25 | Agents | Orchestration |
| Days 26-30 | Production + interviews | Engineering judgment |

Notice the progression: from understanding → implementing → integrating → designing → engineering.

By Day 30, you are engineering AI systems, not just implementing them. That is the difference between a junior and a senior AI engineer.
