# Day 30 — 250+ Interview Questions
## Complete AI Engineer Interview Bank

---

## PYTHON (30 Questions)

**P1.** What is the GIL and how does it affect AI Python backends?
**P2.** Explain async/await. When does async NOT help?
**P3.** What is the difference between `os.getenv()` and `os.environ[]`?
**P4.** How do you implement a retry decorator with exponential backoff?
**P5.** What is a generator? When would you use it over a list in AI?
**P6.** Explain `@dataclass` vs `pydantic.BaseModel`. When to use each?
**P7.** What is `asyncio.gather()` and when is it critical for AI backends?
**P8.** How do you prevent mutable default arguments in dataclasses?
**P9.** What does `functools.wraps` do in a decorator?
**P10.** How do you safely parse JSON from LLM output?
**P11.** What is a context manager and give an AI use case?
**P12.** What is the difference between `time.sleep` and `asyncio.sleep`?
**P13.** How do you implement a simple in-memory LRU cache?
**P14.** Explain Python's `__post_init__` in dataclasses.
**P15.** What is `asyncio.wait_for()` and why is it critical for LLM calls?
**P16.** How do you profile Python code for performance bottlenecks?
**P17.** What is a Protocol in Python typing and how is it used in AI frameworks?
**P18.** How do you handle secrets securely in a Python AI application?
**P19.** What is the difference between `@classmethod` and `@staticmethod`?
**P20.** How do you write thread-safe code when sharing state in async Python?
**P21.** What does `__slots__` do and when is it useful for AI data objects?
**P22.** Explain `itertools.batched` (or implement your own batch function).
**P23.** How do you test async code with pytest?
**P24.** What is a deque and when would you use it for conversation history?
**P25.** How do you handle `KeyboardInterrupt` gracefully in a long-running AI pipeline?
**P26.** What is the difference between `deepcopy` and `copy`? When does it matter in AI?
**P27.** How do you implement rate limiting without Redis (in-memory)?
**P28.** Explain Python's `__enter__` and `__exit__` methods.
**P29.** What is `typing.TypeVar` and when is it used in AI code?
**P30.** How do you use `logging.getLogger(__name__)` properly across a project?

---

## MACHINE LEARNING (30 Questions)

**ML1.** Explain the bias-variance tradeoff. Give a concrete example.
**ML2.** When would you use Random Forest vs Logistic Regression?
**ML3.** What is cross-validation and why is it more reliable than a single split?
**ML4.** Explain precision and recall. When is each more important?
**ML5.** What is data leakage and how do you prevent it?
**ML6.** What does `Pipeline(scaler, model).fit()` do that prevents leakage?
**ML7.** Explain gradient descent. What happens with too high a learning rate?
**ML8.** What is regularization? Difference between L1 and L2?
**ML9.** How do you handle imbalanced classes in a classification problem?
**ML10.** What is ROC-AUC and why is it better than accuracy for imbalanced data?
**ML11.** Explain the difference between `fit_transform` and `transform`.
**ML12.** What is overfitting? Name 4 ways to prevent it.
**ML13.** When would you use GridSearchCV vs RandomizedSearchCV?
**ML14.** What is feature importance in Random Forest? How is it computed?
**ML15.** Explain K-means clustering. How do you choose K?
**ML16.** What is the elbow method?
**ML17.** Explain MSE vs MAE. When would you prefer MAE?
**ML18.** What is R² score? What does R²=0 mean?
**ML19.** How do you serialize/deserialize a sklearn model?
**ML20.** What is StandardScaler? On what data should you fit it?
**ML21.** Explain the difference between supervised and unsupervised learning.
**ML22.** What is stratified sampling and when is it required?
**ML23.** How do you evaluate a regression model differently from a classification model?
**ML24.** What is early stopping in gradient-based optimization?
**ML25.** Explain the decision tree splitting criterion (Gini vs entropy).
**ML26.** What is an ensemble method? Name two types.
**ML27.** How does `n_estimators` affect a Random Forest?
**ML28.** What is the curse of dimensionality?
**ML29.** When would you apply PCA before ML?
**ML30.** What is the difference between online and batch learning?

---

## DEEP LEARNING (25 Questions)

**DL1.** What is backpropagation? Explain the chain rule.
**DL2.** Why do we need activation functions? What breaks without them?
**DL3.** When would you use ReLU vs sigmoid vs softmax?
**DL4.** What is vanishing gradient and which activation functions cause it?
**DL5.** What is He initialization? Why is it better than zero initialization?
**DL6.** Explain the PyTorch training loop (5 steps).
**DL7.** What does `model.train()` vs `model.eval()` do?
**DL8.** Why must you call `optimizer.zero_grad()` before `loss.backward()`?
**DL9.** What is dropout and how does it work during training vs inference?
**DL10.** Explain Batch Normalization. Where do you place it?
**DL11.** What is gradient clipping and when is it used?
**DL12.** Explain the difference between CNN and RNN/LSTM.
**DL13.** What problem does LSTM solve compared to vanilla RNN?
**DL14.** What is transfer learning? What are the two stages?
**DL15.** Why do we use Adam over SGD in most cases?
**DL16.** What is a learning rate scheduler? Name two types.
**DL17.** Explain early stopping. What does patience mean?
**DL18.** What is the difference between batch size and epoch?
**DL19.** How do you save and load a PyTorch model checkpoint?
**DL20.** What is `torch.no_grad()` and when must you use it?
**DL21.** What is the difference between `CrossEntropyLoss` and `BCELoss`?
**DL22.** What is `nn.Sequential` vs subclassing `nn.Module`?
**DL23.** What is weight decay and how does it relate to L2 regularization?
**DL24.** Explain the DataLoader parameters: batch_size, shuffle, num_workers.
**DL25.** What is `model.parameters()` vs `model.named_parameters()`?

---

## TRANSFORMERS (25 Questions)

**T1.** Explain scaled dot-product attention. Why divide by √d_k?
**T2.** What are Query, Key, and Value in attention?
**T3.** Why does multi-head attention outperform single-head attention?
**T4.** What is positional encoding and why is it needed?
**T5.** What is the difference between encoder-only and decoder-only transformers?
**T6.** What is a causal mask and why is it required for GPT-style models?
**T7.** What is LayerNorm vs BatchNorm? Why do transformers use LayerNorm?
**T8.** Why are residual connections essential in deep transformers?
**T9.** Explain GELU activation. Why is it used in transformers over ReLU?
**T10.** What is BPE tokenization? Why does it handle rare words better than word tokenization?
**T11.** What is a KV cache? How does it speed up inference?
**T12.** Explain pre-training vs fine-tuning in the context of BERT vs GPT.
**T13.** What is BERT's masked language modeling objective?
**T14.** What is next sentence prediction in BERT?
**T15.** What is the difference between BERT and sentence-transformers?
**T16.** What is the Feed-Forward Network in a transformer block?
**T17.** Explain why transformers parallelize better than RNNs.
**T18.** What is Flash Attention? What problem does it solve?
**T19.** What is LoRA (Low-Rank Adaptation)? How does it reduce fine-tuning cost?
**T20.** What are the components of a transformer block in order?
**T21.** What is "attention is all you need" referring to?
**T22.** Explain cross-attention vs self-attention.
**T23.** What is the vocabulary size of GPT-2 and why does it matter for memory?
**T24.** What is temperature in transformer inference?
**T25.** What is the difference between top-k and top-p sampling?

---

## LLM (30 Questions)

**L1.** Explain the three phases of LLM training: pretraining, SFT, RLHF.
**L2.** What causes LLM hallucination at a technical level?
**L3.** What is the context window? Why can't we just make it infinite?
**L4.** Explain temperature, top-k, and top-p. When do you use each?
**L5.** What is finish_reason="length" vs "stop"?
**L6.** How do you stream LLM responses in Python?
**L7.** What is RLHF and what does it train the model to do?
**L8.** What is DPO and how does it differ from RLHF?
**L9.** Explain the difference between a system prompt and a user message.
**L10.** How do you estimate token count before making an API call?
**L11.** Compare GPT-4o vs Claude 3.5 Sonnet: strengths and use cases.
**L12.** What is an open-source LLM? Name 3 with their context windows.
**L13.** What is Ollama and when would you use it over API LLMs?
**L14.** How does `response_format={"type": "json_object"}` work in OpenAI?
**L15.** What is a prompt template and why should they be versioned?
**L16.** Explain the "lost in the middle" phenomenon.
**L17.** What is few-shot prompting and when is it more effective than zero-shot?
**L18.** What is chain-of-thought prompting? When does it help?
**L19.** How do you handle a situation where the LLM ignores your instructions?
**L20.** What is an LLM's pretraining corpus and what does it determine?
**L21.** What are the risks of using `eval()` in a tool for an LLM agent?
**L22.** What is "emergent capability" in LLMs?
**L23.** How does quantization (INT4, INT8) affect LLM quality and speed?
**L24.** What is model distillation in the context of LLMs?
**L25.** Explain why smaller context windows are sometimes preferable.
**L26.** What is the difference between greedy decoding and sampling?
**L27.** How do you handle rate limit errors from the OpenAI API?
**L28.** What is a "system" role vs "user" role in the messages array?
**L29.** What is token-by-token generation and why does it create latency?
**L30.** How do you select an LLM for a given use case? What factors matter?

---

## RAG (40 Questions)

**R1.** Explain the complete RAG pipeline end-to-end.
**R2.** Why does RAG reduce hallucination compared to vanilla LLM?
**R3.** What is the difference between the ingestion pipeline and the query pipeline?
**R4.** What chunking strategy would you use for a legal document corpus?
**R5.** Why do we add overlap between chunks?
**R6.** What is the recommended default chunk size and why?
**R7.** Explain why you must use the same embedding model for indexing and querying.
**R8.** What is hybrid search? How do you combine BM25 and vector scores?
**R9.** What is a reranker? What is a cross-encoder vs bi-encoder?
**R10.** When should you use a reranker in your RAG pipeline?
**R11.** What is multi-query retrieval and when does it improve recall?
**R12.** What is query rewriting? Give a concrete example.
**R13.** What is parent-child retrieval?
**R14.** How do you detect when a query has no relevant documents?
**R15.** What metadata should you store with each chunk?
**R16.** How do you handle documents that are frequently updated?
**R17.** What is context precision? What is context recall?
**R18.** What is faithfulness in RAG evaluation?
**R19.** What is answer relevance and how is it measured?
**R20.** Explain RAGAS and what metrics it covers.
**R21.** What is LLM-as-judge and what are its limitations?
**R22.** How do you build an evaluation dataset for a RAG system?
**R23.** How do you handle very long documents that don't fit in one chunk?
**R24.** What is semantic chunking and when is it better than recursive splitting?
**R25.** How do you handle PDFs with tables, images, and mixed content?
**R26.** What is Reciprocal Rank Fusion (RRF)?
**R27.** How do you cache RAG query results efficiently?
**R28.** What is content hash deduplication in document ingestion?
**R29.** When would you use pgvector vs Qdrant for production RAG?
**R30.** How do you handle multi-language documents in a RAG system?
**R31.** What is the "needle in a haystack" problem in long-context RAG?
**R32.** How do you stream RAG responses while still providing citations?
**R33.** What happens when retrieved context and LLM training data conflict?
**R34.** How would you debug a RAG system returning irrelevant answers?
**R35.** What is the difference between a sparse and dense vector?
**R36.** How do you measure RAG latency breakdown (retrieval vs generation)?
**R37.** What is the role of the system prompt in a RAG pipeline?
**R38.** When should you fine-tune vs use RAG for enterprise Q&A?
**R39.** How do you handle document deletion in a RAG system?
**R40.** Design a RAG system for a 10-million-document legal corpus.

---

## AGENTS (30 Questions)

**A1.** Explain the ReAct pattern. What does ReAct stand for?
**A2.** Why is MAX_ITERATIONS essential in every agent loop?
**A3.** How does native tool calling differ from parsing JSON from free text?
**A4.** What is the finish_reason field in OpenAI responses and what values can it have?
**A5.** How do you handle a tool that times out in an agent loop?
**A6.** What is prompt injection in the context of agents? Give an example.
**A7.** What is indirect prompt injection? How does it differ from direct?
**A8.** How do you implement a tool call budget per agent run?
**A9.** What is the orchestrator-worker pattern in multi-agent systems?
**A10.** When should you use a workflow instead of an agent?
**A11.** What is MCP (Model Context Protocol)? What problem does it solve?
**A12.** What are MCP resources vs MCP tools?
**A13.** How do you prevent an agent from calling sensitive tools (delete, email)?
**A14.** What is agent memory? Describe short-term and long-term memory.
**A15.** How do you summarize conversation history to handle context overflow?
**A16.** What is a parallel tool call? Give a use case.
**A17.** How do you validate tool arguments before executing a tool?
**A18.** What is the difference between `tool_choice="auto"` and `tool_choice="required"`?
**A19.** How do you test an agent reliably in CI/CD?
**A20.** What makes agents more expensive than RAG pipelines?
**A21.** How do you design a tool schema for maximum LLM tool selection accuracy?
**A22.** What is a "tool error" in an agent context? How should it be handled?
**A23.** Explain the concept of planning in AI agents.
**A24.** What is a hierarchical agent system? When would you use one?
**A25.** How do you log and audit agent tool calls for compliance?
**A26.** What are the main failure modes of production AI agents?
**A27.** How do you implement human-in-the-loop for sensitive agent actions?
**A28.** What is the difference between a stateless and stateful agent?
**A29.** How do you pass structured context between agent iterations?
**A30.** Design an agent that researches a topic and produces a structured report.

---

## FASTAPI (20 Questions)

**F1.** Why is FastAPI better than Flask for AI backends?
**F2.** What does `Depends()` do in FastAPI?
**F3.** How do you implement API key authentication in FastAPI?
**F4.** What is a Pydantic `field_validator`? When do you use it?
**F5.** How do you stream LLM responses in FastAPI?
**F6.** What is `StreamingResponse` and when do you use it?
**F7.** How do you add rate limiting to a FastAPI endpoint?
**F8.** What is the `lifespan` context manager in FastAPI?
**F9.** How do you handle exceptions globally in FastAPI?
**F10.** What is dependency injection and why is it useful for testing?
**F11.** How do you document an AI API endpoint with FastAPI?
**F12.** What is the difference between `async def` and `def` in FastAPI endpoints?
**F13.** How do you add CORS middleware to a FastAPI app?
**F14.** What is `response_model` and why is it important?
**F15.** How do you write integration tests for a FastAPI AI endpoint?
**F16.** What is `httpx.AsyncClient` used for in FastAPI testing?
**F17.** How do you add request ID tracking across all endpoints?
**F18.** What is Server-Sent Events (SSE) and how do you implement it?
**F19.** How do you handle file uploads (documents) in FastAPI?
**F20.** What is uvicorn and how do you configure it for production?

---

## DOCKER (20 Questions)

**D1.** What is the difference between an image and a container?
**D2.** Why should you copy requirements.txt before application code?
**D3.** What is a HEALTHCHECK in Dockerfile and why is it important?
**D4.** How do you run a container as a non-root user?
**D5.** What is Docker Compose and when would you use it over plain Docker?
**D6.** How do you pass environment variables to a Docker container?
**D7.** What is a Docker volume and when do you need one?
**D8.** How do you check the logs of a running container?
**D9.** What is layer caching and how does it affect build time?
**D10.** What does `depends_on` with `condition: service_healthy` do?
**D11.** How do you reduce Docker image size for an AI application?
**D12.** What is the difference between `CMD` and `ENTRYPOINT` in Dockerfile?
**D13.** How do you debug a container that won't start?
**D14.** What is a multi-stage build and when is it useful for AI?
**D15.** How do you configure Nginx as a reverse proxy for FastAPI?
**D16.** How do you implement a CI/CD pipeline that deploys a Docker container?
**D17.** What is `docker system prune` and when should you use it?
**D18.** How do you override the CMD in a docker-compose service?
**D19.** What is a `.dockerignore` file and what should it contain?
**D20.** What is the difference between `bridge` and `host` network mode?

---

## SYSTEM DESIGN (30 Questions)

**SD1.** Design a RAG system for a company with 1 million documents.
**SD2.** How would you scale a RAG system from 10 to 10,000 concurrent users?
**SD3.** Design the database schema for an AI chat application.
**SD4.** How do you handle LLM provider outages in production?
**SD5.** Design a cost optimization strategy for a high-volume LLM application.
**SD6.** How do you implement document-level access control in a RAG system?
**SD7.** Design an evaluation system for a production LLM application.
**SD8.** How would you migrate a RAG system from ChromaDB to pgvector?
**SD9.** Design a streaming chat API that handles 1000 concurrent users.
**SD10.** How do you implement a multi-tenant AI assistant?
**SD11.** What is the role of Redis in an AI backend architecture?
**SD12.** How do you design a background job system for document ingestion?
**SD13.** Design an LLM cost monitoring and alerting system.
**SD14.** How would you A/B test different prompts in production?
**SD15.** Design a system that automatically re-indexes documents when they change.
**SD16.** How do you handle PII in a production RAG system?
**SD17.** What are the trade-offs between self-hosted LLMs vs API LLMs?
**SD18.** How do you design for 99.9% uptime in an AI system?
**SD19.** Design a multi-language support system for a RAG chatbot.
**SD20.** How do you implement semantic caching (beyond exact-match cache)?
**SD21.** What is a circuit breaker pattern for LLM API calls?
**SD22.** Design a conversation history management system.
**SD23.** How do you handle context window limits in a long conversation?
**SD24.** Design a document chunking pipeline that handles PDFs, Word, and HTML.
**SD25.** How would you build a private LLM deployment for an enterprise?
**SD26.** Design an AI application that can answer questions about SQL databases.
**SD27.** How do you implement user feedback (thumbs up/down) for LLM responses?
**SD28.** Design a system that detects when your RAG quality degrades.
**SD29.** How do you implement streaming with source citations in RAG?
**SD30.** Design the complete architecture for the Enterprise AI Knowledge Assistant.

---

## AI SECURITY (20 Questions)

**SEC1.** What is prompt injection? Give a concrete attack example.
**SEC2.** What is indirect prompt injection? How is it different from direct?
**SEC3.** How do you structurally separate system instructions from user input?
**SEC4.** What regex patterns should you scan for in user input?
**SEC5.** How do you detect PII before sending data to external LLM APIs?
**SEC6.** What are output guardrails? Name 4 things they should check.
**SEC7.** How do you implement tool permission control in an agent system?
**SEC8.** What is the principle of least privilege applied to AI agents?
**SEC9.** How do you audit AI system actions for compliance?
**SEC10.** What is a jailbreak? How does it differ from prompt injection?
**SEC11.** How do you prevent sensitive data from appearing in LLM logs?
**SEC12.** What security concerns arise when using RAG with public documents?
**SEC13.** How do you implement document-level access control in RAG?
**SEC14.** What is a secondary injection in the context of agent tool results?
**SEC15.** How do you safely pass user data to external tools in an agent?
**SEC16.** What are the risks of using `eval()` in any AI context?
**SEC17.** How do you rate limit to prevent cost abuse in an AI API?
**SEC18.** What is the difference between authentication and authorization in AI APIs?
**SEC19.** How do you handle secrets (API keys) in a containerized AI application?
**SEC20.** What is the OWASP Top 10 for LLM applications?

---

## LLMOPS (20 Questions)

**LO1.** What is LLMOps and how does it differ from MLOps?
**LO2.** What should be in every structured log entry for an LLM call?
**LO3.** What metrics should you track for a production LLM application?
**LO4.** How do you version prompts like code?
**LO5.** What is a regression in the context of LLM applications?
**LO6.** How do you detect when LLM output quality has degraded?
**LO7.** What is a prompt monitoring dashboard? What does it track?
**LO8.** How do you implement cost per user tracking?
**LO9.** What is the p95 latency and why is it more useful than average?
**LO10.** How do you set up a cost alert for an LLM application?
**LO11.** What is trace ID propagation and why does it matter?
**LO12.** How do you run eval regressions in CI/CD for prompt changes?
**LO13.** What is A/B testing for prompts? How do you implement it?
**LO14.** How do you monitor for hallucination rates in production?
**LO15.** What is LLM observability vs traditional application observability?
**LO16.** How do you implement token budget per user per day?
**LO17.** What is semantic versioning for ML models?
**LO18.** How do you roll back a prompt change in production?
**LO19.** What is feedback loop in LLMOps?
**LO20.** How do you handle a sudden increase in LLM API costs?
