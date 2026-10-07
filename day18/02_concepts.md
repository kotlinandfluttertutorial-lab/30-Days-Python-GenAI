# Day 18 — Concepts
## Vector Databases

---

## 1. Why Vector Databases?

```
SQL/NoSQL databases: optimized for exact queries ("WHERE id = 5")
Vector databases:    optimized for similarity queries ("find top-K nearest vectors")

In-memory NumPy:     O(N) linear scan — fine for <100K vectors
Vector database:     O(log N) approximate search — required for millions of vectors

ANN algorithms:
- HNSW (Hierarchical Navigable Small World): fast, accurate, high memory
- IVF (Inverted File Index): less memory, slightly less accurate
- FAISS IVFFlat: Facebook's optimized IVF
```

---

## 2. ChromaDB (Development / Small-Medium Scale)

```python
import chromadb
from chromadb.utils import embedding_functions

# Embedded mode (no server needed — great for development)
client = chromadb.Client()  # in-memory
# OR
client = chromadb.PersistentClient(path="./chroma_db")  # persists to disk

# Create collection
collection = client.create_collection(
    name="ai_documents",
    metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    ),
)

# Add documents (ChromaDB embeds automatically if you provide embedding_function)
collection.add(
    documents=["RAG reduces hallucination", "Embeddings capture meaning"],
    metadatas=[{"category": "rag"}, {"category": "embeddings"}],
    ids=["doc_001", "doc_002"],
)

# Query
results = collection.query(
    query_texts=["How does RAG work?"],
    n_results=3,
    where={"category": "rag"},         # Metadata filter
)
print(results["documents"])
print(results["distances"])
print(results["metadatas"])

# Or use pre-computed embeddings
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # your own embeddings
    documents=["text1", "text2"],
    ids=["id1", "id2"],
)
```

---

## 3. FAISS (High Performance, In-Process)

```python
import faiss
import numpy as np

# FAISS: Facebook AI Similarity Search
# Pure C++ with Python bindings — very fast
# No server, runs in-process

dims = 384
n_docs = 10000

# Flat L2 index (exact, slower)
index_l2 = faiss.IndexFlatL2(dims)

# Flat Inner Product (exact, fast for normalized vectors = cosine)
index_ip = faiss.IndexFlatIP(dims)

# IVF for large scale (approximate, faster)
nlist = 100  # Number of inverted lists (clusters)
quantizer = faiss.IndexFlatIP(dims)
index_ivf = faiss.IndexIVFFlat(quantizer, dims, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(np.random.randn(10000, dims).astype("float32"))  # Train on sample

# Add vectors
vectors = np.random.randn(n_docs, dims).astype("float32")
faiss.normalize_L2(vectors)  # Normalize for inner product = cosine

index_ip.add(vectors)
print(f"Index size: {index_ip.ntotal}")

# Search
query = np.random.randn(1, dims).astype("float32")
faiss.normalize_L2(query)

distances, indices = index_ip.search(query, k=5)  # top-5
print(f"Top-5 indices: {indices[0]}")
print(f"Top-5 scores:  {distances[0]}")

# Save/Load
faiss.write_index(index_ip, "index.faiss")
loaded_index = faiss.read_index("index.faiss")
```

---

## 4. pgvector (PostgreSQL Extension)

```python
# pgvector: vector search inside PostgreSQL
# Best for: when you already use PostgreSQL, avoid adding another service

# Setup:
# CREATE EXTENSION vector;
# CREATE TABLE documents (
#     id serial PRIMARY KEY,
#     content text,
#     embedding vector(384),
#     metadata jsonb
# );
# CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
#     WITH (lists = 100);

import psycopg2
import numpy as np

conn = psycopg2.connect("postgresql://user:password@localhost/dbname")
cur = conn.cursor()

# Insert
embedding = np.random.randn(384).tolist()
cur.execute(
    "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
    ("RAG reduces hallucination", embedding, '{"category": "rag"}')
)

# Similarity search (cosine)
query_emb = np.random.randn(384).tolist()
cur.execute("""
    SELECT id, content, 1 - (embedding <=> %s::vector) AS score
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_emb, query_emb))

for row in cur.fetchall():
    print(f"  [{row[2]:.4f}] {row[1]}")
```

---

## 5. Which Vector Database to Choose?

```
DEVELOPMENT:        ChromaDB (simple, embedded, auto-embedding)
PRODUCTION SMALL:   ChromaDB Persistent or FAISS
PRODUCTION LARGE:   Pinecone (managed) or Qdrant (open-source)
EXISTING POSTGRES:  pgvector (minimal infrastructure change)
HIGHEST PERF:       FAISS (in-process, no network latency)
COMPLEX QUERIES:    Weaviate or Qdrant (support hybrid search natively)

Decision tree:
Already use Postgres? → pgvector
Need managed service? → Pinecone
Need hybrid search?   → Qdrant
Building RAG quickly? → ChromaDB
High throughput?      → FAISS + custom service
```

---

## 6. Key Operations Every Vector DB Must Support

```python
# CRUD operations
vector_db.add(id, vector, text, metadata)
vector_db.get(id)
vector_db.update(id, new_vector, new_metadata)
vector_db.delete(id)

# Search operations
vector_db.search(query_vector, top_k=5)                          # Basic
vector_db.search(query_vector, top_k=5, filter={"cat": "rag"})  # Filtered
vector_db.search(query_vector, top_k=5, score_threshold=0.7)    # Threshold

# Batch operations
vector_db.add_batch(documents)
vector_db.search_batch(queries, top_k=5)
```
