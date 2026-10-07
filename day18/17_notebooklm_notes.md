# Day 18 — NotebookLM Notes: Vector Databases

## Why Vector DBs Exist

SQL cannot efficiently answer "find top-5 most similar vectors". Vector DBs use ANN (Approximate Nearest Neighbor) algorithms — O(log N) vs O(N) linear scan.

## ANN Algorithms

**HNSW**: Hierarchical Navigable Small World. Graph-based. Fast, accurate, high memory. Default in ChromaDB.
**IVF**: Inverted File Index. Cluster-based. Less memory, slightly less accurate. Used in FAISS IVFFlat.

## Operations

```python
# ChromaDB (development)
collection.add(ids, documents, embeddings, metadatas)
collection.query(query_embeddings, n_results, where)  # where = metadata filter
collection.delete(ids)
collection.get(ids)

# FAISS (production performance)
index.add(vectors)           # Add float32 numpy array
distances, indices = index.search(query, k)  # returns top-k
faiss.write_index(index, path)  # save
faiss.read_index(path)           # load

# pgvector (SQL + vectors)
# embedding <=> query_vec  ← cosine distance
# embedding <-> query_vec  ← L2 distance
# embedding <#> query_vec  ← inner product
```

## Selection Guide

```
Prototyping / development → ChromaDB (simple, embedded)
High performance          → FAISS (in-process, no network)
Existing Postgres         → pgvector (minimal infra change)
Large scale managed       → Pinecone
Open-source production    → Qdrant
```

## Interview Facts

1. ChromaDB returns cosine DISTANCE (0=identical, 2=opposite); score = 1 - distance
2. FAISS IndexFlatIP + normalized vectors = exact cosine similarity
3. IVF requires training on a sample before use (learns cluster centers)
4. pgvector operators: `<=>` cosine, `<->` L2, `<#>` inner product
5. HNSW build time is slow; search time is very fast
6. Metadata filtering + vector search = hybrid approach in most vector DBs

## Common Mistakes

- Not normalizing vectors before inner product search (wrong distances)
- Using chromadb.Client() in production (in-memory, data lost on restart)
- Missing metadata → can't filter later → add it at indexing time
- Not creating HNSW/IVF index in pgvector → full table scan every query
- Changing embedding model → all existing vectors are incompatible
