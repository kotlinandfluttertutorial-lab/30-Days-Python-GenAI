# Day 19 — Day Summary: RAG Fundamentals

## What You Built
Complete basic RAG chatbot: ingestion pipeline (clean → chunk → embed → ChromaDB), query pipeline (embed → search → build prompt → LLM → answer with citations), interactive CLI.

## Key Takeaways
1. RAG is two pipelines: offline ingestion + online query
2. Same embedding model MUST be used for both indexing and querying
3. Chunking with overlap prevents information loss at boundaries
4. Context always includes source citations
5. The system prompt must instruct the LLM to only use provided context

## Tomorrow: Day 20 — Advanced RAG
Hybrid search (BM25 + vector), reranking, query rewriting, multi-query retrieval — all the techniques that improve RAG from "prototype" to "production quality".
