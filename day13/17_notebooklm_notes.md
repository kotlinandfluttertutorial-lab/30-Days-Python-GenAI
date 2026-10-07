# Day 13 — NotebookLM Notes: NLP Fundamentals

## Key Concepts

**Tokenization** — Splitting text into units. Modern LLMs use BPE (Byte-Pair Encoding) subword tokenization. ~0.75 words per token on average. Vocabulary size: 30K-100K tokens.

**TF-IDF** — Term frequency × inverse document frequency. Measures word importance in a document relative to a corpus. Used in BM25 hybrid search alongside vector search.

**Word Embeddings** — Dense vectors representing meaning. Word2Vec/GloVe: fixed per word. Sentence Transformers: contextual, one vector per sentence/document.

**Semantic Similarity** — Meaning similarity (not word overlap). Two different phrasings of the same question → high cosine similarity. Critical for RAG retrieval.

**Stop Words** — Low-information words ("the", "is"). Remove for TF-IDF; irrelevant for LLMs.

**Lemmatization vs Stemming** — Lemma: real root word ("better"→"good"). Stem: crude suffix removal ("studies"→"studi"). Use lemmatization for NLP pipelines.

## Code Patterns

```python
# TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words="english")
X = vectorizer.fit_transform(corpus)

# Sentence embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)  # shape: (n, 384)

# HuggingFace tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello world", return_tensors="pt")
```

## Interview Facts

1. BPE: frequent character pair merges; handles rare/new words as character sequences
2. TF-IDF: high score = frequent in doc, rare in corpus → important discriminator
3. "bank" has same Word2Vec vector in all contexts; BERT gives contextual vectors
4. Semantic similarity ≠ lexical similarity ("dog" and "canine" = similar meaning, different words)
5. sentence-transformers `all-MiniLM-L6-v2`: 384 dims, fast, good quality, free
6. Vocabulary size × embedding dim = large portion of model parameters

## TF-IDF vs Embeddings

| Aspect | TF-IDF | Sentence Embedding |
|--------|--------|-------------------|
| Representation | Sparse (mostly zeros) | Dense (all non-zero) |
| Captures | Exact keyword match | Semantic meaning |
| Similarity | Keyword overlap | Semantic distance |
| Speed | Very fast | Requires model call |
| In RAG | BM25 component | Vector search component |
| Combined | Hybrid search (best) | — |
