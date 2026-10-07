# Day 13 — Concepts
## NLP Fundamentals

---

## 1. Tokenization

```python
# Tokenization: splitting text into units the model can process

# Word tokenization (simple, ignores morphology)
text = "The quick brown fox jumped"
words = text.split()  # ["The", "quick", "brown", "fox", "jumped"]

# Subword tokenization (modern — what LLMs use)
# BPE (Byte-Pair Encoding): merge frequent character pairs
# "tokenization" → ["token", "ization"]  (2 tokens)
# "supercalifragilistic" → ["super", "cal", "ifra", "gi", "listic"] (5 tokens)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer("What is RAG?", return_tensors="pt")
print(encoded["input_ids"])       # Token IDs
print(tokenizer.decode(encoded["input_ids"][0]))  # Back to text

# Special tokens:
# [CLS] = classification token, first token in BERT
# [SEP] = separator between two sentences
# [PAD] = padding to equal length
# [MASK] = masked for MLM pretraining
```

---

## 2. Vocabulary and Token IDs

```python
# Vocabulary: mapping of token → integer ID
# BERT vocabulary: ~30,000 tokens
# GPT-2/GPT-4 vocabulary: ~50,000 tokens (tiktoken BPE)

# Manual vocabulary
vocab = {"[PAD]": 0, "[UNK]": 1, "the": 2, "cat": 3, "sat": 4}
text_ids = [vocab.get(word, vocab["[UNK]"]) for word in "the cat sat".split()]
# text_ids = [2, 3, 4]

# Vocabulary size affects model size:
# embedding_params = vocab_size × embedding_dim
# 50000 × 768 = 38.4M parameters just for embedding layer!
```

---

## 3. TF-IDF (Term Frequency-Inverse Document Frequency)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# TF = how often a word appears in this document
# IDF = log(total_docs / docs_containing_word)
# TF-IDF = TF × IDF
# High TF-IDF = word is frequent in THIS doc but rare overall = important discriminator

corpus = [
    "machine learning is a subset of AI",
    "deep learning uses neural networks",
    "neural networks are used in machine learning",
    "AI systems can learn from data",
]

vectorizer = TfidfVectorizer(max_features=20, stop_words="english")
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

print("Top TF-IDF terms per document:")
for i, doc in enumerate(corpus):
    scores = zip(feature_names, X[i].toarray()[0])
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    print(f"  Doc {i+1}: {[f'{word}({score:.2f})' for word, score in top if score > 0]}")

# TF-IDF for RAG:
# Used in BM25 hybrid search (keyword-based retrieval alongside vector search)
# BM25 = improved TF-IDF with document length normalization
```

---

## 4. Word Embeddings

```python
# Word2Vec / GloVe: classic word embeddings
# Fixed vectors for each word (no context)
# "bank" has same vector whether financial or river bank

# Sentence embeddings: modern approach
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",  # Same meaning, different words
    "Machine learning is powerful",
]

embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")  # (3, 384)

# Similarity
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(embeddings)
print(f"Sentence 1 vs 2: {sims[0][1]:.3f}")  # High (same meaning)
print(f"Sentence 1 vs 3: {sims[0][2]:.3f}")  # Low (different topic)
```

---

## 5. Stop Words, Stemming, Lemmatization

```python
# These are classical NLP preprocessing steps
# Less important with modern LLMs (which handle them internally)
# BUT still used in: BM25/TF-IDF search, keyword extraction, text cleaning

# Stop words: common words with little meaning ("the", "is", "a")
stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
              "have", "has", "had", "do", "does", "did", "will", "would", "could",
              "should", "may", "might", "shall", "can", "need", "dare", "ought", "used",
              "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
              "through", "during", "before", "after", "above", "below", "between"}

def remove_stop_words(text: str) -> str:
    return " ".join(w for w in text.lower().split() if w not in stop_words)

# Stemming: chop suffix (fast but crude)
# "running" → "run", "studies" → "studi" (not a real word!)
# Use: Porter Stemmer, Snowball

# Lemmatization: find actual root form (slower but accurate)
# "running" → "run", "studies" → "study", "better" → "good"
# Use: spaCy or NLTK WordNetLemmatizer

# Example with spaCy
# import spacy
# nlp = spacy.load("en_core_web_sm")
# doc = nlp("The quick brown foxes were running")
# lemmas = [token.lemma_ for token in doc]
# → ["the", "quick", "brown", "fox", "be", "run"]
```

---

## 6. Text Classification Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Classical text classification (pre-LLM)
# Still useful when you have limited data or need explainability

texts = [
    "I love this product, it's amazing!",
    "Great quality, highly recommend",
    "Terrible, waste of money",
    "Awful experience, never buying again",
    "Okay product, nothing special",
    "So-so, could be better",
]
labels = [1, 1, 0, 0, 2, 2]  # 0=negative, 1=positive, 2=neutral

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["negative", "positive", "neutral"]))
```

---

## 7. Semantic Similarity

```python
# Semantic similarity: do two texts mean the same thing?
# Different from lexical similarity (same words)

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

pairs = [
    ("How do I reset my password?", "I forgot my password, what do I do?"),
    ("What are your business hours?", "When is the office open?"),
    ("Cancel my subscription", "How do I unsubscribe?"),
    ("What is RAG?", "Explain machine learning to me"),  # Different topic
]

for q1, q2 in pairs:
    e1, e2 = model.encode([q1, q2])
    sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    print(f"[{sim:.3f}] '{q1[:35]}' vs '{q2[:35]}'")
```
