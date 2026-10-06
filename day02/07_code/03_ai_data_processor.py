"""
Day 02 — AI Data Processing CLI (Mini Project)
===============================================
A complete data processing pipeline for RAG ingestion.

Demonstrates:
- File handling (text, JSON, CSV)
- Type hints throughout
- Dataclasses for structured data
- Generators for memory-efficient processing
- Comprehensions for batch operations
- Exception handling for file errors
- JSON output
- Token estimation

Run: python 03_ai_data_processor.py [--demo]
"""

import csv
import json
import math
import sys
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from collections.abc import Generator


# ─────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    """A document before processing."""
    path: str
    content: str
    file_type: str  # "txt", "json", "csv"
    size_bytes: int


@dataclass
class ProcessedChunk:
    """A processed chunk ready for embedding."""
    chunk_id: str
    source_path: str
    text: str
    chunk_index: int
    token_estimate: int
    word_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingReport:
    """Summary of a processing run."""
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    total_words: int = 0
    skipped_documents: int = 0
    errors: list[str] = field(default_factory=list)
    cost_estimates: dict[str, float] = field(default_factory=dict)

    def add_chunk(self, chunk: ProcessedChunk) -> None:
        self.total_chunks += 1
        self.total_tokens += chunk.token_estimate
        self.total_words += chunk.word_count

    def calculate_costs(self) -> None:
        """Estimate embedding and LLM costs."""
        embedding_cost_per_1m = 0.02  # text-embedding-3-small
        self.cost_estimates = {
            "embedding_usd": (self.total_tokens / 1_000_000) * embedding_cost_per_1m,
            "gpt4o_input_usd": (self.total_tokens / 1_000_000) * 5.0,
            "claude_input_usd": (self.total_tokens / 1_000_000) * 3.0,
        }


# ─────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize text for RAG ingestion.
    Remove excessive whitespace while preserving paragraph breaks.
    """
    if not text:
        return ""
    # Split into paragraphs
    paragraphs = text.split("\n\n")
    # Clean each paragraph
    cleaned_paragraphs = [" ".join(p.split()) for p in paragraphs if p.strip()]
    return "\n\n".join(cleaned_paragraphs)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using the 4-chars-per-token heuristic.
    Accurate to within ~20% of actual tokenization.
    For production: use tiktoken for exact counts.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_length: int = 50,
) -> Generator[str, None, None]:
    """
    Split text into overlapping chunks for RAG.

    Args:
        text: Input text to chunk
        chunk_size: Target characters per chunk
        overlap: Characters to overlap between chunks
        min_chunk_length: Skip chunks shorter than this

    Yields:
        Text chunks
    """
    if not text or len(text) < min_chunk_length:
        return

    # Try to split on sentence boundaries first
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a sentence boundary
        chunk = text[start:end]
        if end < len(text):
            # Look for last sentence end in chunk
            for delimiter in [". ", "! ", "? ", "\n\n", "\n"]:
                last_break = chunk.rfind(delimiter)
                if last_break > chunk_size * 0.5:  # At least half the chunk
                    end = start + last_break + len(delimiter)
                    chunk = text[start:end]
                    break

        chunk = chunk.strip()
        if len(chunk) >= min_chunk_length:
            yield chunk

        if end >= len(text):
            break

        # Advance with overlap
        start = end - overlap


# ─────────────────────────────────────────────────────────
# DOCUMENT LOADERS
# ─────────────────────────────────────────────────────────

def load_text_file(path: str) -> RawDocument:
    """Load a plain text document."""
    file_path = Path(path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    return RawDocument(
        path=str(file_path),
        content=content,
        file_type="txt",
        size_bytes=file_path.stat().st_size,
    )


def load_json_file(path: str) -> RawDocument:
    """
    Load a JSON document.
    Handles: {"text": "..."} or [{"text": "..."}, ...] formats.
    """
    file_path = Path(path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract text content
    if isinstance(data, dict):
        # Try common text fields
        for field_name in ["text", "content", "body", "description"]:
            if field_name in data:
                content = str(data[field_name])
                break
        else:
            # Fall back to serializing the whole dict
            content = json.dumps(data, indent=2)
    elif isinstance(data, list):
        # Join all text fields
        texts = []
        for item in data:
            if isinstance(item, dict):
                for field_name in ["text", "content", "body"]:
                    if field_name in item:
                        texts.append(str(item[field_name]))
                        break
            elif isinstance(item, str):
                texts.append(item)
        content = "\n\n".join(texts)
    else:
        content = str(data)

    return RawDocument(
        path=str(file_path),
        content=content,
        file_type="json",
        size_bytes=file_path.stat().st_size,
    )


def load_csv_file(path: str, text_column: Optional[str] = None) -> RawDocument:
    """
    Load a CSV and concatenate text columns.
    Auto-detects the text column if not specified.
    """
    file_path = Path(path)
    texts: list[str] = []

    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames:
            # Auto-detect text column if not specified
            if text_column is None:
                for candidate in ["text", "content", "body", "description", "review"]:
                    if candidate in reader.fieldnames:
                        text_column = candidate
                        break
                else:
                    # Use the longest column name as heuristic
                    text_column = max(reader.fieldnames, key=len)

            for row in reader:
                value = row.get(text_column, "")
                if value and value.strip():
                    texts.append(value.strip())

    content = "\n\n".join(texts)

    return RawDocument(
        path=str(file_path),
        content=content,
        file_type="csv",
        size_bytes=file_path.stat().st_size,
    )


def load_document(path: str) -> RawDocument:
    """Load a document based on its file extension."""
    ext = Path(path).suffix.lower()
    loaders = {
        ".txt": load_text_file,
        ".md": load_text_file,
        ".json": load_json_file,
        ".csv": load_csv_file,
    }

    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader(path)


# ─────────────────────────────────────────────────────────
# PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────

def process_document(
    doc: RawDocument,
    chunk_size: int = 512,
    overlap: int = 50,
) -> Generator[ProcessedChunk, None, None]:
    """
    Process a single document into chunks.
    Generator — yields chunks one at a time (memory efficient).
    """
    cleaned = clean_text(doc.content)

    for chunk_idx, chunk_text_content in enumerate(
        chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    ):
        source_name = Path(doc.path).stem
        chunk_id = f"{source_name}_chunk_{chunk_idx:04d}"

        yield ProcessedChunk(
            chunk_id=chunk_id,
            source_path=doc.path,
            text=chunk_text_content,
            chunk_index=chunk_idx,
            token_estimate=estimate_tokens(chunk_text_content),
            word_count=len(chunk_text_content.split()),
            metadata={
                "source": doc.path,
                "file_type": doc.file_type,
                "chunk_index": chunk_idx,
            },
        )


def process_directory(
    directory: str,
    extensions: Optional[list[str]] = None,
    chunk_size: int = 512,
    overlap: int = 50,
) -> tuple[list[ProcessedChunk], ProcessingReport]:
    """
    Process all documents in a directory.
    Returns chunks and a processing report.
    """
    if extensions is None:
        extensions = [".txt", ".md", ".json", ".csv"]

    report = ProcessingReport()
    all_chunks: list[ProcessedChunk] = []

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all matching files
    files = [
        f for ext in extensions
        for f in dir_path.glob(f"**/*{ext}")
        if f.is_file()
    ]

    report.total_documents = len(files)

    for file_path in sorted(files):
        try:
            doc = load_document(str(file_path))
            chunks = list(process_document(doc, chunk_size, overlap))

            all_chunks.extend(chunks)
            for chunk in chunks:
                report.add_chunk(chunk)

        except (ValueError, OSError, json.JSONDecodeError) as e:
            report.skipped_documents += 1
            report.errors.append(f"{file_path.name}: {e}")

    report.calculate_costs()
    return all_chunks, report


# ─────────────────────────────────────────────────────────
# SAMPLE DATA CREATION
# ─────────────────────────────────────────────────────────

def create_sample_data(directory: str) -> None:
    """Create sample documents for demonstration."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Text file
    (dir_path / "rag_guide.txt").write_text("""
Retrieval-Augmented Generation (RAG) Guide

Introduction
RAG is a technique that combines information retrieval with text generation.
It was introduced to address two critical problems with large language models:
hallucination and knowledge cutoffs.

How RAG Works
The RAG pipeline has two main phases: ingestion and retrieval.

During ingestion, documents are loaded and cleaned. Then they are split into
chunks of approximately 500 tokens each. Each chunk is converted to a dense
vector embedding using a pre-trained model. These embeddings are stored in a
vector database along with the original text.

During query time, the user's question is also converted to a vector embedding.
The vector database is searched for the most similar chunks using cosine
similarity. The top-K chunks are retrieved and combined into a context.

The LLM then generates an answer based on the retrieved context and the
original question. This grounds the answer in real documents, dramatically
reducing hallucination.

When to Use RAG
Use RAG when you need to answer questions about specific documents.
Use RAG when your knowledge base changes frequently.
Use RAG when you need to cite sources in your answers.
Do NOT use RAG when the LLM already has the knowledge in its training data.
""".strip(), encoding="utf-8")

    # JSON file
    (dir_path / "ai_concepts.json").write_text(json.dumps({
        "text": (
            "Embeddings are dense vector representations of text that capture semantic meaning. "
            "Unlike one-hot encodings, embeddings place similar concepts near each other in vector space. "
            "For example, the embeddings for 'king' and 'queen' will be geometrically close. "
            "Embeddings are used in RAG for both document storage and query matching. "
            "The most common similarity measure for embeddings is cosine similarity, "
            "which measures the angle between two vectors rather than their absolute distance. "
            "This makes it scale-invariant: a long and short document about the same topic "
            "will have similar cosine similarity despite different vector magnitudes."
        )
    }, indent=2), encoding="utf-8")

    # CSV file
    csv_data = [
        {"id": "1", "text": "Large Language Models are trained on massive text corpora to predict the next token."},
        {"id": "2", "text": "The transformer architecture uses self-attention to process sequences in parallel."},
        {"id": "3", "text": "Fine-tuning adapts a pre-trained model to a specific task using supervised examples."},
        {"id": "4", "text": "Prompt engineering involves crafting inputs that guide LLM behavior toward desired outputs."},
        {"id": "5", "text": "Vector databases use approximate nearest neighbor algorithms for fast similarity search."},
    ]
    with open(dir_path / "llm_facts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(csv_data)


# ─────────────────────────────────────────────────────────
# CLI DISPLAY
# ─────────────────────────────────────────────────────────

def display_report(report: ProcessingReport) -> None:
    """Display a formatted processing report."""
    print("\n" + "═" * 60)
    print("  PROCESSING REPORT")
    print("═" * 60)
    print(f"  Documents processed: {report.total_documents - report.skipped_documents}")
    print(f"  Documents skipped:   {report.skipped_documents}")
    print(f"  Total chunks:        {report.total_chunks:,}")
    print(f"  Total words:         {report.total_words:,}")
    print(f"  Total tokens (est):  {report.total_tokens:,}")

    print("\n  Cost Estimates:")
    for service, cost in report.cost_estimates.items():
        print(f"    {service:<25} ${cost:.4f}")

    if report.errors:
        print(f"\n  Errors ({len(report.errors)}):")
        for error in report.errors[:5]:
            print(f"    ✗ {error}")


def display_chunks(chunks: list[ProcessedChunk], max_display: int = 5) -> None:
    """Display processed chunks."""
    print(f"\n  First {min(max_display, len(chunks))} chunks of {len(chunks)} total:")
    print("  " + "─" * 58)
    for chunk in chunks[:max_display]:
        print(f"  [{chunk.chunk_id}]")
        print(f"    Tokens: ~{chunk.token_estimate} | Words: {chunk.word_count}")
        print(f"    Text: {chunk.text[:80]}...")
        print()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 02 — AI DATA PROCESSING CLI                   ║")
    print("║        Preparing Documents for RAG Ingestion             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Create sample data
    sample_dir = "sample_documents"
    print(f"\nCreating sample documents in '{sample_dir}/'...")
    create_sample_data(sample_dir)

    # Process the directory
    print(f"Processing documents...")
    try:
        chunks, report = process_directory(
            sample_dir,
            chunk_size=300,
            overlap=50,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Display results
    display_report(report)
    display_chunks(chunks)

    # Save chunks to JSON (ready for embedding)
    output_path = "processed_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)

    print(f"\n  ✓ Saved {len(chunks)} chunks to '{output_path}'")
    print(f"  → Next step: embed each chunk and store in vector DB")

    # Cleanup
    import shutil
    if "--no-cleanup" not in sys.argv:
        shutil.rmtree(sample_dir, ignore_errors=True)
        os.remove(output_path) if os.path.exists(output_path) else None
        print("  ✓ Cleaned up sample files")


if __name__ == "__main__":
    main()
