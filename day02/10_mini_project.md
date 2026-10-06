# Day 02 — Mini Project
## AI Data Processing CLI

**Code:** `07_code/03_ai_data_processor.py`

---

## What You Build

A production-quality document processing pipeline that prepares text files for RAG ingestion. It demonstrates every Python pattern from today.

## What It Does

1. Loads documents from a directory (`.txt`, `.json`, `.csv`)
2. Cleans text (normalizes whitespace)
3. Chunks text into RAG-ready pieces with overlap
4. Estimates token counts and costs
5. Generates a processing report
6. Exports chunks as JSON ready for embedding

## Running It

```bash
# Run the demo (creates sample docs, processes them, cleans up)
python 03_ai_data_processor.py

# Keep the output files
python 03_ai_data_processor.py --no-cleanup
```

## Expected Output

```
╔══════════════════════════════════════════════════════════╗
║        DAY 02 — AI DATA PROCESSING CLI                   ║
╚══════════════════════════════════════════════════════════╝

Creating sample documents in 'sample_documents/'...
Processing documents...

════════════════════════════════════════════════════════════
  PROCESSING REPORT
════════════════════════════════════════════════════════════
  Documents processed: 3
  Documents skipped:   0
  Total chunks:        12
  Total words:         487
  Total tokens (est):  1,672

  Cost Estimates:
    embedding_usd             $0.0000
    gpt4o_input_usd           $0.0084
    claude_input_usd          $0.0050
```

## Extension Tasks

1. Add support for `.pdf` files using `pypdf`
2. Add a `--dry-run` flag that reports without saving
3. Add metadata enrichment (auto-detect language, estimate reading level)
4. Add deduplication (skip chunks that are too similar to existing ones)
