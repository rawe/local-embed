---
name: rag
description: >
  Semantic search over local text files using the project's RAG indexer.
  Use when the user wants to index text files or directories for search,
  query indexed documents with natural language, find which document contains
  specific information, or answer questions from indexed content.
  Trigger phrases: "index files", "search documents", "query", "find in docs",
  "which file mentions", "RAG", "semantic search", "look up in indexed files".
allowed-tools: Bash, Read, Glob
argument-hint: <query or file path>
---

# RAG — Local Semantic Search

This skill uses `examples/rag/rag.py` to index text files and query them via embeddings.

**Prerequisite:** The embedding service must be running:

```bash
cd /Users/ramon/Documents/Projects/ai/embedding-service
EMBED_E5_MODE=none uv run uvicorn embed_provider.api:app --port 8000
```

Check with: `curl -s http://localhost:8000/health`

## Commands

All commands run from `examples/rag/`.

### Index files

Single file:
```bash
uv run python examples/rag/rag.py index <file_path>
```

Glob pattern (quote it!):
```bash
uv run python examples/rag/rag.py index "<glob_pattern>"
```

Example: `uv run python examples/rag/rag.py index "../test-data/*.txt"`

Indexing is additive — each call appends to the existing index.

### Query

```bash
uv run python examples/rag/rag.py query "<natural language question>" --top-k <N> --min-score <threshold>
```

- `--top-k` (default 3): max number of results.
- `--min-score` (optional): minimum similarity score (e.g. 0.75). Results below this are excluded.
  **Always use `--min-score` when querying.** Default to `0.75` for quality results. Use `0.8+` when the user wants a high bar. Only go below `0.6` if you need broad, exploratory recall.

Output is JSON to stdout with `query` and `results` (each has `source`, `chunk_index`, `text`, `score`).

### Clean index

```bash
uv run python examples/rag/rag.py clean
```

## How to answer user questions with RAG

1. **Run the query** with `rag.py query` and capture the JSON output.
2. **Answer using only the returned `text` fields.** Do not hallucinate beyond what the chunks say.
3. **Cite the source file** (`source` field) for each piece of information.
4. **Score guidance:** 0.8+ = strong match, 0.6-0.8 = good, <0.5 = weak.
5. If the user wants more detail and the answer chunk is insufficient, **ask permission** to read the full source file from `examples/test-data/<source>` (or wherever it was indexed from) for deeper context.
6. If no results are relevant (all scores < 0.4), say so — do not guess.

## Notes

- Supported files: plain text (`.txt`). Other formats are read as raw text.
- Chunks are ~1,000 characters, split at whitespace boundaries.
- The index lives at `examples/rag/rag_data/rag_index.json`.
- Test data is in `examples/test-data/` (25 topic files).
