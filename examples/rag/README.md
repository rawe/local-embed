# RAG Example

A minimal Retrieval-Augmented Generation (RAG) CLI that indexes text files and retrieves relevant chunks via semantic search. Uses only Python stdlib -- no external dependencies beyond the embedding service.

## Prerequisites

The embedding service must be running on `localhost:8000`. For best results with E5 models, start it with the prefix mode set to `none` (the RAG script handles prefixing internally):

```
EMBED_E5_MODE=none uv run uvicorn embed_provider.api:app --port 8000
```

## Index a file

Sample text files are in `../test-data/` (relative to this directory):

```
uv run python rag.py index ../test-data/space-exploration.txt
```

You can index multiple files. Each call appends to the existing index:

```
uv run python rag.py index ../test-data/artificial-intelligence.txt
```

## Index multiple files

Use glob patterns to index all files matching a pattern in one command:

```
uv run python rag.py index "../test-data/*.txt"
```

Quote the pattern so the shell does not expand it before Python sees it.

## Query the index

```
uv run python rag.py query "Who walked on the Moon?"
```

Returns JSON to stdout with the top matching chunks:

```json
{
  "query": "Who walked on the Moon?",
  "results": [
    {
      "source": "space-exploration.txt",
      "chunk_index": 0,
      "text": "The history of space exploration is one of humanity's greatest...",
      "score": 0.8066
    }
  ]
}
```

Use `--top-k` to control the number of results (default: 3):

```
uv run python rag.py query "space stations" --top-k 5
```

Use `--min-score` to filter out low-relevance results:

```
uv run python rag.py query "When was Darwin born?" --min-score 0.8
```

The two flags combine: `--top-k 5 --min-score 0.75` returns up to 5 results, but only those scoring at or above 0.75.

## Clean the index

```
uv run python rag.py clean
```

This deletes the `rag_data/` directory and its contents.

## How it works

1. **Indexing:** Reads a text file, splits it into ~1000-character chunks at whitespace boundaries, sends all chunks to the embedding API in one batch (with "passage: " E5 prefix), and stores vectors + metadata in `rag_data/rag_index.json`.

2. **Querying:** Embeds the query string (with "query: " E5 prefix), computes cosine similarity against all stored vectors, and returns the top-k matches as JSON.

3. **Storage:** A single JSON file holds everything. Human-readable and simple. For this example's scale (dozens to hundreds of chunks), performance is not a concern.
