"""Minimal RAG CLI: index text files and query them via semantic search.

Uses only Python stdlib + the embedding service at localhost:8000.

Usage:
    python rag.py index <file>            # Index a text file
    python rag.py index "*.txt"           # Index all matching files (glob)
    python rag.py query "search text"     # Search indexed chunks
    python rag.py clean                   # Delete the index
"""

import argparse
import glob
import json
import math
import os
import shutil
import sys
import urllib.error
import urllib.request

URL = "http://localhost:8000/v1/embeddings"
CHUNK_SIZE = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "rag_data")
INDEX_PATH = os.path.join(DATA_DIR, "rag_index.json")


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text, max_chars=CHUNK_SIZE):
    """Split text into chunks of roughly max_chars, breaking at whitespace."""
    chunks = []
    while len(text) > max_chars:
        break_at = text.rfind(" ", 0, max_chars)
        if break_at == -1:
            break_at = max_chars
        chunks.append(text[:break_at].strip())
        text = text[break_at:].strip()
    if text:
        chunks.append(text.strip())
    return chunks


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------

def get_embeddings(texts):
    """Send a list of texts to the embedding service and return vectors."""
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(
            f"\nERROR: Could not connect to {URL}\n"
            f"  {e}\n\n"
            "Is the embedding service running? Start it with:\n"
            "  EMBED_E5_MODE=none uv run uvicorn embed_provider.api:app --port 8000",
            file=sys.stderr,
        )
        sys.exit(1)
    return [item["embedding"] for item in data["data"]]


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

def load_index():
    """Load the index from disk, or return an empty structure."""
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {"documents": []}


def save_index(index):
    """Write the index to disk, creating rag_data/ if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _index_single_file(path, index):
    """Index one text file into the given index. Returns number of chunks added."""
    source = os.path.basename(path)
    print(f"Reading {source}...", file=sys.stderr)
    with open(path) as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks (~{CHUNK_SIZE} chars each)", file=sys.stderr)

    # Prepend E5 passage prefix for indexing
    prefixed = [f"passage: {c}" for c in chunks]

    print("Embedding chunks...", file=sys.stderr)
    vectors = get_embeddings(prefixed)

    # Build document entries
    new_docs = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        new_docs.append({
            "source": source,
            "chunk_index": i,
            "text": chunk,
            "embedding": vec,
        })

    index["documents"].extend(new_docs)
    return len(new_docs)


def cmd_index(args):
    """Index a text file (or glob pattern): chunk, embed, and store."""
    pattern = args.file

    # Detect glob pattern
    if "*" in pattern or "?" in pattern:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"Error: no files matched pattern: {pattern}", file=sys.stderr)
            sys.exit(1)
        print(f"Matched {len(files)} file(s):", file=sys.stderr)
        for f in files:
            print(f"  {f}", file=sys.stderr)
        index = load_index()
        total_added = 0
        for filepath in files:
            path = os.path.abspath(filepath)
            if not os.path.isfile(path):
                print(f"Warning: skipping non-file: {path}", file=sys.stderr)
                continue
            total_added += _index_single_file(path, index)
        save_index(index)
        total = len(index["documents"])
        print(
            f"Indexed {total_added} chunks from {len(files)} file(s) "
            f"({total} total chunks in index)",
            file=sys.stderr,
        )
    else:
        path = os.path.abspath(pattern)
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        index = load_index()
        added = _index_single_file(path, index)
        save_index(index)
        total = len(index["documents"])
        print(
            f"Indexed {added} chunks from {os.path.basename(path)} "
            f"({total} total chunks in index)",
            file=sys.stderr,
        )


def cmd_query(args):
    """Query the index: embed the query, find similar chunks, output JSON."""
    index = load_index()
    if not index["documents"]:
        print("Error: index is empty. Run 'rag.py index <file>' first.", file=sys.stderr)
        sys.exit(1)

    query_text = args.query
    top_k = args.top_k
    min_score = args.min_score

    # Prepend E5 query prefix
    prefixed = [f"query: {query_text}"]

    print(f"Searching for: {query_text}", file=sys.stderr)
    query_vec = get_embeddings(prefixed)[0]

    # Score all chunks
    scored = []
    for doc in index["documents"]:
        score = cosine_similarity(query_vec, doc["embedding"])
        scored.append({
            "source": doc["source"],
            "chunk_index": doc["chunk_index"],
            "text": doc["text"],
            "score": round(score, 4),
        })

    # Sort by score descending, apply threshold, take top-k
    scored.sort(key=lambda x: x["score"], reverse=True)
    if min_score is not None:
        scored = [s for s in scored if s["score"] >= min_score]
    results = scored[:top_k]

    output = {
        "query": query_text,
        "results": results,
    }
    print(json.dumps(output, indent=2))


def cmd_clean(args):
    """Delete the rag_data/ directory."""
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print("Removed rag_data/ index directory.", file=sys.stderr)
    else:
        print("Nothing to clean (rag_data/ does not exist).", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Minimal RAG: index text files and query via semantic search"
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index a text file or glob pattern")
    p_index.add_argument("file", help="Path to a text file or glob pattern (e.g. '*.txt')")

    # query
    p_query = sub.add_parser("query", help="Search indexed chunks")
    p_query.add_argument("query", help="Search query text")
    p_query.add_argument(
        "--top-k", type=int, default=3, help="Number of results (default: 3)"
    )
    p_query.add_argument(
        "--min-score", type=float, default=None,
        help="Minimum similarity score threshold (e.g. 0.75)"
    )

    # clean
    sub.add_parser("clean", help="Delete the index")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "clean":
        cmd_clean(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
