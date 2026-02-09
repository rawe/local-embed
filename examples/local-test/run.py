"""Minimal local test: chunk a text file and get embeddings from the service."""

import json
import os
import sys
import urllib.error
import urllib.request

URL = "http://localhost:8000/v1/embeddings"
CHUNK_SIZE = 1000


def chunk_text(text, max_chars=CHUNK_SIZE):
    """Split text into chunks of roughly max_chars, breaking at whitespace."""
    chunks = []
    while len(text) > max_chars:
        # Find the last space or newline within the limit
        break_at = text.rfind(" ", 0, max_chars)
        if break_at == -1:
            break_at = max_chars
        chunks.append(text[:break_at].strip())
        text = text[break_at:].strip()
    if text:
        chunks.append(text.strip())
    return chunks


def get_embedding(text):
    """Send a single text to the embedding service and return the vector."""
    body = json.dumps({"input": text}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["embedding"]


def main():
    # Read the sample file from the shared test-data directory
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "test-data", "space-exploration.txt")

    print(f"Reading {os.path.basename(path)}...")
    with open(path) as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks (max {CHUNK_SIZE} chars each)\n")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:60].replace("\n", " ")
        print(f'Chunk {i} ({len(chunk)} chars):')
        print(f'  "{preview}..."')

        try:
            emb = get_embedding(chunk)
        except urllib.error.URLError as e:
            print(
                f"\nERROR: Could not connect to {URL}\n"
                f"  {e}\n\n"
                "Is the embedding service running? Start it with:\n"
                "  uv run uvicorn embed_provider.api:app --port 8000",
                file=sys.stderr,
            )
            sys.exit(1)

        first5 = [round(v, 4) for v in emb[:5]]
        print(f"  -> Embedding dim: {len(emb)}, first values: {first5}\n")

    print("Done. All chunks embedded successfully.")


if __name__ == "__main__":
    main()
