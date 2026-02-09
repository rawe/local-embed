# Local Embedding Test

A minimal end-to-end test that chunks a text file and sends each chunk to the embedding service.

## Prerequisites

The embedding service must be running on `localhost:8000`.

## Start the service

From the project root:

```
uv run uvicorn embed_provider.api:app --port 8000
```

## Run the example

From this directory:

```
uv run python run.py
```

## Expected output

```
Reading space-exploration.txt...
Split into 4 chunks (max 1000 chars each)

Chunk 1 (952 chars):
  "The history of space exploration is one of humanity's greate..."
  -> Embedding dim: 768, first values: [0.0123, -0.0341, 0.0512, ...]
...
Done. All chunks embedded successfully.
```

Dimension and values depend on the model configured in the service.
