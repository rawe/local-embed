# Embedding Service

Multilingual embedding API powered by [sentence-transformers](https://www.sbert.net/) and FastAPI. Exposes an OpenAI-compatible `/v1/embeddings` endpoint using the `intfloat/multilingual-e5-base` model (768-dimensional embeddings).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and enter the project
cd embedding-service

# Create virtualenv and install dependencies
uv sync --group dev

# Run the API server
uv run uvicorn embed_provider.api:app --host 0.0.0.0 --port 8000

# Run the test suite
uv run pytest -v
```

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL_ID` | `intfloat/multilingual-e5-base` | HuggingFace model identifier |
| `EMBED_DEVICE` | `auto` | Compute device (`auto`, `cpu`, `cuda`, `mps`) |
| `EMBED_NORMALIZE` | `true` | Normalize embeddings to unit length |
| `EMBED_BATCH_SIZE` | `32` | Encoding batch size |
| `EMBED_E5_MODE` | `passage` | E5 prefix mode (`passage`, `query`, `none`) |
| `HF_HOME` | *(system default)* | HuggingFace cache directory for downloaded models |

### Device selection

When `EMBED_DEVICE=auto` (the default), the service picks the best available device:

1. **MPS** (Apple Silicon GPU) -- preferred on macOS
2. **CUDA** (NVIDIA GPU) -- preferred on Linux with GPU
3. **CPU** -- fallback

Set `EMBED_DEVICE=cpu` to force CPU inference, or `EMBED_DEVICE=cuda` / `EMBED_DEVICE=mps` to pin a specific accelerator.

### Model caching

Sentence-transformers downloads the model on first use. By default models are cached in `~/.cache/huggingface/`. Set `HF_HOME` to change the cache location:

```bash
export HF_HOME=/path/to/models
```

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "intfloat/multilingual-e5-base", "device": "mps"}
```

### Generate embeddings

Single string:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}'
```

Multiple strings:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "Bonjour le monde"]}'
```

Response:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.012, ...]},
    {"object": "embedding", "index": 1, "embedding": [-0.034, ...]}
  ],
  "model": "intfloat/multilingual-e5-base",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

## Testing

```bash
# Run all tests (requires model download on first run)
uv run pytest -v

# Skip slow model-dependent tests
SKIP_MODEL_TESTS=1 uv run pytest -v
```

## Model Selection

The default model is `intfloat/multilingual-e5-base` (768-dimensional embeddings).

**Local development** — set the `EMBED_MODEL_ID` environment variable:

```bash
EMBED_MODEL_ID=intfloat/multilingual-e5-large uv run uvicorn embed_provider.api:app
```

**Docker** — the model is pre-downloaded at build time. To change it, override the build arg *and* the runtime env var so they match:

```bash
# Build with a different model baked in
docker build --build-arg DEFAULT_MODEL_ID=intfloat/multilingual-e5-large -t embedding-service .

# Run with the same model set at runtime
docker run -e EMBED_MODEL_ID=intfloat/multilingual-e5-large -p 8000:8000 embedding-service
```

In `docker-compose.yml` both are configured in one place — see the commented `args` block and the `EMBED_MODEL_ID` env var.

## Docker

```bash
# Build and run with docker-compose
docker compose up --build

# Or build manually
docker build -t embedding-service .
docker run -p 8000:8000 embedding-service
```

The Docker image pre-downloads the model at build time, so the container starts without network access to HuggingFace.
