ARG DEFAULT_MODEL_ID=intfloat/multilingual-e5-base

FROM python:3.11-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cache-friendly layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/
RUN uv sync --frozen --no-dev

# Pre-download the model so the container works offline
ENV HF_HOME=/models/hf
ENV EMBED_MODEL_ID=${DEFAULT_MODEL_ID}
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${DEFAULT_MODEL_ID}')"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "embed_provider.api:app", "--host", "0.0.0.0", "--port", "8000"]
