"""FastAPI application exposing OpenAI-compatible embedding endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from embed_provider import config
from embed_provider.model import EmbeddingModel
from embed_provider.schemas import (
    EmbeddingItem,
    EmbeddingsRequest,
    EmbeddingsResponse,
)

_model: EmbeddingModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model once at startup."""
    global _model  # noqa: PLW0603
    _model = EmbeddingModel()
    yield


app = FastAPI(title="Embedding Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    assert _model is not None
    return {
        "status": "ok",
        "model": config.MODEL_ID,
        "device": _model.device,
    }


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    assert _model is not None

    texts: list[str]
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="Input list must not be empty.")

    for i, t in enumerate(texts):
        if t == "":
            raise HTTPException(
                status_code=400, detail=f"Input string at index {i} must not be empty."
            )

    embeddings = _model.embed_many(texts)

    data = [
        EmbeddingItem(index=i, embedding=emb)
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingsResponse(
        data=data,
        model=config.MODEL_ID,
    )
