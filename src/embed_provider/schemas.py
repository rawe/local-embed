"""Pydantic request/response models (OpenAI-compatible)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: UsageInfo = UsageInfo()
