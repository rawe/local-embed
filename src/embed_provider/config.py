"""Configuration loaded from environment variables."""

from __future__ import annotations

import os


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


MODEL_ID: str = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-base")
DEVICE: str = os.getenv("EMBED_DEVICE", "auto")
NORMALIZE: bool = _parse_bool(os.getenv("EMBED_NORMALIZE", "true"))
BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))
HF_HOME: str | None = os.getenv("HF_HOME")
E5_MODE: str = os.getenv("EMBED_E5_MODE", "passage")
