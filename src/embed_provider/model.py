"""EmbeddingModel — thin wrapper around SentenceTransformer."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from embed_provider import config
from embed_provider.device import resolve_device

VALID_E5_MODES = ("passage", "query", "none")


def _prefix_texts(texts: list[str], e5_mode: str) -> list[str]:
    """Apply E5 prefix based on mode."""
    if e5_mode == "passage":
        return [f"passage: {t}" for t in texts]
    if e5_mode == "query":
        return [f"query: {t}" for t in texts]
    return texts


class EmbeddingModel:
    """Load a SentenceTransformer model once and expose encode helpers."""

    def __init__(
        self,
        model_id: str = config.MODEL_ID,
        device: str = config.DEVICE,
        normalize: bool = config.NORMALIZE,
        e5_mode: str = config.E5_MODE,
        batch_size: int = config.BATCH_SIZE,
    ) -> None:
        if e5_mode not in VALID_E5_MODES:
            raise ValueError(
                f"Invalid e5_mode '{e5_mode}'. Choose from: {', '.join(VALID_E5_MODES)}."
            )
        self.device = resolve_device(device)
        self.normalize = normalize
        self.e5_mode = e5_mode
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, device=self.device)

    def embed_one(self, text: str) -> list[float]:
        """Encode a single text and return a list of floats."""
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts and return a list of float lists."""
        prefixed = _prefix_texts(texts, self.e5_mode)
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
        )
        return embeddings.tolist()
