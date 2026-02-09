"""Device selection logic for torch inference."""

from __future__ import annotations

import torch

VALID_DEVICES = ("cpu", "cuda", "mps")


def resolve_device(device: str = "auto") -> str:
    """Resolve the compute device to use for inference.

    If *device* is ``"auto"``, prefer MPS > CUDA > CPU.
    If explicitly set, validate that the device is actually available.
    """
    device = device.strip().lower()

    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if device not in VALID_DEVICES:
        raise ValueError(
            f"Unsupported device '{device}'. Choose from: {', '.join(VALID_DEVICES)} or 'auto'."
        )

    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Device 'mps' requested but MPS is not available on this system.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Device 'cuda' requested but CUDA is not available on this system.")

    return device
