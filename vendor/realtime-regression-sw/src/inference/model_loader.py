"""Build and load the model described by the merged config."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

# Importing the vendored networks package registers the model factories.
from .._vendor import networks as _networks_pkg  # noqa: F401
from .._vendor.checkpoint import load_model, setup_device
from .._vendor.networks import create_model

logger = logging.getLogger(__name__)


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a short sha256 hex digest for a file.

    Args:
        path: File to hash.
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex digest string.
    """
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_and_load_model(config, checkpoint_path: Path, device_name: str):
    """Create the model from config and load checkpoint weights.

    Args:
        config: Merged OmegaConf config.
        checkpoint_path: Path to `model_best.pth` (or equivalent).
        device_name: "cpu", "cuda", or "mps".

    Returns:
        Tuple of (model, torch.device).
    """
    device = setup_device(device_name)
    model = create_model(config)
    model = load_model(model, str(checkpoint_path), device)
    return model, device
