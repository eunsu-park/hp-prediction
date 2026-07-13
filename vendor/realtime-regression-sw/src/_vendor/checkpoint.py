# Vendored from regression-sw/src/utils.py @ 2d89767 on 2026-04-19 — DO NOT EDIT.
# Subset retained: setup_device, load_model (training/analysis helpers dropped).
# Re-sync: see src/_vendor/README.md.
"""Device setup and checkpoint loading helpers."""

import os

import torch


def setup_device(requested_device: str) -> torch.device:
    """Initialize a compute device.

    Args:
        requested_device: Requested device ('cuda', 'mps', 'cpu').

    Returns:
        torch.device for the selected backend (falls back to CPU).
    """
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    elif requested_device == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("MPS not available, using CPU")
    elif requested_device == 'cpu':
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device('cpu')
        print(f"Unknown device '{requested_device}', using CPU")
    return device


def load_model(model: torch.nn.Module, checkpoint_path: str,
               device: torch.device) -> torch.nn.Module:
    """Load model weights from a checkpoint file.

    Args:
        model: PyTorch model instance.
        checkpoint_path: Path to checkpoint file.
        device: Target device.

    Returns:
        Model with weights loaded, moved to device, set to eval mode.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded: {checkpoint_path}")
    return model
