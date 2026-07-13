"""End-to-end inference: build tensor, forward, denormalize."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .._vendor.normalizer import Normalizer

logger = logging.getLogger(__name__)


def assemble_input_tensor(
    event_df: pd.DataFrame,
    input_variables: list[str],
    normalizer: Normalizer,
) -> torch.Tensor:
    """Normalize each column and stack to a `(1, seq_len, num_vars)` tensor."""
    missing = [col for col in input_variables if col not in event_df.columns]
    if missing:
        raise ValueError(f"Event CSV is missing input variables: {missing}")

    normalized_columns = []
    for var in input_variables:
        raw = event_df[var].to_numpy(dtype=np.float64)
        if np.isnan(raw).any():
            raise ValueError(f"Column '{var}' contains NaN at inference time")
        normalized_columns.append(normalizer.normalize_omni(raw, var))

    stacked = np.stack(normalized_columns, axis=1)  # (seq_len, num_vars)
    tensor = torch.from_numpy(stacked).float().unsqueeze(0)  # (1, seq_len, num_vars)
    return tensor


def predict(
    config,
    model: torch.nn.Module,
    normalizer: Normalizer,
    event_csv_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Run inference on a single event CSV and return the denormalized forecast.

    Args:
        config: Merged OmegaConf config (needs data.timeseries.input_variables
            and data.timeseries.target_variables).
        model: PyTorch model in eval mode, on `device`.
        normalizer: Initialized normalizer with training stats.
        event_csv_path: Path to the 96-row event CSV produced by event_builder.
        device: Target device for the forward pass.

    Returns:
        Array of shape (forecast_steps,) with ap30 predictions in the original
        scale (not normalized).
    """
    event_df = pd.read_csv(event_csv_path)
    input_variables = list(config.data.timeseries.input_variables)
    target_variables = list(config.data.timeseries.target_variables)

    if len(target_variables) != 1:
        raise NotImplementedError(
            f"Realtime predictor supports single-target only; got {target_variables}"
        )
    target_variable = target_variables[0]

    tensor = assemble_input_tensor(event_df, input_variables, normalizer).to(device)
    logger.debug("Input tensor shape: %s", tuple(tensor.shape))

    with torch.no_grad():
        y_norm = model(tensor)  # (1, forecast_steps, num_targets)

    if y_norm.dim() != 3:
        raise RuntimeError(f"Unexpected model output shape: {tuple(y_norm.shape)}")

    forecast_norm = y_norm[0, :, 0].detach().cpu().numpy()
    forecast = normalizer.denormalize_omni(forecast_norm, target_variable)
    forecast = np.clip(forecast, a_min=0.0, a_max=None)  # ap30 is non-negative
    return forecast
