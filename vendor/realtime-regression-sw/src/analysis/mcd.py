"""Monte Carlo Dropout inference for uncertainty estimation.

Follows the `regression-sw/analysis/run_mcd.py` approach: keep the model in eval
mode but manually switch every `nn.Dropout` module back to train mode so stochastic
forward passes produce different samples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .._vendor.normalizer import Normalizer

logger = logging.getLogger(__name__)


@dataclass
class MCDResult:
    """Monte Carlo Dropout statistics in the original (denormalized) scale.

    Attributes:
        samples: Array of shape `(num_samples, forecast_steps)` — every sample
            has already been denormalized for ease of downstream plotting.
        mean: `(forecast_steps,)` — element-wise mean of samples.
        std: `(forecast_steps,)` — element-wise standard deviation of samples.
        lower: `(forecast_steps,)` — `mean - n_std * std`, clipped at 0.
        upper: `(forecast_steps,)` — `mean + n_std * std`.
        n_std: Band width in σ units.
    """

    samples: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_std: float


def _enable_dropout(model: nn.Module) -> int:
    """Switch every `nn.Dropout` module in the model to train mode.

    Args:
        model: Model already in eval mode.

    Returns:
        Number of dropout modules switched on (useful for sanity-check logging).
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            count += 1
    return count


def mcd_forecast(
    model: nn.Module,
    input_tensor: torch.Tensor,
    normalizer: Normalizer,
    target_variable: str,
    num_samples: int = 100,
    n_std: float = 2.0,
) -> MCDResult:
    """Run `num_samples` stochastic forward passes and return uncertainty stats.

    Args:
        model: Model in eval mode. This function mutates dropout modules in-place
            (switches them to train). Caller should call `model.eval()` afterwards
            if deterministic inference is subsequently required.
        input_tensor: `(1, seq_len, num_vars)` tensor on the same device as model.
        normalizer: Fitted normalizer used to denormalize each sample.
        target_variable: Name of the target variable (e.g. "ap30").
        num_samples: Number of stochastic forward passes.
        n_std: Band width in units of standard deviations.

    Returns:
        MCDResult with samples already denormalized.
    """
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2 for MCD statistics")

    n_dropout = _enable_dropout(model)
    logger.debug("Enabled %d dropout module(s) for MCD", n_dropout)
    if n_dropout == 0:
        logger.warning(
            "No nn.Dropout modules found; MCD samples will be identical. "
            "Ensure the model was trained with dropout > 0."
        )

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = model(input_tensor)  # (1, forecast_steps, 1)
            norm = out[0, :, 0].detach().cpu().numpy()
            samples.append(normalizer.denormalize_omni(norm, target_variable))

    arr = np.stack(samples, axis=0)  # (num_samples, forecast_steps)
    arr = np.clip(arr, a_min=0.0, a_max=None)  # ap30 is non-negative

    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    lower = np.clip(mean - n_std * std, a_min=0.0, a_max=None)
    upper = mean + n_std * std

    # Restore deterministic eval state for later forward passes.
    model.eval()

    return MCDResult(samples=arr, mean=mean, std=std,
                     lower=lower, upper=upper, n_std=n_std)
