# Vendored from regression-sw/src/pipeline/normalizer.py @ 2d89767 on 2026-04-19 — DO NOT EDIT.
# Subset retained: Normalizer (OnlineStatistics dropped as unused at inference time).
# Re-sync: see src/_vendor/README.md.
"""Normalization utilities for solar wind variables."""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Normalizer:
    """Multi-method normalizer for OMNI variables.

    Supports different normalization methods per variable:
    - zscore: (x - mean) / std
    - log_zscore: log(x) -> z-score (positive values only)
    - log1p_zscore: log(1 + x) -> z-score (non-negative values)
    - minmax: (x - min) / (max - min)
    """

    VALID_METHODS = {'zscore', 'log_zscore', 'log1p_zscore', 'minmax'}

    def __init__(
        self,
        stat_dict: Optional[Dict[str, Dict[str, float]]] = None,
        method_config: Optional[Dict] = None,
    ):
        """Initialize normalizer.

        Args:
            stat_dict: Dictionary of statistics for each variable.
                Format: {variable: {'mean', 'std', 'log_mean', 'log_std', ...}}
            method_config: Normalization method configuration.
                Format: {'default': str, 'methods': {variable: method}}
        """
        self.stat_dict = stat_dict or {}
        self.method_config = method_config or {}
        self.default_method = self.method_config.get('default', 'zscore')

    def get_method(self, variable: str) -> str:
        """Return the normalization method registered for a variable."""
        methods = self.method_config.get('methods', {})
        return methods.get(variable, self.default_method)

    def normalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Normalize OMNI data using the variable-specific method.

        Args:
            data: Raw OMNI data.
            variable: Variable name for statistics and method lookup.

        Returns:
            Normalized data.

        Raises:
            KeyError: If statistics not found for variable.
            ValueError: If unknown normalization method.
        """
        if variable not in self.stat_dict:
            raise KeyError(f"Statistics not found for variable: {variable}")

        stats = self.stat_dict[variable]
        method = self.get_method(variable)

        if method == 'zscore':
            mean = stats['mean']
            std = stats['std']
            return (data - mean) / (std + 1e-8)

        if method == 'log_zscore':
            data_clipped = np.maximum(data, 1e-6)
            log_data = np.log(data_clipped)
            log_mean = stats.get('log_mean', 0.0)
            log_std = stats.get('log_std', 1.0)
            return (log_data - log_mean) / (log_std + 1e-8)

        if method == 'log1p_zscore':
            log1p_data = np.log1p(np.maximum(data, 0))
            log1p_mean = stats.get('log1p_mean', 0.0)
            log1p_std = stats.get('log1p_std', 1.0)
            return (log1p_data - log1p_mean) / (log1p_std + 1e-8)

        if method == 'minmax':
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0)
            return (data - min_val) / (max_val - min_val + 1e-8)

        raise ValueError(f"Unknown normalization method: {method}")

    def denormalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Denormalize OMNI data back to the original scale.

        Args:
            data: Normalized data.
            variable: Variable name for statistics and method lookup.

        Returns:
            Data in the original scale.

        Raises:
            KeyError: If statistics not found for variable.
            ValueError: If unknown normalization method.
        """
        if variable not in self.stat_dict:
            raise KeyError(f"Statistics not found for variable: {variable}")

        stats = self.stat_dict[variable]
        method = self.get_method(variable)

        if method == 'zscore':
            mean = stats['mean']
            std = stats['std']
            return data * std + mean

        if method == 'log_zscore':
            log_mean = stats.get('log_mean', 0.0)
            log_std = stats.get('log_std', 1.0)
            log_data = data * log_std + log_mean
            return np.exp(log_data)

        if method == 'log1p_zscore':
            log1p_mean = stats.get('log1p_mean', 0.0)
            log1p_std = stats.get('log1p_std', 1.0)
            log1p_data = data * log1p_std + log1p_mean
            return np.expm1(log1p_data)

        if method == 'minmax':
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0)
            return data * (max_val - min_val) + min_val

        raise ValueError(f"Unknown normalization method: {method}")
