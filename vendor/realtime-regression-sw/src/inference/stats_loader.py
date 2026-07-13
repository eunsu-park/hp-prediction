"""Load the per-variable normalization statistics produced during training."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Sequence

logger = logging.getLogger(__name__)


def load_stats(stats_path: Path, required_variables: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Unpickle `table_stats.pkl` and verify it covers the required variables.

    Args:
        stats_path: Absolute path to the pickle file produced by training.
        required_variables: Variables that must be present in the stats dict.

    Returns:
        The stats dict (shape: {variable: {mean, std, log_mean, ...}}).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If any required variable is missing in the stats dict.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    with path.open("rb") as fp:
        stats = pickle.load(fp)

    if not isinstance(stats, dict):
        raise TypeError(f"Unexpected stats payload type: {type(stats)!r}")

    missing = [var for var in required_variables if var not in stats]
    if missing:
        raise KeyError(
            f"Stats file {path} is missing variables required by the model: {missing}"
        )

    logger.info("Loaded stats for %d variables from %s", len(stats), path)
    return stats
