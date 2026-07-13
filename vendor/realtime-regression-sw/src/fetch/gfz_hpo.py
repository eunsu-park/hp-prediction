"""Fetch and parse the GFZ Potsdam Hp30/ap30 nowcast file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .._vendor.download import download
from .._vendor.parse_hpo import HP30, parse_hpo

logger = logging.getLogger(__name__)


def _cache_raw_text(text: str, cache_dir: Optional[Path], filename: str) -> None:
    """Persist the raw text response for reproducibility if a cache dir is set."""
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / filename
    out_path.write_text(text, encoding="utf-8")
    logger.debug("Cached raw GFZ response → %s", out_path)


def fetch_hpo(url: str, timeout: int = 30, max_retries: int = 3,
              cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download the GFZ Hp30/ap30 nowcast text and return a typed DataFrame.

    Args:
        url: GFZ Hp30_ap30_nowcast.txt URL.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts.
        cache_dir: Optional directory to cache the raw text.

    Returns:
        DataFrame with columns [datetime, Hp30, ap30, ...] at 30-min cadence.

    Raises:
        RuntimeError: If download fails.
    """
    text = download(url, timeout=timeout, max_retries=max_retries)
    if text is None:
        raise RuntimeError(f"GFZ HPo download failed: {url}")

    _cache_raw_text(text, cache_dir, "hpo.txt")

    df = parse_hpo(text, HP30)
    # Lower-case geomagnetic columns to match sw_30min schema.
    df = df.rename(columns={"Hp30": "hp30", "ap30": "ap30"})

    # Drop rows with unparsable datetime (pd.NaT) and sort.
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info("GFZ HPo fetched: %d rows, %s → %s",
                len(df), df["datetime"].min(), df["datetime"].max())
    return df[["datetime", "hp30", "ap30"]]
