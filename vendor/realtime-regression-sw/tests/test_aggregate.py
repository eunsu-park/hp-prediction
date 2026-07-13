"""Tests for the DB-free 30-min aggregation."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.pipeline.aggregate import aggregate_30min


def _synthetic_1min(start: datetime, minutes: int) -> pd.DataFrame:
    """Create a synthetic 1-min DataFrame with deterministic values.

    v = 100 + minute_index, np = minute_index, t = 1e4 + minute_index,
    bx/by/bz/bt = minute_index.
    """
    idx = pd.date_range(start, periods=minutes, freq="1min")
    series = np.arange(minutes, dtype=float)
    return pd.DataFrame({
        "datetime": idx,
        "v": 100.0 + series,
        "np": series,
        "t": 10000.0 + series,
        "bx": series,
        "by": series,
        "bz": series,
        "bt": series,
    })


def test_flat_schema_contains_21_columns():
    start = datetime(2026, 1, 1, 0, 0)
    df = _synthetic_1min(start, minutes=60)
    out = aggregate_30min(df, start=start, end=start.replace(minute=30))
    assert list(out.columns) == [
        "v_avg", "v_min", "v_max",
        "np_avg", "np_min", "np_max",
        "t_avg", "t_min", "t_max",
        "bx_avg", "bx_min", "bx_max",
        "by_avg", "by_min", "by_max",
        "bz_avg", "bz_min", "bz_max",
        "bt_avg", "bt_min", "bt_max",
    ]
    assert len(out.columns) == 21


def test_aggregation_values_match_hand_calculation():
    start = datetime(2026, 1, 1, 0, 0)
    df = _synthetic_1min(start, minutes=60)
    out = aggregate_30min(df, start=start, end=start.replace(minute=30))

    # First 30-min window contains minute indices 0..29.
    assert out.loc[pd.Timestamp(start), "v_avg"] == 100.0 + (0 + 29) / 2
    assert out.loc[pd.Timestamp(start), "v_min"] == 100.0
    assert out.loc[pd.Timestamp(start), "v_max"] == 100.0 + 29

    second_bin = pd.Timestamp(start) + pd.Timedelta(minutes=30)
    # Second bin: minute indices 30..59.
    assert out.loc[second_bin, "np_avg"] == (30 + 59) / 2
    assert out.loc[second_bin, "bt_max"] == 59


def test_missing_rows_become_nan():
    start = datetime(2026, 1, 1, 0, 0)
    df = _synthetic_1min(start, minutes=15)  # only first 15 minutes provided
    out = aggregate_30min(df, start=start, end=start.replace(minute=30))

    # Second 30-min bin has no source rows → all NaN.
    second_bin = pd.Timestamp(start) + pd.Timedelta(minutes=30)
    assert out.loc[second_bin].isna().all()
    # First bin still has values.
    assert not np.isnan(out.loc[pd.Timestamp(start), "v_avg"])


def test_handles_empty_grid():
    start = datetime(2026, 1, 1, 0, 0)
    out = aggregate_30min(_synthetic_1min(start, 10),
                          start=start, end=start - pd.Timedelta(minutes=30))
    assert len(out) == 0
