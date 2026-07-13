"""Tests for event CSV builder."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.event_builder import build_event_csv


INPUT_VARS = [
    "v_avg", "v_min", "v_max",
    "np_avg", "np_min", "np_max",
    "t_avg", "t_min", "t_max",
    "bx_avg", "bx_min", "bx_max",
    "by_avg", "by_min", "by_max",
    "bz_avg", "bz_min", "bz_max",
    "bt_avg", "bt_min", "bt_max",
    "ap30",
]


def _sample_aligned(rows: int) -> pd.DataFrame:
    """Build a synthetic aligned DataFrame matching the training schema."""
    start = pd.Timestamp("2026-04-17 00:00:00")
    idx = pd.date_range(start, periods=rows, freq="30min")
    data = {col: np.linspace(1.0, 2.0, rows) for col in INPUT_VARS}
    data["hp30"] = np.linspace(0.0, 3.0, rows)
    data["datetime"] = idx
    return pd.DataFrame(data)


def test_csv_schema_and_row_count(tmp_path: Path):
    aligned = _sample_aligned(rows=96)
    t_end = aligned["datetime"].iloc[-1]
    out_path = build_event_csv(aligned, t_end, tmp_path, INPUT_VARS)

    assert out_path.name == t_end.strftime("%Y%m%d%H%M%S") + ".csv"
    written = pd.read_csv(out_path)
    assert list(written.columns) == ["datetime", *INPUT_VARS, "hp30"]
    assert len(written) == 96


def test_datetime_is_monotonic_and_30min_cadence(tmp_path: Path):
    aligned = _sample_aligned(rows=96)
    t_end = aligned["datetime"].iloc[-1]
    out_path = build_event_csv(aligned, t_end, tmp_path, INPUT_VARS)
    written = pd.read_csv(out_path, parse_dates=["datetime"])

    assert written["datetime"].is_monotonic_increasing
    deltas = written["datetime"].diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(minutes=30)


def test_missing_column_raises(tmp_path: Path):
    aligned = _sample_aligned(rows=96).drop(columns=["hp30"])
    t_end = aligned["datetime"].iloc[-1]
    with pytest.raises(ValueError, match="missing columns"):
        build_event_csv(aligned, t_end, tmp_path, INPUT_VARS)
