"""Tests for the predictor (normalization + forward + denormalization)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src._vendor.normalizer import Normalizer
from src.inference.predictor import predict


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


class _ZeroModel(torch.nn.Module):
    """Model that always returns zeros of shape (batch, forecast_steps, 1)."""

    def __init__(self, forecast_steps: int):
        super().__init__()
        self.forecast_steps = forecast_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.zeros(batch, self.forecast_steps, 1)


def _make_config():
    normalization = {
        "default": "zscore",
        "methods": {
            **{v: "zscore" for v in INPUT_VARS if v != "ap30"},
            "ap30": "log1p_zscore",
        },
    }
    return SimpleNamespace(
        data=SimpleNamespace(
            timeseries=SimpleNamespace(
                input_variables=INPUT_VARS,
                target_variables=["ap30"],
                normalization=normalization,
            )
        )
    )


def _make_stats():
    # Use trivial stats: mean=0, std=1 for zscore vars;
    # log1p_mean=0, log1p_std=1 for ap30.
    stats = {v: {"mean": 0.0, "std": 1.0} for v in INPUT_VARS if v != "ap30"}
    stats["ap30"] = {"log1p_mean": 0.0, "log1p_std": 1.0, "mean": 0.0, "std": 1.0}
    return stats


def _write_event_csv(tmp_path: Path, rows: int = 96) -> Path:
    data = {var: np.full(rows, 0.5) for var in INPUT_VARS}
    data["hp30"] = np.full(rows, 0.5)
    data["datetime"] = pd.date_range("2026-04-17", periods=rows, freq="30min")
    out = tmp_path / "20260417000000.csv"
    pd.DataFrame(data).to_csv(out, index=False)
    return out


def test_predict_returns_expected_shape_and_denormalized_values(tmp_path: Path):
    cfg = _make_config()
    stats = _make_stats()
    normalizer = Normalizer(stat_dict=stats, method_config=cfg.data.timeseries.normalization)

    model = _ZeroModel(forecast_steps=12)
    device = torch.device("cpu")
    event_csv = _write_event_csv(tmp_path)

    forecast = predict(cfg, model, normalizer, event_csv, device)

    # Zero normalized → denormalize: log1p_mean=0, log1p_std=1 → expm1(0)=0.
    assert forecast.shape == (12,)
    assert np.allclose(forecast, 0.0)


def test_predict_errors_on_missing_column(tmp_path: Path):
    cfg = _make_config()
    stats = _make_stats()
    normalizer = Normalizer(stat_dict=stats, method_config=cfg.data.timeseries.normalization)
    model = _ZeroModel(forecast_steps=12)

    # Build CSV missing one column.
    data = {var: np.full(96, 0.5) for var in INPUT_VARS if var != "bx_avg"}
    data["hp30"] = np.full(96, 0.5)
    data["datetime"] = pd.date_range("2026-04-17", periods=96, freq="30min")
    out = tmp_path / "20260417000000.csv"
    pd.DataFrame(data).to_csv(out, index=False)

    with pytest.raises(ValueError, match="missing input variables"):
        predict(cfg, model, normalizer, out, torch.device("cpu"))
