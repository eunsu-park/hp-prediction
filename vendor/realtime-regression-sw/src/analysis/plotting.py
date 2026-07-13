"""Matplotlib plotting utilities for realtime forecast + analysis outputs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend — plots are always saved to disk.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .attention import AttentionResult
from .mcd import MCDResult

logger = logging.getLogger(__name__)


def plot_forecast(
    event_df: pd.DataFrame,
    t_end: pd.Timestamp,
    forecast: np.ndarray,
    mcd: Optional[MCDResult],
    save_path: Path,
    target_variable: str = "ap30",
    history_steps: int = 96,
    interval_minutes: int = 30,
    dpi: int = 120,
) -> Path:
    """Plot the input history and the forecast on a single time axis.

    Args:
        event_df: Event CSV loaded as DataFrame; must have `datetime` + `ap30`.
        t_end: Anchor timestamp (end of input window).
        forecast: `(forecast_steps,)` deterministic predictions.
        mcd: Optional MCD result to render as a shaded uncertainty band.
        save_path: Target PNG path.
        target_variable: Column used as the y-axis history trace.
        history_steps: How many recent input steps to draw (≤ len(event_df)).
        interval_minutes: Step size in minutes.
        dpi: Figure DPI.

    Returns:
        The `save_path` on success.
    """
    event_df = event_df.copy()
    event_df["datetime"] = pd.to_datetime(event_df["datetime"])
    event_df = event_df.sort_values("datetime").tail(history_steps)

    forecast_times = pd.date_range(
        t_end + pd.Timedelta(minutes=interval_minutes),
        periods=len(forecast),
        freq=f"{interval_minutes}min",
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(event_df["datetime"], event_df[target_variable],
            color="#1f77b4", linewidth=1.5, label=f"history {target_variable}")
    ax.axvline(t_end, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.plot(forecast_times, forecast, color="#d62728", marker="o",
            markersize=4, linewidth=1.8, label="forecast")

    if mcd is not None:
        ax.fill_between(
            forecast_times, mcd.lower, mcd.upper,
            color="#d62728", alpha=0.18,
            label=f"MCD ±{mcd.n_std:g}σ (n={len(mcd.samples)})",
        )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(target_variable)
    ax.set_title(f"{target_variable} forecast — anchor {t_end:%Y-%m-%d %H:%M} UTC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved forecast plot: %s", save_path)
    return save_path


def plot_attention(
    attention: AttentionResult,
    t_end: pd.Timestamp,
    save_path: Path,
    interval_minutes: int = 30,
    dpi: int = 120,
) -> Path:
    """Render per-layer attention heatmaps plus a temporal-importance strip.

    Layout: one row per Transformer layer. Left column shows the full
    `(seq_len, seq_len)` attention map (averaged over heads); right column shows
    the temporal-importance line (how much attention each input step receives).

    Args:
        attention: AttentionResult from `extract_gnn_attention`.
        t_end: Anchor timestamp — used for x-axis labels.
        save_path: Target PNG path.
        interval_minutes: Step size in minutes, used to infer history tick labels.
        dpi: Figure DPI.

    Returns:
        The `save_path` on success.
    """
    num_layers = len(attention.attention_per_layer)
    if num_layers == 0:
        raise ValueError("No attention layers to plot")

    fig, axes = plt.subplots(
        num_layers, 2,
        figsize=(12, 3.2 * num_layers),
        gridspec_kw={"width_ratios": [1.2, 2.0]},
        squeeze=False,
    )

    seq_len = attention.attention_per_layer[0].shape[-1]
    history_times = pd.date_range(
        end=t_end, periods=seq_len, freq=f"{interval_minutes}min",
    )
    hours_before = np.arange(-(seq_len - 1), 1) * interval_minutes / 60.0

    for layer_idx, (attn, imp) in enumerate(
        zip(attention.attention_per_layer, attention.temporal_importance_per_layer)
    ):
        heatmap_ax = axes[layer_idx, 0]
        line_ax = axes[layer_idx, 1]

        avg = attn.mean(axis=0)  # (seq_len, seq_len), averaged over heads
        im = heatmap_ax.imshow(avg, aspect="auto", origin="lower", cmap="viridis")
        heatmap_ax.set_title(f"Layer {layer_idx + 1} — mean self-attention")
        heatmap_ax.set_xlabel("Key step")
        heatmap_ax.set_ylabel("Query step")
        fig.colorbar(im, ax=heatmap_ax, fraction=0.046, pad=0.04)

        line_ax.plot(hours_before, imp, color="#ff7f0e", linewidth=1.6)
        line_ax.fill_between(hours_before, imp, alpha=0.2, color="#ff7f0e")
        line_ax.set_title(f"Layer {layer_idx + 1} — temporal importance (incoming)")
        line_ax.set_xlabel("Hours before anchor")
        line_ax.set_ylabel("Attention received")
        line_ax.grid(True, alpha=0.3)
        line_ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    fig.suptitle(f"Transformer self-attention — anchor {t_end:%Y-%m-%d %H:%M} UTC",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved attention plot: %s", save_path)
    return save_path


def plot_adjacency(
    attention: AttentionResult,
    t_end: pd.Timestamp,
    save_path: Path,
    dpi: int = 120,
) -> Path:
    """Render the GNN adaptive adjacency as an annotated heatmap.

    Args:
        attention: AttentionResult carrying `.adjacency` and `.node_labels`.
        t_end: Anchor timestamp for the title.
        save_path: Target PNG path.
        dpi: Figure DPI.

    Returns:
        The `save_path` on success.
    """
    adj = attention.adjacency
    labels = attention.node_labels
    n = adj.shape[0]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(adj, cmap="YlOrRd", vmin=0.0, vmax=adj.max())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target node (j)")
    ax.set_ylabel("Source node (i)")
    ax.set_title(f"GNN adjacency A[i,j] — anchor {t_end:%Y-%m-%d %H:%M} UTC")

    threshold = 0.6 * adj.max() if adj.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            val = adj[i, j]
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved adjacency plot: %s", save_path)
    return save_path
