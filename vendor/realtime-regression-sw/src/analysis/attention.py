"""Attention + GNN adjacency extraction for the `gnn_transformer` model.

Replicates the manual forward pass in
`regression-sw/analysis/attention_analysis.py::_extract_gnn_attention` so that
the Transformer self-attention weights can be captured with
`need_weights=True, average_attn_weights=False`. The GNN adjacency matrix is
read directly from the `.adjacency_matrix` property already exposed by the
vendored model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AttentionResult:
    """Container for attention tensors extracted on a single forward pass.

    Attributes:
        attention_per_layer: List of per-layer arrays with shape
            `(num_heads, seq_len, seq_len)` — already averaged over the batch
            dimension (batch size is always 1 at inference time).
        adjacency: `(num_nodes, num_nodes)` GNN adjacency matrix.
        temporal_importance_per_layer: List of `(seq_len,)` arrays, one per
            layer, giving how much attention each input step *receives*
            (averaged over heads).
        node_labels: Ordered node names from the config (e.g. `["v", "np", ...,
            "ap30"]`).
    """

    attention_per_layer: List[np.ndarray]
    adjacency: np.ndarray
    temporal_importance_per_layer: List[np.ndarray]
    node_labels: List[str]


def extract_gnn_attention(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    node_labels: List[str],
) -> AttentionResult:
    """Run a manual forward pass that exposes attention weights.

    Works for the GNNOnlyModel with `gnn_temporal_type == "transformer"`. The
    function does **not** mutate the model.

    Args:
        model: GNNOnlyModel instance (already on the correct device, eval mode).
        input_tensor: `(1, seq_len, num_vars)` tensor.
        node_labels: Ordered node names matching `gnn_variable_groups` keys.

    Returns:
        AttentionResult with all tensors moved to CPU and converted to numpy.

    Raises:
        RuntimeError: If the model does not expose a transformer-based GNN
            temporal encoder (e.g. TCN/BiLSTM variants).
    """
    gnn = getattr(model, "gnn_encoder", None)
    if gnn is None:
        raise RuntimeError("Model has no `gnn_encoder` — attention path only supports GNN models.")
    if gnn.temporal_type != "transformer":
        raise RuntimeError(
            f"Attention extraction only implemented for transformer temporal encoder, "
            f"got {gnn.temporal_type!r}."
        )

    batch_size, seq_len, _ = input_tensor.size()

    with torch.no_grad():
        # 1. Split to variable nodes and project.
        node_groups = gnn._split_to_nodes(input_tensor)
        node_features = []
        for i, group in enumerate(node_groups):
            node_features.append(gnn.node_projections[i](group))
        node_features = torch.stack(node_features, dim=2)

        # 2. Adaptive adjacency + GCN stack (per-timestep message passing).
        adj = gnn._compute_adaptive_adj()
        h = node_features.reshape(batch_size * seq_len, gnn.num_nodes, -1)
        for gcn_layer in gnn.gcn_layers:
            h = gcn_layer(h, adj)
            h = gnn.gcn_activation(h)
            h = gnn.gcn_dropout(h)
        h = h.reshape(batch_size, seq_len, -1)

        # 3. Temporal projection + positional encoding.
        h = gnn.temporal_proj(h)
        h = gnn.pos_encoder(h)

        # 4. Walk every TransformerEncoderLayer, capturing self-attention weights.
        attention_per_layer: List[np.ndarray] = []
        for layer in gnn.temporal_encoder.layers:
            attn_output, attn_weights = layer.self_attn(
                h, h, h,
                need_weights=True,
                average_attn_weights=False,
            )
            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            attention_per_layer.append(
                attn_weights[0].detach().cpu().numpy()
            )

            # Replicate the rest of the standard TransformerEncoderLayer forward
            # so the residual stream stays consistent (not strictly necessary at
            # inference since we only consume attention, but kept for parity).
            h = layer.norm1(h + layer.dropout1(attn_output))
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(h + layer.dropout2(x2))

    # Temporal importance: how much attention each input step receives.
    temporal_importance_per_layer = []
    for attn in attention_per_layer:
        # attn shape: (num_heads, seq_len, seq_len); sum over query dim, average over heads.
        imp = attn.sum(axis=-2).mean(axis=0)  # (seq_len,)
        temporal_importance_per_layer.append(imp)

    adjacency = adj.detach().cpu().numpy()

    return AttentionResult(
        attention_per_layer=attention_per_layer,
        adjacency=adjacency,
        temporal_importance_per_layer=temporal_importance_per_layer,
        node_labels=list(node_labels),
    )
