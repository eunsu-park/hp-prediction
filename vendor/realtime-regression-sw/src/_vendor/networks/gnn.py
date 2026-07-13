# Vendored from regression-sw/src/networks/gnn.py @ 2d89767 on 2026-04-19 — DO NOT EDIT.
# Re-sync: see src/_vendor/README.md.
"""Graph Neural Network (GNN) models for multivariate time series."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import (
    DEFAULT_VARIABLE_NODE_GROUPS,
    _get_model_dimensions,
    build_gnn_node_groups,
)
from ._registry import register_model
from .transformer import PositionalEncoding
from .tcn import TemporalBlock
from .patchtst import PatchEmbedding


class GraphConvLayer(nn.Module):
    """Single graph convolution layer. X' = sigma(A @ X @ W + b)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(adj, x)
        return self.weight(support)


class GNNEncoder(nn.Module):
    """GNN encoder for multivariate time series with pluggable temporal backbone."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        group_sizes: list = None,
        num_nodes: int = None,
        node_feature_dim: int = 32,
        gcn_hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        temporal_type: str = "transformer",
        d_model: int = 128,
        dropout: float = 0.1,
        node_embed_dim: int = 16,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
        patch_len: int = 16,
        patch_stride: int = 8,
    ):
        super().__init__()

        if group_sizes is None:
            group_sizes = [len(v) for v in DEFAULT_VARIABLE_NODE_GROUPS.values()]
        if num_nodes is None:
            num_nodes = len(group_sizes)

        if num_input_variables != sum(group_sizes):
            raise ValueError(
                f"num_input_variables ({num_input_variables}) != "
                f"sum(group_sizes) ({sum(group_sizes)})"
            )

        self._GROUP_SIZES = group_sizes
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.temporal_type = temporal_type
        self.d_model = d_model
        self.input_sequence_length = input_sequence_length

        self.node_projections = nn.ModuleList()
        for size in self._GROUP_SIZES:
            self.node_projections.append(nn.Linear(size, node_feature_dim))

        self.node_embed1 = nn.Parameter(torch.randn(num_nodes, node_embed_dim))
        self.node_embed2 = nn.Parameter(torch.randn(num_nodes, node_embed_dim))

        gcn_layers = []
        in_dim = node_feature_dim
        for i in range(num_gcn_layers):
            out_dim = gcn_hidden_dim if i < num_gcn_layers - 1 else gcn_hidden_dim
            gcn_layers.append(GraphConvLayer(in_dim, out_dim))
            in_dim = out_dim
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.gcn_activation = nn.ReLU()
        self.gcn_dropout = nn.Dropout(dropout)

        temporal_input_dim = num_nodes * gcn_hidden_dim

        if temporal_type == "transformer":
            self.temporal_proj = nn.Linear(temporal_input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, input_sequence_length, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        elif temporal_type == "tcn":
            if tcn_channels is None:
                tcn_channels = [64, 128, 256]
            self.temporal_proj = nn.Linear(temporal_input_dim, tcn_channels[0])
            layers = []
            num_ch = [tcn_channels[0]] + list(tcn_channels)
            for i in range(len(tcn_channels)):
                layers.append(TemporalBlock(
                    num_ch[i], num_ch[i + 1], tcn_kernel_size,
                    dilation=2 ** i, dropout=dropout,
                ))
            self.temporal_encoder = nn.Sequential(*layers)
            self._tcn_out_dim = tcn_channels[-1]
        elif temporal_type == "bilstm":
            self.temporal_proj = nn.Linear(temporal_input_dim, bilstm_hidden_size)
            self.temporal_encoder = nn.LSTM(
                input_size=bilstm_hidden_size,
                hidden_size=bilstm_hidden_size,
                num_layers=bilstm_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if bilstm_num_layers > 1 else 0.0,
            )
            self._bilstm_out_dim = bilstm_hidden_size * 2
        elif temporal_type == "patch_transformer":
            self.temporal_proj = nn.Linear(temporal_input_dim, d_model)
            self._patch_embed = PatchEmbedding(
                patch_len=patch_len,
                stride=patch_stride,
                d_input=d_model,
                d_model=d_model,
                dropout=dropout,
            )
            pad_len = (patch_stride - (input_sequence_length - patch_len) % patch_stride) % patch_stride
            n_patches = (input_sequence_length + pad_len - patch_len) // patch_stride + 1
            self._patch_pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if temporal_type in ("transformer", "patch_transformer"):
            self.output_projection = nn.Linear(d_model, d_model)
        elif temporal_type == "tcn":
            self.output_projection = nn.Linear(self._tcn_out_dim, d_model)
        elif temporal_type == "bilstm":
            self.output_projection = nn.Linear(self._bilstm_out_dim, d_model)

    def _compute_adaptive_adj(self) -> torch.Tensor:
        adj = F.relu(torch.matmul(self.node_embed1, self.node_embed2.T))
        return F.softmax(adj, dim=1)

    def _split_to_nodes(self, x: torch.Tensor) -> list:
        nodes = []
        idx = 0
        for size in self._GROUP_SIZES:
            nodes.append(x[:, :, idx:idx + size])
            idx += size
        return nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        node_groups = self._split_to_nodes(x)
        node_features = []
        for i, group in enumerate(node_groups):
            node_features.append(self.node_projections[i](group))
        node_features = torch.stack(node_features, dim=2)

        adj = self._compute_adaptive_adj()

        h = node_features.reshape(batch_size * seq_len, self.num_nodes, -1)
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, adj)
            h = self.gcn_activation(h)
            h = self.gcn_dropout(h)

        h = h.reshape(batch_size, seq_len, -1)

        if self.temporal_type == "transformer":
            h = self.temporal_proj(h)
            h = self.pos_encoder(h)
            h = self.temporal_encoder(h)
            h = h.transpose(1, 2)
        elif self.temporal_type == "tcn":
            h = self.temporal_proj(h)
            h = h.transpose(1, 2)
            h = self.temporal_encoder(h)
        elif self.temporal_type == "bilstm":
            h = self.temporal_proj(h)
            h, _ = self.temporal_encoder(h)
            h = h.transpose(1, 2)
        elif self.temporal_type == "patch_transformer":
            h = self.temporal_proj(h)
            tokens = self._patch_embed(h)
            tokens = tokens + self._patch_pos_embed[:, :tokens.size(1), :]
            h = self.temporal_encoder(tokens)
            h = h.transpose(1, 2)

        h = self.global_pool(h).squeeze(-1)
        h = self.output_projection(h)
        return h

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            return self._compute_adaptive_adj()


class GNNOnlyModel(nn.Module):
    """Time series model using GNN encoder with pluggable temporal backend."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        gnn_group_sizes: list = None,
        gnn_num_nodes: int = None,
        gnn_node_feature_dim: int = 32,
        gnn_gcn_hidden_dim: int = 64,
        gnn_num_gcn_layers: int = 2,
        gnn_temporal_type: str = "transformer",
        gnn_dropout: float = 0.1,
        gnn_node_embed_dim: int = 16,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
        patch_len: int = 16,
        patch_stride: int = 8,
    ):
        super().__init__()

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.gnn_encoder = GNNEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            group_sizes=gnn_group_sizes,
            num_nodes=gnn_num_nodes,
            node_feature_dim=gnn_node_feature_dim,
            gcn_hidden_dim=gnn_gcn_hidden_dim,
            num_gcn_layers=gnn_num_gcn_layers,
            temporal_type=gnn_temporal_type,
            d_model=d_model,
            dropout=gnn_dropout,
            node_embed_dim=gnn_node_embed_dim,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            bilstm_hidden_size=bilstm_hidden_size,
            bilstm_num_layers=bilstm_num_layers,
            patch_len=patch_len,
            patch_stride=patch_stride,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(gnn_dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables),
        )

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        return self.gnn_encoder.adjacency_matrix

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        gnn_features = self.gnn_encoder(solar_wind_input)
        predictions = self.regression_head(gnn_features)
        output = predictions.reshape(
            predictions.size(0),
            self.target_sequence_length,
            self.num_target_variables,
        )
        if return_features:
            return output, gnn_features, None
        return output


@register_model("gnn")
def _create_gnn(config):
    """Factory function for GNN model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating gnn model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    gnn_group_sizes, gnn_num_nodes = build_gnn_node_groups(config)

    gnn_temporal_type = getattr(config.model, 'gnn_temporal_type', 'transformer')
    gnn_node_feature_dim = getattr(config.model, 'gnn_node_feature_dim', 32)
    gnn_gcn_hidden_dim = getattr(config.model, 'gnn_gcn_hidden_dim', 64)
    gnn_num_gcn_layers = getattr(config.model, 'gnn_num_gcn_layers', 2)
    gnn_dropout = getattr(config.model, 'gnn_dropout', 0.1)
    gnn_node_embed_dim = getattr(config.model, 'gnn_node_embed_dim', 16)

    tcn_channels = getattr(config.model, 'tcn_channels', [64, 128, 256])
    if hasattr(tcn_channels, '__iter__') and not isinstance(tcn_channels, list):
        tcn_channels = list(tcn_channels)
    tcn_kernel_size = getattr(config.model, 'tcn_kernel_size', 3)
    bilstm_hidden_size = getattr(config.model, 'bilstm_hidden_size', 128)
    bilstm_num_layers = getattr(config.model, 'bilstm_num_layers', 2)
    patch_len = getattr(config.model, 'patch_len', 16)
    patch_stride = getattr(config.model, 'patch_stride', 8)

    model = GNNOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        gnn_group_sizes=gnn_group_sizes,
        gnn_num_nodes=gnn_num_nodes,
        gnn_node_feature_dim=gnn_node_feature_dim,
        gnn_gcn_hidden_dim=gnn_gcn_hidden_dim,
        gnn_num_gcn_layers=gnn_num_gcn_layers,
        gnn_temporal_type=gnn_temporal_type,
        gnn_dropout=gnn_dropout,
        gnn_node_embed_dim=gnn_node_embed_dim,
        transformer_nhead=config.model.transformer_nhead,
        transformer_num_layers=config.model.transformer_num_layers,
        transformer_dim_feedforward=config.model.transformer_dim_feedforward,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        bilstm_hidden_size=bilstm_hidden_size,
        bilstm_num_layers=bilstm_num_layers,
        patch_len=patch_len,
        patch_stride=patch_stride,
    )
    print(f"  GNN temporal encoder: {gnn_temporal_type}")
    print(f"  GNN: {gnn_num_gcn_layers} GCN layers, {gnn_num_nodes} nodes, "
          f"groups={gnn_group_sizes}")
    return model
