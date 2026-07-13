# Vendored from regression-sw/src/networks/patchtst.py @ 2d89767 on 2026-04-19 — DO NOT EDIT.
# Retained for its `PatchEmbedding` symbol, imported by gnn.py.
# Re-sync: see src/_vendor/README.md.
"""PatchTST models for patch-based time series processing."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model


class PatchEmbedding(nn.Module):
    """Convert time series into patch tokens via a sliding window."""

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_input: int = 1,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len * d_input, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_input = x.size()

        pad_len = (self.stride - (seq_len - self.patch_len) % self.stride) % self.stride
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        padded_len = x.size(1)
        num_patches = (padded_len - self.patch_len) // self.stride + 1
        patches = x.unfold(1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2)

        patches = patches.reshape(batch_size, num_patches, -1)
        tokens = self.projection(patches)
        tokens = self.dropout(tokens)
        return tokens


class PatchTransformerEncoder(nn.Module):
    """PatchTST-style encoder for time series."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        d_model: int = 128,
        patch_len: int = 16,
        patch_stride: int = 8,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_input_variables = num_input_variables
        self.input_sequence_length = input_sequence_length
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            stride=patch_stride,
            d_input=num_input_variables,
            d_model=d_model,
            dropout=dropout,
        )

        pad_len = (patch_stride - (input_sequence_length - patch_len) % patch_stride) % patch_stride
        self.num_patches = (input_sequence_length + pad_len - patch_len) // patch_stride + 1

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        h = self.transformer_encoder(tokens)
        h = h.transpose(1, 2)
        h = self.global_pool(h).squeeze(-1)
        h = self.output_projection(h)
        return h


class PatchTSTOnlyModel(nn.Module):
    """Time series model using PatchTST encoder."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        patch_len: int = 16,
        patch_stride: int = 8,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.encoder = PatchTransformerEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=d_model,
            patch_len=patch_len,
            patch_stride=patch_stride,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables),
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        features = self.encoder(solar_wind_input)
        predictions = self.regression_head(features)
        output = predictions.reshape(
            predictions.size(0),
            self.target_sequence_length,
            self.num_target_variables,
        )
        if return_features:
            return output, features, None
        return output


@register_model("patchtst")
def _create_patchtst(config):
    """Factory function for PatchTST model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    return PatchTSTOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        patch_len=getattr(config.model, 'patch_len', 16),
        patch_stride=getattr(config.model, 'patch_stride', 8),
        nhead=config.model.transformer_nhead,
        num_layers=config.model.transformer_num_layers,
        dim_feedforward=config.model.transformer_dim_feedforward,
        dropout=getattr(config.model, 'patchtst_dropout', 0.1),
    )
