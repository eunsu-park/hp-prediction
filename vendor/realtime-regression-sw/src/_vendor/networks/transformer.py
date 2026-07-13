# Vendored from regression-sw/src/networks/transformer.py @ 2d89767 on 2026-04-19 — DO NOT EDIT.
# Re-sync: see src/_vendor/README.md.
"""Transformer models for time series processing."""

from typing import Optional, Tuple, Union
import math

import torch
import torch.nn as nn

from ._base import _get_model_dimensions
from ._registry import register_model


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    """Transformer encoder for time series processing."""

    def __init__(self, num_input_variables: int, input_sequence_length: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()

        if num_input_variables <= 0:
            raise ValueError(f"Number of input variables must be positive, got {num_input_variables}")
        if input_sequence_length <= 0:
            raise ValueError(f"Input sequence length must be positive, got {input_sequence_length}")
        if d_model <= 0:
            raise ValueError(f"Model dimension must be positive, got {d_model}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")
        if nhead <= 0 or num_layers <= 0:
            raise ValueError("Number of heads and layers must be positive")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("Dropout must be between 0 and 1")

        self.d_model = d_model
        self.input_sequence_length = input_sequence_length
        self.num_input_variables = num_input_variables

        self.input_projection = nn.Linear(num_input_variables, d_model)
        self.pos_encoder = PositionalEncoding(d_model, input_sequence_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, vars), got {x.dim()}D")

        batch_size, seq_len, num_vars = x.size()

        if seq_len != self.input_sequence_length:
            raise ValueError(f"Expected seq_len {self.input_sequence_length}, got {seq_len}")
        if num_vars != self.num_input_variables:
            raise ValueError(f"Expected {self.num_input_variables} vars, got {num_vars}")

        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.output_projection(x)
        return x


class TransformerOnlyModel(nn.Module):
    """OMNI time series-only model using Transformer."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float,
    ):
        super().__init__()

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.transformer_model = TransformerEncoderModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables),
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        transformer_features = self.transformer_model(solar_wind_input)
        predictions = self.regression_head(transformer_features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )
        if return_features:
            return output, transformer_features, None
        return output


@register_model("transformer")
def _create_transformer(config):
    """Factory function for Transformer model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating transformer model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    return TransformerOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        transformer_nhead=config.model.transformer_nhead,
        transformer_num_layers=config.model.transformer_num_layers,
        transformer_dim_feedforward=config.model.transformer_dim_feedforward,
        transformer_dropout=config.model.transformer_dropout,
    )
