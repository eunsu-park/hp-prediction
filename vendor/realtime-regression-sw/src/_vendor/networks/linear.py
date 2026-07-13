# Vendored from regression-sw/src/networks/linear.py @ 2d89767 on 2026-04-20 — DO NOT EDIT.
# Re-sync: see src/_vendor/README.md.
"""Linear models for time series processing."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ._base import _get_model_dimensions
from ._registry import register_model


class LinearEncoder(nn.Module):
    """Simple linear encoder for time series data.

    Flattens the input and processes through dense layers.

    Args:
        input_size: Total input size after flattening (seq_len * num_vars).
        output_dim: Output feature dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate for regularization.
    """

    def __init__(self, input_size: int, output_dim: int = 256,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        if input_size <= 0:
            raise ValueError("Input size must be positive")
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")

        self.input_size = input_size
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, seq_len, num_vars), got {x.dim()}D")
        return self.encoder(x)


class LinearOnlyModel(nn.Module):
    """OMNI time series-only model using Linear encoder."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length
        self.d_model = d_model

        ts_input_size = input_sequence_length * num_input_variables
        self.ts_encoder = LinearEncoder(
            input_size=ts_input_size,
            output_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables),
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        ts_features = self.ts_encoder(solar_wind_input)
        predictions = self.regression_head(ts_features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )
        if return_features:
            return output, ts_features, None
        return output


@register_model("linear")
def _create_linear(config):
    """Factory function for Linear model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating linear model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    linear_dropout = getattr(config.model, 'baseline_dropout', 0.1)
    return LinearOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        dropout=linear_dropout,
    )
