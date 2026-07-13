# Vendored from regression-sw/src/networks/tcn.py @ 2d89767 on 2026-04-19 — DO NOT EDIT.
# Retained for its `TemporalBlock` symbol, imported by gnn.py.
# Re-sync: see src/_vendor/README.md.
"""Temporal Convolutional Network (TCN) models for time series."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ._base import _get_model_dimensions
from ._registry import register_model


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")
        if dilation <= 0:
            raise ValueError("Dilation must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=self.padding)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=self.padding)
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[:, :, :-self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        res = self.residual(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: int = 128,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]

        self.num_input_variables = num_input_variables
        self.input_sequence_length = input_sequence_length
        self.output_dim = output_dim

        self.input_projection = nn.Linear(num_input_variables, channels[0])

        layers = []
        num_channels = [channels[0]] + list(channels)
        for i in range(len(channels)):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=num_channels[i],
                    out_channels=num_channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(channels[-1], output_dim)

        self._receptive_field = 1 + 2 * (kernel_size - 1) * sum(2 ** i for i in range(len(channels)))

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.output_projection(x)
        return x


class TCNOnlyModel(nn.Module):
    """OMNI time series-only model using TCN encoder."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 128, 256]

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.tcn_encoder = TCNEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            output_dim=d_model,
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
        features = self.tcn_encoder(solar_wind_input)
        predictions = self.regression_head(features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )
        if return_features:
            return output, features, None
        return output


@register_model("tcn")
def _create_tcn(config):
    """Factory function for TCN model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    tcn_channels = getattr(config.model, 'tcn_channels', [64, 128, 256])
    if hasattr(tcn_channels, '__iter__') and not isinstance(tcn_channels, list):
        tcn_channels = list(tcn_channels)

    return TCNOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        tcn_channels=tcn_channels,
        tcn_kernel_size=getattr(config.model, 'tcn_kernel_size', 3),
        dropout=getattr(config.model, 'tcn_dropout', 0.1),
    )
