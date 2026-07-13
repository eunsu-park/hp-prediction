# Vendored from regression-sw/src/networks/timesnet.py @ 2d89767 on 2026-04-20 — DO NOT EDIT.
# Re-sync: see src/_vendor/README.md.
"""TimesNet models for time series with FFT-based period detection."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model


class InceptionBlock(nn.Module):
    """Multi-scale 2D convolution block (Inception-style)."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=2 * k + 1, padding=k)
            for k in range(num_kernels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(conv(x) for conv in self.convs)


class TimesBlock(nn.Module):
    """FFT-based period detection + 2D convolution block."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int = 256,
        top_k: int = 3,
        num_kernels: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        max_k = seq_len // 2
        if top_k > max_k:
            raise ValueError(f"top_k ({top_k}) must be <= seq_len//2 ({max_k})")
        self.top_k = top_k

        self.inception1 = InceptionBlock(d_model, d_ff, num_kernels)
        self.inception2 = InceptionBlock(d_ff, d_model, num_kernels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        x_freq = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_freq).mean(dim=-1)
        amplitude[:, 0] = 0

        _, top_indices = torch.topk(amplitude, self.top_k, dim=1)
        top_indices = top_indices.detach()

        top_amplitudes = torch.gather(amplitude, 1, top_indices)
        weights = F.softmax(top_amplitudes, dim=1)

        results = []
        for k in range(self.top_k):
            freq_idx = top_indices[:, k]
            period = int(seq_len / (freq_idx.float().mean().clamp(min=1)).item())
            period = max(period, 2)

            pad_len = (period - seq_len % period) % period
            x_padded = F.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x
            padded_len = x_padded.size(1)

            x_2d = x_padded.permute(0, 2, 1)
            x_2d = x_2d.reshape(batch_size, d_model, period, padded_len // period)

            x_2d = self.inception1(x_2d)
            x_2d = self.activation(x_2d)
            x_2d = self.inception2(x_2d)

            x_1d = x_2d.reshape(batch_size, d_model, padded_len)
            x_1d = x_1d.permute(0, 2, 1)
            x_1d = x_1d[:, :seq_len, :]

            results.append(x_1d)

        results = torch.stack(results, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        output = (results * weights).sum(dim=1)
        return output + x


class TimesNetEncoder(nn.Module):
    """TimesNet encoder for multivariate time series."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        d_model: int = 64,
        d_ff: int = 128,
        num_blocks: int = 2,
        top_k: int = 3,
        num_kernels: int = 3,
        dropout: float = 0.1,
        output_dim: int = 128,
        enable_cross_variable: bool = True,
    ):
        super().__init__()
        self.num_input_variables = num_input_variables
        self.input_sequence_length = input_sequence_length

        self.input_projection = nn.Linear(num_input_variables, d_model)

        self.blocks = nn.ModuleList([
            TimesBlock(
                seq_len=input_sequence_length,
                d_model=d_model,
                d_ff=d_ff,
                top_k=top_k,
                num_kernels=num_kernels,
            )
            for _ in range(num_blocks)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])

        self.enable_cross_variable = enable_cross_variable
        if enable_cross_variable:
            self.cross_var_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=4,
                dropout=dropout, batch_first=True,
            )
            self.cross_var_norm = nn.LayerNorm(d_model)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)

        for block, dropout, norm in zip(self.blocks, self.dropouts, self.norms):
            h = norm(dropout(block(h)) + h)

        if self.enable_cross_variable:
            residual = h
            h, _ = self.cross_var_attn(h, h, h)
            h = self.cross_var_norm(h + residual)

        h = h.transpose(1, 2)
        h = self.global_pool(h).squeeze(-1)
        h = self.output_projection(h)
        return h


class TimesNetOnlyModel(nn.Module):
    """Time series model using TimesNet encoder."""

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 64,
        d_ff: int = 128,
        output_d_model: int = 128,
        num_blocks: int = 2,
        top_k: int = 3,
        num_kernels: int = 3,
        dropout: float = 0.1,
        enable_cross_variable: bool = True,
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.timesnet_encoder = TimesNetEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=d_model,
            d_ff=d_ff,
            num_blocks=num_blocks,
            top_k=top_k,
            num_kernels=num_kernels,
            dropout=dropout,
            output_dim=output_d_model,
            enable_cross_variable=enable_cross_variable,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(output_d_model, output_d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_d_model // 2, target_sequence_length * num_target_variables),
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        features = self.timesnet_encoder(solar_wind_input)
        predictions = self.regression_head(features)
        output = predictions.reshape(
            predictions.size(0),
            self.target_sequence_length,
            self.num_target_variables,
        )
        if return_features:
            return output, features, None
        return output


@register_model("timesnet")
def _create_timesnet(config):
    """Factory function for TimesNet model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating timesnet model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    tn_d_model = getattr(config.model, 'timesnet_d_model', 64)
    tn_d_ff = getattr(config.model, 'timesnet_d_ff', 128)
    tn_num_blocks = getattr(config.model, 'timesnet_num_blocks', 2)
    tn_top_k = getattr(config.model, 'timesnet_top_k', 3)
    tn_num_kernels = getattr(config.model, 'timesnet_num_kernels', 3)
    tn_dropout = getattr(config.model, 'timesnet_dropout', 0.1)
    tn_cross_var = getattr(config.model, 'timesnet_cross_variable', True)

    return TimesNetOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=tn_d_model,
        d_ff=tn_d_ff,
        output_d_model=config.model.d_model,
        num_blocks=tn_num_blocks,
        top_k=tn_top_k,
        num_kernels=tn_num_kernels,
        dropout=tn_dropout,
        enable_cross_variable=tn_cross_var,
    )
