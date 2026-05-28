from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from field_converter.models.mlp import ActivationStr


def _activation_layer(name: ActivationStr) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualTCNBlock(nn.Module):
    """A simple residual temporal conv block (offline / non-causal)."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: ActivationStr,
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric 'same' padding")
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if dilation <= 0:
            raise ValueError("dilation must be > 0")

        pad = (dilation * (kernel_size - 1)) // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=int(pad),
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=int(pad),
        )

        self.act = _activation_layer(activation)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = self.act(y)
        y = self.drop(y)

        return x + y


class TCNBackbone(nn.Module):
    """Temporal convolutional backbone.

    Input/Output are (B,T,C). Internally uses Conv1d (B,C,T).
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        dilations: Iterable[int],
        kernel_size: int,
        dropout: float,
        activation: ActivationStr,
    ) -> None:
        super().__init__()

        dilations_list = [int(d) for d in dilations]
        if not dilations_list:
            raise ValueError("dilations must be non-empty")

        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(d),
                    dropout=float(dropout),
                    activation=activation,
                )
                for d in dilations_list
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,C) input, got shape={x.shape}")

        # (B,T,C) -> (B,C,T)
        y = x.transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        # (B,C,T) -> (B,T,C)
        return y.transpose(1, 2)
