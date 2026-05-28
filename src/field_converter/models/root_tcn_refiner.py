from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from field_converter.models.mlp import ActivationStr, MLP
from field_converter.models.tcn import TCNBackbone


@dataclass(frozen=True)
class RootTCNRefinerConfig:
    input_dim: int
    encoder_hidden_dims: tuple[int, ...] = (256, 256)
    temporal_hidden_dim: int = 256
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8)
    temporal_kernel_size: int = 3
    activation: ActivationStr = "gelu"
    dropout: float = 0.1
    head_hidden_dims: tuple[int, ...] = (128,)


class RootTCNRefiner(nn.Module):
    """Root translation refiner with an offline (non-causal) TCN.

    Predicts `root_cam_norm_pred` in normalized space with shape (B,T,3).
    """

    def __init__(
        self,
        *,
        input_dim: int,
        encoder_hidden_dims: Iterable[int] = (256, 256),
        temporal_hidden_dim: int = 256,
        temporal_dilations: Iterable[int] = (1, 2, 4, 8),
        temporal_kernel_size: int = 3,
        activation: ActivationStr = "gelu",
        dropout: float = 0.1,
        head_hidden_dims: Iterable[int] = (128,),
    ) -> None:
        super().__init__()

        self.encoder = MLP(
            input_dim=int(input_dim),
            hidden_dims=list(encoder_hidden_dims),
            output_dim=int(temporal_hidden_dim),
            activation=activation,
            dropout=float(dropout),
        )

        self.tcn = TCNBackbone(
            hidden_dim=int(temporal_hidden_dim),
            dilations=list(temporal_dilations),
            kernel_size=int(temporal_kernel_size),
            dropout=float(dropout),
            activation=activation,
        )

        self.head = MLP(
            input_dim=int(temporal_hidden_dim),
            hidden_dims=list(head_hidden_dims),
            output_dim=3,
            activation=activation,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,D), got {x.shape}")
        B, T, D = x.shape

        # Per-frame encoder.
        y = self.encoder(x.reshape(B * T, D))  # (B*T,H)
        y = y.reshape(B, T, -1)

        # Temporal backbone.
        y = self.tcn(y)  # (B,T,H)

        # Root head.
        out = self.head(y.reshape(B * T, -1)).reshape(B, T, 3)
        return out
