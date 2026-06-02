from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from field_converter.models.mlp import ActivationStr, MLP
from field_converter.models.transformer import (
    PositionalEncodingStr,
    TemporalPositionalEncoding,
    TransformerEncoderBackbone,
)


@dataclass(frozen=True)
class RootTransformerRefinerConfig:
    input_dim: int
    encoder_hidden_dims: tuple[int, ...] = (256,)
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: ActivationStr = "gelu"
    positional_encoding: PositionalEncodingStr = "learned"
    max_window_size: int = 81
    norm_first: bool = True
    head_hidden_dims: tuple[int, ...] = (128,)


class RootTransformerRefiner(nn.Module):
    """Root translation refiner with a Transformer Encoder temporal backbone.

    Predicts `root_cam_norm_pred` in normalized space with shape (B,T,3).
    """

    supports_valid_mask = True

    def __init__(
        self,
        *,
        input_dim: int,
        encoder_hidden_dims: Iterable[int] = (256,),
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: ActivationStr = "gelu",
        positional_encoding: PositionalEncodingStr = "learned",
        max_window_size: int = 81,
        norm_first: bool = True,
        head_hidden_dims: Iterable[int] = (128,),
    ) -> None:
        super().__init__()

        self.encoder = MLP(
            input_dim=int(input_dim),
            hidden_dims=list(encoder_hidden_dims),
            output_dim=int(d_model),
            activation=activation,
            dropout=float(dropout),
        )
        self.position = TemporalPositionalEncoding(
            d_model=int(d_model),
            max_window_size=int(max_window_size),
            encoding=positional_encoding,
        )
        self.transformer = TransformerEncoderBackbone(
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=activation,
            norm_first=bool(norm_first),
        )
        self.head = MLP(
            input_dim=int(d_model),
            hidden_dims=list(head_hidden_dims),
            output_dim=3,
            activation=activation,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,D), got {x.shape}")
        B, T, D = x.shape
        if valid_mask is not None and tuple(valid_mask.shape) != (B, T):
            raise ValueError(f"valid_mask must have shape (B,T)={(B, T)}, got {valid_mask.shape}")

        y = self.encoder(x.reshape(B * T, D)).reshape(B, T, -1)
        y = self.position(y)
        y = self.transformer(y, valid_mask=valid_mask)
        return self.head(y.reshape(B * T, -1)).reshape(B, T, 3)
