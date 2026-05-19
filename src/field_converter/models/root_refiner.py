from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from field_converter.models.mlp import ActivationStr, MLP


@dataclass(frozen=True)
class RootRefinerConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    activation: ActivationStr = "gelu"
    dropout: float = 0.1


class RootRefiner(nn.Module):
    """Frame-wise root translation refiner.

    Predicts `root_cam_norm_pred` in normalized space (B,3).
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Iterable[int] = (256, 256, 128),
        activation: ActivationStr = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=int(input_dim),
            hidden_dims=list(hidden_dims),
            output_dim=3,
            activation=activation,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.mlp(x)
