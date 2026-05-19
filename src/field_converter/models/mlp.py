from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import nn


ActivationStr = Literal["relu", "gelu"]


def make_activation(name: ActivationStr) -> nn.Module:
    name_l = str(name).lower()
    if name_l == "relu":
        return nn.ReLU(inplace=True)
    if name_l == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Simple feed-forward MLP.

    Architecture: Linear -> Activation -> Dropout repeated, then Linear output.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: ActivationStr = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dims_list = list(hidden_dims)
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if any(h <= 0 for h in hidden_dims_list):
            raise ValueError("hidden_dims must contain positive ints")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout must be in [0,1)")

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims_list:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = int(h)

        layers.append(nn.Linear(prev, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)
