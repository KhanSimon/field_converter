from __future__ import annotations

from typing import Optional, Protocol

import torch


class TemporalRootModel(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,D) -> (B,T,3)
        ...


def forward_temporal_root_model(
    model: TemporalRootModel,
    x: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward a temporal root model, passing valid_mask when supported."""
    if valid_mask is not None and bool(getattr(model, "supports_valid_mask", False)):
        return model(x, valid_mask=valid_mask)  # type: ignore[misc,call-arg]
    return model(x)
