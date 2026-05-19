from __future__ import annotations

import torch


def loss_root_velocity(*args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
    """Placeholder for future temporal losses.

    Not used in the V1 frame-wise model. A velocity loss requires ordered frames
    (and typically a temporal model or sequence-aware batching).
    """
    raise NotImplementedError("Temporal losses are not enabled in V1")
