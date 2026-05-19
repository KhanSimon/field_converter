from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MeanRootBaseline:
    """Baseline A: constant mean-root predictor in normalized space."""

    mean_root_norm: torch.Tensor  # (3,)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        mean = self.mean_root_norm.to(device=x.device, dtype=x.dtype)
        return mean.unsqueeze(0).expand(b, 3)


@torch.no_grad()
def compute_mean_root_norm(
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compute mean(root_gt_norm) over a dataloader."""
    s = torch.zeros((3,), dtype=torch.float32, device=device)
    n = 0
    for batch in dataloader:
        root = batch["root_gt"].to(device=device, dtype=torch.float32)
        s += root.sum(dim=0)
        n += int(root.shape[0])
    return s / float(max(n, 1))
