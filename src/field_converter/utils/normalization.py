from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from field_converter.data_preparation.normalize import NormalizationStats


@dataclass(frozen=True)
class TorchNormalizationStats:
    """Normalization statistics converted to torch tensors."""

    mean_sam3d_rel: torch.Tensor  # (25,3)
    std_sam3d_rel: torch.Tensor   # (25,3)

    mean_root: torch.Tensor       # (3,)
    std_root: torch.Tensor        # (3,)

    mean_ground_intersection: torch.Tensor  # (3,)
    std_ground_intersection: torch.Tensor   # (3,)

    mean_C: torch.Tensor          # (3,)
    std_C: torch.Tensor           # (3,)

    @staticmethod
    def load(npz_path: Path | str, *, device: torch.device | str = "cpu") -> "TorchNormalizationStats":
        stats = NormalizationStats.load(Path(npz_path))

        def as_t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device)

        return TorchNormalizationStats(
            mean_sam3d_rel=as_t(stats.mean_sam3d_rel),
            std_sam3d_rel=as_t(stats.std_sam3d_rel),
            mean_root=as_t(stats.mean_root),
            std_root=as_t(stats.std_root),
            mean_ground_intersection=as_t(stats.mean_ground_intersection),
            std_ground_intersection=as_t(stats.std_ground_intersection),
            mean_C=as_t(stats.mean_C),
            std_C=as_t(stats.std_C),
        )

    def to(self, device: torch.device | str) -> "TorchNormalizationStats":
        return TorchNormalizationStats(
            mean_sam3d_rel=self.mean_sam3d_rel.to(device),
            std_sam3d_rel=self.std_sam3d_rel.to(device),
            mean_root=self.mean_root.to(device),
            std_root=self.std_root.to(device),
            mean_ground_intersection=self.mean_ground_intersection.to(device),
            std_ground_intersection=self.std_ground_intersection.to(device),
            mean_C=self.mean_C.to(device),
            std_C=self.std_C.to(device),
        )

    def denorm_root(self, root_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize root translation: (..,3) -> (..,3) in meters."""
        return root_norm * self.std_root + self.mean_root

    def denorm_sam3d_rel(self, X_sam_rel_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize SAM3D relative skeleton: (..,25,3) -> (..,25,3) in meters."""
        return X_sam_rel_norm * self.std_sam3d_rel + self.mean_sam3d_rel

    def denorm_ground_intersection(self, ground_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize ground_intersection: (..,3) -> (..,3) in world meters."""
        return ground_norm * self.std_ground_intersection + self.mean_ground_intersection
