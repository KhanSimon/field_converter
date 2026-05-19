from __future__ import annotations

import torch


def world_to_cam(X_world: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Convert world coordinates to camera coordinates (row-vector convention).

    Formula (row-vectors):
        X_cam = X_world @ R.T + t

    Shapes
    ------
    - X_world: (..., 3)
    - R: (..., 3, 3) broadcastable to X_world
    - t: (..., 3) broadcastable to X_world
    """
    Rt = R.transpose(-1, -2)

    # Special-case: (B,3) with per-sample (B,3,3) to avoid matmul broadcasting to (B,B,3).
    if X_world.ndim == 2 and R.ndim == 3:
        if t.ndim != 2:
            raise ValueError(f"Expected t to have shape (B,3) for batched input, got {t.shape}")
        return torch.bmm(X_world.unsqueeze(1), Rt).squeeze(1) + t

    # General case: broadcast t to X_world (e.g. (B,J,3) + (B,3)).
    while t.ndim < X_world.ndim:
        t = t.unsqueeze(-2)
    return torch.matmul(X_world, Rt) + t


def cam_to_world(X_cam: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Convert camera coordinates to world coordinates (row-vector convention).

    Formula (row-vectors):
        X_world = (X_cam - t) @ R

    Shapes
    ------
    - X_cam: (..., 3)
    - R: (..., 3, 3) broadcastable to X_cam
    - t: (..., 3) broadcastable to X_cam
    """
    # Special-case: (B,3) with per-sample (B,3,3) to avoid matmul broadcasting to (B,B,3).
    if X_cam.ndim == 2 and R.ndim == 3:
        if t.ndim != 2:
            raise ValueError(f"Expected t to have shape (B,3) for batched input, got {t.shape}")
        return torch.bmm((X_cam - t).unsqueeze(1), R).squeeze(1)

    # General case: broadcast t to X_cam (e.g. (B,J,3) - (B,3)).
    while t.ndim < X_cam.ndim:
        t = t.unsqueeze(-2)
    return torch.matmul(X_cam - t, R)
