from __future__ import annotations

import torch


def project_cam_to_image(
    X_cam: torch.Tensor,
    K: torch.Tensor,
    k: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Project 3D camera coordinates to 2D image pixels with radial distortion.

    Distortion model (k1,k2):
        x = X/Z
        y = Y/Z
        r2 = x^2 + y^2
        factor = 1 + k1*r2 + k2*r2^2
        x_d = x * factor
        y_d = y * factor
        u = fx*x_d + cx
        v = fy*y_d + cy

    Parameters
    ----------
    X_cam:
        (..., J, 3) camera coordinates.
    K:
        (..., 3, 3) intrinsics (broadcastable to X_cam batch dims).
    k:
        (..., 2) radial distortion (k1,k2), broadcastable.

    Returns
    -------
    uv:
        (..., J, 2) pixel coordinates.
    """
    if X_cam.shape[-1] != 3:
        raise ValueError(f"X_cam must have last dim 3, got {X_cam.shape}")
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must have shape (...,3,3), got {K.shape}")
    if k.shape[-1] != 2:
        raise ValueError(f"k must have shape (...,2), got {k.shape}")

    X = X_cam[..., 0]
    Y = X_cam[..., 1]
    Z = X_cam[..., 2].clamp(min=float(eps))

    x = X / Z
    y = Y / Z

    r2 = x * x + y * y
    k1 = k[..., 0].unsqueeze(-1)
    k2 = k[..., 1].unsqueeze(-1)
    factor = 1.0 + k1 * r2 + k2 * (r2 * r2)

    x_d = x * factor
    y_d = y * factor

    fx = K[..., 0, 0].unsqueeze(-1)
    fy = K[..., 1, 1].unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1)

    u = fx * x_d + cx
    v = fy * y_d + cy

    return torch.stack([u, v], dim=-1)
