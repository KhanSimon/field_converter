from __future__ import annotations

from typing import Any, Dict, Literal

import torch


PredictionModeStr = Literal["absolute", "delta"]


def validate_prediction_mode(mode: str) -> PredictionModeStr:
    mode_l = str(mode).lower()
    if mode_l not in {"absolute", "delta"}:
        raise ValueError(f"prediction_mode must be 'absolute' or 'delta' (got {mode!r})")
    return mode_l  # type: ignore[return-value]


def apply_prediction_mode(
    model_output_norm: torch.Tensor,
    batch: Dict[str, Any],
    *,
    prediction_mode: PredictionModeStr,
) -> torch.Tensor:
    """Convert raw model output to final normalized root prediction."""
    if prediction_mode == "absolute":
        return model_output_norm

    if "root_init_norm" not in batch:
        raise KeyError("prediction_mode='delta' requires batch['root_init_norm']")

    root_init = batch["root_init_norm"].to(device=model_output_norm.device, dtype=model_output_norm.dtype)
    if root_init.shape != model_output_norm.shape:
        raise ValueError(
            "root_init_norm must have the same shape as model output in delta mode, "
            f"got root_init={root_init.shape}, model_output={model_output_norm.shape}"
        )
    return root_init + model_output_norm
