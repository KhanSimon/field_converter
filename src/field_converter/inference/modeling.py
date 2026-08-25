"""Factories for loading trained root-refiner models for qualitative inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from torch import nn

from field_converter.models.root_refiner import RootRefiner
from field_converter.models.root_tcn_refiner import RootTCNRefiner
from field_converter.models.root_transformer_refiner import RootTransformerRefiner
from field_converter.training.config import load_run_config
from field_converter.training.dataset import infer_input_dim
from field_converter.training.tcn.config import load_tcn_run_config
from field_converter.training.transformer.config import load_transformer_run_config
from field_converter.utils.torch_utils import get_device


ModelType = Literal["mlp", "tcn", "transformer"]


@dataclass(frozen=True)
class InferenceModel:
    model_type: ModelType
    config: Any
    config_path: Path
    checkpoint_path: Path
    checkpoint_metadata: dict[str, Any]
    model: nn.Module
    device: torch.device
    input_dim: int
    temporal: bool
    window_size: Optional[int]
    stride: Optional[int]
    pad_mode: Optional[str]
    batch_size: int


def _resolve_checkpoint(checkpoint: str | Path, checkpoints_dir: Path) -> Path:
    value = str(checkpoint)
    if value in {"best", "last"}:
        return checkpoints_dir / f"{value}.pt"
    return Path(value)


def load_inference_model(
    *,
    model_type: ModelType,
    config_path: Path | str,
    checkpoint: Path | str = "best",
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
) -> InferenceModel:
    """Reconstruct a model from its exact YAML and load its checkpoint."""
    config_file = Path(config_path)
    if model_type == "mlp":
        cfg = load_run_config(config_file)
        model: nn.Module = RootRefiner(
            input_dim=infer_input_dim(cfg.input_config),
            hidden_dims=cfg.model.hidden_dims,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
        )
        temporal = False
        window_size = None
        stride = None
        pad_mode = None
    elif model_type == "tcn":
        cfg = load_tcn_run_config(config_file)
        model = RootTCNRefiner(
            input_dim=infer_input_dim(cfg.input_config),
            encoder_hidden_dims=cfg.model.encoder_hidden_dims,
            temporal_hidden_dim=cfg.model.temporal_hidden_dim,
            temporal_dilations=cfg.model.temporal_dilations,
            temporal_kernel_size=cfg.model.temporal_kernel_size,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            head_hidden_dims=cfg.model.head_hidden_dims,
        )
        temporal = True
        window_size = int(cfg.dataset.window_size)
        stride = int(cfg.dataset.stride)
        pad_mode = str(cfg.dataset.pad_mode)
    elif model_type == "transformer":
        cfg = load_transformer_run_config(config_file)
        model = RootTransformerRefiner(
            input_dim=infer_input_dim(cfg.input_config),
            encoder_hidden_dims=cfg.model.encoder_hidden_dims,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            positional_encoding=cfg.model.positional_encoding,
            max_window_size=cfg.model.max_window_size,
            norm_first=cfg.model.norm_first,
            head_hidden_dims=cfg.model.head_hidden_dims,
        )
        temporal = True
        window_size = int(cfg.dataset.window_size)
        stride = int(cfg.dataset.stride)
        pad_mode = str(cfg.dataset.pad_mode)
        if window_size > int(cfg.model.max_window_size):
            raise ValueError(
                f"dataset.window_size={window_size} exceeds transformer max_window_size="
                f"{cfg.model.max_window_size}"
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    checkpoint_path = _resolve_checkpoint(checkpoint, cfg.checkpoints_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_payload, dict) or "model_state_dict" not in checkpoint_payload:
        raise ValueError(f"Invalid checkpoint payload (missing model_state_dict): {checkpoint_path}")
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)

    device_name = str(device_override or cfg.device).lower()
    device = get_device(device_name)  # type: ignore[arg-type]
    model.to(device)
    model.eval()

    batch_size = int(batch_size_override or cfg.eval.batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    checkpoint_metadata = {
        key: checkpoint_payload[key]
        for key in ("epoch", "best_epoch", "best_root_error_mean_m")
        if key in checkpoint_payload
    }
    return InferenceModel(
        model_type=model_type,
        config=cfg,
        config_path=config_file,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        model=model,
        device=device,
        input_dim=infer_input_dim(cfg.input_config),
        temporal=temporal,
        window_size=window_size,
        stride=stride,
        pad_mode=pad_mode,
        batch_size=batch_size,
    )

