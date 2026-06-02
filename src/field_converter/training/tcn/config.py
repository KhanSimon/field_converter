from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from field_converter import pathseeker as ps
from field_converter.training.config import ActivationStr, DeviceStr, InputConfig


PadModeStr = Literal["edge", "zero", "none"]
AggregateOverlapsStr = Literal["mean"]


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value).expanduser()
    raise TypeError(f"Expected a path-like value, got {type(value)}")


def _resolve_auto_dir(value: Any, *, default: Path) -> Path:
    if value is None:
        return default
    if isinstance(value, str) and value.lower() == "auto":
        return default
    return _as_path(value) or default


def _ensure_relative_to_project_root(path: Path) -> Path:
    return path if path.is_absolute() else (ps.PROJECT_ROOT / path)


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError(f"YAML config must be a mapping at top-level: {path}")
        return payload

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON config must be an object at top-level: {path}")
        return payload

    raise ValueError(f"Unsupported config extension: {path.suffix} ({path})")


def _optional_positive_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0 or null")
    return out


@dataclass(frozen=True)
class WindowDatasetConfig:
    max_sequences: Optional[int] = None
    max_windows_per_sequence: Optional[int] = None

    window_size: int = 81
    stride: int = 20
    min_valid_ratio: float = 0.5
    pad_mode: PadModeStr = "edge"

    min_in_image_joints_ratio: Optional[float] = None

    # Optional filtering to remove degenerate small boxes (in pixels).
    min_bbox_width_px: Optional[float] = None
    min_bbox_height_px: Optional[float] = None
    min_bbox_margin_px: Optional[float] = None

    def validate(self) -> None:
        if self.max_sequences is not None and self.max_sequences <= 0:
            raise ValueError("dataset.max_sequences must be positive or null")
        if self.max_windows_per_sequence is not None and self.max_windows_per_sequence <= 0:
            raise ValueError("dataset.max_windows_per_sequence must be positive or null")

        if self.window_size <= 0:
            raise ValueError("dataset.window_size must be > 0")
        if self.stride <= 0:
            raise ValueError("dataset.stride must be > 0")
        if not (0.0 <= float(self.min_valid_ratio) <= 1.0):
            raise ValueError("dataset.min_valid_ratio must be in [0,1]")
        if self.pad_mode not in {"edge", "zero", "none"}:
            raise ValueError("dataset.pad_mode must be one of: edge, zero, none")

        if self.min_in_image_joints_ratio is not None:
            r = float(self.min_in_image_joints_ratio)
            if not (0.0 <= r <= 1.0):
                raise ValueError("dataset.min_in_image_joints_ratio must be in [0,1] or null")

        if self.min_bbox_width_px is not None and float(self.min_bbox_width_px) <= 0:
            raise ValueError("dataset.min_bbox_width_px must be > 0 or null")
        if self.min_bbox_height_px is not None and float(self.min_bbox_height_px) <= 0:
            raise ValueError("dataset.min_bbox_height_px must be > 0 or null")
        if self.min_bbox_margin_px is not None and float(self.min_bbox_margin_px) < 0:
            raise ValueError("dataset.min_bbox_margin_px must be >= 0 or null")


@dataclass(frozen=True)
class TCNModelConfig:
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    temporal_hidden_dim: int = 256
    temporal_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    temporal_kernel_size: int = 3

    activation: ActivationStr = "gelu"
    dropout: float = 0.1

    head_hidden_dims: list[int] = field(default_factory=lambda: [128])

    def validate(self) -> None:
        if any(h <= 0 for h in self.encoder_hidden_dims):
            raise ValueError("model.encoder_hidden_dims must contain positive ints")
        if self.temporal_hidden_dim <= 0:
            raise ValueError("model.temporal_hidden_dim must be > 0")
        if not self.temporal_dilations or any(d <= 0 for d in self.temporal_dilations):
            raise ValueError("model.temporal_dilations must be a non-empty list of positive ints")
        if self.temporal_kernel_size <= 0:
            raise ValueError("model.temporal_kernel_size must be > 0")
        if self.temporal_kernel_size % 2 != 1:
            raise ValueError("model.temporal_kernel_size must be odd to preserve length with symmetric padding")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(f"model.activation must be 'relu' or 'gelu' (got {self.activation!r})")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("model.dropout must be in [0,1)")
        if any(h <= 0 for h in self.head_hidden_dims):
            raise ValueError("model.head_hidden_dims must contain positive ints")


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 40
    num_workers: int = 2
    group_batches_by_sequence: bool = True
    grad_clip_norm: Optional[float] = 1.0
    early_stopping_patience: int = 10

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("training.epochs must be > 0")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be >= 0")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("training.grad_clip_norm must be positive or null")
        if self.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be >= 0")


@dataclass(frozen=True)
class LossWeights:
    root: float = 1.0
    root_axis_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    root_vel: float = 0.2
    root_acc: float = 0.0
    cam3d: float = 0.0
    proj: float = 0.0

    def validate(self) -> None:
        if len(self.root_axis_weights) != 3:
            raise ValueError("loss_weights.root_axis_weights must contain exactly 3 values [x,y,z]")
        if any(float(v) < 0.0 for v in self.root_axis_weights):
            raise ValueError("loss_weights.root_axis_weights values must be >= 0")
        if sum(float(v) for v in self.root_axis_weights) <= 0.0:
            raise ValueError("at least one loss_weights.root_axis_weights value must be > 0")


@dataclass(frozen=True)
class EvalConfig:
    splits: list[Literal["train", "valid", "test"]] = field(default_factory=lambda: ["valid", "test"])
    batch_size: int = 64
    num_workers: int = 2
    save_predictions_npz: bool = True
    save_predictions_csv: bool = True
    aggregate_overlaps: AggregateOverlapsStr = "mean"


@dataclass(frozen=True)
class PlotsConfig:
    enabled: bool = True
    split_for_plots: Literal["train", "valid", "test"] = "valid"
    seq_name: Optional[str] = None
    person_idx: Optional[int] = None
    num_frames_overlay: int = 6
    show_sam2d_overlay: bool = True
    num_players_per_subplot: Optional[int] = 1


@dataclass(frozen=True)
class TCNRunConfig:
    run_name: str
    seed: int = 123
    device: DeviceStr = "auto"

    data_dir: Path = field(default_factory=lambda: ps.DATA_DIR / "features_normalized")
    output_dir: Path = field(default_factory=lambda: ps.PROJECT_ROOT / "outputs")

    input_config: InputConfig = field(default_factory=InputConfig)
    dataset: WindowDatasetConfig = field(default_factory=WindowDatasetConfig)
    model: TCNModelConfig = field(default_factory=TCNModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    eval: EvalConfig = field(default_factory=EvalConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.run_name

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions" / self.run_name

    @property
    def eval_reports_dir(self) -> Path:
        return self.output_dir / "eval_reports" / self.run_name

    @property
    def normalization_stats_path(self) -> Path:
        return self.data_dir / "normalization_stats.npz"

    @property
    def split_json_path(self) -> Path:
        return self.data_dir / "split.json"

    def validate(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must be a non-empty string")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported device: {self.device}")

        self.input_config.validate()
        self.dataset.validate()
        self.model.validate()
        self.training.validate()
        self.loss_weights.validate()


def load_tcn_run_config(config_path: Path | str) -> TCNRunConfig:
    path = Path(config_path)
    cfg = _load_yaml_or_json(path)

    run_name = str(cfg.get("run_name", "root_tcn_v1"))
    seed = int(cfg.get("seed", 123))
    device = str(cfg.get("device", "auto")).lower()

    data_dir = _resolve_auto_dir(cfg.get("data_dir"), default=ps.DATA_DIR / "features_normalized")
    output_dir = _resolve_auto_dir(cfg.get("output_dir"), default=ps.PROJECT_ROOT / "outputs")
    data_dir = _ensure_relative_to_project_root(data_dir)
    output_dir = _ensure_relative_to_project_root(output_dir)

    input_cfg_raw = cfg.get("input_config", {}) or {}
    input_cfg = InputConfig(
        use_x3d_sam_rel=bool(input_cfg_raw.get("use_x3d_sam_rel", True)),
        use_x2d_img=bool(input_cfg_raw.get("use_x2d_img", False)),
        use_x2d_box=bool(input_cfg_raw.get("use_x2d_box", False)),
        use_bbox_feat=bool(input_cfg_raw.get("use_bbox_feat", True)),
        bbox_clean_or_noisy=str(input_cfg_raw.get("bbox_clean_or_noisy", "noisy")),  # type: ignore[arg-type]
        use_cam_feat=bool(input_cfg_raw.get("use_cam_feat", True)),
        cam_feat_type=str(input_cfg_raw.get("cam_feat_type", "boosted_clean")),  # type: ignore[arg-type]
        use_valid_joints_as_input=bool(input_cfg_raw.get("use_valid_joints_as_input", True)),
    )

    dataset_raw = cfg.get("dataset", {}) or {}
    dataset_cfg = WindowDatasetConfig(
        max_sequences=dataset_raw.get("max_sequences"),
        max_windows_per_sequence=dataset_raw.get("max_windows_per_sequence"),
        window_size=int(dataset_raw.get("window_size", 81)),
        stride=int(dataset_raw.get("stride", 20)),
        min_valid_ratio=float(dataset_raw.get("min_valid_ratio", 0.5)),
        pad_mode=str(dataset_raw.get("pad_mode", "edge")).lower(),  # type: ignore[arg-type]
        min_in_image_joints_ratio=(
            None
            if dataset_raw.get("min_in_image_joints_ratio", None) is None
            else float(dataset_raw["min_in_image_joints_ratio"])
        ),
        min_bbox_width_px=(
            None if dataset_raw.get("min_bbox_width_px", None) is None else float(dataset_raw["min_bbox_width_px"])
        ),
        min_bbox_height_px=(
            None if dataset_raw.get("min_bbox_height_px", None) is None else float(dataset_raw["min_bbox_height_px"])
        ),
        min_bbox_margin_px=(
            None if dataset_raw.get("min_bbox_margin_px", None) is None else float(dataset_raw["min_bbox_margin_px"])
        ),
    )

    model_raw = cfg.get("model", {}) or {}
    model_cfg = TCNModelConfig(
        encoder_hidden_dims=list(model_raw.get("encoder_hidden_dims", [256, 256])),
        temporal_hidden_dim=int(model_raw.get("temporal_hidden_dim", 256)),
        temporal_dilations=list(model_raw.get("temporal_dilations", [1, 2, 4, 8])),
        temporal_kernel_size=int(model_raw.get("temporal_kernel_size", 3)),
        activation=str(model_raw.get("activation", "gelu")).lower(),  # type: ignore[arg-type]
        dropout=float(model_raw.get("dropout", 0.1)),
        head_hidden_dims=list(model_raw.get("head_hidden_dims", [128])),
    )

    optim_raw = cfg.get("optimizer", {}) or {}
    optim_cfg = OptimizerConfig(
        lr=float(optim_raw.get("lr", 1e-3)),
        weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
    )

    training_raw = cfg.get("training", {}) or {}
    training_cfg = TrainingConfig(
        batch_size=int(training_raw.get("batch_size", 64)),
        epochs=int(training_raw.get("epochs", 40)),
        num_workers=int(training_raw.get("num_workers", 2)),
        group_batches_by_sequence=bool(training_raw.get("group_batches_by_sequence", True)),
        grad_clip_norm=training_raw.get("grad_clip_norm", 1.0),
        early_stopping_patience=int(training_raw.get("early_stopping_patience", 10)),
    )

    loss_raw = cfg.get("loss_weights", {}) or {}
    root_axis_raw = loss_raw.get("root_axis_weights", loss_raw.get("root_axis", [1.0, 1.0, 1.0]))
    loss_w = LossWeights(
        root=float(loss_raw.get("root", 1.0)),
        root_axis_weights=[float(v) for v in root_axis_raw],
        root_vel=float(loss_raw.get("root_vel", 0.2)),
        root_acc=float(loss_raw.get("root_acc", 0.0)),
        cam3d=float(loss_raw.get("cam3d", 0.0)),
        proj=float(loss_raw.get("proj", 0.0)),
    )

    eval_raw = cfg.get("eval", {}) or {}
    eval_cfg = EvalConfig(
        splits=list(eval_raw.get("splits", ["valid", "test"])),
        batch_size=int(eval_raw.get("batch_size", 64)),
        num_workers=int(eval_raw.get("num_workers", 2)),
        save_predictions_npz=bool(eval_raw.get("save_predictions_npz", True)),
        save_predictions_csv=bool(eval_raw.get("save_predictions_csv", True)),
        aggregate_overlaps=str(eval_raw.get("aggregate_overlaps", "mean")).lower(),  # type: ignore[arg-type]
    )

    plots_raw = cfg.get("plots", {}) or {}
    plots_cfg = PlotsConfig(
        enabled=bool(plots_raw.get("enabled", True)),
        split_for_plots=str(plots_raw.get("split_for_plots", "valid")),  # type: ignore[arg-type]
        seq_name=plots_raw.get("seq_name"),
        person_idx=plots_raw.get("person_idx"),
        num_frames_overlay=int(plots_raw.get("num_frames_overlay", 6)),
        show_sam2d_overlay=bool(plots_raw.get("show_sam2d_overlay", True)),
        num_players_per_subplot=_optional_positive_int(
            plots_raw.get("num_players_per_subplot", 1),
            field_name="plots.num_players_per_subplot",
        ),
    )

    run_cfg = TCNRunConfig(
        run_name=run_name,
        seed=seed,
        device=device,  # type: ignore[arg-type]
        data_dir=data_dir,
        output_dir=output_dir,
        input_config=input_cfg,
        dataset=dataset_cfg,
        model=model_cfg,
        optimizer=optim_cfg,
        training=training_cfg,
        loss_weights=loss_w,
        eval=eval_cfg,
        plots=plots_cfg,
    )

    run_cfg.validate()
    return run_cfg
