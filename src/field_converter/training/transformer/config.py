from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from field_converter import pathseeker as ps
from field_converter.models.transformer import PositionalEncodingStr
from field_converter.training.config import ActivationStr, DeviceStr, InputConfig, PredictionModeStr, _parse_prediction_mode
from field_converter.training.tcn.config import (
    EvalConfig,
    LossWeights,
    OptimizerConfig,
    PlotsConfig,
    TrainingConfig,
    WindowDatasetConfig,
    _ensure_relative_to_project_root,
    _load_yaml_or_json,
    _optional_positive_int,
    _resolve_auto_dir,
)


AggregateOverlapsStr = Literal["mean"]


@dataclass(frozen=True)
class TransformerModelConfig:
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256])
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: ActivationStr = "gelu"
    positional_encoding: PositionalEncodingStr = "learned"
    max_window_size: int = 81
    norm_first: bool = True
    head_hidden_dims: list[int] = field(default_factory=lambda: [128])

    def validate(self) -> None:
        if any(h <= 0 for h in self.encoder_hidden_dims):
            raise ValueError("model.encoder_hidden_dims must contain positive ints")
        if self.d_model <= 0:
            raise ValueError("model.d_model must be > 0")
        if self.num_layers <= 0:
            raise ValueError("model.num_layers must be > 0")
        if self.num_heads <= 0:
            raise ValueError("model.num_heads must be > 0")
        if self.d_model % self.num_heads != 0:
            raise ValueError("model.d_model must be divisible by model.num_heads")
        if self.dim_feedforward <= 0:
            raise ValueError("model.dim_feedforward must be > 0")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("model.dropout must be in [0,1)")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(f"model.activation must be 'relu' or 'gelu' (got {self.activation!r})")
        if self.positional_encoding not in {"learned", "sinusoidal"}:
            raise ValueError("model.positional_encoding must be 'learned' or 'sinusoidal'")
        if self.max_window_size <= 0:
            raise ValueError("model.max_window_size must be > 0")
        if any(h <= 0 for h in self.head_hidden_dims):
            raise ValueError("model.head_hidden_dims must contain positive ints")


@dataclass(frozen=True)
class TransformerRunConfig:
    run_name: str
    seed: int = 123
    device: DeviceStr = "auto"
    prediction_mode: PredictionModeStr = "absolute"

    data_dir: Path = field(default_factory=lambda: ps.DATA_DIR / "features_normalized")
    root_init_dir: Path = field(default_factory=lambda: ps.DATA_DIR / "root_init_cam_normalized")
    output_dir: Path = field(default_factory=lambda: ps.PROJECT_ROOT / "outputs")

    input_config: InputConfig = field(default_factory=InputConfig)
    dataset: WindowDatasetConfig = field(default_factory=WindowDatasetConfig)
    model: TransformerModelConfig = field(default_factory=TransformerModelConfig)
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
        if self.prediction_mode not in {"absolute", "delta"}:
            raise ValueError(f"Unsupported prediction_mode: {self.prediction_mode}")

        self.input_config.validate()
        self.dataset.validate()
        self.model.validate()
        if self.dataset.window_size > self.model.max_window_size:
            raise ValueError("dataset.window_size must be <= model.max_window_size")
        self.training.validate()
        self.loss_weights.validate()


def load_transformer_run_config(config_path: Path | str) -> TransformerRunConfig:
    path = Path(config_path)
    cfg = _load_yaml_or_json(path)

    run_name = str(cfg.get("run_name", "root_transformer_v1"))
    seed = int(cfg.get("seed", 123))
    device = str(cfg.get("device", "auto")).lower()
    prediction_mode = _parse_prediction_mode(cfg.get("prediction_mode", "absolute"))

    data_dir = _resolve_auto_dir(cfg.get("data_dir"), default=ps.DATA_DIR / "features_normalized")
    output_dir = _resolve_auto_dir(cfg.get("output_dir"), default=ps.PROJECT_ROOT / "outputs")
    data_dir = _ensure_relative_to_project_root(data_dir)
    root_init_dir = _resolve_auto_dir(
        cfg.get("root_init_dir"),
        default=data_dir.parent / "root_init_cam_normalized",
    )
    root_init_dir = _ensure_relative_to_project_root(root_init_dir)
    output_dir = _ensure_relative_to_project_root(output_dir)

    input_cfg_raw = cfg.get("input_config", {}) or {}
    input_cfg = InputConfig(
        use_x3d_sam_rel=bool(input_cfg_raw.get("use_x3d_sam_rel", True)),
        use_x2d_img=bool(input_cfg_raw.get("use_x2d_img", False)),
        use_x2d_box=bool(input_cfg_raw.get("use_x2d_box", False)),
        use_pitch_points_2d=bool(input_cfg_raw.get("use_pitch_points_2d", False)),
        use_bbox_feat=bool(input_cfg_raw.get("use_bbox_feat", True)),
        bbox_clean_or_noisy=str(input_cfg_raw.get("bbox_clean_or_noisy", "noisy")),  # type: ignore[arg-type]
        use_cam_feat=bool(input_cfg_raw.get("use_cam_feat", True)),
        cam_feat_type=str(input_cfg_raw.get("cam_feat_type", "boosted_clean")),  # type: ignore[arg-type]
        use_ground_intersection=bool(input_cfg_raw.get("use_ground_intersection", False)),
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
    model_cfg = TransformerModelConfig(
        encoder_hidden_dims=list(model_raw.get("encoder_hidden_dims", [256])),
        d_model=int(model_raw.get("d_model", 256)),
        num_layers=int(model_raw.get("num_layers", 2)),
        num_heads=int(model_raw.get("num_heads", 4)),
        dim_feedforward=int(model_raw.get("dim_feedforward", 512)),
        dropout=float(model_raw.get("dropout", 0.1)),
        activation=str(model_raw.get("activation", "gelu")).lower(),  # type: ignore[arg-type]
        positional_encoding=str(model_raw.get("positional_encoding", "learned")).lower(),  # type: ignore[arg-type]
        max_window_size=int(model_raw.get("max_window_size", 81)),
        norm_first=bool(model_raw.get("norm_first", True)),
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
    root_axis_raw = loss_raw.get("root_axis_weights", loss_raw.get("root_axis", [1.0, 1.0, 2.0]))
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
        root_error_ground_vs_air_histogram=bool(
            plots_raw.get("root_error_ground_vs_air_histogram", False)
        ),
        seq_name=plots_raw.get("seq_name"),
        person_idx=plots_raw.get("person_idx"),
        num_frames_overlay=int(plots_raw.get("num_frames_overlay", 6)),
        show_sam2d_overlay=bool(plots_raw.get("show_sam2d_overlay", True)),
        num_players_per_subplot=_optional_positive_int(
            plots_raw.get("num_players_per_subplot", 1),
            field_name="plots.num_players_per_subplot",
        ),
    )

    run_cfg = TransformerRunConfig(
        run_name=run_name,
        seed=seed,
        device=device,  # type: ignore[arg-type]
        prediction_mode=prediction_mode,
        data_dir=data_dir,
        root_init_dir=root_init_dir,
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
