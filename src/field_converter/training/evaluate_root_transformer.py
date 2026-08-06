from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from field_converter.evaluation.temporal_evaluator import TemporalEvaluator
from field_converter.evaluation.visualization import (
    plot_reprojection_overlay,
    plot_root_error_ground_vs_air_histogram,
    plot_root_timeseries,
    plot_training_curves,
    plot_world_trajectory_xy,
)
from field_converter.models.root_refiner import RootRefiner
from field_converter.models.root_tcn_refiner import RootTCNRefiner
from field_converter.models.root_transformer_refiner import RootTransformerRefiner
from field_converter.training.config import load_run_config
from field_converter.training.dataset import infer_input_dim
from field_converter.training.tcn.config import load_tcn_run_config
from field_converter.training.tcn.window_dataset import NormalizedWindowDataset
from field_converter.training.transformer.config import TransformerRunConfig, load_transformer_run_config
from field_converter.utils.io import ensure_dir, write_json
from field_converter.utils.normalization import TorchNormalizationStats
from field_converter.utils.torch_utils import get_device


class TemporalMeanRootBaseline:
    supports_valid_mask = True

    def __init__(self, mean_root_norm: torch.Tensor) -> None:
        self.mean_root_norm = mean_root_norm

    def __call__(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = int(x.shape[0]), int(x.shape[1])
        mean = self.mean_root_norm.to(device=x.device, dtype=x.dtype)
        return mean.view(1, 1, 3).expand(B, T, 3)


class WindowedFramewiseRootModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,D), got {x.shape}")
        B, T, D = x.shape
        return self.model(x.reshape(B * T, D)).reshape(B, T, 3)


def _resolve_checkpoint(checkpoint: str, checkpoints_dir: Path) -> Path:
    if checkpoint in {"best", "last"}:
        return checkpoints_dir / f"{checkpoint}.pt"
    return Path(checkpoint)


def _make_window_loader(
    *,
    cfg: TransformerRunConfig,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    ds = NormalizedWindowDataset(
        data_dir=cfg.data_dir,
        split=split,  # type: ignore[arg-type]
        input_config=cfg.input_config,
        prediction_mode=cfg.prediction_mode,
        root_init_dir=cfg.root_init_dir,
        seed=cfg.seed,
        max_sequences=cfg.dataset.max_sequences,
        max_windows_per_sequence=None,
        window_size=cfg.dataset.window_size,
        stride=cfg.dataset.stride,
        min_valid_ratio=0.0,
        pad_mode=cfg.dataset.pad_mode,
        min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
        min_bbox_width_px=cfg.dataset.min_bbox_width_px,
        min_bbox_height_px=cfg.dataset.min_bbox_height_px,
        min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
        filter_by_min_valid_ratio=False,
    )
    dl_kwargs: dict[str, object] = {}
    if num_workers > 0:
        dl_kwargs.update({"persistent_workers": True, "prefetch_factor": 1})
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )


@torch.no_grad()
def _compute_temporal_mean_root_norm(
    *,
    cfg: TransformerRunConfig,
    device: torch.device,
) -> torch.Tensor:
    ds = NormalizedWindowDataset(
        data_dir=cfg.data_dir,
        split="train",
        input_config=cfg.input_config,
        seed=cfg.seed,
        max_sequences=cfg.dataset.max_sequences,
        max_windows_per_sequence=cfg.dataset.max_windows_per_sequence,
        window_size=cfg.dataset.window_size,
        stride=cfg.dataset.stride,
        min_valid_ratio=cfg.dataset.min_valid_ratio,
        pad_mode=cfg.dataset.pad_mode,
        min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
        min_bbox_width_px=cfg.dataset.min_bbox_width_px,
        min_bbox_height_px=cfg.dataset.min_bbox_height_px,
        min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
        filter_by_min_valid_ratio=True,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=0)

    total = torch.zeros((3,), dtype=torch.float32, device=device)
    count = torch.zeros((), dtype=torch.float32, device=device)
    for batch in loader:
        root = batch["root_gt"].to(device=device, dtype=torch.float32)
        valid = batch["valid_mask"].to(device=device).bool()
        if bool(valid.any()):
            total += root[valid].sum(dim=0)
            count += valid.sum().to(dtype=torch.float32)
    return (total / count.clamp(min=1.0)).to("cpu")


def _build_transformer(cfg: TransformerRunConfig) -> RootTransformerRefiner:
    return RootTransformerRefiner(
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


def _gain(reference: float, model: float) -> dict[str, float]:
    abs_gain = reference - model
    rel_gain = abs_gain / reference if reference == reference and reference != 0.0 else float("nan")
    return {"gain_abs": abs_gain, "gain_rel": rel_gain}


def _comparison_table(
    *,
    transformer_report: Dict[str, Dict[str, float]],
    other_reports: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    keys = [
        "root_error_mean_m",
        "MPJPE_cam_m",
        "MPJPE_world_m",
        "MPJPE_local_m",
        "reprojection_error_mean_px",
    ]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split, tr_metrics in transformer_report.items():
        split_out: Dict[str, Dict[str, float]] = {}
        for key in keys:
            row = {"transformer": float(tr_metrics.get(key, float("nan")))}
            for name, report in other_reports.items():
                if split in report:
                    row[name] = float(report[split].get(key, float("nan")))
                    gains = _gain(row[name], row["transformer"])
                    row[f"transformer_vs_{name}_gain_abs"] = gains["gain_abs"]
                    row[f"transformer_vs_{name}_gain_rel"] = gains["gain_rel"]
            split_out[key] = row
        out[split] = split_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V1 temporal root Transformer")
    parser.add_argument("--config", type=str, default="configs/transformer/root_transformer_v1.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint name: 'best', 'last', or a path to a .pt file",
    )
    parser.add_argument("--no_baseline", action="store_true", help="Skip mean-root baseline comparison")
    parser.add_argument("--mlp_config", type=str, default="configs/mlp/root_mlp_v1_train.yaml")
    parser.add_argument("--mlp_checkpoint", type=str, default=None, help="Optional MLP checkpoint: best, last, or .pt")
    parser.add_argument("--tcn_config", type=str, default="configs/tcn/root_tcn_v1.yaml")
    parser.add_argument("--tcn_checkpoint", type=str, default=None, help="Optional TCN checkpoint: best, last, or .pt")
    args = parser.parse_args()

    cfg = load_transformer_run_config(Path(args.config))
    device = get_device(cfg.device)

    ckpt_path = _resolve_checkpoint(args.checkpoint, cfg.checkpoints_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ensure_dir(cfg.eval_reports_dir)
    ensure_dir(cfg.predictions_dir)

    cfg_used = cfg.eval_reports_dir / "config_used.yaml"
    if not cfg_used.exists():
        shutil.copyfile(Path(args.config), cfg_used)

    stats = TorchNormalizationStats.load(cfg.normalization_stats_path, device="cpu")

    model = _build_transformer(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    pin_memory = device.type == "cuda"
    evaluator = TemporalEvaluator(
        stats=stats,
        device=device,
        save_predictions_npz=cfg.eval.save_predictions_npz,
        save_predictions_csv=cfg.eval.save_predictions_csv,
        prediction_mode=cfg.prediction_mode,
    )

    report: Dict[str, Dict[str, float]] = {}
    split_loaders: dict[str, torch.utils.data.DataLoader] = {}

    for split in cfg.eval.splits:
        dl = _make_window_loader(
            cfg=cfg,
            split=split,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            pin_memory=pin_memory,
        )
        split_loaders[split] = dl
        out, extras = evaluator.evaluate_split(
            model=model,
            dataloader=dl,
            out_dir=cfg.predictions_dir,
            split_name=split,
            min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
            min_bbox_width_px=cfg.dataset.min_bbox_width_px,
            min_bbox_height_px=cfg.dataset.min_bbox_height_px,
            min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
            root_axis_weights=cfg.loss_weights.root_axis_weights,
        )

        metrics = dict(out.metrics)
        metrics["num_frames_total"] = float(extras.num_frames_total)
        metrics["num_frames_covered"] = float(extras.num_frames_covered)
        metrics["num_frames_uncovered"] = float(extras.num_frames_uncovered)
        report[split] = metrics

    comparison_reports: Dict[str, Dict[str, Dict[str, float]]] = {}

    if not args.no_baseline:
        mean_root_norm = _compute_temporal_mean_root_norm(cfg=cfg, device=device)
        baseline = TemporalMeanRootBaseline(mean_root_norm=mean_root_norm)
        baseline_eval = TemporalEvaluator(stats=stats, device=device, save_predictions_npz=False, save_predictions_csv=False)
        baseline_report: Dict[str, Dict[str, float]] = {}
        for split, dl in split_loaders.items():
            out, _extras = baseline_eval.evaluate_split(
                model=baseline,
                dataloader=dl,
                out_dir=cfg.predictions_dir / "baseline_mean_root",
                split_name=split,
                min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
                min_bbox_width_px=cfg.dataset.min_bbox_width_px,
                min_bbox_height_px=cfg.dataset.min_bbox_height_px,
                min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
                root_axis_weights=cfg.loss_weights.root_axis_weights,
            )
            baseline_report[split] = dict(out.metrics)
        comparison_reports["baseline_mean_root"] = baseline_report

    if args.mlp_checkpoint is not None:
        mlp_cfg = load_run_config(Path(args.mlp_config))
        mlp_model = RootRefiner(
            input_dim=infer_input_dim(mlp_cfg.input_config),
            hidden_dims=mlp_cfg.model.hidden_dims,
            activation=mlp_cfg.model.activation,
            dropout=mlp_cfg.model.dropout,
        )
        mlp_ckpt = _resolve_checkpoint(str(args.mlp_checkpoint), mlp_cfg.checkpoints_dir)
        if not mlp_ckpt.exists():
            raise FileNotFoundError(f"MLP checkpoint not found: {mlp_ckpt}")
        mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location="cpu")["model_state_dict"], strict=True)
        mlp_wrapped = WindowedFramewiseRootModel(mlp_model)
        mlp_report: Dict[str, Dict[str, float]] = {}
        mlp_eval = TemporalEvaluator(
            stats=stats,
            device=device,
            save_predictions_npz=False,
            save_predictions_csv=False,
            prediction_mode=mlp_cfg.prediction_mode,
        )
        for split, dl in split_loaders.items():
            out, _extras = mlp_eval.evaluate_split(
                model=mlp_wrapped,
                dataloader=dl,
                out_dir=cfg.predictions_dir / "compare_mlp",
                split_name=split,
                min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
                min_bbox_width_px=cfg.dataset.min_bbox_width_px,
                min_bbox_height_px=cfg.dataset.min_bbox_height_px,
                min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
                root_axis_weights=cfg.loss_weights.root_axis_weights,
            )
            mlp_report[split] = dict(out.metrics)
        comparison_reports["mlp"] = mlp_report

    if args.tcn_checkpoint is not None:
        tcn_cfg = load_tcn_run_config(Path(args.tcn_config))
        tcn_model = RootTCNRefiner(
            input_dim=infer_input_dim(tcn_cfg.input_config),
            encoder_hidden_dims=tcn_cfg.model.encoder_hidden_dims,
            temporal_hidden_dim=tcn_cfg.model.temporal_hidden_dim,
            temporal_dilations=tcn_cfg.model.temporal_dilations,
            temporal_kernel_size=tcn_cfg.model.temporal_kernel_size,
            activation=tcn_cfg.model.activation,
            dropout=tcn_cfg.model.dropout,
            head_hidden_dims=tcn_cfg.model.head_hidden_dims,
        )
        tcn_ckpt = _resolve_checkpoint(str(args.tcn_checkpoint), tcn_cfg.checkpoints_dir)
        if not tcn_ckpt.exists():
            raise FileNotFoundError(f"TCN checkpoint not found: {tcn_ckpt}")
        tcn_model.load_state_dict(torch.load(tcn_ckpt, map_location="cpu")["model_state_dict"], strict=True)
        tcn_report: Dict[str, Dict[str, float]] = {}
        tcn_eval = TemporalEvaluator(
            stats=stats,
            device=device,
            save_predictions_npz=False,
            save_predictions_csv=False,
            prediction_mode=tcn_cfg.prediction_mode,
        )
        for split, dl in split_loaders.items():
            out, _extras = tcn_eval.evaluate_split(
                model=tcn_model,
                dataloader=dl,
                out_dir=cfg.predictions_dir / "compare_tcn",
                split_name=split,
                min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
                min_bbox_width_px=cfg.dataset.min_bbox_width_px,
                min_bbox_height_px=cfg.dataset.min_bbox_height_px,
                min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
                root_axis_weights=cfg.loss_weights.root_axis_weights,
            )
            tcn_report[split] = dict(out.metrics)
        comparison_reports["tcn"] = tcn_report

    metrics_path = cfg.eval_reports_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "run_name": cfg.run_name,
            "checkpoint": str(ckpt_path),
            "splits": report,
            "comparison_reports": comparison_reports,
            "comparisons": _comparison_table(transformer_report=report, other_reports=comparison_reports),
        },
    )

    if cfg.plots.enabled:
        plots_dir = cfg.eval_reports_dir / "plots"
        ensure_dir(plots_dir)

        plot_training_curves(
            train_log_csv=cfg.eval_reports_dir / "train_log.csv",
            out_path=plots_dir / "training_curves.png",
        )

        split_p = cfg.plots.split_for_plots
        pred_npz = cfg.predictions_dir / f"{split_p}_predictions.npz"
        if pred_npz.exists():
            if cfg.plots.root_error_ground_vs_air_histogram:
                plot_root_error_ground_vs_air_histogram(
                    data_dir=cfg.data_dir,
                    split=split_p,
                    predictions_npz=pred_npz,
                    out_path=plots_dir / f"root_error_ground_vs_air_histogram_{split_p}.png",
                )
            plot_root_timeseries(
                predictions_npz=pred_npz,
                out_path=plots_dir / f"root_xyz_timeseries_{split_p}.png",
                seq_name=cfg.plots.seq_name,
                person_idx=cfg.plots.person_idx,
            )
            plot_world_trajectory_xy(
                predictions_npz=pred_npz,
                out_path=plots_dir / f"world_traj_xy_{split_p}.png",
                seq_name=cfg.plots.seq_name,
                person_idx=cfg.plots.person_idx,
            )
            plot_reprojection_overlay(
                data_dir=cfg.data_dir,
                split=split_p,
                predictions_npz=pred_npz,
                stats=stats,
                out_path=plots_dir / f"reproj_overlay_{split_p}.png",
                seq_name=cfg.plots.seq_name,
                person_idx=cfg.plots.person_idx,
                num_frames=cfg.plots.num_frames_overlay,
                show_sam2d=cfg.plots.show_sam2d_overlay,
                num_players_per_subplot=cfg.plots.num_players_per_subplot,
            )

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {cfg.predictions_dir}")


if __name__ == "__main__":
    main()
