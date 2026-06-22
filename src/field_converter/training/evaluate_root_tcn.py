from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

import torch

from field_converter.evaluation.temporal_evaluator import TemporalEvaluator
from field_converter.evaluation.visualization import (
    plot_root_diagnostic_plots,
    plot_reprojection_overlay,
    plot_root_timeseries,
    plot_training_curves,
    plot_world_trajectory_xy,
)
from field_converter.models.root_tcn_refiner import RootTCNRefiner
from field_converter.training.dataset import infer_input_dim
from field_converter.training.tcn.config import load_tcn_run_config
from field_converter.training.tcn.window_dataset import NormalizedWindowDataset
from field_converter.utils.io import ensure_dir, write_json
from field_converter.utils.normalization import TorchNormalizationStats
from field_converter.utils.torch_utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V1 temporal root TCN")
    parser.add_argument("--config", type=str, default="configs/tcn/root_tcn_v1.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint name: 'best', 'last', or a path to a .pt file",
    )
    args = parser.parse_args()

    cfg = load_tcn_run_config(Path(args.config))
    device = get_device(cfg.device)

    if args.checkpoint in {"best", "last"}:
        ckpt_path = cfg.checkpoints_dir / f"{args.checkpoint}.pt"
    else:
        ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ensure_dir(cfg.eval_reports_dir)
    ensure_dir(cfg.predictions_dir)

    # Keep a copy of the config used for this evaluation.
    cfg_used = cfg.eval_reports_dir / "config_used.yaml"
    if not cfg_used.exists():
        shutil.copyfile(Path(args.config), cfg_used)

    stats = TorchNormalizationStats.load(cfg.normalization_stats_path, device="cpu")

    input_dim = infer_input_dim(cfg.input_config)
    model = RootTCNRefiner(
        input_dim=input_dim,
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        temporal_hidden_dim=cfg.model.temporal_hidden_dim,
        temporal_dilations=cfg.model.temporal_dilations,
        temporal_kernel_size=cfg.model.temporal_kernel_size,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        head_hidden_dims=cfg.model.head_hidden_dims,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    pin_memory = device.type == "cuda"

    dl_kwargs: dict[str, object] = {}
    if cfg.eval.num_workers > 0:
        dl_kwargs.update({"persistent_workers": True, "prefetch_factor": 1})

    evaluator = TemporalEvaluator(
        stats=stats,
        device=device,
        save_predictions_npz=cfg.eval.save_predictions_npz,
        save_predictions_csv=cfg.eval.save_predictions_csv,
    )

    report: Dict[str, Dict[str, float]] = {}

    for split in cfg.eval.splits:
        ds = NormalizedWindowDataset(
            data_dir=cfg.data_dir,
            split=split,
            input_config=cfg.input_config,
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
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.eval.num_workers,
            pin_memory=pin_memory,
            **dl_kwargs,
        )

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

        # Add a few coverage fields into the split report.
        metrics = dict(out.metrics)
        metrics["num_frames_total"] = float(extras.num_frames_total)
        metrics["num_frames_covered"] = float(extras.num_frames_covered)
        metrics["num_frames_uncovered"] = float(extras.num_frames_uncovered)
        report[split] = metrics

    metrics_path = cfg.eval_reports_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "run_name": cfg.run_name,
            "checkpoint": str(ckpt_path),
            "splits": report,
        },
    )

    # Plots
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

    if cfg.diagnostic_plots.enabled:
        split_p = cfg.diagnostic_plots.split_for_plots
        pred_npz = cfg.predictions_dir / f"{split_p}_predictions.npz"
        if pred_npz.exists():
            diagnostic_dir = cfg.eval_reports_dir / "plots" / "diagnostics"
            ensure_dir(diagnostic_dir)
            plot_root_diagnostic_plots(
                data_dir=cfg.data_dir,
                split=split_p,
                predictions_npz=pred_npz,
                out_dir=diagnostic_dir,
                root_error_vs_camera_distance=cfg.diagnostic_plots.root_error_vs_camera_distance,
                root_error_vs_image_center_distance=cfg.diagnostic_plots.root_error_vs_image_center_distance,
                root_error_vs_player_speed=cfg.diagnostic_plots.root_error_vs_player_speed,
                speed_window=cfg.diagnostic_plots.speed_window,
            )

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {cfg.predictions_dir}")


if __name__ == "__main__":
    main()
