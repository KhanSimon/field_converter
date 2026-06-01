from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from field_converter.evaluation.evaluator import Evaluator
from field_converter.evaluation.visualization import (
    plot_reprojection_overlay,
    plot_root_timeseries,
    plot_training_curves,
    plot_world_trajectory_xy,
)
from field_converter.models.root_refiner import RootRefiner
from field_converter.training.config import load_run_config
from field_converter.training.dataset import NormalizedFrameDataset, infer_input_dim
from field_converter.utils.io import ensure_dir, write_json
from field_converter.utils.normalization import TorchNormalizationStats
from field_converter.utils.torch_utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V1 frame-wise root MLP")
    parser.add_argument("--config", type=str, default="configs/mlp/root_mlp_v1_train.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint name: 'best', 'last', or a path to a .pt file",
    )
    args = parser.parse_args()

    cfg = load_run_config(Path(args.config))
    device = get_device(cfg.device)

    # Resolve checkpoint path.
    if args.checkpoint in {"best", "last"}:
        ckpt_path = cfg.checkpoints_dir / f"{args.checkpoint}.pt"
    else:
        ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ensure_dir(cfg.eval_reports_dir)
    ensure_dir(cfg.predictions_dir)

    stats = TorchNormalizationStats.load(cfg.normalization_stats_path, device="cpu")

    input_dim = infer_input_dim(cfg.input_config)
    model = RootRefiner(
        input_dim=input_dim,
        hidden_dims=cfg.model.hidden_dims,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    pin_memory = device.type == "cuda"

    dl_kwargs: dict[str, object] = {}
    if cfg.eval.num_workers > 0:
        dl_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 1,
            }
        )

    evaluator = Evaluator(
        stats=stats,
        device=device,
        save_predictions_npz=cfg.eval.save_predictions_npz,
        save_predictions_csv=cfg.eval.save_predictions_csv,
    )

    report: Dict[str, Dict[str, float]] = {}

    for split in cfg.eval.splits:
        ds = NormalizedFrameDataset(
            data_dir=cfg.data_dir,
            split=split,
            input_config=cfg.input_config,
            seed=cfg.seed,
            max_sequences=cfg.dataset.max_sequences,
            max_samples_per_sequence=cfg.dataset.max_samples_per_sequence,
            subsample_stride=cfg.dataset.subsample_stride,
            min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
            min_bbox_width_px=cfg.dataset.min_bbox_width_px,
            min_bbox_height_px=cfg.dataset.min_bbox_height_px,
            min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.eval.num_workers,
            pin_memory=pin_memory,
            **dl_kwargs,
        )

        out = evaluator.evaluate_split(
            model=model,
            dataloader=dl,
            out_dir=cfg.predictions_dir,
            split_name=split,
        )
        report[split] = out.metrics

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

        # Training curves (if training log exists)
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

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {cfg.predictions_dir}")


if __name__ == "__main__":
    main()
