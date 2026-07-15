from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

from field_converter.evaluation.temporal_evaluator import TemporalEvaluator
from field_converter.models.root_tcn_refiner import RootTCNRefiner
from field_converter.training.dataset import infer_input_dim
from field_converter.training.samplers import SequenceBatchSampler
from field_converter.training.tcn.config import load_tcn_run_config
from field_converter.training.tcn.trainer import train_tcn
from field_converter.training.tcn.window_dataset import NormalizedWindowDataset
from field_converter.utils.io import ensure_dir, write_json
from field_converter.utils.normalization import TorchNormalizationStats
from field_converter.utils.torch_utils import get_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V1 temporal root TCN")
    parser.add_argument("--config", type=str, default="configs/tcn/root_tcn_v1.yaml")
    args = parser.parse_args()

    cfg = load_tcn_run_config(Path(args.config))

    seed_everything(cfg.seed, deterministic=False)
    device = get_device(cfg.device)

    # Helpful diagnostics when running on a GPU partition.
    slurm_gpus = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (slurm_gpus or cuda_visible) and not torch.cuda.is_available():
        print(
            "[WARN] GPU seems allocated (SLURM/CUDA env set) but torch.cuda.is_available() is False. "
            "This usually means a CUDA driver / PyTorch CUDA build mismatch, so training will run on CPU."
        )

    ensure_dir(cfg.checkpoints_dir)
    ensure_dir(cfg.eval_reports_dir)
    ensure_dir(cfg.predictions_dir)

    # Keep a copy of the config used.
    shutil.copyfile(Path(args.config), cfg.eval_reports_dir / "config_used.yaml")

    stats = TorchNormalizationStats.load(cfg.normalization_stats_path, device="cpu")

    # ---- Datasets
    train_ds = NormalizedWindowDataset(
        data_dir=cfg.data_dir,
        split="train",
        input_config=cfg.input_config,
        prediction_mode=cfg.prediction_mode,
        root_init_dir=cfg.root_init_dir,
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

    # For validation we want full coverage for overlap aggregation.
    valid_ds = NormalizedWindowDataset(
        data_dir=cfg.data_dir,
        split="valid",
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

    pin_memory = device.type == "cuda"

    train_dl_kwargs: dict[str, object] = {}
    if cfg.training.num_workers > 0:
        train_dl_kwargs.update({"persistent_workers": True, "prefetch_factor": 1})

    valid_dl_kwargs: dict[str, object] = {}
    if cfg.eval.num_workers > 0:
        valid_dl_kwargs.update({"persistent_workers": True, "prefetch_factor": 1})

    if cfg.training.group_batches_by_sequence:
        train_batch_sampler = SequenceBatchSampler(
            seq_ids=train_ds.index[:, 0],
            batch_size=cfg.training.batch_size,
            seed=cfg.seed,
            drop_last=False,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=pin_memory,
            **train_dl_kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=pin_memory,
            **train_dl_kwargs,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=pin_memory,
        **valid_dl_kwargs,
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    state = train_tcn(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        stats=stats,
        epochs=cfg.training.epochs,
        grad_clip_norm=cfg.training.grad_clip_norm,
        early_stopping_patience=cfg.training.early_stopping_patience,
        w_root=cfg.loss_weights.root,
        root_axis_weights=cfg.loss_weights.root_axis_weights,
        w_root_vel=cfg.loss_weights.root_vel,
        w_root_acc=cfg.loss_weights.root_acc,
        w_cam3d=cfg.loss_weights.cam3d,
        w_proj=cfg.loss_weights.proj,
        prediction_mode=cfg.prediction_mode,
        min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
        min_bbox_width_px=cfg.dataset.min_bbox_width_px,
        min_bbox_height_px=cfg.dataset.min_bbox_height_px,
        min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
        checkpoints_dir=cfg.checkpoints_dir,
        train_log_csv=cfg.eval_reports_dir / "train_log.csv",
    )

    # Save a short summary.
    write_json(
        cfg.eval_reports_dir / "train_summary.json",
        {
            "run_name": cfg.run_name,
            "best_epoch": state.best_epoch,
            "best_root_error_mean_m": state.best_root_error_mean_m,
            "last_epoch": state.epoch,
            "device": str(device),
        },
    )

    # Optional: print final validation metrics summary (best checkpoint).
    if (cfg.checkpoints_dir / "best.pt").exists():
        ckpt = torch.load(cfg.checkpoints_dir / "best.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        evalr = TemporalEvaluator(
            stats=stats,
            device=device,
            save_predictions_npz=False,
            save_predictions_csv=False,
            prediction_mode=cfg.prediction_mode,
        )
        (out, extras) = evalr.evaluate_split(
            model=model,
            dataloader=valid_loader,
            out_dir=cfg.predictions_dir,
            split_name="valid",
            min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
            min_bbox_width_px=cfg.dataset.min_bbox_width_px,
            min_bbox_height_px=cfg.dataset.min_bbox_height_px,
            min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
            root_axis_weights=cfg.loss_weights.root_axis_weights,
        )
        print("Validation (aggregated) — best checkpoint")
        print(f"- root_error_mean_m: {out.metrics.get('root_error_mean_m', float('nan')):.6f}")
        print(f"- covered frames: {extras.num_frames_covered}/{extras.num_frames_total}")

    print("Training done")
    print(f"- best_epoch: {state.best_epoch}")
    print(f"- best_root_error_mean_m: {state.best_root_error_mean_m:.6f}")
    print(f"- checkpoints: {cfg.checkpoints_dir}")
    print(f"- reports: {cfg.eval_reports_dir}")


if __name__ == "__main__":
    main()
