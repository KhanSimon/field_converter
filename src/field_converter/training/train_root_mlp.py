from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

from field_converter.models.root_refiner import RootRefiner
from field_converter.training.config import load_run_config
from field_converter.training.dataset import NormalizedFrameDataset, infer_input_dim
from field_converter.training.samplers import SequenceBatchSampler
from field_converter.training.trainer import train
from field_converter.utils.io import ensure_dir, write_json
from field_converter.utils.normalization import TorchNormalizationStats
from field_converter.utils.torch_utils import get_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V1 frame-wise root MLP")
    parser.add_argument("--config", type=str, default="configs/mlp/root_mlp_v1_train.yaml")
    args = parser.parse_args()

    cfg = load_run_config(Path(args.config))

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

    # Output dirs
    ensure_dir(cfg.checkpoints_dir)
    ensure_dir(cfg.eval_reports_dir)
    ensure_dir(cfg.predictions_dir)

    # Keep a copy of the config used.
    cfg_used_path = cfg.eval_reports_dir / "config_used.yaml"
    shutil.copyfile(Path(args.config), cfg_used_path)

    # Load normalization stats.
    stats = TorchNormalizationStats.load(cfg.normalization_stats_path, device="cpu")

    # Datasets
    train_ds = NormalizedFrameDataset(
        data_dir=cfg.data_dir,
        split="train",
        input_config=cfg.input_config,
        prediction_mode=cfg.prediction_mode,
        root_init_dir=cfg.root_init_dir,
        seed=cfg.seed,
        max_sequences=cfg.dataset.max_sequences,
        max_samples_per_sequence=cfg.dataset.max_samples_per_sequence,
        subsample_stride=cfg.dataset.subsample_stride,
        min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
        min_bbox_width_px=cfg.dataset.min_bbox_width_px,
        min_bbox_height_px=cfg.dataset.min_bbox_height_px,
        min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
    )
    valid_ds = NormalizedFrameDataset(
        data_dir=cfg.data_dir,
        split="valid",
        input_config=cfg.input_config,
        prediction_mode=cfg.prediction_mode,
        root_init_dir=cfg.root_init_dir,
        seed=cfg.seed,
        max_sequences=cfg.dataset.max_sequences,
        max_samples_per_sequence=cfg.dataset.max_samples_per_sequence,
        subsample_stride=cfg.dataset.subsample_stride,
        min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
        min_bbox_width_px=cfg.dataset.min_bbox_width_px,
        min_bbox_height_px=cfg.dataset.min_bbox_height_px,
        min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
    )

    pin_memory = device.type == "cuda"

    train_dl_kwargs: dict[str, object] = {}
    if cfg.training.num_workers > 0:
        train_dl_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 1,
            }
        )

    valid_dl_kwargs: dict[str, object] = {}
    if cfg.eval.num_workers > 0:
        valid_dl_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 1,
            }
        )

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
    model = RootRefiner(
        input_dim=input_dim,
        hidden_dims=cfg.model.hidden_dims,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    state = train(
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
        w_cam3d=cfg.loss_weights.cam3d,
        w_proj=cfg.loss_weights.proj,
        prediction_mode=cfg.prediction_mode,
        checkpoints_dir=cfg.checkpoints_dir,
        train_log_csv=cfg.eval_reports_dir / "train_log.csv",
    )

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

    print("Training done")
    print(f"- best_epoch: {state.best_epoch}")
    print(f"- best_root_error_mean_m: {state.best_root_error_mean_m:.6f}")
    print(f"- checkpoints: {cfg.checkpoints_dir}")
    print(f"- reports: {cfg.eval_reports_dir}")


if __name__ == "__main__":
    main()
