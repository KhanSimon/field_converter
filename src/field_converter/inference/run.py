"""Command-line entry point for qualitative root inference without GT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from field_converter import pathseeker as ps
from field_converter.inference.modeling import load_inference_model
from field_converter.inference.pipeline import (
    predict_prepared_sequence,
    save_inference_result,
)
from field_converter.inference.preprocessing import prepare_sequence, select_sequences


def _positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0.0:
        raise argparse.ArgumentTypeError(f"FPS must be a finite value > 0, got {value!r}")
    return fps


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MLP/TCN/Transformer root inference on data without GT. The exact training "
            "config is required because checkpoints do not contain their architecture or input schema."
        )
    )
    parser.add_argument("--model-type", required=True, choices=("mlp", "tcn", "transformer"))
    parser.add_argument("--config", required=True, type=Path, help="Exact YAML used to train the checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="best, last, or an explicit .pt checkpoint path",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ps.DATA_DIR / "data_inference",
        help="Raw inference root containing boxes/, cameras/, skel_2d/, skel_3d_relative/",
    )
    parser.add_argument(
        "--pitch-points",
        type=Path,
        default=ps.DATA_DIR / "pitch_points.txt",
        help="Canonical training/FIFA pitch_points.txt used by the aligned world frame",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ps.OUTPUTS_DIR / "predictions" / "inference",
        help="Final output root; results go under <output>/<run>/<sequence>/",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Sequence stem to process; repeat the option for several sequences (default: all complete sequences)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Image size; default infers W=2*cx and H=2*cy from K",
    )
    parser.add_argument(
        "--k-to-zero",
        action="store_true",
        help="Force k1=k2=0 during preprocessing, even if the inference camera contains distortion",
    )
    parser.add_argument(
        "--sam3d-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Global SAM3D orientation sign; use -1 only when the new exporter uses the opposite camera convention",
    )
    parser.add_argument(
        "--world-alignment",
        choices=("auto", "none", "rotate_x_180", "rotate_y_180", "rotate_z_180"),
        default="auto",
        help=(
            "Align the inference world axes to the training camera distribution. "
            "auto detects an opposite-side/sign convention; explicit rotations are overrides."
        ),
    )
    parser.add_argument(
        "--box-normalization-min-size-px",
        type=float,
        default=10.0,
        help="Minimum box size used for SAM2D box normalization (training pipeline default: 10)",
    )
    parser.add_argument(
        "--source-fps",
        type=_positive_fps,
        default=None,
        help=(
            "Current FPS of the input arrays. Providing it enables temporal resampling; "
            "omit the option to keep the original timeline unchanged."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=_positive_fps,
        default=50.0,
        help="Target FPS when --source-fps is provided (default: 50)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override config eval.batch_size")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override the device from the model config",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Do not save data/data_inference/features*, ground_intersection and root_init intermediates",
    )
    parser.add_argument("--no-csv", action="store_true", help="Only save NPZ and JSON outputs")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    runtime = load_inference_model(
        model_type=args.model_type,  # type: ignore[arg-type]
        config_path=args.config,
        checkpoint=args.checkpoint,
        device_override=args.device,
        batch_size_override=args.batch_size,
    )
    sequences = select_sequences(args.input_dir, args.sequence)
    explicit_size = tuple(args.image_size) if args.image_size is not None else None

    print("=== Qualitative root inference ===")
    print(f"model_type: {runtime.model_type}")
    print(f"run_name:   {runtime.config.run_name}")
    print(f"config:     {runtime.config_path}")
    print(f"checkpoint: {runtime.checkpoint_path}")
    print(f"device:     {runtime.device}")
    print(f"sequences:  {len(sequences)}")
    if args.source_fps is None:
        print("resampling: disabled (--source-fps not provided)")
    else:
        print(f"resampling: {args.source_fps:g} FPS -> {args.target_fps:g} FPS")

    for sequence in sequences:
        print(f"\n[preprocess] {sequence}")
        prepared = prepare_sequence(
            input_dir=args.input_dir,
            sequence=sequence,
            stats_path=runtime.config.normalization_stats_path,
            pitch_points_path=args.pitch_points,
            image_size=explicit_size,  # type: ignore[arg-type]
            k_to_zero=bool(args.k_to_zero),
            sam3d_sign=int(args.sam3d_sign),
            world_alignment_mode=args.world_alignment,
            box_normalization_min_size_px=float(args.box_normalization_min_size_px),
            source_fps=args.source_fps,
            target_fps=float(args.target_fps),
            save_intermediate=not bool(args.no_save_intermediate),
        )
        if args.source_fps is not None:
            source_frames = int(np.asarray(prepared.raw["source_num_frames"]).item())
            target_frames = int(np.asarray(prepared.raw["frame_numbers"]).shape[0])
            mode = "interpolation" if args.source_fps < args.target_fps else "sampling"
            if np.isclose(args.source_fps, args.target_fps):
                mode = "identity"
            print(
                f"[resample]  {mode}: {source_frames} frames @ {args.source_fps:g} FPS "
                f"-> {target_frames} frames @ {args.target_fps:g} FPS"
            )
        alignment = prepared.world_alignment
        source_center = np.asarray(alignment.camera_center_source_median)
        aligned_center = np.asarray(alignment.camera_center_aligned_median)
        print(
            f"[alignment]  {alignment.selected_transform} "
            f"(applied={alignment.applied}, "
            f"C_source={np.round(source_center, 2).tolist()}, "
            f"C_aligned={np.round(aligned_center, 2).tolist()})"
        )
        print(f"[predict]    {sequence}")
        prediction = predict_prepared_sequence(prepared, runtime)
        result = save_inference_result(
            prepared=prepared,
            prediction=prediction,
            runtime=runtime,
            output_root=args.output_dir,
            save_csv=not bool(args.no_csv),
        )
        predicted = int(result.summary["num_predicted_positions"])
        root_init_valid = int(result.summary["num_root_init_valid_positions"])
        print(f"[saved]      {result.predictions_path}")
        print(f"              predicted={predicted}, root_init_valid={root_init_valid}")
        if bool(result.summary["camera_center_out_of_distribution_warning"]):
            zscore = float(result.summary["camera_center_max_abs_train_zscore"])
            print(
                "[WARN] Camera-center world features remain outside the training distribution "
                f"after alignment (max |z-score|={zscore:.2f}). The physical camera placement "
                "may differ from the training setup."
            )

    print(f"\nDone. Inference outputs: {args.output_dir / runtime.config.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
