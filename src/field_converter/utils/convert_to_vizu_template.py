#!/usr/bin/env python3
import argparse
import numpy as np


def convert_npz_to_npy(
    input_npz: str,
    output_npy: str,
    sequence_name: str = "ARG_FRA_182345",
    key: str = "root_pred_m",
    fill_value: float = np.nan,
):
    data = np.load(input_npz, allow_pickle=True)

    seq_names = data["seq_names"]
    seq_id = data["seq_id"]
    person_idx = data["person_idx"]
    frame_idx = data["frame_idx"]
    root_pred = data[key]  # shape: (N, 3)

    matches = np.where(seq_names == sequence_name)[0]
    if len(matches) == 0:
        raise ValueError(f"Sequence introuvable: {sequence_name}")

    target_seq_id = matches[0]

    mask = seq_id == target_seq_id

    persons = person_idx[mask]
    frames = frame_idx[mask]
    values = root_pred[mask]

    unique_persons = np.unique(persons)
    unique_frames = np.unique(frames)

    person_to_out = {p: i for i, p in enumerate(unique_persons)}
    frame_to_out = {f: i for i, f in enumerate(unique_frames)}

    output = np.full(
        (len(unique_persons), len(unique_frames), 3),
        fill_value,
        dtype=values.dtype,
    )

    for p, f, xyz in zip(persons, frames, values):
        output[person_to_out[p], frame_to_out[f]] = xyz

    np.save(output_npy, output)

    print(f"Sequence: {sequence_name}")
    print(f"Input key: {key}")
    print(f"Persons: {len(unique_persons)}")
    print(f"Frames: {len(unique_frames)}")
    print(f"Output shape: {output.shape}")
    print(f"Saved to: {output_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz")
    parser.add_argument("output_npy")
    parser.add_argument("--sequence", default="ARG_FRA_182345")
    parser.add_argument("--key", default="root_pred_m")

    args = parser.parse_args()

    convert_npz_to_npy(
        input_npz=args.input_npz,
        output_npy=args.output_npy,
        sequence_name=args.sequence,
        key=args.key,
    )

#PYTHONPATH=src python -m field_converter.utils.convert_to_vizu_template outputs/predictions/root_tcn_grid_search_rs1_trial_002/test_predictions.npz ARG_FRA_182345_root_pred.npy