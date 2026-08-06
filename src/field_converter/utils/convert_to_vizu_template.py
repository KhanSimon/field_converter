#!/usr/bin/env python3
import argparse
import numpy as np


def convert_npz_to_npy(
    input_npz: str,
    output_npy: str,
    sequence_name: str = "ARG_FRA_182345",
    key: str = "root_pred_m",
    fill_value: float = np.nan,
    num_persons: int | None = None,
    num_frames: int | None = None,
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

    if persons.size == 0:
        raise ValueError(f"Aucune prédiction disponible pour la séquence: {sequence_name}")
    if np.any(persons < 0) or np.any(frames < 0):
        raise ValueError("person_idx et frame_idx doivent être positifs ou nuls")

    unique_persons = np.unique(persons)
    unique_frames = np.unique(frames)

    # Keep the original person/frame indices. Predictions contain only frames
    # retained by evaluation filters, so compacting unique indices silently
    # shifts every person after a missing track (and likewise for frames).
    inferred_num_persons = int(persons.max()) + 1
    inferred_num_frames = int(frames.max()) + 1
    output_num_persons = inferred_num_persons if num_persons is None else int(num_persons)
    output_num_frames = inferred_num_frames if num_frames is None else int(num_frames)

    if output_num_persons < inferred_num_persons:
        raise ValueError(
            f"num_persons={output_num_persons} est trop petit pour person_idx max={int(persons.max())}"
        )
    if output_num_frames < inferred_num_frames:
        raise ValueError(
            f"num_frames={output_num_frames} est trop petit pour frame_idx max={int(frames.max())}"
        )

    output = np.full(
        (output_num_persons, output_num_frames, 3),
        fill_value,
        dtype=values.dtype,
    )

    output[persons.astype(np.int64), frames.astype(np.int64)] = values

    np.save(output_npy, output)

    print(f"Sequence: {sequence_name}")
    print(f"Input key: {key}")
    print(f"Observed persons: {len(unique_persons)}")
    print(f"Observed frames: {len(unique_frames)}")
    print(f"Dense persons: {output_num_persons}")
    print(f"Dense frames: {output_num_frames}")
    print(f"Output shape: {output.shape}")
    print(f"Saved to: {output_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz")
    parser.add_argument("output_npy")
    parser.add_argument("--sequence", default="ARG_FRA_182345")
    parser.add_argument("--key", default="root_pred_m")
    parser.add_argument(
        "--num-persons",
        type=int,
        default=None,
        help="Nombre total de joueurs attendu; par défaut person_idx.max() + 1.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Nombre total de frames attendu; par défaut frame_idx.max() + 1.",
    )

    args = parser.parse_args()

    convert_npz_to_npy(
        input_npz=args.input_npz,
        output_npy=args.output_npy,
        sequence_name=args.sequence,
        key=args.key,
        num_persons=args.num_persons,
        num_frames=args.num_frames,
    )

# PYTHONPATH=src python -m field_converter.utils.convert_to_vizu_template outputs/predictions/root_transformer_v1_delta_new_root_init/valid_predictions.npz ARG_FRA_203048.npy --sequence ARG_FRA_203048
