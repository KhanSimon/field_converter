import numpy as np
from field_converter import pathseeker as ps

# === CONFIG ===
path = ps.OUTPUTS / "predictions" / "root_tcn_v1_train" / "test_predictions.npz"
threshold = 50.0  # mètres

data = np.load(path, allow_pickle=True)

seq_names = data["seq_names"]      # (num_sequences,)
seq_id = data["seq_id"]            # (num_rows,)
person = data["person_idx"]
frame = data["frame_idx"]

root_pred = data["root_pred_m"]
root_gt = data["root_gt_m"]
error = data["root_error_m"]

mask_nonfinite = (
    ~np.isfinite(root_pred).all(axis=1)
    | ~np.isfinite(root_gt).all(axis=1)
    | ~np.isfinite(error)
)

mask_outliers = error > threshold

idx_sorted = np.argsort(error)[::-1]

print("\n=== TOP OUTLIERS ===")
for i in idx_sorted[:50]:
    sid = int(seq_id[i])
    sname = str(seq_names[sid])
    print(
        f"{sname} | seq_id {sid} | person {person[i]} | frame {frame[i]} | "
        f"error={error[i]:.2f} m | "
        f"pred={root_pred[i]} | gt={root_gt[i]}"
    )

print(f"\n=== OUTLIERS > {threshold} m ===")
idx = np.where(mask_outliers)[0]
print("num outliers:", len(idx))

for i in idx[:50]:
    sid = int(seq_id[i])
    sname = str(seq_names[sid])
    print(
        f"{sname} | seq_id {sid} | person {person[i]} | frame {frame[i]} | "
        f"error={error[i]:.2f} m"
    )

print("\n=== NON FINITE ===")
idx = np.where(mask_nonfinite)[0]
print("num nonfinite:", len(idx))

for i in idx[:50]:
    sid = int(seq_id[i])
    sname = str(seq_names[sid])
    print(
        f"{sname} | seq_id {sid} | person {person[i]} | frame {frame[i]} | "
        f"error={error[i]}"
    )

finite_error = error[np.isfinite(error)]

print("\n=== STATS ===")
print("max:", np.max(finite_error))
print("mean:", np.mean(finite_error))
print("median:", np.median(finite_error))
print("p90:", np.percentile(finite_error, 90))
print("num outliers:", np.sum(mask_outliers))
print("num nonfinite:", np.sum(mask_nonfinite))