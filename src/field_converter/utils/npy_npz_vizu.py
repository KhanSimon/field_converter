import numpy as np
from field_converter import pathseeker as ps

# PYTHONPATH=src python -m field_converter.utils.npy_npz_vizu

"""
path = "ARG_FRA_182345_root_pred.npy"
data_npy = np.load(path)

print(f"Data shape: {data_npy.shape}")

print(data_npy)
"""

path = ps.OUTPUTS_DIR / "predictions" / "root_transformer_v1_delta_new_root_init" / "test_predictions.npz"
path = ps.OUTPUTS_DIR / "predictions" / "inference" / "root_transformer_v1_delta_new_root_init_wo_vj_gi_pp_k" / "ekstraklasa_001999" / "predictions.npz"
path = ps.DATA_DIR / "features" / "ARG_CRO_221101.npz"

#data_npz = np.load(ps.DATA_DIR / "features" / "ARG_CRO_221101.npz", allow_pickle=True)
data_npz = np.load(path, allow_pickle=True)
print(f"Loaded keys: {list(data_npz.keys())}")

# Si le fichier contient un seul tableau stocké directement (np.save) :
if isinstance(data_npz, np.ndarray):
    print(f"Data shape: {data_npz.shape}")
    #print(data_npz)
else:
    # Si le .npz contient plusieurs tableaux (clé → array)
    for k in data_npz.files:
        arr = data_npz[k]
        print(f"\nKey: {k} — shape: {getattr(arr, 'shape', 'scalar')}")
        
            
        #print(arr)
        


