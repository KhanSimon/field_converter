import numpy as np
from field_converter import pathseeker as ps


path = "ARG_FRA_182345_root_pred.npy"
data_npy = np.load(path)

print(f"Data shape: {data_npy.shape}")

print(data_npy)
"""

path = ps.OUTPUTS_DIR / "predictions" / "root_tcn_random_search_v2_trial_001" / "test_predictions.npz"



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
        
            
        print(arr)
        
"""

