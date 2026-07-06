import numpy as np
from field_converter import pathseeker as ps

#PYTHONPATH=src python -m field_converter.utils.test

path = ps.DATA_DIR / "boxes_gt" / "ARG_FRA_182345.npy"
all_boxes = np.load(path)

print(f"Data shape: {all_boxes.shape}")

for boxes in all_boxes:
    for player_boxe in boxes:
        
            if player_boxe[2]-player_boxe[0] < 10:
                print(player_boxe[2]-player_boxe[0])
