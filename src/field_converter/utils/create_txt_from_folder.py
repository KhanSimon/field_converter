from pathlib import Path
from field_converter import pathseeker as ps


folder = ps.DATA_DIR / "boxes_gt"
output_file = ps.DATA_DIR / "sequences_gt.txt"

with output_file.open("w", encoding="utf-8") as f:
    for file in sorted(folder.iterdir(), key=lambda p: p.stem.lower()):
        if file.is_file():
            f.write(file.stem + "\n")