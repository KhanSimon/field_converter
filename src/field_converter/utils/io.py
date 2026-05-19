from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any], *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
