from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch


DeviceStr = Literal["auto", "cpu", "cuda"]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed python/numpy/torch for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: DeviceStr) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device: {device}")
