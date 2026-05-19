from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


@dataclass
class SequenceBatchSampler(Sampler[List[int]]):
    """BatchSampler that groups samples by sequence id.

    Motivation
    ----------
    The dataset stores per-sequence payloads in .npz files. With `shuffle=True`,
    random access tends to thrash the per-worker cache, causing repeated `np.load`
    + decompression and very slow iterations on network filesystems.

    This sampler:
    - shuffles the order of sequences each epoch
    - shuffles indices within each sequence
    - yields batches made of indices from a single sequence

    This keeps access mostly sequential within the same .npz payload.
    """

    seq_ids: Sequence[int] | np.ndarray
    batch_size: int
    seed: int = 0
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        seq_ids_np = np.asarray(self.seq_ids, dtype=np.int64).reshape(-1)
        if seq_ids_np.size == 0:
            self._groups: list[np.ndarray] = []
            self._n_batches: int = 0
            return

        # Group dataset indices by seq_id.
        order = np.argsort(seq_ids_np, kind="stable")
        seq_sorted = seq_ids_np[order]
        boundaries = np.flatnonzero(np.diff(seq_sorted)) + 1
        splits = np.split(order, boundaries)
        self._groups = [s.astype(np.int64, copy=False) for s in splits if s.size > 0]

        n = 0
        for g in self._groups:
            if self.drop_last:
                n += int(g.size // self.batch_size)
            else:
                n += int((g.size + self.batch_size - 1) // self.batch_size)
        self._n_batches = int(n)

        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return int(self._n_batches)

    def __iter__(self) -> Iterator[List[int]]:
        if not self._groups:
            return iter(())

        rng = np.random.default_rng(int(self.seed) ^ (self._epoch * 1000003))

        # Shuffle sequence order.
        group_order = np.arange(len(self._groups), dtype=np.int64)
        rng.shuffle(group_order)

        # Build batches.
        batches: list[list[int]] = []
        for gi in group_order:
            g = self._groups[int(gi)].copy()
            rng.shuffle(g)

            if self.drop_last:
                end = (g.size // self.batch_size) * self.batch_size
                g = g[:end]

            for start in range(0, g.size, self.batch_size):
                chunk = g[start : start + self.batch_size]
                if chunk.size == 0:
                    continue
                batches.append(chunk.tolist())

        # Shuffle batch order for some mixing while keeping per-batch locality.
        rng.shuffle(batches)

        return iter(batches)
