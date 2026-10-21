"""Epoch loader over a flat fp16 embedding store (``<field>_<split>.f16``).

Two modes:
  * ``in_memory=True``: the whole split is loaded into RAM and every epoch is a full random
    permutation (needs ~2.5 KB of RAM per row; 123M train rows of full TCGA = 315 GB).
  * ``in_memory=False`` (default): rows stay on disk. Each epoch visits fixed-size blocks in random
    order, loads ``buffer_blocks`` of them at a time and shuffles rows inside that buffer.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator

import numpy as np
import torch


class EmbeddingStore:
    def __init__(self, store_dir: pathlib.Path, field: str, split: str, max_rows: int | None = None,
                 in_memory: bool = False, block_rows: int = 65_536, buffer_blocks: int = 16):
        meta = json.loads((store_dir / "metadata.json").read_text())["fields"][field]
        self.d = int(meta["d"])
        n = int(meta[f"n_{split}"])
        self.mm = np.memmap(store_dir / f"{field}_{split}.f16", mode="r", dtype=np.float16, shape=(n, self.d))
        if max_rows is not None:
            self.mm = self.mm[:max_rows]
        self.n = self.mm.shape[0]
        self.x = torch.from_numpy(np.array(self.mm)) if in_memory else None
        self.block_rows, self.buffer_blocks = block_rows, buffer_blocks
        self.mean = torch.tensor(meta["train_mean"], dtype=torch.float32)
        self.std = torch.tensor(meta["train_std"], dtype=torch.float32)

    def rows(self, idx: np.ndarray) -> torch.Tensor:
        """Rows at (sorted) indices ``idx`` as fp32."""
        if self.x is not None:
            return self.x[torch.from_numpy(idx)].float()
        return torch.from_numpy(np.asarray(self.mm[idx])).float()

    def batches(self, batch_size: int, seed: int) -> Iterator[torch.Tensor]:
        """One epoch of shuffled fp32 batches (the last incomplete batch is dropped)."""
        g = torch.Generator().manual_seed(seed)
        if self.x is not None:
            perm = torch.randperm(self.n, generator=g)
            for s in range(0, self.n - batch_size + 1, batch_size):
                yield self.x[perm[s : s + batch_size]].float()
            return
        n_blocks = -(-self.n // self.block_rows)
        order = torch.randperm(n_blocks, generator=g).tolist()
        carry = torch.empty((0, self.d), dtype=torch.float16)
        for i in range(0, n_blocks, self.buffer_blocks):
            parts = [np.asarray(self.mm[b * self.block_rows : (b + 1) * self.block_rows]) for b in order[i : i + self.buffer_blocks]]
            buf = torch.from_numpy(np.concatenate(parts))
            buf = torch.cat([carry, buf[torch.randperm(len(buf), generator=g)]])
            n_full = len(buf) // batch_size * batch_size
            for s in range(0, n_full, batch_size):
                yield buf[s : s + batch_size].float()
            carry = buf[n_full:]

    def steps_per_epoch(self, batch_size: int) -> int:
        return self.n // batch_size

    def random_rows(self, n: int, seed: int = 0) -> torch.Tensor:
        """``n`` rows drawn uniformly without replacement (fp32)."""
        rng = np.random.default_rng(seed)
        return self.rows(np.sort(rng.choice(self.n, size=min(n, self.n), replace=False)))

    def sequential(self, batch_size: int) -> Iterator[torch.Tensor]:
        for s in range(0, self.n, batch_size):
            src = self.x[s : s + batch_size] if self.x is not None else torch.from_numpy(np.array(self.mm[s : s + batch_size]))
            yield src.float()
