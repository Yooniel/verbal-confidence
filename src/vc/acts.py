"""Memory-mapped access to the collected activations (results/acts/acts<k>.npy + meta<k>.json)."""
from __future__ import annotations

import json
import os
from glob import glob

import numpy as np

from .env import RESULTS


class Acts:
    def __init__(self, shards, metas):
        self.shards, self.metas = shards, metas
        self.layers = metas[0]["layers"]
        self.pos = metas[0]["pos"]
        self.idx = [i for m in metas for i in m["idx"]]  # trial idx (into data/trials.jsonl) of every row
        self.n = len(self.idx)
        self._row = {t: r for r, t in enumerate(self.idx)}
        self.logits = np.concatenate([np.array(m["logits"], dtype=np.float32) for m in metas])

    def rows(self, trial_idx):
        return np.array([self._row[t] for t in trial_idx])

    def get(self, li: int, pos: str) -> np.ndarray:
        """All trials' residual stream at sweep-layer index ``li`` and named position -> float32 [n, d]."""
        pi = self.pos.index(pos)
        return np.concatenate(
            [np.asarray(s[:, li, pi], dtype=np.float32) * np.float32(m["scales"][li]) for s, m in zip(self.shards, self.metas)]
        )


def load_acts(d=None, in_memory: bool = True) -> Acts:
    """in_memory=True reads each shard sequentially into RAM (12.8 GB as float16); slicing a memory-map
    position-by-position means thousands of scattered small reads, which is very slow on the network disk."""
    d = d or (RESULTS / "acts")
    files = sorted(glob(str(d / "acts*.npy")))
    shards = [np.load(f, mmap_mode=None if in_memory else "r") for f in files]
    metas = [json.load(open(os.path.join(os.path.dirname(f), os.path.basename(f).replace("acts", "meta").replace(".npy", ".json")))) for f in files]
    return Acts(shards, metas)
