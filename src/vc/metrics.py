"""Loading of intervention results and the metrics of the paper (C.1.5), shared by the analysis scripts.

confidence = midpoint of the argmax class; logit difference = logit of the reference class minus the mean
logit of the other nine classes; token change rate = fraction of trials whose argmax class differs from the
reference run.
"""
from __future__ import annotations

from glob import glob

import numpy as np

from .plotting import sem
from .prompts import MIDPOINTS

MID = np.array(MIDPOINTS)


def have(results, exp) -> bool:
    return len(glob(str(results / exp / "shard*.pt"))) > 0


def load(results, exp) -> dict:
    """Concatenate the shards of one experiment (results/<exp>/shard*.pt)."""
    import torch

    parts = [torch.load(f, weights_only=False) for f in sorted(glob(str(results / exp / "shard*.pt")))]
    out = dict(parts[0])
    for k in ("logits", "ref", "clean", "corrupt", "ood"):
        if k in out:
            out[k] = torch.cat([p[k] for p in parts], 0).numpy()
    for k in ("idx", "recs"):
        if k in out:
            out[k] = [x for p in parts for x in p[k]]
    return out


def ldiff(z, y):
    """z [..., 10], y [...] int -> z[y] - mean(z[others])"""
    zy = np.take_along_axis(z, y[..., None], -1)[..., 0]
    return zy - (z.sum(-1) - zy) / 9.0


def mets(logits, base):
    """logits [n, V, L, 10]; base = reference logits, either [n, 10] (clean run) or [n, V, L, 10] (the unedited
    rows computed in the same forward pass as each variant). Returns dict of [n, V, L] arrays."""
    if base.ndim == 2:
        base = np.broadcast_to(base[:, None, None], logits.shape)
    y = base.argmax(-1)
    conf = MID[logits.argmax(-1)]
    ld = ldiff(logits, y)
    return {
        "conf": conf,
        "dconf": conf - MID[y],
        "ld": ld,
        "dld": ld - ldiff(base, y),
        "change": (logits.argmax(-1) != y).astype(float),
    }


def noise_floor(D):
    """Share of trials whose argmax class differs between the unedited rows of the resumed passes and the
    full clean pass (pure numerical noise from the different batch shapes), per layer."""
    return 100 * (D["ref"].argmax(-1) != D["clean"].argmax(-1)[:, None, None]).mean((0, 1))


def ms(a):
    """mean and SEM over trials (axis 0)."""
    return a.mean(0), sem(a, 0)


def peak(layers, mean, mode="max"):
    i = int(np.argmax(mean) if mode == "max" else np.argmin(mean))
    return {"layer": int(layers[i]), "value": float(mean[i])}
