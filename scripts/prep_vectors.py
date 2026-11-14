"""From the collected activations: steering directions, residual norms, mean-ablation vectors, and the
'natural variability' statistics used for the OOD analysis (paper Table 1). Run as a CPU job."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.acts import load_acts
from vc.env import DATA, RESULTS

A = load_acts()
sets = json.load(open(DATA / "sets.json"))
layers, pos_names = A.layers, A.pos
print("acts:", A.n, "trials;", len(layers), "layers;", len(pos_names), "positions", flush=True)

STEER_POS = ["PANL", "CC", "PANL+1", "FCC", "AC", "A_first", "A_last"]
NOISE_POS = ["PANL", "CC", "PANL+1", "AC"]
out = {"unit": {}, "norm": {}, "mean": {}, "raw_norm": {}, "layers": layers}
hi, lo, cal = A.rows(sets["vec_high"]), A.rows(sets["vec_low"]), A.rows(sets["calib"])
for p in STEER_POS:
    U, N, R = [], [], []
    for li in range(len(layers)):
        X = A.get(li, p)  # [n, d] float32
        v = X[hi].mean(0) - X[lo].mean(0)
        R.append(float(np.linalg.norm(v)))
        U.append(v / np.linalg.norm(v))
        N.append(float(np.linalg.norm(X, axis=1).mean()))
    out["unit"][p] = torch.tensor(np.stack(U), dtype=torch.float32)
    out["norm"][p] = torch.tensor(N, dtype=torch.float32)
    out["raw_norm"][p] = torch.tensor(R, dtype=torch.float32)
    print(p, "mean resid norm @L25: %.0f | raw diff-of-means norm @L25: %.0f (%.1f%% of resid norm)" % (
        N[layers.index(25)], R[layers.index(25)], 100 * R[layers.index(25)] / N[layers.index(25)]))
for p in NOISE_POS:
    out["mean"][p] = torch.tensor(np.stack([A.get(li, p)[cal].mean(0) for li in range(len(layers))]), dtype=torch.float32)
torch.save(out, RESULTS / "vectors.pt")

# ---- OOD reference statistics (Table 1): natural pairwise variability, and steering-induced shifts
rng = np.random.default_rng(0)
ood = {}
for L in [25]:
    li = layers.index(L)
    for p in ["PANL", "PANL+1", "CC"]:
        X = A.get(li, p).astype(np.float64)
        nrm = np.linalg.norm(X, axis=1)
        i, j = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
        keep = i != j
        i, j = i[keep], j[keep]
        cos = (X[i] * X[j]).sum(1) / (nrm[i] * nrm[j])
        nr = nrm[i] / nrm[j]
        d = {"natural_cos_p5_p95": [float(np.percentile(cos, 5)), float(np.percentile(cos, 95))],
             "natural_nr_p5_p95": [float(np.percentile(nr, 5)), float(np.percentile(nr, 95))]}
        if p in out["unit"]:
            u = out["unit"][p][li].numpy().astype(np.float64)
            for s in (2, 5):
                for name, sign in (("high", 1), ("low", -1)):
                    Y = X + sign * s * 0.03 * float(out["norm"][p][li]) * u
                    ny = np.linalg.norm(Y, axis=1)
                    d[f"steer_{name}_s{s}"] = {"cos": float(((X * Y).sum(1) / (nrm * ny)).mean()), "nr": float((ny / nrm).mean())}
        if p in out["mean"]:
            m = out["mean"][p][li].numpy().astype(np.float64)
            d["noise"] = {"cos": float(((X @ m) / (nrm * np.linalg.norm(m))).mean()), "nr": float((np.linalg.norm(m) / nrm).mean())}
        ood[f"L{L}_{p}"] = d
json.dump(ood, open(RESULTS / "ood_natural.json", "w"), indent=1)
print(json.dumps(ood, indent=1))
print("PREP DONE")
