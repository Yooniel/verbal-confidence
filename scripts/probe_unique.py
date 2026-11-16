"""Variance in verbal confidence that activations explain beyond the answer log-probabilities, with a tuned
ridge penalty. Supplement to scripts/probe.py, whose variance partitioning uses the paper's ridge alpha = 1;
that setting over-fits when the confidence varies little, as in the two-turn format.

Two-stage estimate in every CV fold (same folds as probe.py):
  1. ordinary least squares of the confidence on the six answer log-prob summaries      -> R2_base
  2. ridge regression of the stage-1 residual on the z-scored activations, penalty chosen on the training fold
     by leave-one-out error; prediction = stage 1 + stage 2                              -> R2_both
  R2_unique = R2_both - R2_base

Writes results/probe/unique_tuned.json (results/tt/probe/... with VC_FMT=tt). CPU job, about 25 s per task.
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from probe import ALPHAS, LP_KEYS, r2, zs
from vc.acts import load_acts
from vc.data import read_jsonl
from vc.env import DATA, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--positions", default="PANL,CC,PANL+1")
ap.add_argument("--jobs", type=int, default=4)
args = ap.parse_args()

trials = read_jsonl(DATA / "trials.jsonl")
A = load_acts()
KEEP = np.array([trials[i].get("src", "val") == "val" for i in A.idx])
T = [trials[i] for i, k in zip(A.idx, KEEP) if k]
y = np.array([t["conf"] for t in T], dtype=np.float64)
corr = np.array([t["correct"] for t in T], dtype=int)
LP = np.array([[t["lp"][k] for k in LP_KEYS] for t in T], dtype=np.float64)
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(T)), corr))


def task(li, pos):
    X = A.get(li, pos)[KEEP].astype(np.float64)
    base, both, acts, alpha = [], [], [], []
    for tr, te in folds:
        Xtr, Xte = zs(X[tr], X[te])
        Btr, Bte = zs(LP[tr], LP[te])
        Btr1, Bte1 = np.c_[np.ones(len(tr)), Btr], np.c_[np.ones(len(te)), Bte]
        beta = np.linalg.lstsq(Btr1, y[tr], rcond=None)[0]
        res = y[tr] - Btr1 @ beta
        w, Q = np.linalg.eigh(Xtr @ Xtr.T)
        w = np.clip(w, 0.0, None)
        KQ = (Xte @ Xtr.T) @ Q
        Q2 = Q * Q

        def fit(target):
            """Tuned ridge of a centred training target on the activations -> (test prediction, alpha). The
            leave-one-out error is the one of probe.py: the 1/n term is the leverage of the mean, which the
            z-scoring of the training fold removes."""
            Qt = Q.T @ target
            loo = []
            for a in ALPHAS:
                shrink = w / (w + a)
                e = (target - Q @ (shrink * Qt)) / np.clip(1.0 - Q2 @ shrink - 1.0 / len(target), 1e-8, None)
                loo.append(float((e**2).mean()))
            a = ALPHAS[int(np.argmin(loo))]
            return KQ @ (Qt / (w + a)), a

        p_res, a = fit(res)
        p_act, _ = fit(y[tr] - y[tr].mean())
        base.append(r2(y[te], Bte1 @ beta))
        both.append(r2(y[te], Bte1 @ beta + p_res))
        acts.append(r2(y[te], y[tr].mean() + p_act))
        alpha.append(a)
    out = {"pos": pos, "layer": A.layers[li], "r2_base": float(np.mean(base)), "r2_both": float(np.mean(both)),
           "r2_unique": float(np.mean(both) - np.mean(base)), "r2_acts": float(np.mean(acts)), "alpha": alpha}
    print(f"L{out['layer']} {pos}: base {out['r2_base']:.3f} both {out['r2_both']:.3f} unique {out['r2_unique']:.3f} acts alone {out['r2_acts']:.3f}", flush=True)
    return out


tasks = [(li, p) for p in args.positions.split(",") for li in range(len(A.layers))]
res = Parallel(n_jobs=args.jobs, backend="multiprocessing")(delayed(task)(li, p) for li, p in tasks)
json.dump({"n": len(T), "alphas": ALPHAS, "results": res}, open(RESULTS / "probe" / "unique_tuned.json", "w"))
print("PROBE_UNIQUE DONE", len(res))
