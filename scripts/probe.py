"""Linear decoding + variance partitioning on the 3,000-trial collection set (paper 2.6, C.1.8, C.1.9, D.3).

For every (layer, position), with 5-fold cross-validation and features z-scored on the training fold:
  * correctness  : L2-regularised logistic regression (C=1)            -> AUROC
  * confidence   : ridge regression on the verbal-confidence midpoint   -> R^2
                   - alpha = 1.0 (the paper's setting for Figs. 6 and 14)
                   - alpha selected on the training fold by leave-one-out error (cf. Fig. 13d, "best alpha")
  * variance partitioning vs. answer log-prob baselines (ridge alpha=1 on activations + baselines):
        R2_unique = max(0, R2_both - R2_base)   for each single baseline and all six combined

Everything is solved in the dual (n=2400 < d=5376) from ONE eigendecomposition of the training Gram matrix
per fold: ridge for every alpha, its leave-one-out error, the "activations + baselines" fits (Woodbury
update) and the logistic regression (run in the eigenbasis, which is an exact reparametrisation).

Each task writes results/probe/parts/L<layer>_<pos>.json, so the script is resumable;
`--merge-only` just assembles results/probe/probe.json from the parts.
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # parallelism is over (layer, position) tasks

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ALPHAS = [1.0, 10.0, 100.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
LP_KEYS = ["mean", "min", "max", "var", "first", "last"]
POS_ORDER = ["PANL", "CC", "PANL+1", "FCC", "QTT", "AC"] + [f"INT{i}" for i in range(10)] + ["A_first", "A_last"]


def zs(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def r2(y, p):
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fold_fit(Xtr, Xte, ytr, yte, ctr, cte, Btr, Bte, alphas=ALPHAS, lr_iter=500):
    """One CV fold. X: z-scored activations; y: confidence; c: correctness (0/1); B: z-scored baselines."""
    n = len(ytr)
    Ktr, Kte = Xtr @ Xtr.T, Xte @ Xtr.T
    w, Q = np.linalg.eigh(Ktr)
    w = np.clip(w, 0.0, None)
    mu = ytr.mean()
    Qy = Q.T @ (ytr - mu)
    KQ = Kte @ Q
    out = {}
    # ---- ridge for every alpha + leave-one-out error on the training fold
    r2s, loo = [], []
    Q2 = Q * Q
    for a in alphas:
        r2s.append(r2(yte, KQ @ (Qy / (w + a)) + mu))
        shrink = w / (w + a)
        fit = Q @ (shrink * Qy)
        hii = Q2 @ shrink
        e = (ytr - mu - fit) / np.clip(1.0 - hii - 1.0 / n, 1e-8, None)
        loo.append(float((e**2).mean()))
    out["r2_alpha"] = r2s
    out["best"] = int(np.argmin(loo))
    # ---- activations + log-prob baselines, alpha = 1 (Woodbury on the same eigendecomposition)
    a1 = Q @ (Qy / (w + 1.0))
    MiB = Q @ ((Q.T @ Btr) / (w + 1.0)[:, None])
    both = {}
    for name, cols in [(k, [i]) for i, k in enumerate(LP_KEYS)] + [("all", list(range(len(LP_KEYS))))]:
        Bc, Mc = Btr[:, cols], MiB[:, cols]
        S = np.eye(len(cols)) + Bc.T @ Mc
        a = a1 - Mc @ np.linalg.solve(S, Bc.T @ a1)
        both[name] = r2(yte, Kte @ a + Bte[:, cols] @ (Bc.T @ a) + mu)
    out["both"] = both
    # ---- logistic regression in the eigenbasis (Z Z^T = K, so the L2-penalised fit is identical)
    keep = w > 1e-9 * w.max()
    sq = np.sqrt(w[keep])
    Z, Zte = Q[:, keep] * sq, KQ[:, keep] / sq
    clf = LogisticRegression(C=1.0, max_iter=lr_iter)
    clf.fit(Z, ctr)
    out["auroc"] = float(roc_auc_score(cte, clf.decision_function(Zte)))
    out["lr_iter"] = int(clf.n_iter_[0])
    return out


def base_cv(B, y, folds, alpha=1.0):
    out = []
    for tr, te in folds:
        Btr, Bte = zs(B[tr], B[te])
        mu = y[tr].mean()
        w = np.linalg.solve(Btr.T @ Btr + alpha * np.eye(Btr.shape[1]), Btr.T @ (y[tr] - mu))
        out.append(r2(y[te], Bte @ w + mu))
    return float(np.mean(out))


if __name__ == "__main__":
    from joblib import Parallel, delayed

    from vc.acts import load_acts
    from vc.data import read_jsonl
    from vc.env import DATA, RESULTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--positions", default="")
    args = ap.parse_args()
    parts = RESULTS / "probe" / "parts"
    os.makedirs(parts, exist_ok=True)

    trials = read_jsonl(DATA / "trials.jsonl")
    A = load_acts(in_memory=not args.merge_only)
    T = [trials[i] for i in A.idx]
    y_conf = np.array([t["conf"] for t in T], dtype=np.float64)
    y_corr = np.array([t["correct"] for t in T], dtype=int)
    LP = np.array([[t["lp"][k] for k in LP_KEYS] for t in T], dtype=np.float64)
    n = len(T)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(n), y_corr))
    summary = {
        "n": n,
        "r2_base": {**{k: base_cv(LP[:, [i]], y_conf, folds) for i, k in enumerate(LP_KEYS)}, "all": base_cv(LP, y_conf, folds)},
        "auroc_verbal_conf": float(roc_auc_score(y_corr, y_conf)),
        "auroc_mean_logprob": float(roc_auc_score(y_corr, LP[:, 0])),
        "r_logprob_conf": float(np.corrcoef(LP[:, 0], y_conf)[0, 1]),
        "alphas": ALPHAS,
    }
    print(json.dumps(summary, indent=1), flush=True)

    def task(li, pos):
        f = parts / f"L{A.layers[li]}_{pos}.json"
        if f.exists():
            return
        t0 = time.time()
        X = A.get(li, pos).astype(np.float64)
        F = []
        for tr, te in folds:
            Xtr, Xte = zs(X[tr], X[te])
            Btr, Bte = zs(LP[tr], LP[te])
            F.append(fold_fit(Xtr, Xte, y_conf[tr], y_conf[te], y_corr[tr], y_corr[te], Btr, Bte))
        r2a = np.array([f_["r2_alpha"] for f_ in F])
        res = {
            "layer": A.layers[li],
            "pos": pos,
            "r2_alpha": r2a.mean(0).tolist(),
            "r2_a1": float(r2a[:, 0].mean()),
            "r2_tuned": float(np.mean([r2a[k, f_["best"]] for k, f_ in enumerate(F)])),
            "alpha_tuned": [ALPHAS[f_["best"]] for f_ in F],
            "r2_both": {k: float(np.mean([f_["both"][k] for f_ in F])) for k in F[0]["both"]},
            "auroc": float(np.mean([f_["auroc"] for f_ in F])),
            "lr_iter": [f_["lr_iter"] for f_ in F],
            "seconds": round(time.time() - t0, 1),
        }
        json.dump(res, open(str(f) + ".tmp", "w"))
        os.replace(str(f) + ".tmp", f)
        print(f"done L{res['layer']} {pos}: auroc={res['auroc']:.3f} r2(a=1)={res['r2_a1']:.3f} r2(tuned)={res['r2_tuned']:.3f} "
              f"[{res['seconds']}s, lr iters {res['lr_iter']}]", flush=True)

    if not args.merge_only:
        order = [p for p in POS_ORDER if p in A.pos]
        if args.positions:
            order = [p for p in args.positions.split(",")]
        tasks = [(li, p) for p in order for li in range(len(A.layers))]
        t0 = time.time()
        Parallel(n_jobs=args.jobs, backend="multiprocessing")(delayed(task)(li, p) for li, p in tasks)
        print(f"all tasks done in {time.time()-t0:.0f}s", flush=True)
    summary["results"] = [json.load(open(parts / f)) for f in sorted(os.listdir(parts)) if f.endswith(".json")]
    json.dump(summary, open(RESULTS / "probe" / "probe.json", "w"))
    print(f"PROBE MERGED: {len(summary['results'])} (layer, position) results", flush=True)
