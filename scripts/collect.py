"""Activation collection (one shard): residual stream at the analysed positions for the 3,000-trial
collection set, at the paper's sampled layers. Saved as float16 [n, n_layers, n_positions, d]."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import SWEEP_LAYERS, read_jsonl
from vc.env import DATA, RESULTS
from vc.model import VC, batched, log
from vc.prompts import class_token_ids, encode

POS_NAMES = ["PANL", "PANL+1", "CC", "FCC", "AC", "A_first", "A_last", "QTT"] + [f"INT{i}" for i in range(10)]

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", default=str(RESULTS / "acts"))
args = ap.parse_args()

trials = [r for r in read_jsonl(DATA / "trials.jsonl") if r["split"] == "collect"]
trials = trials[args.shard :: args.nshards]
vc = VC()
tok = vc.tok
cls_ids = class_token_ids(tok)
encs = [encode(tok, r["question"], r["answer"], "cat") for r in trials]
P = np.array([[e.pos[k] for k in POS_NAMES[:8]] + e.inter for e in encs])  # [n, 18]
acts = np.zeros((len(trials), len(SWEEP_LAYERS), len(POS_NAMES), vc.d), dtype=np.float32)
logits = np.zeros((len(trials), 10), dtype=np.float32)
order = sorted(range(len(trials)), key=lambda i: encs[i].n)
t0 = time.time()
for bi, b in enumerate(batched(order, args.bs)):
    ids, last = vc.pad([encs[i].ids for i in b])
    pos = torch.tensor(P[b])  # [B, 18]
    ar = torch.arange(len(b))[:, None]

    def cap(l, h):
        return h[ar.to(h.device), pos.to(h.device)].float().cpu()  # [B, 18, d]

    lg, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=SWEEP_LAYERS, capture_fn=cap, sel=cls_ids)
    for li, l in enumerate(SWEEP_LAYERS):
        acts[b, li] = cache[l].numpy()
    logits[b] = lg.cpu().numpy()
    if bi % 10 == 0:
        log(f"batch {bi}: {time.time()-t0:.0f}s")
assert np.isfinite(acts).all()
os.makedirs(args.out, exist_ok=True)
# float16 storage with an exact power-of-two scale per layer (late-layer residuals exceed the float16 range)
scales = []
out16 = np.zeros(acts.shape, dtype=np.float16)
for li in range(len(SWEEP_LAYERS)):
    mx = float(np.abs(acts[:, li]).max())
    s = 2.0 ** max(0, int(np.ceil(np.log2(mx / 16000.0))))
    scales.append(s)
    out16[:, li] = (acts[:, li] / s).astype(np.float16)
assert np.isfinite(out16).all()
np.save(os.path.join(args.out, f"acts{args.shard}.npy"), out16)
json.dump(
    {"idx": [r["idx"] for r in trials], "layers": SWEEP_LAYERS, "pos": POS_NAMES, "logits": logits.tolist(),
     "scales": scales},
    open(os.path.join(args.out, f"meta{args.shard}.json"), "w"),
)
# consistency with the behavioural run
agree = np.mean([int(np.argmax(logits[i])) == trials[i]["cls"] for i in range(len(trials))])
log(f"COLLECT DONE {acts.shape}; class agreement with behavioural run: {agree:.4f}")
