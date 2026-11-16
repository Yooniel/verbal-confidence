"""Numerical test: how many confidence tokens change between attention implementations and batch shapes?

The paper ran its attention experiments with the "eager" attention implementation (C.1.1). If its clean
reference came from a differently computed pass, the reported token-change rates include a floor of purely
numerical flips. This measures that floor on the 500 trials of the categorical attention-blocking experiment:
stock Hugging Face forward with sdpa vs eager attention, batched vs one prompt at a time, and against this
repository's own layer loop.
"""
import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import read_jsonl
from vc.env import DATA, RESULTS
from vc.model import VC, batched, log
from vc.prompts import class_token_ids, encode

trials = read_jsonl(DATA / "trials.jsonl")
idx = json.load(open(DATA / "sets.json"))["attn_cat"]
N1 = 150  # trials also run one at a time


def passes(attn):
    vc = VC(attn_implementation=attn)
    cls = class_token_ids(vc.tok)
    encs = [encode(vc.tok, trials[i]["question"], trials[i]["answer"], "cat") for i in idx]
    order = sorted(range(len(idx)), key=lambda j: encs[j].n)
    out = {"hf_batched": np.zeros((len(idx), 10), np.float32), "hf_single": np.zeros((N1, 10), np.float32)}
    if attn == "sdpa":
        out["loop_batched"] = np.zeros((len(idx), 10), np.float32)
    for b in batched(order, 16):
        ids, last = vc.pad([encs[j].ids for j in b])
        out["hf_batched"][b] = vc.hf_logits(ids, last)[:, cls].cpu().numpy()
        if attn == "sdpa":
            lg, _, _ = vc.run(vc.embed(ids), last_idx=last, sel=cls)
            out["loop_batched"][b] = lg.cpu().numpy()
    for j in range(N1):
        ids, last = vc.pad([encs[j].ids])
        out["hf_single"][j] = vc.hf_logits(ids, last)[:, cls].cpu().numpy()
    log(f"{attn}: done")
    del vc
    gc.collect()
    torch.cuda.empty_cache()
    return out


S, E = passes("sdpa"), passes("eager")


def cmp(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return {"n": n, "token_change_pct": float(100 * (a.argmax(1) != b.argmax(1)).mean()), "mean_abs_logit_diff": float(np.abs(a - b).mean()),
            "max_abs_logit_diff": float(np.abs(a - b).max())}


res = {
    "eager_vs_sdpa_batched": cmp(E["hf_batched"], S["hf_batched"]),
    "eager_vs_sdpa_single": cmp(E["hf_single"], S["hf_single"]),
    "sdpa_single_vs_batched": cmp(S["hf_single"], S["hf_batched"]),
    "eager_single_vs_batched": cmp(E["hf_single"], E["hf_batched"]),
    "eager_single_vs_sdpa_batched": cmp(E["hf_single"], S["hf_batched"]),
    "own_loop_vs_hf_sdpa_batched": cmp(S["loop_batched"], S["hf_batched"]),
    "eager_batched_vs_own_loop": cmp(E["hf_batched"], S["loop_batched"]),
}
srt = np.sort(S["hf_batched"], 1)
res["pct_trials_top2_within_1_logit"] = float(100 * ((srt[:, -1] - srt[:, -2]) < 1).mean())
for k, v in res.items():
    log(k, json.dumps(v))
json.dump(res, open(RESULTS / "numtest_eager.json", "w"), indent=1)
log("NUMTEST_EAGER DONE")
