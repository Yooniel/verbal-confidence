"""Pilot 2: which post-answer separator ("\\n", "\\n\\n", "\\n\\n\\n") reproduces the paper's behavioural
statistics (Fig. 9: 69% 'Almost certain', ECE 0.12, AUROC 0.71; minimal prompt: digit first token,
ECE 0.17, AUROC 0.68)?  Also reports max |activation| per layer (for float16 storage)."""
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.metrics import roc_auc_score

from vc.data import SWEEP_LAYERS, alias_match, load_triviaqa
from vc.model import VC, batched, log
from vc.prompts import CLASSES, MIDPOINTS, class_token_ids, digit_token_ids, encode, phase0_prompt

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
vc = VC()
tok = vc.tok
cls_ids, dig_ids = class_token_ids(tok), digit_token_ids(tok)
rows = load_triviaqa()[:N]


def stop_fn(txt):
    if "**Confidence**:" not in txt:
        return txt.count("\n") >= 3
    tail = txt.split("**Confidence**:", 1)[1]
    return any(c in tail for c in CLASSES) or ("\n" in tail.strip(" ") and len(tail.strip()) > 0)


order = sorted(range(len(rows)), key=lambda i: len(rows[i]["question"]))
for b in batched(order, 32):
    gens = vc.generate([phase0_prompt(rows[i]["question"]) for i in b], max_new_tokens=40, stop_fn=stop_fn)
    for i, (ids, lps) in zip(b, gens):
        txt = tok.decode(ids)
        rows[i]["answer"] = txt.split("\n")[0].strip()
        rows[i]["em"] = alias_match(rows[i]["answer"], rows[i]["norm_aliases"])
        conf = None
        if "**Confidence**:" in txt:
            tail = txt.split("**Confidence**:", 1)[1].strip()
            for ci in sorted(range(10), key=lambda c: -len(CLASSES[c])):
                if tail.startswith(CLASSES[ci]):
                    conf = ci
                    break
        rows[i]["p0"] = conf
rows = [r for r in rows if r["answer"]]
em = np.array([r["em"] for r in rows])
log(f"phase0 done; n={len(rows)} EM acc={em.mean():.3f}")
p0 = np.array([r["p0"] if r["p0"] is not None else -1 for r in rows])
log("phase0 dist:", {CLASSES[c]: int((p0 == c).sum()) for c in range(10) if (p0 == c).any()},
    "AUROC(EM)=%.3f" % roc_auc_score(em[p0 >= 0], p0[p0 >= 0]))


def ece(conf, correct):
    return float(sum((conf == c).mean() * abs(correct[conf == c].mean() - c) for c in np.unique(conf)))


def run_kind(kind, panl, tids):
    encs = [encode(tok, r["question"], r["answer"], kind, panl=panl) for r in rows]
    o = sorted(range(len(rows)), key=lambda i: encs[i].n)
    top = np.zeros(len(rows), dtype=np.int64)
    lg_c = np.zeros((len(rows), 10), dtype=np.float32)
    for b in batched(o, 48):
        ids, last = vc.pad([encs[i].ids for i in b])
        lg, _, _ = vc.run(vc.embed(ids), last_idx=last)
        top[b] = lg.argmax(-1).cpu().numpy()
        lg_c[b] = lg[:, tids].cpu().numpy()
    return top, lg_c


for panl in ["\n", "\n\n", "\n\n\n"]:
    top, lg = run_kind("cat", panl, cls_ids)
    cls = lg.argmax(-1)
    conf = np.round(np.array(MIDPOINTS)[cls], 2)
    srt = np.sort(lg, -1)
    log(f"[cat panl={panl!r}] first token valid: {np.isin(top, cls_ids).mean():.3f} | dist: "
        + ", ".join(f"{CLASSES[c]}={100*(cls==c).mean():.1f}%" for c in range(10) if (cls == c).any())
        + f" | ECE={ece(conf, em):.3f} AUROC={roc_auc_score(em, conf):.3f} mean conf={conf.mean():.3f}"
        + f" | agree w/ phase0: {(cls == p0).mean():.2f}; r={np.corrcoef(cls[p0>=0], p0[p0>=0])[0,1]:.2f}"
        + f" | top1-top2 margin median={np.median(srt[:,-1]-srt[:,-2]):.2f}, <0.5: {(srt[:,-1]-srt[:,-2] < 0.5).mean():.2f}")
    log("     acc by class: " + ", ".join(f"{CLASSES[c]}:{em[cls==c].mean():.2f}(n={int((cls==c).sum())})" for c in range(10) if (cls == c).any()))
    top, lg = run_kind("min", panl, dig_ids)
    dig = lg.argmax(-1)
    log(f"[min panl={panl!r}] first token: {Counter(tok.decode(int(t)) for t in top).most_common(6)} | digit dist: "
        + ", ".join(f"{d}={100*(dig==d).mean():.1f}%" for d in range(10) if (dig == d).any())
        + f" | ECE={ece(np.round((dig+0.5)/10, 2), em):.3f} AUROC={roc_auc_score(em, dig):.3f}")

# magnitude of activations per layer (decides the float16 storage scale)
encs = [encode(tok, r["question"], r["answer"], "cat") for r in rows[:32]]
ids, last = vc.pad([e.ids for e in encs])
_, cache, _ = vc.run(vc.embed(ids), capture=range(vc.L), capture_fn=lambda l, h: (float(h.float().abs().max()), float(h.float().norm(dim=-1).mean())))
log("max |component| / mean norm by layer: " + " ".join(f"L{l}:{cache[l][0]:.0f}/{cache[l][1]:.0f}" for l in SWEEP_LAYERS))
log("PILOT2 DONE")
