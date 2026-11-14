"""Numerical-noise test: are identical rows in one batch computed identically? How much do logits move
between batch shapes (bf16)?"""
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import read_jsonl
from vc.env import DATA
from vc.model import VC, log
from vc.prompts import class_token_ids, encode

trials = read_jsonl(DATA / "trials.jsonl")
sets = json.load(open(DATA / "sets.json"))
vc = VC(); tok = vc.tok; cls_ids = class_token_ids(tok)
idx = sets["noise"][:8] + sets["noise"][200:208]
encs = sorted([encode(tok, trials[i]["question"], trials[i]["answer"], "cat") for i in idx], key=lambda e: e.n)
for B in (4, 8):
    E = encs[:B]
    ids, last = vc.pad([e.ids for e in E])
    clean, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=[0, 25, 40], sel=cls_ids)
    for l in (0, 25, 40):
        for k in (2, 8, 16 // (B // 4)):
            H = cache[l].repeat(k, 1, 1)
            lg, _, _ = vc.run(H, start=l + 1, last_idx=last.repeat(k), sel=cls_ids)
            lg = lg.view(k, B, -1)
            within = float((lg - lg[0:1]).abs().max())
            vs_clean = float((lg[0] - clean.to(lg.device)).abs().max())
            flips = int((lg[0].argmax(-1) != clean.to(lg.device).argmax(-1)).sum())
            log(f"B={B} layer={l} copies={k} rows={k*B}: max|copy_i - copy_0|={within:.6f} | max|resumed - full clean pass|={vs_clean:.4f} | argmax flips vs clean: {flips}/{B}")
    # same rows, different chunk size
    H2 = cache[25].repeat(2, 1, 1); H8 = cache[25].repeat(8, 1, 1)
    a, _, _ = vc.run(H2, start=26, last_idx=last.repeat(2), sel=cls_ids)
    b, _, _ = vc.run(H8, start=26, last_idx=last.repeat(8), sel=cls_ids)
    log(f"B={B}: rows={2*B} vs rows={8*B} (same inputs): max|diff|={float((a[:B]-b[:B]).abs().max()):.4f}")
    # shuffled row order inside a batch
    perm = torch.randperm(B)
    c, _, _ = vc.run(cache[25][perm.to(cache[25].device)], start=26, last_idx=last[perm], sel=cls_ids)
    d, _, _ = vc.run(cache[25], start=26, last_idx=last, sel=cls_ids)
    log(f"B={B}: permuted rows vs original order: max|diff|={float((c - d[perm.to(d.device)]).abs().max()):.6f}")
log("NUMTEST DONE")
