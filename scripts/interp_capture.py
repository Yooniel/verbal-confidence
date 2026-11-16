"""Activations for the interpretability tools (NLA verbalizer, sparse autoencoder, Jacobian lens).
One GPU job with the base model; with VC_FMT=tt the same on the two-turn chat format.

Trials: the steering test set (single prompt: 100 high- and 100 low-confidence trials; two-turn: 60 + 60).
Writes results[/tt]/interp/capture.pt:

  clean      [n, 62, P, d]   residual stream at the output of every block, at the named positions
  logits     [n, 10]         class logits of the unedited prompt
  steer      [n, V, 2, Q, d] residual stream at blocks 31 and 41 at positions Q after steering variant V
                             (high / low confidence direction at scales 2 and 5, and a random direction of the
                             scale-5 size), applied at block STEER_LAYER on PANL (two-turn: on the five tokens
                             between answer and request)
  steer_logits [n, V, 10]
  full       {31, 41: [m, T, d]}  all positions of FULL_N high + FULL_N low trials (right-padded), with `full_idx`,
                             `full_len` and `full_ids`
  ref        [22, d]         block-41 residuals of the reference prompt of the NLA repository (pipeline check)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import SWEEP_LAYERS, read_jsonl
from vc.env import DATA, FMT, NLA_LAYER, RESULTS, SAE_LAYER
from vc.model import VC, batched, log
from vc.prompts import HIGH_CLASSES, TT_POST5, class_token_ids, encode_trial

TT = FMT == "tt"
POS = ["A_last", "PANL", "PANL+1", "INT4", "CC"] + (["UNL", "REQ0"] if TT else [])
STEER_LAYER = 25 if TT else 24
STEER_AT = TT_POST5 if TT else ["PANL"]
STEER_Q = ["PANL", "PANL+1", "CC"]
VARIANTS = [("high", 2), ("high", 5), ("low", 2), ("low", 5), ("rand", 5)]
READ_LAYERS = [SAE_LAYER, NLA_LAYER]
FULL_N = 12
REF_PROMPT = "What is the capital of France and what is it known for?"

trials = read_jsonl(DATA / "trials.jsonl")
sets = json.load(open(DATA / "sets.json"))
idx = list(sets["steer"])
if os.environ.get("VC_LIMIT"):
    idx = idx[: int(os.environ["VC_LIMIT"])]
out_dir = RESULTS / "interp"
os.makedirs(out_dir, exist_ok=True)

vc = VC()
tok = vc.tok
cls_ids = class_token_ids(tok)
encs = [encode_trial(tok, trials[i]) for i in idx]


def where(e, name):
    return e.pos[name] if name in e.pos else e.inter[int(name[3:])]


V = torch.load(RESULTS / "vectors.pt")
li = SWEEP_LAYERS.index(STEER_LAYER)
g = torch.Generator().manual_seed(0)
rand_unit = torch.nn.functional.normalize(torch.randn(vc.d, generator=g), dim=0)


def delta(kind, scale):
    """[len(STEER_AT), d]: scale x 3% of the mean residual norm along the steering (or a random) direction."""
    rows = []
    for q in STEER_AT:
        u = rand_unit if kind == "rand" else V["unit"][q][li] * (1.0 if kind == "high" else -1.0)
        rows.append(scale * 0.03 * V["norm"][q][li] * u)
    return torch.stack(rows)


n = len(idx)
clean = torch.zeros(n, vc.L, len(POS), vc.d)
logits = torch.zeros(n, 10)
steer = torch.zeros(n, len(VARIANTS), len(READ_LAYERS), len(STEER_Q), vc.d)
steer_logits = torch.zeros(n, len(VARIANTS), 10)
order = sorted(range(n), key=lambda j: encs[j].n)
t0 = time.time()
for bi, b in enumerate(batched(order, 8)):
    E = [encs[j] for j in b]
    B = len(b)
    ids, last = vc.pad([e.ids for e in E])
    P = torch.tensor([[where(e, p) for p in POS] for e in E])
    Q = torch.tensor([[where(e, p) for p in STEER_Q] for e in E])
    S = torch.tensor([[where(e, p) for p in STEER_AT] for e in E])
    ar = torch.arange(B)[:, None]
    keep = {}

    def cap(l, h):
        if l == STEER_LAYER:
            keep["base"] = h
        return h[ar.to(h.device), P.to(h.device)].float().cpu()

    lg, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=range(vc.L), capture_fn=cap, sel=cls_ids)
    logits[b] = lg.cpu()
    for l in range(vc.L):
        clean[b, l] = cache[l]
    base = keep["base"]
    H = []
    for kind, scale in VARIANTS:
        h = base.clone()
        h[ar.to(h.device), S.to(h.device)] += delta(kind, scale).to(h.device, h.dtype)
        H.append(h)
    H = torch.cat(H, 0)
    arv = torch.arange(H.shape[0])[:, None]
    Qv = Q.repeat(len(VARIANTS), 1)
    lg, cache, _ = vc.run(H, start=STEER_LAYER + 1, last_idx=last.repeat(len(VARIANTS)), capture=READ_LAYERS,
                          capture_fn=lambda l, h: h[arv.to(h.device), Qv.to(h.device)].float().cpu(), sel=cls_ids)
    steer_logits[b] = lg.cpu().view(len(VARIANTS), B, 10).permute(1, 0, 2)
    for ri, l in enumerate(READ_LAYERS):
        steer[b, :, ri] = cache[l].view(len(VARIANTS), B, len(STEER_Q), vc.d).permute(1, 0, 2, 3)
    del H, base, cache
    log(f"batch {bi}: {time.time() - t0:.0f}s")

# ---- all positions for a few trials
hi = [j for j in range(n) if trials[idx[j]]["cls"] in HIGH_CLASSES][:FULL_N]
lo = [j for j in range(n) if trials[idx[j]]["cls"] not in HIGH_CLASSES][:FULL_N]
fj = hi + lo
T = max(encs[j].n for j in fj)
full = {l: torch.zeros(len(fj), T, vc.d) for l in READ_LAYERS}
for b in batched(list(range(len(fj))), 8):
    ids, last = vc.pad([encs[fj[k]].ids for k in b])
    _, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=READ_LAYERS, capture_fn=lambda l, h: h.float().cpu())
    for l in READ_LAYERS:
        full[l][b, : ids.shape[1]] = cache[l]

# ---- reference prompt of the NLA repository
enc = tok.apply_chat_template([{"role": "user", "content": REF_PROMPT}], tokenize=True, add_generation_prompt=True)
ref_ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
ids, last = vc.pad([ref_ids])
_, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=[NLA_LAYER], capture_fn=lambda l, h: h[0].float().cpu())
ref = cache[NLA_LAYER]
log("reference prompt:", len(ref_ids), "tokens; block-41 norms of tokens 4-7:", [round(float(x), 1) for x in ref[4:8].norm(dim=-1)])

agree = sum(int(logits[j].argmax()) == trials[idx[j]]["cls"] for j in range(n)) / n
torch.save(
    {"idx": idx, "pos": POS, "clean": clean, "logits": logits, "steer": steer, "steer_logits": steer_logits,
     "variants": VARIANTS, "steer_layer": STEER_LAYER, "steer_at": STEER_AT, "steer_q": STEER_Q, "read_layers": READ_LAYERS,
     "full": full, "full_idx": [idx[j] for j in fj], "full_len": [encs[j].n for j in fj],
     "full_ids": [encs[j].ids for j in fj], "full_pos": [{p: where(encs[j], p) for p in POS + ["AC"]} for j in fj],
     "ref": ref, "ref_ids": ref_ids},
    out_dir / "capture.pt",
)
log(f"INTERP CAPTURE DONE {tuple(clean.shape)}; class agreement with the behavioural run {agree:.3f}")
