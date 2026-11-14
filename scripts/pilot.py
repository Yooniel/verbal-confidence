"""Pilot: validate the engine and the prompt format on a small sample, and measure throughput.

1. my layer loop == stock HF forward (logits at last token)
2. resume-from-layer == full pass with an in-loop edit (fast path == reference path)
3. attention-edge blocking via boolean mask: no-op mask == unmasked
4. behaviour: Phase-0 generations, Phase-1 class distribution, validity of the first confidence token
5. throughput of full passes at several batch sizes
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import SWEEP_LAYERS, alias_match, load_triviaqa
from vc.model import VC, batched, log
from vc.prompts import CLASSES, class_token_ids, digit_token_ids, encode, phase0_prompt

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
vc = VC()
tok = vc.tok
cls_ids = class_token_ids(tok)
dig_ids = digit_token_ids(tok)
rows = load_triviaqa()[:N]
log(f"{len(rows)} questions; GPU mem after load: " + ", ".join(f"{torch.cuda.memory_allocated(i)/2**30:.1f}G" for i in range(torch.cuda.device_count())))


# ---------------- Phase 0 ----------------
def stop_fn(txt):
    if "**Confidence**:" not in txt:
        return txt.count("\n") >= 3
    tail = txt.split("**Confidence**:", 1)[1]
    return any(c in tail for c in CLASSES) or "\n" in tail.strip(" ") and len(tail.strip()) > 0


t0 = time.time()
gens = []
for b in batched(rows, 32):
    gens += vc.generate([phase0_prompt(r["question"]) for r in b], max_new_tokens=40, stop_fn=stop_fn)
log(f"phase0: {len(rows)} generations in {time.time()-t0:.0f}s ({len(rows)/(time.time()-t0):.2f}/s)")
n_ok, n_em = 0, 0
for r, (ids, lps) in zip(rows, gens):
    txt = tok.decode(ids)
    r["p0_text"] = txt
    ans = txt.split("\n")[0].strip()
    r["answer"] = ans
    conf = None
    if "**Confidence**:" in txt:
        tail = txt.split("**Confidence**:", 1)[1].strip()
        for c in sorted(CLASSES, key=len, reverse=True):
            if tail.startswith(c):
                conf = c
                break
    r["p0_conf"] = conf
    r["em"] = alias_match(ans, r["norm_aliases"])
    n_ok += conf is not None and len(ans) > 0
    n_em += r["em"]
log(f"phase0 valid format: {n_ok}/{len(rows)}; alias-EM accuracy: {n_em/len(rows):.3f}")
for r in rows[:12]:
    log("  Q:", r["question"][:70], "| GEN:", repr(r["p0_text"][:90]), "| gold:", r["gold"], "| em:", r["em"])
from collections import Counter

log("phase0 conf dist:", Counter(r["p0_conf"] for r in rows).most_common())

# ---------------- Phase 1 (categorical) ----------------
valid = [r for r in rows if r["answer"]]
encs = [encode(tok, r["question"], r["answer"], "cat") for r in valid]
order = sorted(range(len(valid)), key=lambda i: encs[i].n)
t0 = time.time()
argmax_tok, cls_logits = {}, {}
for b in batched(order, 32):
    ids, last = vc.pad([encs[i].ids for i in b])
    lg, _, _ = vc.run(vc.embed(ids), last_idx=last)
    for j, i in enumerate(b):
        argmax_tok[i] = int(lg[j].argmax())
        cls_logits[i] = lg[j, cls_ids].cpu()
dt = time.time() - t0
log(f"phase1: {len(valid)} forward passes in {dt:.0f}s ({len(valid)/dt:.2f}/s); mean len {sum(e.n for e in encs)/len(encs):.0f} tokens")
n_valid = sum(argmax_tok[i] in cls_ids for i in range(len(valid)))
log(f"phase1 argmax token is a class first-token: {n_valid}/{len(valid)}")
log("phase1 argmax tokens:", Counter(tok.decode(argmax_tok[i]) for i in range(len(valid))).most_common(15))
p1 = [int(cls_logits[i].argmax()) for i in range(len(valid))]
log("phase1 class dist:", [(CLASSES[c], n) for c, n in sorted(Counter(p1).items())])
import numpy as np

em = np.array([r["em"] for r in valid])
conf = np.array([0.05 + 0.1 * c for c in p1])
for c in sorted(set(p1)):
    m = np.array(p1) == c
    log(f"   {CLASSES[c]:20s} n={m.sum():4d} acc(EM)={em[m].mean():.2f}")
p0c = [CLASSES.index(r["p0_conf"]) if r["p0_conf"] else -1 for r in valid]
log("phase0 vs phase1 class agreement:", np.mean(np.array(p0c) == np.array(p1)))

# ---------------- minimal numeric prompt ----------------
encm = [encode(tok, r["question"], r["answer"], "min") for r in valid]
dig = []
am = []
for b in batched(list(range(len(valid))), 64):
    ids, last = vc.pad([encm[i].ids for i in b])
    lg, _, _ = vc.run(vc.embed(ids), last_idx=last)
    am += [int(x) for x in lg.argmax(-1)]
    dig += [int(x) for x in lg[:, dig_ids].argmax(-1)]
log("minimal prompt argmax tokens:", Counter(tok.decode(a) for a in am).most_common(12))
log("minimal prompt digit dist:", sorted(Counter(dig).items()))
d = np.array(dig)
for k in sorted(set(dig)):
    log(f"   digit {k}: n={(d==k).sum():4d} acc(EM)={em[d==k].mean():.2f}")

# ---------------- engine checks ----------------
b = order[:8]
ids, last = vc.pad([encs[i].ids for i in b])
emb = vc.embed(ids)
lg_mine, cache, _ = vc.run(emb, last_idx=last, capture=SWEEP_LAYERS)
lg_hf = vc.hf_logits(ids, last)
log(f"[check1] my loop vs HF forward: max|dlogit|={float((lg_mine-lg_hf).abs().max()):.4f}; "
    f"class-logit max diff={float((lg_mine[:, cls_ids]-lg_hf[:, cls_ids].to(lg_mine.device)).abs().max()):.4f}; "
    f"argmax equal: {bool((lg_mine.argmax(-1)==lg_hf.argmax(-1).to(lg_mine.device)).all())}")
# unpadded single-row == padded batch row
i0 = b[0]
ids1, last1 = vc.pad([encs[i0].ids])
lg1, _, _ = vc.run(vc.embed(ids1), last_idx=last1)
log(f"[check1b] padded-batch row vs single row: max|d class logit|={float((lg1[0, cls_ids]-lg_mine[0, cls_ids]).abs().max()):.4f}")

# fast path vs reference path: add a perturbation at PANL at layer l
for l in [0, 22, 25, 40, 61]:
    panl = torch.tensor([encs[i].pos["PANL"] for i in b])
    delta = torch.randn(len(b), vc.d, generator=torch.Generator().manual_seed(l)).to(vc.dtype)

    def edit(i, h, l=l):
        if i == l:
            h = h.clone()
            h[torch.arange(len(b)), panl.to(h.device)] += (delta.to(h.device) * 0.1 * h[torch.arange(len(b)), panl.to(h.device)].norm(dim=-1, keepdim=True) / delta.to(h.device).norm(dim=-1, keepdim=True)).to(h.dtype)
        return h

    lg_ref, _, _ = vc.run(emb, last_idx=last, edit=edit)
    h_mod = edit(l, cache[l])
    lg_fast, _, _ = vc.run(h_mod, start=l + 1, last_idx=last)
    log(f"[check2] layer {l:2d}: resume vs in-loop edit max|d class logit|={float((lg_ref[:, cls_ids]-lg_fast[:, cls_ids]).abs().max()):.4f}; "
        f"effect size vs clean={float((lg_ref[:, cls_ids]-lg_mine[:, cls_ids]).abs().max()):.3f}")

# attention blocking: empty mask == no mask ; blocking CC->everything-but-self changes output
T = ids.shape[1]
none = torch.zeros(len(b), T, T, dtype=torch.bool)
lg_blk0, _, _ = vc.run(emb, last_idx=last, block={"layers": set(range(vc.L)), "mask": none})
log(f"[check3] empty block mask vs unmasked: max|d class logit|={float((lg_blk0[:, cls_ids]-lg_mine[:, cls_ids]).abs().max()):.4f}")
m = none.clone()
for j, i in enumerate(b):
    e = encs[i]
    m[j, e.pos["CC"], e.a_span[0] : e.a_span[1]] = True
lg_blk, _, _ = vc.run(emb, last_idx=last, block={"layers": set(range(vc.L)), "mask": m})
log(f"[check3b] block CC->A (all layers): max|d class logit|={float((lg_blk[:, cls_ids]-lg_mine[:, cls_ids]).abs().max()):.3f}")
# windowed block via resume: block layers 20..31 == resume from cache[19] with block
m2 = none.clone()
for j, i in enumerate(b):
    e = encs[i]
    m2[j, e.pos["PANL"], e.a_span[0] : e.a_span[1]] = True
win = set(range(20, 32))
lg_w_ref, _, _ = vc.run(emb, last_idx=last, block={"layers": win, "mask": m2})
_, c19, _ = vc.run(emb, capture=[19], stop=20)
lg_w_fast, _, _ = vc.run(c19[19], start=20, last_idx=last, block={"layers": win, "mask": m2})
log(f"[check3c] windowed block PANL->A L20-31: ref vs resume max|d|={float((lg_w_ref[:, cls_ids]-lg_w_fast[:, cls_ids]).abs().max()):.4f}; effect={float((lg_w_ref[:, cls_ids]-lg_mine[:, cls_ids]).abs().max()):.3f}")

# residual norms at PANL / CC
for l in [0, 10, 25, 40, 61]:
    h = cache[l]
    pn = h[torch.arange(len(b)), torch.tensor([encs[i].pos["PANL"] for i in b]).to(h.device)].float().norm(dim=-1).mean()
    cn = h[torch.arange(len(b)), last.to(h.device)].float().norm(dim=-1).mean()
    log(f"   layer {l:2d}: mean |resid| PANL={float(pn):.1f} CC={float(cn):.1f}")

# ---------------- throughput ----------------
del cache
torch.cuda.empty_cache()
pool = order[len(order) // 2 : len(order) // 2 + 96]
for bs in [16, 32, 64, 96]:
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    t0 = time.time()
    n = 0
    for bb in batched(pool, bs):
        ids, last = vc.pad([encs[i].ids for i in bb])
        vc.run(vc.embed(ids), last_idx=last)
        n += len(bb)
    torch.cuda.synchronize()
    dt = time.time() - t0
    log(f"[speed] full pass bs={bs}: {n/dt:.2f} prompts/s; peak mem " + ", ".join(f"{torch.cuda.max_memory_allocated(i)/2**30:.1f}G" for i in range(torch.cuda.device_count())))
# resume from layer 25 at bs 64
ids, last = vc.pad([encs[i].ids for i in pool[:64]])
_, c, _ = vc.run(vc.embed(ids), capture=[25], stop=26)
t0 = time.time()
vc.run(c[25], start=26, last_idx=last)
torch.cuda.synchronize()
log(f"[speed] resume from L25 bs=64: {64/(time.time()-t0):.2f} prompts/s")
log("PILOT DONE")
