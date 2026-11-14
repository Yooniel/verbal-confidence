"""Causal interventions on Gemma 3 27B (one shard of one experiment).

    python scripts/interv.py --exp steer|patch|noise|swap|attn_min|attn_cat --shard k --nshards n

Every intervention at (layer l, position p) is run by resuming the forward pass from the cached residual
stream at the output of block l (bit-identical to editing inside a full pass; see scripts/pilot.py).
Each experiment also runs a "null" variant (no edit, same code path) to measure the numerical-noise floor.

Outputs (results/<exp>/shard<k>.pt): for every trial, variant and layer the float32 logits of the 10
confidence classes (or digits) at the last prompt token, plus the clean-run logits.
"""
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
from vc.prompts import class_token_ids, digit_token_ids, encode

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--bs", type=int, default=8, help="trials per clean batch")
ap.add_argument("--rows", type=int, default=64, help="max rows per resumed pass")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--panl-n", type=int, default=2, help="number of newlines after the answer (the PANL token)")
ap.add_argument("--tag", default="")
ap.add_argument("--layers", default="", help="comma-separated subset of the sweep layers (smoke tests)")
ap.add_argument("--set", default="", help="override the trial set (key in data/sets.json) of the patch experiment")
args = ap.parse_args()

trials = read_jsonl(DATA / "trials.jsonl")
sets = json.load(open(DATA / "sets.json"))
vc = VC()
tok = vc.tok
cls_ids, dig_ids = class_token_ids(tok), digit_token_ids(tok)
LAYERS = [int(x) for x in args.layers.split(",")] if args.layers else SWEEP_LAYERS
VI = {l: SWEEP_LAYERS.index(l) for l in LAYERS}  # index of each layer in the stored vectors


def enc_of(i, kind="cat"):
    r = trials[i]
    return encode(tok, r["question"], r["answer"], kind, panl="\n" * args.panl_n)


def shard(lst):
    lst = list(lst)
    if args.limit:
        lst = lst[: args.limit]
    return lst[args.shard :: args.nshards]


def run_variants(base, layer, variants, last, sel):
    """base: [B,T,d] residual at the output of `layer`. variants: list of callables f(h_clone)->h (in-place
    edit of a [B,T,d] clone). Every forward pass also carries an unedited copy of the batch; its logits are
    returned as the reference ("ref") for the variants computed in that same pass, so that edited and
    unedited rows share exactly the same numerics (bf16 results depend slightly on the batch shape).
    Returns (logits [n_var, B, K], ref [n_var, B, K])."""
    B = base.shape[0]
    per = max(1, args.rows // B - 1)
    outs, refs = [], []
    for chunk in batched(variants, per):
        H = torch.cat([base] + [f(base.clone()) for f in chunk], 0)
        lg, _, _ = vc.run(H, start=layer + 1, last_idx=last.repeat(len(chunk) + 1), sel=sel)
        lg = lg.view(len(chunk) + 1, B, -1).cpu()
        outs.append(lg[1:])
        refs.append(lg[0:1].expand(len(chunk), -1, -1))
        del H
    return torch.cat(outs, 0), torch.cat(refs, 0)


def setter(pos, vec):
    """Variant that overwrites row b at position pos[b] with vec[b] (vec: [B,d] or [d])."""

    def f(h):
        ar = torch.arange(h.shape[0], device=h.device)
        h[ar, pos.to(h.device)] = vec.to(h.device, h.dtype)
        return h

    return f


def adder(pos, vec):
    def f(h):
        ar = torch.arange(h.shape[0], device=h.device)
        h[ar, pos.to(h.device)] += vec.to(h.device, h.dtype)
        return h

    return f


def null(h):
    return h


def pos_tensor(encs, name):
    return torch.tensor([e.pos[name] for e in encs])


def run_exp(exp):
  t_start = time.time()
  outdir = RESULTS / (exp + args.tag)
  os.makedirs(outdir, exist_ok=True)
  if os.path.exists(outdir / f"shard{args.shard}.pt") and not args.limit:
    log(f"{exp}: shard {args.shard} already done, skipping")
    return
  # =====================================================================================================
  if exp == "steer":
      V = torch.load(RESULTS / "vectors.pt")  # unit[pos][layer_idx], norm[pos][layer_idx]
      POS = ["PANL", "CC", "PANL+1", "FCC", "AC", "A_first", "A_last"]
      SCALES = {p: [2, 5] for p in POS}
      SCALES["PANL"] = SCALES["CC"] = [1, 2, 5, 10]
      names = [f"{p}|{d}|{s}" for p in POS for s in SCALES[p] for d in ("high", "low")]
      idx = shard(sets["steer"])
      encs = {i: enc_of(i) for i in set(idx)}
      res = torch.zeros(len(idx), len(names), len(LAYERS), 10)
      refs = torch.zeros_like(res)
      clean = torch.zeros(len(idx), 10)
      order = sorted(range(len(idx)), key=lambda j: encs[idx[j]].n)
      for bi, b in enumerate(batched(order, args.bs)):
          E = [encs[idx[j]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          lg, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=LAYERS, sel=cls_ids)
          clean[b] = lg.cpu()
          for li, l in enumerate(LAYERS):
              variants = []
              for p in POS:
                  pt = pos_tensor(E, p)
                  for s in SCALES[p]:
                      for sign in (1.0, -1.0):
                          delta = sign * s * 0.03 * V["norm"][p][VI[l]] * V["unit"][p][VI[l]]
                          variants.append(adder(pt, delta))
              out, ref = run_variants(cache[l], l, variants, last, cls_ids)  # [n_var, B, 10]
              res[b, :, li] = out.permute(1, 0, 2)
              refs[b, :, li] = ref.permute(1, 0, 2)
          del cache
          log(f"steer batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": names, "layers": LAYERS, "logits": res, "ref": refs, "clean": clean}, outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp == "noise":
      POS = ["PANL", "CC", "PANL+1", "AC"]
      if args.panl_n == 2:
          V = torch.load(RESULTS / "vectors.pt")
      else:
          # prompt variant: recompute the mean-ablation vectors from clean passes over the calibration set
          cal = [enc_of(i) for i in sets["calib"]]
          acc = {p: torch.zeros(len(SWEEP_LAYERS), vc.d, dtype=torch.float64) for p in POS}
          for b in batched(sorted(range(len(cal)), key=lambda j: cal[j].n), 32):
              E = [cal[j] for j in b]
              ids, last = vc.pad([e.ids for e in E])
              PT = torch.tensor([[e.pos[p] for p in POS] for e in E])
              ar2 = torch.arange(len(b))[:, None]
              _, cch, _ = vc.run(vc.embed(ids), capture=SWEEP_LAYERS, capture_fn=lambda l, h: h[ar2.to(h.device), PT.to(h.device)].double().sum(0).cpu())
              for li2, l2 in enumerate(SWEEP_LAYERS):
                  for pi2, p in enumerate(POS):
                      acc[p][li2] += cch[l2][pi2]
          V = {"mean": {p: (acc[p] / len(cal)).float() for p in POS}}
          log(f"noise: recomputed ablation means from {len(cal)} calibration trials (panl_n={args.panl_n})")
      names = list(POS)
      idx = shard(sets["noise"])
      encs = {i: enc_of(i) for i in set(idx)}
      res = torch.zeros(len(idx), len(names), len(LAYERS), 10)
      refs = torch.zeros_like(res)
      clean = torch.zeros(len(idx), 10)
      ood = torch.zeros(len(idx), len(POS), len(LAYERS), 2)  # cosine(clean, mean), norm ratio mean/clean
      order = sorted(range(len(idx)), key=lambda j: encs[idx[j]].n)
      for bi, b in enumerate(batched(order, args.bs)):
          E = [encs[idx[j]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          lg, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=LAYERS, sel=cls_ids)
          clean[b] = lg.cpu()
          ar = torch.arange(len(b))
          for li, l in enumerate(LAYERS):
              variants = []
              for pi, p in enumerate(POS):
                  pt = pos_tensor(E, p)
                  mean = V["mean"][p][VI[l]]
                  variants.append(setter(pt, mean))
                  hc = cache[l][ar.to(cache[l].device), pt.to(cache[l].device)].float().cpu()
                  ood[b, pi, li, 0] = torch.nn.functional.cosine_similarity(hc, mean[None].float(), dim=-1)
                  ood[b, pi, li, 1] = mean.float().norm() / hc.norm(dim=-1)
              out, ref = run_variants(cache[l], l, variants, last, cls_ids)
              res[b, :, li] = out.permute(1, 0, 2)
              refs[b, :, li] = ref.permute(1, 0, 2)
          del cache
          log(f"noise batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": names, "layers": LAYERS, "logits": res, "ref": refs, "clean": clean, "ood": ood, "ood_pos": POS},
                 outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp == "patch":
      # corruption: answer-token embeddings replaced by the calibration-set mean embedding at the same
      # answer position (C.1.4); then clean activations are restored at one (position, layer).
      POS = ["PANL", "CC", "PANL+1"]
      names = POS + ["AC"]  # AC: restoring it is a no-op by causality (AC precedes the corrupted tokens)
      calib = sets["calib"]
      cal_enc = [enc_of(i) for i in calib]
      max_a = max(e.a_span[1] - e.a_span[0] for e in cal_enc)
      W = vc.lm.embed_tokens
      sums = torch.zeros(max_a, vc.d)
      cnt = torch.zeros(max_a)
      for e in cal_enc:
          a_ids = torch.tensor(e.ids[e.a_span[0] : e.a_span[1]])
          emb = vc.embed(a_ids[None])[0].float().cpu()
          sums[: len(a_ids)] += emb
          cnt[: len(a_ids)] += 1
      mean_emb = sums / cnt[:, None]
      overall = sums.sum(0) / cnt.sum()
      log(f"patch: calibration answer lengths up to {max_a}; counts per answer position {cnt.tolist()}")

      idx = shard(sets[args.set or "patch"])
      encs = {i: enc_of(i) for i in set(idx)}
      res = torch.zeros(len(idx), len(names), len(LAYERS), 10)
      refs = torch.zeros_like(res)  # corrupted run without any restoration, same forward pass as the patches
      clean = torch.zeros(len(idx), 10)
      corrupt = torch.zeros(len(idx), 10)
      ood = torch.zeros(len(idx), len(POS), len(LAYERS), 2)  # cosine(clean, corrupt), norm ratio corrupt/clean
      order = sorted(range(len(idx)), key=lambda j: encs[idx[j]].n)
      for bi, b in enumerate(batched(order, args.bs)):
          E = [encs[idx[j]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          emb = vc.embed(ids)
          lg, cc, _ = vc.run(emb, last_idx=last, capture=LAYERS, sel=cls_ids)
          clean[b] = lg.cpu()
          emb_c = emb.clone()
          for r, e in enumerate(E):
              for k, t in enumerate(range(e.a_span[0], e.a_span[1])):
                  emb_c[r, t] = (mean_emb[k] if k < max_a else overall).to(emb_c.device, emb_c.dtype)
          lg, cx, _ = vc.run(emb_c, last_idx=last, capture=LAYERS, sel=cls_ids)
          corrupt[b] = lg.cpu()
          ar = torch.arange(len(b))
          for li, l in enumerate(LAYERS):
              variants = []
              for pi, p in enumerate(names):
                  pt = pos_tensor(E, p)
                  hc = cc[l][ar.to(cc[l].device), pt.to(cc[l].device)]
                  variants.append(setter(pt, hc))
                  if p in POS:
                      hx = cx[l][ar.to(cx[l].device), pt.to(cx[l].device)].float().cpu()
                      ood[b, pi, li, 0] = torch.nn.functional.cosine_similarity(hc.float().cpu(), hx, dim=-1)
                      ood[b, pi, li, 1] = hx.norm(dim=-1) / hc.float().cpu().norm(dim=-1)
              out, ref = run_variants(cx[l], l, variants, last, cls_ids)
              res[b, :, li] = out.permute(1, 0, 2)
              refs[b, :, li] = ref.permute(1, 0, 2)
          del cc, cx
          log(f"patch batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": names, "layers": LAYERS, "logits": res, "ref": refs, "clean": clean, "corrupt": corrupt,
                  "ood": ood, "ood_pos": POS}, outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp == "swap":
      POS = ["PANL", "CC", "PANL+1"]
      plan = json.load(open(DATA / "swap_plan.json"))  # recipients: list of {rec, kind, donor_high, donor_low}
      recs = shard(plan["recipients"])
      need = sorted({r["donor_high"] for r in recs} | {r["donor_low"] for r in recs})
      encs = {i: enc_of(i) for i in set(need) | {r["rec"] for r in recs}}
      # ---- donor activations (clean) at the three positions, all sweep layers
      donor = {}
      order = sorted(need, key=lambda i: encs[i].n)
      for b in batched(order, 32):
          E = [encs[i] for i in b]
          ids, last = vc.pad([e.ids for e in E])
          P = torch.tensor([[e.pos[p] for p in POS] for e in E])
          ar = torch.arange(len(b))[:, None]
          _, cache, _ = vc.run(vc.embed(ids), capture=LAYERS, capture_fn=lambda l, h: h[ar.to(h.device), P.to(h.device)].cpu())
          st = torch.stack([cache[l] for l in LAYERS], 1)  # [B, n_layers, 3, d]
          for j, i in enumerate(b):
              donor[i] = st[j]
      log(f"swap: cached {len(donor)} donors in {time.time()-t_start:.0f}s")
      names = [f"{p}|{d}" for p in POS for d in ("high", "low")]
      res = torch.zeros(len(recs), len(names), len(LAYERS), 10)
      refs = torch.zeros_like(res)
      clean = torch.zeros(len(recs), 10)
      order = sorted(range(len(recs)), key=lambda j: encs[recs[j]["rec"]].n)
      for bi, b in enumerate(batched(order, args.bs)):
          E = [encs[recs[j]["rec"]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          lg, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=LAYERS, sel=cls_ids)
          clean[b] = lg.cpu()
          for li, l in enumerate(LAYERS):
              variants = []
              for pi, p in enumerate(POS):
                  pt = pos_tensor(E, p)
                  for d in ("high", "low"):
                      vec = torch.stack([donor[recs[j][f"donor_{d}"]][li, pi] for j in b])
                      variants.append(setter(pt, vec))
              out, ref = run_variants(cache[l], l, variants, last, cls_ids)
              res[b, :, li] = out.permute(1, 0, 2)
              refs[b, :, li] = ref.permute(1, 0, 2)
          del cache
          if bi % 5 == 0:
              log(f"swap batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"recs": recs, "names": names, "layers": LAYERS, "logits": res, "ref": refs, "clean": clean}, outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp.startswith("attn_min"):
      # minimal numeric prompt; attention edges blocked in 12 consecutive layers centred on c (Fig. 7).
      # "attn_min" uses the default variant; "attn_min_<key>" selects a prompt variant explicitly
      # (key: pm / pmq = separator "\n\n" without / with the trailing quote; pm1 / pmq1 = separator "\n").
      MINV = {"pm": ("min", "\n\n"), "pmq": ("minq", "\n\n"), "pm1": ("min", "\n"), "pmq1": ("minq", "\n")}
      mkind, mpanl = MINV[exp.split("_")[2]] if exp.count("_") == 2 else (sets["min_kind"], sets["min_panl"])
      CENTERS = [10, 15, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 45, 50, 56]
      CONDS = ["none", "CC>PANL", "CC>PANL+1", "CC>Q+A", "PANL>A", "PANL>last_A", "CC>Q", "CC>A"]
      idx = shard(sets["attn_min"])
      encs = {i: encode(tok, trials[i]["question"], trials[i]["answer"], mkind, panl=mpanl) for i in set(idx)}
      res = torch.zeros(len(idx), len(CONDS), len(CENTERS), 10)
      clean = torch.zeros(len(idx), 10)
      order = sorted(range(len(idx)), key=lambda j: encs[idx[j]].n)
      starts = [c - 6 for c in CENTERS]
      for bi, b in enumerate(batched(order, 16)):
          E = [encs[idx[j]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          B, T = ids.shape
          lg, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=[s - 1 for s in starts], sel=dig_ids)
          clean[b] = lg.cpu()
          M = torch.zeros(len(CONDS), B, T, T, dtype=torch.bool)
          for r, e in enumerate(E):
              cc, nl, q, a = e.pos["CC"], e.pos["PANL"], e.q_span, e.a_span
              M[1, r, cc, nl] = True
              M[2, r, cc, nl + 1] = True
              M[3, r, cc, q[0] : q[1]] = True
              M[3, r, cc, a[0] : a[1]] = True
              M[4, r, nl, a[0] : a[1]] = True
              M[5, r, nl, a[1] - 1] = True
              M[6, r, cc, q[0] : q[1]] = True
              M[7, r, cc, a[0] : a[1]] = True
          Mf = M.view(-1, T, T)
          for wi, s in enumerate(starts):
              H = cache[s - 1].repeat(len(CONDS), 1, 1)
              out, _, _ = vc.run(H, start=s, last_idx=last.repeat(len(CONDS)), sel=dig_ids,
                                 block={"layers": set(range(s, s + 12)), "mask": Mf})
              res[b, :, wi] = out.view(len(CONDS), B, -1).permute(1, 0, 2).cpu()
          del cache
          if bi % 5 == 0:
              log(f"attn_min batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": CONDS, "centers": CENTERS, "logits": res, "clean": clean}, outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp == "attn_cat":
      # categorical prompt; edges blocked in ALL layers (Fig. 21)
      CONDS = ["none", "CC>NL+1", "CC>NL", "CC>A", "CC>Q", "CC>Q+A", "ALL>NL", "ALL>last_A", "ALL>NL+last_A",
               "ALL>last_A_keepNL", "ALL>A", "ALL>NL+A", "ALL>A_keepNL"]
      idx = shard(sets["attn_cat"])
      encs = {i: enc_of(i) for i in set(idx)}
      res = torch.zeros(len(idx), len(CONDS), 10)
      clean = torch.zeros(len(idx), 10)
      order = sorted(range(len(idx)), key=lambda j: encs[idx[j]].n)
      for bi, b in enumerate(batched(order, 4)):
          E = [encs[idx[j]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          B, T = ids.shape
          emb = vc.embed(ids)
          lg, _, _ = vc.run(emb, last_idx=last, sel=cls_ids)
          clean[b] = lg.cpu()
          M = torch.zeros(len(CONDS), B, T, T, dtype=torch.bool)
          for r, e in enumerate(E):
              cc, nl, q, a = e.pos["CC"], e.pos["PANL"], e.q_span, e.a_span
              la = a[1] - 1
              M[1, r, cc, nl + 1] = True
              M[2, r, cc, nl] = True
              M[3, r, cc, a[0] : a[1]] = True
              M[4, r, cc, q[0] : q[1]] = True
              M[5, r, cc, q[0] : q[1]] = True
              M[5, r, cc, a[0] : a[1]] = True
              M[6, r, nl + 1 :, nl] = True  # all later tokens -> PANL
              M[7, r, nl:, la] = True  # all tokens after the answer (incl. PANL) -> last answer token
              M[8, r, nl:, la] = True
              M[8, r, nl + 1 :, nl] = True
              M[9, r, nl + 1 :, la] = True  # ... but PANL keeps its edge to the last answer token
              M[10, r, nl:, a[0] : a[1]] = True
              M[11, r, nl:, a[0] : a[1]] = True
              M[11, r, nl + 1 :, nl] = True
              M[12, r, nl + 1 :, a[0] : a[1]] = True
          Mf = M.view(-1, T, T)
          H = emb.repeat(len(CONDS), 1, 1)
          out, _, _ = vc.run(H, last_idx=last.repeat(len(CONDS)), sel=cls_ids, block={"layers": set(range(vc.L)), "mask": Mf})
          res[b] = out.view(len(CONDS), B, -1).permute(1, 0, 2).cpu()
          if bi % 10 == 0:
              log(f"attn_cat batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": CONDS, "logits": res, "clean": clean}, outdir / f"shard{args.shard}.pt")
  else:
      raise SystemExit(f"unknown exp {exp}")

  log(f"INTERV DONE {exp} shard {args.shard} in {time.time()-t_start:.0f}s")


for _exp in args.exp.split(","):
    run_exp(_exp)
log("ALL DONE", args.exp)
