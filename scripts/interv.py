"""Causal interventions on Gemma 3 27B (one shard of one experiment).

    python scripts/interv.py --exp steer|patch|noise|swap|attn_min|attn_cat --shard k --nshards n

Additions to the paper's experiments: --exp patchwin holds a position at its clean activations over a range of
layers instead of restoring it at one layer; --corrupt-span q|qa corrupts the question tokens instead of or as
well as the answer tokens; --exp attn_q blocks attention to the question in an unedited run.

With VC_FMT=tt the same experiments run on the two-turn chat format (data/tt, results/tt): PANL is then the
<end_of_turn> token that closes the model's answer turn and CC the last token of the prompt, further positions
between the answer and the request are added, and "POST5" edits the five tokens between answer and request
together. Attention blocking in that format: --exp attn_tt (all layers) and attn_ttw (12-layer windows).

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
from vc.env import DATA, FMT, RESULTS
from vc.model import VC, batched, log
from vc.prompts import TT_POST5, class_token_ids, digit_token_ids, encode, encode_trial

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
ap.add_argument("--vec-file", default="vectors.pt", help="steering: file in results/ holding the steering directions")
ap.add_argument("--steer-pos", default="", help="steering: comma-separated subset of positions (default: all seven)")
ap.add_argument("--steer-scales", default="", help="steering: comma-separated scales for every selected position")
ap.add_argument("--steer-dirs", default="high,low", help="steering: which signs of the direction to run (high = +, low = -)")
ap.add_argument("--patch-pos", default="", help="patch experiment: comma-separated positions to restore instead of the default ones "
                                                "(INSTR = all tokens between PANL and CC, restored together)")
ap.add_argument("--corrupt", default="pos", choices=["pos", "abs", "global"],
                help="patch experiment: how the mean embedding that replaces an answer token is built. pos: mean over the "
                     "calibration answers that have a token at the same answer position; abs: mean over all calibration "
                     "prompts at the same absolute sequence position; global: one mean over all calibration answer tokens")
ap.add_argument("--patch-windows", default="0-61,15-28,20-28,20-35,28-35",
                help="patchwin experiment: layer ranges in which a position is held at its clean activations")
ap.add_argument("--patch-block", default="", choices=["", "pre"],
                help="patchwin experiment: pre = in the same passes, block in all layers the attention of every token after PANL "
                     "to the question, the answer label and the answer, so that PANL is the only token that reads them")
ap.add_argument("--corrupt-span", default="a", choices=["a", "q", "qa"],
                help="patch experiment: which token embeddings are corrupted. a: the answer tokens (the paper); q: the question "
                     "tokens; qa: both")
args = ap.parse_args()

trials = read_jsonl(DATA / "trials.jsonl")
sets = json.load(open(DATA / "sets.json"))
vc = VC()
tok = vc.tok
cls_ids, dig_ids = class_token_ids(tok), digit_token_ids(tok)
LAYERS = [int(x) for x in args.layers.split(",")] if args.layers else SWEEP_LAYERS
VI = {l: SWEEP_LAYERS.index(l) for l in LAYERS}  # index of each layer in the stored vectors


TT = FMT == "tt"
GROUPS = {"POST5": TT_POST5} if TT else {}  # named positions that are edited together


def enc_of(i, kind="cat"):
    return encode_trial(tok, trials[i], kind, panl="\n" * args.panl_n)


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


def take(h, pos):
    """Residuals of h [B,T,d] at pos: [B] -> [B,d]; a group of positions [B,k] -> [B,k,d]."""
    ar = torch.arange(h.shape[0], device=h.device)
    return h[ar[:, None] if pos.dim() == 2 else ar, pos.to(h.device)]


def setter(pos, vec):
    """Variant that overwrites row b at position pos[b] with vec[b] (vec: [B,d] or [d]); for a group of
    positions pos is [B,k] and vec [B,k,d] or [k,d]."""

    def f(h):
        ar = torch.arange(h.shape[0], device=h.device)
        h[ar[:, None] if pos.dim() == 2 else ar, pos.to(h.device)] = vec.to(h.device, h.dtype)
        return h

    return f


def adder(pos, vec):
    def f(h):
        ar = torch.arange(h.shape[0], device=h.device)
        h[ar[:, None] if pos.dim() == 2 else ar, pos.to(h.device)] += vec.to(h.device, h.dtype)
        return h

    return f


def null(h):
    return h


def pos_tensor(encs, name):
    """Token index [B] of a named position, or [B,k] for a group of positions."""
    if name == "BOS":
        return torch.zeros(len(encs), dtype=torch.long)
    if name in encs[0].pos:
        return torch.tensor([e.pos[name] for e in encs])
    return torch.tensor([e.groups[name] for e in encs])


def run_exp(exp):
  t_start = time.time()
  outdir = RESULTS / (exp + args.tag)
  os.makedirs(outdir, exist_ok=True)
  if os.path.exists(outdir / f"shard{args.shard}.pt") and not args.limit:
    log(f"{exp}: shard {args.shard} already done, skipping")
    return
  # =====================================================================================================
  if exp == "steer":
      V = torch.load(RESULTS / args.vec_file)  # unit[pos][layer_idx], norm[pos][layer_idx]
      POS = ["PANL", "CC", "PANL+1", "FCC", "AC", "A_first", "A_last"]
      SCALES = {p: [2, 5] for p in POS}
      SCALES["PANL"] = SCALES["CC"] = [1, 2, 5, 10]
      if TT:
          POS = ["PANL", "CC", "PANL+1", "SOT", "USER", "UNL", "POST5", "REQ0", "CC-1", "AC", "A_last"]
          SCALES = {p: [5] for p in POS}
          SCALES["PANL"] = SCALES["CC"] = SCALES["POST5"] = [2, 5]

      def steer_delta(p, li, c):
          """c x residual norm x unit direction at position p (for a group: one row per member position)."""
          if p in GROUPS:
              return torch.stack([c * V["norm"][q][li] * V["unit"][q][li] for q in GROUPS[p]])
          return c * V["norm"][p][li] * V["unit"][p][li]

      if args.steer_pos:
          POS = [p for p in POS if p in args.steer_pos.split(",")]
      if args.steer_scales:
          SCALES = {p: [int(x) for x in args.steer_scales.split(",")] for p in POS}
      DIRS = [(d, {"high": 1.0, "low": -1.0}[d]) for d in args.steer_dirs.split(",")]
      names = [f"{p}|{d}|{s}" for p in POS for s in SCALES[p] for d, _ in DIRS]
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
                      for _, sign in DIRS:
                          variants.append(adder(pt, steer_delta(p, VI[l], sign * s * 0.03)))
              out, ref = run_variants(cache[l], l, variants, last, cls_ids)  # [n_var, B, 10]
              res[b, :, li] = out.permute(1, 0, 2)
              refs[b, :, li] = ref.permute(1, 0, 2)
          del cache
          log(f"steer batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": names, "layers": LAYERS, "logits": res, "ref": refs, "clean": clean}, outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp == "noise":
      POS = ["PANL", "CC", "PANL+1", "AC"]
      if TT:
          POS = ["PANL", "CC", "PANL+1", "UNL", "POST5", "REQ0", "AC"]
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
                  if p in GROUPS:  # every member position is set to its own calibration mean
                      variants.append(setter(pt, torch.stack([V["mean"][q][VI[l]] for q in GROUPS[p]])))
                      continue
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
  elif exp in ("patch", "patchwin"):
      # corruption: answer-token embeddings replaced by the calibration-set mean embedding at the same
      # answer position (C.1.4); then clean activations are restored at one (position, layer).
      # patchwin: the same corruption, but a position is held at its clean activations in every layer of a
      # range (--patch-windows) while the rest of the corrupted pass runs; one forward pass per range.
      POS = ["PANL", "CC", "PANL+1"]
      if TT:  # REQ: all request tokens together; TAIL4: the four tokens between the request and CC
          POS = ["PANL", "CC", "PANL+1", "SOT", "USER", "UNL", "POST5", "REQ0", "REQ", "TAIL4", "CC-1"]
      if args.patch_pos:
          POS = args.patch_pos.split(",")
      # AC: restoring it is a no-op by causality when only the answer is corrupted (AC precedes the answer tokens).
      # When the question is corrupted too, AC is a real condition (the token that produced the answer) and the
      # no-op control is the first token of the prompt (BOS).
      names = POS + ["AC"] + (["BOS"] if "q" in args.corrupt_span else [])
      calib = sets["calib"]
      cal_enc = [enc_of(i) for i in calib]

      def span_means(span_of):
          """Mean calibration embedding per position within a span, and over all its tokens."""
          n_max = max(span_of(e)[1] - span_of(e)[0] for e in cal_enc)
          sums, cnt = torch.zeros(n_max, vc.d), torch.zeros(n_max)
          for e in cal_enc:
              t_ids = torch.tensor(e.ids[span_of(e)[0] : span_of(e)[1]])
              sums[: len(t_ids)] += vc.embed(t_ids[None])[0].float().cpu()
              cnt[: len(t_ids)] += 1
          return sums / cnt[:, None], sums.sum(0) / cnt.sum(), n_max, cnt

      mean_emb, overall, max_a, cnt = span_means(lambda e: e.a_span)
      mean_q, overall_q, max_q, _ = span_means(lambda e: e.q_span)
      log(f"patch: calibration answer lengths up to {max_a}; counts per answer position {cnt.tolist()}")
      # alternative readings of C.1.4, eq. (1), in which every position averages over all calibration trials
      T_abs = min(e.n for e in cal_enc)
      mean_abs = vc.embed(torch.tensor([e.ids[:T_abs] for e in cal_enc])).float().mean(0).cpu()  # [T_abs, d]
      log(f"patch: corruption = {args.corrupt}; absolute-position means available up to position {T_abs}")

      def corrupt_vec(k, t, span="a"):
          """Replacement for the k-th token of the answer (or question) span, which sits at sequence position t."""
          if args.corrupt == "abs":
              return mean_abs[t]
          if span == "q":
              return overall_q if args.corrupt == "global" or k >= max_q else mean_q[k]
          if args.corrupt == "global":
              return overall
          return mean_emb[k] if k < max_a else overall

      idx = shard(sets[args.set or "patch"])
      encs = {i: enc_of(i) for i in set(idx)}
      res = torch.zeros(len(idx), len(names), len(LAYERS), 10)
      refs = torch.zeros_like(res)  # corrupted run without any restoration, same forward pass as the patches
      clean = torch.zeros(len(idx), 10)
      corrupt = torch.zeros(len(idx), 10)
      ood = torch.zeros(len(idx), len(POS), len(LAYERS), 2)  # cosine(clean, corrupt), norm ratio corrupt/clean
      WINS = [tuple(int(x) for x in w.split("-")) for w in args.patch_windows.split(",")]
      WVAR = [(p, w0, w1) for (w0, w1) in WINS for p in names if p != "BOS"] + ([("BOS", 0, vc.L - 1)] if "BOS" in names else [])
      if args.patch_block:
          WVAR.append(("NONE", 0, -1))  # the corrupted run with the attention block and nothing restored
      wres = torch.zeros(len(idx), len(WVAR), 10)
      order = sorted(range(len(idx)), key=lambda j: encs[idx[j]].n)
      for bi, b in enumerate(batched(order, args.bs)):
          E = [encs[idx[j]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          emb = vc.embed(ids)
          if exp == "patchwin":  # clean activations of the patched positions at every layer
              PT = {p: pos_tensor(E, p) for p in names}
              lg, cw, _ = vc.run(emb, last_idx=last, capture=range(vc.L), sel=cls_ids,
                                 capture_fn=lambda l, h: {p: take(h, PT[p]).cpu() for p in names})
          else:
              lg, cc, _ = vc.run(emb, last_idx=last, capture=LAYERS, sel=cls_ids)
          clean[b] = lg.cpu()
          emb_c = emb.clone()
          for r, e in enumerate(E):
              for span in args.corrupt_span:
                  s0, s1 = e.q_span if span == "q" else e.a_span
                  for k, t in enumerate(range(s0, s1)):
                      emb_c[r, t] = corrupt_vec(k, t, span).to(emb_c.device, emb_c.dtype)
          lg, cx, _ = vc.run(emb_c, last_idx=last, capture=LAYERS if exp == "patch" else None, sel=cls_ids)
          corrupt[b] = lg.cpu()
          ar = torch.arange(len(b))
          if exp == "patchwin":
              B = len(b)
              Mb = None
              if args.patch_block:
                  Mb = torch.zeros(B, emb_c.shape[1], emb_c.shape[1], dtype=torch.bool)
                  for r, e in enumerate(E):
                      Mb[r, e.pos["PANL"] + 1 :, e.q_span[0] : e.pos["PANL"]] = True
              for chunk in batched(list(range(len(WVAR))), max(1, args.rows // B)):

                  def edit(l, h, chunk=chunk):
                      for vi, w in enumerate(chunk):
                          p, w0, w1 = WVAR[w]
                          if w0 <= l <= w1:
                              pt, rows = PT[p].to(h.device), ar.to(h.device) + vi * B
                              h[rows[:, None] if pt.dim() == 2 else rows, pt] = cw[l][p].to(h.device, h.dtype)
                      return h

                  blk = {"layers": set(range(vc.L)), "mask": Mb.repeat(len(chunk), 1, 1)} if Mb is not None else None
                  lgw, _, _ = vc.run(emb_c.repeat(len(chunk), 1, 1), last_idx=last.repeat(len(chunk)), edit=edit, sel=cls_ids, block=blk)
                  wres[torch.tensor(b)[:, None], torch.tensor(chunk)[None]] = lgw.view(len(chunk), B, 10).permute(1, 0, 2).cpu()
              del cw
              log(f"patchwin batch {bi}: {time.time()-t_start:.0f}s")
              continue
          for li, l in enumerate(LAYERS):
              variants = []
              for pi, p in enumerate(names):
                  pt = pos_tensor(E, p)
                  hc = take(cc[l], pt)
                  variants.append(setter(pt, hc))
                  if p in POS and pt.dim() == 1:
                      hx = cx[l][ar.to(cx[l].device), pt.to(cx[l].device)].float().cpu()
                      ood[b, pi, li, 0] = torch.nn.functional.cosine_similarity(hc.float().cpu(), hx, dim=-1)
                      ood[b, pi, li, 1] = hx.norm(dim=-1) / hc.float().cpu().norm(dim=-1)
              out, ref = run_variants(cx[l], l, variants, last, cls_ids)
              res[b, :, li] = out.permute(1, 0, 2)
              refs[b, :, li] = ref.permute(1, 0, 2)
          del cc, cx
          log(f"patch batch {bi}: {time.time()-t_start:.0f}s")
      if exp == "patchwin":
          torch.save({"idx": idx, "win_names": WVAR, "logits": wres, "clean": clean, "corrupt": corrupt,
                      "corrupt_span": args.corrupt_span, "block": args.patch_block}, outdir / f"shard{args.shard}.pt")
      else:
          torch.save({"idx": idx, "names": names, "layers": LAYERS, "logits": res, "ref": refs, "clean": clean, "corrupt": corrupt,
                      "ood": ood, "ood_pos": POS, "corrupt_span": args.corrupt_span}, outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp == "swap":
      POS = ["PANL", "CC", "PANL+1"]
      DPOS = POS  # single positions whose donor activations are cached
      if TT:
          POS, DPOS = ["PANL", "CC", "POST5"], TT_POST5 + ["CC"]
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
          P = torch.tensor([[e.pos[p] for p in DPOS] for e in E])
          ar = torch.arange(len(b))[:, None]
          _, cache, _ = vc.run(vc.embed(ids), capture=LAYERS, capture_fn=lambda l, h: h[ar.to(h.device), P.to(h.device)].cpu())
          st = torch.stack([cache[l] for l in LAYERS], 1)  # [B, n_layers, len(DPOS), d]
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
              for p in POS:
                  pt = pos_tensor(E, p)
                  pi = [DPOS.index(q) for q in GROUPS[p]] if p in GROUPS else DPOS.index(p)
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
  # =====================================================================================================
  elif exp == "attn_q":
      # categorical prompt, edges blocked in ALL layers: who needs to read the question? NL = PANL, LATER = every
      # token after PANL, ALL = PANL and every later token, PRE = question, answer label and answer tokens
      # (everything from the first question token up to PANL; the fixed header before the question stays visible).
      CONDS = ["none", "LATER>Q", "LATER>A", "LATER>Q+A", "LATER>PRE", "ALL>Q", "ALL>PRE", "NL>Q", "NL>A", "NL>PRE"]
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
              nl, q, a = e.pos["PANL"], e.q_span, e.a_span
              M[1, r, nl + 1 :, q[0] : q[1]] = True
              M[2, r, nl + 1 :, a[0] : a[1]] = True
              M[3, r, nl + 1 :, q[0] : q[1]] = True
              M[3, r, nl + 1 :, a[0] : a[1]] = True
              M[4, r, nl + 1 :, q[0] : nl] = True  # PANL is the only token after the answer that sees question or answer
              M[5, r, nl:, q[0] : q[1]] = True
              M[6, r, nl:, q[0] : nl] = True  # nothing after the answer sees question or answer
              M[7, r, nl, q[0] : q[1]] = True
              M[8, r, nl, a[0] : a[1]] = True
              M[9, r, nl, q[0] : nl] = True  # PANL sees neither; later tokens still do
          out, _, _ = vc.run(emb.repeat(len(CONDS), 1, 1), last_idx=last.repeat(len(CONDS)), sel=cls_ids,
                             block={"layers": set(range(vc.L)), "mask": M.view(-1, T, T)})
          res[b] = out.view(len(CONDS), B, -1).permute(1, 0, 2).cpu()
          if bi % 10 == 0:
              log(f"attn_q batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": CONDS, "logits": res, "clean": clean}, outdir / f"shard{args.shard}.pt")
  # =====================================================================================================
  elif exp == "attn_tt":
      # two-turn format, edges blocked in ALL layers. NL = <end_of_turn> after the answer (PANL), POST5 = the
      # five tokens between answer and request, REQ = request tokens, TAIL = the five tokens after the request
      # (the last one is CC). "X>Y": tokens X may not attend to tokens Y.
      CONDS = ["none", "CC>NL+1", "CC>NL", "CC>POST5", "CC>A", "CC>Q", "CC>Q+A", "ALL>NL", "ALL>POST5", "ALL>last_A",
               "ALL>last_A_keepNL", "ALL>A", "ALL>A_keepNL", "ALL>A_keepPOST5", "TAIL>A", "REQ>A", "ALL>A+POST5", "TAIL>A+POST5"]
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
              la, rq, tl = a[1] - 1, e.pos["REQ0"], e.pos["EOT2"]  # last answer token, first request token, first TAIL token
              M[1, r, cc, nl + 1] = True
              M[2, r, cc, nl] = True
              M[3, r, cc, nl:rq] = True
              M[4, r, cc, a[0] : a[1]] = True
              M[5, r, cc, q[0] : q[1]] = True
              M[6, r, cc, q[0] : q[1]] = True
              M[6, r, cc, a[0] : a[1]] = True
              M[7, r, nl + 1 :, nl] = True  # all later tokens -> end-of-turn token
              M[8, r, rq:, nl:rq] = True  # request and final tokens -> the five tokens after the answer
              M[9, r, nl:, la] = True  # all tokens after the answer -> last answer token
              M[10, r, nl + 1 :, la] = True  # ... but the end-of-turn token keeps that edge
              M[11, r, nl:, a[0] : a[1]] = True  # all tokens after the answer -> answer tokens
              M[12, r, nl + 1 :, a[0] : a[1]] = True  # ... but the end-of-turn token keeps its edges
              M[13, r, rq:, a[0] : a[1]] = True  # ... but all five tokens after the answer keep theirs
              M[14, r, tl:, a[0] : a[1]] = True  # final five tokens -> answer tokens
              M[15, r, rq:tl, a[0] : a[1]] = True  # request tokens -> answer tokens
              M[16, r, rq:, a[0] : rq] = True  # request and final tokens -> answer and the five tokens after it
              M[17, r, tl:, a[0] : rq] = True  # final five tokens -> answer and the five tokens after it
          Mf = M.view(-1, T, T)
          H = emb.repeat(len(CONDS), 1, 1)
          out, _, _ = vc.run(H, last_idx=last.repeat(len(CONDS)), sel=cls_ids, block={"layers": set(range(vc.L)), "mask": Mf})
          res[b] = out.view(len(CONDS), B, -1).permute(1, 0, 2).cpu()
          if bi % 10 == 0:
              log(f"attn_tt batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": CONDS, "logits": res, "clean": clean}, outdir / f"shard{args.shard}.pt")

  # =====================================================================================================
  elif exp == "attn_ttw":
      # two-turn format, edges blocked in 12 consecutive layers centred on c (the timing analysis of Fig. 7,
      # here on the categorical two-turn prompt). ALL>REQ0 is the control: one uninformative token is hidden.
      CENTERS = [10, 15, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 45, 50, 56]
      CONDS = ["none", "NL>A", "POST5>A", "ALL>NL", "ALL>POST5", "LATER>A", "CC>POST5", "ALL>REQ0"]
      idx = shard(sets["attn_win"])
      encs = {i: enc_of(i) for i in set(idx)}
      res = torch.zeros(len(idx), len(CONDS), len(CENTERS), 10)
      clean = torch.zeros(len(idx), 10)
      order = sorted(range(len(idx)), key=lambda j: encs[idx[j]].n)
      starts = [c - 6 for c in CENTERS]
      for bi, b in enumerate(batched(order, 8)):
          E = [encs[idx[j]] for j in b]
          ids, last = vc.pad([e.ids for e in E])
          B, T = ids.shape
          lg, cache, _ = vc.run(vc.embed(ids), last_idx=last, capture=[s - 1 for s in starts], sel=cls_ids)
          clean[b] = lg.cpu()
          M = torch.zeros(len(CONDS), B, T, T, dtype=torch.bool)
          for r, e in enumerate(E):
              cc, nl, a, rq = e.pos["CC"], e.pos["PANL"], e.a_span, e.pos["REQ0"]
              M[1, r, nl, a[0] : a[1]] = True  # end-of-turn token -> answer tokens
              M[2, r, nl:rq, a[0] : a[1]] = True  # the five tokens after the answer -> answer tokens
              M[3, r, nl + 1 :, nl] = True  # all later tokens -> end-of-turn token
              M[4, r, rq:, nl:rq] = True  # request and final tokens -> the five tokens after the answer
              M[5, r, rq:, a[0] : a[1]] = True  # request and final tokens -> answer tokens
              M[6, r, cc, nl:rq] = True  # last token -> the five tokens after the answer
              M[7, r, rq + 1 :, rq] = True  # all later tokens -> first request token (control)
          Mf = M.view(-1, T, T)
          for wi, s in enumerate(starts):
              H = cache[s - 1].repeat(len(CONDS), 1, 1)
              out, _, _ = vc.run(H, start=s, last_idx=last.repeat(len(CONDS)), sel=cls_ids,
                                 block={"layers": set(range(s, s + 12)), "mask": Mf})
              res[b, :, wi] = out.view(len(CONDS), B, -1).permute(1, 0, 2).cpu()
          del cache
          if bi % 5 == 0:
              log(f"attn_ttw batch {bi}: {time.time()-t_start:.0f}s")
      torch.save({"idx": idx, "names": CONDS, "centers": CENTERS, "logits": res, "clean": clean}, outdir / f"shard{args.shard}.pt")
  else:
      raise SystemExit(f"unknown exp {exp}")

  log(f"INTERV DONE {exp} shard {args.shard} in {time.time()-t_start:.0f}s")


for _exp in args.exp.split(","):
    run_exp(_exp)
log("ALL DONE", args.exp)
