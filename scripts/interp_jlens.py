"""Jacobian lens (Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in Language
Models") on the confidence prompt. One small GPU job, no model forward passes: it reads the residuals of
scripts/interp_capture.py and the steering directions of scripts/prep_vectors.py.

    python scripts/interp_jlens.py --fmt raw|tt

Readout at layer l: logits = unembed(h @ J[l].T) with the released J of neuronpedia/jacobian-lens (fitted on
828 wikitext prompts); the logit lens (J = identity) runs through the same code as the comparison.

Writes results/interp/jlens_<fmt>.pt:
  cls        [2, 62, P, n, 10]  lens logits of the ten confidence-class tokens (0: Jacobian lens, 1: logit lens)
  top1       [2, 62, P, n]      top lens token of every trial
  pin_auroc  [2, 62, P, K]      pinned tokens: AUROC of "low-confidence trial" from the token's lens rank
  pin_rank   [2, 62, P, 2, K]   pinned tokens: median vocabulary rank on high- / low-confidence trials
  kurt       [2, 62, P, n]      excess kurtosis of the lens logits over the vocabulary
  group_top  {(lens, layer, pos): tokens most expressed for high- and low-confidence trials, and the tokens whose
              mean (row-centred) logit differs most between the two groups}
  vec        per steering direction (position, sweep layer, sign): lens readout of the direction, change of the
              readout when the direction is added to real activations, sparse decomposition into J-lens vectors
              (k = 25) with the fraction of the direction it reconstructs, lens kurtosis, and the same numbers
              for random directions and for differences between two random trials
  family     the "Incorrect" lens direction (mean unit J-lens vector of the tokens in FAMILY): its cosine with
              the steering direction, and on the 3,000-trial collection set at the sweep layers how its readout
              follows the stated confidence and the correctness of the answer
and, as inputs of the steering controls (scripts/interv.py --vec-file), in results[/tt]/interp/:
  vectors_jpart*.pt / vectors_jrest*.pt   the steering directions split into their J-space part and the remainder
  vectors_lens_incorrect.pt               minus the "Incorrect" lens direction (so "high" removes it, "low" adds it)
  vectors_lens_control.pt                 the same construction for eight unrelated tokens
  vectors_meanpart.pt / vectors_meanrest.pt  the steering directions split into their projection on the average
                                          activation at the position and the remainder
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.acts import load_acts
from vc.data import SWEEP_LAYERS, read_jsonl
from vc.env import DATA_RAW, MODEL_ID, RESULTS_RAW, load_env
from vc.interp import JLens, Unembed, excess_kurtosis, sparse_decomposition
from vc.prompts import CLASSES, HIGH_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--fmt", default="raw", choices=["raw", "tt"])
ap.add_argument("--k", type=int, default=25)
ap.add_argument("--n-base", type=int, default=10, help="random directions / random trial pairs per baseline")
ap.add_argument("--part", default="all", choices=["all", "collection"],
                help="collection: only redo part 3 (the collection-set readout) and update the existing output file")
args = ap.parse_args()
os.environ["VC_FMT"] = args.fmt
sub = "" if args.fmt == "raw" else "tt"
RES = RESULTS_RAW / sub
OUT = RESULTS_RAW / "interp"
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES / "interp", exist_ok=True)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


load_env()
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
lead = "" if args.fmt == "tt" else " "
cls_ids = [tok.encode(lead + c, add_special_tokens=False)[0] for c in CLASSES]
WORDS = ["certain", "sure", "confident", "correct", "right", "yes", "definitely", "known", "true", "uncertain", "unsure",
         "unknown", "wrong", "incorrect", "maybe", "guess", "doubt", "unlikely", "likely", "no", "not", "false",
         "confidence", "probability", "answer", "actually", "unfortunately", "excellent", "congratulations", "failed",
         "mistaken", "mistakenly", "erroneous", "misguided", "inaccurate", "supposedly", "perfect"]
FAMILY = [" Incorrect", "Incorrect", " incorrect", "incorrect", " wrong", "wrong", " Wrong", "Wrong"]
CONTROL = [" Tuesday", " blue", " river", " seven", " window", " music", " garden", " paper"]
pin = {}
for w in WORDS:
    for form in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        ids = tok.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            pin[form] = ids[0]
pin_names, pin_ids = list(pin), torch.tensor(list(pin.values()))
fam_ids = [pin[w] for w in FAMILY]
ctl_ids = [tok.encode(w, add_special_tokens=False) for w in CONTROL]
assert all(len(x) == 1 for x in ctl_ids)
ctl_ids = [x[0] for x in ctl_ids]

C = torch.load(RES / "interp" / "capture.pt")
trials = read_jsonl(DATA_RAW / sub / "trials.jsonl")
idx, POS = C["idx"], C["pos"]
high = torch.tensor([trials[i]["cls"] in HIGH_CLASSES for i in idx])
n, L = len(idx), C["clean"].shape[1]
dev = "cuda:0"
un, lens = Unembed(dev), JLens(dev)
log(f"lens: {len(lens.layers)} layers, fitted on {lens.n_prompts} prompts; {n} trials ({int(high.sum())} high); pinned tokens: {len(pin_names)}")

# check of the unembedding: class logits of the last block at CC against the model's own
own = un(C["clean"][:, L - 1, POS.index("CC")])[:, cls_ids].cpu()
err = float((own - C["logits"]).abs().max())
log(f"unembedding check: max |lens logits - model logits| over classes = {err:.3f} (logit range {float(C['logits'].max() - C['logits'].min()):.1f})")
assert err < 1.0


def names(ids):
    return [tok.decode([int(i)]) for i in ids]


def top(v, k=15):
    t = v.topk(k)
    return {"tok": names(t.indices), "id": t.indices.tolist(), "val": [round(float(x), 3) for x in t.values]}


V = torch.load(RES / "vectors.pt")
vec_pos = [p for p in ("PANL", "PANL+1", "CC", "A_last") if p in V["unit"] and p in POS]
if args.part == "all":
    # ------------------------------------------------------------------------------------------------------------
    # 1. readouts of real activations
    # ------------------------------------------------------------------------------------------------------------
    cls = torch.zeros(2, L, len(POS), n, 10)
    top1 = torch.zeros(2, L, len(POS), n, dtype=torch.long)
    pin_auroc = torch.zeros(2, L, len(POS), len(pin_ids))
    pin_rank = torch.zeros(2, L, len(POS), 2, len(pin_ids))
    kurt = torch.zeros(2, L, len(POS), n)
    group_top = {}
    for l in range(L):
        for pi, p in enumerate(POS):
            X = C["clean"][:, l, pi]
            for a, use_j in enumerate((True, False)):
                lg = un(lens.transport(X, l, use_j))  # [n, V]
                cls[a, l, pi] = lg[:, cls_ids].cpu()
                top1[a, l, pi] = lg.argmax(-1).cpu()
                pv = lg[:, pin_ids.to(dev)]
                rk = torch.stack([(lg > pv[:, j: j + 1]).sum(-1) for j in range(len(pin_ids))], 1).float().cpu()  # [n, K]
                pin_rank[a, l, pi, 0], pin_rank[a, l, pi, 1] = rk[high].median(0).values, rk[~high].median(0).values
                pin_auroc[a, l, pi] = torch.tensor([roc_auc_score((~high).numpy(), -rk[:, j].numpy()) for j in range(rk.shape[1])])
                kurt[a, l, pi] = excess_kurtosis(lg).cpu()
                c = lg - lg.mean(-1, keepdim=True)
                mh, ml = c[high.to(dev)].mean(0), c[~high.to(dev)].mean(0)
                group_top[(a, l, p)] = {"high": top(mh), "low": top(ml), "high_minus_low": top(mh - ml), "low_minus_high": top(ml - mh)}
                del lg, c
        if l % 10 == 0:
            log(f"readout layer {l}")

    # ------------------------------------------------------------------------------------------------------------
    # 2. the steering directions
    # ------------------------------------------------------------------------------------------------------------
    g = torch.Generator().manual_seed(0)
    vec, parts = {}, {name: {p: torch.zeros(len(SWEEP_LAYERS), un.W.shape[1]) for p in vec_pos}
                      for name in ("jpart", "jpart_neg", "jrest", "jrest_neg")}
    for li, l in enumerate(SWEEP_LAYERS):
        D = lens.dictionary(un, l)  # [V, d]
        dn = D.norm(dim=-1)
        use = dn > 1e-6 * dn.max()  # unused vocabulary rows have (near-)zero unembedding
        dn = torch.where(use, dn, torch.full_like(dn, float("inf")))

        def describe(u):
            """u: unit direction in layer-l coordinates."""
            u = u.to(dev)
            scores = un(lens.transport(u[None], l))[0]
            sel, coef, recon, proj = sparse_decomposition(u, D, dn, args.k)
            return {"top": top(scores), "kurt": float(excess_kurtosis(scores)),
                    "atoms": names(sel), "atom_ids": sel, "coef": [round(float(c * dn[j]), 4) for c, j in zip(coef, sel)],
                    "fve": float(1 - (u - recon).pow(2).sum() / u.pow(2).sum()),
                    "fve_span": float(1 - (u - proj).pow(2).sum() / u.pow(2).sum())}, recon.cpu(), proj.cpu()

        for p in vec_pos:
            u = V["unit"][p][li]
            X = C["clean"][:, l, POS.index(p)].to(dev)
            step = 5 * 0.03 * float(V["norm"][p][li])
            base = un(lens.transport(X, l))
            for sign, tag in ((1.0, "high"), (-1.0, "low")):
                d, recon, proj = describe(sign * u)
                delta = (un(lens.transport(X + sign * step * u.to(dev), l)) - base).mean(0)
                delta = delta - delta.mean()
                d["steered_up"], d["steered_down"] = top(delta), top(-delta)
                d["steered_cls"] = [round(float(x), 3) for x in delta[cls_ids]]
                vec[(p, l, tag)] = d
                # parts for the steering control: J-space part = non-negative reconstruction, rest = the remainder
                parts["jpart" if sign > 0 else "jpart_neg"][p][li] = sign * recon
                parts["jrest" if sign > 0 else "jrest_neg"][p][li] = sign * (sign * u - recon)
            del base
            # baselines: isotropic random directions, and differences between two random trials at this position
            rnd, pair = [], []
            for _ in range(args.n_base):
                r = torch.nn.functional.normalize(torch.randn(un.W.shape[1], generator=g), dim=0)
                d, _, _ = describe(r)
                rnd.append({"fve": d["fve"], "kurt": d["kurt"]})
                i, j = torch.randperm(n, generator=g)[:2].tolist()
                d, _, _ = describe(torch.nn.functional.normalize(X[i] - X[j], dim=0))
                pair.append({"fve": d["fve"], "kurt": d["kurt"]})
            vec[(p, l, "baseline")] = {"random": rnd, "pair": pair}
            log(f"L{l} {p}: fve high {vec[(p, l, 'high')]['fve']:.3f} low {vec[(p, l, 'low')]['fve']:.3f} | random "
                f"{sum(x['fve'] for x in rnd) / len(rnd):.3f} pair {sum(x['fve'] for x in pair) / len(pair):.3f} | "
                f"high atoms: {vec[(p, l, 'high')]['atoms'][:8]}")
        del D

    for name, u in parts.items():
        torch.save({"unit": u, "norm": {p: V["norm"][p] for p in vec_pos}, "layers": SWEEP_LAYERS}, RES / "interp" / f"vectors_{name}.pt")

# ------------------------------------------------------------------------------------------------------------
# 3. the "Incorrect" lens direction: steering inputs, cosine with the steering direction, and its readout on the
#    3,000-trial collection set (stated confidence, correctness of the answer)
# ------------------------------------------------------------------------------------------------------------
A = load_acts(RES / "acts")
keep = np.array([trials[i].get("src", "val") == "val" for i in A.idx])
conf = np.array([trials[i]["conf"] for i in A.idx])[keep]
correct = np.array([bool(trials[i]["correct"]) for i in A.idx])[keep]
fam_pos = [p for p in ("A_last", "PANL", "PANL+1", "CC") if p in A.pos]
lens_vec = {name: {p: torch.zeros(len(SWEEP_LAYERS), un.W.shape[1]) for p in vec_pos} for name in ("lens_incorrect", "lens_control")}
family = {"tokens": FAMILY, "control": CONTROL, "cos_steer": {}, "collection": {}, "mean_check": {}}
# The steering direction is a difference of class means at one prompt position, and a large part of it lies along
# the AVERAGE activation at that position (the cosine reaches 0.8 and changes sign from layer to layer). Split:
# mean part = its projection on the calibration-set mean activation, rest = the remainder.
mean_pos = [p for p in vec_pos if p in V["mean"]]
mean_vec = {name: {p: torch.zeros(len(SWEEP_LAYERS), un.W.shape[1]) for p in mean_pos} for name in ("meanpart", "meanrest")}
for li, l in enumerate(SWEEP_LAYERS):
    Jm = lens.matrix(l) if l in lens.J else torch.eye(un.W.shape[1], device=dev)
    d = {}
    for name, ids in (("lens_incorrect", fam_ids), ("lens_control", ctl_ids)):
        v = torch.nn.functional.normalize(un.W[ids] @ Jm, dim=-1).mean(0)
        d[name] = torch.nn.functional.normalize(v, dim=0)
        for p in vec_pos:
            lens_vec[name][p][li] = -d[name].cpu()
    for p in vec_pos:
        family["cos_steer"][(p, l)] = {name: float(V["unit"][p][li].to(dev) @ d[name]) for name in d}
    for p in mean_pos:
        m = torch.nn.functional.normalize(V["mean"][p][li].to(dev), dim=0)
        u = V["unit"][p][li].to(dev)
        a = float(u @ m)
        mean_vec["meanpart"][p][li], mean_vec["meanrest"][p][li] = (a * m).cpu(), (u - a * m).cpu()
        sc = un(lens.transport(torch.nn.functional.normalize(u - a * m, dim=0)[None], l))[0]
        family["mean_check"][(p, l)] = {"cos_mean": a, "rest_high": top(sc), "rest_low": top(-sc), "rest_kurt": float(excess_kurtosis(sc)),
                                        "mean_top": top(un(lens.transport(m[None], l))[0]),
                                        "cos_incorrect_rest": float(torch.nn.functional.normalize(u - a * m, dim=0) @ d["lens_incorrect"]),
                                        "cos_incorrect_mean": float(m @ d["lens_incorrect"])}
    for p in fam_pos:
        X = torch.as_tensor(A.get(li, p)[keep])
        sc, sc1 = [], []
        for i in range(0, len(X), 500):
            lg = un(lens.transport(X[i: i + 500], l))
            rk = torch.stack([(lg > lg[:, j: j + 1]).sum(-1) for j in fam_ids], 1).float()  # vocabulary rank of each family token
            sc.append(-torch.log1p(rk).mean(-1).cpu())  # high = the family is ranked near the top
            sc1.append(-rk[:, 0].cpu())  # the single token " Incorrect"
        sc, sc1 = torch.cat(sc).numpy(), torch.cat(sc1).numpy()
        family["collection"][(p, l)] = {"r_conf": float(spearmanr(sc, conf)[0]), "r_conf_token": float(spearmanr(sc1, conf)[0]),
                                        "auroc_wrong_answer": float(roc_auc_score(~correct, sc)),
                                        "auroc_wrong_answer_token": float(roc_auc_score(~correct, sc1)),
                                        "auroc_low_conf": float(roc_auc_score(conf < 0.5, sc)),
                                        "auroc_low_conf_token": float(roc_auc_score(conf < 0.5, sc1))}
    log(f"family L{l}: " + " | ".join(f"{p} r={family['collection'][(p, l)]['r_conf']:.2f} wrong-answer auroc "
                                       f"{family['collection'][(p, l)]['auroc_wrong_answer']:.2f}" for p in fam_pos)
        + " | cos with steering direction: " + ", ".join(f"{p} {family['cos_steer'][(p, l)]['lens_incorrect']:.3f}" for p in vec_pos))
for name, u in lens_vec.items():
    torch.save({"unit": u, "norm": {p: V["norm"][p] for p in vec_pos}, "layers": SWEEP_LAYERS}, RES / "interp" / f"vectors_{name}.pt")
for name, u in mean_vec.items():
    torch.save({"unit": u, "norm": {p: V["norm"][p] for p in mean_pos}, "layers": SWEEP_LAYERS}, RES / "interp" / f"vectors_{name}.pt")
family["stated_conf_auroc_wrong_answer"] = float(roc_auc_score(~correct, -conf))
if args.part == "collection":
    R = torch.load(OUT / f"jlens_{args.fmt}.pt")
    R["family"] = family
    torch.save(R, OUT / f"jlens_{args.fmt}.pt")
else:
    torch.save({"idx": idx, "pos": POS, "high": high, "cls_ids": cls_ids, "pin_names": pin_names, "pin_ids": pin_ids.tolist(),
                "cls": cls, "top1": top1, "pin_auroc": pin_auroc, "pin_rank": pin_rank, "kurt": kurt, "group_top": group_top, "vec": vec,
                "family": family,
                "sweep_layers": SWEEP_LAYERS, "k": args.k, "n_prompts": lens.n_prompts}, OUT / f"jlens_{args.fmt}.pt")
log("INTERP JLENS DONE")
