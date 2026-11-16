"""Sparse autoencoder features on the confidence prompt: Gemma Scope 2, residual stream after block 31 of
Gemma 3 27B IT, 1,048,576 features, JumpReLU (google/gemma-scope-2-27b-it,
resid_post/layer_31_width_1m_l0_big). One GPU job (two cards: encoder on one, decoder on the other); no model
forward passes.

    python scripts/interp_sae.py --fmt raw|tt

Inputs: the stored block-31 activations of the 3,000-trial collection set (results[/tt]/acts), the steering
directions (vectors.pt), and results[/tt]/interp/capture.pt (steered activations, full token sequences).

Writes results/interp/sae_<fmt>.json:
  check      reconstruction quality and number of active features per position
  stats      per position: features whose activation follows the stated confidence (correlation with confidence,
             with correctness and with the answer log-probability; activation rate; class means), the most
             active features, and how well k features predict confidence (cross-validated)
  vector     the steering direction at block 31 (PANL and CC): share reconstructed by the SAE and by its top-k
             features, the features with the largest contribution, cosine with decoder directions
  steered    change of feature activations when the direction is added upstream, against the natural
             high-minus-low difference
  features   for every feature named above: Jacobian-lens and logit-lens tokens of its decoder direction, lens
             kurtosis, where in the prompt it is active, and its top-activating corpus examples and promoted
             tokens from the SAE's examples.safetensors (read over HTTP)
and results[/tt]/interp/vectors_sae*.pt, the SAE split of the steering directions for the steering control.
"""
import argparse
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.acts import load_acts
from vc.data import SWEEP_LAYERS, read_jsonl
from vc.env import DATA_RAW, MODEL_ID, RESULTS_RAW, SAE_DIR, SAE_LAYER, SAE_REPO, load_env
from vc.interp import JLens, Unembed, excess_kurtosis
from vc.prompts import HIGH_CLASSES, LOW_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--fmt", default="raw", choices=["raw", "tt"])
ap.add_argument("--n-examples", type=int, default=10)
args = ap.parse_args()
sub = "" if args.fmt == "raw" else "tt"
RES, OUT = RESULTS_RAW / sub, RESULTS_RAW / "interp"
os.makedirs(OUT, exist_ok=True)
D0, D1 = "cuda:0", "cuda:1"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


load_env()
from huggingface_hub import HfFileSystem, hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
cfg = json.load(open(hf_hub_download(SAE_REPO, f"{SAE_DIR}/config.json")))
assert cfg["hf_hook_point_in"] == f"model.layers.{SAE_LAYER}.output" and cfg["architecture"] == "jump_relu"
with safe_open(hf_hub_download(SAE_REPO, f"{SAE_DIR}/params.safetensors"), "pt") as f:
    W_enc = f.get_tensor("w_enc").to(D0)  # [d, F]
    b_enc, thr = f.get_tensor("b_enc").to(D0), f.get_tensor("threshold").to(D0)
    W_dec = f.get_tensor("w_dec").to(D1)  # [F, d]
    b_dec = f.get_tensor("b_dec")
F = W_dec.shape[0]
log(f"SAE loaded: {F} features, target L0 {cfg['l0']}")


@torch.no_grad()
def encode(X, bs=192, cols=None):
    """JumpReLU activations of X [n, d]. Returns a scipy CSR matrix [n, F] (or dense [n, len(cols)] for a
    subset of features)."""
    X = torch.as_tensor(X, dtype=torch.float32)
    if cols is not None:
        c = torch.as_tensor(cols, device=D0)
        pre = X.to(D0) @ W_enc[:, c] + b_enc[c]
        return torch.where(pre > thr[c], pre, torch.zeros_like(pre)).cpu().numpy()
    r, cc, v = [], [], []
    for i in range(0, len(X), bs):
        pre = X[i: i + bs].to(D0) @ W_enc + b_enc
        nz = (pre > thr).nonzero()
        r.append((nz[:, 0] + i).cpu())
        cc.append(nz[:, 1].cpu())
        v.append(pre[nz[:, 0], nz[:, 1]].cpu())
    return sp.csr_matrix((torch.cat(v).numpy(), (torch.cat(r).numpy(), torch.cat(cc).numpy())), shape=(len(X), F))


@torch.no_grad()
def decode(A, with_bias=True):
    """Reconstruction of a CSR activation matrix -> float32 [n, d]."""
    out = torch.zeros(A.shape[0], W_dec.shape[1])
    for i in range(0, A.shape[0], 64):
        a = A[i: i + 64].tocoo()
        rec = torch.zeros(a.shape[0], W_dec.shape[1], device=D1)
        rec.index_add_(0, torch.as_tensor(a.row, device=D1, dtype=torch.long),
                       W_dec[torch.as_tensor(a.col, device=D1, dtype=torch.long)] * torch.as_tensor(a.data, device=D1)[:, None])
        out[i: i + 64] = rec.cpu()
    return out + b_dec if with_bias else out


def fve(X, R):
    """Share of the between-trial variance at one position that the reconstruction keeps (the mean
    reconstruction error over trials, a constant offset, is removed first)."""
    X, R = torch.as_tensor(X), torch.as_tensor(R)
    E = (X - R) - (X - R).mean(0)
    return float(1 - E.pow(2).sum() / (X - X.mean(0)).pow(2).sum())


# ------------------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------------------
trials = read_jsonl(DATA_RAW / sub / "trials.jsonl")
sets = json.load(open(DATA_RAW / sub / "sets.json"))
A = load_acts(RES / "acts")
li = A.layers.index(SAE_LAYER)
keep = np.array([trials[i].get("src", "val") == "val" for i in A.idx])  # two-turn: without the added reservoir trials
T = [trials[i] for i in A.idx]
conf = np.array([t["conf"] for t in T])
cls = np.array([t["cls"] for t in T])
correct = np.array([float(t["correct"]) for t in T])
lpm = np.array([t["lp"]["mean"] for t in T])
POS = [p for p in ["PANL", "PANL+1", "UNL", "REQ0", "A_last", "AC", "INT4", "CC"] if p in A.pos]
res = {"n_features": F, "n_trials": int(keep.sum()), "check": {}, "stats": {}, "vector": {}, "steered": {}, "features": {}}

acts, X31 = {}, {}
for p in POS:
    X31[p] = A.get(li, p)
    acts[p] = encode(X31[p])
    rec = decode(acts[p])
    res["check"][p] = {"fve": fve(X31[p], rec), "l0_mean": float(acts[p].getnnz(1).mean()),
                       "between_trial_sd_rel": float((torch.as_tensor(X31[p]) - torch.as_tensor(X31[p]).mean(0)).norm(dim=1).div(torch.as_tensor(X31[p]).norm(dim=1)).mean()),
                       "rel_err": float((torch.as_tensor(X31[p]) - rec).norm(dim=1).div(torch.as_tensor(X31[p]).norm(dim=1)).mean()),
                       "cos": float(torch.nn.functional.cosine_similarity(torch.as_tensor(X31[p]), rec).mean())}
    log(p, res["check"][p])
# the alternative convention (decoder bias subtracted from the input) as a check that the one above is right
alt = encode(X31["PANL"] - b_dec.numpy())
res["check"]["PANL_input_minus_b_dec"] = {"fve": fve(X31["PANL"], decode(alt)), "l0_mean": float(alt.getnnz(1).mean())}
log("alternative encoder input convention:", res["check"]["PANL_input_minus_b_dec"])

# ------------------------------------------------------------------------------------------------------------
# features that follow the stated confidence
# ------------------------------------------------------------------------------------------------------------
selected = {}


def pick(fid, why):
    selected.setdefault(int(fid), []).append(why)


def corr_cols(M, y):
    yc = (y - y.mean()) / y.std()
    Mc = M - M.mean(0)
    sd = Mc.std(0)
    return (Mc * yc[:, None]).mean(0) / np.where(sd > 0, sd, np.inf)


def cv_r2(M, y, ks=(1, 5, 20, 100), folds=5, seed=0):
    """Confidence from the k features with the largest |correlation| in the training fold (ridge, alpha = 1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    out = {k: [] for k in ks}
    for f in range(folds):
        te = perm[f::folds]
        tr = np.setdiff1d(perm, te)
        r = np.abs(corr_cols(M[tr], y[tr]))
        order = np.argsort(-r)
        for k in ks:
            c = order[:k]
            mu, sd = M[tr][:, c].mean(0), M[tr][:, c].std(0) + 1e-8
            Xtr, Xte = (M[tr][:, c] - mu) / sd, (M[te][:, c] - mu) / sd
            w = np.linalg.solve(Xtr.T @ Xtr + np.eye(len(c)), Xtr.T @ (y[tr] - y[tr].mean()))
            pred = Xte @ w + y[tr].mean()
            out[k].append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return {str(k): float(np.mean(v)) for k, v in out.items()}


hi_m, lo_m = np.isin(cls, HIGH_CLASSES) & keep, np.isin(cls, LOW_CLASSES) & keep
for p in POS:
    Ap = acts[p][keep]
    cnt = np.asarray((Ap > 0).sum(0)).ravel()
    cols = np.where(cnt >= 0.01 * Ap.shape[0])[0]
    M = np.asarray(Ap[:, cols].todense(), dtype=np.float64)
    y = conf[keep]
    r_conf, r_corr, r_lp = corr_cols(M, y), corr_cols(M, correct[keep]), corr_cols(M, lpm[keep])
    Mh, Ml = np.asarray(acts[p][hi_m][:, cols].mean(0)).ravel(), np.asarray(acts[p][lo_m][:, cols].mean(0)).ravel()

    def row(j):
        return {"feature": int(cols[j]), "r_conf": float(r_conf[j]), "r_correct": float(r_corr[j]), "r_logprob": float(r_lp[j]),
                "rate": float(cnt[cols[j]] / Ap.shape[0]), "mean_high": float(Mh[j]), "mean_low": float(Ml[j]),
                "rate_high": float((acts[p][hi_m][:, cols[j]] > 0).mean()), "rate_low": float((acts[p][lo_m][:, cols[j]] > 0).mean())}

    order = np.argsort(r_conf)
    meanact = np.asarray(Ap[:, cols].mean(0)).ravel()
    st = {"n_candidates": int(len(cols)), "neg": [row(j) for j in order[:20]], "pos": [row(j) for j in order[::-1][:20]],
          "most_active": [row(j) for j in np.argsort(-meanact)[:20]], "cv_r2": cv_r2(M, y),
          "n_abs_r_over_0.3": int((np.abs(r_conf) > 0.3).sum()), "n_abs_r_over_0.5": int((np.abs(r_conf) > 0.5).sum())}
    res["stats"][p] = st
    npick = 15 if p in ("PANL", "CC") else 6
    for k in ("neg", "pos"):
        for d in st[k][:npick]:
            pick(d["feature"], f"{p}:{k}")
    for d in st["most_active"][:12 if p == "PANL" else 4]:
        pick(d["feature"], f"{p}:active")
    log(p, "candidates", len(cols), "| top + :", [(d["feature"], round(d["r_conf"], 2)) for d in st["pos"][:5]],
        "| top - :", [(d["feature"], round(d["r_conf"], 2)) for d in st["neg"][:5]], "| cv r2", st["cv_r2"])

# ------------------------------------------------------------------------------------------------------------
# the steering direction at block 31
# ------------------------------------------------------------------------------------------------------------
V = torch.load(RES / "vectors.pt")
hi, lo = A.rows(sets["vec_high"]), A.rows(sets["vec_low"])
KS = [1, 5, 10, 25, 50, 100, 250]
parts = {name: {} for name in ("saepart", "saerest", "saefull", "saeerr")}
for p in [q for q in ("PANL", "CC") if q in POS]:
    X = torch.as_tensor(X31[p])
    raw = X[hi].mean(0) - X[lo].mean(0)
    u = raw / raw.norm()
    assert float((u - V["unit"][p][li]).abs().max()) < 1e-3
    dA = np.asarray(acts[p][hi].mean(0) - acts[p][lo].mean(0)).ravel()  # feature-wise difference of means
    nz = np.where(dA != 0)[0]
    Wn = W_dec[torch.as_tensor(nz, device=D1)]  # [m, d]
    contrib = torch.as_tensor(dA[nz], device=D1, dtype=torch.float32)[:, None] * Wn  # each feature's share of the difference
    full = contrib.sum(0).cpu()
    # ranking: projection of each feature's contribution on the direction
    score = (contrib @ u.to(D1)).cpu().numpy() / float(raw.norm())
    order = np.argsort(-np.abs(score))
    cum = {}
    for k in KS + [len(nz)]:
        part = contrib[torch.as_tensor(order[:k].copy(), device=D1)].sum(0).cpu()
        cum[str(k)] = {"fve": float(1 - (raw - part).pow(2).sum() / raw.pow(2).sum()),
                       "cos": float(torch.nn.functional.cosine_similarity(raw, part, dim=0))}
    cosd = (torch.nn.functional.normalize(W_dec, dim=-1) @ u.to(D1)).cpu()
    tc, bc = cosd.topk(15), (-cosd).topk(15)
    top = [{"feature": int(nz[j]), "share": float(score[j]), "delta_act": float(dA[nz[j]]),
            "cos": float(cosd[nz[j]])} for j in order[:30]]
    res["vector"][p] = {"norm_raw": float(raw.norm()), "n_features_differ": int(len(nz)), "cumulative": cum, "top": top,
                        "cos_top": [{"feature": int(i), "cos": float(c)} for c, i in zip(tc.values, tc.indices)],
                        "cos_bottom": [{"feature": int(i), "cos": float(-c)} for c, i in zip(bc.values, bc.indices)]}
    for d in top[:20]:
        pick(d["feature"], f"vec {p}:{'+' if d['share'] > 0 else '-'}")
    for d in res["vector"][p]["cos_top"][:6] + res["vector"][p]["cos_bottom"][:6]:
        pick(d["feature"], f"vec {p}:cos")
    k25 = contrib[torch.as_tensor(order[:25].copy(), device=D1)].sum(0).cpu()
    for name, v in (("saepart", k25), ("saerest", raw - k25), ("saefull", full), ("saeerr", raw - full)):
        t = torch.zeros(len(SWEEP_LAYERS), raw.shape[0])
        t[li] = v / raw.norm()  # in units of the steering direction, so that scripts/interv.py scales it the same way
        parts[name][p] = t
    log(f"vector {p}: SAE reconstructs {cum[str(len(nz))]['fve']:.3f} of the direction (cos {cum[str(len(nz))]['cos']:.3f}); "
        f"top-25 features {cum['25']['fve']:.3f}; top shares {[(d['feature'], round(d['share'], 3)) for d in top[:6]]}")
os.makedirs(RES / "interp", exist_ok=True)
for name, u_ in parts.items():
    torch.save({"unit": u_, "norm": {p: V["norm"][p] for p in u_}, "layers": SWEEP_LAYERS}, RES / "interp" / f"vectors_{name}.pt")

# ------------------------------------------------------------------------------------------------------------
# steering upstream: do the features move like the natural high-minus-low difference?
# ------------------------------------------------------------------------------------------------------------
C = torch.load(RES / "interp" / "capture.pt")
ctr = [trials[i] for i in C["idx"]]
c_hi = np.array([t["cls"] in HIGH_CLASSES for t in ctr])
ri = C["read_layers"].index(SAE_LAYER)
for q in C["steer_q"]:
    clean = encode(C["clean"][:, SAE_LAYER, C["pos"].index(q)])
    nat = np.asarray(clean[c_hi].mean(0) - clean[~c_hi].mean(0)).ravel()
    out = {}
    for vi, (kind, scale) in enumerate(C["variants"]):
        st = encode(C["steer"][:, vi, ri, C["steer_q"].index(q)])
        rows = ~c_hi if kind == "high" else (c_hi if kind == "low" else np.ones(len(c_hi), bool))
        d = np.asarray(st[rows].mean(0) - clean[rows].mean(0)).ravel()
        target = nat if kind != "low" else -nat
        m = (d != 0) | (nat != 0)
        out[f"{kind}{scale}"] = {"corr_with_natural": float(np.corrcoef(d[m], target[m])[0, 1]), "n_features": int(m.sum()),
                                 "l0_steered": float(st.getnnz(1).mean()), "l0_clean": float(clean.getnnz(1).mean()),
                                 "top_up": [{"feature": int(j), "delta": float(d[j]), "natural": float(target[j])} for j in np.argsort(-d)[:10]],
                                 "top_down": [{"feature": int(j), "delta": float(d[j]), "natural": float(target[j])} for j in np.argsort(d)[:10]]}
    res["steered"][q] = out
    log("steered", q, {k: round(v["corr_with_natural"], 3) for k, v in out.items()})

# ------------------------------------------------------------------------------------------------------------
# describing the selected features
# ------------------------------------------------------------------------------------------------------------
fids = sorted(selected)
log(f"{len(fids)} features to describe")
un, lens = Unembed(D1), JLens(D1)
Wsel = W_dec[torch.as_tensor(fids, device=D1)]


def toks(v, k=10):
    t = v.topk(k)
    return [tok.decode([int(i)]) for i in t.indices]


jl = un(lens.transport(Wsel, SAE_LAYER))
ll = un(Wsel)
kj = excess_kurtosis(jl).cpu()
# reference distribution of the lens kurtosis: 4,000 random features
g = torch.Generator().manual_seed(0)
rnd = torch.randperm(F, generator=g)[:4000]
kref = torch.cat([excess_kurtosis(un(lens.transport(W_dec[rnd[i: i + 200].to(D1)], SAE_LAYER))).cpu() for i in range(0, 4000, 200)])
res["kurtosis_reference"] = {"quantiles": {str(q): float(kref.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}}

# where in the prompt the features are active: all tokens of 12 high- and 12 low-confidence trials
Ffull = C["full"][SAE_LAYER]
maps = []
for k, n_tok in enumerate(C["full_len"]):
    maps.append(encode(Ffull[k, :n_tok], cols=fids))  # [T, n_sel]
fp = C["full_pos"]

for n_, fid in enumerate(fids):
    where = {}
    for name in ("AC", "A_last", "PANL", "PANL+1", "INT4", "CC"):
        where[name] = float(np.mean([maps[k][fp[k][name], n_] for k in range(len(maps))]))
    peak = [int(np.argmax(maps[k][:, n_])) for k in range(len(maps))]
    res["features"][str(fid)] = {
        "why": selected[fid], "jlens_top": toks(jl[n_]), "jlens_bottom": toks(-jl[n_]), "logit_top": toks(ll[n_]),
        "logit_bottom": toks(-ll[n_]), "lens_kurtosis": float(kj[n_]),
        "kurtosis_percentile": float((kref < kj[n_]).float().mean()),
        "act_at": where, "frac_tokens_active": float(np.mean([(maps[k][:, n_] > 0).mean() for k in range(len(maps))])),
        "peak_token": [tok.decode([C["full_ids"][k][t]]) for k, t in enumerate(peak)][:6],
        "peak_rel_panl": [t - fp[k]["PANL"] for k, t in enumerate(peak)][:6],
    }

# top-activating corpus examples and promoted tokens, read over HTTP from the SAE's examples file
fs = HfFileSystem()
path = f"{SAE_REPO}/{SAE_DIR}/examples.safetensors"
hlen = struct.unpack("<Q", fs.cat_file(path, start=0, end=8))[0]
hdr = json.loads(fs.cat_file(path, start=8, end=8 + hlen))
DT = {"I64": np.int64, "I32": np.int32, "F32": np.float32, "F64": np.float64}


def read_row(name, i):
    h = hdr[name]
    width = int(np.prod(h["shape"][1:])) if len(h["shape"]) > 1 else 1
    size = np.dtype(DT[h["dtype"]]).itemsize * width
    s = 8 + hlen + h["data_offsets"][0] + i * size
    for attempt in range(5):
        try:
            return np.frombuffer(fs.cat_file(path, start=s, end=s + size), dtype=DT[h["dtype"]])
        except Exception as e:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            err = e
    raise err


def describe_remote(fid):
    a, s, ps = read_row("activations", fid), read_row("seq_ids", fid), read_row("positions", fid)
    order = np.argsort(-a)[: args.n_examples]
    ex = []
    for j in order:
        if a[j] <= 0:
            continue
        seq = read_row("tokens", int(s[j]))
        t = int(ps[j])
        ex.append({"act": float(a[j]), "before": tok.decode(seq[max(0, t - 40): t].tolist()), "token": tok.decode([int(seq[t])]),
                   "after": tok.decode(seq[t + 1: t + 9].tolist())})
    return fid, {"examples": ex, "n_examples_stored": int((a > 0).sum()),
                 "frequency": float(read_row("feature_frequencies", fid)[0]),
                 "top_tokens": [tok.decode([int(i)]) for i in read_row("top_tokens", fid)],
                 "bottom_tokens": [tok.decode([int(i)]) for i in read_row("bottom_tokens", fid)]}


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for fid, d in ex.map(describe_remote, fids):
        res["features"][str(fid)].update(d)
log(f"corpus examples read in {time.time() - t0:.0f}s")
json.dump(res, open(OUT / f"sae_{args.fmt}.json", "w"), indent=1, ensure_ascii=False)
log("INTERP SAE DONE")
