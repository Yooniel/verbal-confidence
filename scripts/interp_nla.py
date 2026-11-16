"""Natural Language Autoencoder (Fraser-Taliente et al. 2026) on the confidence prompt: verbalize block-41
residuals of Gemma 3 27B with the released activation verbalizer (AV) and score every explanation with the
released activation reconstructor (AR).

    python scripts/interp_nla.py --fmt raw|tt [--stage av|ar|both]

Inputs: results[/tt]/interp/capture.pt (scripts/interp_capture.py). Output: results/interp/nla_<fmt>.jsonl, one
row per explanation, and results/interp/nla_<fmt>_ref.txt (pipeline check against the example published with
the models).

Protocol (nla-inference README, nla_meta.yaml): the AV prompt is the sidecar's template in the chat format; the
embedding of its placeholder token is replaced by the activation rescaled to L2 norm 60000; the explanation is
the text inside <explanation> tags. The AR reads "Summary of the following text: <text>...</text> <summary>"
(with <bos>), and a linear head on the block-41 output of the last token gives the reconstruction; both vectors
are compared by direction (cosine; fve = 1 - 2(1 - cos) / 0.0579, the published normalisation for this model).

What is verbalized (see build_items): the tokens around the answer of high- and low-confidence trials, the same
tokens after steering, class means and the steering direction, a window of tokens, and the reference prompt.
"""
import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import read_jsonl
from vc.env import DATA_RAW, INTERP_CACHE, NLA_AR, NLA_AV, NLA_LAYER, RESULTS_RAW, load_env
from vc.prompts import HIGH_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--fmt", default="raw", choices=["raw", "tt"])
ap.add_argument("--stage", default="both", choices=["av", "ar", "both"])
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N items")
args = ap.parse_args()

N_MAIN, N_STEER, N_OOD, N_WIN, N_SAMPLES = 60, 40, 20, 3, 2
FVE_DENOM = 0.0579  # Var of unit-norm block-41 activations on the NLA training data (published with the model)
OUT = RESULTS_RAW / "interp"
os.makedirs(OUT, exist_ok=True)
out_file = OUT / f"nla_{args.fmt}.jsonl"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def build_items(fmt):
    """Deterministic list of (metadata, vector) to verbalize."""
    sub = "" if fmt == "raw" else "tt"
    C = torch.load(RESULTS_RAW / sub / "interp" / "capture.pt")
    trials = read_jsonl(DATA_RAW / sub / "trials.jsonl")
    idx, pos = C["idx"], C["pos"]
    cls = [trials[i]["cls"] for i in idx]
    high = [j for j in range(len(idx)) if cls[j] in HIGH_CLASSES]
    low = [j for j in range(len(idx)) if cls[j] not in HIGH_CLASSES]
    items = []

    def add(kind, vec, j=None, **kw):
        m = {"fmt": fmt, "kind": kind, **kw}
        if j is not None:
            m.update(trial=idx[j], cls=cls[j], group="high" if cls[j] in HIGH_CLASSES else "low")
        items.append((m, vec.float().clone()))

    if fmt == "raw":  # pipeline check: tokens 4-11 of the reference prompt, greedy
        for t in range(4, 12):
            add("ref", C["ref"][t], tok=t, layer=NLA_LAYER, sample=0)
    main = high[:N_MAIN] + low[:N_MAIN]
    for p in ["PANL", "A_last", "CC", "PANL+1", "INT4"]:
        for j in main:
            add("clean", C["clean"][j, NLA_LAYER, pos.index(p)], j, pos=p, layer=NLA_LAYER, sample=0)
    for s in range(1, N_SAMPLES + 1):  # temperature-1 samples (the setting the NLA was trained with)
        for j in main:
            add("clean", C["clean"][j, NLA_LAYER, pos.index("PANL")], j, pos="PANL", layer=NLA_LAYER, sample=s)
    for layer in (25, 31):  # away from the layer the NLA was trained on: exploratory
        for j in high[:N_OOD] + low[:N_OOD]:
            add("clean", C["clean"][j, layer, pos.index("PANL")], j, pos="PANL", layer=layer, sample=0)
    ri = C["read_layers"].index(NLA_LAYER)
    for vi, (kind, scale) in enumerate(C["variants"]):
        if scale != 5:
            continue
        for q in ("PANL", "CC"):
            for j in high[:N_STEER] + low[:N_STEER]:
                add("steer", C["steer"][j, vi, ri, C["steer_q"].index(q)], j, pos=q, layer=NLA_LAYER, sample=0,
                    variant=f"{kind}{scale}")
    for p in ("PANL", "CC"):  # class means and the high-minus-low direction at block 41
        X = C["clean"][:, NLA_LAYER, pos.index(p)]
        mh, ml = X[high].mean(0), X[low].mean(0)
        for name, v in (("mean_high", mh), ("mean_low", ml), ("high_minus_low", mh - ml), ("low_minus_high", ml - mh)):
            for s in range(0, 4):
                add("aggregate", v, pos=p, layer=NLA_LAYER, sample=s, name=name)
    F = C["full"][NLA_LAYER]  # a window of tokens: answer colon ... two tokens after PANL
    for k in list(range(N_WIN)) + list(range(len(C["full_idx"]) // 2, len(C["full_idx"]) // 2 + N_WIN)):
        fp, j = C["full_pos"][k], idx.index(C["full_idx"][k])
        for t in range(fp["AC"] - 2, fp["PANL"] + 3):
            add("window", F[k, t], j, tok=t, rel=t - fp["PANL"], token_id=C["full_ids"][k][t], layer=NLA_LAYER, sample=0)
    for n, (m, _) in enumerate(items):
        m["item"] = n
    return items


def parse_explanation(text):
    """Text inside the first <explanation> region (as the reference client); the raw text if there is none."""
    m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
    if m:
        return m.group(1), "ok"
    m = re.search(r"<explanation>\s*(.*)", text, re.DOTALL)
    return (m.group(1).strip(), "unclosed") if m else (text.strip(), "missing")


items = build_items(args.fmt)
if args.limit:
    items = items[: args.limit]
log(f"{len(items)} items for format {args.fmt}")
load_env()

# =========================================================================================================
if args.stage in ("av", "both"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    av_dir = INTERP_CACHE / NLA_AV.split("/")[1]
    meta = yaml.safe_load(open(av_dir / "nla_meta.yaml"))
    assert meta["role"] == "av" and meta["extraction_layer_index"] == NLA_LAYER
    tk, scale = meta["tokens"], float(meta["extraction"]["injection_scale"])
    tok = AutoTokenizer.from_pretrained(av_dir)
    assert tok.encode(tk["injection_char"], add_special_tokens=False) == [tk["injection_token_id"]]
    enc = tok.apply_chat_template(
        [{"role": "user", "content": meta["prompt_templates"]["av"].format(injection_char=tk["injection_char"])}],
        tokenize=True, add_generation_prompt=True)
    ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
    (ipos,) = [i for i, t in enumerate(ids) if t == tk["injection_token_id"]]
    assert ids[ipos - 1] == tk["injection_left_neighbor_id"] and ids[ipos + 1] == tk["injection_right_neighbor_id"]
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id
    av = AutoModelForCausalLM.from_pretrained(av_dir, dtype=torch.bfloat16, device_map="auto",
                                              max_memory={0: "30GiB", 1: "44GiB"}).eval()
    emb = av.get_input_embeddings()
    dev = emb.weight.device
    with torch.no_grad():
        base = emb(torch.tensor([ids], device=dev))  # [1, T, d], already scaled by sqrt(d) in Gemma 3
    assert abs(float(base[0, 1].float().norm()) / float(emb.weight[ids[1]].float().norm()) - av.config.hidden_size ** 0.5) < 0.5
    log(f"AV loaded; prompt {len(ids)} tokens, placeholder at {ipos}; injection norm {scale:.0f}; "
        f"median prompt-embedding norm {float(base[0].float().norm(dim=-1).median()):.0f}")
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]

    done = {json.loads(l)["item"] for l in open(out_file)} if out_file.exists() else set()
    todo = [(m, v) for m, v in items if m["item"] not in done]
    f = open(out_file, "a")
    t0 = time.time()
    for sampled in (False, True):
        part = [(m, v) for m, v in todo if (m["sample"] > 0) == sampled]
        for b0 in range(0, len(part), args.bs):
            chunk = part[b0: b0 + args.bs]
            vec = torch.stack([v for _, v in chunk]).double()
            vec = vec / vec.norm(dim=-1, keepdim=True) * scale
            e = base.repeat(len(chunk), 1, 1)
            e[:, ipos] = vec.to(dev, e.dtype)
            torch.manual_seed(1000 + b0)
            kw = dict(do_sample=True, temperature=1.0, top_k=None, top_p=None) if sampled else dict(do_sample=False, top_k=None, top_p=None, temperature=None)
            with torch.no_grad():
                out = av.generate(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long, device=dev),
                                  max_new_tokens=256, eos_token_id=stop, pad_token_id=tok.pad_token_id, **kw)
            for (m, v), row in zip(chunk, out):
                raw = tok.decode(row, skip_special_tokens=True)
                text, status = parse_explanation(raw)
                f.write(json.dumps({**m, "norm": float(v.norm()), "text": text, "status": status,
                                    "n_tokens": int((row != tok.pad_token_id).sum())}) + "\n")
            f.flush()
            log(f"{'sampled' if sampled else 'greedy'} {b0 + len(chunk)}/{len(part)}: {time.time() - t0:.0f}s")
    f.close()
    del av, emb, base
    gc.collect()
    torch.cuda.empty_cache()

# =========================================================================================================
if args.stage in ("ar", "both"):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ar_dir = snapshot_download(NLA_AR)
    meta = yaml.safe_load(open(os.path.join(ar_dir, "nla_meta.yaml")))
    assert meta["role"] == "ar" and abs(meta["extraction"]["mse_scale"] - 5376 ** 0.5) < 1e-6
    tok = AutoTokenizer.from_pretrained(ar_dir)
    ar = AutoModelForCausalLM.from_pretrained(ar_dir, dtype=torch.bfloat16, device_map="auto",
                                              max_memory={0: "30GiB", 1: "44GiB"}).eval()
    assert ar.config.num_hidden_layers == NLA_LAYER + 1
    ar.model.norm = torch.nn.Identity()  # the head reads the raw output of block 41
    head = torch.nn.Linear(ar.config.hidden_size, ar.config.hidden_size, bias=False, dtype=torch.bfloat16)
    head.load_state_dict(load_file(os.path.join(ar_dir, "value_head.safetensors")))
    head = head.to(ar.model.layers[-1].mlp.down_proj.weight.device).eval()
    suffix = meta["tokens"]["critic_suffix_ids"]
    rows = [json.loads(l) for l in open(out_file)]
    vecs = {m["item"]: v for m, v in items}
    rows = [r for r in rows if r["item"] in vecs]
    enc = [tok(meta["prompt_templates"]["ar"].format(explanation=r["text"]), add_special_tokens=True)["input_ids"] for r in rows]
    assert all(e[0] == tok.bos_token_id for e in enc)
    bad = sum(e[-len(suffix):] != suffix for e in enc)
    log(f"AR loaded; {len(rows)} explanations; {bad} prompts whose tail differs from the sidecar's suffix")
    pred = torch.zeros(len(rows), ar.config.hidden_size)
    order = sorted(range(len(rows)), key=lambda i: len(enc[i]))
    for b0 in range(0, len(order), 16):
        b = order[b0: b0 + 16]
        T = max(len(enc[i]) for i in b)
        ids = torch.full((len(b), T), tok.pad_token_id, dtype=torch.long)
        for k, i in enumerate(b):
            ids[k, : len(enc[i])] = torch.tensor(enc[i])
        mask = (ids != tok.pad_token_id).long()
        with torch.no_grad():
            h = ar.model(input_ids=ids.to(ar.device), attention_mask=mask.to(ar.device), use_cache=False).last_hidden_state
            last = torch.tensor([len(enc[i]) - 1 for i in b], device=h.device)
            p = head(h[torch.arange(len(b), device=h.device), last].to(head.weight.device))
        pred[b] = p.float().cpu()
    gold = torch.stack([vecs[r["item"]] for r in rows])
    pn = torch.nn.functional.normalize(pred.double(), dim=-1)
    gn = torch.nn.functional.normalize(gold.double(), dim=-1)
    cos = (pn * gn).sum(-1)
    for r, c in zip(rows, cos):
        r["cos"] = float(c)
        r["fve"] = float(1 - 2 * (1 - c) / FVE_DENOM)

    # Does the reconstruction keep the confidence direction? For every (kind=clean, position, layer) group the
    # high-minus-low axis is the difference of class means of the ORIGINAL unit-norm activations, computed
    # without the trial itself; originals and reconstructions are projected on it. Steered activations and
    # sampled explanations use the axis of the clean greedy group at the same position.
    key = lambda r: (r["pos"], r["layer"])
    groups = {}
    for i, r in enumerate(rows):
        if r["kind"] == "clean" and r["sample"] == 0:
            groups.setdefault(key(r), []).append(i)
    for k, g in groups.items():
        G = gn[g]
        hi = torch.tensor([rows[i]["group"] == "high" for i in g])
        sh, sl, nh, nl = G[hi].sum(0), G[~hi].sum(0), int(hi.sum()), int((~hi).sum())
        tr = {rows[i]["trial"]: n for n, i in enumerate(g)}
        centre = G.mean(0)
        for i, r in enumerate(rows):
            if r["kind"] not in ("clean", "steer") or key(r) != k:
                continue
            n = tr.get(r["trial"])
            if n is None:
                continue
            h_ = (sh - G[n]) / (nh - 1) if hi[n] else sh / nh
            l_ = sl / nl if hi[n] else (sl - G[n]) / (nl - 1)
            u = torch.nn.functional.normalize(h_ - l_, dim=0)
            r["proj_gold"] = float((gn[i] - centre) @ u)
            r["proj_pred"] = float((pn[i] - centre) @ u)
        # specificity: cosine of a reconstruction with the originals of OTHER trials at the same position
        P = pn[g]
        cross = P @ G.T
        off = (cross.sum(1) - cross.diag()) / (len(g) - 1)
        for n, i in enumerate(g):
            rows[i]["cos_other"] = float(off[n])
        rows[g[0]]["group_mean_pairwise_cos_gold"] = float(((G @ G.T).sum() - len(g)) / (len(g) * (len(g) - 1)))
    with open(out_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = np.array([r["cos"] for r in rows if r["kind"] == "clean" and r["sample"] == 0 and r["layer"] == NLA_LAYER])
    log(f"clean block-41 explanations: mean cosine {c.mean():.4f}, mean fve {np.mean(1 - 2 * (1 - c) / FVE_DENOM):.3f}")

# =========================================================================================================
ref = [r for r in map(json.loads, open(out_file)) if r["kind"] == "ref"]
if ref:
    with open(OUT / f"nla_{args.fmt}_ref.txt", "w") as f:
        f.write("\n\n".join(f"[{r['tok']:3d}] ||v||={r['norm']:.1f} cos={r.get('cos', float('nan')):.3f} fve={r.get('fve', float('nan')):.3f}\n{r['text']}"
                              for r in ref) + "\n")
log("INTERP NLA DONE", out_file)
