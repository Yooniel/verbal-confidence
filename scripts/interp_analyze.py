"""Figures, tables and summary numbers of the interpretability extension (NLA verbalizer, sparse autoencoder,
Jacobian lens). Reads only the tracked outputs in results/interp/ and the steering-control shards.

    python scripts/interp_analyze.py

Writes figures/in01..in09_*.png, results/interp/summary.json and results/interp/tables.md.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import SWEEP_LAYERS, read_jsonl
from vc.env import DATA_RAW, FIGURES, RESULTS_RAW
from vc.plotting import (AQUA, BLUE, GREEN, INK2, MAGENTA, MUTED, ORANGE, RED, VIOLET, hline, layer_axis, line, plt, save,
                         setup)
from vc.prompts import HIGH_CLASSES, MIDPOINTS

setup()
OUT = RESULTS_RAW / "interp"
FMTS = [f for f in ("raw", "tt") if (OUT / f"jlens_{f}.pt").exists()]
FMT_NAME = {"raw": "single prompt", "tt": "two-turn chat"}
PCOL = {"A_last": VIOLET, "PANL": BLUE, "PANL+1": AQUA, "CC": ORANGE, "INT4": MUTED, "UNL": MAGENTA, "REQ0": GREEN}
PLAB = {"raw": {"A_last": "last answer token", "PANL": "PANL (newline after the answer)", "PANL+1": "PANL+1 (first instruction token)",
                "CC": "CC (confidence colon)", "INT4": "class name in the instruction list"},
        "tt": {"A_last": "last answer token", "PANL": "end-of-turn token after the answer", "PANL+1": "newline after it",
               "CC": "last token (emits the class)", "INT4": "class name in the request"}}
S = {}  # summary numbers
T = []  # markdown tables


def md(*lines):
    T.extend(lines)


def tk(s):
    """Token for a markdown table."""
    s = s.replace("\n", "\\n").replace("|", "\\|").replace("`", "'")
    return f"`{s}`"


def first(curve, layers, thr):
    for l, v in zip(layers, curve):
        if v >= thr:
            return int(l)
    return None


# ================================================================================================================
# 1. Jacobian lens
# ================================================================================================================
J = {f: torch.load(OUT / f"jlens_{f}.pt") for f in FMTS}
FAM = " Incorrect"
fig, axes = plt.subplots(2, len(FMTS), figsize=(5.6 * len(FMTS), 6.4), squeeze=False, sharey="row")
for c, f in enumerate(FMTS):
    R = J[f]
    POS, high, names = R["pos"], R["high"].numpy(), R["pin_names"]
    L = R["cls"].shape[1]
    k = names.index(FAM)
    score = (R["cls"][..., 7:].mean(-1) - R["cls"][..., :3].mean(-1)).numpy()  # [2, L, P, n]
    S[f"lens_{f}"] = {"n_trials": int(len(high)), "n_high": int(high.sum()), "token": {}, "class_score": {}, "top_rank": {}}
    for p in ("A_last", "PANL", "PANL+1", "CC"):
        pi = POS.index(p)
        a = R["pin_auroc"][0, :, pi, k].numpy()
        axes[0, c].plot(range(L), a, color=PCOL[p], lw=2, label=PLAB[f][p])
        cs = np.array([roc_auc_score(high, score[0, l, pi]) for l in range(L)])
        axes[1, c].plot(range(L), cs, color=PCOL[p], lw=2, label=PLAB[f][p])
        S[f"lens_{f}"]["token"][p] = {"auroc": {str(l): round(float(a[l]), 3) for l in range(L)},
                                      "first_layer_over_0.8": first(a, range(L), 0.8), "first_layer_over_0.9": first(a, range(L), 0.9),
                                      "auroc_logit_lens": {str(l): round(float(R["pin_auroc"][1, l, pi, k]), 3) for l in range(L)},
                                      "first_layer_over_0.8_logit_lens": first(R["pin_auroc"][1, :, pi, k].numpy(), range(L), 0.8),
                                      "median_rank_high": {str(l): int(R["pin_rank"][0, l, pi, 0, k]) for l in range(L)},
                                      "median_rank_low": {str(l): int(R["pin_rank"][0, l, pi, 1, k]) for l in range(L)}}
        S[f"lens_{f}"]["class_score"][p] = {"auroc": {str(l): round(float(cs[l]), 3) for l in range(L)},
                                            "first_layer_over_0.8": first(cs, range(L), 0.8), "first_layer_over_0.9": first(cs, range(L), 0.9)}
    for r in (0, 1):
        hline(axes[r, c], 0.5)
        axes[r, c].set_xlim(0, L - 1)
        axes[r, c].set_xlabel("Layer")
    axes[0, c].set_title(f"{FMT_NAME[f]}: lens token \"Incorrect\"")
    axes[1, c].set_title(f"{FMT_NAME[f]}: the ten class tokens")
    axes[0, c].legend(loc="upper left")
axes[0, 0].set_ylabel("AUROC, low- vs high-confidence trials\n(rank of \"Incorrect\" in the Jacobian lens)")
axes[1, 0].set_ylabel("AUROC, high- vs low-confidence trials\n(top-3 minus bottom-3 class logits in the lens)")
fig.suptitle("Jacobian lens: \"Incorrect\" is readable at the token after the answer before the class is readable anywhere",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, FIGURES / "in01_lens_incorrect.png")

# ---- the same readout on the 3,000-trial collection set (sweep layers)
fig, axes = plt.subplots(2, len(FMTS), figsize=(5.6 * len(FMTS), 6.0), squeeze=False, sharey="row")
for c, f in enumerate(FMTS):
    col = J[f]["family"]["collection"]
    S[f"lens_{f}"]["collection"] = {}
    for p in ("A_last", "PANL", "PANL+1", "CC"):
        if (p, SWEEP_LAYERS[0]) not in col:
            continue
        r = [col[(p, l)]["r_conf"] for l in SWEEP_LAYERS]
        w = [col[(p, l)]["auroc_wrong_answer"] for l in SWEEP_LAYERS]
        x = layer_axis(axes[0, c], SWEEP_LAYERS)
        line(axes[0, c], x, r, color=PCOL[p], label=PLAB[f][p])
        x = layer_axis(axes[1, c], SWEEP_LAYERS)
        line(axes[1, c], x, w, color=PCOL[p], label=PLAB[f][p])
        S[f"lens_{f}"]["collection"][p] = {"r_conf": {str(l): round(v, 3) for l, v in zip(SWEEP_LAYERS, r)},
                                           "auroc_wrong_answer": {str(l): round(v, 3) for l, v in zip(SWEEP_LAYERS, w)}}
    hline(axes[0, c], 0.0)
    hline(axes[1, c], 0.5)
    axes[0, c].set_title(f"{FMT_NAME[f]}: correlation with the stated confidence")
    axes[1, c].set_title(f"{FMT_NAME[f]}: does it tell wrong answers from right ones?")
    axes[1, c].legend(loc="upper left")
axes[0, 0].set_ylabel("Rank correlation, lens \"Incorrect\" score\nvs stated confidence")
axes[1, 0].set_ylabel("AUROC, wrong vs correct answer\n(lens \"Incorrect\" score)")
fig.suptitle("The lens readout of \"incorrect\" and \"wrong\" on the 3,000-trial collection set (all confidence classes)", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, FIGURES / "in02_lens_collection.png")

# ---- the steering direction through the lens
fig, axes = plt.subplots(2, len(FMTS), figsize=(5.6 * len(FMTS), 6.0), squeeze=False, sharey="row")
for c, f in enumerate(FMTS):
    V = J[f]["vec"]
    S[f"lens_{f}"]["direction"] = {}
    x = layer_axis(axes[0, c], SWEEP_LAYERS[:-1])
    layer_axis(axes[1, c], SWEEP_LAYERS[:-1])
    for p in ("PANL", "CC"):
        ku = [V[(p, l, "high")]["kurt"] for l in SWEEP_LAYERS[:-1]]
        fv = [(V[(p, l, "high")]["fve"] + V[(p, l, "low")]["fve"]) / 2 for l in SWEEP_LAYERS[:-1]]
        kr = [np.mean([b["kurt"] for b in V[(p, l, "baseline")]["random"]]) for l in SWEEP_LAYERS[:-1]]
        fr = [np.mean([b["fve"] for b in V[(p, l, "baseline")]["random"]]) for l in SWEEP_LAYERS[:-1]]
        kp = [np.mean([b["kurt"] for b in V[(p, l, "baseline")]["pair"]]) for l in SWEEP_LAYERS[:-1]]
        fp = [np.mean([b["fve"] for b in V[(p, l, "baseline")]["pair"]]) for l in SWEEP_LAYERS[:-1]]
        line(axes[0, c], x, ku, color=PCOL[p], label=f"steering direction at {p}")
        line(axes[0, c], x, kp, color=PCOL[p], label=f"difference of two random trials at {p}", ls="--", hollow=True, lw=1.2)
        line(axes[1, c], x, fv, color=PCOL[p], label=f"steering direction at {p}")
        line(axes[1, c], x, fp, color=PCOL[p], label=f"difference of two random trials at {p}", ls="--", hollow=True, lw=1.2)
        if p == "CC":
            line(axes[0, c], x, kr, color=MUTED, label="random direction", ls=":", marker=None, lw=1.5)
            line(axes[1, c], x, fr, color=MUTED, label="random direction", ls=":", marker=None, lw=1.5)
        S[f"lens_{f}"]["direction"][p] = {
            "kurtosis": {str(l): round(v, 2) for l, v in zip(SWEEP_LAYERS, ku)}, "kurtosis_random": {str(l): round(float(v), 2) for l, v in zip(SWEEP_LAYERS, kr)},
            "kurtosis_pair": {str(l): round(float(v), 2) for l, v in zip(SWEEP_LAYERS, kp)},
            "share_25_vectors": {str(l): round(v, 3) for l, v in zip(SWEEP_LAYERS, fv)}, "share_random": {str(l): round(float(v), 3) for l, v in zip(SWEEP_LAYERS, fr)},
            "share_pair": {str(l): round(float(v), 3) for l, v in zip(SWEEP_LAYERS, fp)},
            "cos_incorrect_direction": {str(l): round(J[f]["family"]["cos_steer"][(p, l)]["lens_incorrect"], 3) for l in SWEEP_LAYERS},
            "cos_control_direction": {str(l): round(J[f]["family"]["cos_steer"][(p, l)]["lens_control"], 3) for l in SWEEP_LAYERS}}
    axes[0, c].set_title(f"{FMT_NAME[f]}: how peaked is the lens readout of the direction?")
    axes[1, c].set_title(f"{FMT_NAME[f]}: share rebuilt from 25 lens vectors")
    axes[1, c].legend(loc="upper left", fontsize=7)
axes[0, 0].set_ylabel("Excess kurtosis of the lens logits\nover the vocabulary")
axes[1, 0].set_ylabel("Share of the direction's squared norm")
fig.suptitle("The steering direction through the Jacobian lens", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, FIGURES / "in03_lens_direction.png")

# ---- tables: lens tokens
for f in FMTS:
    R = J[f]
    md(f"### Jacobian lens, {FMT_NAME[f]}: tokens of the steering direction", "",
       "| Position | Layer | Added by the high-confidence direction | Added by the low-confidence direction |", "|---|---|---|---|")
    for p, layers in (("PANL", [20, 22, 24, 26, 28, 31, 35, 40]), ("CC", [28, 30, 31, 33, 35, 40, 50])):
        for l in layers:
            md(f"| {p} | {l} | " + " ".join(tk(t) for t in R["vec"][(p, l, "high")]["top"]["tok"][:10]) + " | "
               + " ".join(tk(t) for t in R["vec"][(p, l, "low")]["top"]["tok"][:10]) + " |")
    md("", f"### Jacobian lens, {FMT_NAME[f]}: tokens that differ most between high- and low-confidence trials at PANL", "",
       "| Layer | Stronger on high-confidence trials | Stronger on low-confidence trials |", "|---|---|---|")
    for l in [15, 20, 22, 24, 26, 28, 31, 35, 41, 50]:
        g = R["group_top"][(0, l, "PANL")]
        md(f"| {l} | " + " ".join(tk(t) for t in g["high_minus_low"]["tok"][:10]) + " | " + " ".join(tk(t) for t in g["low_minus_high"]["tok"][:10]) + " |")
    md("", f"### Jacobian lens, {FMT_NAME[f]}: top tokens at PANL (mean over trials)", "",
       "| Layer | High-confidence trials | Low-confidence trials |", "|---|---|---|")
    for l in [10, 20, 24, 26, 28, 31, 35, 41, 50, 61]:
        g = R["group_top"][(0, l, "PANL")]
        md(f"| {l} | " + " ".join(tk(t) for t in g["high"]["tok"][:8]) + " | " + " ".join(tk(t) for t in g["low"]["tok"][:8]) + " |")
    md("")

# ================================================================================================================
# 2. Sparse autoencoder
# ================================================================================================================
SA = {f: json.load(open(OUT / f"sae_{f}.json")) for f in FMTS if (OUT / f"sae_{f}.json").exists()}


def flabel(F_, fid, n=4):
    d = F_[str(fid)]
    toks = [t.strip() for t in d["jlens_top"] if t.strip().isascii() and re.search("[A-Za-z]{2}", t) and "<" not in t]
    seen, out = set(), []
    for t in toks:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return f"#{fid}  " + (", ".join(out[:n]) if out else "(no word among its top lens tokens)")


if SA:
    fig, axes = plt.subplots(1, len(SA), figsize=(6.2 * len(SA), 5.2), squeeze=False)
    for c, (f, R) in enumerate(SA.items()):
        st = R["stats"]["PANL"]
        rows = st["neg"][:8][::-1] + st["pos"][:6][::-1]
        y = np.arange(len(rows))
        ax = axes[0, c]
        for yi, d in zip(y, rows):
            ax.plot([d["mean_high"], d["mean_low"]], [yi, yi], color=MUTED, lw=1.2, zorder=1)
        ax.scatter([d["mean_high"] for d in rows], y, color=BLUE, s=34, zorder=3, label="high-confidence trials", edgecolor="white", linewidth=0.8)
        ax.scatter([d["mean_low"] for d in rows], y, color=RED, s=34, zorder=3, label="low-confidence trials", edgecolor="white", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([flabel(R["features"], d["feature"]) for d in rows], fontsize=7)
        ax.set_xlim(left=-15)
        ax.set_xlabel("Mean activation of the feature at the token after the answer")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{FMT_NAME[f]}")
        if c == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
        S[f"sae_{f}"] = {
            "n_trials": R["n_trials"], "check": {p: {k: round(v, 4) for k, v in d.items()} for p, d in R["check"].items()},
            "cv_r2": {p: R["stats"][p]["cv_r2"] for p in R["stats"]},
            "n_features_abs_r_over_0.3": {p: R["stats"][p]["n_abs_r_over_0.3"] for p in R["stats"]},
            "top_neg": {p: [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for d in R["stats"][p]["neg"][:5]] for p in R["stats"]},
            "top_pos": {p: [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for d in R["stats"][p]["pos"][:5]] for p in R["stats"]},
            "vector": {p: {"cumulative": v["cumulative"], "n_features_differ": v["n_features_differ"]} for p, v in R["vector"].items()},
            "steered_corr": {q: {k: round(v["corr_with_natural"], 3) for k, v in d.items()} for q, d in R["steered"].items()},
            "kurtosis_reference": R["kurtosis_reference"]["quantiles"]}
    fig.suptitle("SAE features (block 31) that follow the stated confidence at the token after the answer\n"
                 "rows: feature number and the words among its top Jacobian-lens tokens; the eight most negative and six most positive correlations",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "in04_sae_features.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for f, R in SA.items():
        for p in ("PANL", "CC"):
            cum = R["vector"][p]["cumulative"]
            ks = sorted(int(k) for k in cum)
            line(axes[0], ks, [cum[str(k)]["cos"] for k in ks], color=PCOL[p], ls="-" if f == "raw" else "--", hollow=f != "raw",
                 label=f"{p}, {FMT_NAME[f]}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of SAE features (largest share first)")
    axes[0].set_ylabel("Cosine with the steering direction")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("How much of the block-31 direction do SAE features rebuild?")
    axes[0].legend(loc="lower right")
    names = ["high2", "low2", "high5", "low5", "rand5"]
    lab = ["high ×2", "low ×2", "high ×5", "low ×5", "random ×5"]
    w, k = 0.8 / (2 * len(SA)), 0
    for f, R in SA.items():
        for q in ("PANL", "CC"):
            vals = [R["steered"][q][n]["corr_with_natural"] for n in names]
            axes[1].bar(np.arange(len(names)) + (k - (2 * len(SA) - 1) / 2) * w, vals, w * 0.92, color=PCOL[q], alpha=1.0 if f == "raw" else 0.45,
                        label=f"read at {q}, {FMT_NAME[f]}")
            k += 1
    hline(axes[1], 0)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(lab)
    axes[1].set_ylabel("Correlation over features: change under steering\nvs natural high-minus-low difference")
    axes[1].set_title("Steering earlier moves the block-31 features the natural way")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, FIGURES / "in05_sae_direction.png")

    for f, R in SA.items():
        Fz = R["features"]
        for p in ("PANL", "CC"):
            md(f"### SAE features that follow the stated confidence at {p}, {FMT_NAME[f]}", "",
               "| Feature | r with confidence | Active in high / low trials | Jacobian-lens tokens of the feature | Strongest corpus example (activating token in «») |",
               "|---|---|---|---|---|")
            for d in R["stats"][p]["neg"][:8] + R["stats"][p]["pos"][:6]:
                ft = Fz.get(str(d["feature"]))
                if ft is None:
                    continue
                ex = ft.get("examples", [])
                e = (ex[0]["before"][-60:] + "«" + ex[0]["token"] + "»" + ex[0]["after"][:20]) if ex else ""
                md(f"| {d['feature']} | {d['r_conf']:+.2f} | {d['rate_high']:.0%} / {d['rate_low']:.0%} | " + " ".join(tk(t) for t in ft["jlens_top"][:6])
                   + " | " + tk(e) + " |")
            md("")
        for p in ("PANL", "CC"):
            md(f"### SAE features with the largest share of the steering direction at {p} (block 31), {FMT_NAME[f]}", "",
               "| Feature | Share | Activation, high minus low | Jacobian-lens tokens | Strongest corpus example |", "|---|---|---|---|---|")
            for d in R["vector"][p]["top"][:10]:
                ft = Fz[str(d["feature"])]
                ex = ft.get("examples", [])
                e = (ex[0]["before"][-60:] + "«" + ex[0]["token"] + "»" + ex[0]["after"][:20]) if ex else ""
                md(f"| {d['feature']} | {d['share']:+.3f} | {d['delta_act']:+.0f} | " + " ".join(tk(t) for t in ft["jlens_top"][:6]) + " | " + tk(e) + " |")
            md("")

# ================================================================================================================
# 3. NLA
# ================================================================================================================
# Word lists for the explanations. Neutral pairs ("correct or incorrect", "correct/incorrect") are removed first.
NEUTRAL = r"correct\s*(?:/|or)\s*incorrect|incorrect\s*(?:/|or)\s*correct|right\s*(?:/|or)\s*wrong|true\s*(?:/|or)\s*false"
UNC = r"incorrect|wrong|mistake|error|inaccura|correction|corrected|the correct answer (?:is|was)|not correct|misconception"
CER = r"(?:is|was|were|are|that's|that is|you are|you're) (?:correct|right)|great|well done|good job|excellent|congrat|perfect"


def has(pat, text):
    return bool(re.search(pat, re.sub(NEUTRAL, " ", text, flags=re.I), re.I))


NLA = {f: [json.loads(l) for l in open(OUT / f"nla_{f}.jsonl")] for f in FMTS if (OUT / f"nla_{f}.jsonl").exists()}
NLA = {f: r for f, r in NLA.items() if r and "cos" in r[0]}


def text_auroc(rows):
    """Cross-validated AUROC of telling high- from low-confidence trials from the explanation text alone
    (TF-IDF of words and word pairs, logistic regression, 5 folds)."""
    y = np.array([r["group"] == "high" for r in rows])
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit_transform([r["text"] for r in rows])
    p = cross_val_predict(LogisticRegression(C=10, max_iter=2000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
    return float(roc_auc_score(y, p))


if NLA:
    SHORT = {"raw": {"A_last": "last answer\ntoken", "PANL": "PANL", "PANL+1": "PANL+1", "INT4": "class name\nin the list", "CC": "CC"},
             "tt": {"A_last": "last answer\ntoken", "PANL": "end of\nturn", "PANL+1": "newline\nafter it", "INT4": "class name\nin the request",
                    "CC": "last\ntoken"}}
    fig, axes = plt.subplots(3, len(NLA), figsize=(5.8 * len(NLA), 9.4), squeeze=False, sharey="row")
    for c, (f, rows) in enumerate(NLA.items()):
        clean = [r for r in rows if r["kind"] == "clean" and r["sample"] == 0 and r["layer"] == 41]
        POSN = [p for p in ("A_last", "PANL", "PANL+1", "INT4", "CC") if any(r["pos"] == p for r in clean)]
        d = S[f"nla_{f}"] = {"n_explanations": len(rows), "status": dict(Counter(r["status"] for r in rows)), "positions": {}}
        for p in POSN:
            g = [r for r in clean if r["pos"] == p]
            hi, lo = [r for r in g if r["group"] == "high"], [r for r in g if r["group"] == "low"]
            rate = lambda rs, pat: float(np.mean([has(pat, r["text"]) for r in rs]))
            y = np.array([r["group"] == "high" for r in g])
            d["positions"][p] = {
                "n": len(g), "cos": round(float(np.mean([r["cos"] for r in g])), 4), "fve": round(float(np.mean([r["fve"] for r in g])), 3),
                "cos_other_trials": round(float(np.mean([r["cos_other"] for r in g])), 4),
                "pairwise_cos_originals": round([r["group_mean_pairwise_cos_gold"] for r in g if "group_mean_pairwise_cos_gold" in r][0], 4),
                "norm": round(float(np.mean([r["norm"] for r in g]))),
                "doubt_words_high": rate(hi, UNC), "doubt_words_low": rate(lo, UNC), "sure_words_high": rate(hi, CER), "sure_words_low": rate(lo, CER),
                "text_auroc": round(text_auroc(g), 3),
                "axis_auroc_original": round(float(roc_auc_score(y, [r["proj_gold"] for r in g])), 3),
                "axis_auroc_reconstruction": round(float(roc_auc_score(y, [r["proj_pred"] for r in g])), 3),
                "axis_r_original_vs_reconstruction": round(float(np.corrcoef([r["proj_gold"] for r in g], [r["proj_pred"] for r in g])[0, 1]), 3)}
        x = np.arange(len(POSN))
        for r_, (key, title) in enumerate((("doubt", "words of error or correction"), ("sure", "words of approval"))):
            ax = axes[r_, c]
            ax.bar(x - 0.2, [d["positions"][p][f"{key}_words_high"] for p in POSN], 0.38, color=BLUE, label="high-confidence trials")
            ax.bar(x + 0.2, [d["positions"][p][f"{key}_words_low"] for p in POSN], 0.38, color=RED, label="low-confidence trials")
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT[f][p] for p in POSN])
            ax.set_ylim(0, 1)
            ax.set_title(f"{FMT_NAME[f]}: {title}")
            if c == 0:
                ax.set_ylabel("Share of explanations")
            if r_ == 0 and c == 0:
                h1 = ax.get_legend_handles_labels()
        ax = axes[2, c]
        ax.bar(x - 0.2, [d["positions"][p]["axis_auroc_original"] for p in POSN], 0.38, color=INK2, label="original activation")
        ax.bar(x + 0.2, [d["positions"][p]["axis_auroc_reconstruction"] for p in POSN], 0.38, color=ORANGE, label="rebuilt from the explanation text")
        hline(ax, 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[f][p] for p in POSN])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{FMT_NAME[f]}: is the confidence direction kept in the text?")
        if c == 0:
            ax.set_ylabel("AUROC, high vs low confidence,\nprojection on the high-minus-low axis")
            h2 = ax.get_legend_handles_labels()
        # further groups
        for name, sel in (("PANL_samples", lambda r: r["kind"] == "clean" and r["sample"] > 0),
                          ("PANL_layer25", lambda r: r["kind"] == "clean" and r["layer"] == 25),
                          ("PANL_layer31", lambda r: r["kind"] == "clean" and r["layer"] == 31)):
            g = [r for r in rows if sel(r)]
            if g:
                hi, lo = [r for r in g if r["group"] == "high"], [r for r in g if r["group"] == "low"]
                d[name] = {"n": len(g), "cos": round(float(np.mean([r["cos"] for r in g])), 4), "fve": round(float(np.mean([r["fve"] for r in g])), 3),
                           "doubt_words_high": float(np.mean([has(UNC, r["text"]) for r in hi])),
                           "doubt_words_low": float(np.mean([has(UNC, r["text"]) for r in lo])),
                           "text_auroc": round(text_auroc(g), 3)}
        d["steer"] = {}
        for q in ("PANL", "CC"):
            base = {r["trial"]: r for r in clean if r["pos"] == q}
            for v in ("high5", "low5", "rand5"):
                g = [r for r in rows if r["kind"] == "steer" and r["pos"] == q and r["variant"] == v]
                if not g:
                    continue
                d["steer"][f"{q}|{v}"] = {
                    "n": len(g), "cos": round(float(np.mean([r["cos"] for r in g])), 4),
                    "doubt_words": {grp: float(np.mean([has(UNC, r["text"]) for r in g if r["group"] == grp])) for grp in ("high", "low")},
                    "doubt_words_unsteered": {grp: float(np.mean([has(UNC, base[r["trial"]]["text"]) for r in g if r["group"] == grp])) for grp in ("high", "low")},
                    "shift_on_axis_original": {grp: float(np.mean([r["proj_gold"] - base[r["trial"]]["proj_gold"] for r in g if r["group"] == grp])) for grp in ("high", "low")},
                    "shift_on_axis_reconstruction": {grp: float(np.mean([r["proj_pred"] - base[r["trial"]]["proj_pred"] for r in g if r["group"] == grp])) for grp in ("high", "low")}}
    fig.suptitle("Natural Language Autoencoder: explanations of block-41 activations at five prompt positions", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.legend(h1[0] + h2[0], h1[1] + h2[1], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    save(fig, FIGURES / "in06_nla.png")

# ================================================================================================================
# 4. steering controls: lens directions, and the steering direction split by its average-activation part, by the
#    SAE and by the lens. Effects are reported on the trials a push can move: "high" variants (towards high
#    confidence) on the low-confidence trials, "low" variants on the high-confidence trials.
# ================================================================================================================
def steer_run(tags, sub=""):
    """{variant: {layer: (mean, sem, n)}} of the change in stated confidence on the relevant trial group,
    merged over result folders; "<variant>|all" holds the mean over all trials."""
    trials = read_jsonl(DATA_RAW / sub / "trials.jsonl")
    out = {}
    for tag in tags:
        files = sorted(glob.glob(str(RESULTS_RAW / sub / f"steer{tag}" / "shard*.pt")))
        if not files:
            return None
        sh = [torch.load(f_) for f_ in files]
        idx = [i_ for s_ in sh for i_ in s_["idx"]]
        hi = np.array([trials[i_]["cls"] in HIGH_CLASSES for i_ in idx])
        lg, ref = torch.cat([s_["logits"] for s_ in sh]), torch.cat([s_["ref"] for s_ in sh])
        mid = torch.tensor(MIDPOINTS)
        dconf = (mid[lg.argmax(-1)] - mid[ref.argmax(-1)]).numpy()  # [n, variants, layers]
        for vi, vn in enumerate(sh[0]["names"]):
            g = ~hi if vn.split("|")[1] == "high" else hi
            out[vn] = {int(l): (float(dconf[g, vi, k].mean()), float(dconf[g, vi, k].std(ddof=1) / np.sqrt(g.sum())), int(g.sum()))
                       for k, l in enumerate(sh[0]["layers"])}
            out[vn + "|all"] = {int(l): (float(dconf[:, vi, k].mean()), float(dconf[:, vi, k].std(ddof=1) / np.sqrt(len(hi))), len(hi))
                                for k, l in enumerate(sh[0]["layers"])}
    return out


CTRL = {"full": [""], "lens_incorrect": ["_lensinc"], "lens_control": ["_lensctl"], "mean_part": ["_meanpart"], "mean_rest": ["_meanrest"],
        "sae_top25": ["_saepart"], "sae_rest": ["_saerest"], "sae_all": ["_saefull"], "sae_error": ["_saeerr"],
        "lens25": ["_jpart", "_jpartneg"], "lens25_rest": ["_jrest", "_jrestneg"]}
runs = {name: steer_run(tags) for name, tags in CTRL.items()}
runs = {k: v for k, v in runs.items() if v is not None}
if len(runs) > 1:
    S["steer_controls"] = {name: {vn: {str(l): [round(m, 3), round(e, 3), n_] for l, (m, e, n_) in d.items()} for vn, d in r.items()}
                           for name, r in runs.items()}
GROUP_TITLE = {"high": "low-confidence trials, pushed towards high confidence", "low": "high-confidence trials, pushed towards low confidence"}


def control_panel(ax, p, d, series, layers):
    x = np.arange(len(layers))
    for name, scale, col, lab, ls, hollow in series:
        if name not in runs or f"{p}|{d}|{scale}" not in runs[name]:
            continue
        r = runs[name][f"{p}|{d}|{scale}"]
        line(ax, x, [r[l][0] for l in layers], err=[r[l][1] for l in layers], color=col, label=lab, ls=ls, hollow=hollow, lw=1.8)
    hline(ax, 0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer of the edit")


if "lens_incorrect" in runs:
    LL = sorted(runs["lens_incorrect"]["PANL|high|5"])
    series = [("full", 2, INK2, "steering direction, scale 2", "-", False), ("full", 5, INK2, "steering direction, scale 5", "--", True),
              ("lens_incorrect", 2, MAGENTA, "\"Incorrect\" lens direction, scale 2", "-", False),
              ("lens_incorrect", 5, MAGENTA, "\"Incorrect\" lens direction, scale 5", "--", True),
              ("lens_control", 2, AQUA, "lens direction of eight unrelated tokens, scale 2", "-", False),
              ("lens_control", 5, AQUA, "lens direction of eight unrelated tokens, scale 5", "--", True)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, sharey="row")
    for ri_, d in enumerate(("high", "low")):
        for ax, p in zip(axes[ri_], ("PANL", "CC")):
            control_panel(ax, p, d, series, LL)
            ax.set_title(f"Edit at {p}: {GROUP_TITLE[d]}", fontsize=9)
        axes[ri_, 0].set_ylabel("Change in stated confidence")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Steering with the lens direction of \"Incorrect\" (towards high confidence = against it, towards low = along it)", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "in07_steer_lens_direction.png")

if "mean_rest" in runs:
    LL = sorted(runs["mean_rest"]["PANL|high|5"])
    series = [("full", 5, INK2, "steering direction", "-", False), ("mean_rest", 5, BLUE, "without its average-activation part", "-", False),
              ("mean_part", 5, ORANGE, "only its average-activation part", "-", False)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), squeeze=False, sharey="row")
    for ri_, d in enumerate(("high", "low")):
        for ax, p in zip(axes[ri_], ("PANL", "CC")):
            control_panel(ax, p, d, series, LL)
            ax.set_title(f"Edit at {p}: {GROUP_TITLE[d]}", fontsize=9)
        axes[ri_, 0].set_ylabel("Change in stated confidence")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("The steering direction split into its part along the average activation and the rest (scale 5)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "in08_steer_mean_split.png")

if "sae_top25" in runs or "lens25" in runs:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, groups, layer_of, title in (
            (axes[0], [("full", "steering\ndirection"), ("sae_all", "all SAE\nfeatures"), ("sae_top25", "top-25 SAE\nfeatures"),
                       ("sae_rest", "direction minus\ntop-25 features"), ("sae_error", "what the SAE\nmisses")], {"PANL": 31, "CC": 31},
             "Block 31: the direction split by the SAE"),
            (axes[1], [("full", "steering\ndirection"), ("lens25", "25 lens\nvectors"), ("lens25_rest", "direction minus\n25 lens vectors")],
             {"PANL": 24, "CC": 31}, "The direction split by the Jacobian lens (PANL at block 24, CC at block 31)")):
        groups = [g for g in groups if g[0] in runs]
        w = 0.2
        for k, (p, d, col, alpha) in enumerate((("PANL", "high", BLUE, 0.45), ("PANL", "low", RED, 0.45), ("CC", "high", BLUE, 1.0), ("CC", "low", RED, 1.0))):
            vals = [runs[g][f"{p}|{d}|5"][layer_of[p]][0] for g, _ in groups]
            errs = [runs[g][f"{p}|{d}|5"][layer_of[p]][1] for g, _ in groups]
            ax.bar(np.arange(len(groups)) + (k - 1.5) * w, vals, w * 0.92, yerr=errs, color=col, alpha=alpha, ecolor=INK2, error_kw={"lw": 0.8},
                   label=f"edit at {p}: " + ("low-confidence trials pushed up" if d == "high" else "high-confidence trials pushed down"))
        hline(ax, 0)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[1] for g in groups], fontsize=7.5)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Change in stated confidence")
    fig.suptitle("Steering with the parts of the direction that the SAE and the lens do and do not capture (scale 5)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    save(fig, FIGURES / "in09_steer_parts.png")

json.dump(S, open(OUT / "summary.json", "w"), indent=1)
open(OUT / "tables.md", "w").write("\n".join(T).rstrip("\n") + "\n")
print("INTERP ANALYZE DONE; summary keys:", list(S))
