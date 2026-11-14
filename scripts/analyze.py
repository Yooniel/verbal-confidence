"""Aggregate all results into figures (figures/*.png) and a machine-readable summary (results/summary.json).

Run after the stages whose outputs you want analysed; missing experiments are skipped.
Metrics follow the paper (C.1.5): confidence = midpoint of the argmax class; logit difference = logit of the
clean trial's class minus the mean logit of the other nine classes; token change rate = fraction of trials
whose argmax class differs from the clean run.
"""
import json
import os
import sys
from glob import glob

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib.pyplot as plt

from vc import plotting as P
from vc.data import read_jsonl
from vc.env import DATA, FIGURES, RESULTS
from vc.prompts import CLASSES, HIGH_CLASSES, LOW_CLASSES, MIDPOINTS

P.setup()
os.makedirs(FIGURES, exist_ok=True)
MID = np.array(MIDPOINTS)
S = {}  # summary


def have(exp):
    return len(glob(str(RESULTS / exp / "shard*.pt"))) > 0


def load(exp):
    parts = [torch.load(f, weights_only=False) for f in sorted(glob(str(RESULTS / exp / "shard*.pt")))]
    out = dict(parts[0])
    for k in ("logits", "ref", "clean", "corrupt", "ood"):
        if k in out:
            out[k] = torch.cat([p[k] for p in parts], 0).numpy()
    for k in ("idx", "recs"):
        if k in out:
            out[k] = [x for p in parts for x in p[k]]
    return out


def ldiff(z, y):
    """z [..., 10], y [...] int -> z[y] - mean(z[others])"""
    zy = np.take_along_axis(z, y[..., None], -1)[..., 0]
    return zy - (z.sum(-1) - zy) / 9.0


def mets(logits, base):
    """logits [n, V, L, 10]; base = reference logits, either [n, 10] (clean run) or [n, V, L, 10] (the unedited
    rows computed in the same forward pass as each variant). Returns dict of [n, V, L] arrays."""
    if base.ndim == 2:
        base = np.broadcast_to(base[:, None, None], logits.shape)
    y = base.argmax(-1)
    conf = MID[logits.argmax(-1)]
    ld = ldiff(logits, y)
    return {
        "conf": conf,
        "dconf": conf - MID[y],
        "ld": ld,
        "dld": ld - ldiff(base, y),
        "change": (logits.argmax(-1) != y).astype(float),
    }


def noise_floor(D):
    """Share of trials whose argmax class differs between the unedited rows of the resumed passes and the
    full clean pass (pure numerical noise from the different batch shapes), per layer."""
    return 100 * (D["ref"].argmax(-1) != D["clean"].argmax(-1)[:, None, None]).mean((0, 1))


def ms(a):
    """mean and SEM over trials (axis 0)."""
    return a.mean(0), P.sem(a, 0)


def peak(layers, mean, mode="max"):
    i = int(np.argmax(mean) if mode == "max" else np.argmin(mean))
    return {"layer": int(layers[i]), "value": float(mean[i])}


trials = read_jsonl(DATA / "trials.jsonl") if (DATA / "trials.jsonl").exists() else None

# =====================================================================================================
# Behaviour (Fig. 9)
# =====================================================================================================
if (RESULTS / "behav" / "summary.json").exists():
    B = json.load(open(RESULTS / "behav" / "summary.json"))
    S["behaviour"] = B
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=P.AXIS, lw=1.2, zorder=1)
    ax.annotate("perfect calibration", (0.56, 0.60), rotation=41, color=P.MUTED, fontsize=7.5, ha="center", va="bottom")
    n = np.array([B["class_counts"][c] for c in CLASSES])
    acc = np.array([B["class_accuracy"][c] if B["class_accuracy"][c] is not None else np.nan for c in CLASSES])
    keep = n >= 20
    ax.scatter(MID[keep], acc[keep], s=18 + 260 * n[keep] / n.max(), color=P.BLUE, edgecolor=P.SURFACE, linewidth=1.2, zorder=3)
    for i, (m, a, k) in enumerate(zip(MID[keep], acc[keep], n[keep])):
        ax.annotate(f"n={k}", (m, a), textcoords="offset points", xytext=(0, 9 if i % 2 == 0 else -15), ha="center", fontsize=7, color=P.INK2)
    ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.02)
    ax.set_xlabel("Verbal confidence (class midpoint)"), ax.set_ylabel("Accuracy")
    ax.set_title(f"Calibration: ECE {B['ece']:.2f}, AUROC {B['auroc_verbal_conf']:.2f}")
    ax.grid(True, axis="both")
    ax = axes[1]
    bars = ax.bar(np.arange(10), n, width=0.55, color=P.BLUE, zorder=3)
    for i, k in enumerate(n):
        ax.annotate(str(k), (i, k), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5, color=P.INK2)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax.set_ylabel("Trials")
    ax.set_title(f"Confidence classes (n={B['n_valid']}, accuracy {100*B['accuracy']:.1f}%)")
    fig.tight_layout()
    P.save(fig, FIGURES / "fig09_calibration.png")

# =====================================================================================================
# Steering (Figs. 2, 11, 13a, 29)
# =====================================================================================================
if have("steer"):
    D = load("steer")
    L, names = D["layers"], D["names"]
    M = mets(D["logits"], D["ref"])
    x = np.arange(len(L))
    S["steer"] = {"n": len(D["idx"]), "baseline_conf": float(MID[D["clean"].argmax(-1)].mean()),
                  "noise_floor_change_pct_by_layer": noise_floor(D).tolist()}

    def curve(pos, d, s, mask=None):
        v = M["dconf"][:, names.index(f"{pos}|{d}|{s}")]
        return ms(v if mask is None else v[mask])

    def panel(ax, pos, scales=(2, 5), title=None, mask=None):
        P.hline(ax, 0, color=P.AXIS)
        for s in scales:
            strong = s == max(scales)
            for d, col, lab in (("high", P.BLUE, "high-confidence vector"), ("low", P.RED, "low-confidence vector")):
                m, e = curve(pos, d, s, mask)
                P.line(ax, x, m, e, color=col, label=f"{lab}, scale {s}", ls="-" if strong else (0, (3, 1.5)),
                       hollow=not strong, lw=2.0 if strong else 1.4, ms=4.5 if strong else 4)
        P.layer_axis(ax, L)
        ax.set_title(title or P.POS_LABEL[pos])
        ax.set_ylabel("Δ confidence")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    for ax, pos in zip(axes.ravel(), ["PANL", "CC", "PANL+1", "FCC"]):
        panel(ax, pos)
    for a in axes[1]:
        a.set_ylim(axes[0, 0].get_ylim())  # controls on the same scale as PANL
    axes[1, 1].legend(loc="upper right", ncol=1)
    fig.suptitle(f"Activation steering, Gemma 3 27B / TriviaQA (n={S['steer']['n']} trials; mean ± SEM)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    P.save(fig, FIGURES / "fig02_steering.png")

    # first vs last answer token coincide for single-token answers, so the first-token panel uses multi-token answers
    multi_mask = np.array([trials[i]["n_a"] for i in D["idx"]]) >= 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    panel(axes[0], "A_first", title=f"First answer token (multi-token answers, n={int(multi_mask.sum())})", mask=multi_mask)
    panel(axes[1], "A_last", title="Last answer token (all 200 trials)")
    panel(axes[2], "AC", title="AC (answer colon, last token before the answer)")
    for a in axes[1:]:
        a.set_ylabel("")
    axes[2].legend(loc="lower right")
    fig.tight_layout()
    P.save(fig, FIGURES / "fig11_13a_steering_answer_tokens_AC.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, pos in zip(axes, ["PANL", "CC"]):
        panel(ax, pos, scales=(1, 10), title=P.POS_LABEL[pos] + " — scales 1 and 10")
    axes[0].legend(loc="best")
    fig.tight_layout()
    P.save(fig, FIGURES / "fig29_steering_scales_1_10.png")

    for pos in ["PANL", "CC", "PANL+1", "FCC", "AC", "A_first", "A_last"]:
        S["steer"][pos] = {}
        for sc in ([1, 2, 5, 10] if pos in ("PANL", "CC") else [2, 5]):
            hm, he = curve(pos, "high", sc)
            lm, le = curve(pos, "low", sc)
            S["steer"][pos][f"scale{sc}"] = {"high_peak": peak(L, hm, "max"), "low_peak": peak(L, lm, "min"),
                                             "high_curve": hm.tolist(), "low_curve": lm.tolist(),
                                             "high_sem": he.tolist(), "low_sem": le.tolist()}
    S["steer"]["layers"] = list(L)
    # first vs last answer token are the same position for single-token answers: separate them out
    n_a = np.array([trials[i]["n_a"] for i in D["idx"]])
    multi = n_a >= 2
    S["steer"]["n_multi_token_answers"] = int(multi.sum())
    for pos in ["A_first", "A_last"]:
        for d in ("high", "low"):
            v = M["dconf"][multi][:, names.index(f"{pos}|{d}|5")].mean(0)
            S["steer"][pos][f"scale5_multi_token_{d}_peak"] = peak(L, v, "max" if d == "high" else "min")
    # by confidence band of the (unedited) trial: ceiling/floor effects
    yb = D["clean"].argmax(-1)
    for band, cls in (("high", HIGH_CLASSES), ("low", LOW_CLASSES)):
        mk = np.isin(yb, cls)
        for pos in ["PANL", "CC"]:
            for d in ("high", "low"):
                v = M["dconf"][mk][:, names.index(f"{pos}|{d}|5")].mean(0)
                S["steer"][pos][f"scale5_{d}_vector_on_{band}_trials_peak"] = peak(L, v, "max" if d == "high" else "min")

# =====================================================================================================
# Patching (Fig. 3)
# =====================================================================================================
if have("patch"):
    D = load("patch")
    L, names = D["layers"], D["names"]
    M = mets(D["logits"], D["clean"])
    y = D["clean"].argmax(-1)
    clean_ld, corr_ld = ldiff(D["clean"], y), ldiff(D["corrupt"], y)
    clean_conf, corr_conf = MID[y], MID[D["corrupt"].argmax(-1)]
    corr_change = (D["corrupt"].argmax(-1) != y).astype(float)
    x = np.arange(len(L))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    rows = [("ld", "Logit difference", clean_ld.mean(), corr_ld.mean(), 1.0),
            ("change", "Token change rate (%)", 0.0, 100 * corr_change.mean(), 100.0),
            ("conf", "Confidence", clean_conf.mean(), corr_conf.mean(), 1.0)]
    for ax, (k, lab, cl, co, sc) in zip(axes, rows):
        P.hline(ax, cl, color=P.GREEN, label="clean")
        P.hline(ax, co, color=P.RED, label="corrupted (no patch)")
        for pos in ["PANL+1", "CC", "PANL"]:
            m, e = ms(sc * M[k][:, names.index(pos)])
            P.line(ax, x, m, e, color=P.POS_COLOR[pos], label=P.POS_LABEL[pos], marker={"PANL": "o", "CC": "s", "PANL+1": "^"}[pos])
        ax.set_ylabel(lab)
    P.layer_axis(axes[-1], L)
    axes[1].legend(loc="center left", ncol=1)
    axes[0].set_title(f"Activation patching after answer-token corruption (n={len(y)} high-confidence trials)")
    fig.tight_layout()
    P.save(fig, FIGURES / "fig03_patching.png")

    # supplementary metric: margin of the clean class over the class the corrupted run falls into
    c = D["corrupt"].argmax(-1)
    ar = np.arange(len(y))
    gap_clean = D["clean"][ar, y] - D["clean"][ar, c]
    gap = lambda z: z[ar[:, None], np.arange(z.shape[1])[None], y[:, None]] - z[ar[:, None], np.arange(z.shape[1])[None], c[:, None]]  # z [n, L, 10]

    ref_y = np.broadcast_to(y[:, None, None], D["ref"].shape[:-1])
    S["patch_ref_check"] = {"max_abs_AC_minus_ref": float(np.abs(D["logits"][:, names.index("AC")] - D["ref"][:, names.index("AC")]).max()),
                            "corrupt_ld_same_pass_by_layer": ldiff(D["ref"], ref_y)[:, 0].mean(0).tolist()}
    S["patch"] = {"n": len(y), "clean": {"ld": float(clean_ld.mean()), "conf": float(clean_conf.mean())},
                  "corrupt": {"ld": float(corr_ld.mean()), "conf": float(corr_conf.mean()), "change": float(corr_change.mean())},
                  "layers": list(L)}
    for pos in names:
        v = names.index(pos)
        cm, lm_, chm = M["conf"][:, v].mean(0), M["ld"][:, v].mean(0), M["change"][:, v].mean(0)
        rec_conf = 100 * (cm - corr_conf.mean()) / (clean_conf.mean() - corr_conf.mean())
        rec_ld = 100 * (lm_ - corr_ld.mean()) / (clean_ld.mean() - corr_ld.mean())
        rec_tok = 100 * (corr_change.mean() - chm) / corr_change.mean()
        g_ref, g_pat = gap(D["ref"][:, v]).mean(0), gap(D["logits"][:, v]).mean(0)
        rec_gap = 100 * (g_pat - g_ref) / (gap_clean.mean() - g_ref)
        S["patch"][pos] = {"gap_patched": g_pat.tolist(), "gap_corrupt": g_ref.tolist(), "gap_clean": float(gap_clean.mean()),
                           "recovery_gap_pct": rec_gap.tolist(), "peak_gap_recovery": peak(L, rec_gap),
                           "conf": cm.tolist(), "ld": lm_.tolist(), "change": chm.tolist(),
                           "recovery_conf_pct": rec_conf.tolist(), "recovery_ld_pct": rec_ld.tolist(), "recovery_token_pct": rec_tok.tolist(),
                           "peak_conf_recovery": peak(L, rec_conf), "peak_ld_recovery": peak(L, rec_ld)}
    if "ood" in D:
        li = list(L).index(25)
        S["patch"]["ood_L25"] = {p: {"cos": float(D["ood"][:, i, li, 0].mean()), "nr": float(D["ood"][:, i, li, 1].mean())} for i, p in enumerate(D["ood_pos"])}

# =====================================================================================================
# Noising (Fig. 12, 13c)
# =====================================================================================================
if have("noise"):
    D = load("noise")
    L, names = D["layers"], D["names"]
    M = mets(D["logits"], D["ref"])
    y = D["clean"].argmax(-1)
    clean_ld = ldiff(D["clean"], y)
    x = np.arange(len(L))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    P.hline(axes[0], 0, color=P.AXIS)
    for pos in ["PANL+1", "AC", "CC", "PANL"]:
        m, e = ms(M["dld"][:, names.index(pos)])
        P.line(axes[0], x, m, e, color=P.POS_COLOR[pos], label=P.POS_LABEL[pos], marker={"PANL": "o", "CC": "s", "PANL+1": "^", "AC": "D"}[pos])
        m, e = ms(100 * M["change"][:, names.index(pos)])
        P.line(axes[1], x, m, e, color=P.POS_COLOR[pos], label=P.POS_LABEL[pos], marker={"PANL": "o", "CC": "s", "PANL+1": "^", "AC": "D"}[pos])
    nf = noise_floor(D)
    axes[0].set_ylabel(f"Change in logit difference\n(clean: {clean_ld.mean():.1f})"), axes[1].set_ylabel("Token change rate (%)")
    P.layer_axis(axes[1], L)
    h, l_ = axes[1].get_legend_handles_labels()
    axes[1].legend(h[::-1], l_[::-1], loc="upper left")
    axes[0].set_title(f"Activation noising (mean ablation), n={len(y)} trials")
    fig.tight_layout()
    P.save(fig, FIGURES / "fig12_noising.png")
    S["noise"] = {"n": len(y), "clean_ld": float(clean_ld.mean()), "layers": list(L), "noise_floor_change_pct_by_layer": nf.tolist(),
                  "n_high": int(np.isin(y, HIGH_CLASSES).sum()), "n_low": int(np.isin(y, LOW_CLASSES).sum())}
    for pos in names:
        v = names.index(pos)
        ch, dl = 100 * M["change"][:, v].mean(0), M["dld"][:, v].mean(0)
        S["noise"][pos] = {"change_pct": ch.tolist(), "dld": dl.tolist(), "peak_change": peak(L, ch), "peak_dld": peak(L, dl, "min")}
        # split by clean confidence band
        for band, cls in (("high", HIGH_CLASSES), ("low", LOW_CLASSES)):
            mk = np.isin(y, cls)
            if mk.sum() > 5:
                S["noise"][pos][f"change_pct_{band}"] = (100 * M["change"][mk][:, v].mean(0)).tolist()
    if "ood" in D:
        li = list(L).index(25)
        S["noise"]["ood_L25"] = {p: {"cos": float(D["ood"][:, i, li, 0].mean()), "nr": float(D["ood"][:, i, li, 1].mean())} for i, p in enumerate(D["ood_pos"])}

# =====================================================================================================
# Swap (Figs. 5, 20)
# =====================================================================================================
if have("swap"):
    D = load("swap")
    L, names = D["layers"], D["names"]
    M = mets(D["logits"], D["ref"])
    kind = np.array([r["kind"] for r in D["recs"]])
    x = np.arange(len(L))
    CONDS = [("high", "high", "high→high (same)", P.BLUE), ("high", "low", "high→low (cross)", P.ORANGE),
             ("low", "high", "low→high (cross)", P.AQUA), ("low", "low", "low→low (same)", P.YELLOW)]
    S["swap"] = {"n_high": int((kind == "high").sum()), "n_low": int((kind == "low").sum()), "layers": list(L)}

    def swap_fig(positions, fname, sharey=False, legend_at=(1, 0), legend_loc="upper right"):
        fig, axes = plt.subplots(3, len(positions), figsize=(6.4 * len(positions), 8.8), sharex=True, sharey="row" if sharey else False, squeeze=False)
        for c, pos in enumerate(positions):
            for r, (k, lab, sc) in enumerate([("dld", "Logit-difference change", 1.0), ("change", "Token change rate (%)", 100.0), ("dconf", "Confidence change", 1.0)]):
                ax = axes[r, c]
                P.hline(ax, 0, color=P.AXIS)
                for rk, dk, name, col in CONDS:
                    m, e = ms(sc * M[k][kind == rk][:, names.index(f"{pos}|{dk}")])
                    P.line(ax, x, m, e, color=col, label=name, hollow=(rk == dk), lw=1.5 if rk == dk else 2.0)
                if c == 0 or not sharey:
                    ax.set_ylabel(lab)
                if r == 0:
                    ax.set_title(f"Swap at {P.POS_LABEL[pos]}")
            P.layer_axis(axes[-1, c], L)
        axes[legend_at].legend(loc=legend_loc, ncol=1, title="recipient→donor confidence")
        fig.tight_layout()
        P.save(fig, FIGURES / fname)

    # PANL next to its control on the same y-scale, CC separately (its late-layer effects are much larger)
    swap_fig(["PANL", "PANL+1"], "fig05_swap_PANL_vs_PANL1.png", sharey=True, legend_at=(1, 1))
    swap_fig(["CC"], "fig20_swap_CC.png", legend_at=(1, 0), legend_loc="upper left")
    for pos in ["PANL", "CC", "PANL+1"]:
        S["swap"][pos] = {}
        for rk, dk, name, _ in CONDS:
            sub = {k: M[k][kind == rk][:, names.index(f"{pos}|{dk}")].mean(0) for k in ("dld", "change", "dconf")}
            S["swap"][pos][f"{rk}->{dk}"] = {
                "dld": sub["dld"].tolist(), "change_pct": (100 * sub["change"]).tolist(), "dconf": sub["dconf"].tolist(),
                "peak_change": peak(L, 100 * sub["change"]),
                "peak_dconf": peak(L, sub["dconf"], "max" if dk == "high" else "min"),
                "peak_dld": peak(L, sub["dld"], "min"),
            }
    S["swap"]["noise_floor_change_pct_by_layer"] = noise_floor(D).tolist()
    S["swap"]["match_stats"] = json.load(open(DATA / "swap_plan.json"))["match_stats"]

# =====================================================================================================
# Attention blocking, minimal numeric prompt (Fig. 7) - one figure per prompt variant
# =====================================================================================================
MIN_DESC = {
    "pmq": "prompt ends  **:'  (digit is the next token), separator \\n\\n",
    "pmq1": "prompt ends  **:'  (digit is the next token), separator \\n",
    "pm": "prompt ends  **:  (digit logits read at CC), separator \\n\\n",
    "pm1": "prompt ends  **:  (digit logits read at CC), separator \\n",
}
for key in ["pmq", "pmq1", "pm", "pm1"]:
    exp = f"attn_min_{key}"
    if not have(exp):
        continue
    D = load(exp)
    C, names = D["centers"], D["names"]
    lg = D["logits"]  # [n, cond, window, 10]
    base = lg[:, names.index("none")]  # same code path, empty mask: reference for each window
    y = base.argmax(-1)  # [n, W]
    x = np.arange(len(C))

    def am(cond):
        z = lg[:, names.index(cond)]
        return (z.argmax(-1) != y).astype(float) * 100, ldiff(z, y) - ldiff(base, y)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True)
    groups = [(["CC>PANL", "CC>PANL+1", "CC>Q+A"], [P.BLUE, P.AQUA, P.MAGENTA], "A  Does CC read from PANL, or from question/answer tokens?"),
              (["PANL>A", "PANL>last_A", "CC>PANL+1"], [P.VIOLET, P.ORANGE, P.AQUA], "B  Does PANL read from the answer tokens?")]
    LAB = {"CC>PANL": "CC→PANL", "CC>PANL+1": "CC→PANL+1 (control)", "CC>Q+A": "CC→Q+A", "PANL>A": "PANL→A (all answer tokens)",
           "PANL>last_A": "PANL→last A", "CC>Q": "CC→Q", "CC>A": "CC→A"}
    cl = D["clean"].argmax(-1)
    S[exp] = {"n": lg.shape[0], "centers": list(C), "variant": MIN_DESC[key],
              "noise_floor_change_pct": float(100 * (y != cl[:, None]).mean()),
              "baseline_digit_counts": {int(d): int((cl == d).sum()) for d in range(10)},
              "baseline_logit_diff": float(ldiff(D["clean"], cl).mean())}
    for c, (conds, cols, title) in enumerate(groups):
        for cond, col in zip(conds, cols):
            ch, dl = am(cond)
            P.line(axes[0, c], x, *ms(ch), color=col, label=LAB[cond])
            P.line(axes[1, c], x, *ms(dl), color=col, label=LAB[cond])
        P.hline(axes[1, c], 0, color=P.AXIS)
        axes[0, c].set_title(title)
        axes[0, c].set_ylabel("Token change rate (%)"), axes[1, c].set_ylabel("Logit-difference change")
        axes[0, c].legend(loc="upper right")
        axes[0, c].set_ylim(bottom=0)
        axes[1, c].set_xticks(x)
        axes[1, c].set_xticklabels([str(v) for v in C], rotation=45)
        axes[1, c].set_xlabel("Centre of the 12-layer blocking window")
    fig.suptitle(f"Attention blocking, minimal 0–9 prompt — {MIN_DESC[key]} (n={lg.shape[0]})", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    P.save(fig, FIGURES / f"fig07_attention_blocking_minimal_{key}.png")
    for cond in names[1:]:
        ch, dl = am(cond)
        S[exp][cond] = {"change_pct": ch.mean(0).tolist(), "dld": dl.mean(0).tolist(), "dld_sem": P.sem(dl, 0).tolist(),
                        "peak_change": peak(C, ch.mean(0)), "peak_dld": peak(C, dl.mean(0), "min")}

# =====================================================================================================
# Attention blocking, categorical prompt (Fig. 21)
# =====================================================================================================
if have("attn_cat"):
    D = load("attn_cat")
    names = D["names"]
    lg = D["logits"]  # [n, cond, 10]
    base = lg[:, 0]
    y = base.argmax(-1)
    conds = names[1:]
    ch = np.stack([(lg[:, names.index(c)].argmax(-1) != y).astype(float) * 100 for c in conds], 1)
    dl = np.stack([ldiff(lg[:, names.index(c)], y) - ldiff(base, y) for c in conds], 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    x = np.arange(len(conds))
    for ax, v, lab in ((axes[0], ch, "Token change rate (%)"), (axes[1], dl, "Logit-difference change")):
        m, e = ms(v)
        ax.bar(x, m, width=0.42, color=P.BLUE, zorder=3)
        ax.errorbar(x, m, yerr=e, fmt="none", ecolor=P.INK2, elinewidth=1, capsize=2, zorder=4)
        P.hline(ax, 0, color=P.AXIS)
        ax.set_ylabel(lab)
        for xi, mi, ei in zip(x, m, e):
            ax.annotate(f"{mi:.1f}", (xi, mi + (ei if mi >= 0 else -ei)), textcoords="offset points", xytext=(0, 3 if mi >= 0 else -10),
                        ha="center", fontsize=7, color=P.INK2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace(">", "→") for c in conds], rotation=40, ha="right")
    axes[0].set_title(f"Attention blocking in all layers, categorical prompt (n={lg.shape[0]})")
    fig.tight_layout()
    P.save(fig, FIGURES / "fig21_attention_blocking_categorical.png")
    S["attn_cat"] = {"n": lg.shape[0], "noise_floor_change_pct": float(100 * (y != D["clean"].argmax(-1)).mean())}
    for i, c in enumerate(conds):
        S["attn_cat"][c] = {"change_pct": float(ch[:, i].mean()), "dld": float(dl[:, i].mean())}

# =====================================================================================================
# Decoding and variance partitioning (Figs. 6, 13d, 14)
# =====================================================================================================
if (RESULTS / "probe" / "probe.json").exists():
    R = json.load(open(RESULTS / "probe" / "probe.json"))
    L = sorted({r["layer"] for r in R["results"]})
    x = np.arange(len(L))
    tab = {(r["pos"], r["layer"]): r for r in R["results"]}
    ok = lambda p: all((p, l) in tab for l in L)

    def series(pos, key, sub=None):
        return np.array([(tab[(pos, l)][key] if sub is None else tab[(pos, l)][key][sub]) for l in L])

    def unique(pos, k):
        return np.clip(series(pos, "r2_both", k) - R["r2_base"][k], 0, None)

    inter = [p for p in (f"INT{i}" for i in range(10)) if ok(p)]
    main = [p for p in ["PANL+1", "FCC", "QTT", "CC", "PANL"] if ok(p)]
    MK = {"PANL": "o", "CC": "s", "PANL+1": "^", "FCC": "v", "QTT": "X", "AC": "D"}
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 10.4), sharex=True)
    ax = axes[0]
    for i, p in enumerate(inter):
        ax.plot(x, series(p, "auroc"), color=P.AXIS, lw=1.0, zorder=1, label="intermediate tokens (first token of each class name)" if i == 0 else None)
    for p in main:
        P.line(ax, x, series(p, "auroc"), color=P.POS_COLOR[p], label=P.POS_LABEL[p], ms=4, marker=MK[p])
    P.hline(ax, R["auroc_verbal_conf"], color=P.VIOLET, lw=1.2, label=f"verbal confidence itself ({R['auroc_verbal_conf']:.2f})")
    P.hline(ax, R["auroc_mean_logprob"], color=P.GREEN, lw=1.2, label=f"mean answer log-prob ({R['auroc_mean_logprob']:.2f})")
    ax.set_ylabel("AUROC (answer correctness)")
    h, l_ = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l_[::-1], loc="lower right", bbox_to_anchor=(1.0, 0.17), ncol=1, fontsize=7)
    ax.set_title(f"Linear decoding from the residual stream (n={R['n']} trials, 5-fold CV)")
    ax = axes[1]
    for p in inter:
        ax.plot(x, np.clip(series(p, "r2_a1"), 0, None), color=P.AXIS, lw=1.0, zorder=1)
    for p in main:
        P.line(ax, x, np.clip(series(p, "r2_a1"), 0, None), color=P.POS_COLOR[p], label=P.POS_LABEL[p], ms=4, marker=MK[p])
    ax.set_ylabel("R² (verbal confidence)\nridge α=1, clipped at 0")
    ax = axes[2]
    for p in [q for q in ["CC", "PANL"] if ok(q)]:
        P.line(ax, x, unique(p, "mean"), color=P.POS_COLOR[p], label=P.POS_LABEL[p], ms=4, marker=MK[p])
    ax.set_ylabel(f"R² unique to activations\n(mean log-prob alone: {R['r2_base']['mean']:.3f})")
    h, l_ = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l_[::-1], loc="upper left")
    P.layer_axis(ax, L)
    fig.tight_layout()
    P.save(fig, FIGURES / "fig06_decoding.png")

    if ok("PANL"):
        # Fig. 14: unique variance at PANL vs each baseline and all six combined
        fig, ax = plt.subplots(figsize=(7.4, 3.9))
        cols = {"mean": P.BLUE, "min": P.ORANGE, "max": P.AQUA, "last": P.YELLOW, "first": P.MAGENTA, "var": P.GREEN}
        for k, col in cols.items():
            ax.plot(x, unique("PANL", k), color=col, lw=1.3, label=f"vs answer {k} log-prob (R² {R['r2_base'][k]:.3f})")
        ax.plot(x, unique("PANL", "all"), color=P.INK, lw=2.4, label=f"vs all six combined (R² {R['r2_base']['all']:.3f})")
        P.layer_axis(ax, L)
        ax.set_ylabel("R² unique to PANL activations")
        ax.set_title("Variance in verbal confidence explained by PANL beyond log-prob baselines")
        ax.legend(loc="upper left", ncol=1, fontsize=7)
        fig.tight_layout()
        P.save(fig, FIGURES / "fig14_variance_partitioning.png")

    if all(ok(p) for p in ["AC", "PANL", "PANL+1"]):
        # Fig. 13d: tuned-alpha ridge at AC vs PANL vs PANL+1
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        for p in ["AC", "PANL+1", "PANL"]:
            P.line(ax, x, np.clip(series(p, "r2_tuned"), 0, None), color=P.POS_COLOR[p], label=P.POS_LABEL[p], ms=4, marker=MK[p])
        P.layer_axis(ax, L)
        ax.set_ylabel("R² (verbal confidence)\nridge, α chosen on training folds")
        ax.set_title("Decodability of verbal confidence: answer colon vs post-answer positions")
        h, l_ = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l_[::-1], loc="upper left")
        fig.tight_layout()
        P.save(fig, FIGURES / "fig13d_decoding_AC.png")

    S["probe"] = {"n": R["n"], "layers": L, "r2_base": R["r2_base"], "auroc_verbal_conf": R["auroc_verbal_conf"],
                  "auroc_mean_logprob": R["auroc_mean_logprob"], "r_logprob_conf": R["r_logprob_conf"], "n_results": len(R["results"])}
    for p in ["PANL", "PANL+1", "CC", "FCC", "QTT", "AC", "A_first", "A_last"]:
        if not ok(p):
            continue
        a, r1, rt = series(p, "auroc"), series(p, "r2_a1"), series(p, "r2_tuned")
        first_pos = next((int(l) for l, v in zip(L, r1) if v > 0.05), None)
        first_auc = next((int(l) for l, v in zip(L, a) if v > 0.75), None)
        S["probe"][p] = {"auroc": a.tolist(), "r2_a1": r1.tolist(), "r2_tuned": rt.tolist(), "r2_unique_mean": unique(p, "mean").tolist(),
                         "r2_unique_all6": unique(p, "all").tolist(), "peak_auroc": peak(L, a), "peak_r2_a1": peak(L, r1), "peak_r2_tuned": peak(L, rt),
                         "peak_unique_all6": peak(L, unique(p, "all")), "peak_unique_mean": peak(L, unique(p, "mean")),
                         "first_layer_r2_a1_above_0.05": first_pos, "first_layer_auroc_above_0.75": first_auc}

# =====================================================================================================
# Robustness: single newline after the answer (PANL token "\n" instead of "\n\n")
# =====================================================================================================
def key_numbers(tag):
    out = {}
    if have("patch" + tag):
        D = load("patch" + tag)
        L, names = D["layers"], D["names"]
        y = D["clean"].argmax(-1)
        M = mets(D["logits"], D["clean"])
        cl, co = ldiff(D["clean"], y).mean(), ldiff(D["corrupt"], y).mean()
        cc, cx = MID[y].mean(), MID[D["corrupt"].argmax(-1)].mean()
        d = {"n": len(y), "clean_ld": float(cl), "corrupt_ld": float(co), "corrupt_conf": float(cx), "clean_conf": float(cc),
             "corrupt_change": float((D["corrupt"].argmax(-1) != y).mean())}
        for pos in ["PANL", "CC", "PANL+1"]:
            v = names.index(pos)
            rec_ld = 100 * (M["ld"][:, v].mean(0) - co) / (cl - co)
            rec_conf = 100 * (M["conf"][:, v].mean(0) - cx) / (cc - cx)
            d[pos] = {"peak_ld_recovery": peak(L, rec_ld), "peak_conf_recovery": peak(L, rec_conf),
                      "recovery_ld_pct": rec_ld.tolist(), "recovery_conf_pct": rec_conf.tolist(),
                      "conf": M["conf"][:, v].mean(0).tolist(), "ld": M["ld"][:, v].mean(0).tolist(),
                      "change_pct": (100 * M["change"][:, v].mean(0)).tolist()}
        out["patch"] = d
    if have("noise" + tag):
        D = load("noise" + tag)
        L, names = D["layers"], D["names"]
        M = mets(D["logits"], D["ref"])
        d = {"n": D["logits"].shape[0]}
        for pos in names:
            v = names.index(pos)
            ch, dl = 100 * M["change"][:, v].mean(0), M["dld"][:, v].mean(0)
            d[pos] = {"peak_change": peak(L, ch), "peak_dld": peak(L, dl, "min"), "change_pct": ch.tolist(), "dld": dl.tolist()}
        out["noise"] = d
    if have("attn_cat" + tag):
        D = load("attn_cat" + tag)
        names, lg = D["names"], D["logits"]
        base = lg[:, 0]
        y = base.argmax(-1)
        out["attn_cat"] = {c: {"change_pct": float(100 * (lg[:, names.index(c)].argmax(-1) != y).mean()),
                               "dld": float((ldiff(lg[:, names.index(c)], y) - ldiff(base, y)).mean())} for c in names[1:]}
        out["attn_cat"]["n"] = lg.shape[0]
    return out


for tag in ["_nl1", "_p0"]:  # _nl1: single newline after the answer; _p0: patching trials selected by Phase-0 confidence
    kn = key_numbers(tag)
    if kn:
        S["robust" + tag] = kn
S["robust_main_reference"] = key_numbers("")

if (RESULTS / "ood_natural.json").exists():
    S["ood_natural"] = json.load(open(RESULTS / "ood_natural.json"))

json.dump(S, open(RESULTS / "summary.json", "w"), indent=1)
print("summary keys:", list(S))
