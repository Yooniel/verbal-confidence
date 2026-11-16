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
from vc.metrics import have as _have
from vc.metrics import ldiff, mets, ms, noise_floor, peak
from vc.metrics import load as _load
from vc.prompts import CLASSES, HIGH_CLASSES, LOW_CLASSES, MIDPOINTS

P.setup()
os.makedirs(FIGURES, exist_ok=True)
MID = np.array(MIDPOINTS)
S = {}  # summary


def have(exp):
    return _have(RESULTS, exp)


def load(exp):
    return _load(RESULTS, exp)


trials = read_jsonl(DATA / "trials.jsonl") if (DATA / "trials.jsonl").exists() else None

# =====================================================================================================
# Behaviour (Fig. 9)
# =====================================================================================================
PAPER_FIG9_COUNTS = [119, 96, 259, 2, 1, 0, 304, 4, 1632, 5441]  # paper Fig. 9B, printed on the bars
PAPER_FIG9_ACC = {0: 0.17, 1: 0.21, 2: 0.26, 6: 0.375, 8: 0.63, 9: 0.855}  # paper Fig. 9A, read off the plot

if (RESULTS / "behav" / "summary.json").exists():
    B = json.load(open(RESULTS / "behav" / "summary.json"))
    S["behaviour"] = B
    S["behaviour"]["paper_fig9"] = {"class_counts": dict(zip(CLASSES, PAPER_FIG9_COUNTS)), "class_accuracy_read_off": {CLASSES[c]: a for c, a in PAPER_FIG9_ACC.items()}}
    pn = np.array(PAPER_FIG9_COUNTS)
    # two phases: confidence stated together with the answer (Phase 0; what the paper's Fig. 9 matches) and
    # confidence for the answer fed back in the Fig. 8 prompt (Phase 1; what every experiment acts on)
    PH = [("Confidence stated with the answer (Phase 0)", B["phase0"]["class_counts"], B["phase0"]["class_accuracy"], B["phase0"]["ece"], B["phase0"]["auroc_verbal_conf"], True),
          ("Confidence for the fed-back answer (Phase 1)", B["class_counts"], B["class_accuracy"], B["ece"], B["auroc_verbal_conf"], False)]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), gridspec_kw={"width_ratios": [1, 1.3]})
    for r, (title, counts, accs, ece_, auc_, with_paper) in enumerate(PH):
        n = np.array([counts[c] for c in CLASSES])
        acc = np.array([accs[c] if accs[c] is not None else np.nan for c in CLASSES])
        keep = n >= 20
        ax = axes[r, 0]
        ax.plot([0, 1], [0, 1], color=P.AXIS, lw=1.2, zorder=1)
        ax.scatter(MID[keep], acc[keep], s=18 + 260 * n[keep] / n.max(), color=P.BLUE, edgecolor=P.SURFACE, linewidth=1.2, zorder=3, label="this replication")
        if with_paper:
            pc = sorted(PAPER_FIG9_ACC)
            ax.scatter(MID[pc], [PAPER_FIG9_ACC[c] for c in pc], s=46, facecolor="none", edgecolor=P.INK2, linewidth=1.4, zorder=4, label="paper, Fig. 9")
            ax.legend(loc="upper left")
        ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.02)
        ax.set_xlabel("Verbal confidence (class midpoint)"), ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.text(0.97, 0.05, f"ECE {ece_:.2f}, AUROC {auc_:.2f}" + ("\npaper: ECE 0.12, AUROC 0.71" if with_paper else ""),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=P.INK2)
        ax.grid(True, axis="both")
        ax = axes[r, 1]
        x = np.arange(10)
        if with_paper:
            ax.bar(x - 0.2, 100 * n / n.sum(), width=0.36, color=P.BLUE, zorder=3, label="this replication")
            ax.bar(x + 0.2, 100 * pn / pn.sum(), width=0.36, color=P.MUTED, zorder=3, label="paper, Fig. 9")
            ax.legend(loc="upper left")
        else:
            ax.bar(x, 100 * n / n.sum(), width=0.55, color=P.BLUE, zorder=3)
        for i, k in enumerate(n):
            if k:
                ax.annotate(f"{100*k/n.sum():.0f}" if 100 * k / n.sum() >= 1 else "<1", (i - (0.2 if with_paper else 0), 100 * k / n.sum()),
                            textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5, color=P.INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES if r == 1 else [""] * 10, rotation=40, ha="right")
        ax.set_ylabel("Trials (%)")
        ax.set_title("Share of trials in each confidence class")
    fig.suptitle(f"Gemma 3 27B on TriviaQA: n = {B['n_valid']:,}, accuracy {100*B['accuracy']:.1f}% (GPT-4o-mini grader; paper 77.4%)" if B.get("grader") == "gpt"
                 else f"Gemma 3 27B on TriviaQA: n = {B['n_valid']:,}, accuracy {100*B['accuracy']:.1f}%", fontsize=10.5)
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
# Main configuration: the one closest to the paper's text (C.1.4) - trials chosen by Phase-0 confidence, every
# answer position replaced by a mean over all 100 calibration prompts. Falls back to the first reading
# (Phase-1 top-3 trials, per-answer-position means) if that run is not there.
PATCH_MAIN = next((e for e in ("patch_p0_cabs", "patch") if have(e)), None)
if PATCH_MAIN:
    D = load(PATCH_MAIN)
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
    who = "trials with a top-3 class in Phase 0" if "p0" in PATCH_MAIN else "high-confidence trials"
    axes[0].set_title(f"Activation patching after answer-token corruption (n={len(y)} {who})")
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
    S["patch"] = {"n": len(y), "config": PATCH_MAIN, "clean": {"ld": float(clean_ld.mean()), "conf": float(clean_conf.mean())},
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


# _nl1: single newline after the answer; _p0: patching trials selected by Phase-0 confidence;
# _cabs / _cglobal: answer corruption by absolute-position / global mean embeddings (interv.py --corrupt)
for tag in ["_nl1", "_p0", "_cabs", "_cglobal", "_p0_cabs", "_p0_cglobal"]:
    kn = key_numbers(tag)
    if kn:
        S["robust" + tag] = kn
S["robust_main_reference"] = key_numbers("")


# =====================================================================================================
# Sensitivity: what the sizes of the effects depend on (README "What the effect sizes depend on")
# =====================================================================================================
def top2_margin(z):
    """Logit gap between the two most likely classes."""
    s = np.sort(z, -1)
    return s[..., -1] - s[..., -2]


def boot_ci(fn, idx, B=1000):
    """95% bootstrap interval of fn(trial indices), resampling trials."""
    r = np.random.default_rng(0)
    v = [fn(idx[r.integers(0, len(idx), len(idx))]) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


SENS = {}
SENS_PATCH = ("patch_cabs", "patch_p0_cabs") if have("patch_cabs") and have("patch_p0_cabs") else ("patch", "patch_p0")
if all(have(e) for e in SENS_PATCH) and trials is not None:
    SENS_LAYER = 25 if SENS_PATCH[0] == "patch_cabs" else 22  # the layer where PANL recovery peaks in these runs
    # Restoring PANL there: recovery as a function of how far the corruption pushes the model towards "No chance"
    # and of the answer length. Main and Phase-0-selected runs pooled, trials whose clean class is in the top three.
    rows = []
    for exp in SENS_PATCH:
        D = load(exp)
        v, li = D["names"].index("PANL"), list(D["layers"]).index(SENS_LAYER)
        y, pat, ar = D["clean"].argmax(-1), D["logits"][:, v, li], np.arange(len(D["idx"]))
        rows.append({"y": y, "ldc": ldiff(D["clean"], y), "ldx": ldiff(D["corrupt"], y), "ldp": ldiff(pat, y),
                     "cc": MID[y], "cx": MID[D["corrupt"].argmax(-1)], "cp": MID[pat.argmax(-1)],
                     "restored": (pat.argmax(-1) == y).astype(float), "margin": D["corrupt"][:, 0] - D["corrupt"][ar, y],
                     "n_a": np.array([trials[i]["n_a"] for i in D["idx"]])})
    T = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    rec_ld = lambda i: 100 * (T["ldp"][i].mean() - T["ldx"][i].mean()) / (T["ldc"][i].mean() - T["ldx"][i].mean())
    rec_cf = lambda i: 100 * (T["cp"][i].mean() - T["cx"][i].mean()) / (T["cc"][i].mean() - T["cx"][i].mean())

    def group(i):
        return {"n": len(i), "no_chance_margin": float(T["margin"][i].mean()), "answer_tokens": float(T["n_a"][i].mean()),
                "clean_conf": float(T["cc"][i].mean()), "ld_recovery_pct": float(rec_ld(i)), "ld_recovery_ci": boot_ci(rec_ld, i),
                "conf_recovery_pct": float(rec_cf(i)), "conf_recovery_ci": boot_ci(rec_cf, i),
                "patched_conf": float(T["cp"][i].mean()), "clean_class_restored_pct": float(100 * T["restored"][i].mean())}

    top = np.where(T["y"] >= 7)[0]
    SENS["patch_PANL"] = {
        "runs": list(SENS_PATCH), "layer": SENS_LAYER,
        "all": group(top),
        "by_margin_quartile": [group(i) for i in np.array_split(top[np.argsort(T["margin"][top], kind="stable")], 4)],
        "by_answer_tokens": {lab: group(top[(T["n_a"][top] >= a) & (T["n_a"][top] <= b)])
                             for lab, a, b in (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 10**6))},
    }

if have("swap"):
    # Same-confidence swaps at PANL, L25: token change against the recipient's distance from a class boundary
    D = load("swap")
    kind = np.array([r["kind"] for r in D["recs"]])
    li, mg = list(D["layers"]).index(25), top2_margin(D["clean"])
    SENS["swap_same_confidence_PANL_L25"] = {}
    for k in ("low", "high"):
        v, m = D["names"].index(f"PANL|{k}"), kind == k
        ch = D["logits"][:, v, li].argmax(-1) != D["ref"][:, v, li].argmax(-1)
        SENS["swap_same_confidence_PANL_L25"][f"{k}->{k}"] = {
            "median_top2_margin": float(np.median(mg[m])), "pct_within_1_logit": float(100 * (mg[m] < 1).mean()),
            "change_pct_by_top2_margin": {lab: {"n": int((m & s).sum()), "change_pct": float(100 * ch[m & s].mean())}
                                          for lab, s in (("<1", mg < 1), ("1-3", (mg >= 1) & (mg < 3)), (">=3", mg >= 3))}}

if have("attn_cat"):
    # Attention blocking, categorical prompt: same-pass reference vs a separately computed clean run
    D = load("attn_cat")
    names, lg = D["names"], D["logits"]
    same, sep, mg = lg[:, 0].argmax(-1), D["clean"].argmax(-1), top2_margin(D["clean"])
    SENS["attn_cat"] = {"unedited_rows_vs_clean_run_change_pct": float(100 * (same != sep).mean()),
                        "pct_within_1_logit": float(100 * (mg < 1).mean()), "conditions": {}}
    for c in names[1:]:
        a = lg[:, names.index(c)].argmax(-1)
        SENS["attn_cat"]["conditions"][c] = {
            "change_pct_vs_same_pass": float(100 * (a != same).mean()), "change_pct_vs_clean_run": float(100 * (a != sep).mean()),
            "change_pct_within_1_logit": float(100 * (a != same)[mg < 1].mean()),
            "change_pct_beyond_1_logit": float(100 * (a != same)[mg >= 1].mean())}

if have("steer"):
    # Steering at PANL: change in confidence against the scale, separately for high- and low-confidence trials
    D = load("steer")
    L, names = list(D["layers"]), D["names"]
    dc = MID[D["logits"].argmax(-1)] - MID[D["ref"].argmax(-1)]
    hi = MID[D["clean"].argmax(-1)] > 0.5
    SENS["steer_PANL_dose_response"] = {
        f"{sign}_L{lay}": {grp: {sc: float(dc[m][:, names.index(f"PANL|{sign}|{sc}"), L.index(lay)].mean()) for sc in ("1", "2", "5", "10")}
                           for grp, m in (("all", np.ones(len(hi), bool)), ("high_conf_trials", hi), ("low_conf_trials", ~hi))}
        for sign, lay in (("high", 24), ("low", 22))}
# Steering at PANL with directions built from other trials (subset of layers): random tie-break draws and the
# "most extreme trials first" tie-break, all with the current correctness labels
alt = {}
for tag in ("_rand1", "_rand2", "_extreme_gpt"):
    if have("steer" + tag):
        D = load("steer" + tag)
        L2, nm = list(D["layers"]), D["names"]
        dc = MID[D["logits"].argmax(-1)] - MID[D["ref"].argmax(-1)]
        alt[tag[1:]] = {"layers": L2, "n": dc.shape[0],
                        "dconf": {n: dc[:, nm.index(n)].mean(0).tolist() for n in nm},
                        "peak": {n: peak(L2, dc[:, nm.index(n)].mean(0), "max" if "|high|" in n else "min") for n in nm}}
if alt:
    SENS["steer_PANL_other_vector_trials"] = alt
if alt and have("steer"):
    # figure: the main run (random tie-break, draw 0) next to the other choices of the 25 + 25 trials
    Dm = load("steer")
    Lm, nmm = list(Dm["layers"]), Dm["names"]
    dm = MID[Dm["logits"].argmax(-1)] - MID[Dm["ref"].argmax(-1)]
    sub = alt[next(iter(alt))]["layers"]
    series = [("ties broken at random, draw 0 (main run)", P.BLUE,
               {n: dm[:, nmm.index(n)].mean(0)[[Lm.index(l) for l in sub]] for n in nmm if n.startswith("PANL|")})]
    for key, lab, col in (("rand1", "ties broken at random, draw 1", P.AQUA), ("rand2", "ties broken at random, draw 2", P.YELLOW),
                          ("extreme_gpt", "most extreme trials first", P.ORANGE)):
        if key in alt:
            series.append((lab, col, {n: np.array(v) for n, v in alt[key]["dconf"].items()}))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    xs = np.arange(len(sub))
    for ax, sc in zip(axes, ("2", "5")):
        P.hline(ax, 0, color=P.AXIS)
        for lab, col, d in series:
            P.line(ax, xs, d[f"PANL|high|{sc}"], color=col, label=lab)
            P.line(ax, xs, d[f"PANL|low|{sc}"], color=col, hollow=True, ls=(0, (3, 1.5)), lw=1.6)
        P.layer_axis(ax, sub)
        ax.set_title(f"Scale {sc}")
    axes[0].set_ylabel("Δ confidence")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, title="Trials that define the steering direction", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Steering at PANL with directions built from different trials: high vector (solid), low vector (dashed)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    P.save(fig, FIGURES / "fig02b_steering_vector_trials.png")
    SENS["steer_PANL_main_on_same_layers"] = {"layers": sub, "dconf": {n: v.tolist() for n, v in series[0][2].items()}}
for key, fname in (("steering_direction_cosines", "vector_sensitivity.json"), ("attention_implementation_noise", "numtest_eager.json"),
                   ("prompt_format_pilot", "pilot_format/summary.json")):
    if (RESULTS / fname).exists():
        SENS[key] = json.load(open(RESULTS / fname))
S["sensitivity"] = SENS

if (RESULTS / "ood_natural.json").exists():
    S["ood_natural"] = json.load(open(RESULTS / "ood_natural.json"))

json.dump(S, open(RESULTS / "summary.json", "w"), indent=1)
print("summary keys:", list(S))
