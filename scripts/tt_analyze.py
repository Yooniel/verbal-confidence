"""Two-turn chat format: figures (figures/tt_*.png), results/tt/summary.json and results/tt/comparison.md.

Run with VC_FMT=tt after the two-turn stages; missing experiments are skipped. The numbers of the main
(raw-format) replication are read from results/summary.json for the side-by-side table.

Position names (see vc.prompts.encode_tt): PANL is the <end_of_turn> token that closes the model's answer turn
(the first token after the answer, like the paper's post-answer newline) and CC is the last prompt token (the
one that emits the confidence class, like the paper's confidence colon).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib.pyplot as plt

from vc import plotting as P
from vc.data import read_jsonl
from vc.env import DATA, DATA_RAW, FIGURES, FMT, RESULTS, RESULTS_RAW
from vc.metrics import have as _have
from vc.metrics import ldiff, mets, ms, noise_floor, peak
from vc.metrics import load as _load
from vc.prompts import CLASSES, HIGH_CLASSES, LOW_CLASSES, MIDPOINTS

assert FMT == "tt", "run with VC_FMT=tt"
P.setup()
os.makedirs(FIGURES, exist_ok=True)
MID = np.array(MIDPOINTS)
S = {}
RAW = json.load(open(RESULTS_RAW / "summary.json"))
have = lambda exp: _have(RESULTS, exp)  # noqa: E731
load = lambda exp: _load(RESULTS, exp)  # noqa: E731
trials = read_jsonl(DATA / "trials.jsonl")

LAB = {
    "PANL": "end-of-turn after the answer",
    "PANL+1": "newline after it",
    "SOT": "<start_of_turn>",
    "USER": '"user"',
    "UNL": "newline before the request",
    "POST5": "all five tokens between answer and request",
    "REQ0": "first request token",
    "INT4": "request token (5th class name)",
    "REQ": "all request tokens",
    "EOT2": "end-of-turn after the request",
    "TAIL4": "the four tokens before the last",
    "CC-1": '"model" (token before the last)',
    "CC": "last prompt token",
    "AC": "newline before the answer",
    "A_first": "first answer token",
    "A_last": "last answer token",
    "QTT": "3rd question token",
}
COL = {"PANL": P.BLUE, "PANL+1": P.AQUA, "SOT": P.YELLOW, "USER": P.MAGENTA, "UNL": P.GREEN, "POST5": P.INK, "REQ0": P.MUTED,
       "INT4": P.MUTED, "REQ": P.VIOLET, "EOT2": P.RED_LIGHT, "TAIL4": P.MAGENTA, "CC-1": P.RED, "CC": P.ORANGE, "AC": P.MAGENTA,
       "A_first": P.GREEN, "A_last": P.VIOLET, "QTT": P.AXIS}
MK = {"PANL": "o", "PANL+1": "^", "SOT": "v", "USER": "P", "UNL": "D", "POST5": "*", "REQ0": "X", "INT4": "x", "REQ": "h", "EOT2": "<",
      "TAIL4": ">", "CC-1": "d", "CC": "s", "AC": "D", "A_first": "p", "A_last": "p", "QTT": "X"}
POST = ["PANL", "PANL+1", "SOT", "USER", "UNL"]
# attention-blocking conditions of interv.py --exp attn_tt (and, where they exist, of the single-prompt attn_cat run)
ATT_LAB = {
    "CC>NL+1": "last token → second token after the answer",
    "CC>NL": "last token → first token after the answer",
    "CC>POST5": "last token → the five tokens between answer and request",
    "CC>A": "last token → answer tokens",
    "CC>Q": "last token → question tokens",
    "CC>Q+A": "last token → question and answer tokens",
    "ALL>NL": "all later tokens → first token after the answer",
    "ALL>POST5": "request and final tokens → the five tokens between answer and request",
    "ALL>last_A": "all later tokens → last answer token",
    "ALL>last_A_keepNL": "… except that the first token after the answer keeps that edge",
    "ALL>A": "all later tokens → all answer tokens",
    "ALL>A_keepNL": "… except that the first token after the answer keeps its edges",
    "ALL>A_keepPOST5": "… except that all five tokens after the answer keep theirs",
    "REQ>A": "request tokens → answer tokens",
    "TAIL>A": "the five final tokens → answer tokens",
    "ALL>A+POST5": "request and final tokens → answer tokens and the five tokens after them",
    "TAIL>A+POST5": "the five final tokens → answer tokens and the five tokens after them",
}


def at(L, curve, layer):
    return float(np.asarray(curve)[list(L).index(layer)])


def hl(z):
    """Continuous confidence readout: mean logit of the three top classes minus mean logit of the three bottom
    classes. The paper's metrics depend on the argmax class, and in the two-turn format the class logits are far
    more peaked than in the single-prompt format, so the same shift in the logits flips fewer argmax classes."""
    return z[..., 7:].mean(-1) - z[..., :3].mean(-1)


def hl_gap(clean):
    """Mean high-minus-low logit of high-confidence trials minus that of low-confidence trials (unedited runs)."""
    y = clean.argmax(-1)
    return float(hl(clean)[np.isin(y, HIGH_CLASSES)].mean() - hl(clean)[np.isin(y, LOW_CLASSES)].mean())


def margins(clean):
    srt = np.sort(clean, -1)
    return srt[:, -1] - srt[:, -2]


def suptitle(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="bold")


# =====================================================================================================
# Behaviour
# =====================================================================================================
B = json.load(open(RESULTS / "behav" / "summary.json"))
B["whitespace_tokens_dropped_pct"] = float(100 * np.mean([t["n_ws_tokens"] > 0 for t in trials if t.get("src", "val") == "val"]))
S["behaviour"] = B
RB = RAW["behaviour"]
# how peaked the class logits are in the two formats (all valid validation trials)
S["logit_margins"] = {}
for tag, T_ in (("two_turn", [t for t in trials if t.get("src", "val") == "val"]), ("single_prompt", read_jsonl(DATA_RAW / "trials.jsonl"))):
    z = np.array([t["p1_logits"] for t in T_])
    yz = z.argmax(-1)
    S["logit_margins"][tag] = {
        "median_top2_margin": float(np.median(margins(z))), "within_one_logit_pct": float(100 * (margins(z) < 1).mean()),
        "median_top2_margin_high_trials": float(np.median(margins(z)[yz >= 7])), "median_top2_margin_low_trials": float(np.median(margins(z)[yz <= 2])),
        "mean_logit_diff": float(ldiff(z, yz).mean()), "mean_high_minus_low_logit_high_trials": float(hl(z)[yz >= 7].mean()),
        "mean_high_minus_low_logit_low_trials": float(hl(z)[yz <= 2].mean())}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
ax = axes[0]
ax.plot([0, 1], [0, 1], color=P.AXIS, lw=1, zorder=1)
series = [("two-turn chat format", B["class_accuracy"], B["class_counts"], P.BLUE, "o"),
          ("single prompt, answer fed back (Phase 1)", RB["class_accuracy"], RB["class_counts"], P.ORANGE, "s"),
          ("single prompt, stated with the answer (Phase 0)", RB["phase0"]["class_accuracy"], RB["phase0"]["class_counts"], P.AQUA, "^")]
for name, acc, cnt, col, mk in series:
    pts = [(MID[c], acc[CLASSES[c]]) for c in range(10) if cnt[CLASSES[c]] >= 20]
    P.line(ax, [p[0] for p in pts], [p[1] for p in pts], color=col, label=name, marker=mk)
PF = RB["paper_fig9"]["class_accuracy_read_off"]
ax.plot([MID[CLASSES.index(k)] for k in PF], list(PF.values()), ls="none", marker="D", ms=5, color=P.INK, markerfacecolor="none", label="paper, Fig. 9 (read off the plot)", zorder=4)
ax.set_xlabel("Stated confidence (class midpoint)"), ax.set_ylabel("Accuracy")
ax.set_title("Calibration (classes with at least 20 trials)")
ax.legend(loc="upper left")
ax = axes[1]
w = 0.27
for k, (name, acc, cnt, col, mk) in enumerate(series):
    tot = sum(cnt.values())
    ax.bar(np.arange(10) + (k - 1) * w, [100 * cnt[c] / tot for c in CLASSES], width=w - 0.03, color=col, label=name, zorder=3)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(CLASSES, rotation=35, ha="right")
pc_ = RB["paper_fig9"]["class_counts"]
ax.plot(np.arange(10), [100 * pc_[c] / sum(pc_.values()) for c in CLASSES], ls="none", marker="D", ms=5, color=P.INK, markerfacecolor="none", zorder=4)
ax.set_ylabel("Share of trials (%)")
ax.set_title("Confidence classes used")
fig.tight_layout()
P.save(fig, FIGURES / "tt01_behaviour.png")

# =====================================================================================================
# Steering
# =====================================================================================================
if have("steer"):
    D = load("steer")
    L, names = D["layers"], D["names"]
    M = mets(D["logits"], D["ref"])
    x = np.arange(len(L))
    yb = D["clean"].argmax(-1)
    S["steer"] = {"n": len(D["idx"]), "baseline_conf": float(MID[yb].mean()), "layers": list(L),
                  "noise_floor_change_pct_by_layer": noise_floor(D).tolist()}
    positions = list(dict.fromkeys(n.split("|")[0] for n in names))
    scales = {p: sorted({int(n.split("|")[2]) for n in names if n.split("|")[0] == p}) for p in positions}

    def curve(pos, d, s, mask=None):
        v = M["dconf"][:, names.index(f"{pos}|{d}|{s}")]
        return ms(v if mask is None else v[mask])

    order = [p for p in ["PANL", "PANL+1", "SOT", "USER", "UNL", "POST5", "CC", "CC-1", "A_last", "AC", "REQ0"] if p in positions]
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow), sharey=True, squeeze=False)
    rawL = RAW["steer"]["layers"]
    for ax, pos in zip(axes.ravel(), order):
        P.hline(ax, 0, color=P.AXIS)
        if pos in ("PANL", "CC"):  # the same position in the paper's single-prompt format, scale 5
            for d in ("high", "low"):
                ax.plot(np.arange(len(rawL)), RAW["steer"][pos]["scale5"][f"{d}_curve"], color=P.MUTED, lw=1.2, ls=(0, (1, 1.5)), zorder=2,
                        label="single-prompt format, scale 5" if d == "high" else None)
        for s in scales[pos]:
            strong = s == max(scales[pos])
            for d, col, lab in (("high", P.BLUE, "high-confidence vector"), ("low", P.RED, "low-confidence vector")):
                m, e = curve(pos, d, s)
                P.line(ax, x, m, e, color=col, label=f"{lab}, scale {s}", ls="-" if strong else (0, (3, 1.5)), hollow=not strong,
                       lw=1.8 if strong else 1.3, ms=3.6)
        P.layer_axis(ax, L, every=2)
        ax.set_title(LAB[pos], fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ confidence")
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    h, l_ = axes[0, 0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l_, loc="center", fontsize=8) if len(order) < nrow * ncol else axes[0, 0].legend(fontsize=6.5)
    suptitle(fig, f"Two-turn format: activation steering (n={S['steer']['n']} trials, baseline confidence {S['steer']['baseline_conf']:.2f}; mean ± SEM)")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt02_steering.png")

    for pos in positions:
        S["steer"][pos] = {}
        for sc in scales[pos]:
            hm, he = curve(pos, "high", sc)
            lm, le = curve(pos, "low", sc)
            S["steer"][pos][f"scale{sc}"] = {"high_peak": peak(L, hm, "max"), "low_peak": peak(L, lm, "min"), "high_curve": hm.tolist(),
                                             "low_curve": lm.tolist(), "high_sem": he.tolist(), "low_sem": le.tolist()}
        sc = max(scales[pos])
        for band, cls in (("high", HIGH_CLASSES), ("low", LOW_CLASSES)):
            mk = np.isin(yb, cls)
            for d in ("high", "low"):
                v = M["dconf"][mk][:, names.index(f"{pos}|{d}|{sc}")].mean(0)
                S["steer"][pos][f"scale{sc}_{d}_vector_on_{band}_trials_peak"] = peak(L, v, "max" if d == "high" else "min")

    # continuous readout (shift of the high-minus-low logit, as a share of the gap between unedited high- and
    # low-confidence trials), for this format and for the single-prompt run
    DR = _load(RESULTS_RAW, "steer")
    cont = {}
    for tag, DD in (("two_turn", D), ("single_prompt", DR)):
        gap = hl_gap(DD["clean"])
        dh = (hl(DD["logits"]) - hl(DD["ref"])) / gap  # [n, variant, layer]
        cont[tag] = {"gap_logits": gap, "median_top2_margin": float(np.median(margins(DD["clean"]))), "layers": list(DD["layers"]), "names": DD["names"], "d": dh}
        S["steer"][f"hl_{tag}"] = {"gap_logits": gap, "median_top2_margin": cont[tag]["median_top2_margin"]}
        for nm in DD["names"]:
            pos, d, sc = nm.split("|")
            m = dh[:, DD["names"].index(nm)].mean(0)
            S["steer"][f"hl_{tag}"][nm] = {"curve_pct_of_gap": (100 * m).tolist(), "peak_pct_of_gap": peak(DD["layers"], 100 * m, "max" if d == "high" else "min")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    for ax, pos, title in zip(axes, ["PANL", "POST5", "CC"], ["First token after the answer", "All five tokens between answer and request (two-turn only)", "Last prompt token"]):
        P.hline(ax, 0, color=P.AXIS)
        for tag, lab, ls, lw, hollow in (("single_prompt", "single prompt", (0, (3, 1.5)), 1.4, True), ("two_turn", "two-turn", "-", 2.0, False)):
            c = cont[tag]
            for d, col in (("high", P.BLUE), ("low", P.RED)):
                nm = f"{pos}|{d}|5"
                if nm in c["names"]:
                    v = 100 * c["d"][:, c["names"].index(nm)]
                    P.line(ax, np.arange(len(c["layers"])), v.mean(0), P.sem(v, 0), color=col, label=f"{lab}, {d}-confidence vector", ls=ls, lw=lw, hollow=hollow, ms=4)
        P.layer_axis(ax, L, every=2)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Shift of the high-minus-low logit\n(% of the high–low trial gap)")
    axes[0].legend(loc="upper left", fontsize=7)
    suptitle(fig, "Steering at scale 5 on a continuous readout: two-turn format vs single-prompt format")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt02b_steering_logits.png")

# =====================================================================================================
# Patching
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
    flat = [p for p in ["SOT", "USER", "UNL", "REQ0", "CC-1"] if p in names]  # single tokens that recover nothing
    cols = [("One token restored", [p for p in ["PANL+1", "CC", "PANL"] if p in names]),
            ("Several tokens restored together", [p for p in ["TAIL4", "REQ", "POST5"] if p in names])]
    rows = [("ld", "Logit difference", clean_ld.mean(), corr_ld.mean(), 1.0),
            ("change", "Token change rate (%)", 0.0, 100 * corr_change.mean(), 100.0),
            ("conf", "Confidence", clean_conf.mean(), corr_conf.mean(), 1.0)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 8.8), sharex=True, sharey="row")
    for c, (title, poss) in enumerate(cols):
        for r, (k, lab, cl, co, sc) in enumerate(rows):
            ax = axes[r, c]
            P.hline(ax, cl, color=P.GREEN, label="clean")
            P.hline(ax, co, color=P.RED, label="corrupted (no patch)")
            if c == 0:
                for i, pos in enumerate(flat):
                    ax.plot(x, (sc * M[k][:, names.index(pos)]).mean(0), color=P.MUTED, lw=0.9, zorder=2,
                            label='each of: <start_of_turn>, "user", newline before the request,\nfirst request token, "model"' if i == 0 else None)
            for pos in poss:
                m, e = ms(sc * M[k][:, names.index(pos)])
                P.line(ax, x, m, e, color=COL[pos], label=LAB[pos], marker=MK[pos], ms=4.5)
            if c == 0:
                ax.set_ylabel(lab)
        axes[0, c].set_title(title)
        h, l_ = axes[0, c].get_legend_handles_labels()
        axes[0, c].legend(h[::-1], l_[::-1], loc="upper left", fontsize=7.5)
        P.layer_axis(axes[-1, c], L)
    axes[0, 0].set_ylim(top=1.45 * clean_ld.mean())  # room for the legends above the clean line
    suptitle(fig, f"Two-turn format: activation patching after answer-token corruption (n={len(y)} high-confidence trials; mean ± SEM)")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt03_patching.png")

    S["patch"] = {"n": len(y), "clean": {"ld": float(clean_ld.mean()), "conf": float(clean_conf.mean())},
                  "corrupt": {"ld": float(corr_ld.mean()), "conf": float(corr_conf.mean()), "change": float(corr_change.mean()),
                              "class_counts": {CLASSES[c]: int((D["corrupt"].argmax(-1) == c).sum()) for c in range(10)}},
                  "layers": list(L),
                  "max_abs_AC_minus_ref": float(np.abs(D["logits"][:, names.index("AC")] - D["ref"][:, names.index("AC")]).max())}
    for pos in names:
        v = names.index(pos)
        cm, lm_, chm = M["conf"][:, v].mean(0), M["ld"][:, v].mean(0), M["change"][:, v].mean(0)
        rec_conf = 100 * (cm - corr_conf.mean()) / (clean_conf.mean() - corr_conf.mean())
        rec_ld = 100 * (lm_ - corr_ld.mean()) / (clean_ld.mean() - corr_ld.mean())
        S["patch"][pos] = {"conf": cm.tolist(), "ld": lm_.tolist(), "change": chm.tolist(), "recovery_conf_pct": rec_conf.tolist(),
                           "recovery_ld_pct": rec_ld.tolist(), "peak_conf_recovery": peak(L, rec_conf), "peak_ld_recovery": peak(L, rec_ld)}
    if "ood" in D:
        li = list(L).index(25)
        S["patch"]["ood_L25"] = {p: {"cos": float(D["ood"][:, i, li, 0].mean()), "nr": float(D["ood"][:, i, li, 1].mean())}
                                 for i, p in enumerate(D["ood_pos"]) if D["ood"][:, i, li, 1].any()}
    # trials that the corruption sends to a bottom-3 class: share back in a top-3 class after the patch at L25
    cq = D["corrupt"].argmax(-1)
    fell, li25 = cq <= 2, list(L).index(25)
    S["patch"]["corrupt"]["n_fell_to_bottom3"], S["patch"]["corrupt"]["n_stayed_top3"] = int(fell.sum()), int((cq >= 7).sum())
    S["patch"]["back_in_top3_at_L25_pct_of_fallen"] = {p: float(100 * (D["logits"][fell][:, names.index(p), li25].argmax(-1) >= 7).mean()) for p in names}

# Relay through the tokens between the answer and the last token, in both formats. The single-prompt run
# patch_p0_cabs_instr restores PANL, CC, PANL+1 and all instruction tokens between PANL and CC together.
RAW_INSTR = "patch_p0_cabs_instr"
if have("patch") and _have(RESULTS_RAW, RAW_INSTR):
    relay = {}
    for tag, DD in (("single_prompt", _load(RESULTS_RAW, RAW_INSTR)), ("two_turn", load("patch"))):
        yy = DD["clean"].argmax(-1)
        MM = mets(DD["logits"], DD["clean"])
        c0, c1 = MID[DD["corrupt"].argmax(-1)].mean(), MID[yy].mean()
        l0, l1 = ldiff(DD["corrupt"], yy).mean(), ldiff(DD["clean"], yy).mean()
        relay[tag] = {"layers": list(DD["layers"]), "n": len(yy), "clean_conf": float(c1), "corrupt_conf": float(c0), "clean_ld": float(l1), "corrupt_ld": float(l0)}
        for pos in DD["names"]:
            v = DD["names"].index(pos)
            relay[tag][pos] = {"recovery_conf_pct": (100 * (MM["conf"][:, v].mean(0) - c0) / (c1 - c0)).tolist(),
                               "recovery_ld_pct": (100 * (MM["ld"][:, v].mean(0) - l0) / (l1 - l0)).tolist()}
            relay[tag][pos]["peak_conf_recovery"] = peak(DD["layers"], relay[tag][pos]["recovery_conf_pct"])
            relay[tag][pos]["peak_ld_recovery"] = peak(DD["layers"], relay[tag][pos]["recovery_ld_pct"])
    S["patch_relay"] = relay
    SHOW = {"single_prompt": [("PANL", "post-answer newline (PANL)", P.BLUE, "o"), ("INSTR", "all instruction tokens between PANL and CC", P.VIOLET, "h"),
                              ("CC", "confidence colon (CC, last token)", P.ORANGE, "s")],
            "two_turn": [("POST5", "the five tokens between answer and request", P.INK, "*"), ("PANL", "end-of-turn after the answer alone", P.BLUE, "o"),
                         ("REQ", "all request tokens", P.VIOLET, "h"), ("TAIL4", "the four tokens before the last", P.MAGENTA, ">"),
                         ("CC", "last prompt token", P.ORANGE, "s")]}
    TITLE = {"single_prompt": "Single prompt (the paper's format)", "two_turn": "Two-turn chat format"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True, sharey="row")
    for c, tag in enumerate(("single_prompt", "two_turn")):
        r_ = relay[tag]
        xx = np.arange(len(r_["layers"]))
        for r, (key, lab) in enumerate((("recovery_conf_pct", "Confidence recovered (%)"), ("recovery_ld_pct", "Logit difference recovered (%)"))):
            ax = axes[r, c]
            P.hline(ax, 0, color=P.AXIS)
            for pos, name, col, mk in SHOW[tag]:
                P.line(ax, xx, r_[pos][key], color=col, label=name, marker=mk, ms=4.5)
            if c == 0:
                ax.set_ylabel(lab)
        axes[0, c].set_title(f"{TITLE[tag]}: clean confidence {r_['clean_conf']:.2f}, corrupted {r_['corrupt_conf']:.2f} (n={r_['n']})", fontsize=9)
        axes[1, c].legend(loc="upper left", fontsize=7.5)
        P.layer_axis(axes[1, c], r_["layers"])
    suptitle(fig, "Where the answer's evaluation sits, layer by layer: clean activations restored after answer-token corruption")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt03b_patching_relay.png")

# =====================================================================================================
# Noising
# =====================================================================================================
if have("noise"):
    D = load("noise")
    L, names = D["layers"], D["names"]
    M = mets(D["logits"], D["ref"])
    y = D["clean"].argmax(-1)
    clean_ld = ldiff(D["clean"], y)
    x = np.arange(len(L))
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True)
    P.hline(axes[0], 0, color=P.AXIS)
    for pos in [p for p in ["REQ0", "AC", "PANL+1", "UNL", "CC", "POST5", "PANL"] if p in names]:
        for ax, k, sc in ((axes[0], "dld", 1.0), (axes[1], "change", 100.0)):
            m, e = ms(sc * M[k][:, names.index(pos)])
            P.line(ax, x, m, e, color=COL[pos], label=LAB[pos], marker=MK[pos], ms=4, lw=2.0 if pos in ("PANL", "POST5", "CC") else 1.3)
    axes[0].set_ylabel(f"Change in logit difference\n(clean: {clean_ld.mean():.1f})"), axes[1].set_ylabel("Token change rate (%)")
    P.layer_axis(axes[1], L)
    h, l_ = axes[1].get_legend_handles_labels()
    axes[1].legend(h[::-1], l_[::-1], loc="upper left", fontsize=7)
    axes[0].set_title(f"Two-turn format: activation noising (mean ablation), n={len(y)} trials")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt04_noising.png")
    S["noise"] = {"n": len(y), "clean_ld": float(clean_ld.mean()), "layers": list(L), "noise_floor_change_pct_by_layer": noise_floor(D).tolist(),
                  "n_high": int(np.isin(y, HIGH_CLASSES).sum()), "n_low": int(np.isin(y, LOW_CLASSES).sum())}
    for pos in names:
        v = names.index(pos)
        ch, dl = 100 * M["change"][:, v].mean(0), M["dld"][:, v].mean(0)
        S["noise"][pos] = {"change_pct": ch.tolist(), "dld": dl.tolist(), "peak_change": peak(L, ch), "peak_dld": peak(L, dl, "min"),
                           "peak_dld_pct_of_clean": peak(L, 100 * dl / clean_ld.mean(), "min")}
        for band, cls in (("high", HIGH_CLASSES), ("low", LOW_CLASSES)):
            mk = np.isin(y, cls)
            if mk.sum() > 5:
                S["noise"][pos][f"change_pct_{band}"] = (100 * M["change"][mk][:, v].mean(0)).tolist()
    # by confidence band, layers 20-28, for this format and the single-prompt run
    S["noise"]["bands"] = {}
    for tag, DD in (("two_turn", D), ("single_prompt", _load(RESULTS_RAW, "noise"))):
        MM, yy = mets(DD["logits"], DD["ref"]), DD["clean"].argmax(-1)
        mid = [i for i, l in enumerate(DD["layers"]) if 20 <= l <= 28]
        for band, mk in (("high", yy >= 7), ("low", yy <= 2)):
            cl = ldiff(DD["clean"][mk], yy[mk]).mean()
            for pos in [q for q in ("PANL", "PANL+1", "POST5") if q in DD["names"]]:
                v = DD["names"].index(pos)
                S["noise"]["bands"][f"{tag}|{band}|{pos}"] = {
                    "n": int(mk.sum()), "max_change_pct_L20_28": float(100 * MM["change"][mk][:, v].mean(0)[mid].max()),
                    "min_dld_pct_of_clean_L20_28": float(100 * MM["dld"][mk][:, v].mean(0)[mid].min() / cl)}

# =====================================================================================================
# Swap
# =====================================================================================================
if have("swap"):
    D = load("swap")
    L, names = D["layers"], D["names"]
    M = mets(D["logits"], D["ref"])
    kind = np.array([r["kind"] for r in D["recs"]])
    x = np.arange(len(L))
    CONDS = [("high", "high", "high→high (same)", P.BLUE), ("high", "low", "high→low (cross)", P.ORANGE),
             ("low", "high", "low→high (cross)", P.AQUA), ("low", "low", "low→low (same)", P.YELLOW)]
    positions = [p for p in ["PANL", "POST5", "CC"] if any(n.startswith(p + "|") for n in names)]
    S["swap"] = {"n_high": int((kind == "high").sum()), "n_low": int((kind == "low").sum()), "layers": list(L),
                 "noise_floor_change_pct_by_layer": noise_floor(D).tolist(),
                 "match_stats": json.load(open(DATA / "swap_plan.json"))["match_stats"]}
    fig, axes = plt.subplots(2, len(positions), figsize=(4.6 * len(positions), 6.0), sharex=True, sharey="row", squeeze=False)
    for c, pos in enumerate(positions):
        for r, (k, lab, sc) in enumerate([("change", "Token change rate (%)", 100.0), ("dconf", "Confidence change", 1.0)]):
            ax = axes[r, c]
            P.hline(ax, 0, color=P.AXIS)
            for rk, dk, name, col in CONDS:
                m, e = ms(sc * M[k][kind == rk][:, names.index(f"{pos}|{dk}")])
                P.line(ax, x, m, e, color=col, label=name, hollow=(rk == dk), lw=1.5 if rk == dk else 2.0, ms=4)
            if c == 0:
                ax.set_ylabel(lab)
            if r == 0:
                ax.set_title(f"Swap at: {LAB[pos]}", fontsize=9)
        P.layer_axis(axes[-1, c], L, every=2)
    axes[1, 0].legend(loc="upper left", ncol=1, title="recipient→donor confidence", fontsize=7)
    suptitle(fig, f"Two-turn format: activation swap ({S['swap']['n_high']} high- and {S['swap']['n_low']} low-confidence recipients)")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt05_swap.png")
    for pos in positions:
        S["swap"][pos] = {}
        for rk, dk, name, _ in CONDS:
            sub = {k: M[k][kind == rk][:, names.index(f"{pos}|{dk}")].mean(0) for k in ("dld", "change", "dconf")}
            S["swap"][pos][f"{rk}->{dk}"] = {"dld": sub["dld"].tolist(), "change_pct": (100 * sub["change"]).tolist(), "dconf": sub["dconf"].tolist(),
                                             "peak_change": peak(L, 100 * sub["change"]),
                                             "peak_dconf": peak(L, sub["dconf"], "max" if dk == "high" else "min")}
    # continuous readout: shift of the high-minus-low logit as a share of the gap between high and low recipients
    for tag, DD in (("two_turn", D), ("single_prompt", _load(RESULTS_RAW, "swap"))):
        kd = np.array([r["kind"] for r in DD["recs"]])
        gap = float(hl(DD["clean"])[kd == "high"].mean() - hl(DD["clean"])[kd == "low"].mean())
        dh = 100 * (hl(DD["logits"]) - hl(DD["ref"])) / gap
        S["swap"][f"hl_{tag}"] = {"gap_logits": gap}
        for nm in DD["names"]:
            pos, dk = nm.split("|")
            for rk in ("high", "low"):
                m = dh[kd == rk][:, DD["names"].index(nm)].mean(0)
                S["swap"][f"hl_{tag}"][f"{pos}|{rk}->{dk}"] = {"curve_pct_of_gap": m.tolist(), "peak_pct_of_gap": peak(DD["layers"], m, "max" if dk == "high" else "min")}

# =====================================================================================================
# Attention blocking
# =====================================================================================================
ARROW = lambda c: c.replace(">", "→").replace("NL", "EOT")  # noqa: E731
if have("attn_tt"):
    D = load("attn_tt")
    names = D["names"]
    lg = D["logits"]
    base = lg[:, 0]
    y = base.argmax(-1)
    conds = names[1:]
    ch = np.stack([(lg[:, names.index(c)].argmax(-1) != y).astype(float) * 100 for c in conds], 1)
    dl = np.stack([ldiff(lg[:, names.index(c)], y) - ldiff(base, y) for c in conds], 1)
    dc = np.stack([MID[lg[:, names.index(c)].argmax(-1)] - MID[y] for c in conds], 1)
    DRa = _load(RESULTS_RAW, "attn_cat")
    rb = DRa["logits"][:, 0]
    ry = rb.argmax(-1)
    raw_ld = float(ldiff(rb, ry).mean())
    order = [c for c in ATT_LAB if c in conds][::-1]
    yy = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6), sharey=True)
    for ax, key, lab in ((axes[0], "ch", "Token change rate (%)"), (axes[1], "dl", "Change of the logit difference (% of its clean value)")):
        v = ch if key == "ch" else 100 * dl / ldiff(base, y).mean()
        m, e = ms(v[:, [conds.index(c) for c in order]])
        ax.barh(yy, m, height=0.55, color=P.BLUE, zorder=3, label="two-turn chat format")
        ax.errorbar(m, yy, xerr=e, fmt="none", ecolor=P.INK2, elinewidth=1, capsize=2, zorder=4)
        rv = [(i, (100 * (DRa["logits"][:, DRa["names"].index(c)].argmax(-1) != ry).mean() if key == "ch" else
                   100 * (ldiff(DRa["logits"][:, DRa["names"].index(c)], ry) - ldiff(rb, ry)).mean() / raw_ld)) for i, c in enumerate(order) if c in DRa["names"]]
        ax.plot([r[1] for r in rv], [r[0] for r in rv], ls="none", marker="D", ms=6, color=P.ORANGE, markerfacecolor=P.SURFACE, markeredgewidth=1.6, zorder=5,
                label="single-prompt format, same edges")
        P.hline(ax, 0, color=P.AXIS) if False else ax.axvline(0, color=P.AXIS, lw=1.0, zorder=1)
        ax.set_xlabel(lab)
        ax.grid(axis="x", color=P.GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([ATT_LAB[c][0].upper() + ATT_LAB[c][1:] for c in order], fontsize=8)
    axes[0].legend(loc="lower right", fontsize=8)
    suptitle(fig, f"Attention edges blocked in all layers (two-turn n={lg.shape[0]}, single prompt n={len(ry)}; mean ± SEM)")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt06_attention_blocking.png")
    S["attn_tt"] = {"n": lg.shape[0], "noise_floor_change_pct": float(100 * (y != D["clean"].argmax(-1)).mean()),
                    "clean_ld": float(ldiff(base, y).mean())}
    for i, c in enumerate(conds):
        S["attn_tt"][c] = {"change_pct": float(ch[:, i].mean()), "dld": float(dl[:, i].mean()), "dconf": float(dc[:, i].mean())}

if have("attn_ttw"):
    D = load("attn_ttw")
    C, names = D["centers"], D["names"]
    lg = D["logits"]  # [n, cond, window, 10]
    base = lg[:, names.index("none")]
    y = base.argmax(-1)
    x = np.arange(len(C))

    def am(cond):
        z = lg[:, names.index(cond)]
        return (z.argmax(-1) != y).astype(float) * 100, ldiff(z, y) - ldiff(base, y)

    groups = [(["NL>A", "POST5>A", "ALL>REQ0"], [P.BLUE, P.INK, P.MUTED], "A  When do the tokens after the answer read the answer?"),
              (["ALL>NL", "ALL>POST5", "LATER>A", "CC>POST5", "ALL>REQ0"], [P.BLUE, P.INK, P.VIOLET, P.ORANGE, P.MUTED],
               "B  When do later tokens read them (or the answer itself)?")]
    WLAB = {"NL>A": "end-of-turn token → answer", "POST5>A": "five tokens after the answer → answer", "ALL>NL": "all later tokens → end-of-turn token",
            "ALL>POST5": "request and final tokens → the five tokens", "LATER>A": "request and final tokens → answer",
            "CC>POST5": "last token → the five tokens", "ALL>REQ0": "later tokens → first request token (control)"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.6), sharex=True, sharey="row")
    for c, (conds, cols, title) in enumerate(groups):
        for cond, col in zip(conds, cols):
            chw, dlw = am(cond)
            P.line(axes[0, c], x, *ms(chw), color=col, label=WLAB[cond], ms=4)
            P.line(axes[1, c], x, *ms(dlw), color=col, label=WLAB[cond], ms=4)
        P.hline(axes[1, c], 0, color=P.AXIS)
        axes[0, c].set_title(title)
        axes[0, c].legend(loc="upper right", fontsize=7)
        axes[1, c].set_xticks(x)
        axes[1, c].set_xticklabels([str(v) for v in C], rotation=45)
        axes[1, c].set_xlabel("Centre of the 12-layer blocking window")
    axes[0, 0].set_ylabel("Token change rate (%)"), axes[1, 0].set_ylabel("Logit-difference change")
    axes[0, 0].set_ylim(bottom=0)
    suptitle(fig, f"Two-turn format: attention blocking in 12-layer windows (n={lg.shape[0]})")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt07_attention_windows.png")
    cl = D["clean"].argmax(-1)
    S["attn_ttw"] = {"n": lg.shape[0], "centers": list(C), "noise_floor_change_pct": float(100 * (y != cl[:, None]).mean()),
                     "clean_ld": float(ldiff(D["clean"], cl).mean())}
    for cond in names[1:]:
        chw, dlw = am(cond)
        S["attn_ttw"][cond] = {"change_pct": chw.mean(0).tolist(), "dld": dlw.mean(0).tolist(), "peak_change": peak(C, chw.mean(0)),
                               "peak_dld": peak(C, dlw.mean(0), "min")}

# =====================================================================================================
# Decoding and variance partitioning
# =====================================================================================================
if (RESULTS / "probe" / "probe.json").exists():
    R = json.load(open(RESULTS / "probe" / "probe.json"))
    L = sorted({r["layer"] for r in R["results"]})
    x = np.arange(len(L))
    tab = {(r["pos"], r["layer"]): r for r in R["results"]}
    ok = lambda p: all((p, l) in tab for l in L)  # noqa: E731
    ser = lambda pos, key, sub=None: np.array([(tab[(pos, l)][key] if sub is None else tab[(pos, l)][key][sub]) for l in L])  # noqa: E731
    uniq = lambda pos, k: np.clip(ser(pos, "r2_both", k) - R["r2_base"][k], 0, None)  # noqa: E731
    # variance beyond the log-prob summaries with a tuned penalty (scripts/probe_unique.py), both formats
    UQ = {}
    for tag, f in (("two_turn", RESULTS / "probe" / "unique_tuned.json"), ("single_prompt", RESULTS_RAW / "probe" / "unique_tuned.json")):
        if f.exists():
            U = json.load(open(f))
            UQ[tag] = {p: {r["layer"]: r for r in U["results"] if r["pos"] == p} for p in sorted({r["pos"] for r in U["results"]})}
    shown = [("QTT", 1.0, "-"), ("INT4", 1.1, (0, (3, 1.5))), ("AC", 1.2, "-"), ("A_last", 1.3, "-"), ("UNL", 1.5, "-"), ("PANL+1", 2.0, "-"), ("CC", 2.2, "-"), ("PANL", 2.2, "-")]
    shown = [t for t in shown if ok(t[0])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, key, lab in ((axes[0], "auroc", "AUROC for answer correctness"), (axes[1], "r2_tuned", "R² for the stated confidence\n(ridge, penalty tuned on the training folds)")):
        for p, lw, ls in shown:
            P.line(ax, x, ser(p, key), color=COL[p], label=LAB[p], marker=MK[p], ms=3.5, lw=lw, ls=ls)
        P.layer_axis(ax, L, every=2)
        ax.set_ylabel(lab)
    P.hline(axes[0], R["auroc_verbal_conf"], color=P.INK2, lw=1.0, ls=(0, (1, 1.5)))
    axes[0].annotate(f"the stated confidence itself ({R['auroc_verbal_conf']:.2f})", (len(L) - 1, R["auroc_verbal_conf"]), textcoords="offset points", xytext=(0, -10),
                     fontsize=7.5, color=P.INK2, ha="right")
    P.hline(axes[0], R["auroc_mean_logprob"], color=P.INK2, lw=1.0, ls=(0, (4, 2)))
    axes[0].annotate(f"mean answer log-prob ({R['auroc_mean_logprob']:.2f})", (len(L) - 1, R["auroc_mean_logprob"]), textcoords="offset points", xytext=(0, -10),
                     fontsize=7.5, color=P.INK2, ha="right")
    axes[0].set_title("A  Correctness")
    axes[1].set_title("B  Stated confidence")
    h, l_ = axes[1].get_legend_handles_labels()
    axes[1].legend(h[::-1], l_[::-1], loc="upper left", fontsize=7.5)
    ax = axes[2]
    P.hline(ax, 0, color=P.AXIS)
    UL = [("two_turn", "PANL", "two-turn: end-of-turn after the answer", P.BLUE, "o", "-", False), ("two_turn", "PANL+1", "two-turn: newline after it", P.AQUA, "^", "-", False),
          ("two_turn", "CC", "two-turn: last prompt token", P.ORANGE, "s", "-", False), ("single_prompt", "PANL", "single prompt: PANL", P.BLUE, "o", (0, (3, 1.5)), True),
          ("single_prompt", "CC", "single prompt: CC", P.ORANGE, "s", (0, (3, 1.5)), True)]
    for tag, p, lab, col, mk, ls, hollow in UL:
        if tag in UQ and p in UQ[tag]:
            P.line(ax, x, [UQ[tag][p][l]["r2_unique"] for l in L], color=col, label=lab, marker=mk, ms=3.5, lw=2.0 if not hollow else 1.4, ls=ls, hollow=hollow)
    P.layer_axis(ax, L, every=2)
    ax.set_ylabel("R² beyond six answer log-prob summaries\n(tuned ridge on the residual)")
    ax.set_title("C  Stated confidence, beyond log-probabilities")
    if UQ:
        ax.legend(loc="upper left", fontsize=7.5)
    suptitle(fig, f"Two-turn format: linear decoding from the residual stream (n={R['n']} trials, 5-fold CV)")
    fig.tight_layout()
    P.save(fig, FIGURES / "tt08_decoding.png")
    if UQ:
        S["probe_unique_tuned"] = {tag: {p: {"r2_unique": [d[l]["r2_unique"] for l in L], "r2_both": [d[l]["r2_both"] for l in L], "r2_base": d[L[0]]["r2_base"],
                                             "peak": peak(L, [d[l]["r2_unique"] for l in L]), "at_L25": d[25]["r2_unique"], "at_L40": d[40]["r2_unique"]}
                                         for p, d in v.items()} for tag, v in UQ.items()}
    S["probe"] = {"n": R["n"], "layers": L, "r2_base": R["r2_base"], "auroc_verbal_conf": R["auroc_verbal_conf"],
                  "auroc_mean_logprob": R["auroc_mean_logprob"], "r_logprob_conf": R["r_logprob_conf"], "n_results": len(R["results"])}
    for p in sorted({r["pos"] for r in R["results"]}):
        if not ok(p):
            continue
        a, r1, rt = ser(p, "auroc"), ser(p, "r2_a1"), ser(p, "r2_tuned")
        S["probe"][p] = {"auroc": a.tolist(), "r2_a1": r1.tolist(), "r2_tuned": rt.tolist(), "r2_unique_all6": uniq(p, "all").tolist(),
                         "peak_auroc": peak(L, a), "peak_r2_a1": peak(L, r1), "peak_r2_tuned": peak(L, rt), "peak_unique_all6": peak(L, uniq(p, "all")),
                         "first_layer_r2_a1_above_0.05": next((int(l) for l, v in zip(L, r1) if v > 0.05), None),
                         "first_layer_auroc_above_0.75": next((int(l) for l, v in zip(L, a) if v > 0.75), None)}

if (RESULTS / "ood_natural.json").exists():
    S["ood_natural"] = json.load(open(RESULTS / "ood_natural.json"))
json.dump(S, open(RESULTS / "summary.json", "w"), indent=1)
print("wrote", RESULTS / "summary.json")

# =====================================================================================================
# Side-by-side table: paper | single-prompt replication | two-turn format
# =====================================================================================================
MINUS = "−"
f2 = lambda v: "–" if v is None else f"{v:.2f}".replace("-", MINUS)  # noqa: E731
sg = lambda v: "–" if v is None else ("0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", MINUS))  # noqa: E731
pc = lambda v: "–" if v is None else f"{v:.1f}%".replace("-", MINUS)  # noqa: E731
p0 = lambda v: "–" if v is None else f"{v + 0.0:.0f}%".replace("-0%", "0%").replace("-", MINUS)  # noqa: E731
pk = lambda d, f=sg: f"{f(d['value'])} @ L{d['layer']}"  # noqa: E731
rows = [("Quantity", "Paper", "Single prompt (main replication)", "Two-turn chat format"), None]


def row(name, paper, raw, tt):
    rows.append((name, paper, raw, tt))


def head(name):
    rows.append((f"**{name}**", "", "", ""))


def share(cnt, names_):
    return 100 * sum(cnt[c] for c in names_) / sum(cnt.values())


head("Behaviour")
row("Valid trials", "7,858", f"{RB['n_valid']:,}", f"{B['n_valid']:,}")
row("Accuracy (GPT-4o-mini grading)", "77.4%", pc(100 * RB["accuracy"]), pc(100 * B["accuracy"]))
row("ECE / AUROC of stated confidence", "0.12 / 0.71", f"{f2(RB['ece'])} / {f2(RB['auroc_verbal_conf'])} (answer fed back); "
    f"{f2(RB['phase0']['ece'])} / {f2(RB['phase0']['auroc_verbal_conf'])} (stated with the answer)", f"{f2(B['ece'])} / {f2(B['auroc_verbal_conf'])}")
row('"Almost certain" / "Highly likely" / bottom-3 classes', "69% / 21% / 6.0%",
    f"{pc(share(RB['class_counts'], ['Almost certain']))} / {pc(share(RB['class_counts'], ['Highly likely']))} / {pc(share(RB['class_counts'], CLASSES[:3]))}",
    f"{pc(share(B['class_counts'], ['Almost certain']))} / {pc(share(B['class_counts'], ['Highly likely']))} / {pc(share(B['class_counts'], CLASSES[:3]))}")
row("Correlation of confidence with mean answer log-prob", "0.29", f2(RB["r_logprob_p1conf"]), f2(B["r_logprob_conf"]))
row("AUROC of mean answer log-prob for correctness", "0.75", f2(RB["auroc_mean_logprob"]), f2(B["auroc_mean_logprob"]))
if "steer" in S:
    rs, ts = RAW["steer"], S["steer"]
    head("Steering (scale 5; peak change, high / low vector)")
    row("Baseline confidence of the test trials", "0.55", f"{f2(rs['baseline_conf'])} (n = {rs['n']})", f"{f2(ts['baseline_conf'])} (n = {ts['n']})")
    row("Median gap between the top two class logits of these trials, unsteered", "–", f2(ts["hl_single_prompt"]["median_top2_margin"]), f2(ts["hl_two_turn"]["median_top2_margin"]))
    hp = lambda tag, nm: pk(ts[f"hl_{tag}"][nm]["peak_pct_of_gap"], p0) if nm in ts[f"hl_{tag}"] else "–"  # noqa: E731
    for pos, name, paper in (("PANL", "first token after the answer", "≈ +0.18 @ L25–26 / −0.22 @ L24"), ("CC", "last prompt token", "≈ +0.46 @ L30–33 / −0.42 @ L30")):
        row(f"Δ confidence, {name}", paper, f"{pk(rs[pos]['scale5']['high_peak'])} / {pk(rs[pos]['scale5']['low_peak'])}",
            f"{pk(ts[pos]['scale5']['high_peak'])} / {pk(ts[pos]['scale5']['low_peak'])}")
        row(f"Shift of the high-minus-low logit (% of the gap between high- and low-confidence trials), {name}", "–",
            f"{hp('single_prompt', pos + '|high|5')} / {hp('single_prompt', pos + '|low|5')}", f"{hp('two_turn', pos + '|high|5')} / {hp('two_turn', pos + '|low|5')}")
    for pos in [p for p in ["PANL+1", "SOT", "USER", "UNL", "POST5", "REQ0", "CC-1", "A_last", "AC"] if p in ts]:
        r_ = f"{pk(rs[pos]['scale5']['high_peak'])} / {pk(rs[pos]['scale5']['low_peak'])}" if pos in rs else "–"
        r2_ = f"{hp('single_prompt', pos + '|high|5')} / {hp('single_prompt', pos + '|low|5')}" if pos in rs else "–"
        nm = "second token after the answer (single prompt: first instruction token; two-turn: newline after the end-of-turn token)" if pos == "PANL+1" else LAB[pos]
        row(f"Δ confidence; logit shift — {nm}", "below 0.01" if pos == "PANL+1" else "–",
            f"{r_}; {r2_}" if pos in rs else "–",
            f"{pk(ts[pos]['scale5']['high_peak'])} / {pk(ts[pos]['scale5']['low_peak'])}; {hp('two_turn', pos + '|high|5')} / {hp('two_turn', pos + '|low|5')}")
if "patch" in S:
    rp, tp = RAW["patch"], S["patch"]
    head("Patching (clean activations restored after answer-token corruption)")
    row("Clean → corrupted logit difference", "11.9 → −0.5", f"{rp['clean']['ld']:.1f} → {rp['corrupt']['ld']:.1f}".replace("-", MINUS), f"{tp['clean']['ld']:.1f} → {tp['corrupt']['ld']:.1f}".replace("-", MINUS))
    row("Clean → corrupted confidence; tokens changed by corruption", "–", f"{f2(rp['clean']['conf'])} → {f2(rp['corrupt']['conf'])}; {p0(100 * rp['corrupt']['change'])}",
        f"{f2(tp['clean']['conf'])} → {f2(tp['corrupt']['conf'])}; {p0(100 * tp['corrupt']['change'])}")
    for pos, name, paper in (("PANL", "First token after the answer", "0.38 @ L25; ≈ 20% @ L25"), ("PANL+1", "Second token after the answer", "no recovery"),
                             ("CC", "Last prompt token", "≈ 22% of confidence @ L35, ≈ 95% @ L40")):
        cr = lambda d: (f"{f2(at(d['layers'], d[pos]['conf'], 25))} / {f2(at(d['layers'], d[pos]['conf'], 30))} / {f2(at(d['layers'], d[pos]['conf'], 35))} / "  # noqa: E731
                        f"{f2(at(d['layers'], d[pos]['conf'], 40))}; {pk(d[pos]['peak_ld_recovery'], p0)}")
        row(f"{name}: confidence after patch at L25 / L30 / L35 / L40; peak logit-difference recovery", paper, cr(rp), cr(tp))
    for pos in [p for p in ["SOT", "USER", "UNL", "REQ0", "CC-1", "POST5", "REQ", "TAIL4"] if p in tp]:
        row(f"{LAB[pos][0].upper() + LAB[pos][1:]}: peak confidence after patch; peak logit-difference recovery", "–", "–",
            f"{pk(peak(tp['layers'], tp[pos]['conf']), f2)}; {pk(tp[pos]['peak_ld_recovery'], p0)}")
    if "patch_relay" in S:
        ri = S["patch_relay"]["single_prompt"]
        row("All instruction tokens between PANL and CC (single prompt) / all request tokens (two-turn): peak confidence recovery; layers with more than half recovered", "–",
            f"{pk(ri['INSTR']['peak_conf_recovery'], p0)}; L" + "–L".join(str(x) for x in (lambda ls: (ls[0], ls[-1]))([l for l, v in zip(ri['layers'], ri['INSTR']['recovery_conf_pct']) if v > 50] or [0])),
            f"{pk(S['patch_relay']['two_turn']['REQ']['peak_conf_recovery'], p0)}; L" + "–L".join(str(x) for x in (lambda ls: (ls[0], ls[-1]))([l for l, v in zip(tp['layers'], S['patch_relay']['two_turn']['REQ']['recovery_conf_pct']) if v > 50] or [0])))
if "noise" in S:
    rn, tn = RAW["noise"], S["noise"]
    head("Noising (mean ablation): peak token change; peak change of the logit difference")
    row("Trials; clean logit difference", "400", f"{rn['n']}; {rn['clean_ld']:.1f}", f"{tn['n']}; {tn['clean_ld']:.1f}")
    for pos, name, paper in (("PANL", "First token after the answer", "≈ 14% @ L25; ≈ −0.9"), ("PANL+1", "Second token after the answer", "3–4%"),
                             ("CC", "Last prompt token", "19% @ L35, 71% @ L40"), ("AC", "Token before the answer", "3–4%")):
        row(name, paper, f"{pk(rn[pos]['peak_change'], pc)}; {pk(rn[pos]['peak_dld'], f2)}", f"{pk(tn[pos]['peak_change'], pc)}; {pk(tn[pos]['peak_dld'], f2)}")
    for pos in [p for p in ["UNL", "POST5", "REQ0"] if p in tn]:
        row(LAB[pos][0].upper() + LAB[pos][1:], "–", "–", f"{pk(tn[pos]['peak_change'], pc)}; {pk(tn[pos]['peak_dld'], f2)}")
if "swap" in S:
    rw, tw = RAW["swap"], S["swap"]
    head("Swap (activations transplanted from a donor trial): peak Δ confidence; peak logit shift as % of the high–low gap")
    row("Recipients (high / low confidence)", "–", f"{rw['n_high']} / {rw['n_low']}", f"{tw['n_high']} / {tw['n_low']}")
    SW_PAPER = {("PANL", "low->high"): "≈ +0.21 @ L26", ("PANL", "high->low"): "≈ −0.065", ("PANL", "low->low"): "≈ ±0.02", ("PANL", "high->high"): "≈ ±0.02",
                ("CC", "low->high"): "≈ +0.68 (late layers)", ("CC", "high->low"): "≈ −0.65 (late layers)"}
    hw = lambda d, tag, key: pk(d[f"hl_{tag}"][key]["peak_pct_of_gap"], p0) if key in d[f"hl_{tag}"] else "–"  # noqa: E731
    for pos, name in (("PANL", "first token after the answer"), ("POST5", "all five tokens between answer and request"), ("CC", "last prompt token")):
        if pos not in tw:
            continue
        for cond in ("low->high", "high->low", "low->low", "high->high"):
            rk, dk = cond.split("->")
            r_ = f"{pk(rw[pos][cond]['peak_dconf'])}; {hw(tw, 'single_prompt', f'{pos}|{rk}->{dk}')}" if pos in rw else "–"
            row(f"{name[0].upper() + name[1:]}, {rk}-confidence recipient ← {dk}-confidence donor", SW_PAPER.get((pos, cond), "–"), r_,
                f"{pk(tw[pos][cond]['peak_dconf'])}; {hw(tw, 'two_turn', f'{pos}|{rk}->{dk}')}")
if "attn_tt" in S:
    ra, ta = RAW["attn_cat"], S["attn_tt"]
    PAPER_ATT = {"CC>NL+1": "≈ 10%", "CC>NL": "≈ 9–12%", "CC>A": "≈ 9–12%", "CC>Q": "≈ 9–12%", "CC>Q+A": "≈ 9–12%", "ALL>NL": "≈ 18%",
                 "ALL>last_A": "≈ 51%", "ALL>last_A_keepNL": "≈ 24%", "ALL>A": "≈ 68%", "ALL>A_keepNL": "≈ 51%"}
    raw_ld = float(np.mean(ldiff(_load(RESULTS_RAW, "attn_cat")["logits"][:, 0], _load(RESULTS_RAW, "attn_cat")["logits"][:, 0].argmax(-1))))
    head("Attention edges blocked in all layers: token change; change of the logit difference as % of its clean value")
    row("Trials; clean logit difference", "–", f"{ra['n']}; {raw_ld:.1f}", f"{ta['n']}; {ta['clean_ld']:.1f}")
    for c in [k for k in ATT_LAB if k in ta]:
        r_ = f"{pc(ra[c]['change_pct'])}; {p0(100 * ra[c]['dld'] / raw_ld)}" if c in ra else "–"
        row(ATT_LAB[c][0].upper() + ATT_LAB[c][1:], PAPER_ATT.get(c, "–"), r_, f"{pc(ta[c]['change_pct'])}; {p0(100 * ta[c]['dld'] / ta['clean_ld'])}")
if "attn_ttw" in S:
    tw_ = S["attn_ttw"]
    head("Attention edges blocked in 12-layer windows (two-turn: categorical prompt; paper and single prompt: minimal 0–9 prompt)")
    rm = RAW.get("attn_min_pmq", {})
    row("First token after the answer → answer tokens: window with the largest drop of the logit difference", "≈ L22–28",
        pk(rm["PANL>A"]["peak_dld"], f2) if "PANL>A" in rm else "–", pk(tw_["NL>A"]["peak_dld"], f2))
    row("Five tokens after the answer → answer tokens", "–", "–", pk(tw_["POST5>A"]["peak_dld"], f2))
    row("Last token → first token after the answer (single prompt) / request and final tokens → the five tokens (two-turn)", "≈ L30–36",
        pk(rm["CC>PANL"]["peak_dld"], f2) if "CC>PANL" in rm else "–", pk(tw_["ALL>POST5"]["peak_dld"], f2))
    row("Request and final tokens → answer tokens", "–", "–", pk(tw_["LATER>A"]["peak_dld"], f2))
    row("Control (one uninformative token hidden): largest change", "–", pk(rm["CC>PANL+1"]["peak_dld"], f2) if "CC>PANL+1" in rm else "–", pk(tw_["ALL>REQ0"]["peak_dld"], f2))
if "probe" in S:
    rr, tr = RAW["probe"], S["probe"]
    head("Decoding (3,000 trials, 5-fold CV)")
    row("AUROC for correctness: stated confidence / mean answer log-prob", "0.71 / 0.75", f"{f2(rr['auroc_verbal_conf'])} / {f2(rr['auroc_mean_logprob'])}",
        f"{f2(tr['auroc_verbal_conf'])} / {f2(tr['auroc_mean_logprob'])}")
    row("R² of stated confidence from six answer log-prob summaries", "0.100", f"{rr['r2_base']['all']:.3f}", f"{tr['r2_base']['all']:.3f}")
    for pos, name, paper in (("PANL", "First token after the answer", "L25; 0.48"), ("CC", "Last prompt token", "L28; 0.84")):
        row(f"{name}: first layer with R² > 0.05 at ridge α = 1; peak R² at α = 1", paper,
            f"L{rr[pos]['first_layer_r2_a1_above_0.05']}; {pk(rr[pos]['peak_r2_a1'], f2)}", f"L{tr[pos]['first_layer_r2_a1_above_0.05']}; {pk(tr[pos]['peak_r2_a1'], f2)}")
    for pos in [p for p in ["PANL", "PANL+1", "SOT", "USER", "UNL", "REQ0", "INT4", "EOT2", "CC-1", "CC", "A_last", "AC", "QTT"] if p in tr]:
        nm = {"PANL": "first token after the answer", "PANL+1": "second token after the answer", "CC": "last prompt token", "AC": "token before the answer"}.get(pos, LAB[pos])
        r_ = f"{f2(at(rr['layers'], rr[pos]['r2_tuned'], 25))} / {pk(rr[pos]['peak_r2_tuned'], f2)}; {f2(at(rr['layers'], rr[pos]['auroc'], 25))} / {pk(rr[pos]['peak_auroc'], f2)}" if pos in rr else "–"
        row(f"{nm[0].upper() + nm[1:]}: tuned-ridge R² of confidence at L25 / peak; correctness AUROC at L25 / peak", "–", r_,
            f"{f2(at(tr['layers'], tr[pos]['r2_tuned'], 25))} / {pk(tr[pos]['peak_r2_tuned'], f2)}; {f2(at(tr['layers'], tr[pos]['auroc'], 25))} / {pk(tr[pos]['peak_auroc'], f2)}")
    row("R² unique to the first token after the answer beyond the six log-prob summaries (ridge α = 1), L40 / peak", "0.38 @ L40",
        f"{f2(at(rr['layers'], rr['PANL']['r2_unique_all6'], 40))} / {pk(rr['PANL']['peak_unique_all6'], f2)}",
        f"{f2(at(tr['layers'], tr['PANL']['r2_unique_all6'], 40))} / {pk(tr['PANL']['peak_unique_all6'], f2)}")

with open(RESULTS / "comparison.md", "w") as f:
    f.write("# Two-turn chat format vs the single-prompt replication and the paper\n\n")
    f.write("Positions are named by their place in the prompt. The first token after the answer is the post-answer newline (PANL) in the paper's "
            "single prompt and the `<end_of_turn>` token of the model's answer turn in the two-turn format; the last prompt token is the confidence "
            "colon (CC) and the newline that opens the model's reply. Layer numbers after @ are where the peak sits. Generated by "
            "`scripts/tt_analyze.py`; raw numbers in `results/tt/summary.json`.\n\n")
    for r in rows:
        f.write("|---|---|---|---|\n" if r is None else "| " + " | ".join(r) + " |\n")
print("wrote", RESULTS / "comparison.md", f"({len(rows) - 2} rows)")
