"""Patching with the question corrupted as well: figure and summary numbers.

The paper corrupts the answer tokens and restores clean activations at one position. Here the question tokens
are corrupted too (or alone), so that no later token can read the question or the answer, and clean
activations are restored at PANL, CC and the comparison positions (scripts/patch_qa.sh).

    python scripts/patch_qa_analyze.py

Reads results/patch_p0_cabs_instr (answer only), patch_p0_cabs_q, patch_p0_cabs_qa and, for the two-turn
format, results/tt/patch and patch_qa; and the runs in which a position is held at its clean activations over
a range of layers (results[/tt]/patchwin_*), and the attention-blocking run that cuts tokens off from the
question (results/attn_q), and the layer-range runs in which the tokens after PANL are also cut off from
question and answer (results/patchwin_*_blk). Writes figures/qa01_patching_question_answer.png,
figures/qa02_patching_layer_ranges.png, figures/qa03_attention_question.png, figures/qa04_panl_only_reader.png
and results/patch_qa_summary.json. Metrics as in scripts/analyze.py: confidence = midpoint of the argmax class;
logit-difference recovery = share of the gap between corrupted and clean run that the patch closes (logit of
the clean class minus the mean of the other nine).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.env import FIGURES, RESULTS_RAW
from vc.metrics import MID, have, ldiff, load, mets
from vc.plotting import AQUA, BLUE, GREEN, INK2, MAGENTA, MUTED, ORANGE, RED, VIOLET, YELLOW, hline, layer_axis, line, plt, save, setup

setup()
RUNS = {"raw": {"a": "patch_p0_cabs_instr", "q": "patch_p0_cabs_q", "qa": "patch_p0_cabs_qa"},
        "tt": {"a": "patch", "q": "patch_q", "qa": "patch_qa"}}
RES = {"raw": RESULTS_RAW, "tt": RESULTS_RAW / "tt"}
SPAN = {"a": "answer tokens corrupted (the paper)", "q": "question tokens corrupted", "qa": "question and answer tokens corrupted"}
FMT = {"raw": "Single prompt", "tt": "Two-turn chat"}
SHOW = {"raw": ["PANL", "INSTR", "CC", "PANL+1", "A_last", "AC"], "tt": ["PANL", "POST5", "REQ", "CC", "PANL+1", "A_last", "AC"]}
COL = {"PANL": BLUE, "CC": ORANGE, "PANL+1": AQUA, "INSTR": YELLOW, "REQ": YELLOW, "POST5": GREEN, "A_last": VIOLET, "AC": MAGENTA}
LAB = {"raw": {"PANL": "PANL", "CC": "CC", "PANL+1": "PANL+1 (control)", "INSTR": "all instruction tokens between PANL and CC",
               "A_last": "last answer token", "AC": "answer colon"},
       "tt": {"PANL": "end-of-turn token after the answer", "CC": "last token", "PANL+1": "newline after the end-of-turn token",
              "POST5": "the five tokens between answer and request", "REQ": "all request tokens", "A_last": "last answer token",
              "AC": "token before the answer"}}
S = {}
fmts = [f for f in RUNS if have(RES[f], RUNS[f]["qa"])]
fig, axes = plt.subplots(len(fmts), 3, figsize=(16, 4.4 * len(fmts)), squeeze=False, sharey="row")
for r, f in enumerate(fmts):
    S[f] = {}
    for c, span in enumerate(("a", "q", "qa")):
        if not have(RES[f], RUNS[f][span]):
            axes[r, c].set_axis_off()  # the question-only corruption was run for the single prompt only
            continue
        D = load(RES[f], RUNS[f][span])
        L, names = list(D["layers"]), D["names"]
        M = mets(D["logits"], D["clean"])
        y = D["clean"].argmax(-1)
        clean_ld, corr_ld = ldiff(D["clean"], y), ldiff(D["corrupt"], y)
        clean_conf, corr_conf = MID[y], MID[D["corrupt"].argmax(-1)]
        d = S[f][span] = {"run": RUNS[f][span], "n": int(len(y)), "layers": L,
                          "clean": {"conf": float(clean_conf.mean()), "ld": float(clean_ld.mean())},
                          "corrupt": {"conf": float(corr_conf.mean()), "ld": float(corr_ld.mean()),
                                      "class_changed": float((D["corrupt"].argmax(-1) != y).mean()),
                                      "in_bottom3": float((D["corrupt"].argmax(-1) <= 2).mean())}}
        ax = axes[r, c]
        x = layer_axis(ax, L)
        hline(ax, clean_conf.mean(), color=GREEN, label="clean run")
        hline(ax, corr_conf.mean(), color=RED, label="corrupted run, nothing restored")
        for p in names:
            v = names.index(p)
            cm, lm = M["conf"][:, v].mean(0), M["ld"][:, v].mean(0)
            se = M["conf"][:, v].std(0, ddof=1) / np.sqrt(len(y))
            rec = 100 * (lm - corr_ld.mean()) / (clean_ld.mean() - corr_ld.mean())
            d[p] = {"conf": [round(float(a), 4) for a in cm], "ld_recovery_pct": [round(float(a), 2) for a in rec],
                    "peak_conf": {"layer": int(L[int(np.argmax(cm))]), "value": float(cm.max())},
                    "peak_ld_recovery_pct": {"layer": int(L[int(np.argmax(rec))]), "value": float(rec.max())}}
            if p in SHOW[f] and not (p == "AC" and span == "a"):  # AC precedes the answer: a no-op when only the answer is corrupted
                line(ax, x, cm, se, color=COL[p], label=LAB[f][p], ms=3.5, lw=1.7)
        noop = "BOS" if "BOS" in names else "AC"
        d["noop_check_max_abs_logit_difference"] = float(np.abs(D["logits"][:, names.index(noop)] - D["ref"][:, names.index(noop)]).max())
        ax.set_title(f"{FMT[f]}: {SPAN[span]}", fontsize=9)
        ax.set_ylim(0, 1)
        if c == 0:
            ax.set_ylabel("Stated confidence after restoring\nclean activations at one position and layer")
    axes[r, 2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5)
fig.suptitle("Activation patching when the question is corrupted too: which position brings the confidence back?", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, FIGURES / "qa01_patching_question_answer.png")

# ---- a position held at its clean activations over a range of layers
WRUNS = {"raw": {"a": "patchwin_p0_cabs_a", "q": "patchwin_p0_cabs_q", "qa": "patchwin_p0_cabs_qa"},
         "tt": {"a": "patchwin_a", "q": "patchwin_q", "qa": "patchwin_qa"}}
WSHOW = {"raw": ["PANL", "PANL+1", "A_last", "AC", "INSTR"], "tt": ["PANL", "PANL+1", "POST5", "A_last", "AC", "REQ"]}
wf = [f for f in WRUNS if have(RES[f], WRUNS[f]["qa"])]
if wf:
    fig, axes = plt.subplots(len(wf), 3, figsize=(16, 4.2 * len(wf)), squeeze=False, sharey="row")
    for r, f in enumerate(wf):
        S.setdefault(f, {})["ranges"] = {}
        for c, span in enumerate(("a", "q", "qa")):
            ax = axes[r, c]
            if not have(RES[f], WRUNS[f][span]):
                ax.set_axis_off()
                continue
            D = load(RES[f], WRUNS[f][span])
            y = D["clean"].argmax(-1)
            clean_ld, corr_ld = ldiff(D["clean"], y), ldiff(D["corrupt"], y)
            clean_conf, corr_conf = MID[y].mean(), MID[D["corrupt"].argmax(-1)].mean()
            wins = sorted({(w0, w1) for _, w0, w1 in D["win_names"]}, key=lambda w: (w != (0, 61), w))
            d = S[f]["ranges"][span] = {"run": WRUNS[f][span], "n": int(len(y)), "clean_conf": float(clean_conf), "corrupt_conf": float(corr_conf)}
            for v, (p, w0, w1) in enumerate(D["win_names"]):
                z = D["logits"][:, v]
                conf = MID[z.argmax(-1)]
                rec = 100 * (ldiff(z, y).mean() - corr_ld.mean()) / (clean_ld.mean() - corr_ld.mean())
                d[f"{p}|{w0}-{w1}"] = {"conf": round(float(conf.mean()), 4), "sem": round(float(conf.std(ddof=1) / np.sqrt(len(y))), 4),
                                         "ld_recovery_pct": round(float(rec), 2)}
            pos = [p for p in WSHOW[f] if f"{p}|0-61" in d and not (p == "AC" and span == "a")]
            w = 0.8 / len(pos)
            for k, p in enumerate(pos):
                ax.bar(np.arange(len(wins)) + (k - (len(pos) - 1) / 2) * w, [d[f"{p}|{a}-{b}"]["conf"] for a, b in wins], w * 0.92,
                       yerr=[d[f"{p}|{a}-{b}"]["sem"] for a, b in wins], color=COL[p], label=LAB[f][p], error_kw={"lw": 0.8})
            hline(ax, clean_conf, color=GREEN, label="clean run")
            hline(ax, corr_conf, color=RED, label="corrupted run, nothing restored")
            ax.set_xticks(range(len(wins)))
            ax.set_xticklabels(["all layers" if (a, b) == (0, 61) else f"layers {a}–{b}" for a, b in wins], fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f"{FMT[f]}: {SPAN[span]}", fontsize=9)
            if c == 0:
                ax.set_ylabel("Stated confidence when one position is held\nat its clean activations in these layers")
        axes[r, 2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5) if axes[r, 2].has_data() else axes[r, 1].legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5)
    fig.suptitle("The same corruption, with one position held clean over a range of layers", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "qa02_patching_layer_ranges.png")
# ---- PANL held clean while the tokens after PANL cannot read the question, the answer label or the answer
BRUNS = {span: f"patchwin_p0_cabs_{span}_blk" for span in ("a", "q", "qa")}
if all(have(RESULTS_RAW, e) for e in BRUNS.values()) and "ranges" in S.get("raw", {}):
    S["raw"]["ranges_blocked"] = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True)
    cats = [("NONE", "nothing\nrestored"), ("PANL", "PANL\nheld clean"), ("PANL+1", "PANL+1 held clean\n(control)"),
            ("A_last", "last answer token\nheld clean")]
    for ax, span in zip(axes, ("a", "q", "qa")):
        D = load(RESULTS_RAW, BRUNS[span])
        y = D["clean"].argmax(-1)
        clean_ld, corr_ld = ldiff(D["clean"], y), ldiff(D["corrupt"], y)
        d = S["raw"]["ranges_blocked"][span] = {"run": BRUNS[span], "n": int(len(y)), "clean_conf": float(MID[y].mean()),
                                                "corrupt_conf_no_block": float(MID[D["corrupt"].argmax(-1)].mean())}
        for v, (p, w0, w1) in enumerate(D["win_names"]):
            z = D["logits"][:, v]
            conf = MID[z.argmax(-1)]
            d[f"{p}|{w0}-{w1}" if w1 >= w0 else "NONE"] = {
                "conf": round(float(conf.mean()), 4), "sem": round(float(conf.std(ddof=1) / np.sqrt(len(y))), 4),
                "ld_recovery_pct": round(float(100 * (ldiff(z, y).mean() - corr_ld.mean()) / (clean_ld.mean() - corr_ld.mean())), 2),
                "in_bottom3_pct": round(float(100 * (z.argmax(-1) <= 2).mean()), 1)}
        free = S["raw"]["ranges"][span]
        open_ = [free["corrupt_conf"]] + [free[f"{p}|20-28"]["conf"] for p, _ in cats[1:]]
        shut = [d["NONE"]["conf"]] + [d[f"{p}|20-28"]["conf"] for p, _ in cats[1:]]
        x = np.arange(len(cats))
        ax.bar(x - 0.19, open_, 0.36, color=MUTED, label="tokens after PANL can read question and answer")
        ax.bar(x + 0.19, shut, 0.36, color=BLUE, label="tokens after PANL cannot: PANL is the only reader")
        for xi, (a_, b_) in enumerate(zip(open_, shut)):
            ax.annotate(f"{a_:.2f}", (xi - 0.19, a_), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=INK2)
            ax.annotate(f"{b_:.2f}", (xi + 0.19, b_), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=INK2)
        hline(ax, d["clean_conf"], color=GREEN, label="clean run")
        ax.set_xticks(x)
        ax.set_xticklabels([c_[1] for c_ in cats], fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"Single prompt: {SPAN[span]}", fontsize=9)
    axes[0].set_ylabel("Stated confidence")
    fig.suptitle("Is a clean PANL enough? Only when the later tokens cannot read the corrupted tokens themselves "
                 "(held clean = kept at its clean activations in layers 20–28)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    save(fig, FIGURES / "qa04_panl_only_reader.png")

# ---- attention blocking in all layers: who needs to read the question? (single prompt, no corruption)
if have(RESULTS_RAW, "attn_q"):
    D = load(RESULTS_RAW, "attn_q")
    names, lg = D["names"], D["logits"]
    base = lg[:, 0]
    y = base.argmax(-1)
    hi, lo = y >= 7, y <= 2
    LABEL = {"NL>Q": "PANL cannot read the question", "NL>A": "PANL cannot read the answer",
             "NL>PRE": "PANL cannot read question, answer label or answer", "LATER>Q": "tokens after PANL cannot read the question",
             "LATER>A": "tokens after PANL cannot read the answer", "LATER>Q+A": "tokens after PANL cannot read question or answer",
             "LATER>PRE": "... nor the answer label: PANL is the only reader", "ALL>Q": "PANL and later tokens cannot read the question",
             "ALL>PRE": "PANL and later tokens cannot read question, label or answer"}
    conds = [c for c in LABEL if c in names]
    d = S["attention_question"] = {"n": int(len(y)), "n_high": int(hi.sum()), "n_low": int(lo.sum()), "clean_conf": float(MID[y].mean()),
                                   "clean_conf_high": float(MID[y][hi].mean()), "clean_conf_low": float(MID[y][lo].mean()),
                                   "noise_floor_change_pct": float(100 * (y != D["clean"].argmax(-1)).mean())}
    for c_ in conds:
        z = lg[:, names.index(c_)]
        conf = MID[z.argmax(-1)]
        d[c_] = {"change_pct": float(100 * (z.argmax(-1) != y).mean()), "conf": float(conf.mean()), "conf_high": float(conf[hi].mean()),
                 "sem_high": float(conf[hi].std(ddof=1) / np.sqrt(hi.sum())),
                 "conf_low": float(conf[lo].mean()), "sem_low": float(conf[lo].std(ddof=1) / np.sqrt(lo.sum())),
                 "low_still_bottom3_pct": float(100 * (z.argmax(-1)[lo] <= 2).mean()),
                 "ld_change_pct": float(100 * (ldiff(z, y).mean() - ldiff(base, y).mean()) / ldiff(base, y).mean()),
                 "in_bottom3_high_pct": float(100 * (z.argmax(-1)[hi] <= 2).mean())}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    yv = np.arange(len(conds))[::-1]
    for k, (key, col, lab) in enumerate((("high", BLUE, f"{int(hi.sum())} trials with a top-3 class"), ("low", RED, f"{int(lo.sum())} trials with a bottom-3 class"))):
        ax.barh(yv - (k - 0.5) * 0.36, [d[c_][f"conf_{key}"] for c_ in conds], 0.34, xerr=[d[c_][f"sem_{key}"] for c_ in conds], color=col,
                error_kw={"lw": 0.8}, label=lab)
        for yi, c_ in zip(yv, conds):
            ax.annotate(f"{d[c_][f'conf_{key}']:.2f}", (d[c_][f"conf_{key}"] + d[c_][f"sem_{key}"], yi - (k - 0.5) * 0.36), textcoords="offset points",
                        xytext=(4, -3), fontsize=7.5, color=INK2)
    ax.axvline(d["clean_conf_high"], color=BLUE, lw=1.0, ls="--", label="the same trials with nothing blocked")
    ax.axvline(d["clean_conf_low"], color=RED, lw=1.0, ls="--")
    ax.set_yticks(yv)
    ax.set_yticklabels([LABEL[c_] for c_ in conds], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Stated confidence")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, -0.3), ncol=3, fontsize=8)
    ax.set_title("Attention blocked in all layers, nothing corrupted (single prompt)", fontsize=10)
    fig.tight_layout()
    save(fig, FIGURES / "qa03_attention_question.png")

json.dump(S, open(RESULTS_RAW / "patch_qa_summary.json", "w"), indent=1)
for f in ("raw", "tt"):
    for span, d in S.get(f, {}).items():
        if span not in ("a", "q", "qa"):
            continue
        print(f"{f} {span}: n {d['n']} clean {d['clean']['conf']:.2f} corrupted {d['corrupt']['conf']:.2f} (ld {d['clean']['ld']:.1f} -> {d['corrupt']['ld']:.1f}); "
              + "; ".join(f"{p} {d[p]['peak_conf']['value']:.2f}@L{d[p]['peak_conf']['layer']}" for p in d if isinstance(d[p], dict) and "peak_conf" in d[p]))
print("PATCH_QA ANALYZE DONE")
