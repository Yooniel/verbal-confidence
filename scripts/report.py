"""Build results/comparison.md: replication numbers side by side with the values reported in the paper.

Paper values are read off the text, tables and figures of arXiv:2603.17839v4 (figure read-offs are approximate).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.env import RESULTS

S = json.load(open(RESULTS / "summary.json"))
rows = []  # (section, quantity, paper, ours)


def add(sec, q, paper, ours):
    """`ours` may be a zero-argument callable so that rows whose inputs are not available yet are skipped."""
    if callable(ours):
        try:
            ours = ours()
        except KeyError:
            return
    rows.append((sec, q, paper, ours))


def at(d, layers, l):
    return d[layers.index(l)]


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pk(p, nd=2, unit=""):
    return f"{p['value']:.{nd}f}{unit} @ L{p['layer']}"


if "behaviour" in S:
    B = S["behaviour"]
    add("Behaviour", "valid trials", "7,858", f"{B['n_valid']:,}")
    add("Behaviour", "accuracy", "77.4% (GPT-4o-mini grader)", f"{100*B['accuracy']:.1f}% (alias match or Gemma grader); {100*B['accuracy_em_only']:.1f}% alias match only")
    add("Behaviour", "ECE (categorical prompt)", "0.12", f(B["ece"]))
    add("Behaviour", "AUROC of verbal confidence", "0.71", f(B["auroc_verbal_conf"]))
    cc = B["class_counts"]
    n = B["n_valid"]
    add("Behaviour", "share 'Almost certain' / 'Highly likely'", "69% / 21%", f"{100*cc['Almost certain']/n:.0f}% / {100*cc['Highly likely']/n:.0f}%")
    add("Behaviour", "share of bottom-3 classes", "6.0%", f"{100*B['n_low']/n:.1f}%")
    # Phase-0 confidence (stated in the same generation as the answer): the paper's Fig. 9 numbers look like this one
    from sklearn.metrics import roc_auc_score

    from vc.data import read_jsonl
    from vc.env import DATA
    from vc.prompts import CLASSES, MIDPOINTS

    T = [t for t in read_jsonl(DATA / "trials.jsonl") if t["p0_conf"] is not None]
    c0 = np.round(np.array([MIDPOINTS[t["p0_conf"]] for t in T]), 2)
    y0 = np.array([t["correct"] for t in T])
    k0 = np.array([t["p0_conf"] for t in T])
    ece0 = float(sum((c0 == v).mean() * abs(y0[c0 == v].mean() - v) for v in np.unique(c0)))
    add("Behaviour", "Phase-0 confidence (same generation as the answer): ECE / AUROC / share 'Almost certain' / 'Highly likely'",
        "(Fig. 9 values: 0.12 / 0.71 / 69% / 21%)", f"{ece0:.2f} / {roc_auc_score(y0, c0):.2f} / {100*(k0==9).mean():.0f}% / {100*(k0==8).mean():.0f}%")
    add("Behaviour", "AUROC of mean answer log-prob", "0.75", f(B["auroc_mean_logprob"]))
    add("Behaviour", "r(log-prob, phase-1 confidence)", "0.29", f(B["r_logprob_p1conf"]))
    add("Behaviour", "r(log-prob, phase-0 confidence)", "0.23", f(B["r_logprob_p0conf"]))
    add("Behaviour", "r(phase-0 conf, phase-1 conf)", "0.63", f(B["r_p0conf_p1conf"]))
    mp = B["min_prompt"][B["min_key"]]
    add("Behaviour", "minimal 0-9 prompt: ECE / AUROC", "0.17 / 0.68", f"{mp['ece']:.2f} / {mp['auroc']:.2f} (first token is a digit in {100*mp['first_token_is_digit']:.0f}% of trials)")

if "steer" in S:
    T = S["steer"]
    add("Steering", "baseline confidence of test trials", "0.55", f(T["baseline_conf"]))
    for pos, ph, pl in (("PANL", "≈ +0.17 @ L24–26 (peak range L21–25)", "≈ −0.22 @ L24"), ("CC", "≈ +0.46 @ L30–34 (peak range L30–35)", "≈ −0.42 @ L30")):
        add("Steering", f"{pos}, scale 5: peak Δconf, high vector", ph, pk(T[pos]["scale5"]["high_peak"], 3))
        add("Steering", f"{pos}, scale 5: peak Δconf, low vector", pl, pk(T[pos]["scale5"]["low_peak"], 3))
    add("Steering", "PANL, scale 2: peak Δconf high / low", "≈ +0.16 / −0.09", f"{pk(T['PANL']['scale2']['high_peak'],3)} / {pk(T['PANL']['scale2']['low_peak'],3)}")
    add("Steering", "CC, scale 2: peak Δconf high / low", "≈ +0.42 / −0.11", f"{pk(T['CC']['scale2']['high_peak'],3)} / {pk(T['CC']['scale2']['low_peak'],3)}")
    for pos, ref in (("PANL+1", "below 0.01 in absolute value"), ("FCC", "below 0.01 in absolute value"), ("AC", "about 0.03 or less, not significant")):
        s5 = T[pos]["scale5"]
        big = max(max(abs(v) for v in s5["high_curve"]), max(abs(v) for v in s5["low_curve"]))
        add("Steering", f"{pos}, scale 5: largest absolute Δconf over all layers and both vectors", ref, f"{big:.3f}")
    add("Steering", "last answer token, scale 5: peak Δconf high / low", "effective (≈ +0.33 / −0.35)",
        f"{pk(T['A_last']['scale5']['high_peak'],3)} / {pk(T['A_last']['scale5']['low_peak'],3)}")
    add("Steering", f"first answer token, scale 5, multi-token answers only (n={T['n_multi_token_answers']}): peak Δconf high / low", "ineffective",
        f"{pk(T['A_first']['scale5_multi_token_high_peak'],3)} / {pk(T['A_first']['scale5_multi_token_low_peak'],3)} "
        f"(last answer token on the same trials: {pk(T['A_last']['scale5_multi_token_high_peak'],3)} / {pk(T['A_last']['scale5_multi_token_low_peak'],3)})")

if "patch" in S:
    Pp = S["patch"]
    L = Pp["layers"]
    add("Patching", "clean logit diff / corrupted logit diff", "≈ 11.9 / ≈ −0.5", f"{Pp['clean']['ld']:.2f} / {Pp['corrupt']['ld']:.2f}")
    add("Patching", "corrupted run: confidence / token change", "≈ 0.05 / 100%", f"{Pp['corrupt']['conf']:.2f} / {100*Pp['corrupt']['change']:.0f}%")
    add("Patching", "PANL: peak confidence after patch", "≈ 0.38 @ L25", pk({"layer": L[int(np.argmax(Pp['PANL']['conf']))], "value": max(Pp['PANL']['conf'])}))
    add("Patching", "PANL: peak confidence recovery", "≈ 40% @ L25", pk(Pp["PANL"]["peak_conf_recovery"], 1, "%"))
    add("Patching", "PANL: peak logit-diff recovery", "≈ 20% @ L25", pk(Pp["PANL"]["peak_ld_recovery"], 1, "%"))
    add("Patching", "PANL: recovery of the margin (clean class − corrupted-run class), supplementary", "not reported", pk(Pp["PANL"]["peak_gap_recovery"], 1, "%") + f" (margin {Pp['PANL']['gap_corrupt'][0]:.1f} → {max(Pp['PANL']['gap_patched']):.1f}; clean {Pp['PANL']['gap_clean']:.1f})")
    add("Patching", "PANL: token change rate at best layer", "≈ 78% @ L25", pk({"layer": L[int(np.argmin(Pp['PANL']['change']))], "value": 100*min(Pp['PANL']['change'])}, 0, "%"))
    add("Patching", "PANL+1: peak confidence recovery", "≈ 0% (−1.4% @ L25)", pk(Pp["PANL+1"]["peak_conf_recovery"], 1, "%"))
    add("Patching", "CC: confidence recovery @ L35 / L40 / L61", "≈ 22% / ≈ 95% / 100% (rises after L30)", " / ".join(f"{at(Pp['CC']['recovery_conf_pct'], L, l):.0f}%" for l in (35, 40, 61)))
    add("Patching", "CC: confidence recovery @ L25 / L30", "≈ 0% / ≈ 0%", " / ".join(f"{at(Pp['CC']['recovery_conf_pct'], L, l):.0f}%" for l in (25, 30)))

if "noise" in S:
    N = S["noise"]
    L = N["layers"]
    add("Noising", "clean logit diff", "≈ 9.4", f(N["clean_ld"]))
    add("Noising", "PANL: peak token change rate", "≈ 14% @ L25", pk(N["PANL"]["peak_change"], 1, "%"))
    add("Noising", "PANL: largest logit-diff drop", "≈ −0.9 @ L25–26", pk(N["PANL"]["peak_dld"]))
    add("Noising", "PANL+1: peak token change rate", "≈ 3–4% (flat)", pk(N["PANL+1"]["peak_change"], 1, "%"))
    add("Noising", "AC: peak token change rate", "≈ PANL+1 control", pk(N["AC"]["peak_change"], 1, "%"))
    add("Noising", "CC: token change @ L30 / L35 / L40 / L61", "≈ 5% / 19% / 71% / 77%", " / ".join(f"{at(N['CC']['change_pct'], L, l):.0f}%" for l in (30, 35, 40, 61)))
    add("Noising", "numerical noise floor (unedited rows vs full clean pass), max over layers", "not reported", f"{max(N['noise_floor_change_pct_by_layer']):.1f}% (removed by the same-pass reference)")

if "swap" in S:
    W = S["swap"]
    ref = {"low->high": ("≈ 37% @ L26", "≈ +0.21 @ L26", "≈ −1.15 @ L26–27"), "high->low": ("≈ 28% @ L26–27", "≈ −0.065 @ L26", "≈ −1.85 @ L26"),
           "high->high": ("≈ 16% @ L25–26", "≈ +0.02", "≈ −0.2"), "low->low": ("≈ 13% @ L25", "≈ −0.01", "≈ −0.2")}
    for c, (rc, rd, rl) in ref.items():
        add("Swap (PANL)", f"{c} (recipient→donor): peak token change", rc, pk(W["PANL"][c]["peak_change"], 1, "%"))
    for c, (rc, rd, rl) in ref.items():
        add("Swap (PANL)", f"{c}: peak confidence change", rd, pk(W["PANL"][c]["peak_dconf"], 3))
    for c, (rc, rd, rl) in ref.items():
        add("Swap (PANL)", f"{c}: largest logit-diff change", rl, pk(W["PANL"][c]["peak_dld"]))
    add("Swap (PANL+1)", "cross-confidence peak confidence change (L→H / H→L)", "≈ 0 (no effect)", f"{pk(W['PANL+1']['low->high']['peak_dconf'],3)} / {pk(W['PANL+1']['high->low']['peak_dconf'],3)}")
    add("Swap (CC)", "cross-confidence peak confidence change (L→H / H→L)", "≈ +0.68 / −0.65 @ L40–61 (onset after L31)", f"{pk(W['CC']['low->high']['peak_dconf'],3)} / {pk(W['CC']['high->low']['peak_dconf'],3)}")

if "probe" in S:
    Q = S["probe"]
    L = Q["layers"]
    b = Q["r2_base"]
    add("Decoding", "R²_CV of log-prob baselines: mean / min / first / var / last / max / all six", "0.084 / 0.101 / 0.070 / 0.051 / 0.039 / 0.025 / 0.100", lambda: " / ".join(f"{b[k]:.3f}" for k in ("mean", "min", "first", "var", "last", "max", "all")))
    add("Decoding", "AUROC: verbal confidence / mean log-prob (collection set)", "0.71 / 0.75", lambda: f"{Q['auroc_verbal_conf']:.2f} / {Q['auroc_mean_logprob']:.2f}")
    add("Decoding", "correctness AUROC from PANL @ L20 / L28 / peak", "≈ 0.75 / ≈ 0.83 / ≈ 0.84", lambda: f"{at(Q['PANL']['auroc'], L, 20):.2f} / {at(Q['PANL']['auroc'], L, 28):.2f} / {pk(Q['PANL']['peak_auroc'])}")
    add("Decoding", "correctness AUROC from CC @ L20 / L28 / peak", "≈ 0.71 / ≈ 0.81 / ≈ 0.84", lambda: f"{at(Q['CC']['auroc'], L, 20):.2f} / {at(Q['CC']['auroc'], L, 28):.2f} / {pk(Q['CC']['peak_auroc'])}")
    add("Decoding", "correctness AUROC from QTT (control), peak", "≈ 0.57", lambda: pk(Q["QTT"]["peak_auroc"]))
    add("Decoding", "confidence R² (ridge α=1) from PANL: first layer > 0.05 / peak", "L25 / ≈ 0.48 @ L40", lambda: f"L{Q['PANL']['first_layer_r2_a1_above_0.05']} / {pk(Q['PANL']['peak_r2_a1'])}")
    add("Decoding", "confidence R² (ridge α=1) from CC: first layer > 0.05 / peak", "L28 / ≈ 0.84 @ L50", lambda: f"L{Q['CC']['first_layer_r2_a1_above_0.05']} / {pk(Q['CC']['peak_r2_a1'])}")
    add("Decoding", "R² unique to PANL beyond all six log-prob baselines, peak", "0.38 @ L40", lambda: pk(Q["PANL"]["peak_unique_all6"]))
    add("Decoding", "R² unique to CC beyond all six log-prob baselines, peak", "≈ 0.76 @ L50 (vs mean log-prob)", lambda: pk(Q["CC"]["peak_unique_all6"]))
    add("Decoding", "confidence R² (tuned ridge): AC / PANL / PANL+1, peak", "≈ 0.2 / ≈ 0.75 / ≈ 0.70", lambda: f"{pk(Q['AC']['peak_r2_tuned'])} / {pk(Q['PANL']['peak_r2_tuned'])} / {pk(Q['PANL+1']['peak_r2_tuned'])}")

for key in ("pmq", "pmq1", "pm", "pm1"):
    exp = f"attn_min_{key}"
    if exp not in S:
        continue
    A = S[exp]
    sec = f"Attention blocking (minimal prompt; {A['variant']})"
    add(sec, "baseline: clean logit diff / control-path noise floor", "n/a", f"{A['baseline_logit_diff']:.2f} / {A['noise_floor_change_pct']:.1f}% token change")
    add(sec, "CC→PANL: peak token change", "≈ 21% @ windows centred L30–36", pk(A["CC>PANL"]["peak_change"], 1, "%"))
    add(sec, "CC→PANL: largest logit-diff change", "≈ −0.8 @ L34", pk(A["CC>PANL"]["peak_dld"]))
    add(sec, "CC→PANL+1 (control): peak token change / largest logit-diff change", "≈ 10% / ≈ −0.1", f"{pk(A['CC>PANL+1']['peak_change'], 1, '%')} / {pk(A['CC>PANL+1']['peak_dld'])}")
    add(sec, "CC→Q+A: peak token change / largest logit-diff change", "≈ 10–12% (= control) / ≈ −0.25", f"{pk(A['CC>Q+A']['peak_change'], 1, '%')} / {pk(A['CC>Q+A']['peak_dld'])}")
    add(sec, "PANL→A: peak token change / largest logit-diff change", "≈ 20% @ L24–30 / ≈ −0.5 @ L22–26", f"{pk(A['PANL>A']['peak_change'], 1, '%')} / {pk(A['PANL>A']['peak_dld'])}")
    add(sec, "PANL→last A: peak token change / largest logit-diff change", "≈ 18% @ L24–30 / ≈ −0.5", f"{pk(A['PANL>last_A']['peak_change'], 1, '%')} / {pk(A['PANL>last_A']['peak_dld'])}")

if "attn_cat" in S:
    C = S["attn_cat"]
    ref = {"CC>NL+1": 10, "CC>NL": 9, "CC>A": 12, "CC>Q": 11, "CC>Q+A": 12, "ALL>NL": 18, "ALL>last_A": 51, "ALL>NL+last_A": 48,
           "ALL>last_A_keepNL": 24, "ALL>A": 68, "ALL>NL+A": 59, "ALL>A_keepNL": 51}
    for c, r in ref.items():
        add("Attention blocking (categorical, all layers)", f"{c.replace('>', '→')}: token change / Δ logit diff", f"≈ {r}%", f"{C[c]['change_pct']:.1f}% / {C[c]['dld']:.2f}")

if "ood_natural" in S:
    O = S["ood_natural"]
    ref = {"L25_PANL": {"steer_high_s2": (0.999, 0.96), "steer_low_s2": (0.999, 1.04), "steer_high_s5": (0.992, 0.91), "steer_low_s5": (0.994, 1.10),
                        "nat": ("[0.998, 1.000]", "[0.90, 1.10]"), "noise": (0.999, 0.995), "patch": (0.999, 0.94)},
           "L25_PANL+1": {"steer_high_s2": (0.999, 1.04), "steer_low_s2": (0.998, 0.96), "steer_high_s5": (0.995, 1.10), "steer_low_s5": (0.993, 0.91),
                          "nat": ("[0.997, 1.000]", "[0.90, 1.11]"), "noise": (0.999, 0.999), "patch": (0.999, 0.98)}}
    for key, r in ref.items():
        pos = key.split("_", 1)[1]
        d = O[key]
        sec = f"OOD check, layer 25, {pos} (Table 1)"
        add(sec, "natural pairwise cosine [p5, p95] / norm ratio [p5, p95]", f"{r['nat'][0]} / {r['nat'][1]}",
            f"[{d['natural_cos_p5_p95'][0]:.3f}, {d['natural_cos_p5_p95'][1]:.3f}] / [{d['natural_nr_p5_p95'][0]:.2f}, {d['natural_nr_p5_p95'][1]:.2f}]")
        for k, lab in (("steer_high_s2", "steering, high vector, scale 2"), ("steer_low_s2", "steering, low vector, scale 2"),
                       ("steer_high_s5", "steering, high vector, scale 5"), ("steer_low_s5", "steering, low vector, scale 5")):
            add(sec, f"{lab}: cosine / norm ratio", f"{r[k][0]:.3f} / {r[k][1]:.2f}", f"{d[k]['cos']:.3f} / {d[k]['nr']:.2f}")
        add(sec, "noising (mean replacement): cosine / norm ratio", f"{r['noise'][0]:.3f} / {r['noise'][1]:.3f}", f"{d['noise']['cos']:.3f} / {d['noise']['nr']:.3f}")
        if "patch" in S and "ood_L25" in S["patch"]:
            q = S["patch"]["ood_L25"][pos]
            add(sec, "patching (clean vs corruption-propagated): cosine / norm ratio", f"{r['patch'][0]:.3f} / {r['patch'][1]:.2f}", f"{q['cos']:.3f} / {q['nr']:.2f} (corrupted ÷ clean)")

out = ["| Experiment | Quantity | Paper | This replication |", "|---|---|---|---|"]
for sec, q, p, o in rows:
    out.append(f"| {sec} | {q} | {p} | {o} |")
open(RESULTS / "comparison.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
