"""Merge behavioural shards -> data/trials.jsonl, splits, experiment trial sets and summary statistics."""
import json
import os
import sys
from collections import Counter
from glob import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.metrics import roc_auc_score

from vc.data import read_jsonl, write_jsonl
from vc.env import DATA, RESULTS
from vc.prompts import CLASSES, HIGH_CLASSES, LOW_CLASSES, MIDPOINTS

rows = []
for f in sorted(glob(str(RESULTS / "behav" / "shard*.jsonl"))):
    rows += read_jsonl(f)
print(f"merged {len(rows)} rows")

# Grader. The paper marks answers with GPT-4o-mini; scripts/grade_gpt.py stores its verdicts. Without that file
# (or with GRADER=gemma) an answer counts as correct if it matches a TriviaQA alias or Gemma itself accepts it.
GPT_FILE = RESULTS / "behav" / "gpt4omini.jsonl"
GRADER = os.environ.get("GRADER", "gpt" if GPT_FILE.exists() else "gemma")
gpt = {r["qid"]: r["verdict"] for r in read_jsonl(GPT_FILE)} if GPT_FILE.exists() else {}
print(f"grader: {GRADER} ({len(gpt)} GPT-4o-mini verdicts on file)")

MAX_ANS_TOK = 12
# minimal 0-9 prompt variants run in behav.py: key -> (prompt kind, separator after the answer)
MIN_VARIANTS = {"pm": ("min", "\n\n"), "pmq": ("minq", "\n\n"), "pm1": ("min", "\n"), "pmq1": ("minq", "\n")}
MIN_KEY = os.environ.get("MIN_KEY", "pmq")  # variant used for the attention-blocking experiment
valid = []
drop = Counter()
for r in rows:
    if not r.get("answer"):
        drop["empty answer"] += 1
    elif "p1_logits" not in r or any(f"{k}_logits" not in r for k in MIN_VARIANTS):
        drop["tokenisation round-trip"] += 1
    elif not r["p1_top_valid"]:
        drop["phase-1 first token is not a confidence class"] += 1
    elif r["n_a"] > MAX_ANS_TOK:
        drop["answer longer than %d tokens" % MAX_ANS_TOK] += 1
    else:
        valid.append(r)
print("dropped:", dict(drop), "| valid:", len(valid))

for r in valid:
    r["cls"] = int(np.argmax(r["p1_logits"]))
    r["conf"] = MIDPOINTS[r["cls"]]
    r["judge"] = r["judge_margin"] > 0
    r["correct_gemma"] = bool(r["em"] or r["judge"])  # alias match or Gemma-as-grader
    r["gpt"] = gpt.get(r["qid"])  # GPT-4o-mini verdict: True / False / None (not graded or unparsed)
    r["correct"] = bool(r["gpt"]) if GRADER == "gpt" and r["gpt"] is not None else r["correct_gemma"]
    p = np.exp(np.array(r["p1_logits"]) - np.max(r["p1_logits"]))
    r["conf_exp"] = float((p / p.sum()) @ np.array(MIDPOINTS))  # expected confidence, used only to break ties
    for k in MIN_VARIANTS:
        r[f"dig_{k}"] = int(np.argmax(r[f"{k}_logits"]))
    lp = np.array(r["p0_ans_lp"])
    r["lp"] = {
        "mean": float(lp.mean()),
        "min": float(lp.min()),
        "max": float(lp.max()),
        "var": float(lp.var()),
        "first": float(lp[0]),
        "last": float(lp[-1]),
    }

rng = np.random.default_rng(0)
perm = rng.permutation(len(valid))
N_COLLECT = 3000
for rank, i in enumerate(perm):
    valid[i]["split"] = "collect" if rank < N_COLLECT else "test"

cls = np.array([r["cls"] for r in valid])
conf = np.array([r["conf"] for r in valid])
correct = np.array([r["correct"] for r in valid])
em = np.array([r["em"] for r in valid])
judge = np.array([r["judge"] for r in valid])
split = np.array([r["split"] for r in valid])
hi = np.isin(cls, HIGH_CLASSES)
lo = np.isin(cls, LOW_CLASSES)


def ece(conf, correct):
    e = 0.0
    for c in np.unique(conf):
        m = conf == c
        e += m.mean() * abs(correct[m].mean() - c)
    return float(e)


summ = {
    "n_total": len(rows),
    "n_valid": len(valid),
    "dropped": dict(drop),
    "accuracy": float(correct.mean()),
    "accuracy_em_only": float(em.mean()),
    "accuracy_judge_only": float(judge.mean()),
    "em_and_judge_agree": float((em == judge).mean()),
    "em_true_judge_false": int((em & ~judge).sum()),
    "em_false_judge_true": int((~em & judge).sum()),
    "ece": ece(conf, correct),
    "auroc_verbal_conf": float(roc_auc_score(correct, conf)),
    "class_counts": {CLASSES[c]: int((cls == c).sum()) for c in range(10)},
    "class_accuracy": {CLASSES[c]: (float(correct[cls == c].mean()) if (cls == c).any() else None) for c in range(10)},
    "n_high": int(hi.sum()),
    "n_low": int(lo.sum()),
    "n_low_test": int((lo & (split == "test")).sum()),
    "n_low_collect": int((lo & (split == "collect")).sum()),
    "mean_conf": float(conf.mean()),
}
# phase-0 confidence (same run as the answer) vs phase-1 confidence, and log-prob relations
p0 = np.array([MIDPOINTS[r["p0_conf"]] if r["p0_conf"] is not None else np.nan for r in valid])
lpm = np.array([r["lp"]["mean"] for r in valid])
ok = ~np.isnan(p0)
cg = np.array([r["correct_gemma"] for r in valid])
has_gpt = np.array([r["gpt"] is not None for r in valid])
g = np.array([bool(r["gpt"]) for r in valid])
summ["grader"] = GRADER
summ["accuracy_gemma_or_alias"] = float(cg.mean())
summ["gpt_graded"] = int(has_gpt.sum())
if has_gpt.any():
    summ["accuracy_gpt"] = float(g[has_gpt].mean())
    summ["gpt_agrees_with_gemma_or_alias"] = float((g == cg)[has_gpt].mean())
    summ["gpt_agrees_with_alias"] = float((g == em)[has_gpt].mean())
    summ["gpt_true_alias_false"] = int((g & ~em & has_gpt).sum())
    summ["gpt_false_alias_true"] = int((~g & em & has_gpt).sum())
# Phase-0 confidence as a behavioural measure in its own right (what the paper's Fig. 9 resembles)
p0c = np.array([r["p0_conf"] if r["p0_conf"] is not None else -1 for r in valid])
summ["phase0"] = {
    "ece": ece(np.round(p0[ok], 2), correct[ok]),
    "auroc_verbal_conf": float(roc_auc_score(correct[ok], p0[ok])),
    "class_counts": {CLASSES[c]: int((p0c == c).sum()) for c in range(10)},
    "class_accuracy": {CLASSES[c]: (float(correct[p0c == c].mean()) if (p0c == c).any() else None) for c in range(10)},
    "balanced_baseline": float((p0[ok & (p0c >= 7)].mean() + p0[ok & (p0c <= 2)].mean()) / 2),
}
summ["balanced_baseline"] = float((conf[hi].mean() + conf[lo].mean()) / 2)
summ["n_p0_conf_valid"] = int(ok.sum())
summ["r_p0conf_p1conf"] = float(np.corrcoef(p0[ok], conf[ok])[0, 1])
summ["r_logprob_p0conf"] = float(np.corrcoef(lpm[ok], p0[ok])[0, 1])
summ["r_logprob_p1conf"] = float(np.corrcoef(lpm, conf)[0, 1])
summ["auroc_mean_logprob"] = float(roc_auc_score(correct, lpm))
# minimal numeric prompt variants
summ["min_prompt"] = {}
for k, (kind, panl) in MIN_VARIANTS.items():
    dig = np.array([r[f"dig_{k}"] for r in valid])
    dv = np.array([r[f"{k}_top_valid"] for r in valid])
    dconf = (dig + 0.5) / 10.0
    summ["min_prompt"][k] = {
        "kind": kind,
        "panl": panl,
        "first_token_is_digit": float(dv.mean()),
        "digit_counts": {int(d): int((dig == d).sum()) for d in range(10)},
        "digit_accuracy": {int(d): (float(correct[dig == d].mean()) if (dig == d).any() else None) for d in range(10)},
        "ece": ece(np.round(dconf, 2), correct),
        "auroc": float(roc_auc_score(correct, dig)),
        "auroc_digit_first_only": float(roc_auc_score(correct[dv], dig[dv])) if dv.sum() > 50 and len(set(correct[dv])) > 1 else None,
        "r_digit_p1conf": float(np.corrcoef(dig, conf)[0, 1]),
    }
summ["min_key"] = MIN_KEY
dvalid = np.array([r[f"{MIN_KEY}_top_valid"] for r in valid])

# ---------------------------------------------------------------- experiment trial sets (indices into `valid`)
test = split == "test"
col = split == "collect"


def pick(mask, n, replace=False):
    idx = np.flatnonzero(mask)
    if replace or len(idx) < n:
        return [int(i) for i in rng.choice(idx, n, replace=len(idx) < n)]
    return [int(i) for i in rng.choice(idx, n, replace=False)]


sets = {}
sets["steer"] = pick(test & hi, 100) + pick(test & lo, 100)
sets["patch"] = pick(test & hi, 200)
sets["noise"] = pick(test & hi, 200) + pick(test & lo, 200)
sets["swap_high"] = pick(test & hi, 400)
low_pool = np.flatnonzero(test & lo)
sets["swap_low_pool"] = [int(i) for i in low_pool]
sets["swap_low"] = [int(i) for i in rng.choice(low_pool, 400, replace=len(low_pool) < 400)]
sets["calib"] = pick(col & hi, 50) + pick(col & lo, 50)  # mean-ablation / corruption calibration set
sets["attn_cat"] = pick(test, 500)
sets["attn_min"] = pick(test & dvalid, 500)
sets["min_kind"], sets["min_panl"], sets["min_key"] = MIN_VARIANTS[MIN_KEY][0], MIN_VARIANTS[MIN_KEY][1], MIN_KEY
# steering-vector trials: correct answers only, 25 highest / 25 lowest ranked by confidence (collection set).
# Many trials tie for those ranks (all "Almost certain" / "No chance"), and the paper does not say how ties are
# broken. Main sets: ties broken at random (own seeded RNG, so the other sets do not depend on it); two more
# random draws and the "most extreme trials first" tie-break (by expected confidence) are kept for comparison.
cc = np.flatnonzero(col & correct)
key = np.array([(valid[i]["conf"], valid[i]["conf_exp"]) for i in cc])


def vec_sets(tiebreak):
    o = np.lexsort((tiebreak, key[:, 0]))
    return [int(i) for i in cc[o[:25]]], [int(i) for i in cc[o[-25:]]]


draws = [vec_sets(np.random.default_rng(100 + k).random(len(cc))) for k in range(3)]
sets["vec_low"], sets["vec_high"] = draws[0]
sets["vec_draws"] = [{"low": lo_, "high": hi_} for lo_, hi_ in draws]
sets["vec_low_extreme"], sets["vec_high_extreme"] = vec_sets(key[:, 1])
# patching robustness set: trials chosen by Phase-0 confidence, as the paper does (own RNG: other sets unchanged)
p0_pool = [int(i) for i in np.flatnonzero(test) if valid[i]["p0_conf"] in (7, 8, 9)]
sets["patch_p0"] = sorted(int(i) for i in np.random.default_rng(123).choice(p0_pool, 200, replace=False))
summ["vec_low_classes"] = dict(Counter(CLASSES[valid[i]["cls"]] for i in sets["vec_low"]))
summ["vec_high_classes"] = dict(Counter(CLASSES[valid[i]["cls"]] for i in sets["vec_high"]))
summ["set_sizes"] = {k: len(v) for k, v in sets.items()}
summ["swap_low_unique"] = len(set(sets["swap_low"]))

for i, r in enumerate(valid):
    r["idx"] = i
write_jsonl(DATA / "trials.jsonl", valid)
json.dump(sets, open(DATA / "sets.json", "w"))
json.dump(summ, open(RESULTS / "behav" / "summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))

# ---------------------------------------------------------------- swap plan (C.1.7): donors matched to
# recipients on tokenised question / answer length via quantile bins
nq = np.array([r["n_q"] for r in valid])
na = np.array([r["n_a"] for r in valid])
hi_pool = np.array(sorted(set(sets["swap_high"])))
lo_pool = np.array(sorted(set(sets["swap_low_pool"])))
uni = np.concatenate([hi_pool, lo_pool])
q_edges = np.quantile(nq[uni], [0.2, 0.4, 0.6, 0.8])
a_edges = np.quantile(na[uni], [1 / 3, 2 / 3])
qb = np.digitize(nq, q_edges, right=True)
ab = np.digitize(na, a_edges, right=True)
print("question-length bin edges", q_edges, "answer-length bin edges", a_edges)


def donor_for(rec, pool):
    cand = pool[pool != rec]
    c = cand[(qb[cand] == qb[rec]) & (ab[cand] == ab[rec])]
    level = 0
    if len(c) == 0:
        c = cand[qb[cand] == qb[rec]]
        level = 1
        if len(c) == 0:
            c, level = cand, 2
        d = np.abs(na[c] - na[rec])
        c = c[d == d.min()]
    return int(rng.choice(c)), level


plan = []
for kind, recs in (("high", sets["swap_high"]), ("low", sets["swap_low"])):
    for rec in recs:
        dh, lh = donor_for(rec, hi_pool)
        dl, ll = donor_for(rec, lo_pool)
        plan.append({"rec": int(rec), "kind": kind, "donor_high": dh, "donor_low": dl, "lvl_high": lh, "lvl_low": ll})
stats = {}
for kind in ("high", "low"):
    for d in ("high", "low"):
        P = [p for p in plan if p["kind"] == kind]
        dq = np.array([abs(nq[p["rec"]] - nq[p[f"donor_{d}"]]) for p in P])
        da = np.array([abs(na[p["rec"]] - na[p[f"donor_{d}"]]) for p in P])
        stats[f"{kind}<-{d}"] = {
            "q_bin_match": float(np.mean([qb[p["rec"]] == qb[p[f"donor_{d}"]] for p in P])),
            "a_bin_match": float(np.mean([ab[p["rec"]] == ab[p[f"donor_{d}"]] for p in P])),
            "mean_abs_dLQ": float(dq.mean()),
            "mean_abs_dLA": float(da.mean()),
        }
json.dump({"recipients": plan, "match_stats": stats}, open(DATA / "swap_plan.json", "w"))
print("swap matching:", json.dumps(stats, indent=1))
