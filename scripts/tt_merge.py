"""Two-turn format: merge the behavioural shards -> data/tt/trials.jsonl, trial sets, swap plan, summary.

Run with VC_FMT=tt after scripts/tt_behav.py and scripts/grade_gpt.py. Same structure as scripts/merge_behav.py:
3,000 random trials form the collection set, all interventions are tested on the remaining trials.

Grading: an answer that equals the main run's answer after normalisation keeps that run's GPT-4o-mini verdict;
the others are graded by scripts/grade_gpt.py (results/tt/behav/gpt4omini.jsonl). Without a GPT verdict the
fallback is alias match or Gemma-as-grader, as in the main run.

Low-confidence reservoir: the model states a bottom-3 class in under 1% of two-turn trials, and only about one
in seven of those answers is correct, so the 3,000-trial collection set holds too few of them for the steering
vector (25 correct low-confidence trials) and the calibration means (50 low-confidence trials). If
results/tt/behav_sup exists (questions of the TriviaQA train split, scripts/tt_behav.py --split train), its valid
bottom-3 trials are appended to the collection set (src = "train"). Everything else uses validation questions
only: the behavioural statistics, the probes and all test sets. The test sets are balanced on the number of
low-confidence test trials there are.
"""
import json
import os
import sys
from collections import Counter
from glob import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.metrics import roc_auc_score

from vc.data import normalize_answer, read_jsonl, write_jsonl
from vc.env import DATA, DATA_RAW, FMT, RESULTS
from vc.prompts import CLASSES, HIGH_CLASSES, LOW_CLASSES, MIDPOINTS

assert FMT == "tt", "run with VC_FMT=tt"
rows = [r for f in sorted(glob(str(RESULTS / "behav" / "shard*.jsonl"))) for r in read_jsonl(f)]
print(f"merged {len(rows)} rows")
raw = {r["qid"]: r for r in read_jsonl(DATA_RAW / "trials.jsonl")}
GPT_FILE = RESULTS / "behav" / "gpt4omini.jsonl"
gpt_new = {r["qid"]: r["verdict"] for r in read_jsonl(GPT_FILE)} if GPT_FILE.exists() else {}

MAX_ANS_TOK = 12


def keep_valid(rows):
    valid, drop = [], Counter()
    for r in rows:
        if not r.get("answer"):
            drop["empty answer"] += 1
        elif r.get("truncated"):
            drop["answer turn not finished within 40 tokens"] += 1
        elif "p1_logits" not in r:
            drop["tokenisation round-trip"] += 1
        elif not r["p1_top_valid"]:
            drop["first token of the reply is not a confidence class"] += 1
        elif r["n_a"] > MAX_ANS_TOK:
            drop["answer longer than %d tokens" % MAX_ANS_TOK] += 1
        else:
            valid.append(r)
    return valid, drop


valid, drop = keep_valid(rows)
print("dropped:", dict(drop), "| valid:", len(valid))


def add_stats(r):
    r["cls"] = int(np.argmax(r["p1_logits"]))
    r["conf"] = MIDPOINTS[r["cls"]]
    p = np.exp(np.array(r["p1_logits"]) - np.max(r["p1_logits"]))
    r["conf_exp"] = float((p / p.sum()) @ np.array(MIDPOINTS))
    lp = np.array(r["p0_ans_lp"])
    r["lp"] = {"mean": float(lp.mean()), "min": float(lp.min()), "max": float(lp.max()), "var": float(lp.var()),
               "first": float(lp[0]), "last": float(lp[-1])}


src = Counter()
for r in valid:
    add_stats(r)
    r["src"] = "val"
    r["judge"] = r["judge_margin"] > 0
    r["correct_gemma"] = bool(r["em"] or r["judge"])
    m = raw.get(r["qid"])
    r["same_as_raw"] = m is not None and normalize_answer(r["answer"]) == normalize_answer(m["answer"])
    if gpt_new.get(r["qid"]) is not None:
        r["gpt"], r["gpt_src"] = gpt_new[r["qid"]], "graded"
    elif r["same_as_raw"] and m["gpt"] is not None:
        r["gpt"], r["gpt_src"] = m["gpt"], "main run"
    else:
        r["gpt"], r["gpt_src"] = None, "none"
    src[r["gpt_src"]] += 1
    r["correct"] = bool(r["gpt"]) if r["gpt"] is not None else r["correct_gemma"]
    r["raw_cls"] = m["cls"] if m is not None else None  # Phase-1 class of the same question in the main run
    r["raw_p0_conf"] = m["p0_conf"] if m is not None else None
print("GPT-4o-mini verdicts:", dict(src))

rng = np.random.default_rng(0)
perm = rng.permutation(len(valid))
N_COLLECT = 3000
for rank, i in enumerate(perm):
    valid[i]["split"] = "collect" if rank < N_COLLECT else "test"
n_val = len(valid)

cls = np.array([r["cls"] for r in valid])
conf = np.array([r["conf"] for r in valid])
correct = np.array([r["correct"] for r in valid])
em = np.array([r["em"] for r in valid])
split = np.array([r["split"] for r in valid])
same = np.array([r["same_as_raw"] for r in valid])
lpm = np.array([r["lp"]["mean"] for r in valid])
hi, lo = np.isin(cls, HIGH_CLASSES), np.isin(cls, LOW_CLASSES)
test, col = split == "test", split == "collect"


def ece(conf, correct):
    return float(sum((conf == c).mean() * abs(correct[conf == c].mean() - c) for c in np.unique(conf)))


has_raw = np.array([r["raw_cls"] is not None for r in valid])
raw_conf = np.array([MIDPOINTS[r["raw_cls"]] if r["raw_cls"] is not None else np.nan for r in valid])
raw_p0 = np.array([MIDPOINTS[r["raw_p0_conf"]] if r["raw_p0_conf"] is not None else np.nan for r in valid])
both = same & has_raw
summ = {
    "n_total": len(rows),
    "n_valid": len(valid),
    "dropped": dict(drop),
    "gpt_verdict_source": dict(src),
    "accuracy": float(correct.mean()),
    "accuracy_em_only": float(em.mean()),
    "accuracy_gemma_or_alias": float(np.mean([r["correct_gemma"] for r in valid])),
    "ece": ece(conf, correct),
    "auroc_verbal_conf": float(roc_auc_score(correct, conf)),
    "class_counts": {CLASSES[c]: int((cls == c).sum()) for c in range(10)},
    "class_accuracy": {CLASSES[c]: (float(correct[cls == c].mean()) if (cls == c).any() else None) for c in range(10)},
    "n_high": int(hi.sum()),
    "n_low": int(lo.sum()),
    "n_low_test": int((lo & test).sum()),
    "n_low_collect": int((lo & col).sum()),
    "n_low_collect_correct": int((lo & col & correct).sum()),
    "mean_conf": float(conf.mean()),
    "balanced_baseline": float((conf[hi].mean() + conf[lo].mean()) / 2),
    "r_logprob_conf": float(np.corrcoef(lpm, conf)[0, 1]),
    "auroc_mean_logprob": float(roc_auc_score(correct, lpm)),
    "answer_tokens_mean": float(np.mean([r["n_a"] for r in valid])),
    "answer_single_token_pct": float(100 * np.mean([r["n_a"] == 1 for r in valid])),
    "prompt_tokens_mean": float(np.mean([r["n_tok"] for r in valid])),
    # relation to the main (raw-format) run on the same questions
    "same_answer_as_main_run_pct": float(100 * same[has_raw].mean()),
    "n_same_answer": int(both.sum()),
    "r_conf_main_phase1_same_answer": float(np.corrcoef(conf[both], raw_conf[both])[0, 1]),
    "r_conf_main_phase0_same_answer": float(np.corrcoef(conf[both & ~np.isnan(raw_p0)], raw_p0[both & ~np.isnan(raw_p0)])[0, 1]),
    "same_class_as_main_phase1_pct": float(100 * (conf[both] == raw_conf[both]).mean()),
    "mean_conf_main_phase1_same_answer": float(raw_conf[both].mean()),
    "mean_conf_two_turn_same_answer": float(conf[both].mean()),
}

# ---------------------------------------------------------------- low-confidence reservoir (train split)
SUP = RESULTS / "behav_sup"
sup_files = sorted(glob(str(SUP / "shard*.jsonl")))
if sup_files:
    sup_rows = [r for f in sup_files for r in read_jsonl(f)]
    sup_valid, sup_drop = keep_valid(sup_rows)
    gpt_sup = {r["qid"]: r["verdict"] for r in read_jsonl(SUP / "gpt4omini.jsonl")} if (SUP / "gpt4omini.jsonl").exists() else {}
    for r in sup_valid:
        add_stats(r)
    sup_cls = np.array([r["cls"] for r in sup_valid])
    sup_low = [r for r in sup_valid if r["cls"] in LOW_CLASSES]
    for r in sup_low:
        r["src"], r["split"] = "train", "collect"
        r["gpt"] = gpt_sup.get(r["qid"])
        r["gpt_src"] = "graded" if r["gpt"] is not None else "none"
        r["correct_gemma"] = bool(r["em"])  # no Gemma-as-grader pass for the reservoir: alias match is the fallback
        r["correct"] = bool(r["gpt"]) if r["gpt"] is not None else r["correct_gemma"]
        r["same_as_raw"], r["raw_cls"], r["raw_p0_conf"] = False, None, None
    valid += sup_low
    summ["reservoir"] = {
        "n_questions": len(sup_rows), "n_valid": len(sup_valid), "dropped": dict(sup_drop),
        "class_counts": {CLASSES[c]: int((sup_cls == c).sum()) for c in range(10)},
        "n_low": len(sup_low), "low_pct": float(100 * len(sup_low) / len(sup_valid)),
        "n_low_gpt_graded": int(sum(r["gpt"] is not None for r in sup_low)),
        "n_low_correct": int(sum(r["correct"] for r in sup_low)),
    }
    print("reservoir:", json.dumps(summ["reservoir"]))

# ---------------------------------------------------------------- experiment trial sets (indices into `valid`)
cls = np.array([r["cls"] for r in valid])
conf = np.array([r["conf"] for r in valid])
correct = np.array([r["correct"] for r in valid])
is_val = np.array([r["src"] == "val" for r in valid])
test = np.array([r["split"] == "test" for r in valid])
col = ~test
hi, lo = np.isin(cls, HIGH_CLASSES) & is_val, np.isin(cls, LOW_CLASSES)  # high: validation questions only


def pick(mask, n, seed):
    """n trials drawn without replacement (all of them, in random order, if fewer are available). Every set
    has its own seed, so that one set does not change when the pool of another one does."""
    idx = np.flatnonzero(mask)
    return [int(i) for i in np.random.default_rng([0, seed]).choice(idx, min(n, len(idx)), replace=False)]


sets = {}
n_lo = int((test & lo).sum())  # all low-confidence test trials are used; high-confidence trials are matched in number
sets["steer"] = pick(test & hi, n_lo, 1) + pick(test & lo, n_lo, 2)
sets["patch"] = pick(test & hi, 200, 3)
sets["noise"] = pick(test & hi, n_lo, 4) + pick(test & lo, n_lo, 5)
sets["swap_high"] = pick(test & hi, 100, 6)
sets["swap_low_pool"] = [int(i) for i in np.flatnonzero(test & lo)]
sets["swap_low"] = pick(test & lo, n_lo, 7)
sets["calib"] = pick(col & hi, 50, 8) + pick(col & lo, 50, 9)  # mean-ablation / corruption calibration set
sets["attn_cat"] = pick(test & is_val, 500, 10)
sets["attn_win"] = sets["attn_cat"][:200]
# steering-vector trials: correct answers only, 25 highest / 25 lowest ranked by confidence class in the
# collection set, ties broken at random (as in the main run)
cc = np.flatnonzero(col & correct)
o = np.lexsort((np.random.default_rng(100).random(len(cc)), conf[cc]))
sets["vec_low"], sets["vec_high"] = [int(i) for i in cc[o[:25]]], [int(i) for i in cc[o[-25:]]]
summ["vec_low_classes"] = dict(Counter(CLASSES[valid[i]["cls"]] for i in sets["vec_low"]))
summ["vec_high_classes"] = dict(Counter(CLASSES[valid[i]["cls"]] for i in sets["vec_high"]))
summ["vec_low_from_reservoir"] = int(sum(valid[i]["src"] == "train" for i in sets["vec_low"]))
summ["set_sizes"] = {k: len(v) for k, v in sets.items()}
summ["set_from_reservoir"] = {k: int(sum(valid[i]["src"] == "train" for i in v)) for k, v in sets.items()}
summ["low_pool"] = {"test": int((test & lo).sum()), "collect": int((col & lo).sum()), "collect_correct": int((col & lo & correct).sum()),
                    "class_counts": {CLASSES[c]: int((cls[lo] == c).sum()) for c in LOW_CLASSES}}

for i, r in enumerate(valid):
    r["idx"] = i
os.makedirs(DATA, exist_ok=True)
write_jsonl(DATA / "trials.jsonl", valid)
json.dump(sets, open(DATA / "sets.json", "w"))
json.dump(summ, open(RESULTS / "behav" / "summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))

# ---------------------------------------------------------------- swap plan: donors matched to recipients on
# tokenised question / answer length via quantile bins (same procedure as scripts/merge_behav.py)
rng = np.random.default_rng([0, 11])
nq = np.array([r["n_q"] for r in valid])
na = np.array([r["n_a"] for r in valid])
hi_pool = np.array(sorted(set(sets["swap_high"])))
lo_pool = np.array(sorted(set(sets["swap_low_pool"])))
uni = np.concatenate([hi_pool, lo_pool])
qb = np.digitize(nq, np.quantile(nq[uni], [0.2, 0.4, 0.6, 0.8]), right=True)
ab = np.digitize(na, np.quantile(na[uni], [1 / 3, 2 / 3]), right=True)


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
        stats[f"{kind}<-{d}"] = {
            "q_bin_match": float(np.mean([qb[p["rec"]] == qb[p[f"donor_{d}"]] for p in P])),
            "a_bin_match": float(np.mean([ab[p["rec"]] == ab[p[f"donor_{d}"]] for p in P])),
            "mean_abs_dLQ": float(np.mean([abs(nq[p["rec"]] - nq[p[f"donor_{d}"]]) for p in P])),
            "mean_abs_dLA": float(np.mean([abs(na[p["rec"]] - na[p[f"donor_{d}"]]) for p in P])),
        }
json.dump({"recipients": plan, "match_stats": stats}, open(DATA / "swap_plan.json", "w"))
print("swap matching:", json.dumps(stats, indent=1))
