"""Pilots for the two-turn chat format, run with VC_FMT=tt on a sample of the questions of the main run.

1. Default: does the model reply to the confidence request with a class name, and does the paper's
   "**Answer**: / **Confidence**:" scaffolding matter? (results/tt/pilot/pilot.jsonl, summary.json). This pilot
   used a first draft of the question turn (W0 below). Three ways of asking are scored on the same questions:
  A  nothing pre-filled, the reply is the class (the form used for the two-turn experiments)
       user: W0 + question | model: <answer> | user: TT_REQ | model: <class>
  B  the paper's "**Answer**: / **Confidence**:" scaffolding kept inside the turns
       user: asks for "**Answer**: $ANSWER" | model: **Answer**: <answer> | user: Fig. 8 request |
       model: **Confidence**: <class>      (class read after the teacher-forced "**Confidence**:")
  A2 the answer turn of A with the request of B

2. `--wording`: eight wordings of the question turn (results/tt/pilot/wording.json). W0 and most others make the
   model end its answer with a full stop, often followed by blank lines; "w1" gives a bare name or phrase and is
   the wording of vc.prompts.encode_tt. The confidence classes stated afterwards are distributed alike for all.

Correctness in these pilots is a proxy: the GPT-4o-mini verdict of the main run where the answer is the same after
normalisation, alias matching otherwise. `--report` recomputes the summary without the model.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import alias_match, normalize_answer, read_jsonl, write_jsonl
from vc.env import DATA_RAW, FMT, RESULTS
from vc.prompts import CAT_TAIL, CLASSES, EOT, MIDPOINTS, SOT, TT_Q_HEAD, TT_Q_TAIL, TT_REQ

assert FMT == "tt", "run with VC_FMT=tt"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--shard", type=int, default=0)  # accepted for scripts/job.sh; the pilot is a single job
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--report", action="store_true")
ap.add_argument("--wording", action="store_true", help="compare wordings of the question turn instead (results/tt/pilot/wording.json)")
args = ap.parse_args()

OUT = RESULTS / "pilot"
CCS = "**Confidence**:"
W0 = "Answer the following question. State your answer as concisely as possible (NO REASONING OR EXPLANATION).\n\nQuestion: "
B_Q = (
    "You will be shown a question. Provide your answer (NO REASONING OR EXPLANATION).\n\n"
    "At the very end of your output, format your answer as\n**Answer**: $ANSWER\n"
    "where ANSWER is your answer, stated as concisely as possible.\n\nQuestion: "
)
B_REQ = CAT_TAIL[: -len("\n\n" + CCS)]
MID = np.array(MIDPOINTS)
trials = read_jsonl(DATA_RAW / "trials.jsonl")


def chat(turns, prefill=""):
    """Chat-template text without <bos>; turns = [(role, text), ...]; ends with an open model turn."""
    return "".join(f"{SOT}{role}\n{text}{EOT}\n" for role, text in turns) + f"{SOT}model\n" + prefill


def parse_class(txt):
    t = txt.strip().strip('"*').strip()
    return next((c for c in sorted(range(10), key=lambda c: -len(CLASSES[c])) if t.startswith(CLASSES[c])), None)


def report(rows):
    from sklearn.metrics import roc_auc_score

    S = {"n": len(rows)}
    for v in ("A", "B", "A2"):
        R = [r for r in rows if v in r]
        T = [trials[r["i"]] for r in R]
        ans_key = "A" if v == "A2" else v
        same = np.array([normalize_answer(r[ans_key]["answer"]) == normalize_answer(t["answer"]) for r, t in zip(R, T)])
        em = np.array([alias_match(r[ans_key]["answer"], t["norm_aliases"]) for r, t in zip(R, T)])
        correct = np.where(same, np.array([t["correct"] for t in T]), em)
        lg = np.array([r[v]["logits"] for r in R])
        cls = lg.argmax(1)
        conf = MID[cls]
        ntok = np.array([r[ans_key]["n_tok"] for r in R])
        lp = np.array([np.mean(r[ans_key]["lp"]) if r[ans_key]["lp"] else np.nan for r in R])
        ok = ~np.isnan(lp)
        d = {
            "n": len(R),
            "same_answer_as_raw_pct": float(100 * same.mean()),
            "answer_tokens_median": float(np.median(ntok)),
            "answer_over_12_tokens_pct": float(100 * (ntok > 12).mean()),
            "answer_ends_with_period_pct": float(100 * np.mean([r[ans_key]["answer"].endswith(".") for r in R])),
            "alias_accuracy_pct": float(100 * em.mean()),
            "alias_accuracy_raw_pct": float(100 * np.mean([t["em"] for t in T])),
            "proxy_accuracy_pct": float(100 * correct.mean()),
            "accuracy_raw_pct": float(100 * np.mean([t["correct"] for t in T])),
            "first_token_is_class_pct": float(100 * np.mean([r[v]["top_valid"] for r in R])),
            "reply_parses_to_class_pct": float(100 * np.mean([r[v].get("reply_cls") is not None for r in R])),
            "reply_class_equals_logit_class_pct": float(100 * np.mean([r[v].get("reply_cls") == int(c) for r, c in zip(R, cls)])),
            "reply_is_exactly_a_class_pct": float(100 * np.mean([r[v].get("reply", "").strip() in CLASSES for r in R])),
            "share_pct": (100 * np.bincount(cls, minlength=10) / len(R)).round(1).tolist(),
            "n_top3": int((cls >= 7).sum()),
            "n_bottom3": int((cls <= 2).sum()),
            "accuracy_by_class": {CLASSES[c]: round(float(correct[cls == c].mean()), 3) for c in range(10) if (cls == c).sum() >= 10},
            "auroc": float(roc_auc_score(correct, conf)),
            "ece": float(sum(abs(correct[cls == c].mean() - MID[c]) * (cls == c).mean() for c in range(10) if (cls == c).any())),
            "r_raw_phase1_conf": float(np.corrcoef(conf, [t["conf"] for t in T])[0, 1]),
            "r_mean_logprob": float(np.corrcoef(conf[ok], lp[ok])[0, 1]),
            "auroc_mean_logprob": float(roc_auc_score(correct[ok], lp[ok])),
            "most_common_first_tokens": sorted(((k, int(c)) for k, c in zip(*np.unique([r[v]["top_str"] for r in R], return_counts=True))), key=lambda kv: -kv[1])[:6],
            "example_turns": [r[ans_key]["turn"][:80] for r in R[:8]],
            "example_replies": [r[v].get("reply", "")[:40] for r in R[:8]],
            "non_class_replies": [r[v].get("reply", "")[:60] for r in R if r[v].get("reply_cls") is None][:8],
        }
        S[v] = d
    S["raw_phase1_share_pct_same_trials"] = (100 * np.bincount([trials[r["i"]]["cls"] for r in rows], minlength=10) / len(rows)).round(1).tolist()
    os.makedirs(OUT, exist_ok=True)
    json.dump(S, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if args.report:
    report(read_jsonl(OUT / "pilot.jsonl"))
    sys.exit(0)

# ------------------------------------------------------------------------------------------------- model part
from vc.model import VC, batched, log  # noqa: E402

vc = VC()
tok = vc.tok
BOS = tok.bos_token
cls_ns = [tok(c, add_special_tokens=False).input_ids[0] for c in CLASSES]
cls_sp = [tok(" " + c, add_special_tokens=False).input_ids[0] for c in CLASSES]
pick = sorted(np.random.default_rng(1).choice(len(trials), args.n, replace=False).tolist())
rows = {i: {"i": i} for i in pick}
stop = lambda txt: EOT in txt  # noqa: E731
by_len = sorted(pick, key=lambda i: len(trials[i]["question"]))


def gen(prompts, n_new):
    out = []
    for b in batched(list(range(len(prompts))), args.bs):
        out += vc.generate([prompts[j] for j in b], max_new_tokens=n_new, stop_fn=stop)
    return out


def read(prompts, sel):
    """Class logits and full-vocabulary top token at the last prompt token."""
    enc = [tok(BOS + p, add_special_tokens=False).input_ids for p in prompts]
    assert all(e[0] == tok.bos_token_id and e[1] != tok.bos_token_id for e in enc)
    lgs, tops = [None] * len(enc), [None] * len(enc)
    for b in batched(sorted(range(len(enc)), key=lambda j: len(enc[j])), args.bs):
        ids, last = vc.pad([enc[j] for j in b])
        lg, _, top = vc.run(vc.embed(ids), last_idx=last, sel=sel)
        for k, j in enumerate(b):
            lgs[j], tops[j] = [round(float(x), 4) for x in lg[k]], int(top[k])
    return lgs, tops


# ---- wording of the question turn: how does the model end its answer, and how confident is it afterwards?
if args.wording:
    WORDINGS = {  # {q} = question
        "w0": W0 + "{q}",
        "w1": TT_Q_HEAD + "{q}" + TT_Q_TAIL,  # the wording adopted for the experiments
        "w2": "Answer the following question in as few words as possible.\n\nQuestion: {q}",
        "w3": "{q}\n\nReply with only the answer, as concisely as possible.",
        "w4": "You will be shown a question. Provide your answer, stated as concisely as possible (NO REASONING OR EXPLANATION).\n\nQuestion: {q}\nAnswer:",
        "w5": "You will be shown a question. Provide your answer, stated as concisely as possible (NO REASONING OR EXPLANATION).\n\nQuestion: {q}",
        "w6": "Answer the following question with just the answer: no sentence and no punctuation.\n\nQuestion: {q}",
        "w7": "{q}",
    }
    W = {}
    for name, tmpl in WORDINGS.items():
        users = [tmpl.format(q=trials[i]["question"].strip()) for i in by_len]
        g = gen([chat([("user", u)]) for u in users], 40)
        turns = [tok.decode(ids) for ids, _ in g]
        ans = [t.strip() for t in turns]
        keep = [j for j in range(len(by_len)) if ans[j] and len(g[j][0]) < 40]
        lgs, tops = read([chat([("user", users[j]), ("model", ans[j]), ("user", TT_REQ)]) for j in keep], cls_ns)
        cls = np.array(lgs).argmax(1)
        T = [trials[by_len[j]] for j in keep]
        same = np.array([normalize_answer(ans[j]) == normalize_answer(t["answer"]) for j, t in zip(keep, T)])
        em = np.array([alias_match(ans[j], t["norm_aliases"]) for j, t in zip(keep, T)])
        correct = np.where(same, np.array([t["correct"] for t in T]), em)
        ntok = np.array([len(tok(ans[j], add_special_tokens=False).input_ids) for j in keep])
        from sklearn.metrics import roc_auc_score

        W[name] = {
            "wording": tmpl, "n": len(by_len), "kept": len(keep),
            "ends_with_period_pct": float(100 * np.mean([a.endswith(".") for a in ans])),
            "trailing_whitespace_generated_pct": float(100 * np.mean([t != t.rstrip() for t in turns])),
            "contains_newline_pct": float(100 * np.mean(["\n" in a for a in ans])),
            "contains_markdown_pct": float(100 * np.mean(["**" in a for a in ans])),
            "answer_tokens_median": float(np.median(ntok)), "answer_over_12_tokens_pct": float(100 * (ntok > 12).mean()),
            "same_answer_as_raw_pct": float(100 * same.mean()), "alias_accuracy_pct": float(100 * em.mean()),
            "alias_accuracy_raw_pct": float(100 * np.mean([t["em"] for t in T])), "proxy_accuracy_pct": float(100 * correct.mean()),
            "first_token_is_class_pct": float(100 * np.mean([t in cls_ns for t in tops])),
            "share_pct": (100 * np.bincount(cls, minlength=10) / len(keep)).round(1).tolist(), "n_bottom3": int((cls <= 2).sum()),
            "auroc": float(roc_auc_score(correct, MID[cls])),
            "r_raw_phase1_conf": float(np.corrcoef(MID[cls], [t["conf"] for t in T])[0, 1]),
            "examples": [repr(t)[:60] for t in turns[:10]],
        }
        log(f"wording {name}: period {W[name]['ends_with_period_pct']:.0f}% | trailing ws {W[name]['trailing_whitespace_generated_pct']:.0f}% | "
            f"same as raw {W[name]['same_answer_as_raw_pct']:.0f}% | shares {W[name]['share_pct']} | bottom-3 {W[name]['n_bottom3']}/{len(keep)}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(W, open(OUT / "wording.json", "w"), indent=1)
    log("TT_PILOT WORDING DONE")
    sys.exit(0)

# ---- turn 1
for v, head in (("A", W0), ("B", B_Q)):
    g = gen([chat([("user", head + trials[i]["question"].strip())]) for i in by_len], 40)
    for i, (ids, lps) in zip(by_len, g):
        turn = tok.decode(ids).strip()
        ans = turn.split("**Answer**:", 1)[1].strip() if "**Answer**:" in turn else turn
        k = len(tok(turn, add_special_tokens=False).input_ids) - len(tok(ans, add_special_tokens=False).input_ids) if v == "B" else 0
        rows[i][v] = {"turn": turn, "answer": ans, "n_tok": len(tok(ans, add_special_tokens=False).input_ids), "lp": [round(x, 4) for x in lps[k:]]}
    log(f"turn 1 generated for {v}")

# ---- turn 2
SPECS = {  # variant -> (question head, answer-turn source, request, prefill, class tokens)
    "A": (W0, "A", TT_REQ, "", cls_ns),
    "B": (B_Q, "B", B_REQ, CCS, cls_sp),
    "A2": (W0, "A", B_REQ, CCS, cls_sp),
}
for v, (head, src, req, prefill, sel) in SPECS.items():
    idx = [i for i in by_len if rows[i][src]["turn"]]
    base = [chat([("user", head + trials[i]["question"].strip()), ("model", rows[i][src]["turn"]), ("user", req)]) for i in idx]
    lgs, tops = read([p + prefill for p in base], sel)
    reps = gen(base, 14)  # free reply, nothing pre-filled
    for i, lg, top, (rid, _) in zip(idx, lgs, tops, reps):
        rep = tok.decode(rid)
        body = rep.split(CCS, 1)[1] if (prefill and CCS in rep) else (rep if not prefill else "")
        d = rows[i][src if v == src else v] if v == src else rows[i].setdefault(v, {})
        d.update({"logits": lg, "top": top, "top_str": tok.decode(top), "top_valid": top in sel, "reply": rep, "reply_cls": parse_class(body),
                  "reply_has_prefix": bool(prefill) and rep.strip().startswith(CCS)})
    log(f"turn 2 scored for {v}: {len(idx)} trials")

out = [rows[i] for i in pick if all("logits" in rows[i].get(v, {}) for v in SPECS)]
os.makedirs(OUT, exist_ok=True)
write_jsonl(OUT / "pilot.jsonl", out)
log("TT_PILOT DONE", len(out))
report(out)
