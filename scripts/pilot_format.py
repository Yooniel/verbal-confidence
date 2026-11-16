"""Pilot: which input format reproduces the paper's Phase-1 confidence behaviour?

The paper's Fig. 9B prints the class counts of Gemma's confidence reports (n = 7,858), and its steering baseline
(0.55) is what those counts imply, so they describe the confidence prompt the experiments run on. The format
used for the main runs here (raw text, blank line after the answer) gives a different distribution. This script
scores alternative ways of feeding the same Fig. 8 prompt on the same trials.

Phase-1 variants (same question and stored Phase-0 answer; class read from the next-token logits):
  raw_nn / raw_nnn / raw_n : <bos> + prompt, with two / three / one newline after the answer
  raw_nobos                : raw_nn without <bos>
  chat_open                : <bos><start_of_turn>user\n + prompt (turn left open, CC is the last token)
  chat_prefill_full        : chat template, whole prompt as the user turn, model turn pre-filled "**Confidence**:"
  chat_prefill_split       : same, but the final "**Confidence**:" line is removed from the user turn
  chat_user                : chat template, whole prompt as the user turn, nothing pre-filled (diagnostic only:
                             here CC is not the last token, which the paper's layer-61 CC results rule out)
Phase-0 variant:
  p0chat                   : Phase-0 instructions as a chat user turn; the generated answer is then scored with
                             raw_nn and chat_prefill_split

`--report` aggregates the shards (no model needed) into results/pilot_format/summary.json.
"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import alias_match, read_jsonl, write_jsonl
from vc.env import DATA, RESULTS
from vc.prompts import CLASSES, MIDPOINTS, PHASE0_HEAD, PROMPT_KINDS

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n", type=int, default=1000, help="trials sampled from data/trials.jsonl (all shards together)")
ap.add_argument("--n-p0", type=int, default=400, help="of those, questions re-answered with a chat-template Phase 0")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--report", action="store_true")
args = ap.parse_args()

OUT = RESULTS / "pilot_format"
PAPER_COUNTS = [119, 96, 259, 2, 1, 0, 304, 4, 1632, 5441]  # paper Fig. 9B, "No chance" ... "Almost certain"
PAPER_ACC = {0: 0.17, 1: 0.21, 2: 0.26, 6: 0.375, 8: 0.63, 9: 0.855}  # read off paper Fig. 9A
MID = np.array(MIDPOINTS)
P1_VARIANTS = ["raw_nn", "raw_nnn", "raw_n", "raw_nobos", "chat_open", "chat_prefill_full", "chat_prefill_split", "chat_user"]


def describe(cls, correct, em, p0, lp, logits):
    """Summary statistics of one variant. cls: argmax class per trial; logits [n, 10]."""
    from sklearn.metrics import roc_auc_score

    n = len(cls)
    share = np.bincount(cls, minlength=10) / n
    paper = np.array(PAPER_COUNTS) / sum(PAPER_COUNTS)
    conf = MID[cls]
    hi, lo = cls >= 7, cls <= 2
    ece = sum(abs(correct[cls == c].mean() - MID[c]) * (cls == c).mean() for c in range(10) if (cls == c).any())
    srt = np.sort(logits, -1)
    zy = np.take_along_axis(logits, cls[:, None], 1)[:, 0]
    ld = zy - (logits.sum(1) - zy) / 9.0
    ok = p0 >= 0
    return {
        "n": n,
        "share_pct": (100 * share).round(1).tolist(),
        "tv_distance_to_paper": float(0.5 * np.abs(share - paper).sum()),
        "accuracy_by_class": {CLASSES[c]: float(correct[cls == c].mean()) for c in range(10) if (cls == c).sum() >= 15},
        "accuracy_by_class_alias_only": {CLASSES[c]: float(em[cls == c].mean()) for c in range(10) if (cls == c).sum() >= 15},
        "auroc": float(roc_auc_score(correct, conf)) if 0 < correct.mean() < 1 else None,
        "ece": float(ece),
        "balanced_baseline": float((conf[hi].mean() + conf[lo].mean()) / 2) if hi.any() and lo.any() else None,
        "r_phase0_conf": float(np.corrcoef(conf[ok], MID[p0[ok]])[0, 1]),
        "r_mean_logprob": float(np.corrcoef(conf, lp)[0, 1]),
        "median_top2_margin": float(np.median(srt[:, -1] - srt[:, -2])),
        "mean_logit_diff_top3": float(ld[hi].mean()) if hi.any() else None,
    }


if args.report:
    rows = [r for f in sorted(glob(str(OUT / "shard*.jsonl"))) for r in read_jsonl(f)]
    trials = read_jsonl(DATA / "trials.jsonl")
    S = {"n_trials": len(rows), "paper_share_pct": (100 * np.array(PAPER_COUNTS) / sum(PAPER_COUNTS)).round(1).tolist(),
         "paper_accuracy_by_class": {CLASSES[c]: a for c, a in PAPER_ACC.items()}, "paper_balanced_baseline": 0.553, "variants": {}}
    for v in P1_VARIANTS:
        R = [r for r in rows if v in r["v"]]
        T = [trials[r["i"]] for r in R]
        top = np.array([r["v"][v]["top"] for r in R])
        sp, ns = np.array([r["v"][v]["sp"] for r in R]), np.array([r["v"][v]["ns"] for r in R])
        valid_sp, valid_ns = np.array([r["v"][v]["top_sp"] for r in R]), np.array([r["v"][v]["top_ns"] for r in R])
        logits = ns if valid_ns.mean() > valid_sp.mean() else sp
        d = describe(logits.argmax(1), np.array([t["correct"] for t in T]), np.array([t["em"] for t in T]),
                     np.array([-1 if t["p0_conf"] is None else t["p0_conf"] for t in T]), np.array([t["lp"]["mean"] for t in T]), logits)
        d["first_token_is_class_pct"] = float(100 * max(valid_sp.mean(), valid_ns.mean()))
        d["class_tokens_read"] = "no leading space" if valid_ns.mean() > valid_sp.mean() else "leading space"
        cnt = {}
        for r in R:
            cnt[r["v"][v]["top_str"]] = cnt.get(r["v"][v]["top_str"], 0) + 1
        d["most_common_first_tokens"] = sorted(cnt.items(), key=lambda kv: -kv[1])[:6]
        S["variants"][v] = d
    P = [r for r in rows if r.get("p0chat")]
    if P:
        T = [trials[r["i"]] for r in P]
        em_chat = np.array([bool(r["p0chat"]["em"]) for r in P])
        em_raw = np.array([t["em"] for t in T])
        same = np.array([r["p0chat"]["answer"] is not None and r["p0chat"]["answer"].strip().lower() == t["answer"].strip().lower() for r, t in zip(P, T)])
        c0 = np.array([-1 if r["p0chat"]["conf"] is None else r["p0chat"]["conf"] for r in P])
        S["p0chat"] = {"n": len(P), "parsed_pct": float(100 * np.mean([r["p0chat"]["answer"] is not None for r in P])),
                       "alias_accuracy_chat_pct": float(100 * em_chat.mean()), "alias_accuracy_raw_pct_same_questions": float(100 * em_raw.mean()),
                       "same_answer_as_raw_pct": float(100 * same.mean()),
                       "phase0_conf_share_pct": (100 * np.bincount(c0[c0 >= 0], minlength=10) / max(1, (c0 >= 0).sum())).round(1).tolist(),
                       "examples": [r["p0chat"]["text"][:120] for r in P[:6]], "phase1": {}}
        for v in ("raw_nn", "chat_prefill_split"):
            Q = [(r, t) for r, t in zip(P, T) if v in r["p0chat"].get("p1", {})]
            lg = np.array([r["p0chat"]["p1"][v]["sp"] for r, _ in Q])
            share = np.bincount(lg.argmax(1), minlength=10) / len(Q)
            S["p0chat"]["phase1"][v] = {"n": len(Q), "share_pct": (100 * share).round(1).tolist(),
                                        "tv_distance_to_paper": float(0.5 * np.abs(share - np.array(PAPER_COUNTS) / sum(PAPER_COUNTS)).sum())}
    json.dump(S, open(OUT / "summary.json", "w"), indent=1)
    print(f"{len(rows)} trials. Paper Fig. 9B shares (%): {S['paper_share_pct']}; balanced baseline 0.553; AUROC 0.71; ECE 0.12")
    print(f"{'variant':<20s} {'1st tok class':>13s} {'TV to paper':>11s} {'NoCh':>5s} {'RUnl':>5s} {'Slight':>6s} {'Likely':>6s} {'HighL':>6s} {'AlmC':>5s} {'baseline':>8s} {'AUROC':>6s} {'ECE':>5s} {'r(P0)':>6s} {'r(lp)':>6s} {'margin':>6s} {'ld top3':>7s}")
    for v, d in S["variants"].items():
        s = d["share_pct"]
        f = lambda x, w, p=2: f"{x:>{w}.{p}f}" if x is not None else " " * (w - 1) + "-"
        print(f"{v:<20s} {d['first_token_is_class_pct']:>12.1f}% {d['tv_distance_to_paper']:>11.3f} {s[0]:>5.1f} {s[1]:>5.1f} {s[2]:>6.1f} {s[6]:>6.1f} {s[8]:>6.1f} {s[9]:>5.1f} "
              f"{f(d['balanced_baseline'], 8, 3)} {f(d['auroc'], 6)} {f(d['ece'], 5)} {f(d['r_phase0_conf'], 6)} {f(d['r_mean_logprob'], 6)} {f(d['median_top2_margin'], 6)} {f(d['mean_logit_diff_top3'], 7, 1)}")
    for v, d in S["variants"].items():
        a, b = d["accuracy_by_class"], d["accuracy_by_class_alias_only"]
        print(f"  accuracy by class, {v:<20s}: " + ", ".join(f"{k} {a[k]:.2f} ({b[k]:.2f})" for k in a) + f" | first tokens {d['most_common_first_tokens'][:4]}")
    print("  paper accuracy by class:", S["paper_accuracy_by_class"])
    if "p0chat" in S:
        print("p0chat:", json.dumps({k: v for k, v in S["p0chat"].items() if k != "examples"}))
        for e in S["p0chat"]["examples"]:
            print("   ", repr(e))
    sys.exit(0)

# ------------------------------------------------------------------------------------------------- model part
import torch  # noqa: E402

from vc.model import VC, batched, log  # noqa: E402
from vc.prompts import class_token_ids, encode  # noqa: E402

vc = VC()
tok = vc.tok
cls_sp = class_token_ids(tok)
cls_ns = [tok(c, add_special_tokens=False).input_ids[0] for c in CLASSES]
assert len(set(cls_ns)) == 10
SEL = cls_sp + cls_ns
trials = read_jsonl(DATA / "trials.jsonl")
pick = sorted(np.random.default_rng(0).choice(len(trials), args.n, replace=False).tolist())
mine = pick[args.shard :: args.nshards]
p0_set = set(pick[: args.n_p0][args.shard :: args.nshards])
HEAD, TAIL = PROMPT_KINDS["cat"]
CCS = "**Confidence**:"
assert TAIL.endswith("\n\n" + CCS)
BOS = tok.bos_token


def full_text(q, a, panl="\n\n"):
    return f"{HEAD}{q.strip()}\n**Answer**: {a.strip()}{panl}{TAIL}"


def chat(user, prefill=""):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True) + prefill


def ids_of(text):
    ids = tok(text, add_special_tokens=False).input_ids
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id, tok.convert_ids_to_tokens(ids[:3])
    return ids


BUILD = {
    "raw_nn": lambda q, a: encode(tok, q, a, "cat", panl="\n\n").ids,
    "raw_nnn": lambda q, a: encode(tok, q, a, "cat", panl="\n\n\n").ids,
    "raw_n": lambda q, a: encode(tok, q, a, "cat", panl="\n").ids,
    "raw_nobos": lambda q, a: encode(tok, q, a, "cat", panl="\n\n").ids[1:],
    "chat_open": lambda q, a: ids_of(f"{BOS}<start_of_turn>user\n{full_text(q, a)}"),
    "chat_prefill_full": lambda q, a: ids_of(chat(full_text(q, a), CCS)),
    "chat_prefill_split": lambda q, a: ids_of(chat(full_text(q, a)[: -len("\n\n" + CCS)], CCS)),
    "chat_user": lambda q, a: ids_of(chat(full_text(q, a))),
}
log("example chat_prefill_split tail:", repr(tok.decode(BUILD["chat_prefill_split"]("Q?", "A")[-14:])))
log("example chat_user tail:", repr(tok.decode(BUILD["chat_user"]("Q?", "A")[-8:])))


def score(items, variants):
    """items: list of (key, question, answer). Returns {key: {variant: {...}}}."""
    out = {k: {} for k, _, _ in items}
    for v in variants:
        enc = {}
        for k, q, a in items:
            try:
                enc[k] = BUILD[v](q, a)
            except AssertionError:
                pass
        order = sorted(enc, key=lambda k: len(enc[k]))
        for b in batched(order, args.bs):
            ids, last = vc.pad([enc[k] for k in b])
            lg, _, top = vc.run(vc.embed(ids), last_idx=last, sel=SEL)
            for j, k in enumerate(b):
                t = int(top[j])
                out[k][v] = {"sp": [round(float(x), 4) for x in lg[j, :10]], "ns": [round(float(x), 4) for x in lg[j, 10:]],
                             "top": t, "top_str": tok.decode(t), "top_sp": t in cls_sp, "top_ns": t in cls_ns}
        log(f"{v}: {len(enc)}/{len(items)} trials scored")
    return out


res = score([(i, trials[i]["question"], trials[i]["answer"]) for i in mine], P1_VARIANTS)
rows = [{"i": i, "v": res[i]} for i in mine]

# ---- Phase 0 through the chat template: do the answers (and hence Phase 1) change?
p0_idx = [i for i in mine if i in p0_set]


def stop_fn(txt):
    if CCS not in txt:
        return txt.count("\n") >= 6
    tail = txt.split(CCS, 1)[1]
    return any(c in tail for c in CLASSES) or ("\n" in tail.strip(" ") and len(tail.strip()) > 0)


by_i = {r["i"]: r for r in rows}
for b in batched(sorted(p0_idx, key=lambda i: len(trials[i]["question"])), args.bs):
    prompts = [chat(PHASE0_HEAD + f"Question: {trials[i]['question'].strip()}")[len(BOS):] for i in b]
    for i, (ids, _) in zip(b, vc.generate(prompts, max_new_tokens=48, stop_fn=stop_fn)):
        txt = tok.decode(ids)
        ans, conf = None, None
        if "**Answer**:" in txt:
            ans = txt.split("**Answer**:", 1)[1].split("\n", 1)[0].strip() or None
        if CCS in txt:
            tail = txt.split(CCS, 1)[1].strip()
            conf = next((c for c in sorted(range(10), key=lambda c: -len(CLASSES[c])) if tail.startswith(CLASSES[c])), None)
        by_i[i]["p0chat"] = {"text": txt, "answer": ans, "conf": conf, "em": bool(ans) and alias_match(ans, trials[i]["norm_aliases"])}
log(f"p0chat: {len(p0_idx)} generations")
res2 = score([(i, trials[i]["question"], by_i[i]["p0chat"]["answer"]) for i in p0_idx if by_i[i]["p0chat"]["answer"]], ["raw_nn", "chat_prefill_split"])
for i, d in res2.items():
    by_i[i]["p0chat"]["p1"] = d

os.makedirs(OUT, exist_ok=True)
write_jsonl(OUT / f"shard{args.shard}.jsonl", rows)
log("PILOT_FORMAT DONE", len(rows))
