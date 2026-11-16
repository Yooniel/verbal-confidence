"""Behavioural stage (one shard): Phase 0 answers + log-probs, grading, Phase 1 confidence.

For every question in the shard:
  phase 0 : generate "<answer>\n**Confidence**: <class>" (greedy); keep answer-token log-probs
  judge   : Gemma-as-grader verdict on answer vs. TriviaQA reference aliases (stand-in for GPT-4o-mini)
  phase 1 : categorical confidence prompt (Fig. 8)  -> logits of the 10 class first-tokens at CC
  minimal : minimal numeric 0-9 prompt (Fig. 16)    -> logits of the 10 digits at CC
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import alias_match, load_triviaqa, write_jsonl
from vc.env import RESULTS
from vc.model import VC, batched, log
from vc.prompts import CLASSES, class_token_ids, digit_token_ids, encode, judge_message, phase0_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", default=str(RESULTS / "behav"))
args = ap.parse_args()

vc = VC()
tok = vc.tok
cls_ids, dig_ids = class_token_ids(tok), digit_token_ids(tok)
rows = load_triviaqa()
if args.limit:
    rows = rows[: args.limit]
rows = rows[args.shard :: args.nshards]
log(f"shard {args.shard}/{args.nshards}: {len(rows)} questions")


def stop_fn(txt: str) -> bool:
    if "**Confidence**:" not in txt:
        return txt.count("\n") >= 3
    tail = txt.split("**Confidence**:", 1)[1]
    return any(c in tail for c in CLASSES) or ("\n" in tail.strip(" ") and len(tail.strip()) > 0)


# ------------------------------------------------------------------ phase 0
t0 = time.time()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["question"]))
for bi, b in enumerate(batched(order, args.bs)):
    gens = vc.generate([phase0_prompt(rows[i]["question"]) for i in b], max_new_tokens=40, stop_fn=stop_fn)
    for i, (ids, lps) in zip(b, gens):
        r = rows[i]
        txt = tok.decode(ids)
        # answer span = generated tokens before the first token containing a newline
        n_ans = next((k for k, t in enumerate(ids) if "\n" in tok.decode(t)), len(ids))
        r["p0_text"] = txt
        r["answer"] = tok.decode(ids[:n_ans]).strip()
        r["p0_ans_ids"] = ids[:n_ans]
        r["p0_ans_lp"] = lps[:n_ans]
        conf = None
        if "**Confidence**:" in txt:
            tail = txt.split("**Confidence**:", 1)[1].strip()
            for ci in sorted(range(10), key=lambda c: -len(CLASSES[c])):
                if tail.startswith(CLASSES[ci]):
                    conf = ci
                    break
        r["p0_conf"] = conf
        r["em"] = alias_match(r["answer"], r["norm_aliases"])
    if bi % 20 == 0:
        log(f"phase0 batch {bi}: {time.time()-t0:.0f}s")
log(f"phase0 done in {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ judge
yes_id, no_id = tok("Yes", add_special_tokens=False).input_ids[0], tok("No", add_special_tokens=False).input_ids[0]


def judge_ids(r):
    msg = judge_message(r)
    text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


t0 = time.time()
jid = {i: judge_ids(rows[i]) for i in range(len(rows))}
assert jid[0][0] == tok.bos_token_id and jid[0][1] != tok.bos_token_id
order = sorted(range(len(rows)), key=lambda i: len(jid[i]))
for b in batched(order, 64):
    ids, last = vc.pad([jid[i] for i in b])
    lg, _, _ = vc.run(vc.embed(ids), last_idx=last)
    for j, i in enumerate(b):
        rows[i]["judge_margin"] = float(lg[j, yes_id] - lg[j, no_id])
        rows[i]["judge_top_is_yn"] = int(lg[j].argmax()) in (yes_id, no_id)
log(f"judge done in {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ phase 1 (categorical) + minimal numeric
# minimal-prompt variants: separator after the answer ("\n\n" vs "\n") x prompt ending (":" vs ":'", see prompts.py)
KINDS = [
    ("cat", "p1", cls_ids, "\n\n"),
    ("min", "pm", dig_ids, "\n\n"),
    ("minq", "pmq", dig_ids, "\n\n"),
    ("min", "pm1", dig_ids, "\n"),
    ("minq", "pmq1", dig_ids, "\n"),
]
for kind, key, tids, panl in KINDS:
    t0 = time.time()
    idx = [i for i in range(len(rows)) if rows[i]["answer"]]
    encs = {}
    for i in idx:
        try:
            encs[i] = encode(tok, rows[i]["question"], rows[i]["answer"], kind, panl=panl)
        except AssertionError as e:  # answers that do not round-trip through the tokenizer cleanly
            rows[i][f"{key}_err"] = str(e)[:200]
    idx = sorted(encs, key=lambda i: encs[i].n)
    for b in batched(idx, 64 if kind != "cat" else args.bs):
        ids, last = vc.pad([encs[i].ids for i in b])
        lg, _, top = vc.run(vc.embed(ids), last_idx=last, sel=tids)
        for j, i in enumerate(b):
            r = rows[i]
            r[f"{key}_logits"] = [round(float(x), 4) for x in lg[j]]
            r[f"{key}_top"] = int(top[j])
            r[f"{key}_top_valid"] = int(top[j]) in tids
            if kind == "cat":
                e = encs[i]
                r["n_tok"], r["n_q"], r["n_a"] = e.n, e.q_span[1] - e.q_span[0], e.a_span[1] - e.a_span[0]
    log(f"{key} ({kind}) prompt done in {time.time()-t0:.0f}s ({len(idx)} trials)")

os.makedirs(args.out, exist_ok=True)
write_jsonl(os.path.join(args.out, f"shard{args.shard}.jsonl"), rows)
log("BEHAV DONE", len(rows))
