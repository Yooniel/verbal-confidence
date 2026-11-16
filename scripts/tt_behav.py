"""Behavioural stage of the two-turn chat format (one shard; run with VC_FMT=tt).

For every question in the shard:
  turn 1 : the question is a user turn and the model answers in its own turn (greedy); the generated answer
           tokens and their log-probs are kept. Whitespace tokens that the model emits around the answer are
           dropped, as the chat template does when a turn becomes part of the history.
  judge  : Gemma-as-grader verdict on the answer (fallback grader, same prompt as scripts/behav.py)
  turn 2 : the confidence request is appended as a new user turn (vc.prompts.encode_tt) -> logits of the 10
           class first-tokens at the last prompt token, i.e. for the first token of the model's reply

`--split train --n N --no-judge --out results/tt/behav_sup` runs N questions of the TriviaQA train split instead:
a reservoir of extra low-confidence trials (the model rarely states low confidence in this format).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import alias_match, load_triviaqa, write_jsonl
from vc.env import FMT, RESULTS
from vc.model import VC, batched, log
from vc.prompts import CLASSES, EOT, TT_SUFFIX, class_token_ids, encode_tt, judge_message, tt_prefix

assert FMT == "tt", "run with VC_FMT=tt"
ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max-new", type=int, default=40)
ap.add_argument("--out", default=str(RESULTS / "behav"))
ap.add_argument("--split", default="validation", help="TriviaQA split (train: reservoir of extra questions)")
ap.add_argument("--n", type=int, default=0, help="use only n questions of the shuffled split")
ap.add_argument("--offset", type=int, default=0, help="... starting at this position")
ap.add_argument("--no-judge", action="store_true", help="skip the Gemma-as-grader pass")
args = ap.parse_args()

vc = VC()
tok = vc.tok
cls_ids = class_token_ids(tok)
rows = load_triviaqa(split=args.split)
if args.n or args.limit:
    rows = rows[args.offset : args.offset + (args.n or args.limit)]
rows = rows[args.shard :: args.nshards]
log(f"shard {args.shard}/{args.nshards}: {len(rows)} questions")
BOS = tok.bos_token

# ------------------------------------------------------------------ turn 1: the model's own answer
t0 = time.time()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["question"]))
for bi, b in enumerate(batched(order, args.bs)):
    prompts = [tt_prefix(rows[i]["question"])[len(BOS):] for i in b]  # generate() adds <bos> itself
    gens = vc.generate(prompts, max_new_tokens=args.max_new, stop_fn=lambda txt: EOT in txt)
    for i, (ids, lps) in zip(b, gens):
        r = rows[i]
        r["turn"] = tok.decode(ids)
        r["truncated"] = len(ids) >= args.max_new  # no <end_of_turn> within the token budget
        k0, k1 = 0, len(ids)  # drop whitespace-only tokens at both ends
        while k0 < k1 and not tok.decode(ids[k0]).strip():
            k0 += 1
        while k1 > k0 and not tok.decode(ids[k1 - 1]).strip():
            k1 -= 1
        r["answer"] = r["turn"].strip()
        r["a_ids"] = ids[k0:k1]
        r["p0_ans_lp"] = lps[k0:k1]
        r["n_ws_tokens"] = len(ids) - (k1 - k0)
        r["em"] = alias_match(r["answer"], r["norm_aliases"])
    if bi % 20 == 0:
        log(f"turn 1 batch {bi}: {time.time()-t0:.0f}s")
log(f"turn 1 done in {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ judge (Gemma as grader)
yes_id, no_id = tok("Yes", add_special_tokens=False).input_ids[0], tok("No", add_special_tokens=False).input_ids[0]


def judge_ids(r):
    text = tok.apply_chat_template([{"role": "user", "content": judge_message(r)}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


t0 = time.time()
jid = {} if args.no_judge else {i: judge_ids(rows[i]) for i in range(len(rows))}
for b in batched(sorted(jid, key=lambda i: len(jid[i])), 64):
    ids, last = vc.pad([jid[i] for i in b])
    lg, _, _ = vc.run(vc.embed(ids), last_idx=last)
    for j, i in enumerate(b):
        rows[i]["judge_margin"] = float(lg[j, yes_id] - lg[j, no_id])
        rows[i]["judge_top_is_yn"] = int(lg[j].argmax()) in (yes_id, no_id)
log(f"judge done in {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ turn 2: confidence request in a new user turn
t0 = time.time()
encs = {}
for i, r in enumerate(rows):
    if not r["answer"] or r["truncated"]:
        continue
    try:
        encs[i] = encode_tt(tok, r["question"], r["answer"], r["a_ids"])
    except AssertionError as e:  # answers that do not tokenise back to the generated tokens
        r["p1_err"] = str(e)[:200]
for b in batched(sorted(encs, key=lambda i: encs[i].n), args.bs):
    ids, last = vc.pad([encs[i].ids for i in b])
    lg, _, top = vc.run(vc.embed(ids), last_idx=last, sel=cls_ids)
    for j, i in enumerate(b):
        r, e = rows[i], encs[i]
        r["p1_logits"] = [round(float(x), 4) for x in lg[j]]
        r["p1_top"] = int(top[j])
        r["p1_top_valid"] = int(top[j]) in cls_ids
        r["n_tok"], r["n_q"], r["n_a"] = e.n, e.q_span[1] - e.q_span[0], e.a_span[1] - e.a_span[0]
log(f"turn 2 done in {time.time()-t0:.0f}s ({len(encs)} trials)")

# a few complete replies, to confirm that the first token is followed by the rest of the class name
t0 = time.time()
chk = sorted(encs)[:64]
n_ok = 0
for b in batched(chk, 32):
    texts = [tt_prefix(rows[i]["question"])[len(BOS):] + rows[i]["answer"] + TT_SUFFIX for i in b]
    gens = vc.generate(texts, max_new_tokens=8, stop_fn=lambda txt: EOT in txt)
    for i, (ids, _) in zip(b, gens):
        rows[i]["reply"] = tok.decode(ids)
        n_ok += rows[i]["reply"].strip() == CLASSES[max(range(10), key=lambda c: rows[i]["p1_logits"][c])]
log(f"reply check: {n_ok}/{len(chk)} generated replies are exactly the class read from the logits ({time.time()-t0:.0f}s)")

os.makedirs(args.out, exist_ok=True)
write_jsonl(os.path.join(args.out, f"shard{args.shard}.jsonl"), rows)
log("TT_BEHAV DONE", len(rows))
