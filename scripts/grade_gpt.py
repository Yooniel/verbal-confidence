"""Grade the Phase-0 answers with GPT-4o-mini, the grader the paper used, through OpenRouter.

Reads results/behav/shard*.jsonl and appends one row per graded answer to results/behav/gpt4omini.jsonl
({"qid", "verdict", "text", "prompt_tokens", "completion_tokens"}); already graded answers are skipped, so the
script can be re-run. The API key is read from ~/.env (OPENROUTER_KEY) inside the script. Only the question, the
TriviaQA reference answers and the model's answer are sent.

With VC_FMT=tt the answers of the two-turn run (results/tt/behav) are graded. Answers that equal the main run's
answer after normalisation keep that run's verdict and are not sent again. `--dir behav_sup --only-low` grades
the low-confidence trials of the extra train-split questions. `--dry-run` only prints how many answers would be
graded and what that would cost.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.data import normalize_answer, read_jsonl
from vc.env import DATA_RAW, FMT, RESULTS, load_env
from vc.prompts import LOW_CLASSES, judge_message

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai/gpt-4o-mini")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--budget", type=float, default=1.0, help="stop when the estimated spend (USD) reaches this")
ap.add_argument("--price-in", type=float, default=0.15, help="USD per million prompt tokens")
ap.add_argument("--price-out", type=float, default=0.60, help="USD per million completion tokens")
ap.add_argument("--dry-run", action="store_true", help="print the number of answers to grade and the estimated cost, then stop")
ap.add_argument("--dir", default="behav", help="sub-directory of the results folder that holds the behavioural shards")
ap.add_argument("--only-low", action="store_true", help="grade only trials whose confidence class is one of the bottom three")
args = ap.parse_args()

OUT = RESULTS / args.dir / "gpt4omini.jsonl"
rows = [r for f in sorted(glob(str(RESULTS / args.dir / "shard*.jsonl"))) for r in read_jsonl(f)]
done = {r["qid"] for r in read_jsonl(OUT)} if OUT.exists() else set()
todo = [r for r in rows if r.get("answer") and not r.get("truncated") and r["qid"] not in done]
if args.only_low:
    todo = [r for r in todo if "p1_logits" in r and max(range(10), key=lambda c: r["p1_logits"][c]) in LOW_CLASSES]
if FMT != "raw":
    raw = {r["qid"]: r for r in read_jsonl(DATA_RAW / "trials.jsonl")}

    def reusable(r):
        m = raw.get(r["qid"])
        return m is not None and m["gpt"] is not None and normalize_answer(r["answer"]) == normalize_answer(m["answer"])

    n0 = len(todo)
    todo = [r for r in todo if not reusable(r)]
    print(f"{n0 - len(todo)} answers equal the main run's answer and keep its verdict", flush=True)
if args.limit:
    todo = todo[: args.limit]
est = len(todo) * (140 * args.price_in + 1 * args.price_out) / 1e6
print(f"{len(rows)} rows, {len(done)} already graded, {len(todo)} to grade with {args.model} (estimated cost ${est:.3f})", flush=True)
if args.dry_run:
    sys.exit(0)

load_env()
KEY = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
assert KEY, "OPENROUTER_KEY not found in the environment or ~/.env"

lock = threading.Lock()
spent = {"in": 0, "out": 0, "n": 0, "fail": 0}
stop = threading.Event()


def cost():
    return spent["in"] * args.price_in / 1e6 + spent["out"] * args.price_out / 1e6


def call(msg):
    body = json.dumps({"model": args.model, "messages": [{"role": "user", "content": msg}], "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.loads(resp.read())
            if "choices" in d:
                return d
            err = str(d.get("error"))[:200]
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code in (401, 402, 403):  # bad key or no credit: retrying will not help
                raise
        except Exception as e:  # timeouts, connection resets
            err = type(e).__name__
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"giving up after retries: {err}")


def grade(r):
    if stop.is_set():
        return
    try:
        d = call(judge_message(r))
    except Exception as e:
        with lock:
            spent["fail"] += 1
            if spent["fail"] <= 5:
                print("failed:", r["qid"], str(e)[:120], flush=True)
        return
    text = (d["choices"][0]["message"].get("content") or "").strip()
    word = text.lower().lstrip("*\"' ")
    verdict = True if word.startswith("yes") else False if word.startswith("no") else None
    u = d.get("usage") or {}
    with lock:
        spent["in"] += u.get("prompt_tokens", 0)
        spent["out"] += u.get("completion_tokens", 0)
        spent["n"] += 1
        with open(OUT, "a") as f:
            f.write(json.dumps({"qid": r["qid"], "verdict": verdict, "text": text[:40],
                                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}) + "\n")
        if spent["n"] % 500 == 0:
            print(f"{spent['n']} graded, estimated spend ${cost():.3f}", flush=True)
        if cost() >= args.budget:
            stop.set()


t0 = time.time()
with ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(grade, todo))
allr = read_jsonl(OUT) if OUT.exists() else []
yes = sum(1 for r in allr if r["verdict"] is True)
unk = sum(1 for r in allr if r["verdict"] is None)
print(f"this run: {spent['n']} graded, {spent['fail']} failed, {spent['in']} prompt + {spent['out']} completion tokens, "
      f"estimated spend ${cost():.3f}, {time.time()-t0:.0f}s" + (" | STOPPED AT BUDGET" if stop.is_set() else ""), flush=True)
print(f"total graded: {len(allr)} | Yes {yes} ({100*yes/max(1,len(allr)):.1f}%) | unparsed {unk}", flush=True)
print("GRADE DONE", flush=True)
