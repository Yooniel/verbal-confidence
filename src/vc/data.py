"""TriviaQA loading, answer normalisation / alias matching, and small IO helpers."""
from __future__ import annotations

import json
import re
import string
from pathlib import Path

import numpy as np


def load_triviaqa(seed: int = 0, split: str = "validation") -> list[dict]:
    """De-duplicated TriviaQA (rc.nocontext): unique questions in a fixed shuffled order. The validation split
    (9,960 unique questions) is the data set of all experiments; the train split is only used as a reservoir of
    extra low-confidence trials for the two-turn format."""
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    seen, rows = set(), []
    for q, qid, ans in zip(ds["question"], ds["question_id"], ds["answer"]):
        if qid in seen:
            continue
        seen.add(qid)
        rows.append(
            {
                "qid": qid,
                "question": " ".join(q.split()),
                "aliases": list(ans["aliases"]),
                "norm_aliases": list(ans["normalized_aliases"]),
                "gold": ans["value"],
            }
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNC = set(string.punctuation + "‘’´`")


def normalize_answer(s: str) -> str:
    """Normalisation from the official TriviaQA evaluation script."""
    s = s.replace("_", " ").lower()
    s = "".join(ch if ch not in _PUNC else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split()).strip()


def alias_match(answer: str, norm_aliases: list[str]) -> bool:
    """Exact match of the normalised answer against any normalised alias (TriviaQA EM)."""
    a = normalize_answer(answer)
    return bool(a) and a in set(norm_aliases)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# Layers sampled in the paper (C.1.3): 0, 10, 15, 20-35 densely, 40, 50, 61
SWEEP_LAYERS = [0, 10, 15] + list(range(20, 36)) + [40, 50, 61]
