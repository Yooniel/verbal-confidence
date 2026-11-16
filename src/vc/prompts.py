"""Prompts from the paper (Figs. 8, 15, 16) and token-position bookkeeping.

All prompts are fed to the model as raw text (BOS + prompt), so that the confidence-colon (CC) token is the
literal last token of the prompt whose final-layer residual produces the confidence token (paper Fig. 1D),
and the answer-colon (AC) token is the last token of the Phase-0 prompt (paper Sec. 2.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .env import FMT

CLASSES = [
    "No chance",
    "Really unlikely",
    "Chances are slight",
    "Unlikely",
    "Less than even",
    "Better than even",
    "Likely",
    "Very good chance",
    "Highly likely",
    "Almost certain",
]
MIDPOINTS = [0.05 + 0.1 * i for i in range(10)]
HIGH_CLASSES = [7, 8, 9]  # "Very good chance", "Highly likely", "Almost certain"
LOW_CLASSES = [0, 1, 2]  # "No chance", "Really unlikely", "Chances are slight"

_CLASS_LINES = "\n".join(f'- "{c}" ({0.1 * i:.1f}–{0.1 * (i + 1):.1f})' for i, c in enumerate(CLASSES))

# Separator that follows the answer; its (single) token is the post-answer-newline (PANL) token.
# Gemma tokenizes "\n" (107), "\n\n" (108) and "\n\n\n" (109) as single tokens.
PANL_DEFAULT = "\n\n"

# ---------------------------------------------------------------------------------------------------------
# Phase 0: answer generation. "Essentially identical to Fig. 8, except that confidence instructions
# appeared at the start" (paper C.1.2). The prompt ends at the answer colon (AC); the model generates
# " <answer>\n**Confidence**: <class>".
# ---------------------------------------------------------------------------------------------------------
PHASE0_HEAD = (
    "You will be shown a question. Provide your answer, then classify your confidence into one of the "
    "following classes based on how\n"
    "likely your answer is to be correct (NO REASONING OR EXPLANATION):\n"
    "\n"
    f"{_CLASS_LINES}\n"
    "\n"
    "Each category reflects the probability that your answer is correct.\n"
    "\n"
    "At the very end of your output, format your answer and confidence as\n"
    "**Answer**: $ANSWER\n"
    "**Confidence**: $CLASS\n"
    "where ANSWER is your answer, stated as concisely as possible, and CLASS is one of the names "
    "(only the names without the probability ranges)\n"
    "of the classes above.\n"
    "\n"
)


def phase0_prompt(question: str) -> str:
    return f"{PHASE0_HEAD}Question: {question}\n**Answer**:"


# ---------------------------------------------------------------------------------------------------------
# Phase 1: categorical confidence prompt (paper Fig. 8).
# ---------------------------------------------------------------------------------------------------------
CAT_HEAD = "You will be shown a question, together with an answer you previously provided before.\n\nQuestion: "
CAT_TAIL = (
    "Classify your confidence into one of the following classes based on how\n"
    "likely the answer above is to be correct (NO REASONING OR EXPLANATION):\n"
    "\n"
    f"{_CLASS_LINES}\n"
    "\n"
    "Each category reflects the probability that the answer is correct.\n"
    "\n"
    "At the very end of your output, format your confidence as\n"
    "**Confidence**: $CLASS\n"
    "where CLASS is one of the names (only the names without the probability ranges)\n"
    "of the classes above.\n"
    "\n"
    "**Confidence**:"
)

# Minimal numeric (0-9) prompt used for attention blocking (paper Fig. 16).
MIN_HEAD = "You will be shown a question, together with an answer you previously provided before.\nQuestion: "
MIN_TAIL = "**State Confidence(0-9) with NO SPACE**:"

# Full numeric (0-100) prompt (paper Fig. 15).
NUM_HEAD = MIN_HEAD
NUM_TAIL = (
    "State your confidence as an integer between 0 and 100 based\n"
    "on how likely your answer is to be correct.\n"
    "That is, if your confidence is 0, that means that your\n"
    "answer has almost no chance of being correct.\n"
    "If your confidence is 100, then you are almost certain that your\n"
    "answer is correct.\n"
    "\n"
    "At the very end of your output, format your confidence as:\n"
    "**Confidence**:$CONFIDENCE\n"
    "where CONFIDENCE is an integer between 0 and 100.\n"
    "do NOT leave a space before the first digit or you will be\n"
    "scored WRONG!\n"
    "**Confidence**:"
)

PROMPT_KINDS = {
    "cat": (CAT_HEAD, CAT_TAIL),
    "min": (MIN_HEAD, MIN_TAIL),
    # Fig. 16 literally ends with  **:'  (a quote after the colon). Gemma tokenizes  **:'  as  "**" + ":'",
    # so the last prompt token still carries the colon and the model continues with the digit itself.
    "minq": (MIN_HEAD, MIN_TAIL + "'"),
    "num": (NUM_HEAD, NUM_TAIL),
}


# ---------------------------------------------------------------------------------------------------------
# Grading prompt (the paper marks answers with GPT-4o-mini but does not give its prompt).
# ---------------------------------------------------------------------------------------------------------
JUDGE = (
    "You are grading an answer to a trivia question.\n\n"
    "Question: {q}\n"
    "Accepted reference answers: {refs}\n"
    "Submitted answer: {a}\n\n"
    "Is the submitted answer correct, i.e. does it refer to the same thing as one of the accepted reference "
    "answers? Ignore differences in capitalisation, spelling variants, abbreviation or extra detail. "
    "Reply with a single word: Yes or No."
)


def judge_message(r: dict) -> str:
    """Grading prompt for one trial (question, up to 12 distinct reference answers, the model's answer)."""
    refs, seen = [], set()
    for a in [r["gold"]] + r["aliases"]:
        if a.lower() not in seen:
            seen.add(a.lower())
            refs.append(a)
    return JUDGE.format(q=r["question"], refs=" | ".join(refs[:12]), a=r["answer"])


@dataclass
class Encoded:
    """A tokenized confidence prompt plus the token indices of the positions analysed in the paper."""

    ids: list[int]
    pos: dict[str, int] = field(default_factory=dict)  # named single positions
    q_span: tuple[int, int] = (0, 0)  # [start, end) of question tokens
    a_span: tuple[int, int] = (0, 0)  # [start, end) of answer tokens
    inter: list[int] = field(default_factory=list)  # first token of each class name in the instruction list
    groups: dict[str, list[int]] = field(default_factory=dict)  # named groups of positions (two-turn format)

    @property
    def n(self) -> int:
        return len(self.ids)


def _tok_span(offsets, c0: int, c1: int) -> tuple[int, int]:
    """Token index range [t0, t1) of tokens overlapping the character range [c0, c1)."""
    idx = [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1 and e > s]
    return idx[0], idx[-1] + 1


def encode(tok, question: str, answer: str, kind: str = "cat", panl: str = PANL_DEFAULT) -> Encoded:
    """Tokenize a confidence prompt and locate PANL, PANL+1, AC, answer/question tokens, FCC, CC, etc."""
    head, tail = PROMPT_KINDS[kind]
    question, answer = question.strip(), answer.strip()
    pre = f"{head}{question}\n**Answer**:"
    text = f"{pre} {answer}{panl}{tail}"
    enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    assert ids[0] == tok.bos_token_id

    q0 = len(head)
    q_span = _tok_span(offs, q0, q0 + len(question))
    a0 = len(pre)  # the answer span includes its leading space
    a_span = _tok_span(offs, a0, a0 + 1 + len(answer))
    panl_idx = a_span[1]
    ac_idx = a_span[0] - 1
    e = Encoded(ids=ids, q_span=q_span, a_span=a_span)
    e.pos = {
        "PANL": panl_idx,
        "PANL+1": panl_idx + 1,
        "AC": ac_idx,
        "A_first": a_span[0],
        "A_last": a_span[1] - 1,
        "QTT": q_span[0] + 2,  # third question token (decoding control)
        "CC": len(ids) - 1,
    }
    # sanity checks on the structure we rely on
    assert tok.decode(ids[panl_idx]) == panl, (tok.decode(ids[panl_idx]), panl)
    assert tok.decode(ids[ac_idx]) == "**:", tok.decode(ids[ac_idx])
    assert tok.decode(ids[-1]) in ("**:", ":'"), tok.decode(ids[-1])
    assert tok.decode(ids[a_span[0] : a_span[1]]).strip() == answer, (tok.decode(ids[a_span[0] : a_span[1]]), answer)

    t0 = len(pre) + 1 + len(answer) + len(panl)  # char offset where the tail starts
    if kind == "cat":
        c = text.index("**Confidence**: $CLASS", t0)
        fcc = _tok_span(offs, c, c + len("**Confidence**:"))[1] - 1
        assert tok.decode(ids[fcc]) == "**:"
        e.pos["FCC"] = fcc
        for name in CLASSES:
            c = text.index(f'- "{name}"', t0) + 3
            e.inter.append(_tok_span(offs, c, c + len(name))[0])
    elif kind == "num":
        c = text.index("**Confidence**:$CONFIDENCE", t0)
        fcc = _tok_span(offs, c, c + len("**Confidence**:"))[1] - 1
        e.pos["FCC"] = fcc
    e.groups = {"INSTR": list(range(panl_idx + 1, len(ids) - 1))}  # every token between PANL and CC
    return e


# ---------------------------------------------------------------------------------------------------------
# Two-turn chat format (not in the paper; VC_FMT=tt). The question is a user turn, the answer is the model's
# own turn, and the confidence request arrives in a second user turn. Nothing is pre-filled: the class is
# read from the next-token logits at the end of the generation prompt, i.e. it is the first token of the
# model's second turn. The request keeps the wording of the paper's Fig. 8 except for its last sentence,
# which there asks for a "**Confidence**: $CLASS" line.
#
#   <bos><start_of_turn>user\nQuestion: {question}\n{TT_Q_TAIL}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>\n
#   <start_of_turn>user\n{TT_REQ}<end_of_turn>\n<start_of_turn>model\n
#
# The wording of the question turn was chosen in a pilot (scripts/tt_pilot.py --wording): with it the model
# answers with a bare name or phrase, as in the paper's format; most other wordings make it end the answer
# with a full stop and blank lines. The confidence classes it then states are distributed alike for all of them.
# ---------------------------------------------------------------------------------------------------------
SOT, EOT = "<start_of_turn>", "<end_of_turn>"
TT_Q_HEAD = "Question: "
TT_Q_TAIL = "\nAnswer with just the answer itself (a name, number or short phrase), nothing else."
TT_REQ = (
    "Classify your confidence into one of the following classes based on how\n"
    "likely the answer above is to be correct (NO REASONING OR EXPLANATION):\n"
    "\n"
    f"{_CLASS_LINES}\n"
    "\n"
    "Each category reflects the probability that the answer is correct.\n"
    "\n"
    "Reply with only the name of the class (only the name without the probability range)."
)
TT_SUFFIX = f"{EOT}\n{SOT}user\n{TT_REQ}{EOT}\n{SOT}model\n"
# Names of the tokens between the answer and the request, and of the tokens after the request. To reuse the
# experiment code, the first token after the answer keeps the name PANL and the last prompt token the name CC.
TT_POST5 = ["PANL", "PANL+1", "SOT", "USER", "UNL"]  # <end_of_turn>  \n  <start_of_turn>  user  \n
TT_TAIL4 = ["EOT2", "CC-3", "CC-2", "CC-1"]  # <end_of_turn>  \n  <start_of_turn>  model   (then CC = \n)


def tt_prefix(question: str) -> str:
    """Everything before the model's answer, as text (starts with <bos>; tokenise without special tokens)."""
    return f"<bos>{SOT}user\n{TT_Q_HEAD}{question.strip()}{TT_Q_TAIL}{EOT}\n{SOT}model\n"


def encode_tt(tok, question: str, answer: str, a_ids: list[int] | None = None) -> Encoded:
    """Tokenize a two-turn prompt and locate the analysed positions.

    PANL = <end_of_turn> closing the answer turn (first token after the answer), PANL+1 = the newline after
    it, SOT / USER / UNL = "<start_of_turn>", "user", newline opening the request turn, REQ0 = first token of
    the request, EOT2 / CC-3 / CC-2 / CC-1 = "<end_of_turn>", newline, "<start_of_turn>", "model" after the
    request, CC = the final newline (the token that emits the confidence class), AC = the newline before the
    answer (the token that emits the answer). ``a_ids``: the tokens the model generated; the trial is rejected
    if the answer text does not tokenise back to them.
    """
    question, answer = question.strip(), answer.strip()
    pre = tt_prefix(question)
    text = pre + answer + TT_SUFFIX
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id
    q0 = pre.index(TT_Q_HEAD) + len(TT_Q_HEAD)
    q_span = _tok_span(offs, q0, q0 + len(question))
    a_span = _tok_span(offs, len(pre), len(pre) + len(answer))
    assert tok.decode(ids[a_span[0] : a_span[1]]).strip() == answer, (tok.decode(ids[a_span[0] : a_span[1]]), answer)
    if a_ids is not None:
        assert ids[a_span[0] : a_span[1]] == list(a_ids), "answer does not tokenise back to the generated tokens"
    p, n = a_span[1], len(ids)
    dec = lambda i: tok.decode(ids[i])  # noqa: E731
    assert [dec(i) for i in range(p, p + 5)] == [EOT, "\n", SOT, "user", "\n"], [dec(i) for i in range(p, p + 5)]
    assert [dec(i) for i in range(n - 5, n)] == [EOT, "\n", SOT, "model", "\n"], [dec(i) for i in range(n - 5, n)]
    assert [dec(i) for i in range(a_span[0] - 3, a_span[0])] == [SOT, "model", "\n"]
    e = Encoded(ids=ids, q_span=q_span, a_span=a_span)
    e.pos = {
        "PANL": p, "PANL+1": p + 1, "SOT": p + 2, "USER": p + 3, "UNL": p + 4, "REQ0": p + 5,
        "EOT2": n - 5, "CC-3": n - 4, "CC-2": n - 3, "CC-1": n - 2, "CC": n - 1,
        "AC": a_span[0] - 1, "A_first": a_span[0], "A_last": a_span[1] - 1,
        "QTT": q_span[0] + 2,
    }  # fmt: skip
    t0 = len(pre) + len(answer)
    for name in CLASSES:
        c = text.index(f'- "{name}"', t0) + 3
        e.inter.append(_tok_span(offs, c, c + len(name))[0])
    for i, t in enumerate(e.inter):
        e.pos[f"INT{i}"] = t
    e.groups = {"POST5": list(range(p, p + 5)), "REQ": list(range(p + 5, n - 5)), "TAIL4": list(range(n - 5, n - 1))}
    return e


def encode_trial(tok, r: dict, kind: str = "cat", panl: str = PANL_DEFAULT) -> Encoded:
    """Encode one row of trials.jsonl in the active prompt format (VC_FMT)."""
    if FMT == "tt":
        assert kind == "cat", "the two-turn format only has the categorical request"
        return encode_tt(tok, r["question"], r["answer"], r.get("a_ids"))
    return encode(tok, r["question"], r["answer"], kind, panl)


def class_token_ids(tok) -> list[int]:
    """First token of each confidence class where the model emits it: after '**Confidence**:' with a leading
    space (raw format), or at the start of the model's turn without one (two-turn format)."""
    lead = "" if FMT == "tt" else " "
    ids = [tok(lead + c, add_special_tokens=False).input_ids[0] for c in CLASSES]
    assert len(set(ids)) == len(CLASSES), "first tokens of the confidence classes must be unique"
    return ids


def digit_token_ids(tok) -> list[int]:
    ids = [tok(str(d), add_special_tokens=False).input_ids for d in range(10)]
    assert all(len(i) == 1 for i in ids)
    return [i[0] for i in ids]
