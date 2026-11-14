"""Prompts from the paper (Figs. 8, 15, 16) and token-position bookkeeping.

All prompts are fed to the model as raw text (BOS + prompt), so that the confidence-colon (CC) token is the
literal last token of the prompt whose final-layer residual produces the confidence token (paper Fig. 1D),
and the answer-colon (AC) token is the last token of the Phase-0 prompt (paper Sec. 2.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class Encoded:
    """A tokenized confidence prompt plus the token indices of the positions analysed in the paper."""

    ids: list[int]
    pos: dict[str, int] = field(default_factory=dict)  # named single positions
    q_span: tuple[int, int] = (0, 0)  # [start, end) of question tokens
    a_span: tuple[int, int] = (0, 0)  # [start, end) of answer tokens
    inter: list[int] = field(default_factory=list)  # first token of each class name in the instruction list

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
    return e


def class_token_ids(tok) -> list[int]:
    """First token of each confidence class as it follows '**Confidence**:' (with a leading space)."""
    ids = [tok(" " + c, add_special_tokens=False).input_ids[0] for c in CLASSES]
    assert len(set(ids)) == len(CLASSES), "first tokens of the confidence classes must be unique"
    return ids


def digit_token_ids(tok) -> list[int]:
    ids = [tok(str(d), add_special_tokens=False).input_ids for d in range(10)]
    assert all(len(i) == 1 for i in ids)
    return [i[0] for i in ids]
