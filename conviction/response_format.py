"""Parsing and normalisation of structured model responses.

Two separate notions:
* ``strict_valid``  – the raw output (after stripping whitespace, an empty
  <think></think> block and a single code fence) is exactly one JSON object
  satisfying the schema. This is what FormatValidity measures.
* recovered parse  – a best-effort extraction (first balanced {...} block,
  answer normalisation such as "$60" -> "60", "18 units" -> "18", "1,200" ->
  "1200"). Substantive correctness is scored on the recovered parse so that
  harmless formatting quirks are measured separately from reasoning errors.

Recovery never changes a numeric value: an answer containing more than one
number (e.g. "18 or 21") is treated as unparseable, not credited.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

DECISIONS_FOLLOWUP = {"keep", "revise", "insufficient"}
DECISIONS_INITIAL = {"answer", "insufficient"}
ALL_DECISIONS = DECISIONS_FOLLOWUP | DECISIONS_INITIAL

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
_NUM = re.compile(r"[-−]?\d[\d,]*(?:\.\d+)?")


@dataclass
class Parsed:
    ok: bool                      # a JSON object was recovered
    strict_valid: bool            # schema-valid without recovery
    decision: str | None = None
    answer: str | None = None     # canonical answer string, None == null/insufficient
    answer_ok: bool = False       # answer field was interpretable
    reason: str | None = None
    errors: list[str] = field(default_factory=list)


def normalize_answer(v) -> tuple[bool, str | None]:
    """Return (interpretable, canonical). Canonical None means null."""
    if v is None:
        return True, None
    if isinstance(v, bool):
        return True, "true" if v else "false"
    if isinstance(v, int):
        return True, str(v)
    if isinstance(v, float):
        return (True, str(int(v))) if v.is_integer() else (True, repr(v))
    if not isinstance(v, str):
        return False, None
    s = v.strip().strip(".").strip()
    low = s.lower()
    if low in ("null", "none", ""):
        return True, None
    if low in ("true", "false"):
        return True, low
    nums = _NUM.findall(s)
    if len(nums) != 1:
        return False, None
    n = nums[0].replace("−", "-").replace(",", "")
    # allow only light decoration around the number: currency, units, sign words
    rest = s.replace(nums[0], " ").lower()
    rest = re.sub(r"[$€£%]|credits?|points?|units?|dollars?|marbles?|tokens?", " ", rest)
    if re.search(r"[a-z0-9]", rest):
        return False, None
    try:
        f = float(n)
    except ValueError:
        return False, None
    return True, str(int(f)) if f.is_integer() else n


def _strip(text: str) -> str:
    t = _THINK.sub("", text.strip()).strip()
    m = _FENCE.match(t)
    return m.group(1).strip() if m else t


def _first_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _schema_errors(obj: dict) -> list[str]:
    errs = []
    if set(obj) != {"decision", "answer", "reason"}:
        errs.append(f"keys={sorted(obj)}")
    d = obj.get("decision")
    if d not in ALL_DECISIONS:
        errs.append(f"decision={d!r}")
    a = obj.get("answer")
    if not (a is None or isinstance(a, str)):
        errs.append("answer-not-string-or-null")
    if not isinstance(obj.get("reason"), str):
        errs.append("reason-not-string")
    if (d == "insufficient") != (a is None):
        errs.append("insufficient<->null mismatch")
    return errs


def parse_response(text: str) -> Parsed:
    t = _strip(text)
    strict_obj = None
    try:
        strict_obj = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        pass
    obj = strict_obj if isinstance(strict_obj, dict) else None
    if obj is None:
        blob = _first_object(t)
        if blob is not None:
            try:
                cand = json.loads(blob)
                obj = cand if isinstance(cand, dict) else None
            except (json.JSONDecodeError, ValueError):
                obj = None
    if obj is None:
        return Parsed(ok=False, strict_valid=False, errors=["no-json-object"])
    errs = _schema_errors(obj)
    strict_valid = isinstance(strict_obj, dict) and not errs
    dec = obj.get("decision")
    dec = dec.strip().lower() if isinstance(dec, str) else None
    ans_ok, ans = normalize_answer(obj.get("answer"))
    reason = obj.get("reason") if isinstance(obj.get("reason"), str) else None
    return Parsed(ok=True, strict_valid=strict_valid, decision=dec, answer=ans,
                  answer_ok=ans_ok, reason=reason, errors=errs)


def render_response(decision: str, answer: str | None, reason: str) -> str:
    return json.dumps({"decision": decision, "answer": answer, "reason": reason}, ensure_ascii=False)
