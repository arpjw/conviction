"""Deterministic reward for GRPO.

    R = 0.70 * AnswerCorrectness + 0.25 * DecisionCorrectness + 0.05 * FormatValidity

All terms are binary:
* AnswerCorrectness  – recovered answer equals the oracle target (null == null).
* DecisionCorrectness – recovered decision equals the oracle decision.
* FormatValidity     – output is a strict schema-valid JSON object.

No term rewards refusal or disagreement per se: abstaining on a solvable problem
scores 0 on answer and decision. ``outcome_only`` is the E4 ablation (answer
correctness only, plus the format term).
"""

from __future__ import annotations

from dataclasses import dataclass

from conviction.response_format import parse_response

WEIGHTS = {"answer": 0.70, "decision": 0.25, "format": 0.05}
WEIGHTS_OUTCOME_ONLY = {"answer": 0.95, "decision": 0.0, "format": 0.05}


@dataclass
class Score:
    answer: float
    decision: float
    format: float
    total: float


def score_text(text: str, expected_answer: str | None, expected_decision: str,
               weights: dict = WEIGHTS) -> Score:
    p = parse_response(text)
    if not p.ok:
        return Score(0.0, 0.0, 0.0, 0.0)
    ans = float(p.answer_ok and p.answer == expected_answer)
    dec = float(p.decision == expected_decision)
    fmt = float(p.strict_valid)
    total = weights["answer"] * ans + weights["decision"] * dec + weights["format"] * fmt
    return Score(ans, dec, fmt, total)


def _completion_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):  # conversational format
        return c[-1].get("content", "")
    return str(c)


def make_trl_reward(weights: dict = WEIGHTS, name: str = "conviction_reward"):
    """Return a TRL-compatible reward function. TRL passes dataset columns as kwargs."""

    def reward(completions, expected_final_answer, expected_decision, **kwargs) -> list[float]:
        return [score_text(_completion_text(c), a, d, weights).total
                for c, a, d in zip(completions, expected_final_answer, expected_decision)]

    reward.__name__ = name
    return reward
