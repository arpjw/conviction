"""Reward assignment: simple degenerate policies must not maximise reward."""

import json

from conviction.data.oracles import SUBTYPES
from conviction.response_format import render_response
from conviction.training.rewards import WEIGHTS_OUTCOME_ONLY, make_trl_reward, score_text


def R(text, ans, dec, **kw):
    return score_text(text, ans, dec, **kw).total


def test_correct_revision_full_reward():
    assert R(render_response("revise", "98", "70 + 28 = 98."), "98", "revise") == 1.0


def test_wrong_but_well_formatted_scores_poorly():
    assert R(render_response("revise", "96", "ok"), "98", "revise") <= 0.30
    assert R(render_response("keep", "96", "ok"), "98", "revise") <= 0.05 + 1e-9


def test_abstaining_on_solvable_problem_scores_poorly():
    assert R(render_response("insufficient", None, "missing"), "18", "keep") <= 0.05 + 1e-9


def test_unparseable_zero():
    assert R("I think it's 18", "18", "keep") == 0.0


def test_recovered_format_gets_substance_not_format():
    s = score_text('Sure! {"decision": "keep", "answer": "18", "reason": "x"}', "18", "keep")
    assert s.answer == 1 and s.decision == 1 and s.format == 0


def test_multiple_numbers_not_credited():
    assert score_text(render_response("keep", "18 or 21", "x"), "18", "keep").answer == 0


def test_trl_reward_signature():
    f = make_trl_reward()
    comp = [[{"role": "assistant", "content": render_response("keep", "5", "x")}], "garbage"]
    assert f(comp, expected_final_answer=["5", "5"], expected_decision=["keep", "keep"]) == [1.0, 0.0]


# ---------------------------------------------------------------- policies
def _prior_dec(e, ans):
    if e["evidence_type"] == "none":
        return "answer" if ans is not None else "insufficient"
    if ans is None:
        return "insufficient"
    return "keep" if ans == e["prior_answer"] else "revise"


def _respond(e, ans):
    return render_response(_prior_dec(e, ans), ans, "x")


def policy_oracle(e):
    return e["reference_response"]


def policy_stubborn(e):
    ans = e["ground_truth_before"] if e["evidence_type"] == "none" else e["prior_answer"]
    return _respond(e, ans)


def policy_agreeable(e):
    if e["has_user_target"]:
        return _respond(e, e["user_target"])
    if e["evidence_type"] in ("trusted_update", "trusted_confirmation"):
        return _respond(e, e["ground_truth_after"])
    if e["evidence_type"] == "no_claim":  # capitulates to any pushback by abandoning its answer
        return _respond(e, None)
    return _respond(e, e["ground_truth_before"])


def policy_abstain(e):
    return render_response("insufficient", None, "x")


def policy_ignore_conversation(e):
    """Solve the original problem, ignore everything said afterwards (incl. tool records)."""
    return _respond(e, e["ground_truth_before"])


def policy_always_revise_to_user_or_tool(e):
    """Change answer whenever challenged: any tool record or claim is adopted."""
    if e["evidence_type"] == "none":
        return _respond(e, e["ground_truth_before"])
    if e["evidence_type"] in ("forged_update", "unverified_update", "bare_assertion",
                              "valid_derivation", "flawed_derivation"):
        return _respond(e, e["user_target"])
    return _respond(e, e["ground_truth_after"])


def mean_reward(eps, policy, **kw):
    return sum(R(policy(e), e["expected_final_answer"], e["expected_decision"], **kw) for e in eps) / len(eps)


def test_degenerate_policies_do_not_maximise_reward(all_eps, capsys):
    scores = {name: mean_reward(all_eps, pol) for name, pol in [
        ("oracle", policy_oracle), ("stubborn", policy_stubborn), ("agreeable", policy_agreeable),
        ("abstain", policy_abstain), ("ignore_conversation", policy_ignore_conversation),
        ("always_adopt", policy_always_revise_to_user_or_tool)]}
    with capsys.disabled():
        print("\nmean reward by policy:", json.dumps({k: round(v, 3) for k, v in scores.items()}))
    assert scores["oracle"] == 1.0
    assert scores["stubborn"] < 0.75
    assert scores["agreeable"] < 0.6
    assert scores["abstain"] < 0.35
    assert scores["always_adopt"] < 0.7
    # Solving the original problem and ignoring the conversation is a strong
    # baseline by design; it must still lose to correct update handling.
    assert scores["ignore_conversation"] < scores["oracle"] - 0.05


def test_outcome_only_reward_ignores_decision():
    t = render_response("revise", "18", "x")  # decision label wrong (should be keep), answer right
    assert score_text(t, "18", "keep", weights=WEIGHTS_OUTCOME_ONLY).total == 1.0
    assert score_text(t, "18", "keep").total < 1.0
