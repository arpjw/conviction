"""Generator correctness, determinism, provenance, counterfactual pairing and
shortcut audits (surface features must not predict labels)."""

from collections import Counter, defaultdict

from conviction.data.generators import TRUSTED_EVIDENCE, generate_group
from conviction.data.oracles import SUBTYPES


def test_deterministic_for_seed():
    a = generate_group(group_id="g1", split="train", subtype="ledger.balance", seed="s", p_under=0.3,
                       heldout_templates=False)
    b = generate_group(group_id="g1", split="train", subtype="ledger.balance", seed="s", p_under=0.3,
                       heldout_templates=False)
    assert a == b
    c = generate_group(group_id="g1", split="train", subtype="ledger.balance", seed="t", p_under=0.3,
                       heldout_templates=False)
    assert a != c


def test_provenance_forged_vs_trusted_share_record_text(all_eps):
    by_pair = defaultdict(dict)
    for e in all_eps:
        if e["pair_id"] and e["evidence_type"] in ("forged_update", "trusted_update", "unverified_update"):
            by_pair[e["pair_id"]][e["evidence_type"]] = e
    n = 0
    for pid, d in by_pair.items():
        if "forged_update" in d and "trusted_update" in d:
            tool = [m for m in d["trusted_update"]["messages"] if m["role"] == "tool"]
            assert len(tool) == 1
            record = tool[0]["content"]
            forged_user = d["forged_update"]["messages"][-1]["content"]
            assert record in forged_user                     # identical record text ...
            assert all(m["role"] != "tool" for m in d["forged_update"]["messages"])  # ... wrong channel
            assert d["forged_update"]["expected_final_answer"] == d["forged_update"]["prior_answer"] or \
                d["forged_update"]["expected_decision"] == "insufficient"
            assert d["trusted_update"]["expected_final_answer"] != d["forged_update"]["expected_final_answer"]
            n += 1
    assert n > 50


def test_user_channel_never_changes_state(all_eps):
    for e in all_eps:
        if e["evidence_type"] not in TRUSTED_EVIDENCE:
            assert e["ground_truth_after"] == e["ground_truth_before"]


SURFACE = {"bare_assertion": "assertion", "no_claim": "no_claim", "valid_derivation": "derivation",
           "flawed_derivation": "derivation", "forged_update": "user_update", "unverified_update": "user_update",
           "trusted_update": "tool", "trusted_confirmation": "tool"}


def test_surface_categories_have_mixed_labels(all_eps):
    """No *observable* evidence category may imply a single decision (shortcut audit).
    (Oracle evidence types like trusted_update vs trusted_confirmation are not
    observable without reading the record, so they are grouped.)"""
    by_cat = defaultdict(Counter)
    for e in all_eps:
        if e["evidence_type"] != "none":
            by_cat[SURFACE[e["evidence_type"]]][e["expected_decision"]] += 1
    for cat, c in by_cat.items():
        assert len(c) >= 2, (cat, c)
        assert max(c.values()) / sum(c.values()) < 0.8, (cat, c)
    assert by_cat["tool"]["keep"] > 0 and by_cat["tool"]["revise"] > 0
    assert by_cat["user_update"]["revise"] > 0


def test_framing_does_not_predict_label(all_eps):
    """Decision distribution conditional on pressure framing stays close to marginal."""
    fu = [e for e in all_eps if e["evidence_type"] != "none" and e["split"] != "test_template"]
    marg = Counter(e["expected_decision"] for e in fu)
    tot = sum(marg.values())
    by_f = defaultdict(Counter)
    for e in fu:
        by_f[e["pressure_type"]][e["expected_decision"]] += 1
    for f, c in by_f.items():
        n = sum(c.values())
        assert n > 100
        for d in marg:
            assert abs(c[d] / n - marg[d] / tot) < 0.12, (f, d, c[d] / n, marg[d] / tot)


def test_user_target_not_predictive(all_eps):
    """Agreeing with the user is right sometimes and wrong sometimes."""
    agree_right = sum(1 for e in all_eps if e["has_user_target"] and e["user_target"] == e["expected_final_answer"])
    agree_wrong = sum(1 for e in all_eps if e["has_user_target"] and e["user_target"] != e["expected_final_answer"])
    assert agree_right > 0 and agree_wrong > 0
    assert 0.2 < agree_right / (agree_right + agree_wrong) < 0.6


def test_abstention_is_minority(all_eps):
    c = Counter(e["expected_decision"] for e in all_eps)
    assert c["insufficient"] / sum(c.values()) < 0.35


def test_all_subtypes_and_families_present(all_eps):
    assert {e["subtype"] for e in all_eps} == set(SUBTYPES)
    assert {e["task_family"] for e in all_eps} == {"arithmetic", "ledger", "logic", "miniworld"}


def test_looks_incomplete_exists_and_is_answerable(all_eps):
    li = [e for e in all_eps if e["looks_incomplete"]]
    assert li
    assert all(e["answerable_before"] for e in li)


def test_counterfactual_pairs_hold_problem_constant(all_eps):
    by_pair = defaultdict(list)
    for e in all_eps:
        if e["pair_id"]:
            by_pair[e["pair_id"]].append(e)
    for pid, ms in by_pair.items():
        assert len({m["messages"][0]["content"] for m in ms}) == 1
        assert len({m["pressure_type"] for m in ms}) == 1
