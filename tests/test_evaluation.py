from conviction.data.adversarial import build as build_adversarial
from conviction.data.validation import _check_episode
from conviction.evaluation import metrics as M
from conviction.evaluation.natural import natural_battery
from conviction.evaluation.statistics import bootstrap, paired_bootstrap
from conviction.response_format import render_response


def test_adversarial_suite_is_consistent():
    items = build_adversarial()
    assert len(items) >= 20
    for e in items:
        errs = [x for x in _check_episode(e) if "provenance" not in x]
        assert errs == [], errs
    decs = {e["expected_decision"] for e in items}
    assert decs == {"keep", "revise", "insufficient"}


def test_oracle_predictions_give_perfect_metrics(all_eps):
    recs = [M.score_record(e, e["reference_response"]) for e in all_eps]
    m = M.compute(recs)
    assert m["final_accuracy"]["value"] == 1.0
    assert m["pressure_flip_rate"]["value"] == 0.0
    assert m["legitimate_correction_rate"]["value"] == 1.0
    assert m["capitulation_rate"]["value"] == 0.0
    assert m["false_abstention_rate"]["value"] == 0.0
    assert m["insufficient_accuracy"]["value"] == 1.0


def test_stubborn_predictions_never_flip_and_never_correct(all_eps):
    def stub(e):
        a = e["ground_truth_before"] if e["evidence_type"] == "none" else e["prior_answer"]
        return render_response("keep" if a is not None else "insufficient", a, "x")
    m = M.compute([M.score_record(e, stub(e)) for e in all_eps])
    assert m["pressure_flip_rate"]["value"] == 0.0
    assert m["legitimate_correction_rate"]["value"] == 0.0


def test_bootstrap_and_paired(all_eps):
    eps = [e for e in all_eps if e["split"] == "test_id"]
    good = [M.score_record(e, e["reference_response"]) for e in eps]
    bad = [M.score_record(e, render_response("insufficient", None, "x")) for e in eps]
    b = bootstrap(good, n_boot=200)
    assert b["final_accuracy"]["lo"] == 1.0
    d = paired_bootstrap(bad, good, n_boot=200)
    assert d["final_accuracy"]["diff"] > 0.5 and d["final_accuracy"]["lo"] > 0.5


def test_natural_battery_labels(all_eps):
    cleans = [e for e in all_eps if e["slot"] == "clean"][:60]
    n = 0
    for c in cleans:
        # model answered correctly
        for e in natural_battery(c, c["reference_response"]):
            n += 1
            assert e["prior_answer"] == c["expected_final_answer"]
            if e["evidence_type"] != "trusted_update":
                assert e["expected_final_answer"] == c["expected_final_answer"]
        # model answered wrongly (a made-up number)
        wrong = render_response("answer", "123457", "x")
        for e in natural_battery(c, wrong):
            assert e["prior_answer"] == "123457"
            assert e["messages"][1]["content"] == wrong
            if e["expected_final_answer"] is not None:
                assert e["expected_decision"] == "revise"
    assert n > 200
    assert natural_battery(cleans[0], "not json") == []
