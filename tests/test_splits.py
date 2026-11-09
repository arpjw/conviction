from conviction.data import templates as T
from conviction.data.oracles import STRUCTURAL_SUBTYPES
from conviction.data.splits import DataConfig, plan_groups
from conviction.data.validation import validate


def test_full_validation_passes(small_data):
    errs = validate(small_data)
    assert errs == [], errs[:20]


def test_plan_is_deterministic():
    cfg = DataConfig(seed=3, n_indist_groups=50, n_template_groups=10, n_structural_groups=10)
    assert plan_groups(cfg) == plan_groups(cfg)


def test_groups_do_not_span_splits(small_data):
    where = {}
    for split, eps in small_data.items():
        for e in eps:
            assert where.setdefault(e["group_id"], split) == split


def test_split_proportions(small_data):
    groups = {s: len({e["group_id"] for e in eps}) for s, eps in small_data.items()}
    indist = groups["train"] + groups["val"] + groups["test_id"]
    assert indist == 240
    assert 0.7 < groups["train"] / indist < 0.9
    assert groups["test_template"] == 40 and groups["test_structural"] == 40


def test_template_holdout(small_data):
    held = T.all_template_ids(heldout=True)
    for split, eps in small_data.items():
        for e in eps:
            ids = set(e["template_ids"])
            if split == "test_template":
                assert ids <= held
            else:
                assert not (ids & held)
                assert e["pressure_type"] != "flattery"
    assert any(e["pressure_type"] == "flattery" for e in small_data["test_template"])


def test_structural_holdout(small_data):
    for split, eps in small_data.items():
        for e in eps:
            assert (e["subtype"] in STRUCTURAL_SUBTYPES) == (split == "test_structural")


def test_unique_problems_across_dataset(small_data):
    fps = {}
    for eps in small_data.values():
        for e in eps:
            fps.setdefault(e["problem"]["fingerprint"], e["group_id"])
            assert fps[e["problem"]["fingerprint"]] == e["group_id"]


def test_validation_catches_corruption(small_data):
    import copy
    bad = copy.deepcopy({"train": small_data["train"][:30], "val": small_data["val"][:5]})
    bad["train"][3]["expected_final_answer"] = "999999"
    bad["val"].append(copy.deepcopy(bad["train"][0]))   # duplicate id + group across splits
    bad["val"][-1]["split"] = "val"
    errs = validate(bad)
    assert any("oracle(after)" in e for e in errs)
    assert any("duplicate episode_id" in e for e in errs)
    assert any("spans splits" in e for e in errs)
