"""Oracle correctness: hand-verified cases, an independent re-implementation
cross-check, and property-based invariants over random problems."""

import itertools
import random
import re

import pytest
from hypothesis import given, settings, strategies as st

from conviction.data.oracles import (LEDGER_KINDS, SUBTYPES, ChainSubtype, InconsistentPremises,
                                     eval_chain)

S = SUBTYPES


# ----------------------------------------------------------------- hand cases
def test_spec_example_A_inventory():
    p = {"shape": 0, "nums": [27, 9, 1], "ops": ["-", "×"], "phr": 0}
    assert S["arithmetic.expr"].solve(p) == "18"


def test_spec_example_B_multiplication():
    p = {"shape": 0, "nums": [14, 7, 0], "ops": ["×", "+"], "phr": 0}
    assert S["arithmetic.expr"].solve(p) == "98"


def test_spec_example_C_refund_changes_balance():
    led = S["ledger.balance"]
    p = {"acct": "A-1", "opening": 100, "txns": [["withdrawal", 25]], "phr": 0}
    assert led.solve(p) == "75"
    p2 = {**p, "txns": [["withdrawal", 25], ["refund", 15]]}
    assert led.solve(p2) == "90"


def test_spec_example_E_missing_opening_balance():
    led = S["ledger.balance"]
    p = {"acct": "A-1", "opening": None, "txns": [["deposit", 40], ["deposit", 60]], "phr": 0}
    assert led.solve(p) is None
    # ... but the *net change* is still determined without the opening balance
    assert S["ledger.net_change"].solve(p) == "100"


def test_percent_ratio_linear_modular():
    assert S["arithmetic.percent"].solve({"item": "x", "price": 80, "pct": 25, "phr": 0}) == "60"
    assert S["arithmetic.percent"].solve({"item": "x", "price": None, "pct": 25, "phr": 0}) is None
    r = {"a": 3, "b": 5, "total": 64, "ca": "red", "cb": "blue", "container": "jar", "phr": 0}
    assert S["arithmetic.ratio"].solve(r) == "24"
    assert S["arithmetic.ratio"].solve({**r, "total": None}) is None
    lin = {"a": 4, "k": 3, "y": 5, "c": 31, "phr": 0}
    assert S["arithmetic.linear"].solve(lin) == "4"
    assert S["arithmetic.linear"].solve({**lin, "y": None}) is None
    assert S["arithmetic.modular"].solve({"a": 7, "b": 8, "c": 5, "m": 6, "phr": 0}) == "1"


def test_boolean_unknown_variable_sometimes_irrelevant():
    b = S["logic.boolean"]
    e = ["or", ["and", "p", "q"], "r"]
    assert b.solve({"expr": e, "vals": {"p": True, "q": True, "r": None}, "phr": 0}) == "true"
    assert b.solve({"expr": e, "vals": {"p": False, "q": True, "r": None}, "phr": 0}) is None
    assert b.solve({"expr": ["imp", "p", "q"], "vals": {"p": True, "q": False}, "phr": 0}) == "false"


def test_entailment():
    ent = S["logic.entailment"]
    atoms = [["bridge", "raised"], ["gate", "open"], ["lamp", "on"]]
    p = {"atoms": atoms, "facts": [[1, True]], "rules": [[[[1, True]], [2, True]], [[[2, True]], [0, False]]],
         "query": 0, "phr": 0}
    assert ent.solve(p) == "false"
    assert ent.solve({**p, "facts": []}) is None
    with pytest.raises(InconsistentPremises):
        ent.solve({**p, "facts": [[1, True], [0, True]]})


def test_sets():
    s = S["logic.sets"]
    base = {"sets": {"A": [1, 2, 3], "B": [2, 3, 4], "C": [9]}, "x": 3, "phr": 0}
    assert s.solve({**base, "form": 0}) == "true"     # (A∩B)∪C
    assert s.solve({**base, "form": 1}) == "false"    # (A∪B)∩C
    assert s.solve({**base, "form": 2}) == "false"    # A−(B∪C)
    assert s.solve({**base, "x": 1, "form": 2}) == "true"


def test_miniworld():
    t = S["miniworld.tokens"]
    p = {"world": "W", "values": {"blue": 3, "red": 5}, "bag": [["blue", 4], ["red", 2]], "phr": 0}
    assert t.solve(p) == "22"
    assert t.solve({**p, "bag": [["blue", 4], ["copper", 1]]}) is None
    m = S["miniworld.multiplier"]
    q = {"world": "W", "base": {"blue": 3, "red": 4}, "derived": {"green": ["times", "blue", 2],
         "teal": ["plus", "red", 1]}, "bag": [["green", 2], ["blue", 1], ["teal", 1]], "phr": 0}
    assert m.solve(q) == str(2 * 6 + 3 + 5)
    assert m.solve({**q, "base": {"blue": None, "red": 4}}) is None


# -------------------------------------------- independent re-implementations
def _indep(key: str, p: dict):
    """Straightforward re-implementation, sharing no code with the oracles."""
    if key == "arithmetic.percent":
        return None if p["price"] is None else p["price"] - p["price"] * p["pct"] // 100
    if key == "arithmetic.ratio":
        return None if p["total"] is None else p["total"] // (p["a"] + p["b"]) * p["a"]
    if key == "arithmetic.linear":
        return None if p["y"] is None else (p["c"] - p["k"] * p["y"]) // p["a"]
    if key == "arithmetic.modular":
        return (p["a"] * p["b"] + p["c"]) % p["m"]
    if key == "arithmetic.expr":
        n, o = p["nums"], [x.replace("×", "*") for x in p["ops"]]
        if p["shape"] == 0:
            e = f"({n[0]}{o[0]}{n[1]}){o[1]}{n[2]}"
        elif p["shape"] == 1:
            e = f"{n[0]}{o[0]}({n[1]}{o[1]}{n[2]})"
        else:
            e = f"({n[0]}{o[0]}{n[1]}){o[1]}({n[2]}{o[2]}{n[3]})"
        return eval(e)  # noqa: S307 - digits and operators only
    if key in ("ledger.balance", "ledger.net_change"):
        if any(a is None for _, a in p["txns"]):
            return None
        net = sum(LEDGER_KINDS[k][0] * a for k, a in p["txns"])
        if key == "ledger.net_change":
            return net
        return None if p["opening"] is None else p["opening"] + net
    if key == "miniworld.tokens":
        if any(p["values"].get(c) is None for c, _ in p["bag"]):
            return None
        return sum(n * p["values"][c] for c, n in p["bag"])
    if key == "miniworld.multiplier":
        def val(c):
            if c in p["base"]:
                return p["base"][c]
            mode, r, k = p["derived"][c]
            v = p["base"].get(r)
            if v is None:
                return None
            return v * k if mode == "times" else v + k
        vs = [val(c) for c, _ in p["bag"]]
        return None if None in vs else sum(n * v for (c, n), v in zip(p["bag"], vs))
    if key == "logic.sets":
        A, B, C = (set(p["sets"][k]) for k in "ABC")
        res = [(A & B) | C, (A | B) & C, A - (B | C)][p["form"]]
        return p["x"] in res
    if key == "logic.boolean":
        def ev(e, env):
            if isinstance(e, str):
                return env[e]
            if e[0] == "not":
                return not ev(e[1], env)
            a, b = ev(e[1], env), ev(e[2], env)
            return {"and": a and b, "or": a or b, "imp": (not a) or b}[e[0]]
        unk = [v for v, x in p["vals"].items() if x is None]
        outs = set()
        for combo in itertools.product([0, 1], repeat=len(unk)):
            env = {**{k: v for k, v in p["vals"].items() if v is not None}, **dict(zip(unk, map(bool, combo)))}
            outs.add(ev(p["expr"], env))
        return outs.pop() if len(outs) == 1 else None
    if key == "logic.entailment":
        n = len(p["atoms"])
        vals = set()
        for a in itertools.product([False, True], repeat=n):
            if all(a[i] == v for i, v in p["facts"]) and all(
                    not all(a[i] == v for i, v in ants) or a[c] == cv for ants, (c, cv) in p["rules"]):
                vals.add(a[p["query"]])
        return vals.pop() if len(vals) == 1 else None
    raise KeyError(key)


def _canon(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


@pytest.mark.parametrize("key", sorted(SUBTYPES))
def test_oracle_matches_independent_implementation(key):
    sub = SUBTYPES[key]
    rng = random.Random(key)
    for _ in range(300):
        p = sub.sample(rng)
        assert sub.solve(p) == _canon(_indep(key, p)), p
        if sub.can_blank:
            q = sub.blank(p, rng)
            if q is not None:
                assert sub.solve(q) == _canon(_indep(key, q)), q
        for m in sub.modifications(p, rng):
            assert sub.solve(m.params) == _canon(_indep(key, m.params)), m


# --------------------------------------------------------------- properties
KEYS = sorted(SUBTYPES)


@settings(max_examples=150, deadline=None)
@given(key=st.sampled_from(KEYS), seed=st.integers(0, 10**9))
def test_sample_is_determinable_and_derivation_consistent(key, seed):
    sub = SUBTYPES[key]
    rng = random.Random(seed)
    p = sub.sample(rng)
    ans = sub.solve(p)
    assert ans is not None
    steps = sub.derivation(p)
    assert steps and all(isinstance(s, str) and s for s in steps)
    if isinstance(sub, ChainSubtype):
        last = re.findall(r"= (-?\d+)$", steps[-1])
        assert last and last[0] == ans
    assert sub.render(p).strip()
    for d in sub.distractors(p, rng):
        assert d != ans


@settings(max_examples=150, deadline=None)
@given(key=st.sampled_from(KEYS), seed=st.integers(0, 10**9))
def test_flawed_derivation_reaches_wrong_answer(key, seed):
    sub = SUBTYPES[key]
    rng = random.Random(seed)
    p = sub.sample(rng)
    fl = sub.flawed(p, rng)
    if fl is not None:
        steps, x = fl
        assert x != sub.solve(p)
        assert steps


@settings(max_examples=150, deadline=None)
@given(key=st.sampled_from([k for k in KEYS if SUBTYPES[k].can_blank]), seed=st.integers(0, 10**9))
def test_supplies_make_blanked_problem_determinable(key, seed):
    sub = SUBTYPES[key]
    rng = random.Random(seed)
    q = sub.blank(sub.sample(rng), rng)
    if q is None or sub.solve(q) is not None:
        return
    assert sub.insufficiency_reason(q)
    for m in sub.supplies(q, rng):
        assert sub.solve(m.params) is not None


def test_chain_corruption_propagates():
    steps = [("+", 2, 3), ("×", "s0", 4)]
    assert eval_chain(steps) == [5, 20]
    assert eval_chain(steps, {0: 1}) == [6, 24]
