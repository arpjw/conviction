import pytest

from conviction.response_format import normalize_answer, parse_response


@pytest.mark.parametrize("raw,canon", [
    ("18", "18"), (" 18 ", "18"), ("$60", "60"), ("18 units", "18"), ("1,200", "1200"),
    ("−5", "-5"), ("-5", "-5"), ("18.0", "18"), (18, "18"), ("True", "true"), ("false", "false"),
    ("90 credits", "90"), ("22 points", "22"), (True, "true"),
])
def test_normalize_ok(raw, canon):
    assert normalize_answer(raw) == (True, canon)


@pytest.mark.parametrize("raw", ["18 or 21", "about 18 maybe", "x = 4", "unknown", ["18"]])
def test_normalize_rejects(raw):
    assert normalize_answer(raw)[0] is False


@pytest.mark.parametrize("raw", [None, "null", "None", ""])
def test_normalize_null(raw):
    assert normalize_answer(raw) == (True, None)


def test_strict_vs_recovered():
    p = parse_response('{"decision": "keep", "answer": "18", "reason": "ok"}')
    assert p.ok and p.strict_valid and p.answer == "18"
    p = parse_response('```json\n{"decision": "keep", "answer": "18", "reason": "ok"}\n```')
    assert p.strict_valid
    p = parse_response('<think>\n\n</think>\n\n{"decision": "keep", "answer": "18", "reason": "ok"}')
    assert p.strict_valid
    p = parse_response('Answer: {"decision": "keep", "answer": "18", "reason": "ok"} done')
    assert p.ok and not p.strict_valid and p.answer == "18"
    p = parse_response('{"decision": "insufficient", "answer": "18", "reason": "ok"}')
    assert p.ok and not p.strict_valid
    p = parse_response('{"decision": "keep", "answer": 18, "reason": "ok"}')
    assert p.ok and not p.strict_valid and p.answer == "18"   # int not string: substance ok, format not
    assert not parse_response("no json here").ok
    p = parse_response('{"decision": "keep", "answer": "a {b}", "reason": "has } brace"}')
    assert p.ok
