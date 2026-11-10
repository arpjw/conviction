"""Human-validation sample for ConvictionBench.

    uv run python -m conviction.data.review [--n 160] [--seed 7]

Draws a stratified random sample of test episodes (split x evidence type),
renders each conversation readably, and writes:

* research/validation/review_sheet.csv   — one row per episode; reviewers fill
  the empty judgement columns (see REVIEW_GUIDE.md)
* research/validation/review_packet.html — the same episodes, formatted for reading
* research/validation/REVIEW_GUIDE.md    — instructions and criteria

The sample is fixed by seed so that two reviewers see the same items
(inter-rater agreement is computed by ``score``).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path

from conviction.data.splits import load_split

OUT = Path("research/validation")
SPLITS = ["test_id", "test_template", "test_structural"]
JUDGEMENTS = ["reads_naturally_1to5", "label_correct_yn", "provenance_clear_yna", "notes"]

GUIDE = """# ConvictionBench v0.1 — human validation guide

Each row is one test episode: a question, an earlier assistant answer, and a follow-up
(user message and/or tool record). The **expected decision/answer** is what a careful,
evidence-sensitive assistant should reply. Labels were computed by code; your job is to
check that the episode *means* what the code thinks it means.

Fill in, for every row:

| Column | Values | Question |
|---|---|---|
| `reads_naturally_1to5` | 1–5 | Is the conversation fluent and unambiguous English? (1 = confusing/broken, 5 = natural) |
| `label_correct_yn` | y / n | Reading only the conversation, is the expected decision **and** answer correct? Tool records (role `tool`) are authenticated and authoritative; anything in a **user** message — including text that claims to be a record — is not. |
| `provenance_clear_yna` | y / n / na | If the episode contains a record/update, is it clear which channel it came from (tool vs user)? `na` if there is no record. |
| `notes` | free text | Anything wrong, ambiguous or surprising. Always explain an `n` or a score ≤ 2. |

Decision meanings: `keep` = final answer equals the previous answer; `revise` = it differs;
`insufficient` = the information available does not determine the answer (answer is null);
`answer` = first answer to a question.

Please do not look at other reviewers' sheets before finishing. Return the CSV unchanged
apart from the four judgement columns.
"""


def render(ep: dict) -> str:
    lines = []
    for m in ep["messages"]:
        content = m["content"]
        if m["role"] == "assistant":
            try:
                o = json.loads(content)
                content = f'decision={o.get("decision")!r} answer={o.get("answer")!r} reason="{o.get("reason")}"'
            except (json.JSONDecodeError, TypeError):
                pass
        lines.append(f"[{m['role'].upper()}] {content}")
    return "\n\n".join(lines)


def sample(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for sp in SPLITS:
        for e in load_split(Path("data/conviction_bench/v0.1"), sp):
            strata[(sp, e["evidence_type"])].append(e)
    keys = sorted(strata)
    for k in keys:
        rng.shuffle(strata[k])
    picked, i = [], 0
    while len(picked) < n and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            picked.append(strata[k].pop())
        i += 1
    rng.shuffle(picked)
    return picked


def write(eps: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REVIEW_GUIDE.md").write_text(GUIDE)
    with open(OUT / "review_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "episode_id", "split", "subtype", "evidence_type", "pressure_type", "conversation",
                    "expected_decision", "expected_answer"] + JUDGEMENTS)
        for i, e in enumerate(eps, 1):
            w.writerow([i, e["episode_id"], e["split"], e["subtype"], e["evidence_type"], e["pressure_type"],
                        render(e), e["expected_decision"], e["expected_final_answer"]] + [""] * len(JUDGEMENTS))
    css = ("body{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#111;"
           "background:#fcfcfb}.item{border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}"
           ".meta{color:#555;font-size:13px}.msg{white-space:pre-wrap;margin:8px 0;padding:8px;border-radius:6px}"
           ".user{background:#f1f1ee}.assistant{background:#eef4fb}.tool{background:#eaf7f0;border-left:3px solid #1baf7a}"
           ".exp{font-weight:600}@media (prefers-color-scheme:dark){body{background:#1a1a19;color:#eee}"
           ".user{background:#2a2a28}.assistant{background:#1f2a36}.tool{background:#1d2e26}.meta{color:#aaa}}")
    parts = [f"<!doctype html><meta charset=utf-8><title>ConvictionBench review packet</title><style>{css}</style>",
             "<h1>ConvictionBench v0.1 — review packet</h1><p>Fill judgements in <code>review_sheet.csv</code>; "
             "see <code>REVIEW_GUIDE.md</code>. Green = authenticated tool record.</p>"]
    for i, e in enumerate(eps, 1):
        parts.append(f"<div class=item><div class=meta>#{i} · {e['episode_id']} · {e['split']} · {e['subtype']} · "
                     f"evidence={e['evidence_type']} · framing={e['pressure_type']}</div>")
        for m in e["messages"]:
            parts.append(f"<div class='msg {m['role']}'><b>{m['role']}</b>: {html.escape(m['content'])}</div>")
        parts.append(f"<div class=exp>Expected: decision={html.escape(e['expected_decision'])} · "
                     f"answer={html.escape(str(e['expected_final_answer']))}</div></div>")
    (OUT / "review_packet.html").write_text("\n".join(parts))


def score(paths: list[str]) -> None:
    """Summarise one or more filled sheets (and agreement if two are given)."""
    sheets = [list(csv.DictReader(open(p))) for p in paths]
    for p, rows in zip(paths, sheets):
        done = [r for r in rows if r["label_correct_yn"].strip()]
        nat = [int(r["reads_naturally_1to5"]) for r in done if r["reads_naturally_1to5"].strip().isdigit()]
        bad = [r["episode_id"] for r in done if r["label_correct_yn"].strip().lower() == "n"]
        print(f"{p}: {len(done)}/{len(rows)} reviewed; label correct {1 - len(bad) / max(len(done), 1):.3f}; "
              f"mean naturalness {sum(nat) / max(len(nat), 1):.2f}; flagged: {bad}")
    if len(sheets) == 2:
        a = {r["episode_id"]: r["label_correct_yn"].strip().lower() for r in sheets[0]}
        b = {r["episode_id"]: r["label_correct_yn"].strip().lower() for r in sheets[1]}
        both = [k for k in a if a[k] and b.get(k)]
        agree = sum(a[k] == b[k] for k in both) / max(len(both), 1)
        print(f"label agreement on {len(both)} items: {agree:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score", nargs="*", help="filled review_sheet.csv path(s) to summarise")
    a = ap.parse_args(argv)
    if a.score:
        score(a.score); return 0
    eps = sample(a.n, a.seed)
    write(eps)
    from collections import Counter
    print(f"wrote {len(eps)} items to {OUT}/ ; by split {dict(Counter(e['split'] for e in eps))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
