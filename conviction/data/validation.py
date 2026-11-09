"""Dataset integrity checks. ``validate(data)`` returns a list of error strings;
an empty list means the dataset passed every check."""

from __future__ import annotations

import json
from collections import defaultdict

from conviction.data import templates as T
from conviction.data.generators import TRUSTED_EVIDENCE
from conviction.data.oracles import SUBTYPES, InconsistentPremises
from conviction.data.splits import SPLITS
from conviction.response_format import parse_response

REQUIRED = ["episode_id", "group_id", "task_family", "pressure_type", "evidence_type",
            "prior_answer", "prior_answer_correct", "ground_truth_before", "ground_truth_after",
            "expected_decision", "expected_final_answer", "messages", "template_id",
            "generator_seed", "split"]

# Oracle-only strings that must never leak into model-visible text
CONTROL_STRINGS = ["<|im_start|>", "<|im_end|>", "<tool_response>", "</tool_response>",
                   "<tool_call>", "<think>", "expected_", "ground_truth"]


def _check_episode(e: dict) -> list[str]:
    errs = []
    eid = e.get("episode_id", "?")
    for k in REQUIRED:
        if k not in e:
            errs.append(f"{eid}: missing field {k}")
    if errs:
        return errs
    sub = SUBTYPES[e["problem"]["subtype"]]
    try:
        before = sub.solve(e["problem"]["params"])
        after = sub.solve(e["problem"]["params_after"])
    except InconsistentPremises:
        return [f"{eid}: inconsistent premises"]
    if before != e["ground_truth_before"]:
        errs.append(f"{eid}: oracle(before)={before!r} != label {e['ground_truth_before']!r}")
    if after != e["ground_truth_after"] or after != e["expected_final_answer"]:
        errs.append(f"{eid}: oracle(after)={after!r} != label {e['expected_final_answer']!r}")
    # decision consistency
    dec, fin, prior = e["expected_decision"], e["expected_final_answer"], e["prior_answer"]
    if e["evidence_type"] == "none":
        want = "answer" if fin is not None else "insufficient"
    else:
        want = "insufficient" if fin is None else ("keep" if fin == prior else "revise")
    if dec != want:
        errs.append(f"{eid}: decision {dec} inconsistent (expected {want})")
    # provenance: tool messages iff trusted evidence
    roles = [m["role"] for m in e["messages"]]
    has_tool = "tool" in roles
    if has_tool != (e["evidence_type"] in TRUSTED_EVIDENCE):
        errs.append(f"{eid}: tool-channel provenance mismatch")
    if e["evidence_type"] not in TRUSTED_EVIDENCE and e["problem"]["params_after"] != e["problem"]["params"]:
        errs.append(f"{eid}: user-channel evidence changed world state")
    if roles[-1] != "user" and e["evidence_type"] != "none":
        errs.append(f"{eid}: last message is not a user turn")
    if roles[0] != "user":
        errs.append(f"{eid}: first message must be the question")
    for m in e["messages"]:
        if m["role"] == "system":
            errs.append(f"{eid}: system prompt stored in episode")
        for c in CONTROL_STRINGS:
            if c in m["content"]:
                errs.append(f"{eid}: control/oracle string {c!r} in visible text")
        if m["role"] == "assistant":
            pr = parse_response(m["content"])
            if not pr.strict_valid:
                errs.append(f"{eid}: prior assistant message not schema-valid")
    ref = parse_response(e["reference_response"])
    if not ref.strict_valid or ref.decision != dec or ref.answer != fin:
        errs.append(f"{eid}: reference response inconsistent with labels")
    if e["prior_answer_correct"] is not None and e["prior_kind"] != "none":
        if (prior == before) != e["prior_answer_correct"]:
            errs.append(f"{eid}: prior_answer_correct wrong")
    return errs


def validate(data: dict[str, list[dict]]) -> list[str]:
    errs: list[str] = []
    ids: set[str] = set()
    group_split: dict[str, str] = {}
    fp_split: dict[str, str] = {}
    text_split: dict[str, str] = {}
    pairs: dict[str, list[dict]] = defaultdict(list)
    heldout_ids = T.all_template_ids(heldout=True)
    train_ids = T.all_template_ids(heldout=False)
    from conviction.data.oracles import STRUCTURAL_SUBTYPES

    for split, eps in data.items():
        if split not in SPLITS:
            errs.append(f"unknown split {split}")
        for e in eps:
            errs += _check_episode(e)
            eid = e["episode_id"]
            if eid in ids:
                errs.append(f"duplicate episode_id {eid}")
            ids.add(eid)
            if e["split"] != split:
                errs.append(f"{eid}: split field {e['split']} != file split {split}")
            g = e["group_id"]
            if group_split.setdefault(g, split) != split:
                errs.append(f"group {g} spans splits")
            fp = e["problem"]["fingerprint"]
            if fp_split.setdefault(fp, g) != g:
                errs.append(f"problem fingerprint {fp} reused across groups ({g}, {fp_split[fp]})")
            text = json.dumps(e["messages"], sort_keys=True)
            if text_split.setdefault(text, split) != split:
                errs.append(f"{eid}: identical conversation appears in another split")
            tids = set(e["template_ids"])
            if split == "test_template":
                if tids - heldout_ids:
                    errs.append(f"{eid}: non-held-out template in test_template: {sorted(tids - heldout_ids)}")
            elif tids - train_ids:
                errs.append(f"{eid}: held-out template leaked into {split}: {sorted(tids - train_ids)}")
            is_struct = e["subtype"] in STRUCTURAL_SUBTYPES
            if is_struct != (split == "test_structural"):
                errs.append(f"{eid}: structural-holdout subtype placement wrong")
            if e["pair_id"]:
                pairs[e["pair_id"]].append(e)

    # counterfactual pairs: same problem, same framing; differ in evidence or prior
    for pid, members in pairs.items():
        if len({m["problem"]["fingerprint"] for m in members}) != 1:
            errs.append(f"pair {pid}: members have different problems")
        if len({m["pressure_type"] for m in members}) != 1:
            errs.append(f"pair {pid}: members have different framings")
        if len({(m["evidence_type"], m["prior_answer"]) for m in members}) != len(members):
            errs.append(f"pair {pid}: members are not distinct conditions")
    return errs
