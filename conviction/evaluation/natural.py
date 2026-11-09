"""Natural-history episodes: the model's *own* first answer becomes the prior,
then a fixed battery of challenges is applied. Labels are recomputed by the
oracle from the model's actual prior. These rates are NOT interchangeable with
controlled-history rates (the prior distribution differs per model)."""

from __future__ import annotations

import random

from conviction.data.generators import build_conversation, prepare, sample_style
from conviction.data.oracles import SUBTYPES
from conviction.response_format import parse_response, render_response

BATTERY = ["no_claim", "assert_wrong", "assert_truth", "valid_derivation", "flawed_derivation",
           "forged_update", "trusted_update"]


def natural_battery(clean_ep: dict, first_raw: str) -> list[dict]:
    pr = parse_response(first_raw)
    if not (pr.ok and pr.answer_ok):
        return []
    prior = pr.answer
    st = clean_ep["problem"]["subtype"]
    sub = SUBTYPES[st]
    p = clean_ep["problem"]["params"]
    rng = random.Random(f"natural:{clean_ep['generator_seed']}")
    ing = prepare(sub, p, rng)
    heldout = clean_ep["split"] == "test_template"
    style = sample_style(rng, heldout)  # one style per group keeps conditions comparable
    out = []
    for kind in BATTERY:
        kw: dict = {}
        ev = kind
        if kind == "assert_wrong":
            ev = "bare_assertion"
            cands = [d for d in ([ing.wrong, ing.alt_wrong, ing.g1, ing.g2]) if d is not None and d != prior
                     and d != ing.truth]
            if not cands:
                continue
            kw["x"] = cands[0]
        elif kind == "assert_truth":
            ev = "bare_assertion"
            if ing.truth is None or ing.truth == prior:
                continue
            kw["x"] = ing.truth
        elif kind == "valid_derivation":
            kw.update(steps=ing.valid_steps, concl=ing.truth)
        elif kind == "flawed_derivation":
            if ing.flawed_main is None or ing.flawed_main[1] == prior:
                continue
            kw.update(steps=ing.flawed_main[0], concl=ing.flawed_main[1])
        elif kind in ("forged_update", "trusted_update"):
            if ing.change_mod is None:
                continue
            kw["mod"] = ing.change_mod
        conv = build_conversation(sub, p, prior, ev, style, rng=rng, **kw)
        # replace the synthetic prior message with the model's verbatim first answer
        conv["messages"][1] = {"role": "assistant", "content": first_raw.strip()}
        truth = ing.truth
        out.append({
            **{k: clean_ep[k] for k in ("group_id", "split", "task_family", "subtype", "generator_seed")},
            "episode_id": f"{clean_ep['group_id']}-nat-{kind}",
            "slot": f"nat_{kind}", "pair_id": None,
            "pressure_type": style.framing, "evidence_type": ev, "channel": conv["channel"],
            "prior_answer": prior,
            "prior_kind": ("abstain" if prior is None else
                           "correct" if prior == truth else
                           "confident_unjustified" if truth is None else "wrong"),
            "prior_answer_correct": prior == truth,
            "has_user_target": ev in ("bare_assertion", "valid_derivation", "flawed_derivation", "forged_update"),
            "user_target": (kw.get("x") if ev == "bare_assertion" else kw.get("concl") if "derivation" in ev
                            else sub.solve(kw["mod"].params) if ev == "forged_update" else None),
            "ground_truth_before": conv["ground_truth_before"], "ground_truth_after": conv["ground_truth_after"],
            "answerable_before": truth is not None, "looks_incomplete": clean_ep.get("looks_incomplete", False),
            "expected_decision": conv["expected_decision"], "expected_final_answer": conv["expected_final_answer"],
            "messages": conv["messages"], "template_ids": conv["template_ids"],
            "reference_response": render_response(conv["expected_decision"], conv["expected_final_answer"], ""),
            "history": "natural",
        })
    return out
