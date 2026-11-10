"""Stratified sampling of whole problem groups (shared by the HF and API runners)."""

from __future__ import annotations

import random
from collections import defaultdict


def sample_groups(eps: list[dict], n_groups: int | None, seed: int) -> list[dict]:
    """Keep every episode of ``n_groups`` problem groups, stratified round-robin across subtypes
    (whole groups are kept so cluster-bootstrap intervals stay valid). ``n_groups`` falsy = all."""
    if not n_groups:
        return eps
    by_sub: dict[str, list[str]] = defaultdict(list)
    for gid, sub in sorted({(e["group_id"], e["subtype"]) for e in eps}):
        by_sub[sub].append(gid)
    rng = random.Random(seed)
    for gids in by_sub.values():
        rng.shuffle(gids)
    chosen: list[str] = []
    subs = sorted(by_sub)
    while len(chosen) < n_groups and any(by_sub.values()):
        for s in subs:
            if by_sub[s] and len(chosen) < n_groups:
                chosen.append(by_sub[s].pop())
    keep = set(chosen)
    return [e for e in eps if e["group_id"] in keep]
