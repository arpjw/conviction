"""Training-set selection. SFT and GRPO draw from *disjoint groups* of the
train split (hash buckets), so GRPO optimises on problems SFT never saw."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

BUCKETS = {"sft": (0, 50), "grpo": (50, 100)}  # percent ranges of hash(group_id)


def bucket_of(group_id: str) -> int:
    return int(hashlib.sha1(group_id.encode()).hexdigest(), 16) % 100


def select(episodes: list[dict], bucket: str, n: int | None, seed: int,
           one_per_group: bool = False, include_clean: bool = True) -> list[dict]:
    """Select up to n episodes from a bucket, stratified by slot so every
    behaviour is represented. ``one_per_group`` (E5 ablation) drops counterfactual
    siblings: at most one episode per underlying problem."""
    lo, hi = BUCKETS[bucket]
    pool = [e for e in episodes if lo <= bucket_of(e["group_id"]) < hi
            and (include_clean or e["slot"] != "clean")]
    rng = random.Random(seed)
    if one_per_group:
        by_g = defaultdict(list)
        for e in pool:
            by_g[e["group_id"]].append(e)
        pool = [rng.choice(v) for _, v in sorted(by_g.items())]
    by_slot = defaultdict(list)
    for e in pool:
        by_slot[e["slot"]].append(e)
    for v in by_slot.values():
        rng.shuffle(v)
    if n is None or n >= len(pool):
        out = list(pool)
    else:  # round-robin over slots
        out, slots = [], sorted(by_slot)
        while len(out) < n:
            progressed = False
            for s in slots:
                if by_slot[s] and len(out) < n:
                    out.append(by_slot[s].pop()); progressed = True
            if not progressed:
                break
    rng.shuffle(out)
    return out
