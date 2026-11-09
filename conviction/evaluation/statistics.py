"""Cluster bootstrap (resampling matched groups, not episodes) for point
estimates, 95% CIs and paired differences between two models evaluated on
the same episodes."""

from __future__ import annotations

import numpy as np

from conviction.evaluation.metrics import METRICS, Metric


def _group_counts(records: list[dict], m: Metric, groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {g: i for i, g in enumerate(groups)}
    k = np.zeros(len(groups)); n = np.zeros(len(groups))
    for r in records:
        if m.filt(r):
            i = idx[r["group_id"]]
            n[i] += 1
            k[i] += bool(m.success(r))
    return k, n


def bootstrap(records: list[dict], metrics: list[Metric] = METRICS, n_boot: int = 2000,
              seed: int = 0) -> dict[str, dict]:
    groups = sorted({r["group_id"] for r in records})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    out = {}
    for m in metrics:
        k, n = _group_counts(records, m, groups)
        if n.sum() == 0:
            out[m.name] = {"value": None, "lo": None, "hi": None, "n": 0}
            continue
        K, N = k[draws].sum(1), n[draws].sum(1)
        vals = K[N > 0] / N[N > 0]
        out[m.name] = {"value": k.sum() / n.sum(), "lo": float(np.percentile(vals, 2.5)),
                       "hi": float(np.percentile(vals, 97.5)), "n": int(n.sum()),
                       "clusters": int((n > 0).sum())}
    return out


def paired_bootstrap(rec_a: list[dict], rec_b: list[dict], metrics: list[Metric] = METRICS,
                     n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Difference B − A with group-clustered CIs. Both record lists must cover the
    same episodes (same evaluation set)."""
    ids_a = {r["episode_id"] for r in rec_a}
    ids_b = {r["episode_id"] for r in rec_b}
    if ids_a != ids_b:
        raise ValueError("paired bootstrap requires identical episode sets")
    groups = sorted({r["group_id"] for r in rec_a})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    out = {}
    for m in metrics:
        ka, na = _group_counts(rec_a, m, groups)
        kb, nb = _group_counts(rec_b, m, groups)
        if na.sum() == 0 or nb.sum() == 0:
            out[m.name] = {"diff": None}
            continue
        Ka, Na, Kb, Nb = ka[draws].sum(1), na[draws].sum(1), kb[draws].sum(1), nb[draws].sum(1)
        ok = (Na > 0) & (Nb > 0)
        d = Kb[ok] / Nb[ok] - Ka[ok] / Na[ok]
        diff = kb.sum() / nb.sum() - ka.sum() / na.sum()
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        # two-sided bootstrap p-value for H0: diff = 0 (percentile method)
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        out[m.name] = {"diff": diff, "lo": lo, "hi": hi, "p_boot": p}
    return out
