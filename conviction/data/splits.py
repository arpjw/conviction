"""Group-level split assignment and dataset assembly.

Splits
------
train / val / test_id : in-distribution subtypes, training template pools.
                        Assigned per *group* (all matched episodes stay together).
test_template         : in-distribution subtypes, held-out template pools and the
                        held-out ``flattery`` framing (Test B).
test_structural       : structurally held-out subtypes (arithmetic.modular,
                        logic.sets, miniworld.multiplier), training templates (Test C).

Problem fingerprints are globally unique, so no problem instance occurs in two
splits.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from conviction.data.generators import BENCHMARK_VERSION, generate_group
from conviction.data.oracles import IN_DIST_SUBTYPES, STRUCTURAL_SUBTYPES, SUBTYPES

SPLITS = ["train", "val", "test_id", "test_template", "test_structural"]
TEST_SPLITS = ["test_id", "test_template", "test_structural"]


@dataclass
class DataConfig:
    seed: int = 20261109
    n_indist_groups: int = 1800
    n_template_groups: int = 200
    n_structural_groups: int = 200
    indist_fractions: dict = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test_id": 0.1})
    p_under: float = 0.3

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _by_family(keys: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k in keys:
        out.setdefault(SUBTYPES[k].family, []).append(k)
    return out


def plan_groups(cfg: DataConfig) -> list[tuple[str, str, str, bool]]:
    """Return a list of (group_id, split, subtype, heldout_templates). Families are
    balanced first, then subtypes within family."""
    rng = random.Random(f"plan:{cfg.seed}")
    fams = _by_family(IN_DIST_SUBTYPES)
    fam_names = sorted(fams)
    plan = []
    names, weights = zip(*cfg.indist_fractions.items())
    for i in range(cfg.n_indist_groups):
        fam = fam_names[i % len(fam_names)]
        st = rng.choice(sorted(fams[fam]))
        split = rng.choices(names, weights)[0]
        plan.append((f"g{len(plan):06d}", split, st, False))
    for i in range(cfg.n_template_groups):
        fam = fam_names[i % len(fam_names)]
        st = rng.choice(sorted(fams[fam]))
        plan.append((f"g{len(plan):06d}", "test_template", st, True))
    sfams = _by_family(STRUCTURAL_SUBTYPES)
    sf = sorted(sfams)
    for i in range(cfg.n_structural_groups):
        st = rng.choice(sorted(sfams[sf[i % len(sf)]]))
        plan.append((f"g{len(plan):06d}", "test_structural", st, False))
    return plan


def generate_dataset(cfg: DataConfig) -> dict[str, list[dict]]:
    seen: set[str] = set()
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for gid, split, st, heldout in plan_groups(cfg):
        eps = generate_group(group_id=gid, split=split, subtype=st, seed=f"{cfg.seed}:{gid}",
                             p_under=cfg.p_under, heldout_templates=heldout, seen=seen)
        out[split].extend(eps)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset(data: dict[str, list[dict]], out_dir: Path, cfg: DataConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"benchmark_version": BENCHMARK_VERSION, "config": cfg.__dict__, "splits": {}}
    for split, eps in data.items():
        path = out_dir / f"{split}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for e in eps:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.replace(path)
        manifest["splits"][split] = {
            "file": path.name,
            "sha256": _sha256(path),
            "episodes": len(eps),
            "groups": len({e["group_id"] for e in eps}),
            "by_decision": dict(Counter(e["expected_decision"] for e in eps)),
            "by_family": dict(Counter(e["task_family"] for e in eps)),
            "by_evidence": dict(Counter(e["evidence_type"] for e in eps)),
            "by_pressure": dict(Counter(e["pressure_type"] for e in eps)),
        }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_split(data_dir: Path, split: str) -> list[dict]:
    with open(Path(data_dir) / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]
