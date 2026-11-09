"""Generate, validate and write ConvictionBench.

    uv run python -m conviction.data.cli --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from conviction.data.splits import DataConfig, generate_dataset, write_dataset
from conviction.data.validation import validate


def write_examples(data: dict, path: Path, n_per_split: int = 6, seed: int = 0) -> None:
    rng = random.Random(seed)
    lines = ["# ConvictionBench — random examples (for manual inspection)\n"]
    for split, eps in data.items():
        lines.append(f"\n## {split}\n")
        for e in rng.sample(eps, min(n_per_split, len(eps))):
            lines.append(f"\n### {e['episode_id']}\n")
            lines.append(f"`{e['subtype']}` · framing=`{e['pressure_type']}` · evidence=`{e['evidence_type']}` · "
                         f"prior={e['prior_answer']!r} ({e['prior_kind']}) → expected "
                         f"**{e['expected_decision']} {e['expected_final_answer']!r}**\n")
            for m in e["messages"]:
                body = m["content"].replace("\n", "\n> ")
                lines.append(f"> **{m['role']}**: {body}\n>")
            lines.append(f"\nReference: `{e['reference_response']}`\n")
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg_all = yaml.safe_load(open(args.config))
    dcfg = DataConfig.from_dict(cfg_all["data"])
    out = Path(args.out or cfg_all["data"]["out_dir"])
    data = generate_dataset(dcfg)
    errs = validate(data)
    if errs:
        print(f"VALIDATION FAILED ({len(errs)} errors):", *errs[:30], sep="\n  ", file=sys.stderr)
        return 1
    manifest = write_dataset(data, out, dcfg)
    write_examples(data, out / "EXAMPLES.md")
    print(json.dumps({s: {k: v for k, v in m.items() if k in ("episodes", "groups", "by_decision")}
                      for s, m in manifest["splits"].items()}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
