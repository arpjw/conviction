"""Running RunPod budget ledger.

The RunPod account balance is the authoritative record; this ledger is a
planning aid with a deliberate safety margin.

    python -m conviction.utils.budget status
    python -m conviction.utils.budget start --phase sft --gpu "RTX 4090" --rate 0.59 [--storage-rate 0.01]
    python -m conviction.utils.budget stop             # closes the open session
    python -m conviction.utils.budget add --phase storage --cost 0.12 --note "volume, 3 days"
    python -m conviction.utils.budget estimate --rate 0.59 --hours 2.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LEDGER = Path("results/budget_ledger.json")
TOTAL = 40.00
SAFETY_MARGIN = 3.00          # never plan to spend the last $3
PHASE_CAPS = {"setup_baseline": 4.0, "sft": 7.0, "grpo": 15.0, "eval_ablation": 10.0, "storage_reserve": 4.0}


def load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"total_budget": TOTAL, "entries": [], "open": None}


def save(d: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(LEDGER)


def spent(d: dict, now: float | None = None) -> float:
    s = sum(e["cost"] for e in d["entries"])
    if d.get("open"):
        o = d["open"]
        s += (( now or time.time()) - o["start"]) / 3600 * (o["rate"] + o.get("storage_rate", 0.0))
    return s


def by_phase(d: dict) -> dict:
    out: dict[str, float] = {}
    for e in d["entries"]:
        out[e["phase"]] = out.get(e["phase"], 0.0) + e["cost"]
    return out


def status(d: dict) -> str:
    now = time.time()
    s = spent(d, now)
    rem = d["total_budget"] - s
    lines = [f"spent (ledger estimate): ${s:.2f} / ${d['total_budget']:.2f}   remaining: ${rem:.2f}   "
             f"plannable (minus ${SAFETY_MARGIN:.2f} margin): ${rem - SAFETY_MARGIN:.2f}"]
    for ph, cap in PHASE_CAPS.items():
        lines.append(f"  {ph:16s} ${by_phase(d).get(ph, 0.0):6.2f} / cap ${cap:.2f}")
    if d.get("open"):
        o = d["open"]
        el = (now - o["start"]) / 3600
        rate = o["rate"] + o.get("storage_rate", 0.0)
        lines.append(f"OPEN session: phase={o['phase']} gpu={o['gpu']} rate=${rate:.3f}/h elapsed={el:.2f}h "
                     f"(${el * rate:.2f}); affordable remaining runtime at this rate: "
                     f"{max(0.0, (rem - SAFETY_MARGIN) / rate):.1f}h; phase cap leaves "
                     f"{max(0.0, (PHASE_CAPS.get(o['phase'], 0) - by_phase(d).get(o['phase'], 0) - el * rate) / rate):.1f}h")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status")
    s = sp.add_parser("start"); s.add_argument("--phase", required=True); s.add_argument("--gpu", required=True)
    s.add_argument("--rate", type=float, required=True); s.add_argument("--storage-rate", type=float, default=0.0)
    s.add_argument("--note", default="")
    st = sp.add_parser("stop"); st.add_argument("--note", default="")
    a = sp.add_parser("add"); a.add_argument("--phase", required=True); a.add_argument("--cost", type=float, required=True)
    a.add_argument("--note", default="")
    e = sp.add_parser("estimate"); e.add_argument("--rate", type=float, required=True)
    e.add_argument("--hours", type=float, required=True)
    args = ap.parse_args(argv)
    d = load()
    if args.cmd == "start":
        if d.get("open"):
            raise SystemExit("a session is already open; stop it first")
        d["open"] = {"phase": args.phase, "gpu": args.gpu, "rate": args.rate, "storage_rate": args.storage_rate,
                     "start": time.time(), "note": args.note}
    elif args.cmd == "stop":
        o = d.pop("open", None)
        if not o:
            raise SystemExit("no open session")
        hours = (time.time() - o["start"]) / 3600
        d["entries"].append({"phase": o["phase"], "gpu": o["gpu"], "hours": round(hours, 4),
                             "rate": o["rate"] + o.get("storage_rate", 0.0),
                             "cost": round(hours * (o["rate"] + o.get("storage_rate", 0.0)), 4),
                             "start": o["start"], "end": time.time(), "note": f"{o['note']} {args.note}".strip(),
                             "kind": "measured_session"})
        d["open"] = None
    elif args.cmd == "add":
        d["entries"].append({"phase": args.phase, "cost": args.cost, "note": args.note, "time": time.time(),
                             "kind": "manual"})
    elif args.cmd == "estimate":
        c = args.rate * args.hours
        rem = d["total_budget"] - spent(d)
        print(f"estimated ${c:.2f}; remaining after: ${rem - c:.2f} "
              f"({'OK' if rem - c >= SAFETY_MARGIN else 'VIOLATES SAFETY MARGIN'})")
        return 0
    save(d) if args.cmd != "status" else None
    print(status(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
