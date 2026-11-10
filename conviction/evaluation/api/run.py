"""conviction-eval-api: evaluate a provider-API model on a stratified sample of ConvictionBench.

    # 1. free cost estimate (no API calls, no key needed)
    conviction-eval-api --provider anthropic --model claude-opus-5-5 --groups 90 --natural \
        --price-in 4 --price-out 20 --dry-run --out results/A_claude-opus-5-5
    # 2. paid run with a hard spending cap (keys come from the environment)
    conviction-eval-api ... --budget-usd 15 --out results/A_claude-opus-5-5

Protocol (the "API track"; see research/EXPERIMENTS.md): the same base system prompt and response
format as every other model; tool records sent through the provider's native tool-call format;
reasoning left at each model's default unless --effort is given; no sampling parameters unless
--temperature is given (and accepted by the provider). Every request and response is cached on disk
(``<out>/cache.jsonl``), so an interrupted run resumes without paying twice.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from conviction.data.splits import load_split
from conviction.evaluation import metrics as M
from conviction.evaluation.api import convert
from conviction.evaluation.api.providers import PROVIDERS, make_provider
from conviction.evaluation.natural import natural_battery
from conviction.evaluation.runner import BREAKDOWN_KEYS, BREAKDOWN_METRICS, evaluate_records
from conviction.evaluation.statistics import bootstrap
from conviction.evaluation.sampling import sample_groups  # noqa: F401  (re-exported for tests)
from conviction.prompts import build_messages


class BudgetExceeded(RuntimeError):
    pass


class Cache:
    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                self.data[r["key"]] = r
        self.f = open(path, "a")

    @staticmethod
    def key(params: dict) -> str:
        return hashlib.sha256(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, k: str):
        return self.data.get(k)

    def put(self, k: str, rec: dict) -> None:
        rec = {"key": k, **rec}
        self.data[k] = rec
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()


class Spend:
    def __init__(self, price_in: float, price_out: float, budget: float | None):
        self.price_in, self.price_out, self.budget = price_in, price_out, budget
        self.new_in = self.new_out = self.cached = self.calls = 0
        self.errors: list[str] = []
        self.stops: dict[str, int] = defaultdict(int)

    @property
    def usd(self) -> float:
        return (self.new_in * self.price_in + self.new_out * self.price_out) / 1e6

    def check(self) -> None:
        if self.budget is not None and self.usd >= self.budget:
            raise BudgetExceeded(f"budget ${self.budget:.2f} reached (spent ${self.usd:.2f})")


async def run_batch(provider, items: list[tuple[dict, dict]], cache: Cache, spend: Spend, concurrency: int,
                    label: str) -> list[str]:
    """items: (episode, request params). Returns raw texts in order."""
    sem = asyncio.Semaphore(concurrency)
    out: list[str | None] = [None] * len(items)
    done = 0
    t0 = time.time()

    async def one(i: int, ep: dict, params: dict) -> None:
        nonlocal done
        k = Cache.key(params)
        hit = cache.get(k)
        if hit is None:
            async with sem:
                spend.check()
                c = await provider.complete(params, ep)
                spend.new_in += c.input_tokens
                spend.new_out += c.output_tokens
                spend.calls += 1
                spend.stops[c.stop] += 1
                hit = {"text": c.text, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                       "stop": c.stop, "meta": c.meta, "episode_id": ep["episode_id"]}
                cache.put(k, hit)
        else:
            spend.cached += 1
        out[i] = hit["text"]
        done += 1
        if done % 50 == 0 or done == len(items):
            print(f"  {label}: {done}/{len(items)}  new calls {spend.calls}  spent ${spend.usd:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    await asyncio.gather(*(one(i, ep, p) for i, (ep, p) in enumerate(items)))
    return out  # type: ignore[return-value]


def request_for(provider, ep: dict) -> dict:
    system, turns = convert.split_system(build_messages(ep["messages"], "base"))
    return provider.request_params(system, turns)


def estimate(provider, eps: list[dict], natural_factor: float, price_in: float, price_out: float,
             est_out: tuple[int, int]) -> dict:
    """Token estimate from request size (≈ 4 characters per token). Natural-history episodes are
    approximated as ``natural_factor`` × the clean episodes, with the same average size."""
    chars = [len(json.dumps(request_for(provider, e), ensure_ascii=False)) for e in eps]
    n_clean = sum(e["slot"] == "clean" for e in eps)
    n = len(eps) + int(n_clean * natural_factor)
    tin = sum(chars) / 4 / len(eps) * n
    lo, hi = (tin * price_in + n * est_out[0] * price_out) / 1e6, (tin * price_in + n * est_out[1] * price_out) / 1e6
    return {"requests": n, "input_tokens": int(tin), "output_tokens_range": [n * est_out[0], n * est_out[1]],
            "usd_range": [round(lo, 2), round(hi, 2)]}


async def _evaluate(provider, split_eps, args, cache, spend, out, all_metrics, timing) -> None:
    for split, eps in split_eps.items():
        t0 = time.time()
        raws = await run_batch(provider, [(e, request_for(provider, e)) for e in eps], cache, spend,
                               args.concurrency, split)
        recs = [M.score_record(e, r) for e, r in zip(eps, raws)]
        _write_jsonl(out / f"predictions_{split}.jsonl", recs)
        all_metrics[split] = evaluate_records(recs, args.n_boot)
        json.dump({k: M.breakdown(recs, k, BREAKDOWN_METRICS) for k in BREAKDOWN_KEYS},
                  open(out / f"breakdown_{split}.json", "w"), indent=1)
        if args.natural:
            clean = {r["group_id"]: (e, r["raw"]) for e, r in zip(eps, recs) if e["slot"] == "clean"}
            nat = [x for e, raw in clean.values() for x in natural_battery(e, raw)]
            if nat:
                nraws = await run_batch(provider, [(e, request_for(provider, e)) for e in nat], cache, spend,
                                        args.concurrency, f"{split}_natural")
                nrecs = [M.score_record(e, r) for e, r in zip(nat, nraws)]
                _write_jsonl(out / f"predictions_{split}_natural.jsonl", nrecs)
                all_metrics[f"{split}_natural"] = {"all": bootstrap(nrecs, n_boot=args.n_boot)}
        timing[split] = {"episodes": len(eps), "seconds": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=PROVIDERS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None, help="override the provider's API base URL")
    ap.add_argument("--effort", default=None, help="reasoning effort (Anthropic output_config.effort / OpenAI reasoning_effort)")
    ap.add_argument("--temperature", type=float, default=None, help="only for providers/models that accept it")
    ap.add_argument("--extra-json", default=None, help="extra request fields as JSON, merged last (per-model quirks)")
    ap.add_argument("--max-output-tokens", type=int, default=16000, help="includes reasoning tokens on most providers")
    ap.add_argument("--data-dir", default="data/conviction_bench/v0.1")
    ap.add_argument("--splits", default="test_structural")
    ap.add_argument("--groups", type=int, default=90, help="problem groups sampled per split (0 = all)")
    ap.add_argument("--sample-seed", type=int, default=20261110)
    ap.add_argument("--natural", action="store_true", help="also run the natural-history battery")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--price-in", type=float, required=True, help="USD per 1M input tokens (from the provider's pricing page)")
    ap.add_argument("--price-out", type=float, required=True, help="USD per 1M output tokens")
    ap.add_argument("--budget-usd", type=float, default=None, help="hard cap on new spend for this run")
    ap.add_argument("--dry-run", action="store_true", help="estimate requests, tokens and cost; make no API calls")
    ap.add_argument("--est-output-tokens", default="120,1500",
                    help="low,high output tokens per request for the dry-run estimate (high covers reasoning)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if not args.dry_run and args.provider != "mock" and args.budget_usd is None:
        raise SystemExit("--budget-usd is required for paid runs")
    extra = json.loads(args.extra_json) if args.extra_json else None
    # dry runs size requests offline (no client, no key); paid runs build the real provider
    provider = make_provider("mock", args.model, 0, None, None, None, None) if args.dry_run else \
        make_provider(args.provider, args.model, args.max_output_tokens, args.effort, args.temperature,
                      args.base_url, extra)

    split_eps = {}
    for split in args.splits.split(","):
        eps = load_split(Path(args.data_dir), split)
        split_eps[split] = sample_groups(eps, args.groups, args.sample_seed)

    if args.dry_run:
        lo, hi = (int(x) for x in args.est_output_tokens.split(","))
        report = {s: estimate(provider, e, 7.0 if args.natural else 0.0, args.price_in, args.price_out, (lo, hi))
                  for s, e in split_eps.items()}
        tot = [sum(r["usd_range"][i] for r in report.values()) for i in (0, 1)]
        print(json.dumps({"model": args.model, "per_split": report, "total_usd_range": [round(t, 2) for t in tot]}, indent=1))
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(out / "cache.jsonl")
    spend = Spend(args.price_in, args.price_out, args.budget_usd)
    t_start = time.time()
    all_metrics, timing = {}, {}
    try:
        asyncio.run(_evaluate(provider, split_eps, args, cache, spend, out, all_metrics, timing))
    except BudgetExceeded as e:
        _write_info(out, args, spend, timing, t_start, status=f"STOPPED: {e}")
        print(f"STOPPED: {e}. Cached responses are kept; re-run with a higher --budget-usd to resume.")
        return 2

    rows = []
    for split, d in all_metrics.items():
        for view, ms in d.items():
            for name, v in ms.items():
                rows.append({"split": split, "view": view, "metric": name, **v})
    json.dump(all_metrics, open(out / "metrics.json", "w"), indent=1)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "view", "metric", "value", "lo", "hi", "n", "clusters"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    _write_info(out, args, spend, timing, t_start, status="complete")
    for split, d in all_metrics.items():
        a = d["all"]
        print(split, {k: (round(v["value"], 3) if v["value"] is not None else None) for k, v in a.items()
                      if k in ("final_accuracy", "pressure_flip_rate", "legitimate_correction_rate",
                               "fabricated_evidence_resistance", "strict_format_rate")})
    print(f"new spend ${spend.usd:.2f} ({spend.calls} calls, {spend.cached} cached)")
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_info(out: Path, args, spend: Spend, timing: dict, t_start: float, status: str) -> None:
    info = {"track": "api", "status": status, "provider": args.provider, "model": args.model,
            "base_url": args.base_url, "effort": args.effort, "temperature": args.temperature,
            "extra_request_fields": json.loads(args.extra_json) if args.extra_json else None,
            "max_output_tokens": args.max_output_tokens, "condition": "base", "splits": args.splits,
            "groups_per_split": args.groups, "sample_seed": args.sample_seed, "natural": args.natural,
            "data_dir": args.data_dir, "tool_channel": "native provider tool calls (convert.py)",
            "prices_usd_per_mtok": {"input": args.price_in, "output": args.price_out},
            "usage_new": {"input_tokens": spend.new_in, "output_tokens": spend.new_out, "calls": spend.calls,
                          "cached_responses": spend.cached, "usd": round(spend.usd, 4)},
            "stop_reasons_new": dict(spend.stops), "timing": timing, "total_seconds": time.time() - t_start,
            "python": platform.python_version()}
    try:
        import anthropic, openai  # noqa: E401
        info["sdk"] = {"anthropic": anthropic.__version__, "openai": openai.__version__}
    except ImportError:
        pass
    json.dump(info, open(out / "run_info.json", "w"), indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
