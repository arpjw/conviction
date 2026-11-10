"""Pre-registered go/no-go gate for E3b, applied to the 3-step smoke run."""
import json, sys
t = json.load(open(sys.argv[1]))
rows = [h for h in t["log_history"] if "frac_reward_zero_std" in h]
z = sum(h["frac_reward_zero_std"] for h in rows) / len(rows)
c = sum(h.get("completions/clipped_ratio", 0.0) for h in rows) / len(rows)
r = sum(h["reward"] for h in rows) / len(rows)
print(f"GATE zero_std={z:.3f} clipped={c:.3f} reward={r:.3f}")
if z > 0.70 or c > 0.20:
    print("GATE_FAIL"); sys.exit(1)
print("GATE_PASS")
