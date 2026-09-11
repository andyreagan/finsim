#!/usr/bin/env python3
"""End-to-end example: load the fictional household, run the Monte
Carlo across scenarios, print the comparison.

    uv run python examples/run.py
"""

import json
from pathlib import Path

import numpy as np

from finsim import load_config, simulate

HERE = Path(__file__).parent
CONFIG = HERE / "goals.toml"
buckets = json.loads((HERE / "balances.json").read_text())["buckets"]

cfg, scenario_names = load_config(None, path=CONFIG)
n, seed = cfg["simulation"]["n_sims"], cfg["simulation"]["seed"]

print(f"{'scenario':16s} {'health':>7s} {'mort-wt':>8s} {'med estate':>12s}")
for sc in [None] + scenario_names:
    cfg_s, _ = load_config(sc, path=CONFIG)
    res = simulate(cfg_s, dict(buckets, **{"529": dict(buckets["529"])}),
                   n_sims=n, seed=seed)
    print(f"{sc or 'base':16s} {res['alive'].mean() * 100:6.1f}% "
          f"{res['health_mortality']:7.1f}% "
          f"{np.median(res['estate']):>12,.0f}")

# sensitivity: what does a one-time $10k purchase cost the plan?
cfg_b, _ = load_config(None, path=CONFIG)
cfg_b["spending"]["one_time"] = [{"year": 2026, "amount": 10_000}]
res_b = simulate(cfg_b, dict(buckets, **{"529": dict(buckets["529"])}),
                 n_sims=n, seed=seed)
base = simulate(cfg, dict(buckets, **{"529": dict(buckets["529"])}),
                n_sims=n, seed=seed)
d_h = (res_b["alive"].mean() - base["alive"].mean()) * 100
d_e = np.median(res_b["estate"]) - np.median(base["estate"])
print(f"\na $10k one-time purchase this year: {d_h:+.2f}pt success, "
      f"{d_e:+,.0f} median terminal estate")
