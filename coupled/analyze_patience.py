"""Lookahead and endogenous commitment, on the shared collapsed set."""
import json
import sys
from collections import defaultdict

import numpy as np

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results_patience.json"))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


N = len(D["collapsed"])
print(f"collapsed set: N = {N}\n")

print("A  pure lookahead at k = 1")
g = defaultdict(list)
for r in D["lookahead"]:
    g[(r["H"], r["gamma"])].append(r["phi"])
print(f"  {'H':>6} {'gamma':>7} {'recovered':>24} {'mean phi':>10}")
for k in sorted(g):
    v = np.array(g[k])
    rec = int((v > 0.5).sum())
    lo, hi = wilson(rec, v.size)
    print(f"  {k[0]:>6} {k[1]:>7.3f} {rec:>4}/{v.size} = {rec/v.size:.3f} "
          f"({lo:.3f}-{hi:.3f}) {v.mean():>10.3f}")

print("\nB  endogenous commitment vs a fixed k = 1 control")
g = defaultdict(list)
for r in D["endogenous"]:
    g[(r["gamma"], r["mode"])].append(r)
print(f"  {'gamma':>7} {'H':>7} {'mode':>11} {'recovered (phi>1/2)':>26} "
      f"{'mean phi':>9} {'mean rate':>10} {'payoff A':>9} {'payoff B':>9}")
for gam in sorted({k[0] for k in g}):
    for mode in ("fixed", "endogenous"):
        rs = g[(gam, mode)]
        v = np.array([r["phi"] for r in rs])
        rate = np.array([r["rate"] for r in rs])
        pa = np.array([r["payoff_A"] for r in rs])
        pb = np.array([r["payoff_B"] for r in rs])
        rec = int((v > 0.5).sum())
        lo, hi = wilson(rec, v.size)
        print(f"  {gam:>7.3f} {1/(1-gam):>7.0f} {mode:>11} "
              f"{rec:>4}/{v.size} = {rec/v.size:.3f} ({lo:.3f}-{hi:.3f}) "
              f"{v.mean():>9.3f} {rate.mean():>10.4f} {pa.mean():>9.3f} "
              f"{pb.mean():>9.3f}")

print("\n  recovery judged by interaction rate above ten times the eps^2 floor")
for gam in sorted({k[0] for k in g}):
    for mode in ("fixed", "endogenous"):
        rs = g[(gam, mode)]
        rate = np.array([r["rate"] for r in rs])
        rec = int((rate > 1e-3).sum())
        lo, hi = wilson(rec, rate.size)
        print(f"  {gam:>7.3f} {mode:>11} {rec:>4}/{rate.size} = "
              f"{rec/rate.size:.3f} ({lo:.3f}-{hi:.3f})")
