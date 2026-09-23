import json
import sys

import numpy as np

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results_bimodality.json"))
rows = D["base"]
phi = np.array([r["phi"] for r in rows])
N = phi.size


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


print(f"N = {N} realisations, {D['n_steps']:,} steps, occupancy on final 20%")
for lbl, sel in (("phi < 0.25", phi < 0.25),
                 ("0.25 <= phi <= 0.75", (phi >= 0.25) & (phi <= 0.75)),
                 ("phi > 0.75", phi > 0.75)):
    k = int(sel.sum())
    lo, hi = wilson(k, N)
    print(f"  {lbl:>22}: {k:>5} ({k/N:.3f}, 95% CI {lo:.3f}-{hi:.3f})")
coupled = phi > 0.5
print(f"  reached coupled phase: {coupled.mean():.3f}; majority baseline "
      f"{max(coupled.mean(), 1-coupled.mean()):.3f}")

print("\nPredictability of the final phase from a trajectory prefix")
print(f"  {'prefix':>9} {'in-sample':>11} {'split-sample':>22}")
rng = np.random.default_rng(0)
for t in D["checks"]:
    x = np.array([r["prefix"][str(t)] for r in rows])

    def best_acc(xi, yi):
        cands = np.unique(np.round(xi, 4))
        best = 0.0
        thr = None
        for c in cands:
            for sgn in (1, -1):
                acc = (((xi > c) if sgn > 0 else (xi <= c)) == yi).mean()
                if acc > best:
                    best, thr = acc, (c, sgn)
        return best, thr

    ins, _ = best_acc(x, coupled)
    accs = []
    for _ in range(400):
        m = rng.random(N) < 0.5
        if m.sum() < 10 or (~m).sum() < 10:
            continue
        _, thr = best_acc(x[m], coupled[m])
        c, sgn = thr
        pred = (x[~m] > c) if sgn > 0 else (x[~m] <= c)
        accs.append((pred == coupled[~m]).mean())
    accs = np.array(accs)
    print(f"  {t:>9} {ins:>11.3f} {accs.mean():>14.3f} +- {accs.std():.3f}")

print("\nRecovery of the collapsed realisations")
coll = set(D["collapsed"])
print(f"  {len(coll)} collapsed realisations")
by = {}
for r in D["rescue"]:
    by.setdefault(r["mode"], []).append(r["phi"])
for mode, lbl in (("control", "none (control)"), ("A2", "agent A -> level 2"),
                  ("both2", "both -> level 2"),
                  ("lvl2_t0", "level 2 from t=0")):
    v = np.array(by[mode])
    k = int((v > 0.5).sum())
    lo, hi = wilson(k, v.size)
    print(f"  {lbl:>22}: recovered {k}/{v.size} = {k/v.size:.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f})  mean final phi = {v.mean():.3f} "
          f"+- {v.std(ddof=1)/np.sqrt(v.size):.3f}")
