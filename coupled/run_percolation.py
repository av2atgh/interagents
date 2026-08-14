"""R15: does the solitary phase percolate, and does local structure resist it?

Control parameter: c, the cost of an unreciprocated offer. On a network this is
the natural one -- an agent with k relations can serve only one per step, so
offers go unreciprocated constantly and c sets how punishing that is.

Networks, all at mean degree 4:
  LS(l=1)  triadic closure -- local, every edge in a triangle, emergent
           communities (Ramsey number r_kappa = 81)
  CONF     degree-preserving randomization of LS -- IDENTICAL degrees, locality
           destroyed, no finite Ramsey number
  BA(m=2)  preferential attachment -- non-local control, no communities

LS vs CONF is decisive: same degree sequence, so any difference is local
structure alone. This is the control of Vazquez, *Local Network Growth*,
Ch. 'Emergence of communities'.

An edge counts as coupled when its interaction rate exceeds 10x the solitary
baseline eps^2 = 1e-4. Order parameter: S, the largest connected component of
the coupled subgraph as a fraction of n.
"""

import numpy as np
from network import (grow_ls, grow_ba, randomize_degree_preserving,
                     run_network, giant_fraction, all_edges, directed_index)

N, STEPS, SEEDS = 600, 40_000, (1, 2, 3)
THRESH = 1e-3


def coupled_edges(res):
    keep = res["rate"] > THRESH
    return {(min(i, j), max(i, j))
            for i, j, k in zip(res["src"], res["dst"], keep) if k}


def comp_sizes(edges):
    nb = {}
    for i, j in edges:
        nb.setdefault(i, []).append(j)
        nb.setdefault(j, []).append(i)
    seen, sizes = set(), []
    for s in nb:
        if s in seen:
            continue
        stack, sz = [s], 0
        seen.add(s)
        while stack:
            v = stack.pop()
            sz += 1
            for w in nb[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        sizes.append(sz)
    return sorted(sizes, reverse=True)


def build(name, sd):
    ls = grow_ls(N, 1, seed=sd)
    if name == "LS(l=1)":
        return ls
    if name == "CONF":
        return randomize_degree_preserving(ls, seed=sd)
    return grow_ba(N, 2, seed=sd)


MODELS = ("LS(l=1)", "CONF", "BA(m=2)")
CS = (0.002, 0.005, 0.010, 0.020, 0.030, 0.045, 0.070)

print("=" * 84)
print("R15  PERCOLATION OF THE SOLITARY PHASE   (R_I = 0.95, n = 600, <k> = 4)")
print("=" * 84)
print(f"  {'c':>6} {'model':>9} {'coupled edges':>14} {'S':>7} "
      f"{'2nd comp':>9} {'#comp>=3':>9}")
for c in CS:
    for name in MODELS:
        fe, gs, s2, nc = [], [], [], []
        for sd in SEEDS:
            g = build(name, sd)
            res = run_network(g, R_I=0.95, c=c, n_steps=STEPS, seed=sd)
            ce = coupled_edges(res)
            fe.append(len(ce) / len(all_edges(g)))
            gs.append(giant_fraction(N, ce))
            sz = comp_sizes(ce)
            s2.append(sz[1] if len(sz) > 1 else 0)
            nc.append(sum(1 for s in sz if s >= 3))
        print(f"  {c:>6.3f} {name:>9} {np.mean(fe):>14.3f} {np.mean(gs):>7.3f} "
              f"{np.mean(s2):>9.1f} {np.mean(nc):>9.1f}")
    print()

print("=" * 84)
print("  Survival by degree at c = 0.020: attention is scarce, so a hub must")
print("  divide it. p_hat ~ P(active)/k, so hub relations should fail first.")
print("=" * 84)
for name in ("LS(l=1)", "CONF"):
    bins = {}
    for sd in SEEDS:
        g = build(name, sd)
        res = run_network(g, R_I=0.95, c=0.020, n_steps=STEPS, seed=sd)
        ce = coupled_edges(res)
        deg = np.array([len(x) for x in g])
        cd = np.zeros(N)
        for i, j in ce:
            cd[i] += 1
            cd[j] += 1
        for i in range(N):
            b = 2 ** int(np.log2(max(deg[i], 1)))
            bins.setdefault(b, []).append(cd[i] / max(deg[i], 1))
    print(f"  {name:>9}: " + "  ".join(
        f"k~{k}:{np.mean(v):.3f}" for k, v in sorted(bins.items())
        if len(v) >= 15))
