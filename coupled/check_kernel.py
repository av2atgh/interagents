"""Equivalence check for the compiled inner loop.

network.py runs the network dynamics through the numba kernel in
network_kernel.py when numba is available, and through the original NumPy
implementation otherwise. The two consume random numbers in a different order,
so individual runs differ; what must agree is the ensemble. This script runs
both over the same graphs and compares the coupled-edge fraction and the giant
component.

Reference output (6 realisations, n = 600, LS(l=1), 2e5 steps, R_I = 0.95):

       c    impl      coupled frac                 S              wall
   0.010   numba   0.956 +- 0.005     1.000 +- 0.000            19.0s
   0.010   numpy   0.956 +- 0.004     1.000 +- 0.000           144.1s
   0.020   numba   0.806 +- 0.004     0.898 +- 0.048            20.6s
   0.020   numpy   0.806 +- 0.005     0.836 +- 0.087           114.4s
   0.030   numba   0.115 +- 0.001     0.007 +- 0.001            17.2s
   0.030   numpy   0.113 +- 0.003     0.008 +- 0.001            77.6s
"""

import time

import numpy as np

import network
from network import grow_ls, run_network, giant_fraction, all_edges


def stats(use_kernel, c, seeds, n=600, n_steps=200_000):
    saved = network._run_kernel
    if not use_kernel:
        network._run_kernel = None
    out = []
    try:
        for sd in seeds:
            g = grow_ls(n, 1, seed=sd)
            r = run_network(g, R_I=0.95, c=c, n_steps=n_steps, seed=sd,
                            tau_mem=200.0, burn_frac=0.5)
            ce = {(min(i, j), max(i, j))
                  for i, j, k in zip(r["src"], r["dst"], r["rate"] > 1e-3) if k}
            out.append((len(ce) / len(all_edges(g)), giant_fraction(n, ce)))
    finally:
        network._run_kernel = saved
    return np.array(out)


if __name__ == "__main__":
    seeds = range(6)
    print(f"{'c':>8} {'impl':>7} {'coupled frac':>22} {'S':>22} {'wall':>8}")
    for c in (0.010, 0.020, 0.030):
        for use, lbl in ((True, "numba"), (False, "numpy")):
            t0 = time.time()
            v = stats(use, c, seeds)
            se = lambda col: col.std(ddof=1) / np.sqrt(col.size)
            print(f"{c:>8.3f} {lbl:>7} {v[:,0].mean():>10.3f} +- "
                  f"{se(v[:,0]):<8.3f} {v[:,1].mean():>10.3f} +- "
                  f"{se(v[:,1]):<8.3f} {time.time()-t0:>7.1f}s")
