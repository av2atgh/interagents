"""Is the network coupled phase sustained without exploration?

Equation (kc) says a coupled state exists whenever the degree is below k_c, and
at c = 5e-4 that bound is k_c = 450, so a network of mean degree four is nowhere
near it. The dynamics disagrees: at epsilon = 0 the coupled phase decays away at
every cost tested, because each relation has a finite lifetime and nothing
brings it back. At epsilon >= 1e-3 the same runs reach a stationary state.

This is the network counterpart of the pair result that the solitary state is
absorbing at epsilon = 0, and it is the reason the mean-field threshold is
wrong: what is stationary is the balance between an escape rate and a
return rate, and the mean-field calculation contains neither.

Reference output (n = 600, randomized LS, 2e5 steps, 3 realisations, and the
time series at n = 300 in windows of 400 steps):

  eps=0      c=0.0005: f=0.000 S=0.000   c=0.002: f=0.000 S=0.000
             c=0.005:  f=0.000 S=0.000   c=0.010: f=0.000 S=0.000
  eps=0.001  c=0.0005: f=0.957 S=1.000   c=0.002: f=0.967 S=1.000
             c=0.005:  f=0.918 S=1.000   c=0.010: f=0.173 S=0.041
  eps=0.003  c=0.0005: f=0.960 S=1.000   c=0.002: f=0.974 S=1.000
             c=0.005:  f=0.961 S=1.000   c=0.010: f=0.875 S=0.998

  executed-edge fraction, x1e3, per 400-step window, c = 5e-4:
  eps=0      3.97 2.10 1.97 2.27 2.41 2.54 2.73 2.77 2.33 2.16 ... 0.71
  eps=0.001  4.66 2.70 2.95 2.91 3.43 3.94 3.97 4.11 3.66 3.97 ... 4.30
"""

import numpy as np

from network import (grow_ls, randomize_degree_preserving, run_network,
                     giant_fraction, all_edges)

CS = (0.0005, 0.002, 0.005, 0.010)
EPS = (0.0, 0.001, 0.003)

if __name__ == "__main__":
    print("Coupled edge fraction f and giant component S, n = 600, 2e5 steps,"
          " 3 realisations")
    for eps in EPS:
        cells = []
        for c in CS:
            fr, Ss = [], []
            for sd in range(3):
                g = randomize_degree_preserving(grow_ls(600, 1, seed=sd), seed=sd)
                r = run_network(g, R_I=0.95, c=c, n_steps=200_000, seed=sd,
                                tau_mem=200.0, epsilon=eps, burn_frac=0.5)
                ce = {(min(i, j), max(i, j))
                      for i, j, k in zip(r["src"], r["dst"], r["rate"] > 1e-3)
                      if k}
                fr.append(len(ce) / len(all_edges(g)))
                Ss.append(giant_fraction(600, ce))
            cells.append(f"c={c:g}: f={np.mean(fr):.3f} S={np.mean(Ss):.3f}")
        print(f"  eps={eps:<6g} " + "  ".join(cells))

    print("\nExecuted-edge fraction (x1e3) per 400-step window, n = 300,"
          " c = 5e-4")
    g = randomize_degree_preserving(grow_ls(300, 1, seed=0), seed=0)
    for eps in (0.0, 0.001):
        r = run_network(g, R_I=0.95, c=0.0005, n_steps=40_000, seed=0,
                        tau_mem=200.0, epsilon=eps, burn_frac=0.5, record=True,
                        silence_hold=200, window=400)
        print(f"  eps={eps:<6g} "
              + " ".join(f"{1e3*v:.2f}" for v in r["series"][:20]))
