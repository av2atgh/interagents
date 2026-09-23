"""Mean-field percolation thresholds for the degree distributions actually used.

Reports, for each growth rule and each system size, the critical cost c_c at
which the cavity criterion Eq. (cavity) loses its giant component, and the
cruder Molloy-Reed value obtained by setting phi = 1. Both are upper bounds on
the measured threshold. Averaged over the same graph realisations as the
simulations, so that the theory curve carries the same finite-size information.
"""

import numpy as np

from network import grow_ls, grow_ba, randomize_degree_preserving
from theory import degree_pmf, percolates, k_crit, naive_MR


def build(model, n, seed):
    if model == "LS":
        return grow_ls(n, 1, seed=seed)
    if model == "CONF":
        return randomize_degree_preserving(grow_ls(n, 1, seed=seed), seed=seed)
    return grow_ba(n, 2, seed=seed)


def c_crit(pk, R_I=0.95, a=1, eps=0.0, lo=1e-4, hi=0.5, iters=40):
    """Bisect on c: percolating below, not percolating above."""
    if not percolates(pk, R_I, lo, a, eps)[0]:
        return np.nan
    if percolates(pk, R_I, hi, a, eps)[0]:
        return np.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if percolates(pk, R_I, mid, a, eps)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def c_crit_MR(pk, R_I=0.95, a=1, eps=0.0, lo=1e-4, hi=0.5, iters=40):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        kc = k_crit(R_I, mid, a, eps)
        if kc >= 1 and naive_MR(pk, kc) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("Mean-field thresholds, R_I = 0.95, a = 1")
    print(f"  {'model':>6} {'n':>6} {'<k>':>6} {'<k^2>':>8} {'kmax':>5} "
          f"{'c_c cavity':>11} {'c_c MR':>8} {'k_c(0.020)':>11}")
    for model in ("LS", "CONF", "BA"):
        for n in (300, 600, 1200, 2400):
            cav, mr, k2, km, kb = [], [], [], [], []
            for seed in range(6):
                g = build(model, n, seed)
                deg = np.array([len(x) for x in g])
                pk = degree_pmf(deg)
                cav.append(c_crit(pk))
                mr.append(c_crit_MR(pk))
                kb.append(deg.mean())
                k2.append((deg ** 2).mean())
                km.append(deg.max())
            print(f"  {model:>6} {n:>6} {np.mean(kb):>6.2f} {np.mean(k2):>8.1f} "
                  f"{np.mean(km):>5.0f} {np.nanmean(cav):>11.4f} "
                  f"{np.mean(mr):>8.4f} {k_crit(0.95, 0.020):>11d}")
    print("\n  exploration penalty at eps = 0.01 (CONF, n = 600):")
    g = build("CONF", 600, 0)
    pk = degree_pmf(np.array([len(x) for x in g]))
    for eps in (0.0, 0.01):
        print(f"    eps={eps:g}: c_c cavity = {c_crit(pk, eps=eps):.4f}, "
              f"k_c(c=0.020) = {k_crit(0.95, 0.020, 1, eps)}")
