"""R9, corrected. Does alpha in the intermittent window depend on L (a
universality class, as in the i.i.d. paper) or only on phase occupancy (a
tuning artefact)?

Two corrections to run_universality.py:

1. The interior saddle-node p* = (a c)^(a/(a+1)) is only valid while p* <= 1,
   i.e. a <= 1/c. At c=0.35 that caps L at 3.86, so the earlier L=4 and L=5 rows
   were probing a bifurcation that does not exist and R_crit returned garbage.
   Here c=0.1, valid to L=11.

2. alpha is measured at MATCHED phase occupancy across L, and at two different
   occupancies, so the two hypotheses separate:

     alpha depends on L, not occupancy  -> universality class survives learning
     alpha depends on occupancy, not L  -> alpha is a tuning knob, not an order
                                           parameter, and R8 stays open
"""

import numpy as np
from coupled_agents import AgentConfig, run
from run_critical import pl_vs_exp

C = 0.10
EPS = 0.01


def R_crit(a, c=C):
    p = (a * c) ** (a / (a + 1.0))
    if p > 1.0:
        return None, None
    return p ** (1.0 / a) + c / p - c, p


def measure(L, R, tau_mem, n_steps, seeds, pstar):
    taus, fr = [], []
    for sd in seeds:
        cfg = AgentConfig(L=L, level=1, R_I=R, c=C, epsilon=EPS,
                          tau_mem=tau_mem, p_init=pstar)
        r = run(cfg, cfg, n_steps=n_steps, seed=sd)
        taus.append(r["tau"])
        fr.append(float((r["p_traj"] > 0.5 * pstar).mean()))
    return np.concatenate(taus), float(np.mean(fr))


def bisect(L, target, tau_mem, pstar, Rc, n_steps=200_000, iters=11):
    lo, hi = Rc - 0.01, min(Rc + 0.40, 1.30)
    # NOTE: this bisection assumes frac_coupled is a smooth increasing function
    # of R_I. It is not -- the transition is sharp, so the search converges to
    # the jump location regardless of `target`, and every row of R9 returned the
    # same R_win for target 0.20 and 0.50. Kept as run, with the failure
    # documented; see run_occupancy.py for the test that replaces it.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, f = measure(L, mid, tau_mem, n_steps, (31,), pstar)
        if f < target:
            lo = mid
        else:
            hi = mid
    R = 0.5 * (lo + hi)
    _, f = measure(L, R, tau_mem, n_steps, (31,), pstar)
    return R, f


print("=" * 94)
print(f"R9  alpha vs L at MATCHED phase occupancy   (c={C}, eps={EPS})")
print("=" * 94)
print(f"{'L':>3} {'target':>7} {'R_c':>8} {'R_win':>8} {'frac':>6} {'n_ev':>9} "
      f"{'CV':>7} {'alpha':>7} {'1+1/(L-1)':>10} {'lnL_PL-exp':>12}")

for target in (0.20, 0.50):
    for L in (2, 3, 4, 5, 6):
        a = L - 1
        Rc, pstar = R_crit(a)
        if Rc is None:
            print(f"{L:>3} {target:>7.2f}   -- no interior saddle-node at c={C} --")
            continue
        tau_mem = 200.0
        Rw, _ = bisect(L, target, tau_mem, pstar, Rc)
        tau, frac = measure(L, Rw, tau_mem, 1_200_000, (41, 42), pstar)
        if tau.size < 2000:
            print(f"{L:>3} {target:>7.2f} {Rc:>8.4f} {Rw:>8.4f} {frac:>6.3f} "
                  f"{tau.size:>9}   (too few events)")
            continue
        cv = tau.std() / tau.mean()
        alpha, dll, _ = pl_vs_exp(tau)
        a_s = f"{alpha:7.3f}" if not np.isnan(alpha) else "     --"
        d_s = f"{dll:12.1f}" if not np.isnan(dll) else "          --"
        print(f"{L:>3} {target:>7.2f} {Rc:>8.4f} {Rw:>8.4f} {frac:>6.3f} "
              f"{tau.size:>9} {cv:>7.2f} {a_s} {1+1/a:>10.3f} {d_s}")
    print()
