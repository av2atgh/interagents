"""R8, decisive test: does the LEARNED intermittent system fall in the same
universality class as the i.i.d. queueing model?

run_critical.py found a window where the pair switches intermittently between
the coupled and solitary phases, with CV >> 1 and a decisive power-law
preference over an exponential. At L=2 the measured exponent came out at 1.93,
against the paper's alpha = 1 + 1/max(L_j - 1) = 2.

That could be coincidence. The test: locate the intermittent window at each L by
tuning R_I until the pair spends a target fraction of its time coupled, then
measure alpha there and compare with 1 + 1/(L-1).

  agreement across L  => learning preserves the universality class, and alpha
                         survives as the order parameter the theory wants
  disagreement        => 1.93 was a coincidence and R8 stays open
"""

import numpy as np
from coupled_agents import AgentConfig, run
from run_critical import R_crit, pl_vs_exp, C

TARGET = 0.40          # target fraction of time in the coupled phase
EPS = 0.01


def frac_coupled(L, R, tau_mem, n_steps, seed, pstar):
    cfg = AgentConfig(L=L, level=1, R_I=R, c=C, epsilon=EPS,
                      tau_mem=tau_mem, p_init=pstar)
    r = run(cfg, cfg, n_steps=n_steps, seed=seed)
    return float((r["p_traj"] > 0.5 * pstar).mean()), r


def bisect_R(L, tau_mem, pstar, Rc, n_steps=250_000, iters=12):
    """Find R_I giving frac_coupled ~= TARGET. Monotone increasing in R."""
    lo, hi = Rc - 0.002, Rc + 0.30
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f, _ = frac_coupled(L, mid, tau_mem, n_steps, 31, pstar)
        if f < TARGET:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print("=" * 88)
print("R8  UNIVERSALITY TEST:  does the intermittent learned system reproduce")
print("    alpha = 1 + 1/(L-1) ?")
print("=" * 88)
print(f"{'L':>3} {'R_c':>8} {'R_win':>8} {'frac':>6} {'n_ev':>9} {'CV':>8} "
      f"{'alpha_meas':>11} {'alpha_pred':>11} {'lnL_PL-exp':>12}")

for L in (2, 3, 4, 5):
    a = L - 1
    Rc, pstar = R_crit(a)
    tau_mem = 200.0
    Rw = bisect_R(L, tau_mem, pstar, Rc)

    taus, fr = [], []
    for sd in (41, 42, 43, 44):
        f, r = frac_coupled(L, Rw, tau_mem, 1_500_000, sd, pstar)
        taus.append(r["tau"])
        fr.append(f)
    tau = np.concatenate(taus)
    if tau.size < 2000:
        print(f"{L:>3} {Rc:>8.4f} {Rw:>8.4f} {np.mean(fr):>6.3f} {tau.size:>9} "
              f"  (too few events)")
        continue
    cv = tau.std() / tau.mean()
    alpha, dll, _ = pl_vs_exp(tau)
    pred = 1.0 + 1.0 / a
    a_s = f"{alpha:11.3f}" if not np.isnan(alpha) else "         --"
    d_s = f"{dll:12.1f}" if not np.isnan(dll) else "          --"
    print(f"{L:>3} {Rc:>8.4f} {Rw:>8.4f} {np.mean(fr):>6.3f} {tau.size:>9} "
          f"{cv:>8.2f} {a_s} {pred:>11.3f} {d_s}")
