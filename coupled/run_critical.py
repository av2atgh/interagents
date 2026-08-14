"""R8: locate the critical window where the LEARNED system recovers a
scale-free interaction timescale.  (plan.md 6A/R8, 3.5)

The coupled fixed point is born in a saddle-node bifurcation. Writing
a = L-1 and solving for tangency of F(p) = clip(R_I - c(1-p)/p, 0, 1)^a = p:

    p*   = (a c)^(a/(a+1))
    R_c  = p*^(1/a) + c/p* - c

For c=0.35: R_c(L=2) = 0.8332 at p* = 0.5916;  R_c(L=3) = 0.9818 at p* = 0.7885.

Just above R_c the stable and unstable fixed points nearly coincide, the barrier
between the coupled and solitary basins vanishes, and with a nonzero epistemic
reach rate (which makes the solitary phase escapable at all) the pair should
switch intermittently between phases. Intermittent switching with a vanishing
barrier is the standard route to a power law. If the theory wants alpha as its
order parameter, it has to appear HERE or nowhere.

Discriminator: tail log-likelihood of a power law vs an exponential (positive
=> power law favoured). CV >> 1 is necessary but not sufficient -- a mixture of
two exponentials also has CV > 1.
"""

import numpy as np
from coupled_agents import AgentConfig, run, alpha_of

C = 0.35


def R_crit(a, c=C):
    p = (a * c) ** (a / (a + 1.0))
    return p ** (1.0 / a) + c / p - c, p


def pl_vs_exp(tau, q=0.80):
    """Tail comparison. Returns (alpha, loglik_PL - loglik_exp, n_tail)."""
    if tau.size < 2000:
        return np.nan, np.nan, 0
    xmin = max(2.0, float(np.quantile(tau, q)))
    x = tau[tau >= xmin]
    n = x.size
    if n < 500:
        return np.nan, np.nan, n
    s = np.sum(np.log(x / xmin))
    alpha = 1.0 + n / s
    ll_pl = n * np.log((alpha - 1.0) / xmin) - alpha * s
    m = x.mean()
    if m <= xmin:
        return alpha, np.nan, n
    lam = 1.0 / (m - xmin)
    ll_ex = n * np.log(lam) - lam * np.sum(x - xmin)
    return alpha, ll_pl - ll_ex, n


def sweep(L, eps, tau_mem, n_steps, seeds=(21, 22, 23)):
    a = L - 1
    Rc, pstar = R_crit(a)
    print(f"\n  L={L}  a={a}  R_c={Rc:.4f}  p*={pstar:.4f}  "
          f"epsilon={eps}  tau_mem={tau_mem:.0f}  steps={n_steps:,}")
    print(f"  {'R_I':>7} {'R-Rc':>8} {'n_ev':>7} {'CV':>7} {'alpha':>7} "
          f"{'lnL_PL-lnL_exp':>15} {'frac_coupled':>13}")
    for dR in (-0.010, -0.002, 0.000, 0.002, 0.006, 0.012, 0.025, 0.050, 0.100):
        R = Rc + dR
        taus, fracs = [], []
        for sd in seeds:
            cfg = AgentConfig(L=L, level=1, R_I=R, c=C, epsilon=eps,
                              tau_mem=tau_mem, p_init=pstar)
            r = run(cfg, cfg, n_steps=n_steps, seed=sd)
            taus.append(r["tau"])
            fracs.append(float((r["p_traj"] > 0.5 * pstar).mean()))
        tau = np.concatenate(taus)
        if tau.size < 50:
            print(f"  {R:>7.4f} {dR:>8.3f} {tau.size:>7} " + " " * 30 + "(too few events)")
            continue
        cv = tau.std() / tau.mean()
        alpha, dll, ntail = pl_vs_exp(tau)
        a_s = f"{alpha:7.3f}" if not np.isnan(alpha) else "     --"
        d_s = f"{dll:15.1f}" if not np.isnan(dll) else "             --"
        print(f"  {R:>7.4f} {dR:>8.3f} {tau.size:>7} {cv:>7.2f} {a_s} {d_s} "
              f"{np.mean(fracs):>13.3f}")


if __name__ == "__main__":
    print("=" * 92)
    print("R8  CRITICAL WINDOW SEARCH")
    print("=" * 92)

    for eps in (0.0, 1e-3, 1e-2):
        sweep(L=2, eps=eps, tau_mem=50.0, n_steps=400_000)

    print("\n" + "=" * 92)
    print("Memory dependence at the critical point (shorter memory = larger belief noise)")
    print("=" * 92)
    for tm in (10.0, 50.0, 200.0):
        sweep(L=2, eps=1e-2, tau_mem=tm, n_steps=400_000)
