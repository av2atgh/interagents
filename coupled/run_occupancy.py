"""R10: is alpha a function of phase occupancy rather than of L?

R9 could not test this as designed: occupancy is NOT tunable via R_I, because
the coupled/solitary transition is sharp. The bisection returned the same R_win
for target 0.20 and 0.50 in every row.

So vary the seed instead. At a single (L, c, R_I) sitting inside the window,
different noise realisations land at different occupancies. If alpha tracks
occupancy WITHIN one L, and the alpha(occupancy) curve then coincides across
different L, alpha is a tuning-dependent quantity and not a universality class.
"""

import numpy as np
from coupled_agents import AgentConfig, run
from run_critical import pl_vs_exp

EPS = 0.01
TAU_MEM = 200.0


def point(L, c, R, seed, pstar, n_steps=1_200_000):
    cfg = AgentConfig(L=L, level=1, R_I=R, c=c, epsilon=EPS,
                      tau_mem=TAU_MEM, p_init=pstar)
    r = run(cfg, cfg, n_steps=n_steps, seed=seed)
    frac = float((r["p_traj"] > 0.5 * pstar).mean())
    tau = r["tau"]
    if tau.size < 2000:
        return frac, np.nan, np.nan, tau.size
    alpha, dll, _ = pl_vs_exp(tau)
    return frac, alpha, tau.std() / tau.mean(), tau.size


print("=" * 78)
print("R10  alpha vs occupancy, varying ONLY the seed at a fixed operating point")
print("=" * 78)

# two operating points known to sit inside the intermittent window
CASES = [
    dict(L=6, c=0.10, R=0.9715, pstar=(5 * 0.10) ** (5 / 6.0), pred=1.200),
    dict(L=2, c=0.35, R=0.8428, pstar=(1 * 0.35) ** (1 / 2.0), pred=2.000),
]

for cs in CASES:
    print(f"\n  L={cs['L']}  c={cs['c']}  R_I={cs['R']}  "
          f"predicted alpha if universal = {cs['pred']:.3f}")
    print(f"  {'seed':>5} {'frac':>7} {'n_ev':>9} {'CV':>8} {'alpha':>8}")
    rows = []
    for sd in range(60, 72):
        frac, alpha, cv, n = point(cs["L"], cs["c"], cs["R"], sd, cs["pstar"])
        a_s = f"{alpha:8.3f}" if not np.isnan(alpha) else "      --"
        cv_s = f"{cv:8.2f}" if not np.isnan(cv) else "      --"
        print(f"  {sd:>5} {frac:>7.3f} {n:>9} {cv_s} {a_s}")
        if not np.isnan(alpha):
            rows.append((frac, alpha))
    if len(rows) >= 4:
        f = np.array([r[0] for r in rows])
        a = np.array([r[1] for r in rows])
        if f.std() > 1e-6:
            print(f"    corr(frac, alpha) = {np.corrcoef(f, a)[0,1]:+.3f}  "
                  f"over {len(rows)} usable seeds, frac in "
                  f"[{f.min():.3f}, {f.max():.3f}]")
        else:
            print(f"    occupancy did not vary across seeds (frac={f.mean():.3f}); "
                  f"alpha = {a.mean():.3f} +/- {a.std():.3f}")
