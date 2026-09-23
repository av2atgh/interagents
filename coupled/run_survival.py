"""Collapse times, survival probabilities and first-passage distributions.

Referee-requested quantification of the escape into the absorbing state, for
the pair. The mean-field fixed point exists but fluctuations of the belief
estimate carry the pair over the unstable point; the escape is a rare event, so
the first-passage time to the solitary state should be exponentially
distributed with a mean set by the noise amplitude (~1/tau_mem) and by the
distance to the saddle-node.

Collapse is declared at the first step t such that the pair participation
propensity p stays below p*/2 for a whole memory time -- the same p*/2 cut used
for the occupancy phi, made into a sustained first-passage criterion so that
single-step excursions are not counted.

Outputs a JSON blob consumed by make_figures_rev.py.
"""

import copy
import json
import os
from multiprocessing import Pool

import numpy as np

from coupled_agents import AgentConfig, run

OUT = os.environ.get("OUT", "results_survival.json")


def p_star(L, c):
    a = L - 1
    return (a * c) ** (a / (a + 1.0))


def collapse_time(p_traj, thresh, hold):
    """First t with max(p_traj[t:t+hold]) < thresh; None if it never happens."""
    below = p_traj < thresh
    if not below.any():
        return None
    # running max over a window of length `hold`, via cumulative count of "above"
    above = (~below).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(above)])
    n = p_traj.size
    idx = np.arange(0, n - hold + 1)
    win_above = cs[idx + hold] - cs[idx]
    hit = np.flatnonzero(win_above == 0)
    return int(hit[0]) if hit.size else None


def one(args):
    L, c, R_I, eps, tau, seed, n_steps, p_init = args
    cfg = AgentConfig(L=L, level=1, tau_mem=tau, R_I=R_I, c=c,
                      epsilon=eps, p_init=p_init)
    res = run(cfg, copy.deepcopy(cfg), n_steps=n_steps, seed=seed,
              burn_frac=0.0)
    ps = p_star(L, c)
    tc = collapse_time(res["p_traj"], 0.5 * ps, int(tau))
    phi = float((res["p_traj"] > 0.5 * ps).mean())
    return dict(L=L, c=c, R_I=R_I, eps=eps, tau=tau, seed=seed,
                n_steps=n_steps, p_init=p_init, t_collapse=tc, phi=phi,
                rate=res["rate"])


def sweep(name, jobs, pool):
    rows = pool.map(one, jobs)
    print(f"[{name}] {len(rows)} realisations")
    return rows


if __name__ == "__main__":
    NREAL = int(os.environ.get("NREAL", 200))
    NSTEPS = int(os.environ.get("NSTEPS", 200_000))
    out = {}

    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        # --- (1) first-passage distribution at the reference operating point,
        #         L = 2, c = 0.35, R_I = 0.8428 (the bimodality point),
        #         many realisations -> S(t) and the FPT histogram.
        jobs = [(2, 0.35, 0.8428, 1e-2, 200.0, s, NSTEPS, 0.85)
                for s in range(NREAL)]
        out["fpt_reference"] = sweep("fpt_reference", jobs, pool)

        # --- (2) mean collapse time vs tau_mem (noise amplitude)
        jobs = []
        for tau in (25.0, 50.0, 100.0, 200.0, 400.0, 800.0):
            jobs += [(2, 0.35, 0.8428, 1e-2, tau, s, NSTEPS, 0.85)
                     for s in range(NREAL // 2)]
        out["tau_scan"] = sweep("tau_scan", jobs, pool)

        # --- (3) mean collapse time vs distance above R_c (= 0.83324 at c=0.35)
        jobs = []
        for R_I in (0.8400, 0.8428, 0.8500, 0.8700, 0.9000, 0.9500):
            jobs += [(2, 0.35, R_I, 1e-2, 200.0, s, NSTEPS, 0.85)
                     for s in range(NREAL // 2)]
        out["R_scan"] = sweep("R_scan", jobs, pool)

        # --- (4) exploration rate
        jobs = []
        for eps in (0.0, 1e-3, 1e-2, 3e-2, 1e-1):
            jobs += [(2, 0.35, 0.8428, eps, 200.0, s, NSTEPS, 0.85)
                     for s in range(NREAL // 2)]
        out["eps_scan"] = sweep("eps_scan", jobs, pool)

        # --- (5) initial condition
        jobs = []
        for p0 in (0.10, 0.30, 0.50, 0.70, 0.85, 1.00):
            jobs += [(2, 0.35, 0.8428, 1e-2, 200.0, s, NSTEPS, p0)
                     for s in range(NREAL // 2)]
        out["p_init_scan"] = sweep("p_init_scan", jobs, pool)

        # --- (6) observation window: does the inferred phase depend on T?
        jobs = []
        for T in (25_000, 50_000, 100_000, 200_000, 400_000, 800_000):
            jobs += [(2, 0.35, 0.8428, 1e-2, 200.0, s, T, 0.85)
                     for s in range(NREAL // 2)]
        out["T_scan"] = sweep("T_scan", jobs, pool)

    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT)
