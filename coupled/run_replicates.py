"""Replicates and error bars for the pair-level tables of the manuscript.

The first submission reported several of these at a single seed. Here every
entry is an average over NREAL independent realisations (independent random
seeds, hence independent priority draws, exploration events and solitary-task
values), reported as mean +/- standard error, together with the number of
realisations classified as coupled.

Two observables are kept apart, because the first submission conflated them:

  pbar  the time-averaged participation propensity, (p_A + p_B)/2
  phi   the occupancy, the fraction of steps with (p_A+p_B)/2 > p*/2

phi is the order parameter defined in the text; pbar is what the tables of the
first submission actually reported.
"""

import copy
import json
import os
from multiprocessing import Pool

import numpy as np

from coupled_agents import AgentConfig, run, alpha_of

OUT = os.environ.get("OUT", "results_replicates.json")
NREAL = int(os.environ.get("NREAL", 40))
NSTEPS = int(os.environ.get("NSTEPS", 200_000))


def p_star(L, c):
    a = L - 1
    return (a * c) ** (a / (a + 1.0))


def summarise(res, L, c, burn_frac=0.2):
    tr = res["p_traj"]
    b = int(tr.size * burn_frac)
    tr = tr[b:]
    ps = p_star(L, c)
    return dict(pbar=float(tr.mean()), phi=float((tr > 0.5 * ps).mean()),
                rate=res["rate"], n_events=int(res["n_events"]),
                alpha=float(alpha_of(res["tau"])),
                payoff_A=res["payoff_A"], payoff_B=res["payoff_B"])


# ------------------------------------------------------------------ E3 / asym
def job_asym(args):
    R, lvA, lvB, seed = args
    cA = AgentConfig(L=3, level=lvA, R_I=R, tau_mem=50.0, epsilon=0.0)
    cB = AgentConfig(L=3, level=lvB, R_I=R, tau_mem=50.0, epsilon=0.0)
    r = run(cA, cB, n_steps=NSTEPS, seed=seed)
    d = summarise(r, 3, cA.c)
    d.update(R_I=R, lvA=lvA, lvB=lvB, seed=seed)
    return d


# ------------------------------------------------------------------ hold table
def job_hold(args):
    inv_nu, tau, hold, seed = args
    cfg = AgentConfig(L=3, level=2, R_I=0.99, tau_mem=tau, hold_silence=hold)
    r = run(cfg, copy.deepcopy(cfg), n_steps=NSTEPS, observability=0.0,
            avail=1.0 / inv_nu, seed=seed)
    d = summarise(r, 3, cfg.c)
    d.update(inv_nu=inv_nu, tau=tau, hold=hold, seed=seed)
    return d


# ------------------------------------------- level-1 vs level-2 phase boundary
def job_levels(args):
    L, R, lvl, seed = args
    cfg = AgentConfig(L=L, level=lvl, R_I=R)
    r = run(cfg, copy.deepcopy(cfg), n_steps=NSTEPS, seed=seed)
    d = summarise(r, L, cfg.c)
    d.update(L=L, R_I=R, level=lvl, seed=seed)
    return d


if __name__ == "__main__":
    out = {}
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        jobs = [(R, lv[0], lv[1], s)
                for R in (0.94, 0.96, 0.98)
                for lv in ((1, 1), (2, 1), (2, 2))
                for s in range(NREAL)]
        out["asym"] = pool.map(job_asym, jobs)
        print("asym done", flush=True)

        jobs = [(inv, tau, hold, s)
                for inv in (10.0, 100.0, 333.0)
                for tau in (50.0, 5000.0)
                for hold in (False, True)
                for s in range(NREAL)]
        out["hold"] = pool.map(job_hold, jobs)
        print("hold done", flush=True)

        jobs = [(L, R, lvl, s)
                for L in (2, 3)
                for R in (0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 1.00)
                for lvl in (1, 2)
                for s in range(NREAL)]
        out["levels"] = pool.map(job_levels, jobs)
        print("levels done", flush=True)

    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT)
