"""Recovery of collapsed pairs versus commitment length and discount, with
enough collapsed realisations to put a confidence interval on the recovery
fraction.

Protocol identical to run_noreturn.py (L=2, c=0.35, R_I=0.8428, eps=1e-2,
tau_mem=200, p_init=p*, 2e5 steps, agent A promoted to level 2 at the midpoint)
but over NSEED base realisations instead of 140, so that the collapsed set is
several times larger.

Part A is the deterministic commit region -- the smallest belief at which the
rollout still recommends participating -- which needs no statistics because the
decision is a deterministic function of the beliefs. Part B is the measured
recovery, which does.
"""

import json
import os
from multiprocessing import Pool

import numpy as np

from coupled_agents import AgentConfig, Agent, run

OUT = os.environ.get("OUT", "results_noreturn.json")
NSEED = int(os.environ.get("NSEED", 400))
L, C, R_I, EPS, TAU = 2, 0.35, 0.8428, 0.01, 200.0
PSTAR = C ** 0.5
N_STEPS = 200_000
UPGRADE = N_STEPS // 2
CELLS = [(k, g) for k in (1, 10, 100, 1000) for g in (0.97, 0.999)]


def cfg(level=2, horizon=40, gamma=0.97, commit=1):
    return AgentConfig(L=L, level=level, R_I=R_I, c=C, epsilon=EPS,
                       tau_mem=TAU, p_init=PSTAR, horizon=horizon,
                       gamma=gamma, commit_steps=commit)


def occ(p_traj, tail=0.2):
    k = int(len(p_traj) * (1 - tail))
    return float((p_traj[k:] > 0.5 * PSTAR).mean())


def job_base(seed):
    r = run(cfg(1), cfg(1), n_steps=N_STEPS, seed=seed)
    return dict(seed=seed, phi=occ(r["p_traj"]))


def job_cell(args):
    seed, k, gamma = args
    H = max(60, 3 * k)
    a = cfg(1, H, gamma, k)
    b = cfg(1, H, gamma, k)
    r = run(a, b, n_steps=N_STEPS, seed=seed, upgrade_at=UPGRADE,
            upgrade_levels=(2, None))
    return dict(seed=seed, k=k, gamma=gamma, phi=occ(r["p_traj"]),
                rate=r["rate"], payoff_A=r["payoff_A"], payoff_B=r["payoff_B"])


def commit_region():
    """Part A: smallest p_hat at which the agent still commits (deterministic)."""
    rng0 = np.random.default_rng(0)
    grid = np.concatenate([np.geomspace(1e-6, 0.01, 40),
                           np.linspace(0.01, 1.0, 200)])
    out = []
    for k, gamma in CELLS:
        H = max(60, 3 * k)
        ag = Agent(cfg(2, H, gamma, k), rng0)
        lo = None
        for p in grid:
            ag.p_hat, ag.s_hat, ag._commit_left = float(p), float(p), 0
            if ag._rollout(k) > ag._rollout(0):
                lo = float(p)
                break
        out.append(dict(k=k, gamma=gamma, horizon=H, min_p=lo))
    return out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        base = pool.map(job_base, range(1000, 1000 + NSEED))
        collapsed = [r["seed"] for r in base if r["phi"] < 0.25]
        print(f"{len(collapsed)} of {NSEED} collapsed", flush=True)
        jobs = [(s, k, g) for (k, g) in CELLS for s in collapsed]
        cells = pool.map(job_cell, jobs)
    json.dump(dict(base=base, collapsed=collapsed, cells=cells,
                   region=commit_region(), n_steps=N_STEPS), open(OUT, "w"))
    print("wrote", OUT)
