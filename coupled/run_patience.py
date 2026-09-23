"""Lookahead and endogenous commitment on the same collapsed set.

The first submission drew the recovery fractions of its two patience
experiments from two different collapsed subsets, of different run lengths, and
had to say in the text that the baselines were not comparable. Here both
experiments are run on the collapsed set of run_noreturn2.py, so they are.

A  pure lookahead at k = 1: does a longer horizon alone recover anything?
B  endogenous commitment: the agent evaluates the rollout at every candidate
   length in POLICY_GRID and latches onto the argmax, against a fixed k = 1
   control at the same discount.
"""

import json
import os
from multiprocessing import Pool

import numpy as np

from coupled_agents import AgentConfig, run

OUT = os.environ.get("OUT", "results_patience.json")
L, C, R_I, EPS, TAU = 2, 0.35, 0.8428, 0.01, 200.0
PSTAR = C ** 0.5
N_STEPS, UPGRADE = 200_000, 100_000
POLICY_GRID = (0, 16, 64, 256, 1024)


def cfg(level=1, gamma=0.97, horizon=60, grid=None, k=1):
    return AgentConfig(L=L, level=level, R_I=R_I, c=C, epsilon=EPS,
                       tau_mem=TAU, p_init=PSTAR, horizon=horizon,
                       gamma=gamma, commit_steps=k, commit_grid=grid)


def occ(p_traj, tail=0.2):
    j = int(len(p_traj) * (1 - tail))
    return float((p_traj[j:] > 0.5 * PSTAR).mean())


def job_H(args):
    seed, H, gamma = args
    a, b = cfg(1, gamma, H, None, 1), cfg(1, gamma, H, None, 1)
    r = run(a, b, n_steps=N_STEPS, seed=seed, upgrade_at=UPGRADE,
            upgrade_levels=(2, None))
    return dict(kind="H", seed=seed, H=H, gamma=gamma, phi=occ(r["p_traj"]))


def job_endog(args):
    seed, gamma, mode = args
    grid = POLICY_GRID if mode == "endogenous" else None
    a = cfg(1, gamma, 6000, grid, 1)
    b = cfg(1, gamma, 6000, grid, 1)
    r = run(a, b, n_steps=N_STEPS, seed=seed, upgrade_at=UPGRADE,
            upgrade_levels=(2, None))
    return dict(kind="endog", seed=seed, gamma=gamma, mode=mode,
                phi=occ(r["p_traj"]), rate=r["rate"],
                payoff_A=r["payoff_A"], payoff_B=r["payoff_B"])


if __name__ == "__main__":
    collapsed = json.load(open("results_noreturn.json"))["collapsed"]
    print(f"{len(collapsed)} collapsed realisations", flush=True)
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        jobs = [(s, H, g) for H in (10, 40, 200, 1000)
                for g in (0.97, 0.999) for s in collapsed]
        A = pool.map(job_H, jobs)
        print("lookahead done", flush=True)
        jobs = [(s, g, m) for g in (0.97, 0.99, 0.995, 0.999)
                for m in ("fixed", "endogenous") for s in collapsed]
        B = pool.map(job_endog, jobs)
        print("endogenous done", flush=True)
    json.dump(dict(lookahead=A, endogenous=B, collapsed=collapsed),
              open(OUT, "w"))
    print("wrote", OUT)
