"""Bimodality, predictability and recovery, with proper statistics.

Same operating point as the first submission (L=2, c=0.35, R_I=0.8428,
epsilon=1e-2, tau_mem=200, p_init=p*, 3e5 steps, occupancy measured on the final
20% of the trajectory) but over NREAL independent realisations instead of 140,
so that the bimodal split, the predictability of the outcome from a trajectory
prefix and the recovery fractions all carry confidence intervals.

The prefix classifier of the first submission chose its threshold on the same
data it was scored on, which biases the accuracy upward. Here we report both the
in-sample optimum (comparable to the first submission) and a split-sample
estimate: the threshold is fitted on a random half and scored on the other half,
averaged over 400 splits.
"""

import copy
import json
import os
from multiprocessing import Pool

import numpy as np

from coupled_agents import AgentConfig, run

OUT = os.environ.get("OUT", "results_bimodality.json")
NREAL = int(os.environ.get("NREAL", 1000))

L, C, R_I, EPS, TAU = 2, 0.35, 0.8428, 0.01, 200.0
PSTAR = C ** 0.5
N_STEPS = 300_000
CHECKS = (1_000, 3_000, 10_000, 30_000, 60_000, 120_000)


def base(level):
    return AgentConfig(L=L, level=level, R_I=R_I, c=C, epsilon=EPS,
                       tau_mem=TAU, p_init=PSTAR)


def occ(p_traj, tail=0.2):
    k = int(len(p_traj) * (1 - tail))
    return float((p_traj[k:] > 0.5 * PSTAR).mean())


def job_base(seed):
    r = run(base(1), base(1), n_steps=N_STEPS, seed=seed)
    p = r["p_traj"]
    return dict(seed=seed, phi=occ(p),
                prefix={str(t): float((p[:t] > 0.5 * PSTAR).mean())
                        for t in CHECKS})


def job_rescue(args):
    seed, mode = args
    if mode == "control":
        r = run(base(1), base(1), n_steps=N_STEPS, seed=seed,
                upgrade_at=N_STEPS // 2, upgrade_levels=(None, None))
    elif mode == "A2":
        r = run(base(1), base(1), n_steps=N_STEPS, seed=seed,
                upgrade_at=N_STEPS // 2, upgrade_levels=(2, None))
    elif mode == "both2":
        r = run(base(1), base(1), n_steps=N_STEPS, seed=seed,
                upgrade_at=N_STEPS // 2, upgrade_levels=(2, 2))
    elif mode == "lvl2_t0":
        r = run(base(2), copy.deepcopy(base(2)), n_steps=N_STEPS, seed=seed)
    return dict(seed=seed, mode=mode, phi=occ(r["p_traj"]))


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        rows = pool.map(job_base, range(200, 200 + NREAL))
        print(f"base: {len(rows)} realisations", flush=True)
        collapsed = [r["seed"] for r in rows if r["phi"] < 0.25]
        print(f"collapsed: {len(collapsed)}", flush=True)
        jobs = [(s, m) for m in ("control", "A2", "both2", "lvl2_t0")
                for s in collapsed]
        resc = pool.map(job_rescue, jobs)
        print("rescue done", flush=True)
    json.dump(dict(base=rows, rescue=resc, collapsed=collapsed,
                   n_steps=N_STEPS, checks=list(CHECKS)), open(OUT, "w"))
    print("wrote", OUT)
