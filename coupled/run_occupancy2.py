"""Tail index versus phase occupancy, with enough realisations to put a
confidence interval on the correlation.

Same design as run_occupancy.py -- vary only the noise realisation at a fixed
operating point inside the intermittent window, since the occupancy cannot be
tuned through R_I when the transition is sharp -- but with NREAL realisations
instead of 12, so that the correlation between the occupancy and the fitted tail
index carries a bootstrap interval and the alpha(phi) curves at different L can
be compared with error bars rather than by eye.
"""

import copy
import json
import os
from multiprocessing import Pool

import numpy as np

from coupled_agents import AgentConfig, run
from run_critical import pl_vs_exp

OUT = os.environ.get("OUT", "results_occupancy.json")
NREAL = int(os.environ.get("NREAL", 60))
NSTEPS = int(os.environ.get("NSTEPS", 1_200_000))
EPS, TAU_MEM = 0.01, 200.0

CASES = [dict(L=6, c=0.10, R=0.9715, pstar=(5 * 0.10) ** (5 / 6.0), pred=1.200),
         dict(L=2, c=0.35, R=0.8428, pstar=(1 * 0.35) ** (1 / 2.0), pred=2.000)]


def one(args):
    ci, seed = args
    cs = CASES[ci]
    cfg = AgentConfig(L=cs["L"], level=1, R_I=cs["R"], c=cs["c"], epsilon=EPS,
                      tau_mem=TAU_MEM, p_init=cs["pstar"])
    r = run(cfg, copy.deepcopy(cfg), n_steps=NSTEPS, seed=seed)
    phi = float((r["p_traj"] > 0.5 * cs["pstar"]).mean())
    tau = r["tau"]
    out = dict(case=ci, L=cs["L"], seed=seed, phi=phi, n_events=int(tau.size),
               alpha=np.nan, dll=np.nan, cv=np.nan)
    if tau.size >= 2000:
        alpha, dll, _ = pl_vs_exp(tau)
        out.update(alpha=float(alpha), dll=float(dll),
                   cv=float(tau.std() / tau.mean()))
    return out


if __name__ == "__main__":
    jobs = [(ci, sd) for ci in range(len(CASES))
            for sd in range(60, 60 + NREAL)]
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        rows = pool.map(one, jobs)
    json.dump(dict(rows=rows, cases=CASES, n_steps=NSTEPS), open(OUT, "w"))
    print("wrote", OUT)
