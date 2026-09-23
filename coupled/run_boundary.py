"""Measured phase boundary in R_I against the closed-form R_c, for several
(L, c). This is the sensitivity to c and to L: neither enters the dynamics
except through R_c and p*, Eq. (rcrit), so the test of that dependence is
whether the measured boundary moves with them as the formula says.

For each (L, c) we scan R_I and record the fraction of realisations still
coupled at the end of a run of fixed length, plus the censored-MLE mean collapse
time. The boundary is quoted as the R_I at which the coupled fraction crosses
1/2, obtained by linear interpolation on the scan; because the coupled state is
metastable this is a run-length-dependent operational definition, and the run
length is fixed at 2e5 steps throughout so that the comparison across (L, c) is
fair.
"""

import copy
import json
import os
from multiprocessing import Pool

import numpy as np

from coupled_agents import AgentConfig, run

OUT = os.environ.get("OUT", "results_boundary.json")
NREAL = int(os.environ.get("NREAL", 40))
NSTEPS = int(os.environ.get("NSTEPS", 200_000))


def p_star(L, c):
    a = L - 1
    return (a * c) ** (a / (a + 1.0))


def R_crit(L, c):
    a = L - 1
    ps = p_star(L, c)
    return ps ** (1.0 / a) + c / ps - c


CASES = [(2, 0.20), (2, 0.35), (2, 0.50), (3, 0.10), (3, 0.20), (3, 0.30)]
OFFS = (-0.02, -0.01, -0.005, 0.0, 0.005, 0.010, 0.020, 0.040, 0.080)


def one(args):
    L, c, R_I, seed = args
    cfg = AgentConfig(L=L, level=1, tau_mem=200.0, R_I=R_I, c=c,
                      epsilon=1e-2, p_init=0.85)
    res = run(cfg, copy.deepcopy(cfg), n_steps=NSTEPS, seed=seed,
              burn_frac=0.0)
    tr = res["p_traj"]
    ps = p_star(L, c)
    b = int(tr.size * 0.2)
    return dict(L=L, c=c, R_I=R_I, seed=seed,
                phi=float((tr[b:] > 0.5 * ps).mean()),
                n_steps=NSTEPS)


if __name__ == "__main__":
    jobs = []
    for L, c in CASES:
        Rc = R_crit(L, c)
        for d in OFFS:
            for s in range(NREAL):
                jobs.append((L, c, Rc + d, s))
    print(f"{len(jobs)} runs", flush=True)
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        rows = pool.map(one, jobs)
    json.dump(dict(rows=rows,
                   cases=[dict(L=L, c=c, Rc=R_crit(L, c), pstar=p_star(L, c))
                          for L, c in CASES],
                   offsets=list(OFFS)), open(OUT, "w"))
    print("wrote", OUT)
