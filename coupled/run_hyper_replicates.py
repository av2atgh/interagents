"""Replicates for the hypergraph section, which the first submission ran at a
single seed. Every entry is now an average over NREAL independent hypergraph
realisations (independent stub matching and independent dynamics)."""

import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

from hypergraph import regular_hypergraph, run_hyper

OUT = os.environ.get("OUT", "results_hyper.json")
NREAL = int(os.environ.get("NREAL", 12))
R_I = 0.95


def job_md(args):
    m, d, seed = args
    na = 240
    while (na * d) % m:
        na += 1
    ma, me, ne = regular_hypergraph(na, m, d, seed=seed)
    r = run_hyper(ma, me, na, ne, m, R_I=R_I, c=0.020, n_steps=40_000, seed=seed)
    return dict(m=m, d=d, seed=seed, coupled=float((r > 10 * (0.01 ** m)).mean()),
                rate=float(r.mean()))


def job_tau(args):
    c, tau, seed = args
    na = 240
    ma, me, ne = regular_hypergraph(na, 3, 1, seed=seed)
    r = run_hyper(ma, me, na, ne, 3, R_I=R_I, c=c, tau_mem=float(tau),
                  n_steps=60_000, seed=seed)
    return dict(c=c, tau=tau, seed=seed, rate=float(r.mean()))


def job_prop(args):
    m, mode, seed = args
    na = 240
    while na % m:
        na += 1
    ma, me, ne = regular_hypergraph(na, m, 1, seed=seed)
    R = 0.95 if mode == "const" else 0.95 * m
    r = run_hyper(ma, me, na, ne, m, R_I=R, c=0.020, n_steps=40_000, seed=seed)
    return dict(m=m, mode=mode, seed=seed, rate=float(r.mean()),
                coupled=float((r > 10 * (0.01 ** m)).mean()))


if __name__ == "__main__":
    out = {}
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        out["md"] = pool.map(job_md, [(m, d, s) for m in (2, 3, 4)
                                      for d in (1, 2, 4, 8, 16)
                                      for s in range(NREAL)])
        print("md done", flush=True)
        out["tau"] = pool.map(job_tau, [(c, t, s) for c in (0.020, 0.005)
                                        for t in (200, 1000, 5000, 20000)
                                        for s in range(NREAL)])
        print("tau done", flush=True)
        out["prop"] = pool.map(job_prop, [(m, mo, s) for m in (2, 3, 4, 6, 8, 10)
                                          for mo in ("const", "prop")
                                          for s in range(NREAL)])
        print("prop done", flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT)
