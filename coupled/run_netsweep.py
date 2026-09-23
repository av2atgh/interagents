"""Network percolation sweep: many realisations, several sizes, one file per run.

Each run stores the per-edge interaction rate, the degree sequence, the
edge-level first-passage time to sustained silence and the coarse-grained time
series of the executed-edge fraction. All of the analysis the referees ask for
-- error bars, component-size histograms, robustness of the coupled-edge
criterion, collapse times, dependence of the threshold on run length -- is then
done offline by analyze_net.py without re-simulating.

Usage:  python3 run_netsweep.py [tag]
        tag in {main, finitesize, length, sensitivity}
"""

import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

from network import (grow_ls, grow_ba, randomize_degree_preserving,
                     run_network, directed_index)

OUTDIR = os.environ.get("OUTDIR", "results_net")
os.makedirs(OUTDIR, exist_ok=True)


def build(model, n, seed):
    if model == "LS":
        return grow_ls(n, 1, seed=seed)
    if model == "CONF":
        return randomize_degree_preserving(grow_ls(n, 1, seed=seed), seed=seed)
    if model == "BA":
        return grow_ba(n, 2, seed=seed)
    raise ValueError(model)


def tagname(job):
    model, n, c, seed, steps, R_I, eps, tau, L, p0 = job
    return (f"{model}_n{n}_c{c:.4f}_s{seed}_T{steps}_R{R_I:.3f}"
            f"_e{eps:g}_t{tau:g}_L{L}_p{p0:.2f}.npz")


def one(job):
    model, n, c, seed, steps, R_I, eps, tau, L, p0 = job
    path = os.path.join(OUTDIR, tagname(job))
    if os.path.exists(path):
        return path
    g = build(model, n, seed)
    # A coupled edge executes every ten or twenty steps, so a silence of one
    # memory time happens by chance even in the coupled phase and is useless as
    # a death criterion. Ten memory times is not: at the coupled execution rate
    # the probability of such a gap is ~1e-4, so an edge that falls silent for
    # 10 tau_mem has left the coupled state.
    res = run_network(g, R_I=R_I, c=c, L=L, tau_mem=tau, epsilon=eps,
                      n_steps=steps, burn_frac=0.5, seed=seed, p_init=p0,
                      record=True, silence_hold=int(10 * tau),
                      window=max(2000, int(steps // 200)))
    deg = np.array([len(x) for x in g], dtype=np.int32)
    np.savez_compressed(
        path, src=res["src"].astype(np.int32), dst=res["dst"].astype(np.int32),
        rate=res["rate"].astype(np.float32), deg=deg,
        fpt=res["fpt"].astype(np.int64), series=res["series"].astype(np.float32),
        window=res["window"], hold=res["silence_hold"],
        meta=np.array([n, c, seed, steps, R_I, eps, tau, L, p0], dtype=float))
    return path


# ------------------------------------------------------------------ job sets
D = dict(R_I=0.95, eps=0.01, tau=200.0, L=2, p0=0.85)
CS_MAIN = (0.002, 0.010, 0.015, 0.020, 0.025, 0.030, 0.045, 0.070)
CS_FS = (0.015, 0.020, 0.025, 0.030)


def jobs_main():
    """n = 600, all three growth rules, 24 realisations, converged runs."""
    out = []
    for model, c, seed in itertools.product(("LS", "CONF", "BA"), CS_MAIN,
                                            range(24)):
        out.append((model, 600, c, seed, 400_000, D["R_I"], D["eps"],
                    D["tau"], D["L"], D["p0"]))
    return out


def jobs_finitesize():
    """Size ladder for LS and its randomization."""
    out = []
    reps = {300: 24, 600: 24, 1200: 16, 2400: 12, 4800: 8}
    for model, n, c in itertools.product(("LS", "CONF"),
                                         (300, 600, 1200, 2400, 4800),
                                         CS_FS):
        for seed in range(reps[n]):
            out.append((model, n, c, seed, 400_000, D["R_I"], D["eps"],
                        D["tau"], D["L"], D["p0"]))
    return out


def jobs_length():
    """Dependence of the inferred threshold on run length, and on the initial
    condition -- the two things that bias a short-run estimate."""
    out = []
    for c, T, seed in itertools.product((0.015, 0.020, 0.025, 0.030),
                                        (25_000, 50_000, 100_000, 200_000,
                                         400_000, 800_000, 1_600_000),
                                        range(12)):
        out.append(("CONF", 600, c, seed, T, D["R_I"], D["eps"], D["tau"],
                    D["L"], D["p0"]))
    for c, p0, seed in itertools.product((0.010, 0.015, 0.020, 0.025, 0.030),
                                         (0.10, 0.30, 0.50, 0.85), range(12)):
        out.append(("CONF", 600, c, seed, 400_000, D["R_I"], D["eps"],
                    D["tau"], D["L"], p0))
    return out


def jobs_sensitivity():
    """One parameter at a time about the reference point, n = 600, CONF."""
    out = []
    base = dict(D)
    grid = dict(eps=(0.0, 0.003, 0.01, 0.03, 0.1),
                tau=(50.0, 100.0, 200.0, 400.0, 800.0),
                R_I=(0.85, 0.90, 0.95, 1.00),
                L=(2, 3, 4))
    for key, vals in grid.items():
        for v in vals:
            p = dict(base)
            p[key] = v
            for c in (0.010, 0.015, 0.020, 0.025, 0.030):
                for seed in range(8):
                    out.append(("CONF", 600, c, seed, 400_000, p["R_I"],
                                p["eps"], p["tau"], p["L"], p["p0"]))
    # de-duplicate the reference point, which every grid repeats
    seen, uniq = set(), []
    for j in out:
        if j not in seen:
            seen.add(j)
            uniq.append(j)
    return uniq


def jobs_long():
    """Very long runs near the threshold. The coupled fraction is still rising
    at 1.6e6 steps at c = 0.025 and 0.030, so the transition located at 4e5
    steps is a lower bound; these runs say how much further it moves."""
    out = []
    for c, T, seed in itertools.product((0.025, 0.030, 0.035, 0.045, 0.062),
                                        (1_600_000, 6_400_000), range(8)):
        out.append(("CONF", 600, c, seed, T, D["R_I"], D["eps"], D["tau"],
                    D["L"], D["p0"]))
    return out


SETS = dict(main=jobs_main, finitesize=jobs_finitesize,
            length=jobs_length, sensitivity=jobs_sensitivity,
            long=jobs_long)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    jobs = SETS[which]()
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUTDIR, tagname(j)))]
    print(f"[{which}] {len(jobs)} jobs, {len(todo)} to run", flush=True)
    with Pool(int(os.environ.get("NPROC", 10))) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, todo, chunksize=1), 1):
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"[{which}] done", flush=True)
