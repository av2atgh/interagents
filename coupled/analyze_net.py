"""Offline analysis of the network sweep (run_netsweep.py).

Everything the referees ask for on the network side is computed from the stored
per-run arrays: error bars over independent realisations, component-size
distributions, robustness of the coupled-edge criterion, edge-level collapse
times, finite-size behaviour, and the dependence of the inferred threshold on
run length and on the initial condition.
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np

DIR = os.environ.get("RESULTS_NET", "results_net")
PAT = re.compile(r"(?P<model>[A-Z]+)_n(?P<n>\d+)_c(?P<c>[\d.]+)_s(?P<seed>\d+)"
                 r"_T(?P<T>\d+)_R(?P<R>[\d.]+)_e(?P<e>[\d.e+-]+)_t(?P<t>[\d.]+)"
                 r"_L(?P<L>\d+)_p(?P<p>[\d.]+)\.npz")


def load_all():
    out = []
    for f in sorted(os.listdir(DIR)):
        m = PAT.fullmatch(f)
        if not m:
            continue
        d = m.groupdict()
        out.append(dict(path=os.path.join(DIR, f), model=d["model"],
                        n=int(d["n"]), c=float(d["c"]), seed=int(d["seed"]),
                        T=int(d["T"]), R=float(d["R"]), eps=float(d["e"]),
                        tau=float(d["t"]), L=int(d["L"]), p0=float(d["p"])))
    return out


def comp_sizes(n, edges):
    if not edges:
        return [0]
    nb = defaultdict(list)
    for i, j in edges:
        nb[i].append(j)
        nb[j].append(i)
    seen, sizes = set(), []
    for s in nb:
        if s in seen:
            continue
        stack, sz = [s], 0
        seen.add(s)
        while stack:
            v = stack.pop()
            sz += 1
            for w in nb[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        sizes.append(sz)
    return sorted(sizes, reverse=True)


def observables(rec, thresh=1e-3):
    z = np.load(rec["path"])
    src, dst, rate, deg = z["src"], z["dst"], z["rate"], z["deg"]
    n = rec["n"]
    keep = rate > thresh
    edges = {(min(i, j), max(i, j)) for i, j, k in zip(src, dst, keep) if k}
    n_edges = src.size // 2
    sizes = comp_sizes(n, edges)
    S = sizes[0] / n
    # degree-resolved survival of undirected edges
    cd = np.zeros(n)
    for i, j in edges:
        cd[i] += 1
        cd[j] += 1
    surv = {}
    for b in (2, 4, 8, 16, 32):
        sel = (deg >= b) & (deg < 2 * b)
        if sel.sum() >= 5:
            surv[b] = float((cd[sel] / deg[sel]).mean())
    fpt = z["fpt"]
    return dict(frac=len(edges) / n_edges, S=S,
                S2=(sizes[1] / n if len(sizes) > 1 else 0.0),
                ncomp=sum(1 for s in sizes if s >= 3), sizes=sizes,
                surv=surv, rate=rate, fpt=fpt, series=z["series"],
                window=int(z["window"]), hold=int(z["hold"]),
                nnode=n, deg=deg)


def sem_(vals):
    v = np.asarray(vals, float)
    return v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0


def agg(vals):
    v = np.asarray(vals, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0), v.size


def key_of(r, *fields):
    return tuple(r[f] for f in fields)


REF = dict(R=0.95, eps=0.01, tau=200.0, L=2, p0=0.85, T=400000)


def is_ref(r, **over):
    d = dict(REF)
    d.update(over)
    return all(abs(r[k] - v) < 1e-12 if isinstance(v, float) else r[k] == v
               for k, v in d.items())


if __name__ == "__main__":
    recs = load_all()
    print(f"{len(recs)} runs in {DIR}\n")

    # ---------------------------------------------------- 1. main table
    print("=" * 100)
    print("1  n=600, converged runs (T=4e5), 3 growth rules, mean +- s.e.m. "
          "over independent realisations")
    print("=" * 100)
    print(f"  {'c':>6} {'model':>6} {'reals':>6} {'coupled edges':>20} "
          f"{'S':>20} {'S2':>9} {'#comp>=3':>9}")
    sel = [r for r in recs if r["n"] == 600 and is_ref(r)]
    by = defaultdict(list)
    for r in sel:
        by[(r["c"], r["model"])].append(r)
    for (c, model) in sorted(by):
        obs = [observables(r) for r in by[(c, model)]]
        fm, fe, nn = agg([o["frac"] for o in obs])
        sm, se_, _ = agg([o["S"] for o in obs])
        s2m, _, _ = agg([o["S2"] for o in obs])
        ncm, _, _ = agg([o["ncomp"] for o in obs])
        print(f"  {c:>6.3f} {model:>6} {nn:>6} {fm:>12.3f} +- {fe:<5.3f} "
              f"{sm:>12.3f} +- {se_:<5.3f} {s2m:>9.3f} {ncm:>9.1f}")

    # ------------------------------------------- 2. threshold robustness
    print("\n" + "=" * 100)
    print("2  ROBUSTNESS OF THE COUPLED-EDGE CRITERION (n=600, CONF)")
    print("=" * 100)
    ths = (1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3)
    print(f"  {'c':>6} | " + " ".join(f"thr={t:g}".rjust(16) for t in ths))
    for c in sorted({r["c"] for r in sel if r["model"] == "CONF"}):
        rs = [r for r in sel if r["model"] == "CONF" and r["c"] == c]
        cells = []
        for t in ths:
            obs = [observables(r, thresh=t) for r in rs]
            sm, se_, _ = agg([o["S"] for o in obs])
            cells.append(f"{sm:.3f}+-{se_:.3f}")
        print(f"  {c:>6.3f} | " + " ".join(x.rjust(16) for x in cells))

    # -------------------------------------------- 3. per-edge rate histogram
    print("\n" + "=" * 100)
    print("3  DISTRIBUTION OF PER-EDGE INTERACTION RATES (n=600, CONF): is the")
    print("   coupled/solitary split a real gap or an artefact of the cut?")
    print("=" * 100)
    for c in sorted({r["c"] for r in sel if r["model"] == "CONF"}):
        rs = [r for r in sel if r["model"] == "CONF" and r["c"] == c]
        rate = np.concatenate([observables(r)["rate"] for r in rs])
        edges_b = [0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1.0]
        h = np.histogram(rate, bins=edges_b)[0] / rate.size
        print(f"  c={c:>6.3f}  " + "  ".join(
            f"[{edges_b[i]:g},{edges_b[i+1]:g}):{h[i]:.3f}" for i in range(len(h))))

    # ------------------------------------------------ 4. degree-resolved
    print("\n" + "=" * 100)
    print("4  DEGREE-RESOLVED SURVIVAL at c=0.020")
    print("=" * 100)
    for model in ("LS", "CONF", "BA"):
        rs = [r for r in sel if r["model"] == model and abs(r["c"] - 0.020) < 1e-9]
        if not rs:
            continue
        acc = defaultdict(list)
        for r in rs:
            for b, v in observables(r)["surv"].items():
                acc[b].append(v)
        print(f"  {model:>6}: " + "  ".join(
            f"k~{b}: {np.mean(v):.3f}+-{sem_(v):.3f}"
            for b, v in sorted(acc.items())))

    # ------------------------------------------------ 5. finite size
    print("\n" + "=" * 100)
    print("5  FINITE SIZE (LS and CONF)")
    print("=" * 100)
    print(f"  {'model':>6} {'c':>6} {'n':>6} {'reals':>6} {'coupled edges':>18} "
          f"{'S':>18} {'S sd':>8}")
    fs = [r for r in recs if is_ref(r)]
    by = defaultdict(list)
    for r in fs:
        by[(r["model"], r["c"], r["n"])].append(r)
    for k in sorted(by):
        if k[0] == "BA":
            continue
        obs = [observables(r) for r in by[k]]
        fm, fe, nn = agg([o["frac"] for o in obs])
        Sv = np.array([o["S"] for o in obs])
        sm, se_, _ = agg(Sv)
        print(f"  {k[0]:>6} {k[1]:>6.3f} {k[2]:>6} {nn:>6} "
              f"{fm:>11.3f} +- {fe:<5.3f} {sm:>11.3f} +- {se_:<5.3f} "
              f"{Sv.std(ddof=1) if Sv.size>1 else 0:>8.3f}")

    # ------------------------------------------------ 6. run length
    print("\n" + "=" * 100)
    print("6  DEPENDENCE ON RUN LENGTH (CONF, n=600)")
    print("=" * 100)
    by = defaultdict(list)
    for r in recs:
        if r["model"] == "CONF" and r["n"] == 600 and is_ref(r, T=r["T"]):
            by[(r["c"], r["T"])].append(r)
    cs = sorted({k[0] for k in by})
    Ts = sorted({k[1] for k in by})
    print(f"  {'c':>6} | " + " ".join(f"T={T:g}".rjust(16) for T in Ts))
    for c in cs:
        cells = []
        for T in Ts:
            rs = by.get((c, T), [])
            if not rs:
                cells.append("--")
                continue
            obs = [observables(r) for r in rs]
            fm, fe, _ = agg([o["frac"] for o in obs])
            cells.append(f"{fm:.3f}+-{fe:.3f}")
        print(f"  {c:>6.3f} | " + " ".join(x.rjust(16) for x in cells))

    # ------------------------------------------------ 7. initial condition
    print("\n" + "=" * 100)
    print("7  DEPENDENCE ON THE INITIAL BELIEF (CONF, n=600, T=4e5)")
    print("=" * 100)
    by = defaultdict(list)
    for r in recs:
        if r["model"] == "CONF" and r["n"] == 600 and is_ref(r, p0=r["p0"]):
            by[(r["c"], r["p0"])].append(r)
    p0s = sorted({k[1] for k in by})
    print(f"  {'c':>6} | " + " ".join(f"p0={p:g}".rjust(16) for p in p0s))
    for c in sorted({k[0] for k in by}):
        cells = []
        for p in p0s:
            rs = by.get((c, p), [])
            if not rs:
                cells.append("--")
                continue
            obs = [observables(r) for r in rs]
            fm, fe, _ = agg([o["frac"] for o in obs])
            cells.append(f"{fm:.3f}+-{fe:.3f}")
        print(f"  {c:>6.3f} | " + " ".join(x.rjust(16) for x in cells))

    # ------------------------------------------------ 8. sensitivity
    print("\n" + "=" * 100)
    print("8  SENSITIVITY, one parameter at a time (CONF, n=600, T=4e5):")
    print("   coupled-edge fraction at each c")
    print("=" * 100)
    for field, ref in (("eps", 0.01), ("tau", 200.0), ("R", 0.95), ("L", 2)):
        vals = sorted({r[field] for r in recs
                       if r["model"] == "CONF" and r["n"] == 600
                       and is_ref(r, **{field: r[field]})})
        if len(vals) < 2:
            continue
        cs = sorted({r["c"] for r in recs if r["model"] == "CONF"
                     and r["n"] == 600 and is_ref(r, **{field: r[field]})})
        print(f"\n  {field}:")
        print(f"  {'c':>6} | " + " ".join(f"{v:g}".rjust(14) for v in vals))
        for c in cs:
            cells = []
            for v in vals:
                rs = [r for r in recs if r["model"] == "CONF" and r["n"] == 600
                      and r["c"] == c and is_ref(r, **{field: v})
                      and r[field] == v]
                if not rs:
                    cells.append("--")
                    continue
                obs = [observables(r) for r in rs]
                fm, fe, _ = agg([o["frac"] for o in obs])
                cells.append(f"{fm:.3f}+-{fe:.3f}")
            print(f"  {c:>6.3f} | " + " ".join(x.rjust(14) for x in cells))

    # ------------------------------------------------ 9. relaxation and the
    #     global collapse time, from the recorded time series
    print("\n" + "=" * 100)
    print("9  RELAXATION OF THE EXECUTED-EDGE FRACTION (n=600, T=4e5).")
    print("   The interevent times are heavy-tailed, so a silence-based death")
    print("   criterion for a single edge is ambiguous; the global execution")
    print("   rate is not. s0 is the first window, s1 the last, and T_half the")
    print("   first window in which the rate has fallen below s0/2.")
    print("=" * 100)
    print(f"  {'model':>6} {'c':>6} {'N':>4} {'s0 (x1e3)':>18} {'s1 (x1e3)':>18} "
          f"{'s1/s0':>14} {'T_half':>18} {'never':>6}")
    for model in ("LS", "CONF", "BA"):
        for c in sorted({r["c"] for r in sel if r["model"] == model}):
            rs = [r for r in sel if r["model"] == model and r["c"] == c]
            if not rs:
                continue
            s0s, s1s, ratio, thalf, never = [], [], [], [], 0
            for r in rs:
                o = observables(r)
                ser, win = o["series"], o["window"]
                s0 = float(ser[0])
                s1 = float(ser[-5:].mean())
                s0s.append(s0)
                s1s.append(s1)
                ratio.append(s1 / s0 if s0 > 0 else np.nan)
                hit = np.flatnonzero(ser < 0.5 * s0)
                if hit.size:
                    thalf.append(float(hit[0] * win))
                else:
                    never += 1
            m0, e0, _ = agg(s0s)
            m1, e1, _ = agg(s1s)
            mr, er, _ = agg(ratio)
            th = (f"{np.mean(thalf):.3g}+-{sem_(thalf):.2g}" if thalf else "--")
            print(f"  {model:>6} {c:>6.3f} {len(rs):>4} "
                  f"{1e3*m0:>10.3f} +- {1e3*e0:<5.3f} "
                  f"{1e3*m1:>10.3f} +- {1e3*e1:<5.3f} "
                  f"{mr:>8.3f} +- {er:<5.3f} {th:>18} {never:>6}")

    print("\n" + "=" * 100)
    print("10  TIME SERIES of the executed-edge fraction, CONF n=600, in units")
    print("    of 1e-3, averaged over realisations, at 10 points of the run")
    print("=" * 100)
    for c in sorted({r["c"] for r in sel if r["model"] == "CONF"}):
        rs = [r for r in sel if r["model"] == "CONF" and r["c"] == c]
        S = np.array([observables(r)["series"] for r in rs])
        k = S.shape[1]
        idx = np.linspace(0, k - 1, 10).astype(int)
        print(f"  c={c:>6.3f}  " + " ".join(f"{1e3*S[:, i].mean():7.3f}"
                                            for i in idx))
