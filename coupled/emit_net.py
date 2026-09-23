"""LaTeX table bodies for the network section, straight from the stored runs."""

import sys
from collections import defaultdict

import numpy as np


def _compact(val, err, dec):
    """0.843 +- 0.032 -> $0.843(32)$: the error in units of the last digit."""
    e = max(1, round(err * 10 ** dec))
    return f"${val:.{dec}f}({e})$"


from analyze_net import load_all, observables, is_ref, agg, sem_

RECS = load_all()
REFSEL = [r for r in RECS if is_ref(r)]
_CACHE = {}


def obs(r, thresh=1e-3):
    key = (r["path"], thresh)
    if key not in _CACHE:
        _CACHE[key] = observables(r, thresh=thresh)
    return _CACHE[key]


def pm(v, dec=3):
    m, e, nn = agg(v)
    return _compact(m, e, dec)


def percolation_table(n=600):
    """Coupled edge fraction and S for the three growth rules."""
    by = defaultdict(list)
    for r in REFSEL:
        if r["n"] == n:
            by[(r["c"], r["model"])].append(r)
    cs = sorted({k[0] for k in by})
    out = []
    for c in cs:
        cells = []
        for model in ("LS", "CONF", "BA"):
            rs = by.get((c, model), [])
            if not rs:
                cells += ["--", "--"]
                continue
            o = [obs(r) for r in rs]
            cells.append(pm([x["frac"] for x in o]))
            cells.append(pm([x["S"] for x in o]))
        out.append(f"{c:.3f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def finitesize_table():
    by = defaultdict(list)
    for r in REFSEL:
        if r["model"] in ("LS", "CONF"):
            by[(r["model"], r["c"], r["n"])].append(r)
    ns = sorted({k[2] for k in by})
    cs = sorted({k[1] for k in by if k[1] in (0.015, 0.020, 0.025, 0.030)})
    out = []
    for model in ("LS", "CONF"):
        for c in cs:
            cells = []
            for n in ns:
                rs = by.get((model, c, n), [])
                cells.append(pm([obs(r)["S"] for r in rs]) if rs else "--")
            lbl = (r"$LS(\ell=1)$" if model == "LS" else "randomized")
            out.append(f"{lbl if c == cs[0] else ''} & {c:.3f} & "
                       + " & ".join(cells) + r" \\")
        out.append(r"\colrule")
    return "\n".join(out[:-1]), ns


def robustness_table(n=600, model="CONF"):
    ths = (1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3)
    by = defaultdict(list)
    for r in REFSEL:
        if r["n"] == n and r["model"] == model:
            by[r["c"]].append(r)
    out = []
    for c in sorted(by):
        cells = [pm([obs(r, t)["S"] for r in by[c]], 3) for t in ths]
        out.append(f"{c:.3f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out), ths


def length_table(model="CONF", n=600):
    by = defaultdict(list)
    for r in RECS:
        if (r["model"] == model and r["n"] == n
                and is_ref(r, T=r["T"])):
            by[(r["c"], r["T"])].append(r)
    Ts = sorted({k[1] for k in by})
    cs = sorted({k[0] for k in by})
    out = []
    for c in cs:
        cells = []
        for T in Ts:
            rs = by.get((c, T), [])
            cells.append(pm([obs(r)["frac"] for r in rs]) if rs else "--")
        out.append(f"{c:.3f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out), Ts


def pinit_table(model="CONF", n=600):
    by = defaultdict(list)
    for r in RECS:
        if (r["model"] == model and r["n"] == n and is_ref(r, p0=r["p0"])):
            by[(r["c"], r["p0"])].append(r)
    p0s = sorted({k[1] for k in by})
    out = []
    for c in sorted({k[0] for k in by}):
        cells = []
        for p in p0s:
            rs = by.get((c, p), [])
            cells.append(pm([obs(r)["frac"] for r in rs]) if rs else "--")
        out.append(f"{c:.3f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out), p0s


def sensitivity_block(field, model="CONF", n=600):
    vals = sorted({r[field] for r in RECS if r["model"] == model
                   and r["n"] == n and is_ref(r, **{field: r[field]})})
    by = defaultdict(list)
    for r in RECS:
        if (r["model"] == model and r["n"] == n
                and is_ref(r, **{field: r[field]})):
            by[(r["c"], r[field])].append(r)
    out = []
    for c in sorted({k[0] for k in by}):
        cells = []
        for v in vals:
            rs = by.get((c, v), [])
            cells.append(pm([obs(r)["frac"] for r in rs]) if rs else "--")
        out.append(f"{c:.3f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out), vals


def degree_table(c=0.020, n=600):
    out = []
    for model, lbl in (("LS", r"$LS(\ell=1)$"), ("CONF", "randomized"),
                       ("BA", r"$BA(m=2)$")):
        rs = [r for r in REFSEL if r["model"] == model and r["n"] == n
              and abs(r["c"] - c) < 1e-9]
        if not rs:
            continue
        acc = defaultdict(list)
        for r in rs:
            for b, v in obs(r)["surv"].items():
                acc[b].append(v)
        cells = [pm(acc[b]) if b in acc else "--" for b in (2, 4, 8, 16)]
        out.append(f"{lbl} & {len(rs)} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "percolation"
    fns = dict(percolation=percolation_table, finitesize=finitesize_table,
               robustness=robustness_table, length=length_table,
               pinit=pinit_table, degree=degree_table)
    if which.startswith("sens:"):
        body, vals = sensitivity_block(which.split(":")[1])
        print("# columns:", vals)
        print(body)
    else:
        out = fns[which]()
        if isinstance(out, tuple):
            print("# columns:", out[1])
            print(out[0])
        else:
            print(out)
