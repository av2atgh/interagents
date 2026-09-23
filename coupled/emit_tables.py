"""Emit LaTeX table bodies from the stored results, so that nothing is
transcribed by hand."""

import json
import sys
from collections import defaultdict

import numpy as np


def _compact(val, err, dec):
    """0.843 +- 0.032 -> $0.843(32)$: the error in units of the last digit."""
    e = max(1, round(err * 10 ** dec))
    return f"${val:.{dec}f}({e})$"



def mle(rows):
    ev = [r["t_collapse"] for r in rows if r["t_collapse"] is not None]
    exp_ = sum(r["t_collapse"] if r["t_collapse"] is not None else r["n_steps"]
               for r in rows)
    k = len(ev)
    return (exp_ / k if k else np.inf), k, exp_


def fmt_T(m, k, expo):
    if k == 0:
        e = int(np.floor(np.log10(expo)))
        return rf"$>{expo/10**e:.0f}\times10^{{{e}}}$"
    e = int(np.floor(np.log10(m)))
    mant = m / 10 ** e
    rel = 1 / np.sqrt(k)
    return rf"${mant:.2f}(\pm{mant*rel:.2f})\times10^{{{e}}}$"


def sem(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0


def sensitivity_table(path="results_survival.json"):
    D = json.load(open(path))
    blocks = [
        (r"$\tau_{\rm mem}$", "tau_scan", "tau", "{:g}"),
        (r"$R_I$", "R_scan", "R_I", "{:.4f}"),
        (r"$\epsilon$", "eps_scan", "eps", "{:g}"),
        (r"$\hat p(0)$", "p_init_scan", "p_init", "{:.2f}"),
        (r"$T$", "T_scan", "n_steps", "{:g}"),
    ]
    out = []
    for label, key, field, fm in blocks:
        g = defaultdict(list)
        for r in D[key]:
            g[r[field]].append(r)
        first = True
        for v in sorted(g):
            rows = g[v]
            m, k, expo = mle(rows)
            phi = np.array([r["phi"] for r in rows])
            out.append(
                f"{label if first else ''} & {fm.format(v)} & {len(rows)} & "
                f"{k/len(rows):.3f} & {fmt_T(m, k, expo)} & "
                f"${phi.mean():.3f}\\pm{max(sem(phi),0.0005):.3f}$ \\\\")
            first = False
        out.append(r"\colrule")
    return "\n".join(out[:-1])




# ------------------------------------------------------------------ pair tables
def _load(path):
    return json.load(open(path))


def _grp(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def _ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


def asym_table(path="results_replicates.json"):
    g = _grp(_load(path)["asym"], ["R_I", "lvA", "lvB"])
    out = []
    for (R, a, b) in sorted(g):
        rs = g[(R, a, b)]
        phi, ephi = _ms([r["phi"] for r in rs])
        pb, epb = _ms([r["pbar"] for r in rs])
        frac = np.mean([r["phi"] > 0.75 for r in rs])
        phase = "coupled" if frac > 0.5 else "solitary"
        out.append(f"{R:.2f} & ({a},{b}) & {len(rs)} & {phase} & {frac:.2f} & "
                   f"${phi:.3f}\\pm{max(ephi,0.0005):.3f}$ & "
                   f"${pb:.3f}\\pm{max(epb,0.0005):.3f}$ \\\\")
    return "\n".join(out)


def hold_table(path="results_replicates.json"):
    rows = _load(path)["hold"]
    g = _grp(rows, ["inv_nu", "tau", "hold"])
    out = []
    for inv in sorted({k[0] for k in g}):
        cells = []
        for hold in (False, True):
            for tau in (50.0, 5000.0):
                rs = g[(inv, tau, hold)]
                phi, e = _ms([r["phi"] for r in rs])
                cells.append(f"${phi:.3f}\\pm{max(e,0.0005):.3f}$")
        out.append(f"{inv:.0f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def levels_table(path="results_replicates.json"):
    g = _grp(_load(path)["levels"], ["L", "R_I", "level"])
    Ls = sorted({k[0] for k in g})
    Rs = sorted({k[1] for k in g})
    out = []
    for R in Rs:
        cells = []
        for L in Ls:
            for lvl in (1, 2):
                rs = g[(L, R, lvl)]
                frac = np.mean([r["phi"] > 0.75 for r in rs])
                cells.append(f"{frac:.2f}")
        out.append(f"{R:.2f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def hyper_md_table(path="results_hyper.json"):
    g = _grp(_load(path)["md"], ["m", "d"])
    ds = sorted({k[1] for k in g})
    out = []
    for m in sorted({k[0] for k in g}):
        cells = []
        for d in ds:
            rs = g[(m, d)]
            v, e = _ms([r["coupled"] for r in rs])
            cells.append(f"${v:.3f}\\pm{max(e,0.0005):.3f}$")
        out.append(f"{m} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def hyper_tau_table(path="results_hyper.json"):
    g = _grp(_load(path)["tau"], ["c", "tau"])
    taus = sorted({k[1] for k in g})
    out = []
    for c in sorted({k[0] for k in g}):
        cells = []
        for t in taus:
            rs = g[(c, t)]
            v, e = _ms([r["rate"] for r in rs])
            cells.append(f"${v:.4f}\\pm{max(e,0.00005):.4f}$")
        out.append(f"{c:.3f} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def hyper_prop_table(path="results_hyper.json"):
    g = _grp(_load(path)["prop"], ["m", "mode"])
    out = []
    for m in sorted({k[0] for k in g}):
        cells = []
        for mode in ("const", "prop"):
            rs = g[(m, mode)]
            v, e = _ms([r["rate"] for r in rs])
            cells.append(f"${v:.4f}\\pm{max(e,0.00005):.4f}$")
        out.append(f"{m} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def boundary_table(path="results_boundary.json"):
    D = _load(path)
    g = _grp(D["rows"], ["L", "c", "R_I"])
    out = []
    for case in D["cases"]:
        L, c, Rc = case["L"], case["c"], case["Rc"]
        xs, ys = [], []
        for (LL, cc, R) in sorted(g):
            if LL != L or abs(cc - c) > 1e-12:
                continue
            rs = g[(LL, cc, R)]
            xs.append(R)
            ys.append(np.mean([r["phi"] > 0.75 for r in rs]))
        xs, ys = np.array(xs), np.array(ys)
        cross = np.nan
        for i in range(len(xs) - 1):
            if ys[i] < 0.5 <= ys[i + 1]:
                cross = xs[i] + (0.5 - ys[i]) * (xs[i + 1] - xs[i]) / \
                    (ys[i + 1] - ys[i])
                break
        n = len(g[(L, c, xs[0])])
        out.append(f"{L} & {c:.2f} & {Rc:.4f} & {case['pstar']:.4f} & {n} & "
                   f"{cross:.4f} & {cross - Rc:+.4f} \\\\")
    return "\n".join(out)




def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def noreturn_table(path="results_noreturn.json"):
    D = _load(path)
    reg = {(r["k"], r["gamma"]): r["min_p"] for r in D["region"]}
    g = _grp(D["cells"], ["k", "gamma"])
    ks = sorted({k[0] for k in g})
    gammas = sorted({k[1] for k in g})
    out = []
    for k in ks:
        cells = []
        for gam in gammas:
            mp = reg[(k, gam)]
            cells.append(f"${mp:.3f}$" if mp > 1e-4 else
                         f"${mp:.0e}$".replace("e-0", r"\times10^{-") + "}$"
                         if mp else "--")
        for gam in gammas:
            rs = g[(k, gam)]
            rec = sum(1 for r in rs if r["phi"] > 0.5)
            lo, hi = _wilson(rec, len(rs))
            cells.append(f"${100*rec/len(rs):.1f}\%$")
        out.append(f"{k} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def noreturn_ci(path="results_noreturn.json"):
    D = _load(path)
    g = _grp(D["cells"], ["k", "gamma"])
    lines = [f"collapsed realisations: {len(D['collapsed'])} of "
             f"{len(D['base'])}"]
    for key in sorted(g):
        rs = g[key]
        rec = sum(1 for r in rs if r["phi"] > 0.5)
        lo, hi = _wilson(rec, len(rs))
        lines.append(f"  k={key[0]:>4} gamma={key[1]:<6} N={len(rs):>4} "
                     f"recovered {rec:>4} = {rec/len(rs):.3f} "
                     f"(95% CI {lo:.3f}-{hi:.3f})")
    for r in D["region"]:
        lines.append(f"  region k={r['k']:>4} gamma={r['gamma']:<6} "
                     f"min p_hat = {r['min_p']}")
    return "\n".join(lines)


def occupancy_table(path="results_occupancy.json"):
    D = _load(path)
    rows = [r for r in D["rows"] if not np.isnan(r["alpha"])]
    out = []
    for ci, cs in enumerate(D["cases"]):
        rs = [r for r in rows if r["case"] == ci]
        phi = np.array([r["phi"] for r in rs])
        al = np.array([r["alpha"] for r in rs])
        if phi.std() < 1e-9 or len(rs) < 4:
            out.append(f"L={cs['L']}: no variation, alpha = {al.mean():.3f}"
                       f" +- {al.std(ddof=1):.3f} over {len(rs)}")
            continue
        r0 = np.corrcoef(phi, al)[0, 1]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(4000):
            idx = rng.integers(0, len(rs), len(rs))
            if phi[idx].std() < 1e-9:
                continue
            bs.append(np.corrcoef(phi[idx], al[idx])[0, 1])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        out.append(f"L={cs['L']} c={cs['c']}: N={len(rs)} "
                   f"corr(phi,alpha) = {r0:+.3f} (95% CI {lo:+.3f}, {hi:+.3f}) "
                   f"phi in [{phi.min():.3f},{phi.max():.3f}]")
        for lo_, hi_ in ((0.0, 0.35), (0.35, 0.8), (0.8, 1.001)):
            m = (phi >= lo_) & (phi < hi_)
            if m.sum() >= 2:
                out.append(f"    phi in [{lo_},{hi_}): N={m.sum()} "
                           f"alpha = {al[m].mean():.3f} +- "
                           f"{al[m].std(ddof=1)/np.sqrt(m.sum()):.3f} "
                           f"CV = {np.mean([r['cv'] for r,mm in zip(rs,m) if mm]):.2f}")
    return "\n".join(out)


def boundary_report(path="results_boundary.json"):
    D = _load(path)
    g = _grp(D["rows"], ["L", "c", "R_I"])
    lines = []
    for case in D["cases"]:
        L, c, Rc = case["L"], case["c"], case["Rc"]
        lines.append(f"L={L} c={c} R_c={Rc:.4f} p*={case['pstar']:.4f}")
        for (LL, cc, R) in sorted(g):
            if LL != L or abs(cc - c) > 1e-12:
                continue
            rs = g[(LL, cc, R)]
            f = np.mean([r["phi"] > 0.75 for r in rs])
            phi, e = _ms([r["phi"] for r in rs])
            lines.append(f"    R_I={R:.4f} (R-Rc={R-Rc:+.3f}) N={len(rs)} "
                         f"coupled={f:.2f} phi={phi:.3f}+-{e:.3f}")
    return "\n".join(lines)


TABLES = dict(sensitivity=sensitivity_table, asym=asym_table, hold=hold_table,
              levels=levels_table, hyper_md=hyper_md_table,
              hyper_tau=hyper_tau_table, hyper_prop=hyper_prop_table,
              boundary=boundary_table, noreturn=noreturn_table,
              noreturn_ci=noreturn_ci, occupancy=occupancy_table,
              boundary_report=boundary_report)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "sensitivity"
    print(TABLES[which]())
