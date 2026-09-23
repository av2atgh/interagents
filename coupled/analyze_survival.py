"""Analysis of the pair-level collapse-time campaign (run_survival.py).

Escape into the absorbing state is a rare event, so the first-passage times are
right-censored: a realisation that has not collapsed by the end of the run
contributes exposure but no event. The mean collapse time is therefore estimated
by the exponential MLE with censoring,

    1/lambda_hat = (sum of all exposure times) / (number of collapses),

whose relative standard error is 1/sqrt(number of collapses). Exponentiality
itself is checked on the uncensored sample by the coefficient of variation,
which is 1 for an exponential law.
"""

import json
import sys

import numpy as np

IN = sys.argv[1] if len(sys.argv) > 1 else "results_survival.json"
D = json.load(open(IN))


def mle(rows):
    """Censored exponential MLE of the mean first-passage time."""
    ev = [r["t_collapse"] for r in rows if r["t_collapse"] is not None]
    exposure = sum(r["t_collapse"] if r["t_collapse"] is not None else r["n_steps"]
                   for r in rows)
    k = len(ev)
    if k == 0:
        return dict(n=len(rows), k=0, mean=np.inf, se=np.nan, cv=np.nan,
                    exposure=exposure, frac=0.0, median=np.nan)
    m = exposure / k
    ev = np.array(ev, float)
    return dict(n=len(rows), k=k, mean=m, se=m / np.sqrt(k),
                cv=float(ev.std(ddof=1) / ev.mean()) if k > 2 else np.nan,
                exposure=exposure, frac=k / len(rows),
                median=float(np.median(ev)))


def group(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return dict(sorted(out.items()))


def sem(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0


def line(lbl, rows):
    m = mle(rows)
    phi = np.array([r["phi"] for r in rows])
    print(f"  {lbl:>10}  n={m['n']:>4}  collapsed={m['k']:>4} ({m['frac']:.3f})  "
          f"<T_c>={m['mean']:>12.4g} +- {m['se']:<10.3g} CV={m['cv']:>5.2f}  "
          f"med={m['median']:>10.4g}  "
          f"phi={phi.mean():.3f}+-{sem(phi):.3f}  "
          f"[phi<0.25]={(phi < 0.25).mean():.3f} [phi>0.75]={(phi > 0.75).mean():.3f}")


print("=" * 118)
print("1  REFERENCE OPERATING POINT  L=2, c=0.35, R_I=0.8428, eps=1e-2, "
      "tau_mem=200, T=2e5")
print("=" * 118)
rows = D["fpt_reference"]
line("all", rows)
ev = np.array([r["t_collapse"] for r in rows if r["t_collapse"] is not None],
              float)
if ev.size:
    print(f"     first-passage sample: n={ev.size}  mean={ev.mean():.4g}  "
          f"sd={ev.std(ddof=1):.4g}  CV={ev.std(ddof=1)/ev.mean():.3f}  "
          f"min={ev.min():.0f}  max={ev.max():.0f}")
    q = np.quantile(ev, [0.1, 0.25, 0.5, 0.75, 0.9])
    print("     quantiles (10/25/50/75/90): " + " ".join(f"{x:.4g}" for x in q))
    # information in a prefix: fraction of collapses occurring after t
    for t in (1e3, 3e3, 1e4, 3e4, 6e4, 1.2e5):
        print(f"     collapses after t={t:>8.0f}: "
              f"{(ev > t).sum():>4} of {ev.size}  ({(ev > t).mean():.3f})")

for name, key, lbl in (("2  MEMORY TIME", "tau_scan", "tau"),
                       ("3  INTERACTION VALUE", "R_scan", "R_I"),
                       ("4  EXPLORATION RATE", "eps_scan", "eps"),
                       ("5  INITIAL BELIEF", "p_init_scan", "p_init"),
                       ("6  RUN LENGTH", "T_scan", "n_steps")):
    print("\n" + "=" * 118)
    print(name)
    print("=" * 118)
    for v, rs in group(D[key], lbl).items():
        line(f"{lbl}={v:g}", rs)
