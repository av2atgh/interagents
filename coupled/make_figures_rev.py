"""Figures added for the revised manuscript.

fig_survival.pdf    escape from the coupled state is a rare event: survival
                    curve, first-passage density, and the mean collapse time as
                    a function of the memory time and of the distance above R_c
fig_finitesize.pdf  network transition with error bars over independent
                    realisations, at four system sizes, plus the distribution of
                    the largest coupled component and the run-length dependence
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "figure.dpi": 200})

ACC = "#c0392b"
DARK = "#2c3e50"
BLUE = "#2471a3"
GREEN = "#1e8449"
ORANGE = "#d68910"


def mle(rows):
    ev = [r["t_collapse"] for r in rows if r["t_collapse"] is not None]
    exp = sum(r["t_collapse"] if r["t_collapse"] is not None else r["n_steps"]
              for r in rows)
    k = len(ev)
    if k == 0:
        return np.nan, np.nan, 0
    return exp / k, exp / k / np.sqrt(k), k


def group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return dict(sorted(out.items()))


# =========================================================== fig_survival
def fig_survival(path="results_survival.json", out="fig_survival.pdf"):
    D = json.load(open(path))
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.075, right=0.985,
                          bottom=0.21, top=0.90)

    # --- (a) survival of the coupled state: symbols for the measurement,
    #         line for the exponential with the censored-MLE rate
    ax = fig.add_subplot(gs[0])
    rows = D["fpt_reference"]
    T = max(r["n_steps"] for r in rows)
    ev = np.sort([r["t_collapse"] for r in rows if r["t_collapse"] is not None])
    N = len(rows)
    m, _, k = mle(rows)
    tt = np.linspace(0, T, 200)
    ax.plot(tt / 1e3, np.exp(-tt / m), color=DARK, lw=0.9, ls="--",
            label=r"$e^{-t/\langle T_c\rangle}$")
    # empirical survival sampled on a regular grid, so the marks are legible
    ts = np.linspace(0, T, 18)
    surv = 1.0 - np.searchsorted(ev, ts, side="right") / N
    ax.plot(ts / 1e3, surv, "o", ms=3.5, color=ACC, mfc="white", mew=1.0,
            label=f"measured, $N={N}$")
    ax.set_xlabel(r"$t$ ($10^3$ steps)")
    ax.set_ylabel(r"$\Pr(T_c>t)$")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_title("(a)", loc="left", fontsize=8)

    # --- (b) fraction collapsed against run length, against a single rate
    ax = fig.add_subplot(gs[1])
    rows = D["T_scan"]
    k = sum(1 for r in rows if r["t_collapse"] is not None)
    expo = sum(r["t_collapse"] if r["t_collapse"] is not None else r["n_steps"]
               for r in rows)
    lam = k / expo
    g = group(rows, "n_steps")
    Ts = np.array(sorted(g))
    f = np.array([np.mean([r["t_collapse"] is not None for r in g[T_]])
                  for T_ in Ts])
    nn = np.array([len(g[T_]) for T_ in Ts], float)
    err = np.sqrt(f * (1 - f) / nn)
    tt = np.geomspace(Ts.min() * 0.7, Ts.max() * 1.4, 200)
    ax.plot(tt, 1 - np.exp(-lam * tt), color=DARK, lw=0.9, ls="--",
            label=r"$1-e^{-T/\langle T_c\rangle}$")
    ax.errorbar(Ts, f, yerr=err, fmt="o", ms=3.5, color=ORANGE, mfc="white",
                capsize=2, lw=0.9, label="measured")
    ax.set_xscale("log")
    ax.set_xlabel(r"run length $T$ (steps)")
    ax.set_ylabel(r"$\Pr(T_c\leq T)$")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_ylim(0, 0.78)
    ax.set_title("(b)", loc="left", fontsize=8)

    # --- (c) mean collapse time vs tau_mem, with lower bounds where no
    #         realisation collapsed
    ax = fig.add_subplot(gs[2])
    g = group(D["tau_scan"], "tau")
    xs, ys, es, bx, by_ = [], [], [], [], []
    for tau, rs in g.items():
        m_, se, k_ = mle(rs)
        if k_:
            xs.append(tau)
            ys.append(m_)
            es.append(se)
        else:
            bx.append(tau)
            by_.append(sum(r["n_steps"] for r in rs))
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3.5, lw=0.9, color=BLUE,
                capsize=2, mfc="white")
    if bx:
        ax.errorbar(bx, by_, yerr=[[0.0] * len(bx), [0.6 * y for y in by_]],
                    fmt="^", ms=4, color=BLUE, mfc="none", lw=0.9,
                    uplims=False, lolims=True)
        ax.text(0.5 * (bx[0] + bx[-1]), by_[0] * 0.06,
                "no collapse observed\n(lower bounds)",
                fontsize=6.0, color=BLUE, ha="center")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_{\rm mem}$")
    ax.set_ylabel(r"$\langle T_c\rangle$ (steps)")
    ax.set_title("(c)", loc="left", fontsize=8)

    fig.savefig(out)
    print("wrote", out)


# ========================================================= fig_finitesize
PAT = None


def _load_net(DIR="results_net"):
    import re
    pat = re.compile(r"(?P<model>[A-Z]+)_n(?P<n>\d+)_c(?P<c>[\d.]+)_s(?P<seed>\d+)"
                     r"_T(?P<T>\d+)_R(?P<R>[\d.]+)_e(?P<e>[\d.e+-]+)_t(?P<t>[\d.]+)"
                     r"_L(?P<L>\d+)_p(?P<p>[\d.]+)\.npz")
    out = []
    for f in sorted(os.listdir(DIR)):
        m = pat.fullmatch(f)
        if not m:
            continue
        d = m.groupdict()
        out.append(dict(path=os.path.join(DIR, f), model=d["model"],
                        n=int(d["n"]), c=float(d["c"]), seed=int(d["seed"]),
                        T=int(d["T"]), R=float(d["R"]), eps=float(d["e"]),
                        tau=float(d["t"]), L=int(d["L"]), p0=float(d["p"])))
    return out


def _obs(rec, thresh=1e-3):
    z = np.load(rec["path"])
    src, dst, rate = z["src"], z["dst"], z["rate"]
    n = rec["n"]
    keep = rate > thresh
    edges = {(min(i, j), max(i, j)) for i, j, k in zip(src, dst, keep) if k}
    nb = defaultdict(list)
    for i, j in edges:
        nb[i].append(j)
        nb[j].append(i)
    seen, best = set(), 0
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
        best = max(best, sz)
    return len(edges) / (src.size // 2), best / n


def fig_finitesize(DIR="results_net", out="fig_finitesize.pdf"):
    recs = _load_net(DIR)
    ref = [r for r in recs if r["T"] == 400_000 and r["R"] == 0.95
           and abs(r["eps"] - 0.01) < 1e-12 and r["tau"] == 200.0
           and r["L"] == 2 and abs(r["p0"] - 0.85) < 1e-12]
    cache = {}

    def obs(r):
        if r["path"] not in cache:
            cache[r["path"]] = _obs(r)
        return cache[r["path"]]

    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.075, right=0.985,
                          bottom=0.21, top=0.90)

    # --- (a) S vs c at four sizes, CONF
    ax = fig.add_subplot(gs[0])
    cols = {300: "#d6a21a", 600: ORANGE, 1200: ACC, 2400: BLUE,
            4800: DARK}
    # a distinct marker as well as a distinct colour per size, so the sizes
    # stay separable in greyscale and where the curves overlap
    marks = {300: "o", 600: "s", 1200: "^", 2400: "D", 4800: "v"}
    SIZES = (300, 600, 1200, 2400, 4800)
    CFRAG = 0.025
    for n in SIZES:
        by = defaultdict(list)
        for r in ref:
            if r["model"] == "CONF" and r["n"] == n:
                by[r["c"]].append(obs(r)[1])
        if not by:
            continue
        cs = sorted(by)
        m = [np.mean(by[c]) for c in cs]
        e = [np.std(by[c], ddof=1) / np.sqrt(len(by[c])) if len(by[c]) > 1 else 0
             for c in cs]
        ax.errorbar(cs, m, yerr=e, fmt=marks[n] + "-", ms=3.2, lw=0.9,
                    capsize=2, color=cols[n], mfc="white", label=f"$n={n}$")
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$S$")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("(a) randomized", loc="left", fontsize=8)

    # --- (b) S vs c, LS
    ax = fig.add_subplot(gs[1])
    for n in SIZES:
        by = defaultdict(list)
        for r in ref:
            if r["model"] == "LS" and r["n"] == n:
                by[r["c"]].append(obs(r)[1])
        if not by:
            continue
        cs = sorted(by)
        m = [np.mean(by[c]) for c in cs]
        e = [np.std(by[c], ddof=1) / np.sqrt(len(by[c])) if len(by[c]) > 1 else 0
             for c in cs]
        ax.errorbar(cs, m, yerr=e, fmt=marks[n] + "-", ms=3.2, lw=0.9,
                    capsize=2, color=cols[n], mfc="white", label=f"$n={n}$")
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$S$")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title(r"(b) $LS(\ell=1)$", loc="left", fontsize=8)

    # --- (c) distribution of S at c = 0.020, n = 600
    ax = fig.add_subplot(gs[2])
    for model, col, lbl in (("LS", ACC, r"$LS(\ell=1)$"),
                            ("CONF", BLUE, "randomized")):
        v = [obs(r)[1] for r in ref
             if r["model"] == model and r["n"] == 600
             and abs(r["c"] - CFRAG) < 1e-9]
        if not v:
            continue
        ax.hist(v, bins=np.linspace(0, 1, 21), histtype="step", lw=1.1,
                color=col, label=lbl)
    ax.set_xlabel(rf"$S$ at $c={CFRAG:g}$")
    ax.set_ylabel("realisations")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("(c)", loc="left", fontsize=8)

    fig.savefig(out)
    print("wrote", out)


# ============================================================== fig_levels
def fig_levels(path="results_replicates.json", out="fig_levels.pdf"):
    """Asymmetric pairing: what one level-2 agent does to a pair.

    Replaces the table of the first submission. Three panels, one per
    observable; three groups of bars, one per interaction value; three bars per
    group, one per pairing of modelling levels.
    """
    D = json.load(open(path))["asym"]
    g = defaultdict(list)
    for r in D:
        g[(r["R_I"], r["lvA"], r["lvB"])].append(r)
    Rs = sorted({k[0] for k in g})
    lev = [(1, 1), (2, 1), (2, 2)]
    cols = {(1, 1): "#7f8c8d", (2, 1): ORANGE, (2, 2): BLUE}

    def stat(rs, key):
        v = np.array([r[key] for r in rs], float)
        return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0)

    def coupled(rs):
        k = sum(1 for r in rs if r["phi"] > 0.75)
        n = len(rs)
        p = k / n
        return p, np.sqrt(max(p * (1 - p), 1e-12) / n)

    panels = [(r"$\Pr(\phi>3/4)$", coupled),
              (r"$\phi$", lambda rs: stat(rs, "phi")),
              (r"$\bar p$", lambda rs: stat(rs, "pbar"))]

    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.75), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.22, top=0.78,
                        wspace=0.12)
    x = np.arange(len(Rs))
    w = 0.26            # pitch between bar centres within a group
    bw = 0.82 * w       # drawn width, leaving a small gap between columns
    handles = None
    for ax, (lbl, fn), tag in zip(axes, panels, "abc"):
        for i, lv in enumerate(lev):
            m = np.array([fn(g[(R, *lv)])[0] for R in Rs])
            e = np.array([fn(g[(R, *lv)])[1] for R in Rs])
            xs = x + (i - 1) * w
            # the y axis runs slightly below zero, so that a bar of height
            # zero still shows its outline rather than vanishing into the
            # spine: a level-1 pair collapses in every realisation
            ax.bar(xs, m, bw, yerr=e, color=cols[lv], edgecolor=DARK,
                   linewidth=0.6, error_kw=dict(lw=0.7),
                   label=f"$({lv[0]},{lv[1]})$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{R:g}" for R in Rs], fontsize=6.5)
        ax.set_xlabel(r"$R_I$", labelpad=1.5)
        ax.set_title(f"({tag}) {lbl}", loc="left", fontsize=7.5, pad=3)
        ax.set_ylim(-0.075, 1.12)
        # leave room on the left so the inward y ticks clear the first group,
        # and drop the x ticks, which would sit under the bars
        ax.set_xlim(-0.82, len(Rs) - 1 + 0.55)
        ax.tick_params(labelsize=6.5)
        ax.tick_params(axis="x", length=0)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    axes[0].set_yticks([0, 0.5, 1.0])
    fig.legend(*handles, frameon=False, fontsize=6.5, ncol=3,
               loc="upper center", bbox_to_anchor=(0.55, 1.02),
               handlelength=1.0, handletextpad=0.4, columnspacing=1.4)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


# ======================================================== fig_percolation
def fig_percolation(DIR="results_net", out="fig_percolation.pdf"):
    """Regenerated from the stored runs: 20 realisations per point, converged
    runs, error bars, and the two mean-field thresholds marked."""
    recs = _load_net(DIR)
    ref = [r for r in recs if r["T"] == 400_000 and r["R"] == 0.95
           and abs(r["eps"] - 0.01) < 1e-12 and r["tau"] == 200.0
           and r["L"] == 2 and abs(r["p0"] - 0.85) < 1e-12 and r["n"] == 600]
    cache = {}

    def obs(r):
        if r["path"] not in cache:
            cache[r["path"]] = _obs(r)
        return cache[r["path"]]

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.30))
    styles = {"LS": ("o-", ACC, r"$LS(\ell=1)$"),
              "CONF": ("s--", BLUE, "randomized"),
              "BA": ("^:", GREEN, r"$BA(m=2)$")}
    for model, (fmt, col, lbl) in styles.items():
        by = defaultdict(list)
        for r in ref:
            if r["model"] == model:
                by[r["c"]].append(obs(r)[1])
        if not by:
            continue
        cs = sorted(by)
        m = [np.mean(by[c]) for c in cs]
        e = [np.std(by[c], ddof=1) / np.sqrt(len(by[c])) if len(by[c]) > 1 else 0
             for c in cs]
        ax[0].errorbar(cs, m, yerr=e, fmt=fmt, ms=3.5, lw=1.1, capsize=2,
                       color=col, mfc="white", label=lbl)
    ax[0].set_xlabel(r"$c$")
    ax[0].set_ylabel(r"$S$")
    ax[0].set_title("(a) local structure fragments first", fontsize=8,
                    loc="left")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    ax[0].set_ylim(-0.05, 1.12)

    # --- (b) per-edge rate distribution: the coupled/solitary split
    bins = np.geomspace(1e-5, 1.0, 31)
    for c, col, ls in ((0.002, BLUE, "-"), (0.025, ACC, "-"),
                       (0.070, DARK, "--")):
        rs = [r for r in ref if r["model"] == "LS" and abs(r["c"] - c) < 1e-9]
        if not rs:
            continue
        rate = np.concatenate([np.load(r["path"])["rate"] for r in rs])
        rate = np.maximum(rate, 1.1e-5)
        h, _ = np.histogram(rate, bins=bins)
        ax[1].step(bins[:-1], h / rate.size, where="post", color=col, ls=ls,
                   lw=1.1, label=f"$c={c:g}$")
    ax[1].axvline(1e-3, color="0.35", lw=1.0, ls=":",
                  label=r"cut, $10\epsilon^2$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$w$")
    ax[1].set_ylabel(r"$P(w)$")
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    ax[1].set_title("(b) the split is a gap", fontsize=8, loc="left")

    # --- (c) order parameters against the two mean-field thresholds
    by_S, by_f = defaultdict(list), defaultdict(list)
    for r in ref:
        if r["model"] != "CONF":
            continue
        f, S = obs(r)
        by_S[r["c"]].append(S)
        by_f[r["c"]].append(f)
    cs = sorted(by_S)
    hs, ls_ = [], []
    for d, fmt, col, lbl in ((by_S, "o-", DARK, r"$S$"),
                             (by_f, "s--", "#7f8c8d", r"$f_c$")):
        m = [np.mean(d[c]) for c in cs]
        e = [np.std(d[c], ddof=1) / np.sqrt(len(d[c])) if len(d[c]) > 1 else 0
             for c in cs]
        h = ax[2].errorbar(cs, m, yerr=e, fmt=fmt, ms=3.5, lw=1.1, capsize=2,
                           color=col, mfc="white", label=lbl)
        hs.append(h)
        ls_.append(lbl)
    for x, col, sty, lbl in ((0.041, "0.55", ":", "MR"),
                             (0.062, "0.45", "-.", "cavity")):
        hs.append(ax[2].axvline(x, color=col, lw=1.0, ls=sty, label=lbl))
        ls_.append(lbl)
    ax[2].set_xlabel(r"$c$")
    ax[2].set_ylabel(r"$S$, $f_c$")
    ax[2].set_title("(c) measured vs mean field", fontsize=8, loc="left")
    ax[2].legend(hs, ls_, frameon=False, fontsize=6.5, loc="lower left",
                 labelspacing=0.3, handlelength=1.4, handletextpad=0.35,
                 borderaxespad=0.4)
    ax[2].set_ylim(-0.05, 1.12)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "survival") and os.path.exists("results_survival.json"):
        fig_survival()
    if which in ("all", "levels") and os.path.exists("results_replicates.json"):
        fig_levels()
    if which in ("all", "finitesize") and os.path.isdir("results_net"):
        fig_finitesize()
    if which in ("all", "percolation") and os.path.isdir("results_net"):
        fig_percolation()
