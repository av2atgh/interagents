"""One-panel figure for the NetSci 2027 abstract.

Panel (a) of coupled/fig_percolation.pdf, drawn from the same stored runs in
coupled/results_net, with plain-language labels.
Writes netsci2027/fig_netsci.pdf and leaves the paper's figures untouched.

    python3 netsci2027/make_fig_netsci.py      # from the repo root
"""

import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COUPLED = os.path.join(HERE, "..", "coupled")
sys.path.insert(0, COUPLED)
import make_figures_rev as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def main(out=os.path.join(HERE, "fig_netsci.pdf")):
    recs = mf._load_net(os.path.join(COUPLED, "results_net"))
    ref = [r for r in recs if r["T"] == 400_000 and r["R"] == 0.95
           and abs(r["eps"] - 0.01) < 1e-12 and r["tau"] == 200.0
           and r["L"] == 2 and abs(r["p0"] - 0.85) < 1e-12 and r["n"] == 600]
    cache = {}

    def obs(r):
        if r["path"] not in cache:
            cache[r["path"]] = mf._obs(r)
        return cache[r["path"]]

    def mean_se(d, cs):
        m = [np.mean(d[c]) for c in cs]
        e = [np.std(d[c], ddof=1) / np.sqrt(len(d[c])) if len(d[c]) > 1 else 0
             for c in cs]
        return m, e

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    styles = {"LS": ("o-", mf.ACC, "triadic closure"),
              "CONF": ("s--", mf.BLUE, "same network, links reshuffled"),
              "BA": ("^:", mf.GREEN, "preferential attachment")}
    for model, (fmt, col, lbl) in styles.items():
        by = defaultdict(list)
        for r in ref:
            if r["model"] == model:
                by[r["c"]].append(obs(r)[1])
        cs = sorted(by)
        m, e = mean_se(by, cs)
        ax.errorbar(cs, m, yerr=e, fmt=fmt, ms=4, lw=1.2, capsize=2,
                    color=col, mfc="white", label=lbl)
    ax.set_xlabel("cost of being stood up")
    ax.set_ylabel("largest cluster of active ties")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_ylim(-0.05, 1.12)

    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
