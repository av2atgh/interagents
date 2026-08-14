"""Figures for manuscript.tex. Both panels are analytic -- no simulation."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "figure.dpi": 200})


def F(p, R, c, a):
    return np.clip(R - c * (1 - p) / np.maximum(p, 1e-12), 0, 1) ** a


def R_crit(a, c):
    p = (a * c) ** (a / (a + 1.0))
    return (p ** (1.0 / a) + c / p - c, p) if p <= 1 else (np.nan, np.nan)


# Sized for \textwidth in revtex4-2 aps twocolumn (~7.0 in), so the figure
# renders at ~1:1 and the 8 pt labels stay 8 pt on the page.
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))

# (a) return map at fixed a, several R_I, showing the saddle-node
c, a = 0.35, 1
p = np.linspace(1e-4, 1, 4000)
Rc, pstar = R_crit(a, c)
for R, ls in ((Rc - 0.06, ":"), (Rc, "-"), (Rc + 0.06, "--"), (Rc + 0.14, "-.")):
    ax[0].plot(p, F(p, R, c, a), ls, lw=1.1,
               label=rf"$R_I={R:.3f}$" + (r"$=R_c$" if abs(R - Rc) < 1e-9 else ""))
ax[0].plot(p, p, color="0.6", lw=0.7)
ax[0].plot([pstar], [pstar], "o", ms=4, mfc="none", color="k")
ax[0].annotate("saddle-node", xy=(pstar, pstar), xytext=(0.30, 0.80),
               arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=7)
ax[0].set_xlabel(r"$p$")
ax[0].set_ylabel(r"$\mathcal{F}(p)$")
ax[0].set_title(rf"(a) return map, $L=2$, $c={c}$", fontsize=8, loc="left")
ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)

# (b) critical line R_c(L) for several c, with the validity boundary a <= 1/c
for c in (0.05, 0.10, 0.20, 0.35):
    aa = np.arange(1, 25)
    vals = np.array([R_crit(x, c)[0] for x in aa])
    ok = ~np.isnan(vals)
    ax[1].plot(aa[ok] + 1, vals[ok], "o-", ms=2.5, lw=1.0, label=rf"$c={c}$")
    if ok.sum() < len(aa):
        ax[1].plot([aa[ok][-1] + 1], [vals[ok][-1]], "x", ms=6, color="k")
ax[1].axhline(1.0, color="0.6", lw=0.7, ls="--")
ax[1].text(2.2, 1.02, r"$R_I=1$ (clipping bound)", fontsize=6.5, color="0.35")
ax[1].set_xlabel(r"$L$")
ax[1].set_ylabel(r"$R_c$")
ax[1].set_title(r"(b) critical line; $\times$ marks $a=1/c$", fontsize=8, loc="left")
ax[1].legend(frameon=False, fontsize=6.5, loc="lower right")
ax[1].set_xlim(1.5, 22)
ax[1].set_ylim(top=1.07)

fig.tight_layout()
fig.savefig("fig_phase.pdf", bbox_inches="tight")
print("wrote fig_phase.pdf")
