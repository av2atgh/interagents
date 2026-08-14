"""Figures for manuscript.tex.

fig_schematic.pdf  the three settings: pair, network, hypergraph  (schematic)
fig_phase.pdf      return map and critical line                   (analytic)
fig_percolation.pdf order parameter vs control parameter          (measured)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "figure.dpi": 200})

ACC = "#c0392b"      # interacting task
GREY = "#95a5a6"     # private tasks
DARK = "#2c3e50"


# ===================================================== fig 1: schematic
def queue(ax, x, y, label, n_priv=4, h=0.085, w=0.30):
    """A priority queue: one interacting task (red) over n_priv private ones."""
    ax.add_patch(Rectangle((x, y), w, h, fc=ACC, ec=DARK, lw=0.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, "I", ha="center", va="center",
            fontsize=6.5, color="white", zorder=4)
    for k in range(n_priv):
        yy = y - (k + 1) * (h + 0.012)
        ax.add_patch(Rectangle((x, yy), w, h, fc=GREY, ec=DARK, lw=0.4,
                               alpha=0.55, zorder=3))
    ax.text(x + w / 2, y + h + 0.045, label, ha="center", fontsize=8.5,
            color=DARK)
    ax.text(x + w + 0.05, y - 2.2 * (h + 0.012), r"$O$", ha="left",
            va="center", fontsize=7, color=DARK)


fig = plt.figure(figsize=(7.0, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1, 1], wspace=0.28)

# --- (a) two agents
ax = fig.add_subplot(gs[0])
ax.set_xlim(-0.05, 1.55)
ax.set_ylim(-0.72, 0.95)
ax.axis("off")
queue(ax, 0.16, 0.34, "$A$")
queue(ax, 1.02, 0.34, "$B$")
ax.add_patch(FancyArrowPatch((0.48, 0.425), (1.00, 0.425),
                             arrowstyle="<|-|>", mutation_scale=7,
                             lw=1.0, color=ACC, zorder=5))
ax.text(0.74, 0.63, "must coincide", ha="center", fontsize=6.5, color=ACC)
ax.text(0.75, -0.64, "(a) pair", ha="center", fontsize=8, color=DARK)

# --- (b) network
ax = fig.add_subplot(gs[1])
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.45, 1.15)
ax.axis("off")
th = np.linspace(0, 2 * np.pi, 9)[:-1]
pos = np.vstack([np.c_[np.cos(th), np.sin(th)] * 0.86, [[0.0, 0.0]]])
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
         (8, 0), (8, 2), (8, 4), (8, 6), (1, 8), (3, 5)]
coupled = {(0, 1), (2, 3), (4, 5), (8, 0), (8, 4), (6, 7)}
for i, j in edges:
    live = (i, j) in coupled or (j, i) in coupled
    ax.plot(*zip(pos[i], pos[j]), lw=1.5 if live else 0.6,
            color=ACC if live else GREY, ls="-" if live else (0, (2, 2)),
            zorder=1, alpha=1.0 if live else 0.8)
for x, y in pos:
    ax.add_patch(plt.Circle((x, y), 0.115, fc="white", ec=DARK, lw=0.7, zorder=3))
ax.text(0.0, -1.36, "(b) network", ha="center", fontsize=8, color=DARK)

# --- (c) hypergraph: three groups sharing one agent, so that agent has d = 3
ax = fig.add_subplot(gs[2])
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.45, 1.15)
ax.axis("off")
cen = np.array([0.0, 0.0])
pts = [cen]
for base in (90.0, 210.0, 330.0):
    for off in (-26.0, 26.0):
        t = np.radians(base + off)
        pts.append([0.88 * np.cos(t), 0.88 * np.sin(t)])
pts = np.array(pts)
for g in ((0, 1, 2), (0, 3, 4), (0, 5, 6)):
    P = pts[list(g)]
    m = P.mean(axis=0)
    ax.add_patch(Polygon(m + (P - m) * 1.42, closed=True, fc=ACC, ec=ACC,
                         lw=0.8, alpha=0.18, zorder=1, joinstyle="round"))
for x, y in pts:
    ax.add_patch(plt.Circle((x, y), 0.115, fc="white", ec=DARK, lw=0.7, zorder=3))
ax.annotate(r"$d=3$", xy=(0.0, 0.0), xytext=(0.62, -0.72), fontsize=7,
            color=DARK, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.text(0.0, -1.36, r"(c) hypergraph, $m=3$", ha="center", fontsize=8, color=DARK)

fig.savefig("fig_schematic.pdf", bbox_inches="tight")
print("wrote fig_schematic.pdf")


# ===================================================== fig 2: phase structure
def F(p, R, c, a):
    return np.clip(R - c * (1 - p) / np.maximum(p, 1e-12), 0, 1) ** a


def R_crit(a, c):
    p = (a * c) ** (a / (a + 1.0))
    return (p ** (1.0 / a) + c / p - c, p) if p <= 1 else (np.nan, np.nan)


fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
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


# ================================================ fig 3: percolation transition
# Measured values: run_percolation.py (40k steps, 3 realisations) for the
# LS / randomized / BA comparison, and the converged 400k-step runs on the
# randomized network reported in the text.
c_short = np.array([0.002, 0.005, 0.010, 0.020, 0.030, 0.045, 0.070])
S_ls = np.array([1.000, 1.000, 1.000, 0.569, 0.006, 0.003, 0.000])
S_cf = np.array([1.000, 1.000, 1.000, 0.976, 0.006, 0.003, 0.000])
S_ba = np.array([1.000, 1.000, 1.000, 0.967, 0.007, 0.003, 0.000])
c_long = np.array([0.020, 0.030, 0.045, 0.060, 0.080])
S_long = np.array([0.997, 0.007, 0.000, 0.000, 0.000])
E_long = np.array([0.825, 0.106, 0.000, 0.000, 0.000])

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax[0].plot(c_short, S_ls, "o-", ms=3.5, lw=1.1, label=r"$LS(\ell=1)$")
ax[0].plot(c_short, S_cf, "s--", ms=3.5, lw=1.1, label="randomized")
ax[0].plot(c_short, S_ba, "^:", ms=3.5, lw=1.1, label=r"$BA(m=2)$")
ax[0].axvspan(0.019, 0.021, color=ACC, alpha=0.12, lw=0)
ax[0].annotate("same coupled edge\nfraction, different $S$",
               xy=(0.020, 0.55), xytext=(0.032, 0.62), fontsize=6.5,
               arrowprops=dict(arrowstyle="->", lw=0.6))
ax[0].set_xlabel(r"$c$")
ax[0].set_ylabel(r"$S$")
ax[0].set_title("(a) local structure fragments the coupled phase",
                fontsize=8, loc="left")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")
ax[0].set_ylim(-0.05, 1.12)

ax[1].plot(c_long, S_long, "o-", ms=4, lw=1.2, color=DARK, label=r"$S$ (measured)")
ax[1].plot(c_long, E_long, "s--", ms=4, lw=1.0, color=GREY,
           label="coupled edge fraction")
ax[1].axvline(0.025, color=ACC, lw=1.0, ls="-")
ax[1].text(0.0255, 0.55, "measured\n$c_c$", fontsize=6.5, color=ACC)
ax[1].axvline(0.062, color="0.45", lw=1.0, ls="--")
ax[1].text(0.064, 0.55, "mean-field\n$c_c$", fontsize=6.5, color="0.35")
ax[1].set_xlabel(r"$c$")
ax[1].set_ylabel("order parameter")
ax[1].set_title("(b) converged transition vs mean field", fontsize=8, loc="left")
ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
ax[1].set_ylim(-0.05, 1.12)
fig.tight_layout()
fig.savefig("fig_percolation.pdf", bbox_inches="tight")
print("wrote fig_percolation.pdf")
