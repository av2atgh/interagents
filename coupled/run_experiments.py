"""Experiments on the coupled pair (plan.md sections 3.4, 3.5, 6.1, 6.2, 7).

The learned system is a stag hunt with a saddle-node bifurcation. Sweeping the
interaction value R_I at fixed L crosses a critical line R_I^c(L):

    R_I < R_I^c  ->  SOLITARY phase only (the coupled fixed point does not exist)
    R_I > R_I^c  ->  two attractors; near the line they nearly merge, critical
                     slowing down sets in, and finite memory lets the pair
                     switch between them -> intermittency -> heavy-tailed
                     interevent times, i.e. the paper's power law reappears.
"""

import numpy as np
from coupled_agents import AgentConfig, run, alpha_of, regime

N = 150_000


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def row(res, pre):
    a = alpha_of(res["tau"])
    a_s = f"{a:5.2f}" if not np.isnan(a) else "   --"
    cv = (res["tau"].std() / res["tau"].mean()) if res["tau"].size > 10 else np.nan
    cv_s = f"{cv:6.2f}" if not np.isnan(cv) else "    --"
    print(f"{pre}  {regime(res):>9}  p={res['p_final']:.3f}  rate={res['rate']:.4f}  "
          f"alpha={a_s}  CV={cv_s}  payoff={0.5*(res['payoff_A']+res['payoff_B']):.3f}")


hdr("E1  PHASE DIAGRAM. Sweep interaction value R_I at fixed L, both level 1.\n"
    "    CV = coefficient of variation of interevent times. CV >> 1 means no\n"
    "    characteristic interaction timescale.")
for L in (2, 3):
    print(f"\n  L = {L}   (a = {L-1} competing solitary tasks)")
    print(f"  {'R_I':>5}")
    for R in (0.90, 0.94, 0.96, 0.97, 0.98, 0.99, 1.00, 1.02, 1.06):
        cfg = AgentConfig(L=L, level=1, R_I=R)
        row(run(cfg, cfg, n_steps=N, seed=1), f"  {R:>5.2f}")

hdr("E2  DOES THE SELF-MODEL DO WORK? (plan.md 3.4)\n"
    "    Same sweep, level 2 (tracks the partner's model of ME). If the\n"
    "    self-model has a function, it should shift the critical line down:\n"
    "    coupling survives at values of R_I where level-1 pairs collapse.")
for L in (2, 3):
    print(f"\n  L = {L}")
    print(f"  {'R_I':>5} {'lvl':>4}")
    for R in (0.90, 0.94, 0.96, 0.97, 0.98, 0.99, 1.00):
        for lvl in (1, 2):
            cfg = AgentConfig(L=L, level=lvl, R_I=R)
            row(run(cfg, cfg, n_steps=N, seed=2), f"  {R:>5.2f} {lvl:>4}")

hdr("E3  ASYMMETRY (the paper's max(L_j-1) lesson: the worse agent sets the tone)\n"
    "    Does one level-2 agent rescue a level-1 partner?")
print(f"  {'R_I':>5} {'A':>3} {'B':>3}")
for R in (0.94, 0.96, 0.98):
    for lvA, lvB in ((1, 1), (2, 1), (2, 2)):
        row(run(AgentConfig(L=3, level=lvA, R_I=R),
                AgentConfig(L=3, level=lvB, R_I=R), n_steps=N, seed=3),
            f"  {R:>5.2f} {lvA:>3} {lvB:>3}")

hdr("E4  INTERACTION TIMESCALE vs MEMORY TIMESCALE.\n"
    "    observability=0 => the ONLY news about the other arrives at\n"
    "    interaction events, so the interevent time sets the information rate.\n"
    "    Prediction: coupling survives only when tau_mem is commensurate with\n"
    "    the typical interevent time.")
print(f"  {'obs':>4} {'tau_mem':>8}")
for obs in (1.0, 0.3, 0.0):
    for tm in (5.0, 20.0, 100.0, 500.0, 2000.0):
        cfg = AgentConfig(L=3, level=2, R_I=0.99, tau_mem=tm)
        row(run(cfg, cfg, n_steps=N, observability=obs, seed=4),
            f"  {obs:>4.1f} {tm:>8.0f}")

hdr("E5  PREDICTION 4 (plan.md 7): ISOLATION DEGRADES THE SELF-MODEL.\n"
    "    Compare a level-2 agent with a live partner vs one whose partner\n"
    "    never reaches. err_self = |my model of how you see me - your model of me|.")
cfg = AgentConfig(L=3, level=2, R_I=0.99)
r = run(cfg, cfg, n_steps=N, seed=6)
dead = AgentConfig(L=3, level=0, R_I=-10.0)   # a partner that never reaches
r_iso = run(cfg, dead, n_steps=N, seed=6)
print(f"  coupled : regime={regime(r):>9}  err_self={r['err_self']:.4f}  "
      f"err_other={r['err_other']:.4f}  p={r['p_final']:.3f}")
print(f"  isolated: regime={regime(r_iso):>9}  err_self={r_iso['err_self']:.4f}  "
      f"err_other={r_iso['err_other']:.4f}  p={r_iso['p_final']:.3f}")
