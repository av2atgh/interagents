"""E4/E5 rerun: interaction timescale, and isolation.  (plan.md 3.5, 6.5, 7)

`avail` sets the mean gap between OPPORTUNITIES to interact, independently of
whether the agents want to. This separates the timescale question from the
coordination question. With observability=0 the only news about the other
arrives at those opportunities, so 1/avail is the information rate of the
other-model, and tau_mem is the rate at which that model forgets.
"""

import numpy as np
from coupled_agents import AgentConfig, run, alpha_of, regime

N = 200_000


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def row(res, pre):
    a = alpha_of(res["tau"])
    a_s = f"{a:5.2f}" if not np.isnan(a) else "   --"
    mt = res["tau"].mean() if res["tau"].size > 10 else np.nan
    mt_s = f"{mt:8.1f}" if not np.isnan(mt) else "      --"
    print(f"{pre}  {regime(res):>9}  p={res['p_final']:.3f}  rate={res['rate']:.4f}  "
          f"<tau>={mt_s}  alpha={a_s}  payoff={0.5*(res['payoff_A']+res['payoff_B']):.3f}")


hdr("E4a  INTERACTION TIMESCALE vs MEMORY TIMESCALE.\n"
    "     observability=0, level 2, R_I=0.99, L=3.\n"
    "     1/avail is the mean gap between opportunities. tau_mem is how fast\n"
    "     the other-model forgets. Silence is read as rejection (hold=False).")
print(f"  {'1/avail':>8} {'tau_mem':>8}")
for av in (1.0, 0.1, 0.01, 0.003):
    for tm in (5.0, 50.0, 500.0, 5000.0):
        cfg = AgentConfig(L=3, level=2, R_I=0.99, tau_mem=tm)
        row(run(cfg, cfg, n_steps=N, observability=0.0, avail=av, seed=11),
            f"  {1/av:>8.0f} {tm:>8.0f}")

hdr("E4b  SAME, but the other-model is HELD across silence (hold_silence=True):\n"
    "     absence of evidence is not treated as evidence of absence. Holding a\n"
    "     model with no input is precisely P1. If this rescues coupling at long\n"
    "     interaction timescales, then spontaneous activity has a FUNCTION:\n"
    "     keeping the other alive between encounters.")
print(f"  {'1/avail':>8} {'tau_mem':>8}")
for av in (1.0, 0.1, 0.01, 0.003):
    for tm in (5.0, 50.0, 500.0, 5000.0):
        cfg = AgentConfig(L=3, level=2, R_I=0.99, tau_mem=tm, hold_silence=True)
        row(run(cfg, cfg, n_steps=N, observability=0.0, avail=av, seed=11),
            f"  {1/av:>8.0f} {tm:>8.0f}")

hdr("E5  ISOLATION DEGRADES THE SELF-MODEL (plan.md prediction 4).\n"
    "    err_self = |my model of how you see me  -  your actual model of me|")
cfg = AgentConfig(L=3, level=2, R_I=0.99)
dead = AgentConfig(L=3, level=0, R_I=0.0)      # a partner that never reaches
for label, partner in (("live partner", cfg), ("partner never reaches", dead)):
    r = run(cfg, partner, n_steps=N, seed=12)
    print(f"  {label:>22}: regime={regime(r):>9}  p={r['p_final']:.3f}  "
          f"err_self={r['err_self']:.4f}  err_other={r['err_other']:.4f}")
