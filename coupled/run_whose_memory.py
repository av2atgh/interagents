"""R14: whose memory constant sets the price of recovery?

DOES NOT WORK AS WRITTEN -- kept for the record, do not draw conclusions from
it. `recovery_time` thresholds `p_traj`, which stores the MEAN of the two
agents' participation probabilities, so it fires the instant agent A commits
(p_A = 1 makes the mean >= 0.5 whatever the partner does) and reports t_rec = 0
for pairs that never actually coupled. Fixing it needs a per-agent trajectory
(add a `pb_traj` to `run()`), and a gamma chosen between 1.5*tau_A and
1.5*tau_B so that the two hypotheses give opposite predictions. See plan.md R14.


R13 found k_min = 0.739 tau_mem and H* = 1.5 tau_mem, but every run there used
tau_A = tau_B, so 'the partner's memory' and 'the agent's assumption about it'
were indistinguishable. The two enter at different points:

  - the agent DECIDES how long to persist using self.lam, i.e. its OWN tau_A
    (the rollout propagates the partner's estimate with the agent's constant);
  - the partner ACTUALLY updates at rate 1/tau_B.

So a mismatch should matter, and asymmetrically. An agent that underestimates
its partner's memory commits for too short a time and its offer expires before
it lands; one that overestimates persists longer than necessary and succeeds.

Test: set gamma high enough that every agent is willing (H = 10^4), then measure
the TIME from promotion to recovery as tau_A and tau_B are varied separately.
Whichever constant the recovery time scales with is the one that sets the price.
"""

import numpy as np
from coupled_agents import AgentConfig, run

C, PS = 0.35, 0.35 ** 0.5
N, UP = 400_000, 100_000
GRID = (0, 8, 32, 128, 512, 2048)


def cfg(level, tau, grid=None, gamma=0.9999):
    return AgentConfig(L=2, level=level, R_I=0.8428, c=C, epsilon=0.01,
                       tau_mem=float(tau), p_init=PS, horizon=12000,
                       gamma=gamma, commit_grid=grid)


def occ(p, tail=0.2):
    j = int(len(p) * (1 - tail))
    return float((p[j:] > 0.5 * PS).mean())


def recovery_time(p):
    """Steps from promotion until participation is sustained above 0.5."""
    post = p[UP:]
    hi = post > 0.5
    # first index after which it stays high for 5000 consecutive steps
    run_len, start = 0, None
    for i, v in enumerate(hi):
        if v:
            if start is None:
                start = i
            run_len += 1
            if run_len >= 5000:
                return start
        else:
            run_len, start = 0, None
    return np.nan


print("=" * 72)
print("R14  whose memory sets the price?   (gamma = 0.9999, H = 10^4)")
print("=" * 72)

collapsed = [sd for sd in range(200, 400)
             if occ(run(cfg(1, 200), cfg(1, 200), n_steps=200_000, seed=sd)["p_traj"]) < 0.25]
print(f"  {len(collapsed)} collapsed seeds identified at tau=200\n")
print(f"  {'tau_A':>6} {'tau_B':>6} {'recovered':>10} {'median t_rec':>13} "
      f"{'t_rec/tau_A':>12} {'t_rec/tau_B':>12}")

for tA, tB in ((100, 100), (400, 400),
               (100, 400), (400, 100),
               (100, 800), (800, 100)):
    ts, ok = [], []
    for sd in collapsed:
        r = run(cfg(1, tA, GRID), cfg(1, tB, GRID), n_steps=N, seed=sd,
                upgrade_at=UP, upgrade_levels=(2, None))
        ok.append(occ(r["p_traj"]) > 0.5)
        t = recovery_time(r["p_traj"])
        if not np.isnan(t):
            ts.append(t)
    m = np.median(ts) if ts else np.nan
    ms = f"{m:13.0f}" if not np.isnan(m) else "           --"
    a_s = f"{m/tA:12.2f}" if not np.isnan(m) else "          --"
    b_s = f"{m/tB:12.2f}" if not np.isnan(m) else "          --"
    print(f"  {tA:>6} {tB:>6} {np.mean(ok):>9.1%} {ms} {a_s} {b_s}")
