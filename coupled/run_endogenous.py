"""R13: endogenous commitment length. Does an OPTIMAL agent choose to persist?

R12 showed that a fixed commitment length k=1000 with gamma=0.999 recovers every
collapsed pair, but k was imposed. Here the agent evaluates its rollout at every
candidate k and latches onto the argmax, so the commitment length becomes a
property of its beliefs.

The argmax itself is uninformative: it pins to the top of the grid, because once
the partner is participating the agent wants to keep participating anyway, so
longer commitments cost nothing on the upside. The informative quantity is the
MINIMUM patience that beats not persisting,

    k_min(p_hat) = min { k : rollout(k) > rollout(0) },

which is the amount of unrewarded persistence the agent must be willing to
accept before reaching is worthwhile at all from that belief.
"""

import numpy as np
from coupled_agents import AgentConfig, Agent, run

L, C, R_I, EPS, TAU = 2, 0.35, 0.8428, 0.01, 200.0
PSTAR = C ** 0.5
N_STEPS, UPGRADE = 200_000, 100_000
SEEDS = range(200, 340)
rng0 = np.random.default_rng(0)

KLADDER = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
           512, 768, 1024, 1536, 2048, 3072]
POLICY_GRID = (0, 16, 64, 256, 1024)


def cfg(level=2, gamma=0.999, horizon=6000, grid=None, k=1):
    return AgentConfig(L=L, level=level, R_I=R_I, c=C, epsilon=EPS,
                       tau_mem=TAU, p_init=PSTAR, horizon=horizon,
                       gamma=gamma, commit_steps=k, commit_grid=grid)


def occ(p_traj, tail=0.2):
    j = int(len(p_traj) * (1 - tail))
    return float((p_traj[j:] > 0.5 * PSTAR).mean())


print("=" * 76)
print("A  REQUIRED PATIENCE  k_min(p_hat) = min{k : rollout(k) > rollout(0)}")
print("   ('--' = no commitment length in the ladder is worth it)")
print("=" * 76)
beliefs = [1e-6, 1e-3, 0.01, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
print(f"  {'gamma':>7} {'1/(1-g)':>8} | " + " ".join(f"{b:>7g}" for b in beliefs))
for gamma in (0.97, 0.99, 0.995, 0.999, 0.9995):
    ag = Agent(cfg(gamma=gamma, horizon=6000), rng0)
    row = []
    for b in beliefs:
        ag.p_hat = ag.s_hat = float(b)
        v0 = ag._rollout(0)
        km = next((k for k in KLADDER[1:] if ag._rollout(k) > v0), None)
        row.append(f"{km:>7}" if km is not None else f"{'--':>7}")
    print(f"  {gamma:>7.4f} {1/(1-gamma):>8.0f} | " + " ".join(row))

print()
print("=" * 76)
print("B  RECOVERY with the commitment length CHOSEN BY THE AGENT")
print("=" * 76)
collapsed = []
for sd in SEEDS:
    r = run(cfg(level=1), cfg(level=1), n_steps=N_STEPS, seed=sd)
    if occ(r["p_traj"]) < 0.25:
        collapsed.append(sd)
print(f"  {len(collapsed)} of {len(SEEDS)} seeds collapsed as level-1 pairs;"
      f" agent A promoted at step {UPGRADE:,}\n")
print(f"  {'gamma':>8} {'commitment':>12} {'recovered':>11} {'mean occ':>10}")

for gamma in (0.97, 0.99, 0.995, 0.999):
    for label, grid, k in (("fixed k=1", None, 1),
                           ("endogenous", POLICY_GRID, 1)):
        outs = []
        for sd in collapsed:
            a = cfg(level=1, gamma=gamma, horizon=6000, grid=grid, k=k)
            b = cfg(level=1, gamma=gamma, horizon=6000, grid=grid, k=k)
            r = run(a, b, n_steps=N_STEPS, seed=sd, upgrade_at=UPGRADE,
                    upgrade_levels=(2, None))
            outs.append(occ(r["p_traj"]))
        o = np.array(outs)
        print(f"  {gamma:>8.3f} {label:>12} {(o>0.5).mean():>10.1%} {o.mean():>10.3f}")
