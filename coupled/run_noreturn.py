"""Is the ~36% recovery cap (R11) a genuine point of no return, or a limit of
the level-2 rollout?

Three candidate causes, swept separately:
  horizon       - the agent cannot see far enough ahead
  gamma         - the agent discounts the future too steeply
  commit_steps  - the agent never CONSIDERS persisting for more than one step

The third is the suspicious one. With commit_steps=1 the rollout only asks 'is
reaching worth it for one step?'. Reaching once at a collapsed partner barely
moves their estimate before the agent reverts, so it almost never pays --
regardless of how far ahead the agent looks.

Part A is analytic and free: for a grid of beliefs, ask whether the agent
commits at all. If the minimum p_hat at which it commits falls to zero as
patience grows, no belief state is beyond rescue and there is no point of no
return in the policy. Part B measures actual recovery on the collapsed seeds.
"""

import numpy as np
from coupled_agents import AgentConfig, Agent, run

L, C, R_I, EPS, TAU = 2, 0.35, 0.8428, 0.01, 200.0
PSTAR = C ** 0.5
N_STEPS = 200_000
UPGRADE = N_STEPS // 2
SEEDS = range(200, 340)
rng0 = np.random.default_rng(0)


def cfg(level=2, horizon=40, gamma=0.97, commit=1):
    return AgentConfig(L=L, level=level, R_I=R_I, c=C, epsilon=EPS,
                       tau_mem=TAU, p_init=PSTAR,
                       horizon=horizon, gamma=gamma, commit_steps=commit)


def occ(p_traj, tail=0.2):
    k = int(len(p_traj) * (1 - tail))
    return float((p_traj[k:] > 0.5 * PSTAR).mean())


# ---------------------------------------------------------------- Part A
print("=" * 78)
print("A  COMMIT REGION (analytic): lowest belief p_hat at which the agent")
print("   still decides to reach.  s_hat is set equal to p_hat (a collapsed")
print("   pair sees itself as it sees the other).  '--' = never commits.")
print("=" * 78)
grid = np.concatenate([np.geomspace(1e-6, 0.01, 40), np.linspace(0.01, 1.0, 200)])
print(f"  {'commit':>7} {'gamma':>7} {'horizon':>8}   {'min p_hat that commits':>24}")
for commit in (1, 10, 100, 1000):
    for gamma in (0.97, 0.999):
        H = max(60, 3 * commit)
        ag = Agent(cfg(2, H, gamma, commit), rng0)
        lo = None
        for p in grid:
            ag.p_hat, ag.s_hat, ag._commit_left = float(p), float(p), 0
            if ag._rollout(True) > ag._rollout(False):
                lo = float(p)
                break
        s = f"{lo:.2e}" if lo is not None else "--"
        print(f"  {commit:>7} {gamma:>7.3f} {H:>8}   {s:>24}")

# ---------------------------------------------------------------- Part B
print()
print("=" * 78)
print("B  ACTUAL RECOVERY on the seeds that collapsed as level-1 pairs")
print("=" * 78)

collapsed = []
for sd in SEEDS:
    r = run(cfg(1), cfg(1), n_steps=N_STEPS, seed=sd)
    if occ(r["p_traj"]) < 0.25:
        collapsed.append(sd)
print(f"  {len(collapsed)} of {len(SEEDS)} seeds collapsed; "
      f"upgrading agent A at step {UPGRADE:,}\n")
print(f"  {'commit':>7} {'gamma':>7} {'horizon':>8} {'recovered':>11} {'mean occ':>10}")


def trial_params(horizon, gamma, commit):
    outs = []
    for sd in collapsed:
        a = cfg(1, horizon, gamma, commit)
        b = cfg(1, horizon, gamma, commit)
        r = run(a, b, n_steps=N_STEPS, seed=sd, upgrade_at=UPGRADE,
                upgrade_levels=(2, None))
        outs.append(occ(r["p_traj"]))
    return np.array(outs)


print("  -- patience sweep (horizon scaled to commitment) --")
for commit in (1, 10, 100, 1000):
    for gamma in (0.97, 0.999):
        H = max(60, 3 * commit)
        o = trial_params(H, gamma, commit)
        print(f"  {commit:>7} {gamma:>7.3f} {H:>8} {(o>0.5).mean():>10.1%} "
              f"{o.mean():>10.3f}")

print("\n  -- pure lookahead sweep at commit_steps = 1 --")
for H in (10, 40, 200, 1000):
    for gamma in (0.97, 0.999):
        o = trial_params(H, gamma, 1)
        print(f"  {1:>7} {gamma:>7.3f} {H:>8} {(o>0.5).mean():>10.1%} "
              f"{o.mean():>10.3f}")
