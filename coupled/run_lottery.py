"""R11: the developmental lottery.

At identical parameters, some pairs reach the coupled phase and some collapse to
solitary. Two questions:

  A. How early is the outcome decided? If a short prefix of the trajectory
     already predicts the final phase, the lottery is settled in development,
     not maintained by ongoing dynamics.

  B. Can a level-2 agent rescue a pair after the lottery has gone against it?
     Pairs are run as level 1, allowed to collapse, then one or both agents are
     upgraded to level 2 (self-model) partway through. If the pair recovers, the
     solitary attractor is escapable by architecture rather than by luck. If it
     does not, plan.md 3.4's claim about the self-model is limited to
     PREVENTING collapse, not reversing it -- a materially weaker claim.
"""

import numpy as np
from dataclasses import replace
from coupled_agents import AgentConfig, run

L, C, R_I, EPS, TAU = 2, 0.35, 0.8428, 0.01, 200.0
PSTAR = C ** 0.5
N_STEPS = 300_000
SEEDS = range(200, 340)


def base(level):
    return AgentConfig(L=L, level=level, R_I=R_I, c=C, epsilon=EPS,
                       tau_mem=TAU, p_init=PSTAR)


def outcome(p_traj, tail=0.2):
    k = int(len(p_traj) * (1 - tail))
    return float((p_traj[k:] > 0.5 * PSTAR).mean())


print("=" * 76)
print("R11-A  HOW EARLY IS THE LOTTERY DECIDED?")
print("=" * 76)

fracs, prefix = [], {}
CHECKS = (1_000, 3_000, 10_000, 30_000, 60_000, 120_000)
for sd in SEEDS:
    cfg = base(1)
    r = run(cfg, base(1), n_steps=N_STEPS, seed=sd)
    p = r["p_traj"]
    fracs.append(outcome(p))
    for t in CHECKS:
        prefix.setdefault(t, []).append(float((p[:t] > 0.5 * PSTAR).mean()))

fracs = np.array(fracs)
coupled = fracs > 0.5
print(f"  {len(fracs)} seeds, identical parameters")
print(f"  final occupancy: min={fracs.min():.3f} max={fracs.max():.3f}")
print(f"  bimodality: {(fracs<0.25).sum()} seeds < 0.25, "
      f"{((fracs>=0.25)&(fracs<=0.75)).sum()} in between, "
      f"{(fracs>0.75).sum()} seeds > 0.75")
print(f"  reached coupled phase: {coupled.mean():.1%}")
print()
print(f"  {'prefix':>8} {'best accuracy':>14} {'threshold':>10}")
for t in CHECKS:
    x = np.array(prefix[t])
    best, thr = 0.0, 0.0
    for c in np.unique(np.round(x, 4)):
        acc = max(((x > c) == coupled).mean(), ((x <= c) == coupled).mean())
        if acc > best:
            best, thr = acc, c
    print(f"  {t:>8} {best:>14.3f} {thr:>10.4f}")
print(f"  (baseline = always guess the majority class: "
      f"{max(coupled.mean(), 1-coupled.mean()):.3f})")

print()
print("=" * 76)
print("R11-B  CAN A SELF-MODEL REVERSE A COLLAPSE?")
print("=" * 76)

collapsed = [sd for sd, f in zip(SEEDS, fracs) if f < 0.25]
print(f"  {len(collapsed)} of {len(fracs)} seeds collapsed as level-1 pairs.")
print(f"  Re-running those seeds, upgrading at step {N_STEPS//2:,}.")
print()
print(f"  {'upgrade':>16} {'n':>4} {'recovered':>11} {'mean final occ':>15}")

for label, lv in (("neither (control)", (None, None)),
                  ("A only -> lvl 2", (2, None)),
                  ("both -> lvl 2", (2, 2))):
    outs = []
    for sd in collapsed:
        r = run(base(1), base(1), n_steps=N_STEPS, seed=sd,
                upgrade_at=N_STEPS // 2, upgrade_levels=lv)
        outs.append(outcome(r["p_traj"]))
    outs = np.array(outs)
    print(f"  {label:>16} {len(outs):>4} {(outs>0.5).mean():>10.1%} "
          f"{outs.mean():>15.3f}")

print()
print("  For contrast: pairs that are level 2 FROM THE START, same seeds.")
outs = []
for sd in collapsed:
    r = run(base(2), base(2), n_steps=N_STEPS, seed=sd)
    outs.append(outcome(r["p_traj"]))
outs = np.array(outs)
print(f"  {'level 2 from t=0':>16} {len(outs):>4} {(outs>0.5).mean():>10.1%} "
      f"{outs.mean():>15.3f}")
