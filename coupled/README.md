# coupled/ — two coupled agents

Substrate: J. G. Oliveira and A. Vazquez, *Impact of interactions on human
dynamics*, arXiv:0710.4916v3. Results are recorded in `../plan.md` §6A (R0–R8).

| file | what it does | runtime |
|---|---|---|
| `queueing_baseline.py` | Reproduces the paper: α = 1 + 1/max(L_j−1), and the divergence of the mean interevent time with observation window. Validation only — no agents. | ~3 s |
| `coupled_agents.py` | The model. Agents *choose* the interacting-task priority instead of drawing it i.i.d. Levels 0/1/2 = no other-model / other-model / self-model. | library |
| `run_experiments.py` | E1 phase diagram, E2 does the self-model do work, E3 asymmetric levels. | ~60 s |
| `run_timescale.py` | E4 interaction timescale vs memory timescale, E5 isolation. | ~36 s |
| `run_critical.py` | R8: locates the intermittent window near the saddle-node. Also exports `R_crit` and `pl_vs_exp` (power-law vs exponential tail likelihood). | ~8 min |
| `run_universality.py` | R9, first attempt. **Its L=4/L=5 rows are invalid** — see caveat below. Kept for the record. | ~12 min |
| `run_universality2.py` | R9 corrected (c=0.1). Its bisection also fails, for a documented reason: occupancy is not tunable via R_I. | ~7 min |
| `run_occupancy.py` | R10, the test that works: vary the seed at a fixed operating point and correlate α with phase occupancy. | ~5 min |
| `hypergraph.py` | Group meetings: hyperedges of size m requiring all members. Analytic critical group degree `d_crit(m)` and maximum group size. | library |
| `run_hypergraph.py` | R17: validates d_c(m), and shows why m>=3 fails despite the fixed point existing. | ~4 min |
| `run_lottery.py` | R11: bimodal outcomes at identical parameters; how early the outcome is predictable; whether a mid-run upgrade to level 2 reverses a collapse. | ~12 min |

```bash
python3 queueing_baseline.py
python3 run_experiments.py
python3 run_timescale.py
python3 run_occupancy.py      # the R8 verdict
```

## The one equation

An agent's optimal priority for the interacting task is

```
x_I* = R_I - c * (1 - p) / p
```

the value of the interaction discounted by the risk that the other will not be
there, where `p` is the agent's estimate of the partner's reach probability and
`c` is the cost of an unreciprocated reach. Reach probability is `x_I^(L-1)`, so
the number of competing solitary tasks `L` controls how hard it is to stay
coupled. `p = 0` is absorbing.

## Knobs

- `L` — solitary tasks competing with the relation. Larger ⇒ bigger solitary basin.
- `R_I` — value of a successful interaction. Crossing `R_I^c(L)` is the bifurcation.
- `c` — cost of reaching and being refused. This is what makes the other-model necessary.
- `level` — 0 none, 1 other-model, 2 self-model (tracks the partner's model of me).
- `avail` — rate of *opportunities* to interact; sets the interaction timescale.
- `observability` — P(seeing the partner on a non-interaction step).
- `hold_silence` — if `True`, the other-model is held across silent gaps instead
  of decaying. This is P1, and it is what rescues coupling at long interaction
  timescales (R5).

## Caveats

**Level-2 memoisation.** The level-2 decision is cached on a 200×200 belief grid
(`Agent._cache`); without it the rollout costs ~100× more. Decisions are
therefore coarse-grained in belief space. Re-check any result sitting near a
threshold with the cache disabled.

**Validity of the saddle-node formula.** `R_crit(a)` uses
p\* = (a·c)^(a/(a+1)), which is only meaningful while p\* ≤ 1, i.e. **a ≤ 1/c**.
At c = 0.35 that caps L at 3.86. `run_critical.R_crit` does *not* guard this and
returns garbage above the cap — the L=4 and L=5 rows of `run_universality.py`
are invalid for this reason. `run_universality2.R_crit` returns `None` instead.
Above the cap there is no interior bifurcation at all: the coupled phase is born
discontinuously at the clipping bound x_I = 1.

**Occupancy is not tunable via R_I.** The coupled/solitary transition is sharp,
so bisecting R_I to hit a target occupancy converges to the jump location
regardless of the target. Vary the seed instead (`run_occupancy.py`).

**ε-coincidence floor.** Rows showing ~ε²·N events with CV ≈ 1.2, identical
across R_I, are pure exploration noise in the solitary phase — not weak
coupling, and not critical behaviour.
