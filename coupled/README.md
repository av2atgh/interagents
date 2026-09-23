# coupled/ — two coupled agents

Substrate: J. G. Oliveira and A. Vazquez, *Impact of interactions on human
dynamics*, arXiv:0710.4916v3. Results are reported in `../manuscript.tex`.

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
| `network.py` | The model on a fixed network: one interacting task per neighbour, attention divided as 1/(k+a). Also records edge-level collapse times and the coarse-grained time series. Uses the compiled kernel when numba is present and the NumPy path otherwise. | library |
| `network_kernel.py` | numba inner loop for `run_network`, ~7x faster than the NumPy path. Same model; exploration uses geometric gaps and the edge collapse time is derived from the last execution, both exact. | library |
| `check_kernel.py` | Ensemble equivalence check between the compiled and NumPy paths, with the reference output in its docstring. | ~6 min |
| `theory.py` | Analytic percolation: critical degree `k_crit`, cavity equation, Molloy–Reed limit. | library |
| `run_percolation.py` | R15, first pass: three growth rules, 3 realisations, 4·10⁴ steps. Superseded by `run_netsweep.py`. | ~15 min |

### Added for the revision (referee-requested numerics)

| file | what it does | runtime |
|---|---|---|
| `run_bimodality.py` | Bimodal outcomes, prefix predictability and recovery at the operating point of the first submission, over 1000 realisations, with Wilson intervals and a split-sample estimate of the classifier accuracy. | ~12 min |
| `run_noreturn2.py` | Recovery versus commitment length and discount over a collapsed set several times larger than the first submission's. | ~10 min |
| `run_noexplore.py` | Whether the network coupled phase survives at epsilon = 0. It does not, at any cost -- the reason the mean-field threshold is wrong. | ~4 min |
| `run_patience.py` | Lookahead sweep and endogenous commitment on the *same* collapsed set as `run_noreturn2.py`, so the two experiments are directly comparable. | ~14 min |
| `analyze_patience.py` | Wilson intervals on both. | ~2 s |
| `run_boundary.py` | Measured phase boundary in R_I against Eq. (5), for six (L, c) -- the sensitivity to c and L, neither of which enters except through R_c. | ~7 min |
| `run_occupancy2.py` | Tail index versus phase occupancy over 60 realisations per operating point, with a bootstrap interval on the correlation. | ~3 min |
| `emit_tables.py` | Emits the LaTeX table bodies from the stored results, so nothing is transcribed by hand. | ~5 s |
| `run_survival.py` | Pair collapse times: first-passage distribution at the reference point, and the dependence of the mean collapse time on τ_mem, R_I, ε, the initial belief, and the run length. 500 realisations at the reference point. | ~25 min |
| `analyze_bimodality.py` | Wilson intervals, split-sample classifier accuracy. | ~10 s |
| `analyze_survival.py` | Censored-exponential MLE of the mean collapse time, survival curves, bimodality fractions. | ~2 s |
| `run_replicates.py` | Replicates with error bars for the pair tables that the first submission reported at a single seed (asymmetric levels, held-vs-decaying beliefs, the level-1/level-2 phase boundary). 40 realisations per point. | ~25 min |
| `run_hyper_replicates.py` | Same for the hypergraph tables. 12 realisations per point. | ~20 min |
| `run_netsweep.py` | Network sweep, one `.npz` per run (per-edge rates, degrees, edge collapse times, time series). Job sets: `main` (n=600, three rules, 20 realisations, 4·10⁵ steps), `finitesize` (n=300…2400), `length` (run length and initial condition), `sensitivity` (ε, τ_mem, R_I, L). Restartable — existing files are skipped. | ~3 h total |
| `analyze_net.py` | All network analysis offline from the stored runs: error bars, component-size distributions, robustness of the coupled-edge cut, degree-resolved survival, finite size, run-length and initial-condition dependence, edge collapse times. | ~1 min |
| `analyze_theory.py` | Mean-field thresholds (cavity and Molloy–Reed) for the degree distributions actually used, at each size. | ~2 min |
| `make_supplementary.py` | Builds `../supplementary.tex`: the two widest parameter scans, reported in full outside the main text and submitted as a separate PDF. | ~20 s |
| `make_figures_rev.py` | `fig_survival.pdf`, `fig_finitesize.pdf`, and the data-driven `fig_percolation.pdf`. | ~1 min |

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
