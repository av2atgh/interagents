# interagents

**Absorbing phase transition in a queueing model of coupled adaptive agents.**

What decides whether people do things together or separately? Many activities
cannot be carried out alone, and an individual must rank them against the
private tasks competing for the same time.

Built on the two-agent queueing model of J. G. Oliveira and A. Vazquez,
*Impact of interactions on human dynamics*, Physica A **388**, 187 (2009)
(arXiv:0710.4916), in which two agents share a task requiring simultaneous
execution and the interevent time follows a power law with
α = 1 + 1/max(L_j − 1).

In that model the priority assigned to the shared task is drawn from a fixed
distribution. **Here the agents choose it**, which turns a stochastic process
into a coupled inference problem and changes the phenomenology substantially.

## Contents

| path | what |
|---|---|
| `manuscript.tex` | the paper (RevTeX, PRE format) |
| `coupled/` | all code, with its own README |
| `supplementary.tex` | Supplementary Tables I and II: the full parameter scans, submitted as a separate PDF |

## Main results

- Letting agents choose the interacting-task priority yields the policy
  `x_I* = R_I − c(1−p)/p` — the value of the interaction discounted by the risk
  the partner will not participate — and makes the system a stag hunt with an
  absorbing solitary phase.
- The saddle-node is analytic: `p* = (ac)^{a/(a+1)}`,
  `R_c = p*^{1/a} + c/p* − c`, valid only for `a ≤ 1/c`.
- Agents that model *the partner's model of themselves* enlarge the coupled
  phase, and one such agent stabilises a pair.
- Rare interaction is viable only if the partner model is **held** across
  intervals containing no observations.
- Adaptation destroys the universality classes: in the intermittent window near
  the bifurcation the exponent tracks phase occupancy, not queue length.
- Recovering a collapsed pair costs `0.74 τ_mem` of unilateral persistence and
  is chosen only by an agent with horizon `> 1.5 τ_mem`.
- On a network, attention is scarce and divides as `1/(k+a)`, giving a critical
  degree `k_c` and percolation by Molloy–Reed with the second moment truncated
  at `k_c`. The truncation acts at the top of the degree distribution, so
  scale-free contact structures are maximally vulnerable.

Network growth rules: triadic closure (local search at ℓ=1), its
degree-preserving randomization, and Barabási–Albert as a non-local control.

## Reproducing

```bash
cd coupled
python3 queueing_baseline.py    # validates against the 2009 paper
python3 run_experiments.py      # phase diagram, theory-of-mind levels
python3 run_percolation.py      # network percolation, first pass
python3 make_figures.py         # regenerates fig_phase.pdf
cd .. && pdflatex manuscript.tex && pdflatex manuscript.tex
pdflatex supplementary.tex && pdflatex supplementary.tex
```

The full campaign behind the published numbers — replicates, error bars,
collapse times, finite size, robustness and sensitivity — is

```bash
cd coupled
python3 run_survival.py                 # pair collapse times and survival
python3 run_replicates.py               # pair tables, 40 realisations each
python3 run_hyper_replicates.py         # group tables, 12 hypergraphs each
python3 run_bimodality.py               # bimodality, predictability, recovery
python3 run_noreturn2.py                # recovery vs commitment and discount
python3 run_boundary.py                 # measured phase boundary vs Eq. (5)
python3 run_occupancy2.py               # tail index vs occupancy
for t in main length sensitivity finitesize; do
  python3 run_netsweep.py $t            # network: one .npz per run
done
python3 analyze_survival.py             # pair analysis
python3 analyze_net.py                  # network analysis
python3 analyze_theory.py               # mean-field thresholds
python3 emit_tables.py <name>           # LaTeX table bodies
python3 emit_net.py <name>
python3 make_figures_rev.py             # fig_survival, fig_finitesize, fig_percolation
python3 make_supplementary.py           # ../supplementary.tex
```

Requires numpy, scipy, matplotlib; numba is optional but makes the network
sweep about seven times faster (`check_kernel.py` verifies the two paths agree).

## Status

Working repository for the manuscript. Known limitations are stated in the
manuscript itself, and Table I of the paper labels every result as exact,
mean-field, numerical or analogy: the mean-field percolation threshold
overestimates the measured one by a factor ~2.3 and fails qualitatively for
group sizes m >= 3; no critical exponent is measured and no scaling collapse is
attempted; and the network analogue of the partner-modelling agent is not
implemented.

