"""
Baseline: reproduce Oliveira & Vazquez, "Impact of interactions on human
dynamics" (arXiv:0710.4916v3), as the validated substrate for the coupled-agent
model of this project.

Two agents, each with a priority queue holding one *interacting* task I (which
executes only if BOTH agents select it) and an aggregate non-interacting task O
standing for L-1 solitary tasks. Highest-priority-first.

Coarse-grained model (exact, paper Eqs. 2-3):
    given I-priorities u, v ~ U(0,1),
    q = P(both O-priorities fall below their I-priority)
      = u^(L_A-1) * v^(L_B-1)
    tau | (u,v) ~ Geometric(q)

Prediction:  P(tau) ~ tau^-alpha  with  alpha = 1 + 1/max(L_j - 1)

Mechanism (why the exponent is what it is): with a = L_A-1, b = L_B-1 and
m = max(a,b), the density of q near zero goes as q^(1/m - 1), so
E[(1-q)^t] ~ t^(-1/m), giving a tail index 1/m on the survival function and
hence alpha = 1 + 1/m. The *more distracted* agent sets the exponent.
"""

import numpy as np


def simulate_interevent_times(L_A, L_B, n_events, rng):
    """Draw n_events interevent times from the exact coarse-grained model."""
    a, b = L_A - 1, L_B - 1
    u = rng.random(n_events)
    v = rng.random(n_events)
    q = u**a * v**b
    # Geometric on {1,2,...} with success probability q, sampled by inversion.
    # Guard against q underflowing to 0 (tau would be infinite).
    q = np.maximum(q, 1e-300)
    w = rng.random(n_events)
    tau = np.ceil(np.log1p(-w) / np.log1p(-q))
    return np.maximum(tau, 1.0)


def hill_alpha(tau, tau_min):
    """Hill MLE for the power-law tail index. Returns alpha in P(tau)~tau^-alpha."""
    x = tau[tau >= tau_min]
    if x.size < 100:
        return np.nan, x.size
    return 1.0 + x.size / np.sum(np.log(x / tau_min)), x.size


def alpha_over_window(tau, T):
    """Estimate alpha using only the events fitting in an observation window T.

    The paper's Eq. 8 scaling form P(tau) = A tau^-alpha g(tau/T^z) with z=1
    means a finite window imposes a cutoff. This mirrors the finite-T analysis.
    """
    cum = np.cumsum(tau)
    tau_w = tau[cum <= T]
    if tau_w.size < 200:
        return np.nan
    tau_min = max(10.0, np.quantile(tau_w, 0.90))
    a, _ = hill_alpha(tau_w, tau_min)
    return a


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 4_000_000

    print("Reproducing alpha = 1 + 1/max(L_j - 1)   [paper Fig. 4a inset]")
    print(f"{'L_A':>4} {'L_B':>4} {'predicted':>10} {'measured':>10} {'n_tail':>9}")
    cases = [(2, 2), (2, 3), (3, 3), (4, 4), (5, 5), (2, 5), (7, 7), (20, 20)]
    for L_A, L_B in cases:
        tau = simulate_interevent_times(L_A, L_B, n, rng)
        m = max(L_A - 1, L_B - 1)
        pred = 1.0 + 1.0 / m
        # take the tail well above the crossover; the crossover grows with m
        tau_min = float(np.quantile(tau, 0.95))
        meas, n_tail = hill_alpha(tau, tau_min)
        print(f"{L_A:>4} {L_B:>4} {pred:>10.3f} {meas:>10.3f} {n_tail:>9d}")

    print()
    print("Mean interevent time diverges with observation window T as T^(2-alpha)")
    print("=> for alpha <= 2 the interaction has NO characteristic timescale.")
    print(f"{'L':>4} {'alpha':>7}   " + "  ".join(f"T=1e{k}" for k in (4, 6, 8, 10)))
    for L in (2, 3, 5, 10):
        tau = simulate_interevent_times(L, L, n, rng)
        cum = np.cumsum(tau)
        means = []
        for k in (4, 6, 8, 10):
            T = 10.0**k
            sel = tau[cum <= T]
            means.append(sel.mean() if sel.size > 50 else np.nan)
        alpha = 1.0 + 1.0 / (L - 1)
        print(f"{L:>4} {alpha:>7.3f}   " + "  ".join(f"{m:9.2f}" for m in means))
