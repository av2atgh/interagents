"""Agents meeting in GROUPS: the interacting task becomes a hyperedge.

The pairwise model requires two agents to select the same task in the same step.
Here a hyperedge e of size m executes only when ALL m of its members select it,
which is the natural reading of a meeting: it happens only if everybody turns up.
Each agent holds one interacting task per hyperedge it belongs to, plus the
solitary aggregate, and still selects a single item per step.

ANALYTICS
---------
Selection is unchanged, P(i selects e) = X^a/(d_i + a) with d_i the number of
groups i belongs to. But agent i now needs all m-1 others, so

    p_hat = [ X^a / (d + a) ]^{m-1},

and self-consistency X = B - c/p_hat, B = R_I + c, gives

    X^{n+1} - B X^n + c (d + a)^{m-1} = 0,        n = a(m-1).           (H1)

Tangency at X* = nB/(n+1) yields a critical GROUP DEGREE

    d_c(m) = [ n^n B^{n+1} / ( c (n+1)^{n+1} ) ]^{1/(m-1)} - a.          (H2)

At m = 2 this reduces to the pairwise k_c. The exponent 1/(m-1) is the whole
story: the cost of coordination enters multiplicatively in the number of people
who must coincide, so d_c collapses very fast with group size. Setting
d_c(m) >= 1 gives a MAXIMUM GROUP SIZE even for an agent that belongs to nothing
else.
"""

import numpy as np


def d_crit(m, R_I, c, a=1):
    """Eq. (H2). Largest number of groups an agent can sustain at group size m."""
    if m < 2:
        return np.inf
    n = a * (m - 1)
    log_d = (n * np.log(n) + (n + 1) * np.log(R_I + c)
             - np.log(c) - (n + 1) * np.log(n + 1)) / (m - 1)
    return np.exp(log_d) - a


def max_group_size(R_I, c, a=1, d=1, mmax=40):
    """Largest m for which an agent belonging to d groups can sustain them."""
    ok = [m for m in range(2, mmax) if d_crit(m, R_I, c, a) >= d]
    return max(ok) if ok else 0


def regular_hypergraph(n_agents, m, d, seed=0):
    """d-regular, m-uniform hypergraph by stub matching.

    Returns (mem_agent, mem_edge, n_edges). Rejects hyperedges with repeated
    members by reshuffling, so every group has m distinct agents.
    """
    if (n_agents * d) % m:
        raise ValueError("n_agents*d must be divisible by m")
    rng = np.random.default_rng(seed)
    n_edges = n_agents * d // m
    for _ in range(200):
        stubs = rng.permutation(np.repeat(np.arange(n_agents), d))
        groups = stubs.reshape(n_edges, m)
        if all(len(set(g)) == m for g in groups):
            mem_agent = groups.reshape(-1)
            mem_edge = np.repeat(np.arange(n_edges), m)
            return mem_agent, mem_edge, n_edges
    # fall back: repair duplicates by local swaps
    groups = groups.copy()
    for i, g in enumerate(groups):
        while len(set(g)) < m:
            j = rng.integers(0, n_edges)
            k1, k2 = rng.integers(0, m), rng.integers(0, m)
            g[k1], groups[j][k2] = groups[j][k2], g[k1]
    mem_agent = groups.reshape(-1)
    mem_edge = np.repeat(np.arange(n_edges), m)
    return mem_agent, mem_edge, n_edges


def run_hyper(mem_agent, mem_edge, n_agents, n_edges, m,
              R_I=0.95, c=0.02, L=2, tau_mem=200.0, epsilon=0.01,
              n_steps=60_000, burn_frac=0.5, seed=0, p_init=0.85):
    """Level-1 agents on a fixed hypergraph. Returns per-hyperedge meeting rates."""
    a = L - 1
    lam = np.exp(-1.0 / tau_mem)
    rng = np.random.default_rng(seed)
    P = mem_agent.size

    p_hat = np.full(P, float(p_init))
    u = rng.random(P)
    counts = np.zeros(n_edges)
    burn = int(n_steps * burn_frac)

    for t in range(n_steps):
        xstar = np.clip(R_I - c * (1.0 - p_hat) / np.maximum(p_hat, 1e-9), 0.0, 1.0)
        x = u * xstar
        x = np.where(rng.random(P) < epsilon, 1.0, x)

        best = np.zeros(n_agents)
        np.maximum.at(best, mem_agent, x)
        x_O = rng.random(n_agents) ** (1.0 / a)
        active = best > x_O
        sel = (x >= best[mem_agent]) & active[mem_agent]
        # a tie could let one agent select two groups; keep only its argmax
        if sel.sum() > active.sum():
            first = np.full(n_agents, -1)
            order = np.argsort(-x, kind="stable")
            for e in order[sel[order]]:
                if first[mem_agent[e]] < 0:
                    first[mem_agent[e]] = e
            sel = np.zeros_like(sel)
            sel[first[first >= 0]] = True

        cnt = np.bincount(mem_edge[sel], minlength=n_edges)
        fires = cnt == m
        if t >= burn:
            counts += fires

        # i observes whether ALL the others turned up
        others_all = (cnt[mem_edge] - sel.astype(int)) == (m - 1)
        p_hat = np.clip(lam * p_hat + (1 - lam) * others_all.astype(float),
                        1e-9, 1.0)
        if sel.any():
            u = np.where(sel, rng.random(P), u)

    return counts / (n_steps - burn)
