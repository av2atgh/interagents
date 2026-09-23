"""Coupled agents on a network: does the solitary phase percolate?

Generalises the two-agent model. Agent i now holds one interacting task PER
NEIGHBOUR plus the aggregate solitary task, and selects a single highest-priority
item per step. Attention is therefore a scarce resource: a hub must divide it
across many relations, so degree competes against every individual relation.

An edge (i,j) executes only when i selects j AND j selects i, exactly as in the
two-agent case.

Growth rules follow Vazquez, *Local Network Growth* (LocalNetworkGrowth/):

  LS(n, l=1)  triadic closure. Pick a random node, take a 1-step walk, link the
              newcomer to both. Closes a triangle every step. Has emergent
              communities (Ramsey number r_kappa = 81 by block model).
  CONF        degree-preserving randomization of LS. Same degrees, no local
              structure. No finite Ramsey number -- communities destroyed.
  BA(n, m=2)  non-local control, preferential attachment over the whole graph.
              No emergent communities.

LS and BA are matched at mean degree 4; CONF matches LS degree-for-degree. Any
difference between LS and CONF is therefore attributable to local structure
alone, which is the control the book uses.
"""

import numpy as np

try:
    from network_kernel import run_kernel as _run_kernel
except Exception:                                    # numba unavailable
    _run_kernel = None


# ----------------------------------------------------------------- growth
def grow_ls(n, ell=1, seed=0):
    """Local search / triadic closure at ell=1."""
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(n)]
    adj[0].add(1)
    adj[1].add(0)
    for i in range(2, n):
        j = int(rng.integers(0, i))
        targets = {j}
        cur = j
        for _ in range(ell):
            nb = list(adj[cur])
            if not nb:
                break
            cur = nb[int(rng.integers(0, len(nb)))]
            targets.add(cur)
        for t in targets:
            adj[i].add(t)
            adj[t].add(i)
    return adj


def grow_ba(n, m=2, seed=0):
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(n)]
    for i in range(1, min(m + 1, n)):
        adj[i].add(0)
        adj[0].add(i)
    stubs = [v for v in range(min(m + 1, n)) for _ in adj[v]]
    for i in range(m + 1, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(int(stubs[int(rng.integers(0, len(stubs)))]))
        for t in chosen:
            adj[i].add(t)
            adj[t].add(i)
            stubs += [i, t]
    return adj


def randomize_degree_preserving(adj, seed=0, sweeps=20):
    """Double-edge swaps: preserves every degree, destroys local structure."""
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(len(adj)) for j in adj[i] if i < j]
    eset = {(min(a, b), max(a, b)) for a, b in edges}
    for _ in range(sweeps * len(edges)):
        p, q = int(rng.integers(0, len(edges))), int(rng.integers(0, len(edges)))
        if p == q:
            continue
        a, b = edges[p]
        c, d = edges[q]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = (min(a, c), max(a, c)), (min(b, d), max(b, d))
        if e1 in eset or e2 in eset:
            continue
        eset.discard((min(a, b), max(a, b)))
        eset.discard((min(c, d), max(c, d)))
        eset.add(e1)
        eset.add(e2)
        edges[p], edges[q] = e1, e2
    out = [set() for _ in adj]
    for a, b in eset:
        out[a].add(b)
        out[b].add(a)
    return out


# ----------------------------------------------------------------- indexing
def directed_index(adj):
    """Build directed-edge arrays and the reverse-edge permutation."""
    src, dst = [], []
    pos = {}
    for i, nb in enumerate(adj):
        for j in nb:
            pos[(i, j)] = len(src)
            src.append(i)
            dst.append(j)
    src = np.asarray(src)
    dst = np.asarray(dst)
    rev = np.array([pos[(j, i)] for i, j in zip(src, dst)])
    return src, dst, rev


# ----------------------------------------------------------------- dynamics
def run_network(adj, R_I=0.90, c=0.35, L=2, tau_mem=200.0, epsilon=0.01,
                n_steps=60_000, burn_frac=0.3, seed=0, p_init=None,
                record=False, silence_hold=None, window=None):
    """Level-1 agents on a fixed network. Returns per-edge interaction rates.

    record: also return (i) the first-passage time of each directed edge to
      sustained silence -- the first step at which it has gone `silence_hold`
      consecutive steps without executing, which is the edge-level analogue of
      the collapse time of a pair -- and (ii) the executed-edge count binned in
      windows of `window` steps, so the approach to the stationary state can be
      seen rather than assumed. Defaults: silence_hold = ceil(tau_mem),
      window = 10 * silence_hold.
    """
    n = len(adj)
    src, dst, rev = directed_index(adj)
    a = L - 1
    lam = np.exp(-1.0 / tau_mem)
    rng = np.random.default_rng(seed)

    if p_init is None:
        p_hat = np.full(src.size, 0.85)
    elif np.isscalar(p_init):
        p_hat = np.full(src.size, float(p_init))
    else:
        p_hat = np.asarray(p_init, dtype=float).copy()   # per-edge initialisation
    counts = np.zeros(src.size)
    burn = int(n_steps * burn_frac)

    if record:
        hold = int(np.ceil(tau_mem)) if silence_hold is None else int(silence_hold)
        win = 10 * hold if window is None else int(window)
        silence = np.zeros(src.size, dtype=np.int32)
        fpt = np.full(src.size, -1, dtype=np.int64)
        n_win = int(np.ceil(n_steps / win))
        series = np.zeros(n_win)
        n_unset = src.size

    # Node offsets into the directed-edge arrays. directed_index builds them in
    # node order, so each node's edges are contiguous and the per-node maximum
    # is a reduceat rather than a scatter-max.
    deg = np.bincount(src, minlength=n)
    if (deg == 0).any():
        raise ValueError("isolated nodes are not supported")
    starts = np.concatenate(([0], np.cumsum(deg)[:-1]))
    inv_n = 1.0 / a

    if _run_kernel is not None:
        # Compiled inner loop. Same model, scalar loops; see network_kernel.py.
        hold_k = (int(np.ceil(tau_mem)) if silence_hold is None
                  else int(silence_hold))
        win_k = 10 * hold_k if window is None else int(window)
        u0 = rng.random(src.size)
        cnt, fpt_k, series_k = _run_kernel(
            src.astype(np.int64), dst.astype(np.int64), rev.astype(np.int64),
            starts.astype(np.int64), deg.astype(np.int64), n, src.size,
            float(a), float(R_I), float(c), float(lam), float(epsilon),
            int(n_steps), int(burn), hold_k, win_k,
            p_hat.astype(np.float64), u0, int(seed))
        out = dict(rate=cnt / (n_steps - burn), src=src, dst=dst, rev=rev, n=n)
        if record:
            out["fpt"] = fpt_k
            out["series"] = series_k / (win_k * src.size)
            out["window"] = win_k
            out["silence_hold"] = hold_k
        return out

    # Faithful to the queueing substrate: a task's priority is DRAWN from a
    # distribution the policy controls, and held until that task executes. With
    # one relation this is immaterial, but with several it is what lets an agent
    # rotate among them. Making the priority a deterministic function of p_hat
    # instead locks each agent onto its single best partner and forces the
    # coupled subgraph to be a matching.
    u = rng.random(src.size)

    E = src.size
    for t in range(n_steps):
        # policy: value of the relation discounted by the risk of no response
        # (p_hat is kept >= 1e-9 below, so no further guard is needed here)
        xstar = np.clip(R_I - c * (1.0 / p_hat - 1.0), 0.0, 1.0)
        x = u * xstar
        # Exploration sets x = 1 on a Binomial(E, epsilon) subset. Drawing the
        # count and then the indices costs O(#explorers) instead of O(E).
        n_expl = rng.binomial(E, epsilon) if epsilon > 0.0 else 0
        if n_expl:
            expl_idx = rng.integers(0, E, n_expl)
            x[expl_idx] = 1.0

        # each node's best relation, and its solitary alternative
        best = np.maximum.reduceat(x, starts)
        x_O = rng.random(n) ** inv_n
        # a node participates only if its best relation beats its private task
        active = best > x_O
        # select the argmax edge (ties are measure-zero unless two of a node's
        # relations were both explored on the same step)
        sel = (x >= best[src]) & active[src]
        # guard against a node selecting two edges on an exact tie
        if n_expl and sel.sum() > active.sum():
            first = np.full(n, -1)
            order = np.argsort(-x, kind="stable")
            for e in order[sel[order]]:
                if first[src[e]] < 0:
                    first[src[e]] = e
            sel = np.zeros_like(sel)
            sel[first[first >= 0]] = True

        both = sel & sel[rev]
        if t >= burn:
            counts += both
        if record:
            series[t // win] += both.sum()
            silence += 1
            silence[both] = 0
            if n_unset:
                # silence increments by one and resets to zero, so it passes
                # through `hold` exactly once per silent run
                hit = np.flatnonzero(silence == hold)
                if hit.size:
                    hit = hit[fpt[hit] < 0]
                    if hit.size:
                        fpt[hit] = t - hold + 1
                        n_unset -= hit.size

        # An OFFER consumes the attempt, whether or not it was reciprocated, so
        # the priority is redrawn on selection rather than only on execution.
        # Redrawing only on execution freezes the choice: agent i would offer to
        # the same partner forever, and if that partner's best is not i the pair
        # deadlocks with no mechanism to ever change.
        # An OFFER consumes the attempt, so only the selected edges are
        # redrawn; at most one per node, so this is O(n) rather than O(E).
        sel_idx = np.flatnonzero(sel)
        if sel_idx.size:
            u[sel_idx] = rng.random(sel_idx.size)

        # i learns whether j selected i on this relation
        p_hat *= lam
        p_hat[sel[rev]] += 1.0 - lam
        np.maximum(p_hat, 1e-9, out=p_hat)

    rate = counts / (n_steps - burn)
    out = dict(rate=rate, src=src, dst=dst, rev=rev, n=n)
    if record:
        out["fpt"] = fpt                      # -1 = never silent for `hold`
        out["series"] = series / (win * src.size)
        out["window"] = win
        out["silence_hold"] = hold
    return out


# ----------------------------------------------------------------- analysis
def undirected_coupled(res, thresh=0.02):
    """Set of undirected edges whose interaction rate exceeds the threshold."""
    keep = res["rate"] > thresh
    return {(min(i, j), max(i, j))
            for i, j, k in zip(res["src"], res["dst"], keep) if k}


def giant_fraction(n, edges):
    """Largest connected component of a subgraph, as a fraction of n."""
    if not edges:
        return 0.0
    nb = {}
    for i, j in edges:
        nb.setdefault(i, []).append(j)
        nb.setdefault(j, []).append(i)
    seen, best = set(), 0
    for s in nb:
        if s in seen:
            continue
        stack, size = [s], 0
        seen.add(s)
        while stack:
            v = stack.pop()
            size += 1
            for w in nb[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        best = max(best, size)
    return best / n


def triangle_edges(adj):
    """Undirected edges that belong to at least one triangle."""
    out = set()
    for i, nb in enumerate(adj):
        for j in nb:
            if i < j and (adj[i] & adj[j]):
                out.add((i, j))
    return out


def all_edges(adj):
    return {(i, j) for i, nb in enumerate(adj) for j in nb if i < j}
