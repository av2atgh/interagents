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
                n_steps=60_000, burn_frac=0.3, seed=0, p_init=None):
    """Level-1 agents on a fixed network. Returns per-edge interaction rates."""
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

    # Faithful to the queueing substrate: a task's priority is DRAWN from a
    # distribution the policy controls, and held until that task executes. With
    # one relation this is immaterial, but with several it is what lets an agent
    # rotate among them. Making the priority a deterministic function of p_hat
    # instead locks each agent onto its single best partner and forces the
    # coupled subgraph to be a matching.
    u = rng.random(src.size)

    for t in range(n_steps):
        # policy: value of the relation discounted by the risk of no response
        xstar = np.clip(R_I - c * (1.0 - p_hat) / np.maximum(p_hat, 1e-9), 0.0, 1.0)
        x = u * xstar
        expl = rng.random(src.size) < epsilon
        x = np.where(expl, 1.0, x)

        # each node's best relation, and its solitary alternative
        best = np.zeros(n)
        np.maximum.at(best, src, x)
        x_O = rng.random(n) ** (1.0 / a)
        # a node participates only if its best relation beats its private task
        active = best > x_O
        # select the argmax edge (ties are measure-zero)
        sel = (x >= best[src]) & active[src]
        # guard against a node selecting two edges on an exact tie
        if sel.sum() > active.sum():
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

        # An OFFER consumes the attempt, whether or not it was reciprocated, so
        # the priority is redrawn on selection rather than only on execution.
        # Redrawing only on execution freezes the choice: agent i would offer to
        # the same partner forever, and if that partner's best is not i the pair
        # deadlocks with no mechanism to ever change.
        if sel.any():
            u = np.where(sel, rng.random(src.size), u)

        # i learns whether j selected i on this relation
        p_hat = lam * p_hat + (1 - lam) * sel[rev].astype(float)
        p_hat = np.clip(p_hat, 1e-9, 1.0)

    rate = counts / (n_steps - burn)
    return dict(rate=rate, src=src, dst=dst, rev=rev, n=n)


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
