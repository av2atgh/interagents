"""Analytic percolation threshold for the coupled subgraph on a configuration
model with arbitrary degree distribution.

DERIVATION
----------
Priorities are x = u X with u ~ U(0,1) drawn per offer, and the solitary task has
CDF F_O(t) = t^a with a = L-1. The probability that agent i, of degree k, selects
one particular neighbour is

    P(i selects j) = int_0^1 u^{k-1} (uX)^a du = X^a / (k + a).            (1)

So the partner's estimate is p_hat = X^a/(k+a), and self-consistency
X = Psi(p_hat) with Psi(p) = R_I - c(1-p)/p = B - c/p, B = R_I + c, gives

    X^{a+1} - B X^a + c (k + a) = 0.                                       (2)

A coupled state exists only while (2) has a real root. Tangency at
X* = aB/(a+1) yields the CRITICAL DEGREE

    k_c = a^a B^{a+1} / [ c (a+1)^{a+1} ] - a.                             (3)

A node cannot sustain relations once its degree exceeds k_c: attention is
divided k ways, so p_hat ~ 1/k, and beyond k_c it falls under the collapse
threshold c/(R_I+c).

SELF-CONSISTENCY
----------------
Equation (3) uses the FULL degree k, but a node whose relations have already
died divides its attention only among the survivors. The relevant quantity is
the number of live relations m, not k. On a locally tree-like graph this closes
as a cavity equation. Let phi be the probability that an edge, followed to one
of its ends, is sustained by that node. A node reached along an edge has degree
k with probability q_k = k p_k / <k>, has Bin(k-1, phi) other live relations, and
sustains the incoming one iff its live degree m = 1 + Bin(k-1, phi) <= k_c:

    phi = sum_k q_k P[ Bin(k-1, phi) <= k_c - 1 ].                         (4)

An edge is live iff sustained from both ends. Following a live edge, the node at
its end has m - 1 further live edges, so the branching ratio is

    b = sum_k q_k E[ (m-1) 1{m <= k_c} ] / phi,                            (5)

and the coupled subgraph has a giant component iff b > 1. Setting phi = 1 in (4)
recovers the naive truncation sum_{k<=k_c} k(k-1) p_k > <k>, i.e. Molloy-Reed
with the second moment cut at k_c -- which is the phi=1 limit and an upper bound
on the coupled region, since it ignores that dying relations free up attention.

Exploration (rate eps per relation, which sets x=1 and therefore wins the
argmax) consumes the attention slot with probability ~ 1-(1-eps)^k, penalising
hubs further. Including it replaces c(k+a) in (2) by c(k+a)(1-eps)^{-k}.
"""

import numpy as np
from scipy.stats import binom


def k_crit(R_I, c, a=1, eps=0.0, kmax=5000):
    """Critical degree, Eq. (3), optionally with the exploration penalty."""
    B = R_I + c
    rhs = a**a * B**(a + 1) / ((a + 1) ** (a + 1))
    k = np.arange(0, kmax)
    lhs = c * (k + a) * (1.0 - eps) ** (-k.astype(float)) if eps > 0 else c * (k + a)
    ok = np.nonzero(lhs <= rhs)[0]
    return int(ok.max()) if ok.size else -1


def solve_phi(pk, kc, tol=1e-12, itmax=2000):
    """Cavity equation (4). pk is an array over k = 0, 1, 2, ..."""
    k = np.arange(len(pk))
    kbar = (k * pk).sum()
    if kbar <= 0:
        return 0.0
    q = k * pk / kbar
    phi = 1.0
    for _ in range(itmax):
        with np.errstate(invalid="ignore"):
            surv = np.where(k >= 1, binom.cdf(kc - 1, np.maximum(k - 1, 0), phi), 1.0)
        new = float((q * surv).sum())
        if abs(new - phi) < tol:
            return new
        phi = new
    return phi


def branching(pk, kc, phi):
    """Branching ratio (5) of the live subgraph."""
    k = np.arange(len(pk))
    kbar = (k * pk).sum()
    q = k * pk / kbar
    tot = 0.0
    for kk in range(1, len(pk)):
        if q[kk] == 0:
            continue
        m = np.arange(0, kk)                      # other live relations
        w = binom.pmf(m, kk - 1, phi)
        live = (m + 1) <= kc
        tot += q[kk] * (w * m * live).sum()
    return tot / max(phi, 1e-15)


def percolates(pk, R_I, c, a=1, eps=0.0):
    kc = k_crit(R_I, c, a, eps)
    if kc < 1:
        return False, 0.0, kc, 0.0
    phi = solve_phi(pk, kc)
    b = branching(pk, kc, phi)
    return b > 1.0, b, kc, phi


def naive_MR(pk, kc):
    """phi = 1 limit: Molloy-Reed with the second moment truncated at k_c."""
    k = np.arange(len(pk))
    kbar = (k * pk).sum()
    sel = k <= kc
    return (k[sel] * (k[sel] - 1) * pk[sel]).sum() / kbar


def degree_pmf(degrees, kmax=None):
    kmax = kmax or int(degrees.max())
    pk = np.bincount(degrees, minlength=kmax + 1).astype(float)
    return pk / pk.sum()


def critical_c(pk, R_I, a=1, eps=0.0, lo=1e-4, hi=1.0, iters=60):
    """Bisect for the c at which the coupled subgraph stops percolating."""
    if not percolates(pk, R_I, lo, a, eps)[0]:
        return np.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if percolates(pk, R_I, mid, a, eps)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
