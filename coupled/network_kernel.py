"""JIT-compiled inner loop for run_network.

Identical model to the NumPy implementation in network.py, written as scalar
loops so that numba can compile it: the arrays are small (a few thousand
directed edges), so the NumPy version spends most of its time in per-call
overhead rather than in arithmetic.

Two implementation details worth recording, because both are exact rather than
approximations:

* Exploration is an independent Bernoulli(epsilon) trial per edge per step. The
  kernel does not draw one uniform per edge; it draws the gap to the next
  exploration from the geometric distribution and counts down, which is the same
  process and costs O(epsilon E) instead of O(E) random numbers.

* The edge-level first-passage time to `hold` consecutive silent steps is
  obtained from the time of the last execution rather than by incrementing a
  silence counter on every edge at every step. A silent run that starts at
  s = last_fire + 1 reaches length `hold` iff the next execution (or the end of
  the run) is at least `hold` steps later, and the first-passage time is s.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def run_kernel(src, dst, rev, starts, deg, n, E, a, R_I, c, lam, epsilon,
               n_steps, burn, hold, win, p_hat, u, seed):
    np.random.seed(seed)
    inv_a = 1.0 / a
    one_m_lam = 1.0 - lam

    x = np.zeros(E)
    counts = np.zeros(E, dtype=np.int64)
    last_fire = np.full(E, -1, dtype=np.int64)
    fpt = np.full(E, -1, dtype=np.int64)
    recip = np.zeros(E, dtype=np.uint8)
    sel_of_node = np.full(n, -1, dtype=np.int64)
    prev_recip = np.zeros(n, dtype=np.int64)
    n_prev = 0
    n_win = (n_steps + win - 1) // win
    series = np.zeros(n_win)

    # gap to the next exploring edge-slot, geometric with parameter epsilon
    if epsilon > 0.0:
        log1m = np.log(1.0 - epsilon)
        gap = np.int64(np.floor(np.log(np.random.random()) / log1m))
    else:
        log1m = 0.0
        gap = np.int64(1) << 62

    for t in range(n_steps):
        # ---- policy, belief update, per-node argmax ----------------------
        # no observation has been made before the first step, so the beliefs
        # enter it exactly as initialised
        lam_t = lam if t > 0 else 1.0
        omt = one_m_lam if t > 0 else 0.0
        for i in range(n):
            b = -1.0
            arg = -1
            s0 = starts[i]
            s1 = s0 + deg[i]
            for e in range(s0, s1):
                ph = lam_t * p_hat[e] + omt * recip[e]
                if ph < 1e-9:
                    ph = 1e-9
                p_hat[e] = ph
                xs = R_I - c * (1.0 / ph - 1.0)
                if xs < 0.0:
                    xs = 0.0
                elif xs > 1.0:
                    xs = 1.0
                xe = u[e] * xs
                if gap == 0:
                    xe = 1.0
                    gap = np.int64(np.floor(np.log(np.random.random()) / log1m))
                else:
                    gap -= 1
                x[e] = xe
                if xe > b:
                    b = xe
                    arg = e
            xo = np.random.random() ** inv_a
            sel_of_node[i] = arg if b > xo else -1

        # ---- clear last step's reciprocation flags -----------------------
        for k in range(n_prev):
            recip[prev_recip[k]] = 0
        n_prev = 0

        # ---- executions --------------------------------------------------
        fired = 0
        for i in range(n):
            e = sel_of_node[i]
            if e < 0:
                continue
            er = rev[e]
            recip[er] = 1
            prev_recip[n_prev] = er
            n_prev += 1
            # both endpoints selected this edge?
            if sel_of_node[dst[e]] == er:
                fired += 1
                if t >= burn:
                    counts[e] += 1
                # close the silent run that ended here
                if fpt[e] < 0:
                    s = last_fire[e] + 1
                    if t - s >= hold:
                        fpt[e] = s
                last_fire[e] = t
            # an offer consumes the attempt, reciprocated or not
            u[e] = np.random.random()
        series[t // win] = series[t // win] + fired

    # silent runs still open when the simulation ended
    for e in range(E):
        if fpt[e] < 0:
            s = last_fire[e] + 1
            if n_steps - s >= hold:
                fpt[e] = s
    return counts, fpt, series
