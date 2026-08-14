"""R17: agents meeting in groups. Hyperedges of size m require all m members.

Reproduces the hypergraph section:
  A  the analytic critical group degree d_c(m), Eq. (H2), and the maximum group
     size it implies
  B  simulation across (m, d), testing d_c
  C  the failure of d_c at m >= 3, and its fluctuation origin
"""

import numpy as np
from hypergraph import d_crit, max_group_size, regular_hypergraph, run_hyper

R = 0.95
CS = (0.002, 0.010, 0.020, 0.050)


def hdr(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


hdr("A  CRITICAL GROUP DEGREE  d_c(m) = [n^n B^(n+1)/(c(n+1)^(n+1))]^(1/(m-1)) - a\n"
    "   n = a(m-1), B = R_I + c.  m=2 reproduces the pairwise k_c.")
print(f"  {'m':>3} | " + " ".join(f"c={c}".rjust(9) for c in CS))
for m in (2, 3, 4, 5, 6, 8):
    print(f"  {m:>3} | " + " ".join(f"{d_crit(m, R, c):9.2f}" for c in CS))
print("\n  maximum group size for an agent devoted to a single group (d_c >= 1):")
for c in CS:
    print(f"    c = {c:<6} m_max = {max_group_size(R, c)}")

hdr("B  SIMULATION: fraction of groups that stay coupled, vs group size and\n"
    "   group degree. Coupled means a meeting rate above 10x the eps^m floor.")
print(f"  {'m':>3} {'d_c':>7} | " + " ".join(f"d={d}".rjust(7) for d in (1, 2, 4, 8, 16)))
for m in (2, 3, 4):
    row = []
    for d in (1, 2, 4, 8, 16):
        na = 240
        while (na * d) % m:
            na += 1
        ma, me, ne = regular_hypergraph(na, m, d, seed=1)
        r = run_hyper(ma, me, na, ne, m, R_I=R, c=0.020, n_steps=40_000, seed=1)
        row.append((r > 10 * (0.01 ** m)).mean())
    print(f"  {m:>3} {d_crit(m, R, 0.020):>7.2f} | " + " ".join(f"{v:7.3f}" for v in row))
print("\n  m=2 tracks d_c. m=3 does NOT: d_c = 1.60 permits d = 1, yet nothing")
print("  survives. The mean-field fixed point exists but is not reached.")

hdr("C  WHY: the group dies if ANY single member's belief drifts below\n"
    "   threshold, so escape is fluctuation-driven. Reducing belief noise by\n"
    "   lengthening tau_mem should partially rescue it -- and does.")
print(f"  m=3, d=1.  mean meeting rate")
print(f"  {'c':>6} {'d_c':>6} | " + " ".join(f"tau={t}".rjust(9)
                                            for t in (200, 1000, 5000, 20000)))
for c in (0.020, 0.005):
    row = []
    for tau in (200, 1000, 5000, 20000):
        na = 240
        ma, me, ne = regular_hypergraph(na, 3, 1, seed=1)
        r = run_hyper(ma, me, na, ne, 3, R_I=R, c=c, tau_mem=float(tau),
                      n_steps=60_000, seed=1)
        row.append(r.mean())
    print(f"  {c:>6.3f} {d_crit(3, R, c):>6.2f} | " + " ".join(f"{v:9.4f}" for v in row))
print("\n  Rates remain far below the mean-field value (~0.08 at c=0.020), so the")
print("  rescue is partial: groups of three or more are qualitatively more")
print("  fragile than the existence criterion allows.")
