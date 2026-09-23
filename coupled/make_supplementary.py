"""Build ../supplementary.tex from the stored simulation results.

The manuscript keeps the two widest parameter scans out of the main text and
refers to them as Supplementary Tables I and II. Nothing is transcribed by
hand: the numbers come from the same result files as the tables in the paper.

  Supplementary Table I    pair sensitivity, one parameter at a time
  Supplementary Table II   network results against run length, initial belief,
                           and the remaining parameters
  Supplementary Table III  robustness of the coupled-edge criterion
  Supplementary Table IV   finite size
"""

import json
import os
from collections import defaultdict

import numpy as np

import emit_net as EN

OUT = os.environ.get("OUT", "../supplementary.tex")

TITLE = ("Supplementary Material for\\\\ ``Absorbing phase transition in a "
         "queueing model\\\\ of coupled adaptive agents''")


def compact(m, e, dec=3):
    return f"${m:.{dec}f}({max(1, round(e * 10 ** dec))})$"


def tc_cell(mean, k, exposure):
    """Censored-exponential mean collapse time, or a lower bound."""
    if k == 0:
        ex = int(np.floor(np.log10(exposure)))
        return rf"$>{exposure / 10 ** ex:.0f}\times10^{{{ex}}}$"
    ex = int(np.floor(np.log10(mean)))
    mant = mean / 10 ** ex
    return (rf"${mant:.2f}(\pm{mant / np.sqrt(k):.2f})"
            rf"\times10^{{{ex}}}$")


def sci(x):
    """25000 -> 2.5\\times10^{4}, for column headings."""
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    if -2 <= e <= 3:
        return f"${x:g}$"
    m = x / 10 ** e
    mant = f"{m:g}"
    return (rf"$10^{{{e}}}$" if mant == "1"
            else rf"${mant}\times10^{{{e}}}$")


# ------------------------------------------------- Supplementary Table I
def supp_pair_sensitivity(path="results_survival.json"):
    D = json.load(open(path))
    spec = [(r"$\tau_{\rm mem}$", "tau_scan", "tau", "{:g}"),
            (r"$R_I$", "R_scan", "R_I", "{:.4f}"),
            (r"$\epsilon$", "eps_scan", "eps", "{:g}"),
            (r"$\hat p(0)$", "p_init_scan", "p_init", "{:.2f}"),
            (r"$T$", "T_scan", "n_steps", None)]
    body = []
    for label, key, field, fmt in spec:
        g = defaultdict(list)
        for r in D[key]:
            g[r[field]].append(r)
        first = True
        for v in sorted(g):
            rs = g[v]
            ev = [r["t_collapse"] for r in rs if r["t_collapse"] is not None]
            exposure = sum(r["t_collapse"] if r["t_collapse"] is not None
                           else r["n_steps"] for r in rs)
            k = len(ev)
            mean = exposure / k if k else np.inf
            phi = np.array([r["phi"] for r in rs], float)
            sem = phi.std(ddof=1) / np.sqrt(phi.size)
            val = sci(v) if fmt is None else f"${fmt.format(v)}$"
            body.append(f"{label if first else ''} & {val} & {len(rs)} & "
                        f"${k/len(rs):.3f}$ & {tc_cell(mean, k, exposure)} & "
                        f"{compact(phi.mean(), sem)} \\\\")
            first = False
        body.append(r"\colrule")
    caption = (
        r"Sensitivity of the pair at the reference point ($L=2$, $c=0.35$, "
        r"$R_I=0.8428$, $\epsilon=10^{-2}$, $\tau_{\rm mem}=200$), one "
        r"parameter at a time, $N$ independent realisations each. "
        r"``collapsed'' is the fraction of realisations that reached the "
        r"solitary state within the run; $\langle T_c\rangle$ is the "
        r"censored-exponential mean collapse time in steps, with a lower bound "
        r"given where no collapse was observed; $\phi$ is the occupancy, with "
        r"the standard error in units of the last digit. Runs are "
        r"$2\times10^{5}$ steps except in the last block, where the run length "
        r"is the variable. Note that $\langle T_c\rangle$ is independent of "
        r"the run length while $\phi$ is not.")
    return r"""\begin{table}[htbp]
\caption{%s}
\label{supp:sensitivity}
\begin{ruledtabular}
\begin{tabular}{llcccc}
 & value & $N$ & collapsed & $\langle T_c\rangle$ & $\phi$ \\
\colrule
%s
\end{tabular}
\end{ruledtabular}
\end{table}""" % (caption, "\n".join(body[:-1]))


# ------------------------------------------------ Supplementary Table II
def supp_net_sensitivity():
    def rows_of(body, ncol):
        out = []
        for line in body.split("\n"):
            cells = [c.strip() for c in line.rstrip(" \\").split("&")]
            cells += [""] * (ncol + 1 - len(cells))
            out.append(" & ".join(cells) + r" \\")
        return out

    ln_body, Ts = EN.length_table()
    pi_body, p0s = EN.pinit_table()
    sens = [(lbl, *EN.sensitivity_block(f)) for f, lbl in
            (("eps", r"$\epsilon$"), ("tau", r"$\tau_{\rm mem}$"),
             ("R", r"$R_I$"), ("L", r"$L$"))]
    ncol = max([len(Ts), len(p0s)] + [len(v) for _, _, v in sens])

    def block(title, colvals, body, fmt=lambda v: f"${v:g}$"):
        head = ["$c$"] + [fmt(v) for v in colvals]
        head += [""] * (ncol + 1 - len(head))
        return ([rf"\multicolumn{{{ncol+1}}}{{l}}{{\textit{{{title}}}}} \\",
                 " & ".join(head) + r" \\", r"\colrule"]
                + rows_of(body, ncol))

    out = block("run length $T$ (steps)", Ts, ln_body, sci)
    out += [r"\colrule"] + block(r"initial belief $\hat p(0)$", p0s, pi_body)
    for lbl, body, vals in sens:
        out += [r"\colrule"] + block(lbl, vals, body)
    caption = (
        r"Network results against run length, initial belief and the remaining "
        r"parameters. Entries are the coupled edge fraction with the standard "
        r"error in units of the last digit, on the randomized network at "
        r"$n=600$, $\langle k\rangle=4$, over $8$ to $24$ independent "
        r"realisations per entry. Unless it is the variable, the run is "
        r"$4\times10^{5}$ steps, the initial belief $0.85$, "
        r"$\epsilon=10^{-2}$, $\tau_{\rm mem}=200$, $R_I=0.95$ and $L=2$. "
        r"A dash means the combination was not run.")
    return r"""\clearpage
\begin{table}[p]
\caption{%s}
\label{supp:netsens}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{ruledtabular}
\begin{tabular}{l%s}
%s
\end{tabular}
\end{ruledtabular}
\end{table}""" % (caption, "c" * ncol, "\n".join(out))


# ----------------------------------------------- Supplementary Table III
def supp_finitesize():
    body, ns = EN.finitesize_table()
    head = " & ".join([" ", "$c$"] + [f"${n}$" for n in ns])
    caption = (
        r"Finite size. Giant component $S$ of the coupled subgraph at "
        r"$\langle k\rangle=4$, $R_I=0.95$, $4\times10^{5}$ steps, with the "
        r"standard error in units of the last digit, over $N=24$ realisations "
        r"at $n=300$ and $600$, $16$ at $1200$, $12$ at $2400$ and $8$ at "
        r"$4800$. The randomized network shows no size dependence; under "
        r"triadic closure $S$ falls systematically with $n$ at every cost "
        r"below the transition. Figure 6 of the main text plots the same "
        r"measurement over the full range of $c$.")
    return r"""\begin{table}[htbp]
\caption{%s}
\label{supp:finitesize}
\begin{ruledtabular}
\begin{tabular}{ll%s}
%s \\
\colrule
%s
\end{tabular}
\end{ruledtabular}
\end{table}""" % (caption, "c" * len(ns), head, body)


# ------------------------------------------------ Supplementary Table IV
def supp_robust():
    body, ths = EN.robustness_table()
    head = " & ".join(["$c$"] + [sci(t) for t in ths])
    caption = (
        r"Robustness of the coupled-edge criterion. Giant component $S$ of the "
        r"coupled subgraph on the randomized network, $n=600$, "
        r"$4\times10^{5}$ steps, $N=24$ realisations, as the rate $w$ above "
        r"which an edge is called coupled is varied across the gap between the "
        r"two bands of Fig.~4(b) of the main text. The value used throughout "
        r"is $10\epsilon^{2}=10^{-3}$. Over the decade from $3\times10^{-4}$ "
        r"to $3\times10^{-3}$ the order parameter moves by less than $0.012$ "
        r"in the coupled phase; outside it the criterion fails, the "
        r"exploration floor percolating by itself at a cut of $\epsilon^{2}$ "
        r"and genuinely coupled edges being deleted at $5\times10^{-3}$.")
    return r"""\begin{table}[htbp]
\caption{%s}
\label{supp:robust}
\begin{ruledtabular}
\begin{tabular}{l%s}
%s \\
\colrule
%s
\end{tabular}
\end{ruledtabular}
\end{table}""" % (caption, "c" * len(ths), head, body)


PREAMBLE = r"""%% Generated by coupled/make_supplementary.py -- do not edit by hand.
\documentclass[aps,pre,onecolumn,superscriptaddress,nofootinbib]{revtex4-2}

\usepackage{amsmath,amssymb}
\usepackage{booktabs}

%% Tables are numbered as Supplementary Table I, II, ...
\renewcommand{\tablename}{Supplementary Table}

\begin{document}

\title{%s}

\author{Alexei Vazquez}
\email{alexei@nodeslinks.com}
\affiliation{Nodes \& Links Ltd, Salisbury House, Station Road,
Cambridge, CB1 2LA, UK}

\date{\today}

\maketitle

The four tables below are reference material for the main text, which quotes
only the entries its argument uses and reports them here in full. The first two
are the parameter scans; the last two are the
criterion-robustness and finite-size scans behind Sec.~VI\,A. Every entry is produced from the stored simulation
output by \texttt{make\_supplementary.py} in the code repository,
\url{https://github.com/av2atgh/interagents}; nothing is transcribed by hand.
Section, equation and table numbers with no ``Supplementary'' prefix refer to
the main text.

""" % TITLE


def build():
    # Supplementary tables are numbered by the LaTeX table counter, so this
    # list is what fixes I, II, III, IV. It must follow the order in which the
    # main text first cites them.
    tables = [supp_pair_sensitivity(),   # Supp. Tab. I,   cited in Sec. IV A
              supp_net_sensitivity(),    # Supp. Tab. II,  cited in Sec. IV A
              supp_robust(),             # Supp. Tab. III, cited in Sec. VI A
              supp_finitesize()]         # Supp. Tab. IV,  cited in Sec. VI A
    doc = PREAMBLE + "\n\n".join(tables) + "\n\n\\end{document}\n"
    with open(OUT, "w") as fh:
        fh.write(doc)
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    build()
