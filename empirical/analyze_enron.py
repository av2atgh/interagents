"""Four tests of the coupled/solitary transition on Enron ego dyads.

Usage: python -I analyze_enron.py <enron_dyads.pkl> <out.json>

A "start" is a message that opens a thread (no Re:/Fw:). Starts are the
offers: whether each is answered is the dyad's participation outcome, free of
the trivial clustering of replies inside one conversation.

1. Persistence.  Within a dyad and direction, are answered and unanswered
   starts clustered in time beyond a within-dyad permutation?  A slow
   coupled/solitary state predicts lag-1 excess > 0.
2. Silence.  Answer probability of a start vs the silence preceding it, in
   units of the dyad's median gap, centred on the dyad's own answer rate.
   A memoryless dyad is flat; the model predicts decay on the memory scale,
   steeper for high-c offers.  Also: one-sided runs before the first answer.
3. Load.  Answer probability of starts *to* the ego vs the ego's number of
   distinct correspondents in the previous 30 days, centred on the ego.
   The model predicts a drop concentrated on weak ties.
4. Exponent.  Tail of interevent times (deseasonalised clock) vs occupancy
   (fraction of weeks with traffic both ways) and ego load.  Non-adaptive
   queueing: load decides.  Adaptive: occupancy decides.
"""
import sys
import json
import pickle
import collections
import numpy as np

rng = np.random.default_rng(20261013)
DAY = 86400.0
WEEK = 7 * DAY


def load(path):
    D = pickle.load(open(path, "rb"))
    return D["dyads"], D["window"]


# ---------------------------------------------------------------- clock
def seasonal_clock(dyads):
    """Map wall time to activity time using the hour-of-week profile."""
    ts = np.concatenate([v["t"] for v in dyads.values()])
    how = ((ts // 3600 - 72) % 168).astype(int)   # epoch starts Thursday 00:00 UTC
    prof = np.bincount(how, minlength=168).astype(float) + 1.0
    prof /= prof.mean()
    cum = np.concatenate([[0.0], np.cumsum(prof)])  # in hours of activity

    def clock(t):
        h = t / 3600.0
        wk, rem = np.divmod(h - 72, 168)
        i = rem.astype(int)
        return (wk * 168 + cum[i] + (rem - i) * prof[i]) / 24.0   # activity-days
    return clock


# ---------------------------------------------------------------- 1
def persistence(dyads, min_starts=10, n_perm=200):
    out = {}
    for label, sel in (("all", None), ("high_c", True)):
        obs, null = [], []
        n_seq = 0
        for v in dyads.values():
            st = ~v["rep"]
            for d in (+1, -1):
                m = st & (v["d"] == d)
                if sel:
                    m &= v["hc"] | False
                a = v["ans"][m].astype(float)
                if len(a) < (min_starts if not sel else 4) or a.min() == a.max():
                    continue
                n_seq += 1
                obs.append(np.mean(a[1:] == a[:-1]))
                null.append(np.mean([np.mean(p[1:] == p[:-1])
                                     for p in (rng.permutation(a) for _ in range(n_perm))]))
        obs, null = np.array(obs), np.array(null)
        diff = obs - null
        out[label] = dict(n_sequences=n_seq,
                          same_outcome_obs=float(obs.mean()) if n_seq else None,
                          same_outcome_null=float(null.mean()) if n_seq else None,
                          excess=float(diff.mean()) if n_seq else None,
                          excess_se=float(diff.std(ddof=1) / np.sqrt(n_seq)) if n_seq > 1 else None)
    return out


# ---------------------------------------------------------------- 2
def silence(dyads, clock, min_events=8):
    edges = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 1e9])
    acc = {k: [[] for _ in range(len(edges) - 1)] for k in ("low_c", "high_c")}
    runs = collections.defaultdict(list)
    for v in dyads.values():
        if len(v["t"]) < min_events:
            continue
        tc = clock(v["t"])
        gaps = np.diff(tc)
        gaps = gaps[gaps > 0]
        if len(gaps) < 3:
            continue
        med = np.median(gaps)
        st = np.where(~v["rep"])[0]
        st = st[st > 0]
        if len(st) < 4:
            continue
        base = v["ans"][st].mean()
        for k in st:
            x = (tc[k] - tc[k - 1]) / med
            b = np.searchsorted(edges, x, side="right") - 1
            acc["high_c" if v["hc"][k] else "low_c"][b].append(v["ans"][k] - base)
        # one-sided runs: after each silence, count unanswered messages from
        # the first sender before the first answered exchange
        k = 1
        while k < len(tc):
            x = (tc[k] - tc[k - 1]) / med
            if x >= 1:
                s = v["d"][k]
                n = 0
                j = k
                while j < len(tc) and v["d"][j] == s and not v["ans"][j]:
                    n += 1
                    j += 1
                b = np.searchsorted(edges, x, side="right") - 1
                runs[int(b)].append(n)
            k += 1
    res = dict(bins=[f"[{edges[i]:g},{edges[i+1]:g})" for i in range(len(edges) - 1)])
    for key, A in acc.items():
        res[key] = [dict(n=len(a), mean_dev=float(np.mean(a)) if a else None,
                         se=float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None)
                    for a in A]
    res["unanswered_run"] = {res["bins"][b]: dict(n=len(r), mean=float(np.mean(r)),
                                                  frac_ge2=float(np.mean(np.array(r) >= 2)))
                             for b, r in sorted(runs.items())}
    return res


# ---------------------------------------------------------------- 3
def load_test(dyads):
    by_ego = collections.defaultdict(list)          # ego -> [(t, alter)]
    for (e, a), v in dyads.items():
        for t in v["t"]:
            by_ego[e].append((t, a))
    load_at = {}
    for e, lst in by_ego.items():
        lst.sort()
        ts = np.array([x[0] for x in lst])
        al = [x[1] for x in lst]
        load_at[e] = (ts, al)

    def k_prior(e, t, span=30 * DAY):
        ts, al = load_at[e]
        lo, hi = np.searchsorted(ts, [t - span, t])
        return len(set(al[lo:hi]))

    rows = []
    for (e, a), v in dyads.items():
        for i in np.where((v["d"] == -1) & ~v["rep"])[0]:
            t = v["t"][i]
            tie = np.sum((v["t"] < t) & (v["t"] >= t - 90 * DAY))
            rows.append((e, k_prior(e, t), tie, bool(v["hc"][i]), bool(v["ans"][i])))
    egos = np.array([r[0] for r in rows])
    k = np.array([r[1] for r in rows], float)
    tie = np.array([r[2] for r in rows])
    hc = np.array([r[3] for r in rows])
    ans = np.array([r[4] for r in rows], float)
    # centre on each ego
    dev = ans.copy()
    kc = k.copy()
    for e in np.unique(egos):
        m = egos == e
        dev[m] -= ans[m].mean()
        kc[m] = k[m] / max(np.median(k[m]), 1)
    edges = np.array([0, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 1e9])
    res = dict(bins_rel_load=[f"[{edges[i]:g},{edges[i+1]:g})" for i in range(len(edges) - 1)],
               n_offers=int(len(ans)))
    for name, m0 in (("weak_tie", tie <= 2), ("strong_tie", tie > 2),
                     ("high_c", hc), ("all", np.ones_like(hc))):
        b = np.searchsorted(edges, kc[m0], side="right") - 1
        d = dev[m0]
        res[name] = [dict(n=int((b == i).sum()),
                          mean_dev=float(d[b == i].mean()) if (b == i).any() else None,
                          se=float(d[b == i].std(ddof=1) / np.sqrt((b == i).sum())) if (b == i).sum() > 1 else None)
                     for i in range(len(edges) - 1)]
    # absolute load, pooled: is there a threshold in k itself?
    kedges = np.array([0, 10, 20, 30, 40, 60, 80, 120, 1e9])
    b = np.searchsorted(kedges, k, side="right") - 1
    res["abs_load"] = dict(bins=[f"[{kedges[i]:g},{kedges[i+1]:g})" for i in range(len(kedges) - 1)],
                           answer_rate=[float(ans[b == i].mean()) if (b == i).any() else None
                                        for i in range(len(kedges) - 1)],
                           mean_dev=[float(dev[b == i].mean()) if (b == i).any() else None
                                     for i in range(len(kedges) - 1)],
                           n=[int((b == i).sum()) for i in range(len(kedges) - 1)])
    return res


def load_controlled(dyads):
    """Test 3 with the ego's presence controlled.

    Raw load is confounded with being in the office: an absent ego both
    receives from fewer people and answers nobody. Keep only offers after which
    the ego sent something to anyone within 24 h, and regress the answer on
    log inbound load (distinct senders, previous 7 days), log own sends
    (previous 7 days) and log tie strength (dyad messages, previous 90 days),
    with ego fixed effects and ego-clustered errors.
    """
    inb = collections.defaultdict(list)
    outb = collections.defaultdict(list)
    for (e, a), v in dyads.items():
        for t, d in zip(v["t"], v["d"]):
            (outb if d == 1 else inb)[e].append((t, a))
    idx = {}
    for e in set(inb) | set(outb):
        I = sorted(inb[e])
        idx[e] = (np.array([x[0] for x in I]), [x[1] for x in I],
                  np.array(sorted(x[0] for x in outb[e])))
    rows = []
    for (e, a), v in dyads.items():
        ti, ai, to = idx[e]
        for i in np.where((v["d"] == -1) & ~v["rep"])[0]:
            t = v["t"][i]
            lo, hi = np.searchsorted(to, [t, t + DAY])
            if hi == lo:
                continue
            l0, l1 = np.searchsorted(ti, [t - 7 * DAY, t])
            o0, o1 = np.searchsorted(to, [t - 7 * DAY, t])
            tie = np.sum((v["t"] < t) & (v["t"] >= t - 90 * DAY))
            rows.append((e, len(set(ai[l0:l1])), o1 - o0, tie, bool(v["hc"][i]), float(v["ans"][i])))
    E = np.array([r[0] for r in rows])
    cols = [np.log1p(np.array([r[k] for r in rows], float)) for k in (1, 2, 3)]
    tie = np.array([r[3] for r in rows])
    hc = np.array([r[4] for r in rows])
    ans = np.array([r[5] for r in rows])

    def fe(m):
        y = ans[m].copy()
        X = np.column_stack([c[m] for c in cols])
        e = E[m]
        for g in np.unique(e):
            s = e == g
            y[s] -= y[s].mean()
            X[s] -= X[s].mean(0)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        XtXi = np.linalg.inv(X.T @ X)
        meat = np.zeros((3, 3))
        for g in np.unique(e):
            s = e == g
            u = X[s].T @ r[s]
            meat += np.outer(u, u)
        se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        return dict(n=int(m.sum()),
                    **{f"beta_{n}": float(b) for n, b in zip(("log_inbound_k", "log_own_sends", "log_tie"), beta)},
                    **{f"se_{n}": float(x) for n, x in zip(("log_inbound_k", "log_own_sends", "log_tie"), se)})
    allm = np.ones(len(ans), bool)
    return dict(answer_rate_active=float(ans.mean()),
                all=fe(allm), weak_tie=fe(tie <= 2), strong_tie=fe(tie > 2), high_c=fe(hc))


# ---------------------------------------------------------------- 4
def mle_alpha(x, xmin):
    x = x[x >= xmin]
    if len(x) < 50:
        return None, len(x)
    a = 1 + len(x) / np.sum(np.log(x / xmin))
    return float(a), len(x)


def exponent(dyads, window, clock, min_offers=20):
    ego_load = collections.defaultdict(list)
    for (e, a), v in dyads.items():
        for w in np.unique((v["t"] // WEEK).astype(int)):
            ego_load[(e, w)].append(a)
    mean_load = collections.defaultdict(list)
    for (e, w), al in ego_load.items():
        mean_load[e].append(len(set(al)))
    mean_load = {e: float(np.mean(x)) for e, x in mean_load.items()}

    recs = []
    for (e, a), v in dyads.items():
        for d in (+1, -1):
            t = v["t"][v["d"] == d]
            if len(t) < min_offers:
                continue
            g = np.diff(clock(t))
            g = g[g > 1e-4]
            if len(g) < min_offers - 1:
                continue
            w_all = np.arange(int(v["t"][0] // WEEK), int(v["t"][-1] // WEEK) + 1)
            wk_out = set((v["t"][v["d"] == +1] // WEEK).astype(int))
            wk_in = set((v["t"][v["d"] == -1] // WEEK).astype(int))
            occ = np.mean([w in wk_out and w in wk_in for w in w_all])
            B = (g.std() - g.mean()) / (g.std() + g.mean())
            recs.append(dict(occ=float(occ), load=mean_load[e], B=float(B),
                             g=g / np.median(g)))
    occ = np.array([r["occ"] for r in recs])
    ld = np.array([r["load"] for r in recs])
    B = np.array([r["B"] for r in recs])
    # standardised regression B ~ occ + log load
    X = np.column_stack([np.ones_like(occ), (occ - occ.mean()) / occ.std(),
                         (np.log(ld) - np.log(ld).mean()) / np.log(ld).std()])
    beta, *_ = np.linalg.lstsq(X, B, rcond=None)
    resid = B - X @ beta
    s2 = resid @ resid / (len(B) - 3)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    out = dict(n_sequences=len(recs),
               burstiness_regression=dict(beta_occ=float(beta[1]), se_occ=float(se[1]),
                                          beta_logload=float(beta[2]), se_logload=float(se[2])),
               corr_occ_logload=float(np.corrcoef(occ, np.log(ld))[0, 1]))
    qo = np.quantile(occ, [0, 1 / 3, 2 / 3, 1])
    ql = np.quantile(ld, [0, 1 / 3, 2 / 3, 1])
    table = []
    for i in range(3):
        row = []
        for j in range(3):
            sel = [r for r in recs
                   if qo[i] <= r["occ"] <= qo[i + 1] and ql[j] <= r["load"] <= ql[j + 1]]
            if not sel:
                row.append(None)
                continue
            g = np.concatenate([r["g"] for r in sel])
            a, n = mle_alpha(g, 3.0)
            row.append(dict(alpha=a, n_tail=n, n_seq=len(sel),
                            B=float(np.mean([r["B"] for r in sel]))))
        table.append(row)
    out["alpha_table"] = dict(rows="occupancy tercile (low->high)", cols="ego load tercile (low->high)",
                              occ_edges=qo.tolist(), load_edges=ql.tolist(), cells=table)
    return out


def main(inp, outp):
    dyads, window = load(inp)
    clock = seasonal_clock(dyads)
    res = dict(n_dyads=len(dyads), n_egos=len(window))
    res["1_persistence"] = persistence(dyads)
    print("1 done", flush=True)
    res["2_silence"] = silence(dyads, clock)
    print("2 done", flush=True)
    res["3_load"] = load_test(dyads)
    res["3b_load_controlled"] = load_controlled(dyads)
    print("3 done", flush=True)
    res["4_exponent"] = exponent(dyads, window, clock)
    print("4 done", flush=True)
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
