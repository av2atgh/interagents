# Enron email as a negative case: correspondence shows no coupled/solitary transition

Report of the first empirical investigation, 2026-10-13. Code, commands and the
full numbers are in this directory (`README.md`, `results/enron.json`).

## Question

The manuscript predicts a discontinuous transition between a coupled phase,
in which a joint activity is sustained, and an absorbing solitary phase. The
transition requires a cost `c > 0` of an unreciprocated offer: at `c = 0`
offering is free, the absorbing state disappears, and the non-adaptive queueing
picture with exogenous priority is recovered. The Conclusions state the
corollary for data: exponents close to the non-adaptive family in
correspondence are evidence that priority there is close to exogenous.

Ordinary email is correspondence with `c ≈ 0`: an unanswered message costs
the sender almost nothing. It should therefore be a **negative case**: no
signature of the transition. Email that proposes a synchronous joint activity
(a meeting, a call, a meal) has real `c` and is where the signatures should
appear. This investigation tests both on the Enron corpus.

## Data

- CMU Enron corpus, release 2015-05-07: 517,394 files, deduplicated on
  (sender, timestamp, subject, recipients) to 253,202 messages, mostly
  2000-2001.
- **Ego dyads.** An ego is one of the custodians whose mailbox is in the
  corpus, so both directions of each of its dyads are observed (sent folder
  and inbox). Four custodians were dropped (assistant-kept or duplicate
  mailboxes: `lay-k`, `skilling-j`, `whalley-l`, `phanis-s`).
- Kept: personal messages (1-3 recipients) between an ego and a human
  `enron.com` address, inside the ego's coverage window (5th-95th percentile
  of its traffic). Result: 145 egos, 16,259 dyads, 119,455 messages.
- **Offer and answer.** A thread start (no `Re:`/`Fw:`) is an offer. It is
  answered if the recipient writes back within 7 days and before the sender
  writes again. Using thread starts removes the trivial clustering of
  replies inside one conversation.
- **High-c.** A thread start proposing a synchronous joint activity, by
  regex on proposal phrasing ("can we meet", "are you free", "lunch
  tomorrow?", ...): 1,111 messages. Precision about 65-70% on a hand-checked
  sample of 40.
- **Clock.** Interevent times are measured on a deseasonalised clock (the
  corpus hour-of-week activity profile), so circadian and weekly cycles
  do not generate the tails (Malmgren et al., PNAS 2008).

## Tests and results

### 1. Persistence of the dyad state

If a dyad has a slow coupled/solitary state, the answered/unanswered outcomes
of successive offers should cluster beyond a within-dyad permutation, which
preserves each dyad's answer rate.

| offers | sequences | same-outcome, observed | permutation null | excess |
|---|---|---|---|---|
| all | 929 | 0.682 | 0.654 | +0.028 ± 0.003 |
| high-c | 12 | — | — | too few |

The excess is significant but small. It is also the signature of any common
time-varying driver (project phases, absences), so it does not discriminate
a strategic state from exogenous modulation.

### 2. Silence and the price of recovery

Answer probability of an offer against the silence that precedes it, in units
of the dyad's median gap, centred on the dyad's own answer rate:

| silence (median gaps) | <0.5 | 0.5-1 | 1-2 | 2-4 | 4-8 | 8-16 | 16-32 | >32 |
|---|---|---|---|---|---|---|---|---|
| low-c, deviation | +0.006 | +0.010 | +0.009 | −0.002 | −0.011 | −0.018 | −0.029 | −0.033 |
| low-c, n | 11,422 | 4,758 | 6,730 | 5,540 | 4,263 | 2,670 | 1,338 | 1,226 |
| high-c, n | 198 | 92 | 160 | 122 | 109 | 32 | 27 | 28 |

The decline is gradual and monotone, with no threshold. The high-c bins at
long silence hold about 30 offers each, with standard errors of 0.05-0.09.

The model's price of recovery (unilateral persistence of order `0.74 τ_mem`)
predicts that, after a silence, the run of unanswered messages from the
party restarting the dyad grows with the length of the silence. It does
not:

| silence (median gaps) | 1-2 | 2-4 | 4-8 | 8-16 | 16-32 | >32 |
|---|---|---|---|---|---|---|
| mean one-sided run | 5.7 | 5.9 | 6.2 | 6.0 | 5.6 | 5.4 |
| fraction ≥ 2 | 0.53 | 0.55 | 0.53 | 0.54 | 0.50 | 0.46 |

### 3. Load and the critical degree

The model predicts that attention divides as `1/(k+a)`, so answer probability
should fall beyond a critical load `k_c`, weak ties first.

Measured naively, answer probability *rises* with load (0.11 at fewer than 10
recent correspondents, 0.35 at 80-120). This is a presence confound: an
absent ego both hears from fewer people and answers nobody. Restricting to
offers after which the ego sent something within 24 h, and regressing the
answer on log inbound load (distinct senders, previous 7 days), log own sends
and log tie strength, with ego fixed effects and ego-clustered errors:

| offers | n | log inbound load | log own sends | log tie strength |
|---|---|---|---|---|
| all | 16,200 | −0.024 ± 0.009 | +0.025 ± 0.010 | +0.051 ± 0.005 |
| weak ties (≤2 messages in 90 d) | 5,332 | +0.009 ± 0.012 | +0.013 ± 0.010 | +0.023 ± 0.014 |
| strong ties | 10,868 | −0.042 ± 0.011 | +0.029 ± 0.013 | +0.048 ± 0.012 |
| high-c | 292 | +0.024 ± 0.061 | +0.009 ± 0.037 | +0.053 ± 0.027 |

Load has a weak negative effect, with no threshold in the binned profile, and
it falls on strong ties rather than weak ones. That is the opposite of the
mean-field picture, in which the weakest ties are the first to lose their
coupled state.

### 4. The interevent exponent

Non-adaptive queueing ties the exponent to queue length (proxy: ego load).
The adaptive model predicts that, near the transition, it tracks occupancy
(proxy: fraction of a dyad's weeks with traffic in both directions). Tail
exponent (MLE above 3 median gaps) for 1,090 dyad-directions with ≥20
messages, in terciles:

| occupancy \ ego load | low | mid | high |
|---|---|---|---|
| low | 1.89 | 2.01 | 2.00 |
| mid | 2.00 | 2.09 | 1.99 |
| high | 1.78 | 2.16 | 2.16 |

A regression of burstiness on standardised occupancy and log load gives
`β_occ = +0.003 ± 0.005` and `β_load = −0.010 ± 0.005`: neither predictor
matters. The exponent sits near 2 throughout, inside the non-adaptive family
`α = 1 + 1/max(L−1) ≤ 2`.

## Conclusion

Enron correspondence is a negative case, and the kind the theory expects.
None of the signatures of the transition appears:

- no bistable dyad state beyond what common drivers explain;
- no threshold in silence;
- no price of recovery;
- no critical load, and no preferential loss of weak ties;
- no dependence of the exponent on occupancy.

The exponent near 2, independent of both proxies, is what exogenous priority
gives. Read through the manuscript, this places ordinary email at `c ≈ 0`,
outside the domain in which participation is strategic.

The case in which the theory makes a positive prediction, offers with real
cost, could not be tested here. Proposals for synchronous joint activity are
about 1% of personal email (1,111 messages), only 12 dyad-directions carry
four or more, and every high-c estimate above is consistent with zero and
with the model alike. **This is a null for correspondence, not a test of the
transition.**

## Limitations

- Correspondent aliases (e.g. `k..allen` vs `phillip.allen`) are merged
  only for egos.
- Answers are inferred from timing with a 7-day window; sensitivity to the
  window is not yet checked.
- The high-c classifier was validated on 40 messages only.
- The persistence and silence effects cannot separate strategic withdrawal
  from a project ending.
- The firm's collapse in late 2001 truncates coverage and was not used as
  a shock.

## What would test the positive prediction

A record in which the synchronous joint activity itself is the primary event,
so that unreciprocated offers are frequent and observed:

- calendar invitations with accept/decline (Avocado corpus, LDC2015T03);
- event RSVPs with attendance;
- call logs that record missed calls (MIT Reality Mining, Copenhagen Networks
  Study).

The same four tests, run without change, would then be the experiment of
which Enron is the control.
