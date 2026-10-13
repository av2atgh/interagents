# Empirical tests: Enron email

Tests of the coupled/solitary transition on ego dyads of the CMU Enron corpus
(release 2015-05-07). Raw and derived data are not tracked.

```bash
cd empirical
mkdir -p data/enron_raw data/derived results
curl -L -o data/enron_raw/enron_mail_20150507.tar.gz https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
python -I ingest_enron.py data/enron_raw/enron_mail_20150507.tar.gz data/derived/enron_messages.pkl
python -I build_dyads.py data/derived/enron_messages.pkl data/derived/enron_dyads.pkl
python -I analyze_enron.py data/derived/enron_dyads.pkl results/enron.json
```

`ingest_enron.py` streams the tarball (517,394 files -> 253,202 unique
messages). `build_dyads.py` keeps personal messages (1-3 recipients) between a
custodian and human enron.com addresses inside the custodian's coverage window:
145 egos, 16,259 dyads, 119,455 messages, 1,111 high-c proposals. An offer is
answered if the recipient writes back within 7 days and before the sender
writes again. High-c = thread-opening message proposing a meeting, call, meal
or game (regex; ~65-70% precision on a hand-checked sample of 40).

## First-pass results (2026-10-13)

| test | prediction | result |
|---|---|---|
| 1 persistence of answer outcomes of thread starts | slow coupled/solitary state: excess same-outcome | +0.028 +- 0.003 over within-dyad permutation (929 sequences); small, and confounded with project timing |
| 2 answer probability vs preceding silence (units of dyad median gap) | decay on the memory scale | gradual decline, +0.01 below 2 gaps to -0.03 beyond 16; no threshold |
| 2b one-sided run before the first answer after a silence | grows with silence (price of recovery) | flat, 5.7 -> 5.4 messages; no recovery cost |
| 3 answer probability vs ego load, ego presence controlled | drop beyond k_c, weak ties first | -0.024 +- 0.009 per log unit, carried by strong ties (-0.042 +- 0.011), not weak (+0.009 +- 0.012); no threshold |
| 4 interevent tail vs occupancy and load | adaptive: occupancy decides; non-adaptive: load decides | alpha = 1.8-2.2 in all 9 tercile cells; burstiness depends on neither |

Reading: ordinary Enron email behaves as the low-c control should - no
bistability, no critical load, no occupancy-dependent exponent. The high-c test
is underpowered: 1,111 proposals, only 12 dyad-directions with >=4, and 28-32
offers in each long-silence bin. Testing the high-c prediction needs a record in
which synchronous joint activity is the primary event (calendar accept/decline,
RSVPs, call logs with missed calls).

Not yet done: alias merging for alters, sensitivity to the 7-day answer window,
hand validation of the classifier beyond 40 messages, the late-2001 collapse as
a shock.
