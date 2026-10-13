"""Turn the deduplicated Enron messages into ego-dyad offer/answer events.

Usage: python -I build_dyads.py <enron_messages.pkl> <out.pkl>

An ego is a custodian whose own mailbox is in the corpus, so both directions of
every dyad (ego, alter) are observed: ego -> alter from the sent folders,
alter -> ego from the inbox. Only personal messages (1-3 recipients, both ends
human enron.com addresses) inside the ego's coverage window are kept.

An offer i -> j is *answered* if j -> i follows within T_ANS and before i's next
message to j. Messages are flagged high-c when they open a thread (no Re:/Fw:)
and propose a synchronous joint activity (meeting, call, meal, game), the
case in which an unreciprocated offer costs the sender something.
"""
import sys
import re
import pickle
import collections
import numpy as np

T_ANS = 7 * 86400.0

# custodians whose sent folder is written by an assistant, or duplicates another
DROP = {"lay-k", "skilling-j", "whalley-l", "phanis-s"}

ACT = (r"(lunch|dinner|breakfast|drinks?|a drink|beers?|a beer|coffee|happy hour|"
       r"golf|tennis|a game|together|meet|a meeting|a call|talk|chat|get together)")
PATS = [
    r"\b(want|wanna|would you like|like|care|interested in|up for) (to )?(grab|get|go|meet|have|do|join|catch|play)\b[^.?!\n]{0,30}\b" + ACT,
    r"\blet'?s (meet|grab|get|have|go|do|catch|play|talk|plan|get together|set up)\b",
    r"\b(can|could|shall|should) we (meet|get together|talk|grab|have|do|go|set up|schedule|sit down|discuss)\b",
    r"\bhow about (lunch|dinner|drinks|coffee|breakfast|a beer|golf|tennis|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|next week)\b",
    r"\bare you (free|available|around|in town|up for)\b",
    r"\bdo you have (time|a few minutes|a minute|some time|a sec)\b",
    r"\b(join|meet) (me|us) (for|at)\b",
    r"\b(what|which) (time|day|night) (works|is good|is best|would work|is better|can you|could you|do you)\b",
    r"\b(when|what time) (can|could|would|do|are|will) (we|you)\b[^.?!\n]{0,40}\b(meet|get together|talk|go|have|grab|be free|be available|lunch|dinner)\b",
    r"\b(set up|schedule|arrange) (a|some) (time|meeting|call|lunch|dinner)\b",
    r"\b(lunch|dinner|drinks|happy hour|coffee|breakfast|golf|tennis)\s*(today|tomorrow|tonight|this week|next week|on (mon|tue|wed|thu|fri|sat|sun)\w*)?\s*\?",
]
HC = re.compile("|".join("(?:%s)" % p for p in PATS), re.I)
SUBJ = re.compile(r"^(lunch|dinner|drinks|happy hour|coffee|breakfast|golf|tennis|get together)\b.{0,20}\??$", re.I)
REPLY = re.compile(r"^\s*(re|fw|fwd)\s*:", re.I)
HUMAN = re.compile(r"^[a-z'-]+\.(?:[a-z]\.{1,2})?[a-z'-]+@enron\.com$")
BOT = re.compile(r"announce|no\.address|outlook|team|enron\.|admin|mailbox|help|request|notice|news|daily|update", re.I)


def ego_addresses(rows):
    sent = collections.defaultdict(collections.Counter)
    for r in rows:
        for c, f in r["folders"]:
            if "sent" in f:
                sent[c][r["sender"]] += 1
    egos = {}
    for c, cnt in sent.items():
        if c in DROP:
            continue
        sur = c.rsplit("-", 1)[0].replace("-", "")
        top = cnt.most_common(1)[0][1]
        al = {a for a, n in cnt.items()
              if n >= 0.05 * top and n >= 5 and sur[:4] in a.split("@")[0].replace("-", "").replace("'", "")}
        egos[c] = al or {cnt.most_common(1)[0][0]}
    return egos


def main(inp, out):
    rows = pickle.load(open(inp, "rb"))
    egos = ego_addresses(rows)
    alias = {a: c for c, al in egos.items() for a in al}
    senders = collections.Counter(r["sender"] for r in rows)

    def human(a):
        return bool(HUMAN.match(a)) and not BOT.search(a.split("@")[0]) and senders[a] > 0

    # personal messages touching an ego
    ev = collections.defaultdict(list)       # (ego, alter) -> [(t, dir, highc, is_reply)]
    ego_times = collections.defaultdict(list)
    for r in rows:
        rec = r["to"] + r["cc"]
        if not (1 <= len(rec) <= 3) or not r["sender"].endswith("@enron.com"):
            continue
        s = alias.get(r["sender"], r["sender"])
        is_rep = bool(REPLY.match(r["subject"]))
        hc = (not is_rep) and bool(HC.search(r["body"]) or SUBJ.match(r["subject"]))
        rec_ids = [alias.get(a, a) for a in rec]
        if s in egos:
            for a in rec_ids:
                if a != s and (a in egos or human(a)):
                    ev[(s, a)].append((r["t"], +1, hc, is_rep))
                    ego_times[s].append(r["t"])
        for a in set(rec_ids):
            if a in egos and a != s and (s in egos or human(r["sender"])):
                ev[(a, s)].append((r["t"], -1, hc, is_rep))
                ego_times[a].append(r["t"])

    # coverage window: 5th-95th percentile of the ego's personal traffic,
    # and the ego must have both sent and received inside it
    window = {}
    for e, ts in ego_times.items():
        ts = np.array(ts)
        window[e] = (np.percentile(ts, 5), np.percentile(ts, 95))

    dyads = {}
    for (e, a), lst in ev.items():
        lo, hi = window[e]
        lst = sorted(x for x in set(lst) if lo <= x[0] <= hi)
        if not lst:
            continue
        t = np.array([x[0] for x in lst])
        d = np.array([x[1] for x in lst], dtype=np.int8)
        hc = np.array([x[2] for x in lst], dtype=bool)
        rp = np.array([x[3] for x in lst], dtype=bool)
        ans = np.zeros(len(t), dtype=bool)
        for k in range(len(t)):
            for m in range(k + 1, len(t)):
                if t[m] - t[k] > T_ANS:
                    break
                if d[m] == d[k]:
                    break          # sender wrote again before any answer
                ans[k] = True
                break
        dyads[(e, a)] = dict(t=t, d=d, hc=hc, rep=rp, ans=ans)

    out_obj = dict(dyads=dyads, window=window, egos={e: sorted(v) for e, v in egos.items()},
                   T_ANS=T_ANS)
    pickle.dump(out_obj, open(out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    n_msg = sum(len(v["t"]) for v in dyads.values())
    n_hc = sum(v["hc"].sum() for v in dyads.values())
    print("egos", len(window), "dyads", len(dyads), "events", n_msg, "high-c", n_hc)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
