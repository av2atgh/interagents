"""Stream the CMU Enron tarball into one deduplicated message table.

Usage: python -I ingest_enron.py <enron_mail_20150507.tar.gz> <out.pkl>

Each message appears in several folders (sent, all_documents, discussion
threads), each copy with its own Message-ID, so copies are merged on
(sender, timestamp, subject, recipients). The body kept is the author's own
text: everything from the first quoted or forwarded block onwards is cut.
"""
import sys
import re
import tarfile
import email
import email.utils
from email import policy
import pickle

ADDR = re.compile(r"[\w.+'&-]+@[\w.-]+\.\w+")
CUT = re.compile(
    r"^(-+\s*Original Message\s*-+|-+\s*Forwarded by|>|From:\s.*|"
    r"\S.*\s+on\s+\d\d/\d\d/\d\d\d\d\s+\d\d:\d\d.*|To:\s.*)",
    re.I,
)


def addrs(field):
    if not field:
        return ()
    return tuple(sorted({a.lower() for a in ADDR.findall(str(field))}))


def own_text(body):
    out = []
    for line in body.splitlines():
        if CUT.match(line.strip()):
            break
        out.append(line)
    return "\n".join(out).strip()[:4000]


def main(tar_path, out_path):
    seen = {}
    n_files = 0
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 4 or parts[0] != "maildir":
                continue
            raw = tf.extractfile(m).read()
            n_files += 1
            msg = email.message_from_bytes(raw, policy=policy.compat32)
            try:
                t = email.utils.parsedate_to_datetime(msg["Date"]).timestamp()
            except Exception:
                continue
            sender = addrs(msg["From"])
            if not sender:
                continue
            to, cc, bcc = addrs(msg["To"]), addrs(msg["Cc"]), addrs(msg["Bcc"])
            subj = str(msg["Subject"] or "").strip()
            key = (sender[0], int(t), subj, to + cc)
            folder = "/".join(parts[2:-1]).lower()
            if key in seen:
                seen[key]["folders"].add((parts[1], folder))
                continue
            payload = msg.get_payload(decode=False)
            body = payload if isinstance(payload, str) else ""
            seen[key] = dict(
                t=t,
                sender=sender[0],
                to=to,
                cc=tuple(a for a in cc if a not in to),
                bcc=bcc,
                subject=subj,
                body=own_text(body),
                full_len=len(body),
                folders={(parts[1], folder)},
            )
            if n_files % 50000 == 0:
                print(n_files, len(seen), flush=True)
    rows = list(seen.values())
    print("files", n_files, "unique messages", len(rows))
    with open(out_path, "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
