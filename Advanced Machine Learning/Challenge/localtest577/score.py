#!/usr/bin/python3
"""Portable SOCO test577 scorer (self-contained copy of /opt/soco_scorer/score.py).

Scoring logic is unchanged from the host scorer: a query is correct when the
prediction is within 0.1 * scale of the nearest accepted target point.

Differences from the host copy are path handling only: labels, ledger and the
script itself are resolved relative to this file, so the directory can be moved
anywhere. Prints one JSON line and appends one ledger row.

Usage: ./score.py <submission.csv> <exp_id> [note]
"""
import sys, csv, json, math, hashlib, time, os

BASE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(BASE, 'labels', 'test_labels.json')
LEDGER = os.path.join(BASE, 'ledger.tsv')

if len(sys.argv) < 3:
    sys.exit('usage: score.py submission.csv exp_id [note]')
sub, exp = sys.argv[1], sys.argv[2]
note = ' '.join(sys.argv[3:]).replace('\t', ' ')[:200]

L = json.load(open(P))
raw = open(sub, 'rb').read()
sha = hashlib.sha256(raw).hexdigest()
rows = list(csv.reader(raw.decode().splitlines()))
assert rows[0] == ['id', 'prediction'], 'bad header'
pred = {}
for r in rows[1:]:
    p = json.loads(r[1])
    assert len(p) == 2 and all(math.isfinite(float(v)) for v in p), r[0]
    pred[r[0]] = p
assert set(pred) == set(L) and len(rows) - 1 == len(L), f'need exactly {len(L)} test ids'

ok = 0
for k, v in L.items():
    x, y = map(float, pred[k])
    ok += min(math.hypot(x - a, y - b) for a, b in v['pts']) <= 0.1 * v['scale']

out = {'exp_id': exp, 'correct': ok, 'total': len(L),
       'pck': round(ok / len(L), 5), 'sha256': sha[:16]}

new = not os.path.exists(LEDGER)
with open(LEDGER, 'a') as f:
    if new:
        f.write('time\texp_id\tcorrect\ttotal\tsha16\tnote\n')
    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{exp}\t{ok}\t{len(L)}\t{sha[:16]}\t{note}\n")

print(json.dumps(out))
