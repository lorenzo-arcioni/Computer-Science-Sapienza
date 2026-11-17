#!/usr/bin/env python3
"""Label-free format check for a localtest577 prediction file.

Does not read the answer key. Validates only the submission contract:
  - header is exactly id,prediction
  - one row per id in data/sample_submission.csv, same order, no duplicates
  - every prediction parses as a finite 2-element JSON list

Usage: python3 verify_submission.py <submission.csv> [sample_submission.csv]
"""
import csv, json, math, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
cand = Path(sys.argv[1])
sample = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "data" / "sample_submission.csv"

with cand.open(newline="", encoding="utf-8-sig") as f:
    r = csv.DictReader(f)
    fields = r.fieldnames
    rows = list(r)

with sample.open(newline="", encoding="utf-8-sig") as f:
    sr = csv.DictReader(f)
    sample_ids = [x.get("id", "") for x in sr]

ids = [x.get("id", "") for x in rows]
bad = []
neg = []
spaced = 0
for i, row in enumerate(rows):
    try:
        pt = json.loads(row.get("prediction", ""))
    except Exception:
        bad.append((i, ids[i], "unparseable json"))
        continue
    if not isinstance(pt, list) or len(pt) != 2:
        bad.append((i, ids[i], "not a 2-element list"))
        continue
    try:
        vals = [float(v) for v in pt]
    except Exception:
        bad.append((i, ids[i], "non-numeric"))
        continue
    if not all(math.isfinite(v) for v in vals):
        bad.append((i, ids[i], "non-finite"))
        continue
    if any(v < 0 for v in vals):
        neg.append(ids[i])
    if ", " in row["prediction"]:
        spaced += 1

problems = []
if fields != ["id", "prediction"]:
    problems.append(f"header is {fields}, expected ['id', 'prediction']")
if len(rows) != len(sample_ids):
    problems.append(f"{len(rows)} rows, expected {len(sample_ids)}")
if len(set(ids)) != len(ids):
    problems.append(f"{len(ids) - len(set(ids))} duplicate ids")
if ids != sample_ids:
    mismatch = sum(1 for a, b in zip(ids, sample_ids) if a != b)
    problems.append(f"id order differs from sample_submission.csv ({mismatch} positions)")
if bad:
    problems.append(f"{len(bad)} invalid predictions, first {bad[:3]}")
if neg:
    problems.append(f"{len(neg)} negative coordinates, first {neg[:3]}")

print(json.dumps({
    "file": str(cand),
    "rows": len(rows),
    "unique_ids": len(set(ids)),
    "id_order_matches_sample": ids == sample_ids,
    "invalid_predictions": len(bad),
    "negative_coordinates": len(neg),
    "json_with_space_after_comma": spaced,
    "ok": not problems,
    "problems": problems,
}, indent=2))

sys.exit(0 if not problems else 1)
