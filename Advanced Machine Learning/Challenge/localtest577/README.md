# localtest577

Self-contained scorer for the AML2026 577-query local test. Drop the directory
anywhere and score a frozen prediction file offline — no sudo, no socoscore
account, no network.

```
unzip localtest577.zip && cd localtest577
./score.sh predictions/gated_vitup.csv check_vitup
```

Expected: `{"exp_id": "check_vitup", "correct": 428, "total": 577, "pck": 0.74177, ...}`

## Contents

| Path | What it is |
|---|---|
| `score.sh` | wrapper: `./score.sh <submission.csv> <exp_id> [note]` |
| `score.py` | scorer; same logic as the host copy, paths resolved from its own directory |
| `labels/test_labels.json` | **answer key** — accepted target points and target scale per query (mode 0600) |
| `ledger.tsv` | append-only history of every host-side score, including the two recheck receipts |
| `data/sample_submission.csv` | the 577 ids and their required order |
| `data/test.csv` | test manifest: ids, pair ids, image paths, source points. No labels. |
| `predictions/gated_vitup.csv` | channel rerank, gated: **428/577** |
| `predictions/s17_contrast_gated.csv` | incumbent s17: **417/577** |
| `predictions/gated_dv6.csv`, `gated_ap.csv`, `ungated_*.csv` | the other five variants (409, 378, 385, 384, 351) |
| `rerank/run_channel_rerank.py` | the rerank that produced the six variants |
| `rerank/b_test_top8.npz` | B top-8 peaks, the rerank input |
| `rerank/score.json`, `diagnostic.json`, `REPORT.md` | the B550 run's own numbers |
| `verify/verify_submission.py` | label-free format check: header, id order, 577 rows, finite points |
| `verify/receipts.json` | pinned hashes, canonical receipts, provenance and caveats |

## Metric

A query counts as correct when

```
min over accepted target points  hypot(px - ax, py - ay)  <=  0.1 * scale
```

Every query has equal weight; there is no pair or category averaging. `scale` is
the target object's longest bounding-box side.

## Verifying a file before scoring

```
python3 verify/verify_submission.py predictions/gated_vitup.csv
```

Checks the header, that the ids are exactly `sample_submission.csv` in order,
that there are 577 of them, and that every prediction is a finite `[x,y]`.
It never opens the answer key.

## Provenance and the two caveats that matter

The labels are **SOCO-recovered annotations** for this course release — a derived
proxy, not the official Kaggle private ground truth. Two consequences observed on
this data:

1. **Local scores over-state the leaderboard.** s17 reads 417/577 (0.7227) here
   and scored 0.68811 public; `gated_vitup` reads 428/577 (0.74177) here and
   scored 0.62376 public. The gap is far larger for the gated file because the
   gate rewrites exactly the 145 borderline queries where a proxy and the real
   metric diverge most.
2. **Selecting a winner on these 577 inflates it.** `+11` was the best of six
   variants on the same cohort. With the report's own paired-bootstrap
   half-width of 19.5 queries (sd ≈ 10), the expected maximum of six null
   variants is ≈ +12.6, and P(max ≥ 11) ≈ 0.58. Treat a single-digit win from a
   multi-variant sweep as no evidence of transfer.

A valid lower score is a negative result, not a failed run.

## Handling rules

The archive contains the answer key. Keep it off any path that training,
fitting, pseudo-labelling or hyperparameter search can read — per-query labels
must not reach a model. Aggregate scores from a frozen file are the only
intended use. The separate 497-query locked holdout is not in here and must stay
untouched.

To restrict the key on a shared machine:

```
chmod 600 labels/test_labels.json
```
