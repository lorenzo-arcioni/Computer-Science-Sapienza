# Channel rerank at fusion-B top-8 peaks

**Exceeds s17 (417):** gated_vitup.

Anchor s17: **417/577**. Ungated: argmax channel sample among 8 B peaks (always replace s17). Gated: replace only if B margin < 0.35 and pick > 0.04×L from s17.

## Scores
- **gated_ap:** 378/577 (Δ-39, fixed 18, broke 57)
- **gated_dv6:** 409/577 (Δ-8, fixed 27, broke 35)
- **gated_vitup:** 428/577 (Δ+11, fixed 45, broke 34) **beats s17**
- **ungated_ap:** 351/577 (Δ-66, fixed 28, broke 94)
- **ungated_dv6:** 385/577 (Δ-32, fixed 40, broke 72)
- **ungated_vitup:** 384/577 (Δ-33, fixed 54, broke 87)

## Oracle within top-8 (diagnostic)

- **dv6:** on s17 misses, argmax hits 40/160; margin<0.35 subset 32/103; ungated breaks 72/417 s17 hits
- **ap:** on s17 misses, argmax hits 28/160; margin<0.35 subset 20/103; ungated breaks 94/417 s17 hits
- **vitup:** on s17 misses, argmax hits 54/160; margin<0.35 subset 48/103; ungated breaks 87/417 s17 hits

## Takeaway
Tests whether dv6/ap/viT-Up similarity at B's shortlist beats s17 when B margin is tight.
CPU-only; labels in hit/boot/diagnostic only.
