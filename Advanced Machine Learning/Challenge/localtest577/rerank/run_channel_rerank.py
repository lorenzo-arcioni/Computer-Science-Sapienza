#!/usr/bin/env python3
"""Top-8 B peaks reranked by alternate channel samples (CPU, pre-declared rules)."""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path("/mnt/Netac/Projects/AML2026Project")
NM = ROOT / "results/kaggle_improvement/nearmiss_20261114"
OUT = ROOT / "results/kaggle_improvement/s17_catchup_20261116/channel_rerank"
TOP8 = ROOT / "results/kaggle_improvement/test_choice_diagnosis_20261113/b_test_top8.npz"
S17_PATH = ROOT / "results/kaggle_improvement/nearmiss3_20261114/s17_contrast_gated.csv"
VUT = ROOT / "results/kaggle_improvement/vitup_fold_pilot_20261110/test_out_v1/out"

MARGIN_GATE = 0.35
DIST_GATE = 0.04
CHANNELS = ("dv6", "ap", "vitup")

sys.path.insert(0, str(NM))
from nm_core import boot, hit, load_maps, load_preds, o2g, setup, write_preds  # noqa: E402


def bilinear(m: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Sample map at fractional (col, row); OOB -> nearest in-range via clip."""
    gx = np.clip(np.asarray(gx, dtype=np.float64), 0.0, 219.0)
    gy = np.clip(np.asarray(gy, dtype=np.float64), 0.0, 219.0)
    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = np.minimum(x0 + 1, 219)
    y1 = np.minimum(y0 + 1, 219)
    fx = gx - x0
    fy = gy - y0
    return (
        m[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + m[y0, x1] * fx * (1.0 - fy)
        + m[y1, x0] * (1.0 - fx) * fy
        + m[y1, x1] * fx * fy
    )


def sample_peaks(channel: np.ndarray, peaks_xy: np.ndarray, orig, res) -> np.ndarray:
    """channel (220,220); peaks_xy (8,2) original pixels -> (8,) scores."""
    cols = np.empty(8, dtype=np.float64)
    rows = np.empty(8, dtype=np.float64)
    for j in range(8):
        c, r = o2g(peaks_xy[j], orig, res)
        cols[j] = c
        rows[j] = r
    return bilinear(channel, cols, rows)


def load_vitup_z(ids) -> np.ndarray:
    acc: dict[str, np.ndarray | None] = {q: None for q in ids}
    cnt: dict[str, int] = {q: 0 for q in ids}
    for k in range(4):
        pattern = str(VUT / f"fold{k}_test_896" / "test_pair_*.npz")
        for f in sorted(glob.glob(pattern)):
            with np.load(f) as z:
                for i, q in enumerate(z["ids"]):
                    q = str(q)
                    m = np.asarray(z["maps"][i, 0, 0, 0], dtype=np.float32)
                    if acc[q] is None:
                        acc[q] = m.copy()
                    else:
                        acc[q] += m
                    cnt[q] += 1
    vitup = np.empty((len(ids), 220, 220), dtype=np.float32)
    for i, q in enumerate(ids):
        if cnt[q] != 4:
            raise SystemExit(f"vitup fold coverage {q}: {cnt[q]} != 4")
        m = acc[q] / 4.0
        m = m - m.mean()
        m = m / max(float(m.std()), 1e-6)
        vitup[i] = m
    return vitup


def peak_pick(peaks_xy: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return peaks_xy[int(np.argmax(scores))]


def apply_gated(s17_pt, pick, margin, L: float) -> np.ndarray:
    if margin < MARGIN_GATE and float(np.linalg.norm(pick - s17_pt)) > DIST_GATE * L:
        return pick
    return s17_pt


def score_preds(name: str, preds: dict, ids, labels, pair_of, base_h) -> dict:
    H = {q: hit(preds[q], labels[q])[0] for q in ids}
    hits = int(sum(H.values()))
    delta, fixed, broke, lo, hi = boot(ids, base_h, H, pair_of)
    return {
        "csv": name,
        "hits": hits,
        "delta_vs_s17": delta,
        "fixed": fixed,
        "broke": broke,
        "ci95_pair_boot_delta": [lo, hi],
    }


def diagnostic_channel(
    ch: str,
    picks: dict,
    ids,
    labels,
    base_h,
    margin,
    L_arr,
) -> dict:
    misses = [q for q in ids if not base_h[q]]
    low_margin_misses = [q for q in misses if margin[q] < MARGIN_GATE]
    hits_on_miss = 0
    hits_low_miss = 0
    for q in misses:
        if hit(picks[q], labels[q])[0]:
            hits_on_miss += 1
    for q in low_margin_misses:
        if hit(picks[q], labels[q])[0]:
            hits_low_miss += 1
                        
    s17_hits_q = [q for q in ids if base_h[q]]
    ungated_breaks = 0
    ungated_name = f"ungated_{ch}"
    # picks[q] is ungated peak choice for all q
    for q in s17_hits_q:
        if not hit(picks[q], labels[q])[0]:
            ungated_breaks += 1
    return {
        "channel": ch,
        "n_s17_misses": len(misses),
        "argmax_in_top8_hits_on_s17_misses": hits_on_miss,
        "same_restricted_B_margin_lt_0.35": hits_low_miss,
        "n_s17_misses_B_margin_lt_0.35": len(low_margin_misses),
        "ungated_would_break_s17_hits": ungated_breaks,
        "ungated_csv": ungated_name,
    }


def report_md(s17_hits: int, scores: dict, diag: dict) -> str:
    beats = [k for k, v in scores.items() if v["hits"] > s17_hits]
    lead = ""
    if beats:
        lead = f"**Exceeds s17 ({s17_hits}):** {', '.join(beats)}.\n\n"
    lines = [
        "# Channel rerank at fusion-B top-8 peaks",
        "",
        lead + f"Anchor s17: **{s17_hits}/577**. "
        f"Ungated: argmax channel sample among 8 B peaks (always replace s17). "
        f"Gated: replace only if B margin < {MARGIN_GATE} and pick > {DIST_GATE}×L from s17.",
        "",
        "## Scores",
    ]
    for key in sorted(scores.keys()):
        s = scores[key]
        tag = " **beats s17**" if s["hits"] > s17_hits else ""
        lines.append(
            f"- **{key}:** {s['hits']}/577 (Δ{s['delta_vs_s17']:+d}, "
            f"fixed {s['fixed']}, broke {s['broke']}){tag}"
        )
    lines += ["", "## Oracle within top-8 (diagnostic)", ""]
    for ch in CHANNELS:
        d = diag[ch]
        lines.append(
            f"- **{ch}:** on s17 misses, argmax hits {d['argmax_in_top8_hits_on_s17_misses']}/"
            f"{d['n_s17_misses']}; "
            f"margin<0.35 subset {d['same_restricted_B_margin_lt_0.35']}/"
            f"{d['n_s17_misses_B_margin_lt_0.35']}; "
            f"ungated breaks {d['ungated_would_break_s17_hits']}/417 s17 hits"
        )
    lines += [
        "",
        "## Takeaway",
        "Tests whether dv6/ap/viT-Up similarity at B's shortlist beats s17 when B margin is tight.",
        "CPU-only; labels in hit/boot/diagnostic only.",
    ]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, _rows, ids, pair_of = setup()
    s17 = load_preds(S17_PATH)
    if set(s17) != set(ids):
        raise SystemExit("s17 id mismatch")

    base_h = {q: hit(s17[q], labels[q])[0] for q in ids}
    s17_hits = int(sum(base_h.values()))
    assert s17_hits == 417, s17_hits

    z = np.load(TOP8)
    assert list(z["ids"]) == ids
    assert bool((z["B_peak"][:, 0] >= z["B_peak"][:, 1]).all())
    B_xy = z["B_xy"]
    B_peak = z["B_peak"]
    orig = z["orig"]
    res = z["res"]
    margin = {q: float(B_peak[i, 0] - B_peak[i, 1]) for i, q in enumerate(ids)}
    L_arr = {q: float(max(orig[i, 0], orig[i, 1])) for i, q in enumerate(ids)}

    maps, _go, _gr = load_maps(ids)
    vitup = load_vitup_z(ids)
    channel_maps = {"dv6": maps["dv6"], "ap": maps["ap"], "vitup": vitup}

    ungated: dict[str, dict] = {ch: {} for ch in CHANNELS}
    gated: dict[str, dict] = {ch: {} for ch in CHANNELS}
    peak_picks: dict[str, dict] = {ch: {} for ch in CHANNELS}
    n_gated_switch = {ch: 0 for ch in CHANNELS}

    for i, q in enumerate(ids):
        o, r = orig[i], res[i]
        for ch in CHANNELS:
            sc = sample_peaks(channel_maps[ch][i], B_xy[i], o, r)
            pick = peak_pick(B_xy[i], sc)
            peak_picks[ch][q] = pick
            ungated[ch][q] = pick
            gpt = apply_gated(s17[q], pick, margin[q], L_arr[q])
            gated[ch][q] = gpt
            if not np.allclose(gpt, s17[q]):
                n_gated_switch[ch] += 1

    all_outputs: dict[str, dict] = {}
    for ch in CHANNELS:
        all_outputs[f"ungated_{ch}"] = ungated[ch]
        all_outputs[f"gated_{ch}"] = gated[ch]

    for name, preds in all_outputs.items():
        write_preds(OUT / f"{name}.csv", ids, preds)

    scores = {
        name: score_preds(name, preds, ids, labels, pair_of, base_h)
        for name, preds in all_outputs.items()
    }
    diag = {ch: diagnostic_channel(ch, peak_picks[ch], ids, labels, base_h, margin, L_arr) for ch in CHANNELS}

    payload = {
        "protocol": {
            "anchor": str(S17_PATH.relative_to(ROOT)),
            "s17_hits": s17_hits,
            "top8_npz": str(TOP8.relative_to(ROOT)),
            "B_margin_gate": MARGIN_GATE,
            "dist_gate_frac_L": DIST_GATE,
            "channels": list(CHANNELS),
        },
        "variants": scores,
        "n_gated_switches": n_gated_switch,
        "diagnostic_ref": "diagnostic.json",
    }
    (OUT / "score.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "diagnostic.json").write_text(json.dumps(diag, indent=2) + "\n")
    (OUT / "REPORT.md").write_text(report_md(s17_hits, scores, diag))

    hit_line = {k: scores[k]["hits"] for k in sorted(scores.keys())}
    above = {k: v for k, v in hit_line.items() if v > s17_hits}
    if above:
        print("EXCEEDS s17 (417):", json.dumps(above))
    print("hit_counts:", json.dumps(hit_line, indent=2))


if __name__ == "__main__":
    main()