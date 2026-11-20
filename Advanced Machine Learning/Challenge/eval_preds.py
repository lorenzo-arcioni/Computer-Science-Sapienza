#!/usr/bin/env python3
"""
Valuta un CSV di predizioni sul test di sub-SOCO (stessa logica di pck() del notebook,
con statistiche aggiuntive).

Uso:
    python eval_preds.py submission.csv
    python eval_preds.py submission.csv --root datasets --alpha 0.1 --top 15
    python eval_preds.py submission.csv --vs altra_submission.csv
    python eval_preds.py submission.csv --save-details dettagli.csv

Richiede (relativi a --root, default "datasets"):
    SOCO/Images, SOCO/PairAnnotations/{intra,cross}
    sub-SOCO/test.csv, sub-SOCO/test_pairs.json
    sub_to_soco_mapping.csv      <- generato dal notebook check_sub_soco
La ground truth viene ricostruita una volta sola e messa in cache (ROOT/gt_cache.pkl).
"""
import argparse, ast, json, pickle, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PAT = re.compile(r"^\d+-(.+)-(.+)\.json$")
cat_of = lambda name: name.rsplit("_", 1)[0]
skey = lambda name: f"{cat_of(name)}/{name}"                    # "aeroplane/aeroplane_1"
key_from_path = lambda p: f"{Path(p).parent.name}/{Path(p).stem}"


# ----------------------------------------------------------------------------------------
# 1) Ground truth (equivalente a G2 del notebook)
# ----------------------------------------------------------------------------------------
def build_gt(root: Path) -> pd.DataFrame:
    soco, sub = root / "SOCO", root / "sub-SOCO"

    # mappa sub-path relativo ("images/test/img_0000.jpeg") -> chiave SOCO ("cat/cat_N")
    mp = pd.read_csv(root / "sub_to_soco_mapping.csv")
    sub2soco = {}
    for r in mp.itertuples():
        parts = Path(r.sub_path).parts
        i = len(parts) - 1 - parts[::-1].index("images")
        sub2soco["/".join(parts[i:])] = key_from_path(r.soco_path)

    # annotazioni SOCO: (a, b) -> json
    def pairs_from(folder):
        out = {}
        for f in (soco / "PairAnnotations" / folder).rglob("*.json"):
            a, b = PAT.match(f.name).groups()
            out[(skey(a), skey(b))] = f
        return out

    cross, intra = pairs_from("cross"), pairs_from("intra")
    idx = {}
    for swapped in (True, False):      # direzione giusta sovrascrive l'invertita
        for d in (cross, intra):       # intra sovrascrive cross
            for (a, b), f in d.items():
                idx[(b, a) if swapped else (a, b)] = (f, swapped)

    t = pd.read_csv(sub / "test.csv")
    tp = json.load(open(sub / "test_pairs.json"))["pairs"]

    rows = []
    for p in tp:
        sa, sb = sub2soco[p["source_image"]], sub2soco[p["target_image"]]
        f, swapped = idx[(sa, sb)]
        J = json.load(open(f))
        s_key, t_key = ("trg", "src") if swapped else ("src", "trg")
        skps, tkps = J[f"{s_key}_keypoints"], J[f"{t_key}_keypoints"]
        spos = np.array([k["pos"] for k in skps])
        for i, (x, y) in enumerate(p["source_xy"]):
            j = int(np.hypot(*(spos - [x, y]).T).argmin())
            k = skps[j]
            cands = [c["pos"] for c in tkps if c["concept_id"] == k["concept_id"]]
            named = [c["pos"] for c in tkps if c["name"] == k["name"]]
            rows.append(dict(
                pair_id=p["pair_id"], i=i, kp=k["name"], concept=k.get("concept_name", ""),
                cat=sb.split("/")[0], cands=cands, named=named[0] if named else None,
                n_cand=len(cands), has_name=bool(named),
                t_w=p["target_size"][0], t_h=p["target_size"][1],
                bbox=J["src_bndbox" if swapped else "trg_bndbox"]))
    Q = pd.DataFrame(rows)
    G = t.merge(Q, left_on=["pair_id", "query_index"], right_on=["pair_id", "i"], how="left")
    assert G.cands.notna().all(), "query di test.csv senza ground truth"
    return G


def load_gt(root: Path, rebuild=False) -> pd.DataFrame:
    cache = root / "gt_cache.pkl"
    if cache.exists() and not rebuild:
        return pickle.load(open(cache, "rb"))
    print("[info] costruisco la ground truth (una tantum)...", file=sys.stderr)
    G = build_gt(root)
    pickle.dump(G, open(cache, "wb"))
    return G


# ----------------------------------------------------------------------------------------
# 2) Predizioni -> tabella per-query
# ----------------------------------------------------------------------------------------
def _parse(x):
    try:
        return json.loads(x)
    except Exception:
        return ast.literal_eval(x)


def per_query(csv_path, G, clip=False) -> pd.DataFrame:
    s = pd.read_csv(csv_path)
    assert list(s.columns) == ["id", "prediction"], f"colonne errate: {list(s.columns)}"
    assert s.id.is_unique and set(s.id) == set(G.id), "id mancanti/extra/duplicati"
    xy = np.array(s.prediction.map(_parse).tolist(), dtype=float)
    assert xy.shape == (len(s), 2) and np.isfinite(xy).all(), "predizioni non valide"
    s["px"], s["py"] = xy[:, 0], xy[:, 1]

    e = G.merge(s[["id", "px", "py"]], on="id", how="left").copy()
    bb = np.array(e.bbox.tolist(), dtype=float)
    if clip:
        bb[:, [0, 2]] = bb[:, [0, 2]].clip(0, e.t_w.to_numpy()[:, None])
        bb[:, [1, 3]] = bb[:, [1, 3]].clip(0, e.t_h.to_numpy()[:, None])
    e["L_bbox"] = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
    e["L_img"] = np.maximum(e.t_w, e.t_h).astype(float)
    e["bbox_frac"] = e.L_bbox / e.L_img

    err, nearest = [], []
    for cs, px, py in zip(e.cands, e.px, e.py):
        C = np.array(cs, dtype=float)
        d = np.hypot(C[:, 0] - px, C[:, 1] - py)
        err.append(d.min()); nearest.append(tuple(C[d.argmin()]))
    e["err_px"] = err
    e["nearest"] = nearest
    e["err_name_px"] = [np.hypot(n[0] - px, n[1] - py) if n is not None else np.nan
                        for n, px, py in zip(e.named, e.px, e.py)]
    e["nerr"] = e.err_px / e.L_bbox                      # errore normalizzato (metrica ufficiale)
    e["nerr_name"] = e.err_name_px / e.L_bbox
    e["nerr_img"] = e.err_px / e.L_img
    return e


# ----------------------------------------------------------------------------------------
# 3) Report
# ----------------------------------------------------------------------------------------
def boot_ci(ok, groups, n=2000, seed=0):
    """CI 95% bootstrap con ricampionamento a livello di coppia (le query di una coppia sono correlate)."""
    g = pd.DataFrame({"ok": ok.astype(float), "g": groups}).groupby("g").ok.agg(["sum", "count"])
    S, C = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(g), size=(n, len(g)))
    r = S[ix].sum(1) / C[ix].sum(1)
    return np.percentile(r, [2.5, 97.5])


def sec(title):
    print(f"\n{'=' * 8} {title} {'=' * (70 - len(title))}")


def report(e, path, alpha, alphas, top, clip):
    n = len(e)
    ok = (e.nerr <= alpha).to_numpy()

    print(f"File: {path}")
    print(f"Query: {n} | coppie: {e.pair_id.nunique()} | categorie: {e.cat.nunique()} "
          f"| bbox {'clippato' if clip else 'originale'}")
    lo, hi = boot_ci(ok, e.pair_id.to_numpy())
    print(f"\n>>> PCK@{alpha:.2f} = {ok.mean():.4f}  ({ok.sum()}/{n})   "
          f"IC95% (bootstrap per coppia): [{lo:.4f}, {hi:.4f}]")

    sec("PCK per soglia")
    m = e.has_name.to_numpy()
    rows = []
    for a in alphas:
        rows.append({
            "alpha": a,
            "bbox | min-cand": (e.nerr <= a).mean(),
            "n ok": int((e.nerr <= a).sum()),
            f"bbox | solo-nome (n={m.sum()})": (e.nerr_name[m] <= a).mean(),
            "img | min-cand": (e.nerr_img <= a).mean(),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    sec("Errore")
    q = [50, 75, 90, 95, 99]
    print(f"px   : medio {e.err_px.mean():.1f} | mediano {e.err_px.median():.1f} | max {e.err_px.max():.1f}")
    print(f"norm : medio {e.nerr.mean():.3f} | mediano {e.nerr.median():.3f} | "
          + " | ".join(f"p{k} {np.percentile(e.nerr, k):.3f}" for k in q))
    for lbl, thr in (("<=alpha", (e.nerr <= alpha)), ("alpha..2*alpha", (e.nerr > alpha) & (e.nerr <= 2 * alpha)),
                     (">2*alpha", e.nerr > 2 * alpha)):
        print(f"  query con err {lbl:>14}: {int(thr.sum()):4d} ({thr.mean():.1%})")

    def grp(by, name, order=None):
        g = e.assign(ok=ok).groupby(by, observed=True).agg(
            n=("ok", "size"), PCK=("ok", "mean"), err_px=("err_px", "mean"), nerr_med=("nerr", "median"))
        if order is not None:
            g = g.loc[order]
        print(f"\n-- per {name}")
        print(g.to_string(float_format=lambda x: f"{x:.3f}"))

    sec(f"Breakdown (PCK@{alpha:.2f})")
    grp(pd.cut(e.n_cand, [0, 1, 2, 99], labels=["1 candidato", "2 candidati", "3+ candidati"]),
        "n. candidati target per il concetto (ambiguità)")
    grp(e.has_name.map({True: "con omonimo", False: "senza omonimo"}), "presenza keypoint omonimo")
    grp(pd.cut(e.bbox_frac, [0, .2, .4, .7, 1.01], labels=["<20%", "20-40%", "40-70%", ">70%"]),
        "dimensione bbox / immagine target")

    sec(f"Categorie peggiori (top {top})")
    c = e.assign(ok=ok).groupby("cat").agg(n=("ok", "size"), PCK=("ok", "mean"),
                                           err_px=("err_px", "mean"), nerr_med=("nerr", "median"))
    print(c.sort_values(["PCK", "n"], ascending=[True, False]).head(top).to_string(float_format=lambda x: f"{x:.3f}"))

    sec(f"Keypoint (concetti) peggiori, n>=3 (top {top})")
    k = e.assign(ok=ok).groupby("concept").agg(n=("ok", "size"), PCK=("ok", "mean"), nerr_med=("nerr", "median"))
    print(k[k.n >= 3].sort_values(["PCK", "n"], ascending=[True, False]).head(top).to_string(float_format=lambda x: f"{x:.3f}"))

    sec("Coppie")
    pp = e.assign(ok=ok).groupby("pair_id").agg(n=("ok", "size"), PCK=("ok", "mean"), cat=("cat", "first"))
    print(f"coppie con PCK=0: {(pp.PCK == 0).sum()}/{len(pp)} | con PCK=1: {(pp.PCK == 1).sum()}/{len(pp)}")
    print(pp.sort_values(["PCK", "n"], ascending=[True, False]).head(min(top, 8)).to_string(float_format=lambda x: f"{x:.2f}"))

    sec(f"Peggiori {top} query")
    w = e.sort_values("nerr", ascending=False).head(top)
    out = pd.DataFrame({
        "id": w.id, "pair_id": w.pair_id, "cat": w.cat, "kp": w.kp, "n_cand": w.n_cand,
        "pred": [f"({x:.0f},{y:.0f})" for x, y in zip(w.px, w.py)],
        "gt_vicino": [f"({x:.0f},{y:.0f})" for x, y in w.nearest],
        "err_px": w.err_px.round(1), "nerr": w.nerr.round(3)})
    print(out.to_string(index=False))

    # query "quasi giuste": utile per capire quanto margine c'è
    near = ((e.nerr > alpha) & (e.nerr <= 1.5 * alpha)).sum()
    print(f"\nquery appena sbagliate (alpha < err <= 1.5*alpha): {near}")


def compare(e1, e2, alpha, name1, name2):
    sec(f"Confronto  A={name1}  vs  B={name2}")
    a, b = (e1.nerr <= alpha).to_numpy(), (e2.set_index("id").loc[e1.id].nerr <= alpha).to_numpy()
    print(f"PCK@{alpha:.2f}: A={a.mean():.4f} B={b.mean():.4f} | giuste A={a.sum()} B={b.sum()}")
    print(f"solo A: {(a & ~b).sum()} | solo B: {(~a & b).sum()} | entrambe: {(a & b).sum()} | nessuna: {(~a & ~b).sum()}")
    # test dei segni (McNemar esatto) sulle query discordanti
    from math import comb
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    nd = n01 + n10
    if nd:
        k = min(n01, n10)
        p = min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / 2 ** nd)
        print(f"McNemar esatto (query discordanti={nd}): p = {p:.3f}")
    pa = e1.set_index("id")[["px", "py"]]; pb = e2.set_index("id").loc[e1.id][["px", "py"]]
    d = np.hypot(pa.px - pb.px, pa.py - pb.py)
    print(f"distanza tra predizioni A e B (px): mediana {d.median():.2f} | media {d.mean():.2f} | max {d.max():.1f}")


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV di predizioni (colonne: id, prediction=[x,y])")
    ap.add_argument("--root", default="datasets", help="cartella con SOCO/, sub-SOCO/ e sub_to_soco_mapping.csv")
    ap.add_argument("--alpha", type=float, default=0.10, help="soglia principale (default 0.10)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--clip", action="store_true", help="ritaglia il bbox ai bordi dell'immagine")
    ap.add_argument("--top", type=int, default=10, help="n. righe nelle classifiche peggiori")
    ap.add_argument("--vs", help="seconda submission da confrontare (test dei segni sulle query)")
    ap.add_argument("--save-details", help="salva il dettaglio per-query in CSV")
    ap.add_argument("--rebuild-gt", action="store_true", help="ricostruisce la cache della ground truth")
    args = ap.parse_args()

    pd.set_option("display.width", 200, "display.max_columns", 50)
    G = load_gt(Path(args.root), args.rebuild_gt)
    e = per_query(args.csv, G, args.clip)
    report(e, args.csv, args.alpha, args.alphas, args.top, args.clip)
    if args.vs:
        compare(e, per_query(args.vs, G, args.clip), args.alpha, Path(args.csv).name, Path(args.vs).name)
    if args.save_details:
        e.drop(columns=["cands", "bbox", "named", "nearest"]).to_csv(args.save_details, index=False)
        print(f"\n[info] dettaglio per-query salvato in {args.save_details}")


if __name__ == "__main__":
    main()
