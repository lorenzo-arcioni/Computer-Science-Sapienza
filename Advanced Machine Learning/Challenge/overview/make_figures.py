#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py - figure SVG della nota "Pipeline di semantic correspondence (AML Challenge)".

Uso
    python make_figures.py                  # scrive figs/F00_*.svg ... figs/F24_*.svg
    python make_figures.py --png preview/   # in piu' scrive anteprime PNG

Dipendenze: numpy, scipy, matplotlib. Nessuna installazione di LaTeX.

Formule
    Le formule sono scritte in sintassi LaTeX ($...$) e composte dal motore "mathtext" di matplotlib
    (font Computer Modern). Con svg.fonttype = "path" ogni glifo diventa un tracciato vettoriale:
    gli SVG si vedono identici in qualsiasi visualizzatore, senza font installati.

Origine dei dati
    - Figure di RISULTATI (F06, F10, F14, F15, F16 e il pannello destro di F12): numeri letti dai tre
      notebook (val-split, 695 query, 102 coppie).
    - Figure di MECCANISMO (F04, F07, F08, F09, F11, F12-sinistra): mappe e punti SINTETICI, costruiti
      per illustrare il funzionamento; non sono output reali del modello. Lo dichiara la didascalia.
    - Gli schemi (F00-F03, F05, F13, F17) non contengono dati.
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch
from scipy.ndimage import zoom as ndzoom

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figs", help="cartella di output degli SVG")
ap.add_argument("--png", default=None, help="se indicata, salva anche anteprime PNG")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
if args.png:
    os.makedirs(args.png, exist_ok=True)

plt.rcParams.update({
    "svg.fonttype": "path", "svg.hashsalt": "aml-note",
    "font.family": "DejaVu Sans", "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# palette (fill, stroke, text)
PAL = {
    "gray":   ("#F1EFE8", "#5F5E5A", "#2C2C2A"),
    "purple": ("#EEEDFE", "#534AB7", "#26215C"),
    "teal":   ("#E1F5EE", "#0F6E56", "#04342C"),
    "amber":  ("#FAEEDA", "#854F0B", "#412402"),
    "coral":  ("#FAECE7", "#993C1D", "#4A1B0C"),
    "blue":   ("#E6F1FB", "#185FA5", "#042C53"),
    "red":    ("#FCEBEB", "#A32D2D", "#501313"),
    "green":  ("#EAF3DE", "#3B6D11", "#173404"),
}
INK, MUTED = "#2C2C2A", "#5F5E5A"


# --------------------------------------------------------------------------- helpers
def save(fig, name):
    fig.savefig(os.path.join(args.out, name + ".svg"), format="svg", bbox_inches="tight", pad_inches=0.1)
    if args.png:
        fig.savefig(os.path.join(args.png, name + ".png"), dpi=110, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("ok", name)


def canvas(w, h, W=None, H=None):
    """Figura 'a coordinate di schermo': x verso destra, y verso il basso; W,H in unita' dati."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    W, H = W or w * 100, H or h * 100
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig._WH = (W, H)
    return fig, ax


def inset(fig, x, y, w, h):
    W, H = fig._WH
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def T(ax, x, y, s, fs=10, c=INK, ha="center", va="center", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, fontweight=weight, **kw)


def node(ax, x, y, w, h, title, sub=None, extra=None, c="gray", dashed=False, fs=10, fs2=8.8):
    f, s, t = PAL[c]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=f, ec=s, lw=1.2,
                                ls=("--" if dashed else "-")))
    n = 1 + (sub is not None) + (extra is not None)
    ys = {1: [0.5], 2: [0.33, 0.72], 3: [0.22, 0.52, 0.80]}[n]
    ax.text(x + w / 2, y + h * ys[0], title, ha="center", va="center", fontsize=fs, color=t, fontweight="bold")
    k = 1
    if sub is not None:
        ax.text(x + w / 2, y + h * ys[k], sub, ha="center", va="center", fontsize=fs2, color=s)
        k += 1
    if extra is not None:
        ax.text(x + w / 2, y + h * ys[k], extra, ha="center", va="center", fontsize=fs2 + 0.6, color=t)


def arrow(ax, p, q, c=MUTED, lw=1.4, ls="-", rad=0.0, ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def polyline_arrow(ax, pts, c=MUTED, lw=1.4, ls="-"):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs + (pts[-1][0],), ys + (pts[-1][1],), color=c, lw=lw, ls=ls, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], c=c, lw=lw, ls=ls)


def img(ax, x, y, w, h, obj=None, pts=None, pcol="#E24B4A", fc="#FBFAF7", ec=MUTED, lw=1.1, psize=4.8):
    """Immagine schematica: pts e obj in coordinate normalizzate [0,1] dentro il riquadro."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    if obj is not None:
        cx, cy, rx, ry = obj
        ax.add_patch(Ellipse((x + cx * w, y + cy * h), 2 * rx * w, 2 * ry * h, fc=PAL["purple"][0],
                             ec=PAL["purple"][1], lw=1))
    for p in (pts if pts is not None else []):
        ax.plot(x + p[0] * w, y + p[1] * h, "o", ms=psize, color=pcol, mec="white", mew=0.7, zorder=5)


def synth(h, w, peaks, noise=0.0, seed=0, bg=0.25):
    """Mappa di similarita' sintetica h x w: somma di gaussiane + rumore (solo illustrativa)."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    xx, yy = (xx + 0.5) / w, (yy + 0.5) / h
    M = bg * np.ones((h, w))
    for px, py, a, s in peaks:
        M += a * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * s * s))
    return M + noise * rng.randn(h, w)


def box_from(pts, W, H, margin):
    """Stessa logica di box_from() del notebook 3 (riquadro dei punti + margine)."""
    pts = np.asarray(pts, float)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = margin * max(x1 - x0, y1 - y0, 0.1 * max(W, H))
    return (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))


def rect(ax, b, ox=0, oy=0, ec="#854F0B", lw=1.6, ls="-", fc="none"):
    ax.add_patch(Rectangle((ox + b[0], oy + b[1]), b[2] - b[0], b[3] - b[1], fc=fc, ec=ec, lw=lw, ls=ls, zorder=4))


def ladder(ax, rows, xmax=0.75, fs=9.4):
    """Barre orizzontali con IC95%. rows: (etichetta, valore, (lo, hi) | None, chiave-colore | None)."""
    n = len(rows)
    y = np.arange(n)[::-1]
    for yy, (lab, v, ci, c) in zip(y, rows):
        ax.barh(yy, v, color=(PAL[c][1] if c else "#B4B2A9"), alpha=0.88, height=0.66)
        if ci is not None:
            ax.plot(ci, [yy, yy], color=INK, lw=1.3)
            for e in ci:
                ax.plot([e, e], [yy - 0.14, yy + 0.14], color=INK, lw=1.3)
        ax.text((ci[1] if ci is not None else v) + 0.006, yy, "%.3f" % v, va="center", fontsize=fs - 0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=fs)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="y", length=0)
    return y




# --------------------------------------------------------------------------- F00 panoramica
def f00():
    fig, ax = canvas(12, 6.9, 1200, 690)
    T(ax, 600, 26, "Pipeline completa (ricetta finale: DINOv3 ViT-L, notebook 5)", fs=13, weight="bold")
    T(ax, 28, 66, "A · Addestramento", ha="left", fs=11.5, weight="bold", c=PAL["purple"][1])
    w, g, x0 = 176, 18, 27
    A = [("A1 Dati e split", "componenti connesse", r"$\mathcal{I}_{trn}\cap\mathcal{I}_{val}=\emptyset$", "gray"),
         ("A2 Augmentation", "inversione, flip, ritagli", r"$x\mapsto W-x$", "purple"),
         ("A3 Backbone", "ViT-L: 20 congelati + 4", r"$f=\mathrm{norm}(u+r(u))$", "purple"),
         ("A4 Logit", "similarita' sul target", r"$S=\langle f_s,F_t\rangle/\tau$", "purple"),
         ("A5 Loss", "CE verso gaussiane", r"$\mathcal{L}=-\sum_{ij}y_{ij}\log p_{ij}$", "teal"),
         ("A6 Ottimizzazione", "AdamW, coseno, EMA", r"$\theta'\leftarrow\beta\theta'+(1-\beta)\theta$", "teal")]
    for i, (t, s, e, c) in enumerate(A):
        node(ax, x0 + i * (w + g), 82, w, 112, t, s, e, c)
        if i:
            arrow(ax, (x0 + i * (w + g) - g + 2, 138), (x0 + i * (w + g) - 2, 138))
    T(ax, 28, 272, "B · Inferenza", ha="left", fs=11.5, weight="bold", c=PAL["purple"][1])
    B = [("B1 Mappe multiscala", "2 scale, griglia canonica", r"$\bar S=\frac{1}{|V|}\sum_v S_v$", "purple", False),
         ("B2 Soft-argmax", "tau 0.03, finestra 7x7", r"$\hat y=\sum_c w_c\,c$", "purple", False),
         ("B3 Zoom, 1 giro", "margine 0.3 sul riquadro", r"$(x,y)=(x_c,y_c)+(x_0,y_0)$", "purple", False),
         ("B4 2° backbone", "PROPOSTO", r"$\bar S=\sum_m\alpha_m\bar S_m$", "amber", True),
         ("B5 Re-ranker", "PROPOSTO", r"$\hat c=\arg\max_c\,p_c$", "amber", True),
         ("B6 Output", "clip e JSON", r"$x\leftarrow\min(\max(x,0),W-1)$", "gray", False)]
    for i, (t, s, e, c, d) in enumerate(B):
        node(ax, x0 + i * (w + g), 288, w, 112, t, s, e, c, dashed=d)
        if i:
            arrow(ax, (x0 + i * (w + g) - g + 2, 344), (x0 + i * (w + g) - 2, 344))
    xA6 = x0 + 5 * (w + g) + w / 2
    polyline_arrow(ax, [(xA6, 194), (xA6, 240), (x0 + w / 2, 240), (x0 + w / 2, 286)], c=PAL["teal"][1])
    T(ax, 640, 230, "snapshot della testa (EMA) all'epoca migliore sul val", fs=9, c=PAL["teal"][1])
    T(ax, 28, 470, "C · Validazione e modello finale", ha="left", fs=11.5, weight="bold", c=PAL["purple"][1])
    C = [("C1 K-fold", "per componenti connesse", r"$\mathrm{PCK}_{OOF}$ per scegliere", "amber", True),
         ("C2 Snapshot validato", "early stopping, train-split", r"epoca $e^\ast=\arg\max$ val", "gray", False),
         ("C3 Submission", "CSV: id, [x, y]", "nessun riaddestramento", "gray", False)]
    for i, (t, s, e, c, d) in enumerate(C):
        node(ax, x0 + i * 260, 486, 240, 100, t, s, e, c, dashed=d)
        if i:
            arrow(ax, (x0 + i * 260 - 18, 536), (x0 + i * 260 - 2, 536))
    ax.add_patch(FancyBboxPatch((820, 486), 350, 100, boxstyle="round,pad=0,rounding_size=8", fc="white", ec=MUTED, lw=1))
    ax.add_patch(Rectangle((838, 504), 22, 14, fc=PAL["purple"][0], ec=PAL["purple"][1], lw=1.2))
    T(ax, 870, 511, "validato nei notebook", ha="left", fs=9.5)
    ax.add_patch(Rectangle((838, 530), 22, 14, fc=PAL["amber"][0], ec=PAL["amber"][1], lw=1.2, ls="--"))
    T(ax, 870, 537, "proposto, non ancora testato", ha="left", fs=9.5)
    ax.add_patch(Rectangle((838, 556), 22, 14, fc=PAL["teal"][0], ec=PAL["teal"][1], lw=1.2))
    T(ax, 870, 563, "supervisione e ottimizzazione", ha="left", fs=9.5)
    T(ax, 600, 640, "Scartati perche' non hanno aiutato: flip-TTA, ri-ranking geometrico e cycle, allineamento di posa, ensemble A+B, 2° e 3° giro di zoom (con B), LoRA, mix dei layer.",
      fs=9.4, c=MUTED)
    save(fig, "F00_panoramica")


# --------------------------------------------------------------------------- F01 split e unita'
def f01():
    fig, ax = canvas(11.5, 5.4, 1150, 540)
    T(ax, 30, 26, "A1 · Split per componenti connesse e unita' di training", ha="left", fs=12.5, weight="bold")
    comps = [("trn", [(90, 140), (170, 105), (165, 200), (255, 150), (245, 235)], [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]),
             ("trn", [(105, 360), (190, 325), (200, 410), (290, 370)], [(0, 1), (0, 2), (1, 3), (2, 3)]),
             ("val", [(450, 140), (530, 110), (525, 205)], [(0, 1), (1, 2), (0, 2)]),
             ("val", [(440, 350), (530, 320), (525, 410), (615, 365)], [(0, 1), (0, 2), (1, 3), (2, 3)])]
    for kind, nodes, edges in comps:
        c = "blue" if kind == "trn" else "amber"
        for a, b in edges:
            ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=PAL[c][1], lw=1.4, zorder=1)
        for (x, y) in nodes:
            ax.add_patch(Circle((x, y), 15, fc=PAL[c][0], ec=PAL[c][1], lw=1.4, zorder=2))
    ax.plot([355, 355], [70, 450], color=MUTED, lw=1.2, ls="--")
    T(ax, 180, 80, "train-split: 1800 query", fs=10, c=PAL["blue"][1], weight="bold")
    T(ax, 535, 80, "val-split: 695 query", fs=10, c=PAL["amber"][1], weight="bold")
    T(ax, 355, 466, r"$\mathcal{I}_{trn}\cap\mathcal{I}_{val}=\emptyset$  (nessuna immagine condivisa)", fs=10.5)
    T(ax, 330, 500, "nodo = immagine, arco = coppia (sorgente, target); si separano le componenti connesse, mai le singole coppie",
      fs=9, c=MUTED)
    T(ax, 30, 50, "517 immagini, 392 coppie, 142 componenti, 2495 query", ha="left", fs=9.5, c=MUTED)
    # unita'
    X = 720
    T(ax, X, 70, "Unita' di training = una coppia con tutte le sue query", ha="left", fs=10.5, weight="bold")
    img(ax, X, 100, 160, 120, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.38, 0.4), (0.6, 0.65), (0.52, 0.8)])
    img(ax, X + 230, 100, 160, 120, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.4, 0.38)], pcol="#0F6E56")
    ax.plot([X + 230 + 0.62 * 160, X + 230 + 0.62 * 160 - 0.0], [100 + 0.66 * 120] * 2, "o", ms=4.8, color="#0F6E56", mec="white", mew=.7, zorder=5)
    ax.plot([X + 230 + 0.74 * 160], [100 + 0.66 * 120], "o", ms=4.8, color="#0F6E56", mec="white", mew=.7, zorder=5)
    ax.plot([X + 230 + 0.55 * 160], [100 + 0.85 * 120], "o", ms=4.8, color="#0F6E56", mec="white", mew=.7, zorder=5)
    arrow(ax, (X + 165, 160), (X + 225, 160))
    T(ax, X + 80, 236, r"sorgente: $p_1,p_2,p_3$", fs=9.5)
    T(ax, X + 310, 236, r"target: $|A(q_1)|=1,\ |A(q_2)|=2,\ |A(q_3)|=1$", fs=9.5)
    T(ax, X + 40, 262, "(il punto 2 ha due target accettati, ad es. le due ruote)", ha="left", fs=9, c=MUTED)
    img(ax, X, 300, 160, 120, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.4, 0.38), (0.55, 0.85)], pcol="#0F6E56")
    img(ax, X + 230, 300, 160, 120, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.38, 0.4), (0.52, 0.8)])
    arrow(ax, (X + 165, 360), (X + 225, 360))
    T(ax, X + 195, 446, "unita' invertita: solo query con un solo target", fs=9.5)
    T(ax, X + 195, 470, r"nb3: $534$ intere $+\,1032$ ritagli; val $102$ (nb2: $290$ dirette $+\,244$ invertite)", fs=9, c=MUTED)
    save(fig, "F01_split_unita")


# --------------------------------------------------------------------------- F02 augmentation
def f02():
    fig, ax = canvas(12.4, 5.3, 1240, 530)
    T(ax, 30, 26, "A2 · Augmentation (tutte validate nel nb3)", ha="left", fs=12.5, weight="bold")
    # (a) inversione
    T(ax, 30, 62, "(a) coppia invertita", ha="left", fs=10.5, weight="bold", c=PAL["purple"][1])
    img(ax, 30, 92, 110, 90, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.4, 0.4)])
    img(ax, 190, 92, 110, 90, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.62, 0.45)], pcol="#0F6E56")
    arrow(ax, (144, 137), (186, 137))
    img(ax, 30, 252, 110, 90, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.62, 0.45)], pcol="#0F6E56")
    img(ax, 190, 252, 110, 90, obj=(0.5, 0.55, 0.3, 0.32), pts=[(0.4, 0.4)])
    arrow(ax, (144, 297), (186, 297))
    arrow(ax, (165, 188), (165, 246), c=PAL["purple"][1])
    T(ax, 150, 366, r"si usa solo se $|A(q)|=1$", fs=9.5)
    T(ax, 150, 388, "altrimenti il punto inverso e' ambiguo", fs=9, c=MUTED)
    T(ax, 150, 418, r"$p'=a,\quad A'(q')=\{p\}$", fs=10.5)
    # (b) flip
    T(ax, 350, 62, "(b) specchio, probabilita' 0.5", ha="left", fs=10.5, weight="bold", c=PAL["purple"][1])
    img(ax, 350, 92, 110, 90, obj=(0.3, 0.55, 0.22, 0.3), pts=[(0.25, 0.4)])
    img(ax, 510, 92, 110, 90, obj=(0.7, 0.55, 0.22, 0.3), pts=[(0.75, 0.4)])
    arrow(ax, (464, 137), (506, 137))
    T(ax, 485, 125, "flip", fs=8.8, c=MUTED)
    T(ax, 485, 214, "sorgente e target insieme", fs=9.5)
    T(ax, 485, 250, r"$(x,y)\mapsto(W-x,\ y)$", fs=11)
    T(ax, 485, 282, r"anche per tutti i target:  $x_a\mapsto W_t-x_a$", fs=9.5)
    T(ax, 485, 322, "il flip e' applicato anche in cache", fs=9, c=MUTED)
    T(ax, 485, 340, "(token specchiati precalcolati)", fs=9, c=MUTED)
    # (c) ritaglio
    T(ax, 690, 62, "(c) ritaglio centrato sull'oggetto", ha="left", fs=10.5, weight="bold", c=PAL["purple"][1])
    pts_s = np.array([[0.30, 0.34], [0.42, 0.30], [0.55, 0.38], [0.36, 0.55], [0.5, 0.62], [0.62, 0.52]])
    img(ax, 690, 92, 200, 160, obj=(0.46, 0.46, 0.2, 0.22), pts=pts_s)
    bb = box_from(pts_s * np.array([200, 160]), 200, 160, 0.3)
    rect(ax, bb, 690, 92)
    bj = (bb[0] + 7, bb[1] - 6, bb[2] + 7, bb[3] - 6)
    rect(ax, bj, 690, 92, ec=PAL["coral"][1], ls="--", lw=1.4)
    T(ax, 790, 266, "riquadro dei punti sorgente", fs=9.5)
    T(ax, 790, 284, "+ margine + jitter (tratteggio)", fs=9.5)
    img(ax, 930, 92, 200, 160, obj=(0.5, 0.5, 0.36, 0.38), pts=[(0.3, 0.3), (0.55, 0.25), (0.68, 0.45)], pcol="#0F6E56")
    rect(ax, (36, 28, 164, 140), 930, 92, ec=PAL["coral"][1], lw=1.6)
    T(ax, 1030, 266, "riquadro dei target corretti", fs=9.5)
    T(ax, 1030, 284, "+ margine + jitter", fs=9.5)
    T(ax, 910, 330, r"$\mathrm{pad}=m\cdot\max\!\left(\Delta x,\ \Delta y,\ 0.1\max(W,H)\right)$", fs=11)
    T(ax, 910, 362, r"$m_s\sim\mathcal{U}(0.15,0.5),\quad m_t\sim\mathcal{U}(0.15,0.6),\quad \mathrm{jitter}=\pm8\%$", fs=10)
    T(ax, 910, 392, r"si tengono solo query dentro il ritaglio; lato minimo $\geq 48$ px; 2 ritagli casuali per coppia", fs=9.3, c=MUTED)
    T(ax, 910, 414, "ritaglio rielaborato a CROP_SIZE px sul lato lungo (640 nel nb3, 512 nel nb5)", fs=9.3, c=MUTED)
    T(ax, 620, 480, "PROPOSTO: augmentation fotometrica (colore, luminosita', blur), jitter di scala e piu' varianti di ritaglio per coppia", fs=9.8,
      c=PAL["amber"][1])
    save(fig, "F02_augmentation")


# --------------------------------------------------------------------------- F03 architettura
def f03():
    fig, ax = canvas(12.4, 6.6, 1240, 660)
    T(ax, 30, 26, "A3 · Backbone DINOv3 ViT-L/16 spezzato in tronco congelato e testa addestrabile", ha="left", fs=12.5, weight="bold")
    w, h, g, x0 = 270, 110, 40, 30
    r1 = [("Immagine", "lato lungo 512 o 768 px", r"$I\in\mathbb{R}^{3\times H\times W}$", "gray"),
          ("Patch + registri", "patch 16x16, 4 registri, RoPE", r"$N=g_hg_w+1+4$ token", "gray"),
          ("Layer 1-20 (congelati)", "autocast: fp16 dove sicuro", r"$u^{(20)}\in\mathbb{R}^{N\times 1024}$", "gray"),
          ("Cache dei token", "anche versione specchiata", "2.1 MB per immagine o ritaglio", "gray")]
    for i, (t, s, e, c) in enumerate(r1):
        node(ax, x0 + i * (w + g), 70, w, h, t, s, e, c)
        if i:
            arrow(ax, (x0 + i * (w + g) - g + 3, 125), (x0 + i * (w + g) - 3, 125))
    r2 = [("Layer 21-24 (addestrabili)", "copia dei 4 ultimi blocchi, fp32", r"$u^{(24)}=\mathrm{Block}_{24}\circ\cdots\circ\mathrm{Block}_{21}(u^{(20)})$", "purple"),
          ("Norma e reshape", "scarta CLS e registri, L2", r"$F\in\mathbb{R}^{1024\times g_h\times g_w}$", "purple"),
          ("Upsampling x2", "bilineare", r"$u=\mathrm{up}_{\times2}(F)$", "purple"),
          ("Raffinamento residuo", "r zero-init: epoca 0 = DINOv3-L", r"$f=\mathrm{norm}\!\left(u+r(u)\right)$", "purple")]
    for i, (t, s, e, c) in enumerate(r2):
        node(ax, x0 + i * (w + g), 330, w, h, t, s, e, c, fs2=8.4)
        if i:
            arrow(ax, (x0 + i * (w + g) - g + 3, 385), (x0 + i * (w + g) - 3, 385))
    xr = x0 + 3 * (w + g) + w / 2
    polyline_arrow(ax, [(xr, 180), (xr, 255), (x0 + w / 2, 255), (x0 + w / 2, 328)], c=PAL["purple"][1])
    T(ax, 620, 243, "token in ingresso al layer 21", fs=9.5, c=PAL["purple"][1])
    T(ax, 1210, 468, r"$r:\ 1{\times}1(1024\to128)\to\mathrm{GELU}\to\mathrm{DW}\,3{\times}3\to\mathrm{GELU}\to1{\times}1(128\to1024)$", fs=9.6, ha="right")
    T(ax, 1210, 492, r"ultimo $1{\times}1$ a zero: $f=\mathrm{norm}(u)$ all'epoca 0 (coseno 1.0000 con DINOv3-L completo, verificato)", fs=9.3, c=MUTED, ha="right")
    T(ax, 620, 548, r"A lato lungo 512 px: $g=32\ \Rightarrow\ 64$ celle dopo l'upsampling.     A 768 px: $g=48\ \Rightarrow\ 96$ celle.", fs=10.5)
    T(ax, 620, 582, r"Addestrati: 4 blocchi (lr $4\cdot10^{-5}$, 50.7M parametri) + testa $r$ (lr $3\cdot10^{-4}$). Il tronco non riceve gradiente.", fs=10)
    T(ax, 620, 614, r"Con ViT-B/16 (nb2, nb3): 12 layer, $C=768$, tronco = primi 10 (nb2) o 8 (nb3) layer. Il resto e' identico.", fs=9.4, c=MUTED)
    save(fig, "F03_architettura")


# --------------------------------------------------------------------------- F04 descrittore e logit
def f04():
    fig, ax = canvas(12.6, 5.0, 1260, 500)
    T(ax, 30, 26, "A4 · Descrittore del punto e mappa di similarita'", ha="left", fs=12.5, weight="bold")
    T(ax, 30, 62, "(a) lettura bilineare di $f_s$ sulla sorgente", ha="left", fs=10.5, weight="bold", c=PAL["purple"][1])
    cs, x0, y0 = 56, 40, 100
    for i in range(7):
        for j in range(6):
            ax.add_patch(Rectangle((x0 + i * cs, y0 + j * cs), cs, cs, fc="#FBFAF7", ec="#B4B2A9", lw=0.8))
    u, v = 3.35, 2.70      # coordinate in unita' di cella-centro
    px, py = x0 + (u + 0.5) * cs, y0 + (v + 0.5) * cs
    a, b = u - 3, v - 2
    wts = {(3, 2): (1 - a) * (1 - b), (4, 2): a * (1 - b), (3, 3): (1 - a) * b, (4, 3): a * b}
    for (i, j), ww in wts.items():
        cx, cy = x0 + (i + 0.5) * cs, y0 + (j + 0.5) * cs
        ax.add_patch(Circle((cx, cy), 5, fc=PAL["purple"][1], ec="white", lw=0.8, zorder=4))
        ax.plot([px, cx], [py, cy], color=PAL["purple"][1], lw=0.9, ls=":", zorder=3)
        T(ax, cx + (14 if i == 4 else -14), cy + (-14 if j == 2 else 14), f"{ww:.2f}", fs=9, c=PAL["purple"][1], ha=("left" if i == 4 else "right"))
    ax.plot(px, py, "o", ms=8, color="#E24B4A", mec="white", mew=1, zorder=6)
    T(ax, 215, 452, r"$u=\frac{x}{W}\,w-\frac{1}{2},\quad f_s=\mathrm{norm}(\sum_{k=1}^{4}\omega_k\,F_s(i_k,j_k))$", fs=10.5)
    T(ax, 215, 490, r"pesi bilineari: $\omega=(1-a)(1-b),\ a(1-b),\ (1-a)b,\ ab$", fs=9, c=MUTED)
    T(ax, 480, 62, "(b) mappa di similarita' sul target", ha="left", fs=10.5, weight="bold", c=PAL["purple"][1])
    M = synth(48, 64, [(0.46, 0.52, 0.62, 0.05), (0.78, 0.30, 0.25, 0.06)], noise=0.025, seed=3)
    a2 = inset(fig, 480, 100, 380, 285)
    a2.imshow(M, cmap="YlOrRd", interpolation="nearest")
    i0, j0 = np.unravel_index(np.argmax(M), M.shape)
    a2.plot(j0, i0, "x", color="#042C53", ms=11, mew=2.4)
    T(ax, 670, 405, r"$S(i,j)=\langle f_s,F_t(i,j)\rangle/\tau$,   $\tau=0.05$", fs=10.8)
    T(ax, 670, 432, "cella piu' calda = predizione per argmax (croce)", fs=9, c=MUTED)
    T(ax, 670, 454, "esempio sintetico, griglia 64x48", fs=9, c=MUTED)
    T(ax, 900, 62, "(c) distribuzione sul target", ha="left", fs=10.5, weight="bold", c=PAL["purple"][1])
    T(ax, 900, 120, r"$p_{ij}=\frac{\exp\!\left(S_{ij}\right)}{\sum_{i'j'}\exp\!\left(S_{i'j'}\right)}$", ha="left", fs=12)
    T(ax, 900, 185, "Il training confronta $p$ con le etichette $y$", ha="left", fs=9.6)
    T(ax, 900, 208, "(gaussiane sui target accettati, passo A5).", ha="left", fs=9.6)
    T(ax, 900, 262, "In inferenza si usa direttamente $S$:", ha="left", fs=9.6)
    T(ax, 900, 285, "serve il picco, non una probabilita'.", ha="left", fs=9.6)
    T(ax, 900, 340, r"$f_s$ e $F_t$ sono normalizzati, quindi", ha="left", fs=9.6)
    T(ax, 900, 363, r"$\langle f_s,F_t\rangle=\cos\theta\in[-1,1]$", ha="left", fs=10.4)
    save(fig, "F04_descrittore_logit")


# --------------------------------------------------------------------------- F05 etichette gaussiane
def sigma_sched(E, smax=3.0, smin=0.75):
    e = np.arange(1, E + 1)
    return e, smin + 0.5 * (smax - smin) * (1 + np.cos(np.pi * (e - 1) / max(E - 1, 1)))


def f05():
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    fig.suptitle("A5 · Etichette gaussiane multi-target e annealing di $\\sigma$", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    ax = axs[0]
    x = np.linspace(-9, 9, 400)
    for s, col in ((3.0, "#7F77DD"), (1.875, "#D85A30"), (0.75, "#0F6E56")):
        ax.plot(x, np.exp(-x ** 2 / (2 * s ** 2)), color=col, lw=2, label=r"$\sigma=%g$" % s)
    ax.set_xlabel("distanza dal target (celle della griglia fine)")
    ax.set_ylabel("peso $y$ (prima di normalizzare)")
    ax.set_title(r"un target: $\mathcal{G}(r;\sigma)=e^{-r^2/(2\sigma^2)}$")
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    h, w = 30, 44
    yy, xx = np.mgrid[0:h, 0:w]
    L = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2)) for cx, cy in ((12, 18), (31, 17)))
    ax.imshow(L / L.sum(), cmap="YlOrRd", interpolation="nearest")
    ax.plot([12, 31], [18, 17], "*", color="#042C53", ms=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"due target: $y\propto\sum_{a\in A(q)}\mathcal{G}(\cdot-a;\sigma)$")
    ax.set_xlabel("bimodale: scegliere una ruota o l'altra non e' un errore", fontsize=9)
    ax = axs[2]
    for E, smin, col, lab, mk in ((24, 0.75, "#9C9A92", r"piano nb3: $E=24$, $\sigma_{min}=0.75$", None),
                                  (15, 1.0, "#378ADD", r"nb2: $E=15$, $\sigma_{min}=1$", None),
                                  (12, 1.25, "#D85A30", r"nb5: $E=12$, $\sigma_{min}=1.25$", None)):
        e, s = sigma_sched(E, 3.0, smin)
        ax.plot((e - 1) / (E - 1), s, color=col, lw=2, label=lab)
    e3, s3 = sigma_sched(3)
    ax.plot((e3 - 1) / 2, s3, "o-", color="#534AB7", lw=2, ms=8, label=r"eseguito nb3: $E=3$")
    for xx_, ss in zip((e3 - 1) / 2, s3):
        ax.annotate("%.2f" % ss, (xx_, ss), textcoords="offset points", xytext=(6, 6), fontsize=9, color="#534AB7")
    ax.set_xlabel("avanzamento $(e-1)/(E-1)$")
    ax.set_ylabel(r"$\sigma$ (celle fini)")
    ax.set_title(r"$\sigma(e)=\sigma_{min}+\frac{1}{2}(\sigma_{max}-\sigma_{min})\left(1+\cos\frac{\pi(e-1)}{E-1}\right)$", fontsize=9.6)
    ax.legend(frameon=False, fontsize=8.4)
    fig.text(0.01, -0.04, r"Loss: $\mathcal{L}=-\frac{1}{|Q|}\sum_{q}\sum_{ij}y^{(q)}_{ij}\,\log\,\mathrm{softmax}(S^{(q)}/\tau)_{ij}$, con $y$ normalizzata a somma 1 e $\tau=0.05$.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "F05_etichette_gaussiane")


# --------------------------------------------------------------------------- F06 dinamica di training
def f06():
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.7))
    fig.suptitle("A6 · Ottimizzazione e curve di addestramento (dati letti dai notebook)", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    nb2_val = [0.224, 0.243, 0.299, 0.335, 0.367, 0.377, 0.388, 0.396, 0.399, 0.401, 0.406, 0.410, 0.413, 0.414, 0.416, 0.416]
    nb3_val = [0.226, 0.378, 0.453, 0.473]
    F_val = [0.283, 0.440, 0.452, 0.489, 0.506, 0.506, 0.514, 0.512, 0.512, 0.511, 0.506]
    M_val = [0.283, 0.439, 0.455, 0.483, 0.505, 0.518, 0.515, 0.527, 0.517, 0.505, 0.511, 0.514]
    L_val = [0.283, 0.427, 0.445, 0.432, 0.435, 0.446, 0.452, 0.469, 0.479, 0.475, 0.471, 0.463, 0.463]
    ser = [("nb2: B, 2 blocchi, 15 ep.", nb2_val, "#378ADD", 14), ("nb3 (B): B, 4 blocchi, 3 ep.", nb3_val, "#534AB7", 3),
           ("nb5 F: L, pieno", F_val, "#D85A30", 6), ("nb5 M: L, + mix layer", M_val, "#BA7517", 7), ("nb5 L: L, LoRA", L_val, "#0F6E56", 8)]
    ax = axs[0]
    for lab, v, col, b in ser:
        ax.plot(range(len(v)), v, "o-", ms=3.5, lw=1.6, color=col, label=lab)
        ax.plot(b, v[b], "*", ms=13, color=col, mec="white", mew=0.8, zorder=5)
    ax.axhline(0.245, color="#9C9A92", ls=":", lw=1.2)
    ax.text(0.2, 0.252, "zero-shot B 0.245", fontsize=8.5, color="#5F5E5A")
    ax.set_xlabel("epoca $e$ (stella = snapshot scelto)")
    ax.set_ylabel("PCK@0.10 su val (argmax, EMA, intera 512)")
    ax.set_title("accuratezza di validazione")
    hh, ll = ax.get_legend_handles_labels()
    ax = axs[1]
    nb2_loss = [7.149, 6.538, 6.147, 5.910, 5.716, 5.527, 5.337, 5.149, 4.952, 4.754, 4.577, 4.425, 4.312, 4.245, 4.223]
    Fl = [6.685, 5.726, 5.453, 5.271, 5.087, 4.880, 4.652, 4.413, 4.178, 3.978]
    Ml = [6.678, 5.706, 5.436, 5.259, 5.077, 4.871, 4.645, 4.405, 4.171, 3.970, 3.833]
    Ll = [6.750, 5.867, 5.618, 5.432, 5.238, 5.028, 4.802, 4.566, 4.342, 4.156, 4.031, 3.982]
    for (lab, _, col, _b), l in zip(ser, (nb2_loss, [6.706, 5.146, 3.913], Fl, Ml, Ll)):
        ax.plot(range(1, len(l) + 1), l, "o-", ms=3.5, lw=1.6, color=col)
    ax.set_xlabel("epoca $e$")
    ax.set_ylabel(r"loss $\mathcal{L}$ media")
    ax.set_title(r"la loss scende anche perche' $\sigma$ si stringe")
    ax.legend(hh, ll, frameon=True, framealpha=0.9, fontsize=7.8, loc="upper right")
    ax = axs[2]
    spe = 196
    for E, col, lab in ((3, "#534AB7", "nb3: 3 epoche"), (12, "#D85A30", "nb5: 12 epoche (si ferma prima)")):
        total, warm = E * spe, spe
        s = np.arange(total + 1)
        lr = np.minimum(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * np.minimum(s, total) / total))
        ax.plot(s / spe, lr, color=col, lw=2, label=lab)
    ax.axvline(1, color="#9C9A92", ls=":", lw=1)
    ax.text(1.05, 0.05, "fine warm-up", fontsize=8.5, color="#5F5E5A")
    ax.set_xlabel("epoca (196 passi per epoca, batch 8)")
    ax.set_ylabel(r"$\eta(s)/\eta_0$")
    ax.set_title(r"$\eta(s)=\eta_0\min\!\left(1,\frac{s+1}{s_w}\right)\frac{1}{2}\left(1+\cos\frac{\pi s}{S}\right)$", fontsize=9.6)
    ax.legend(frameon=False, fontsize=8.4)
    fig.text(0.01, -0.07, r"AdamW (weight decay 0.05), clipping del gradiente a norma 1, flip casuale 50%.  EMA: $\theta'\leftarrow\beta\theta'+(1-\beta)\theta$, $\beta=0.97$; si valuta e si salva $\theta'$.", fontsize=9.8)
    fig.text(0.01, -0.115, "Tempi per epoca: 304 s (nb3, B), 331 s (nb5 F), 333 s (M), 282 s (LoRA). nb5: lr blocchi 4e-5, testa 3e-4, LoRA 2e-4; early stopping con pazienza 4 e minimo 5 epoche.", fontsize=9.4, color="#5F5E5A")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "F06_ottimizzazione")


# --------------------------------------------------------------------------- F07 multiscala e griglia canonica
def f07():
    fig, ax = canvas(12.6, 5.5, 1260, 550)
    T(ax, 30, 26, "B1 · Mappe di similarita' a due scale su una griglia canonica comune", ha="left", fs=12.5, weight="bold")
    M512 = synth(48, 64, [(0.46, 0.52, 0.50, 0.045), (0.74, 0.38, 0.56, 0.05)], noise=0.035, seed=1)
    M768 = synth(72, 96, [(0.46, 0.52, 0.62, 0.04), (0.74, 0.38, 0.50, 0.045)], noise=0.025, seed=2)
    C512 = ndzoom(M512, (96 / 48, 128 / 64), order=3)
    C768 = ndzoom(M768, (96 / 72, 128 / 96), order=3)
    Mavg = 0.5 * (C512 + C768)
    vmin, vmax = 0.15, 1.05
    specs = [(40, M512, r"$S_{512}$: griglia $64\times48$", "nearest"),
             (450, M768, r"$S_{768}$: griglia $96\times72$", "nearest"),
             (860, Mavg, r"$\bar S$: canonica $128\times96$", "bilinear")]
    for x, M, ttl, interp in specs:
        a = inset(fig, x, 110, 360, 270)
        a.imshow(M, cmap="YlOrRd", vmin=vmin, vmax=vmax, interpolation=interp, aspect="auto")
        i0, j0 = np.unravel_index(np.argmax(M), M.shape)
        a.plot(j0, i0, "x", color="#042C53", ms=11, mew=2.4)
        px = 0.46 * M.shape[1] - 0.5
        py = 0.52 * M.shape[0] - 0.5
        a.plot(px, py, "*", color="#0F6E56", ms=15, mec="white", mew=1)
        T(ax, x + 180, 92, ttl, fs=10.5)
    T(ax, 425, 245, "+", fs=22, c=MUTED)
    T(ax, 835, 245, "=", fs=22, c=MUTED)
    T(ax, 630, 420, r"$\bar S=\frac{1}{|V|}\sum_{e\in E}\sum_{s\in\{512,768\}}\sum_{v}\ \mathrm{Resample}_{H_c\times W_c}\!\left(S_{e,s,v}\right)$", fs=12)
    T(ax, 630, 456, r"griglia canonica: $H_c=\mathrm{round}\!\left(H_t\cdot\frac{128}{\max(W_t,H_t)}\right)$ e $W_c=\mathrm{round}\!\left(W_t\cdot\frac{128}{\max(W_t,H_t)}\right)$", fs=10.5)
    T(ax, 630, 492, "croce blu = argmax della mappa; stella verde = posizione corretta. Nella scala 512 il picco sbagliato vince, nella media vince quello giusto.", fs=9.4, c=MUTED)
    T(ax, 630, 514, "Mappe sintetiche a scopo illustrativo. Nei notebook: +2.3 punti da multiscala (argmax, ricetta B: 0.473 -> 0.496).", fs=9.4, c=MUTED)
    save(fig, "F07_multiscala_canonica")


# --------------------------------------------------------------------------- F08 soft-argmax
def soft_decode(S, tau, rad):
    i0, j0 = np.unravel_index(np.argmax(S), S.shape)
    i1, i2, j1, j2 = max(0, i0 - rad), min(S.shape[0], i0 + rad + 1), max(0, j0 - rad), min(S.shape[1], j0 + rad + 1)
    w = np.exp((S[i1:i2, j1:j2] - S[i0, j0]) / tau)
    w /= w.sum()
    ii, jj = np.meshgrid(np.arange(i1, i2), np.arange(j1, j2), indexing="ij")
    return (w * jj).sum(), (w * ii).sum(), w, (i0, j0)


def f08():
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    fig.suptitle("B2 · Decoding soft-argmax in una finestra attorno al picco", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    n = 15
    yy, xx = np.mgrid[0:n, 0:n]
    tx, ty = 7.35, 6.80
    S = 0.55 + 0.40 * np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2 * 2.0 ** 2))
    sx, sy, w, (i0, j0) = soft_decode(S, 0.03, 3)
    ax = axs[0]
    ax.imshow(S, cmap="YlOrRd", extent=(-0.5, n - 0.5, n - 0.5, -0.5), interpolation="nearest")
    ax.add_patch(Rectangle((j0 - 3.5, i0 - 3.5), 7, 7, fc="none", ec="#042C53", lw=2))
    ax.plot(j0, i0, "s", color="white", mec="#042C53", ms=8, label="argmax (centro cella)")
    ax.plot(sx, sy, "o", color="#534AB7", mec="white", ms=9, label=r"soft-argmax $\hat y$")
    ax.plot(tx, ty, "*", color="#0F6E56", mec="white", ms=15, label="posizione vera")
    ax.set_xlim(2.5, 12.5)
    ax.set_ylim(11.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=True, fontsize=7.4, loc="lower right", framealpha=0.9)
    ax.set_title(r"finestra $W$ di raggio $r=3$ ($7\times7$ celle)")
    ax.set_xlabel(f"errore argmax {np.hypot(j0 - tx, i0 - ty):.2f} celle, soft {np.hypot(sx - tx, sy - ty):.2f} celle", fontsize=9.3)
    ax = axs[1]
    cols = np.arange(j0 - 3, j0 + 4)
    wd = 0.26
    for k, (tau, col) in enumerate(((0.01, "#0F6E56"), (0.03, "#534AB7"), (0.10, "#D85A30"))):
        _, _, ww, _ = soft_decode(S, tau, 3)
        ax.bar(cols + (k - 1) * wd, ww.sum(0), width=wd, color=col, label=fr"$\tau={tau:g}$")
    ax.set_xlabel("colonna $x$ della griglia")
    ax.set_ylabel(r"peso marginale $\sum_y w_{(y,x)}$")
    ax.set_title(r"$w_c=\frac{e^{(S_c-S_{max})/\tau}}{\sum_{c'\in W}e^{(S_{c'}-S_{max})/\tau}}$", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[2]
    taus = np.logspace(-3, 0, 60)
    err = []
    for tau in taus:
        a_, b_, _, _ = soft_decode(S, tau, 3)
        err.append(np.hypot(a_ - tx, b_ - ty))
    ax.semilogx(taus, err, color="#534AB7", lw=2)
    ax.axhline(np.hypot(j0 - tx, i0 - ty), color="#9C9A92", ls="--", lw=1.2)
    ax.text(taus[0] * 1.1, np.hypot(j0 - tx, i0 - ty) + 0.012, "argmax", fontsize=9, color="#5F5E5A")
    ax.axvline(0.03, color="#0F6E56", ls=":", lw=1.4)
    ax.text(0.034, min(err) + 0.01, r"$\tau=0.03$", fontsize=9, color="#0F6E56")
    ax.set_xlabel(r"temperatura $\tau$")
    ax.set_ylabel("errore di posizione (celle)")
    ax.set_title(r"$\tau\to0$: argmax;  $\tau\to\infty$: centro della finestra", fontsize=10)
    fig.text(0.01, -0.04, r"Poi $\hat y=\left(\sum_c w_c\,j_c,\ \sum_c w_c\,i_c\right)$ e la conversione in pixel: $x=\frac{j+0.5}{W_c}W_t,\ y=\frac{i+0.5}{H_c}H_t$. Mappa sintetica a scopo illustrativo; nel nb3 il guadagno misurato e' +1.6 punti.",
             fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "F08_soft_argmax")


# --------------------------------------------------------------------------- F09 geometria dello zoom
def f09():
    fig, ax = canvas(12.8, 5.9, 1280, 590)
    T(ax, 30, 26, "B3 · Zoom: come si costruiscono i riquadri e come si riportano le coordinate", ha="left", fs=12.5, weight="bold")
    Wp, Hp = 330, 250
    rng = np.random.RandomState(4)
    base = np.array([[0.28, 0.30], [0.45, 0.24], [0.63, 0.32], [0.30, 0.52], [0.47, 0.50], [0.66, 0.55], [0.34, 0.74], [0.52, 0.72], [0.70, 0.76]])
    # (a) sorgente
    xa, ya = 30, 90
    T(ax, xa, 62, "(a) sorgente: tutti i punti della coppia", ha="left", fs=10.3, weight="bold", c=PAL["purple"][1])
    img(ax, xa, ya, Wp, Hp, obj=(0.48, 0.52, 0.28, 0.32), pts=base)
    ps = base * np.array([Wp, Hp])
    bt = (ps[:, 0].min(), ps[:, 1].min(), ps[:, 0].max(), ps[:, 1].max())
    rect(ax, bt, xa, ya, ec="#888780", ls="--", lw=1.2)
    rect(ax, box_from(ps, Wp, Hp, 0.3), xa, ya)
    T(ax, xa + Wp / 2, ya + Hp + 22, "tratteggio: riquadro dei punti; pieno: con margine 0.3", fs=9, c=MUTED)
    # (b) target
    xb, yb = 440, 90
    T(ax, xb, 62, "(b) target: predizioni dello stadio 1", ha="left", fs=10.3, weight="bold", c=PAL["purple"][1])
    img(ax, xb, yb, Wp, Hp, obj=(0.55, 0.5, 0.3, 0.34), pts=None)
    pred = base * 0.92 + np.array([0.09, 0.05]) + rng.randn(9, 2) * 0.018
    pred = np.vstack([pred, [[0.10, 0.88], [0.93, 0.14]]])
    pp = pred * np.array([Wp, Hp])
    for k, (x_, y_) in enumerate(pp):
        out = k >= 9
        ax.plot(xb + x_, yb + y_, "o", ms=5.2, color=("#A32D2D" if out else "#E24B4A"), mec="white", mew=0.7, zorder=5)
    lo, hi = np.percentile(pp, [10, 90], axis=0)
    core = (lo[0], lo[1], hi[0], hi[1])
    rect(ax, core, xb, yb, ec="#0F6E56", ls="--", lw=1.3)
    rect(ax, box_from(np.array([[core[0], core[1]], [core[2], core[3]]]), Wp, Hp, 0.3), xb, yb)
    T(ax, xb + Wp / 2, yb + Hp + 22, "percentili 10-90 (tratteggio verde): ignorano i 2 punti isolati", fs=9, c=MUTED)
    T(ax, xb + 72, yb + 228, "outlier", fs=8.8, c="#A32D2D")
    # (c) giri
    xc, yc = 850, 90
    T(ax, xc, 62, "(c) riquadri dei giri sul target", ha="left", fs=10.3, weight="bold", c=PAL["purple"][1])
    img(ax, xc, yc, 400, Hp, obj=(0.55, 0.5, 0.3, 0.34), pts=None)
    b1 = (100, 30, 330, 222)
    b2 = (125, 55, 300, 200)
    rect(ax, b1, xc, yc, ec="#854F0B", lw=1.8)
    rect(ax, b2, xc, yc, ec="#993C1D", lw=1.8)
    T(ax, xc + 315, yc + 40, "giro 1", fs=9.5, c="#854F0B", ha="right")
    T(ax, xc + 292, yc + 70, "giro 2", fs=9.5, c="#993C1D", ha="right")
    T(ax, xc + 200, yc + Hp + 22, "margini: 0.3 al giro 1, 0.18 al giro 2 (solo nb3)", fs=9, c=MUTED)
    T(ax, 640, 392, r"$\mathrm{pad}=m\cdot\max(\Delta x,\Delta y,\ 0.1\max(W,H))$,   riquadro $=\left[x_{min}-\mathrm{pad},\ x_{max}+\mathrm{pad}\right]\times\left[y_{min}-\mathrm{pad},\ y_{max}+\mathrm{pad}\right]$", fs=10.4)
    T(ax, 640, 428, r"si salta lo zoom se $\max\!\left(\frac{A_{s}}{W_sH_s},\frac{A_{t}}{W_tH_t}\right)>0.9$ oppure il lato minimo $<32$ px, oppure la coppia ha meno di 3 punti", fs=10)
    T(ax, 640, 468, r"ritaglio come coppia nuova a $640$ e $768$ px:   $p_{src}'=p_{src}-(x_0^s,y_0^s)$,   poi si riporta:   $(x,y)=(x_c,y_c)+(x_0^t,y_0^t)$", fs=10.4)
    T(ax, 640, 508, r"nb5: 1 giro. nb3 (B): giro 2 con le predizioni del giro 1 e margine $m_2=0.18$; con 3 giri il risultato peggiora (0.518 contro 0.532)", fs=10)
    T(ax, 640, 548, "Punti e immagini sintetici a scopo illustrativo; i riquadri sono calcolati con la stessa box_from() del notebook.", fs=9.2, c=MUTED)
    save(fig, "F09_zoom_geometria")


# --------------------------------------------------------------------------- F10 effetto per scala
def f10():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.suptitle("B3 · PCK@0.10 per gruppo di query: ViT-B (nb3) contro ViT-L (nb5), val 695 query", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    names = ["L zero-shot", "L fine-tuned, argmax 512 (F0)", "L finale: multiscala + zoom (F2)", "B finale (nb3, riferimento riportato nel nb5)"]
    cols = ["#B4B2A9", "#AFA9EC", "#0F6E56", "#EF9F27"]
    g1 = [("piccolo ($\\rho<0.33$)\n$n=263$", (0.369, 0.490, 0.665, 0.620)), ("medio ($\\rho<0.68$)\n$n=201$", (0.229, 0.448, 0.453, 0.443)),
          ("grande\n$n=231$", (0.203, 0.623, 0.667, 0.519))]
    g2 = [("1 target\n$n=426$", (0.239, 0.448, 0.552, 0.486)), ("$>1$ target\n$n=269$", (0.327, 0.639, 0.688, 0.613))]
    for ax, groups, ttl in ((axs[0], g1, r"scala dell'oggetto $\rho=L/\max(W_t,H_t)$"), (axs[1], g2, "numero di target accettati")):
        x = np.arange(len(groups))
        wd = 0.2
        for k in range(4):
            vals = [g[1][k] for g in groups]
            bars = ax.bar(x + (k - 1.5) * wd, vals, width=wd, color=cols[k], label=names[k], edgecolor="white")
            for b_, v in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, "%.2f" % v, ha="center", fontsize=7.6)
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=9)
        ax.set_ylim(0, 0.8)
        ax.set_title(ttl)
        ax.set_ylabel("PCK@0.10")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.075, "Il ViT-L guadagna soprattutto sugli oggetti grandi (+14.8) e sulle query a target singolo (+6.6). Gli oggetti di scala media restano fermi (+1.0).",
             fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save(fig, "F10_effetto_per_gruppo")


# --------------------------------------------------------------------------- F11 secondo backbone
def f11():
    fig, ax = canvas(12.4, 5.3, 1240, 530)
    T(ax, 30, 26, "B4 · Ensemble con un secondo backbone (PROPOSTO, non testato)", ha="left", fs=12.5, weight="bold", c=PAL["amber"][1])
    M1 = synth(96, 128, [(0.46, 0.52, 0.45, 0.04), (0.74, 0.38, 0.55, 0.045)], noise=0.02, seed=5)
    M2 = synth(96, 128, [(0.46, 0.52, 0.60, 0.04), (0.22, 0.70, 0.52, 0.045)], noise=0.02, seed=6)
    Mavg = 0.5 * M1 + 0.5 * M2
    for x, M, ttl in ((40, M1, r"modello 1: $\bar S_1$"), (450, M2, r"modello 2: $\bar S_2$"), (860, Mavg, r"ensemble: $\bar S=\sum_m\alpha_m\bar S_m$")):
        a = inset(fig, x, 100, 340, 255)
        a.imshow(M, cmap="YlOrRd", vmin=0.15, vmax=0.95, aspect="auto", interpolation="bilinear")
        i0, j0 = np.unravel_index(np.argmax(M), M.shape)
        a.plot(j0, i0, "x", color="#042C53", ms=11, mew=2.4)
        a.plot(0.46 * 128 - 0.5, 0.52 * 96 - 0.5, "*", color="#0F6E56", ms=15, mec="white", mew=1)
        T(ax, x + 170, 80, ttl, fs=10.5)
    T(ax, 395, 230, "+", fs=22, c=MUTED)
    T(ax, 805, 230, "=", fs=22, c=MUTED)
    T(ax, 620, 392, r"stessa griglia canonica $\Rightarrow$ si media senza altro lavoro;  $\alpha_m\geq0,\ \sum_m\alpha_m=1$ scelti su validazione", fs=10.6)
    T(ax, 620, 428, "Funziona solo se i modelli sbagliano in punti diversi (qui: distrattori diversi). Mappe sintetiche.", fs=9.5, c=MUTED)
    T(ax, 620, 468, "Evidenza nei notebook: l'ensemble A+B (stesso backbone, stessi dati) ha dato 0.521 contro 0.532 del singolo B3.", fs=9.8, c=PAL["coral"][1])
    T(ax, 620, 494, "Candidati: DINOv3-L, DINOv2-L, feature di Stable Diffusion; ciascuno con la propria testa fine-tuned.", fs=9.8, c=MUTED)
    save(fig, "F11_ensemble_backbone")


# --------------------------------------------------------------------------- F12 re-ranker
def top_peaks(S, K=5, nms=6):
    S2 = S.copy()
    out = []
    for _ in range(K):
        if not np.isfinite(S2).any():
            break
        i, j = np.unravel_index(np.nanargmax(np.where(np.isfinite(S2), S2, -np.inf)), S2.shape)
        out.append((i, j, S[i, j]))
        S2[max(0, i - nms):i + nms + 1, max(0, j - nms):j + nms + 1] = -np.inf
    return out


def f12():
    fig = plt.figure(figsize=(14.0, 5.2))
    fig.suptitle("B5 · Re-ranker appreso sui picchi candidati (PROPOSTO; il pannello destro e' misurato)", fontsize=12.5, fontweight="bold", x=0.01, ha="left", color=PAL["amber"][1])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.05], wspace=0.22)
    ax = fig.add_subplot(gs[0])
    peaks = [(0.30, 0.62, 0.52, 0.035), (0.46, 0.38, 0.47, 0.035), (0.70, 0.55, 0.44, 0.035), (0.58, 0.80, 0.34, 0.03), (0.15, 0.25, 0.30, 0.03)]
    M = synth(96, 128, peaks, noise=0.012, seed=8)
    ax.imshow(M, cmap="YlOrRd", vmin=0.15, vmax=0.9, interpolation="bilinear")
    for k, (i, j, s) in enumerate(top_peaks(M, 5, 6), 1):
        ax.add_patch(Rectangle((j - 6.5, i - 6.5), 13, 13, fc="none", ec="#042C53", lw=1.1, ls="--"))
        ax.text(j, i, str(k), ha="center", va="center", fontsize=10, fontweight="bold", color="#042C53")
    ax.plot(0.46 * 128, 0.38 * 96, "*", color="#0F6E56", ms=16, mec="white", mew=1, zorder=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"top-$K$ picchi con NMS, $K=5$, raggio 6 celle", fontsize=10)
    ax.set_xlabel("la posizione vera (stella) e' il picco 2, non il 1", fontsize=9)
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(r"feature $\phi_c$ e probabilita' $p_c=\sigma(\mathbf{w}^{\top}\phi_c+b)$", fontsize=10)
    hdr = ["c", r"$S_c$", "rango", r"$\Delta S$", "accordo\nscale", r"$d_{prev}$", r"$d_{cyc}$", r"$p_c$"]
    xs = [0.04, 0.17, 0.30, 0.43, 0.57, 0.71, 0.83, 0.95]
    for x, h_ in zip(xs, hdr):
        ax.text(x, 0.09, h_, ha="center", va="center", fontsize=8.8, fontweight="bold")
    rows = [("1", "0.52", "1", "0.00", "no", "0.31", "0.40", "0.18"), ("2", "0.47", "2", "-0.05", "si", "0.04", "0.02", "0.71"),
            ("3", "0.44", "3", "-0.08", "si", "0.12", "0.09", "0.07"), ("4", "0.34", "4", "-0.18", "no", "0.44", "0.52", "0.02"),
            ("5", "0.30", "5", "-0.22", "no", "0.51", "0.47", "0.01")]
    for r_, row in enumerate(rows):
        y = 0.22 + r_ * 0.11
        if row[0] == "2":
            ax.add_patch(Rectangle((0.0, y - 0.05), 1.0, 0.10, fc="#E1F5EE", ec="none"))
        for x, v in zip(xs, row):
            ax.text(x, y, v, ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.80, r"etichetta: $z_c=\mathbb{1}[\min_{a\in A(q)}\|c-a\|\leq0.10\,L]$ (solo train)", ha="center", fontsize=9.4)
    ax.text(0.5, 0.88, r"scelta: $\hat c=\arg\max_c\,p_c$,   $d_{prev}$: distanza dalla predizione del giro precedente", ha="center", fontsize=9)
    ax.text(0.5, 0.96, "valori di esempio illustrativi", ha="center", fontsize=8.5, color="#5F5E5A")
    ax = fig.add_subplot(gs[2])
    K = np.arange(4)
    vB = [0.512, 0.645, 0.686, 0.753]
    vL = [0.548, 0.660, 0.712, 0.781]
    b1 = ax.bar(K - 0.18, vB, width=0.36, color="#EF9F27", label="ViT-B (nb3, riferimento)")
    b2 = ax.bar(K + 0.18, vL, width=0.36, color="#0F6E56", label="ViT-L (nb5)")
    for bs, vv in ((b1, vB), (b2, vL)):
        for b_, v in zip(bs, vv):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, "%.3f" % v, ha="center", fontsize=7.8)
    ax.set_xticks(K)
    ax.set_xticklabels(["1", "3", "5", "10"])
    ax.set_ylim(0, 0.95)
    ax.set_xlabel(r"$K$ (candidati ammessi)")
    ax.set_ylabel("oracolo: almeno un picco corretto")
    ax.set_title("margine teorico (stadio 1, senza zoom)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    ax.text(1.5, 0.84, "top-1 -> top-5: +16.4 (L), +17.4 (B)", ha="center", fontsize=8.8, color="#A32D2D")
    save(fig, "F12_reranker")


# --------------------------------------------------------------------------- F13 output
def f13():
    fig, ax = canvas(11.6, 3.6, 1160, 360)
    T(ax, 30, 26, "B6 · Output: clip ai bordi e formato di consegna", ha="left", fs=12.5, weight="bold")
    node(ax, 30, 70, 320, 120, "Predizione dello zoom", "coordinate float nel target", r"$(x,\,y)=(301.82,\ 348.02)$", "purple")
    node(ax, 420, 70, 320, 120, "Clip ai bordi", r"$W_t\times H_t$ del target", r"$x\leftarrow\min\!\left(\max(x,0),W_t-1\right)$", "gray")
    node(ax, 810, 70, 320, 120, "submission.csv", "una riga per query", "id, prediction", "gray")
    arrow(ax, (354, 130), (416, 130))
    arrow(ax, (744, 130), (806, 130))
    T(ax, 970, 215, 'test_q_00426,  "[301.8236..., 348.0152...]"', fs=9.6, c=MUTED)
    T(ax, 580, 262, "Controlli prima di consegnare: 577 righe, nessun id duplicato, ID identici a sample_submission.csv, ogni prediction e' una lista JSON di 2 numeri finiti.", fs=9.6)
    T(ax, 580, 292, "L'esempio e' una riga reale prodotta dal notebook 2.", fs=9, c=MUTED)
    save(fig, "F13_output")


# --------------------------------------------------------------------------- F14 scala dei risultati
def f14():
    rows = [("nb1  baseline: punto al centro", .047, (.022, .077), "gray"),
            ("nb1  NN argmax, 448 px", .197, (.164, .233), "gray"),
            ("nb1  + soft-argmax", .229, (.193, .263), "gray"),
            ("nb1  ultimi 4 layer mediati", .278, (.236, .321), "gray"),
            ("nb1  risoluzione 896 px", .296, (.252, .338), "gray"),
            ("nb1  adapter lineare + flip", .374, (.328, .419), "gray"),
            ("nb2  zero-shot, 512 px", .245, (.204, .284), "teal"),
            ("nb2  fine-tuning 2 blocchi, argmax", .410, (.367, .456), "teal"),
            ("nb2  + soft-argmax", .432, (.389, .476), "teal"),
            ("nb2  + flip e multiscala", .455, (.404, .501), "teal"),
            ("nb2  + zoom (finale nb2)", .488, (.440, .531), "teal"),
            ("nb3  fine-tuning 4 blocchi, argmax (A0)", .471, (.425, .519), "purple"),
            ("nb3  + multiscala e soft (B1)", .512, (.462, .563), "purple"),
            ("nb3  + zoom 2 giri + ritagli (B3)", .532, (.480, .583), "purple"),
            ("nb5  ViT-L zero-shot, 512 px", .273, (.234, .311), "coral"),
            ("nb5  ViT-L fine-tuning, argmax (F0)", .522, (.473, .571), "coral"),
            ("nb5  + multiscala e soft (F1)", .550, (.498, .601), "coral"),
            ("nb5  + zoom 1 giro (F2, finale)", .604, (.553, .656), "coral")]
    fig, ax = plt.subplots(figsize=(11.8, 8.6))
    y = ladder(ax, rows, xmax=0.80)
    for yy in (11.5, 6.5, 3.5):
        ax.axhline(yy, color="#D3D1C7", lw=0.8)
    ax.axvline(0.650, color="#BA7517", ls="--", lw=1.1)
    ax.text(0.647, 17.45, r"$\mathrm{PCK}@0.20$ nb3: 0.650", fontsize=8.4, color="#854F0B", va="top", ha="right")
    ax.axvline(0.750, color="#A32D2D", ls="--", lw=1.1)
    ax.text(0.752, 17.45, r"nb5: 0.750", fontsize=8.4, color="#A32D2D", va="top")
    ax.set_xlabel(r"$\mathrm{PCK}@0.10$ sul val-split (695 query)")
    ax.set_title("Dal baseline a 0.604: ogni riga e' un risultato misurato (barre nere: IC 95%, bootstrap per coppia)", fontsize=11, loc="left")
    fig.text(0.01, -0.005, r"$\mathrm{PCK}@\alpha=\frac{1}{N}\sum_q\mathbb{1}[\min_{a\in A(q)}\|\hat y_q-a\|_2\leq\alpha\,L_q]$.  Differenze di 1-3 punti sono dentro l'incertezza (circa $\pm5$). Linee tratteggiate: tetto di sola precisione.", fontsize=9.6)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "F14_scala_risultati")


# --------------------------------------------------------------------------- F15 budget d'errore
def f15():
    cats = [(r"corretto: $e\leq0.10$", "#639922"), (r"vicino: $0.10<e\leq0.20$", "#EF9F27"), (r"parte sbagliata: $0.20<e\leq0.50$", "#D85A30"), (r"altrove: $e>0.50$", "#A32D2D")]
    runs = [("nb3: ViT-B, B3 (2 giri)", np.array([370, 82, 141, 102])), ("nb5: ViT-L, F2 (1 giro)", np.array([420, 101, 112, 62]))]
    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    for yi, (lab, n) in zip((1, 0), runs):
        p = n / n.sum()
        left = 0
        for k in range(4):
            ax.barh(yi, p[k], left=left, color=cats[k][1], height=0.55, edgecolor="white")
            ax.text(left + p[k] / 2, yi, "%d\n%.1f%%" % (n[k], 100 * p[k]), ha="center", va="center", fontsize=9.4, color="white", fontweight="bold")
            left += p[k]
        sem = (n[2] + n[3]) / n.sum()
        ax.text(1.012, yi + 0.1, "errori semantici\n$e>0.20$: %.1f%%" % (100 * sem), fontsize=9.6, va="center", color="#A32D2D")
        ax.text(1.012, yi - 0.28, r"$\mathrm{PCK}@0.10=%.3f$,  @0.20 $=%.3f$" % (n[0] / n.sum(), (n[0] + n[1]) / n.sum()), fontsize=8.8, va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in runs[::-1]], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    handles = [Rectangle((0, 0), 1, 1, fc=c) for _, c in cats]
    fig.legend(handles, [t for t, _ in cats], loc="upper center", ncol=4, frameon=False, fontsize=9.2, bbox_to_anchor=(0.45, 1.0))
    ax.set_title(r"Budget d'errore sul val (695 query); $e=d/L$ e' la distanza normalizzata", fontsize=10.8, loc="left", pad=26)
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    save(fig, "F15_budget_errore")


# --------------------------------------------------------------------------- F16 matrice tecniche x notebook
def f16():
    rows = [("fine-tuning / adapter", (12.2, 16.5, 22.6, 24.9)),
            ("backbone B -> L (a parita' di ricetta)", (None, None, None, 6.9)),
            ("risoluzione e multiscala", (6.7, 3.5, 2.3, 2.8)),
            ("soft-argmax", (3.2, 2.2, 1.6, "incl")),
            ("zoom", (None, 2.6, 2.0, 5.4)),
            ("ritagli in training (con zoom)", (None, None, 2.3, None)),
            ("media di piu' layer (zero-shot)", (4.9, None, None, -2.0)),
            ("mix appreso dei layer", (None, None, None, 0.0)),
            ("LoRA invece di fine-tuning pieno", (None, None, None, -4.7)),
            ("allineamento di posa", (None, 0.7, None, None)),
            ("flip-TTA", (0.0, -1.2, -0.1, None)),
            ("ensemble", (0.4, None, -1.1, None)),
            ("ri-ranking geometrico / cycle", (0.7, -1.7, None, None)),
            ("terzo giro di zoom", (None, None, -1.4, None))]
    cols = ["nb1\nB congelato", "nb2\nB, 2 blocchi", "nb3\nB, 4 blocchi", "nb5\nL, 4 blocchi"]
    fig, ax = plt.subplots(figsize=(10.2, 7.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#E24B4A", "#FBFAF7", "#639922"])
    for i, (lab, vals) in enumerate(rows):
        for j, v in enumerate(vals):
            if v is None or v == "incl":
                ax.add_patch(Rectangle((j, i), 1, 1, fc="#F1EFE8", ec="white", lw=2))
                ax.text(j + .5, i + .5, "n/d" if v is None else "incl.*", ha="center", va="center", fontsize=9, color="#888780")
            else:
                z = max(-1, min(1, v / 6))
                ax.add_patch(Rectangle((j, i), 1, 1, fc=cmap((z + 1) / 2), ec="white", lw=2))
                ax.text(j + .5, i + .5, "%+.1f" % v, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlim(0, 4)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_xticklabels(cols, fontsize=9.6)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.045, r"punti di $\mathrm{PCK}@0.10$ (val). Verde: aiuta; rosso: peggiora; n/d: non testato. Il colore satura a $\pm6$.", fontsize=9.2)
    fig.text(0.01, 0.015, "* nb5 misura soft-argmax e multiscala insieme (+2.8). nb1 geometria: +0.7 al meglio, ma la selezione sul train ha scelto lambda=0.", fontsize=8.8, color="#5F5E5A")
    ax.set_title("Effetto di ciascuna tecnica nei quattro notebook (si ripete?)", fontsize=11.5, loc="left", pad=36)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "F16_matrice_tecniche")


# --------------------------------------------------------------------------- F17 K-fold e modello finale
def f17():
    fig, ax = canvas(12.2, 5.6, 1220, 560)
    T(ax, 30, 26, "C · Validazione K-fold per componenti e scelta del modello finale", ha="left", fs=12.5, weight="bold")
    T(ax, 30, 60, "(1) PROPOSTO: K = 5 fold per componenti connesse; ogni giro addestra su 4 fold e predice quello escluso", ha="left", fs=10.2, c=PAL["amber"][1], weight="bold")
    for r in range(5):
        for k in range(5):
            hold = (r == k)
            ax.add_patch(Rectangle((200 + k * 90, 84 + r * 36), 84, 28, fc=(PAL["amber"][0] if hold else PAL["purple"][0]),
                                   ec=(PAL["amber"][1] if hold else PAL["purple"][1]), lw=1.2))
            T(ax, 242 + k * 90, 98 + r * 36, ("predice" if hold else "train"), fs=8.8, c=(PAL["amber"][2] if hold else PAL["purple"][2]))
        T(ax, 120, 98 + r * 36, "giro %d" % (r + 1), fs=9.5)
    T(ax, 650, 100, r"$\mathrm{PCK}_{OOF}=\frac{1}{5}\sum_{k=1}^{5}\mathrm{PCK}(\mathrm{fold}\ k)$", ha="left", fs=11)
    T(ax, 650, 160, r"$\Rightarrow$ predizioni out-of-fold su tutto il train", ha="left", fs=10.5)
    T(ax, 650, 190, "servono per (a) scegliere gli iperparametri con meno rumore,", ha="left", fs=9.6)
    T(ax, 650, 212, "(b) fissare il numero di epoche e (c) addestrare il re-ranker", ha="left", fs=9.6)
    T(ax, 650, 234, "su predizioni di un modello che non ha visto l'immagine", ha="left", fs=9.6)
    T(ax, 30, 306, "(2) Modello finale: due opzioni", ha="left", fs=10.2, c=PAL["purple"][1], weight="bold")
    node(ax, 30, 330, 360, 104, "Opzione A (nb5, in uso)", "testa del train-split, snapshot migliore", "nessun riaddestramento", "purple")
    node(ax, 420, 330, 360, 104, "Opzione B (proposta)", "K-fold fissa le epoche, poi train + val", "da validare: nb3 andava in overfitting", "amber", dashed=True)
    node(ax, 810, 330, 380, 104, "Inferenza", "soft-argmax e zoom a 1 giro", r"$\bar S$ su 512 e 768 px", "gray")
    arrow(ax, (394, 382), (416, 382))
    arrow(ax, (784, 382), (806, 382))
    T(ax, 610, 478, "Nel nb3 il riaddestramento su train+val con schedule compressa compare interrotto e non e' stato validato; nel nb5 e' stato escluso.", fs=9.6, c=PAL["coral"][1])
    T(ax, 610, 504, "L'ensemble di 3 seed del piano nb3 non e' stato validato e nel nb5 non e' previsto.", fs=9.6, c=MUTED)
    save(fig, "F17_kfold_e_finale")


# --------------------------------------------------------------------------- f18
def f18():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1.1, 0.9]))
    fig.suptitle("Notebook 1 · Decodifica, risoluzione e layer (DINOv3-B congelato, val 695 query)", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    ax = axs[0]
    G = np.array([[0.212, 0.222, 0.229], [0.204, 0.210, 0.199], [0.201, 0.206, 0.183]])
    im = ax.imshow(G, cmap="RdYlGn", vmin=0.17, vmax=0.24)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%.3f" % G[i, j], ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["r = 3", "r = 6", "r = 12"])
    ax.set_yticks(range(3))
    ax.set_yticklabels([r"$\tau=0.01$", r"$\tau=0.03$", r"$\tau=0.1$"])
    ax.set_title("soft-argmax: griglia $(\\tau, r)$ a 448 px", fontsize=10.2)
    ax.set_xlabel("riferimento: argmax = 0.197;  la migliore = 0.229", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax = axs[1]
    rows = [("224", .158, (.124, .192), "gray"), ("336", .222, (.183, .260), "gray"), ("448", .229, (.193, .263), "gray"), ("672", .262, (.225, .300), "gray"), ("896", .296, (.252, .338), "gray")]
    x = np.arange(5)
    for xi, (lab, v, ci, c) in zip(x, rows):
        ax.bar(xi, v, color=PAL["gray"][1], alpha=0.88, width=0.62)
        ax.plot([xi, xi], ci, color=INK, lw=1.3)
        ax.text(xi, ci[1] + 0.008, "%.3f" % v, ha="center", fontsize=8.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlabel("lato lungo dell'immagine (px)")
    ax.set_ylabel("PCK@0.10")
    ax.set_ylim(0, 0.4)
    ax.set_title("risoluzione (decoding soft)", fontsize=10.2)
    ax = axs[2]
    rows = [("ultimo", .229, (.193, .263)), ("ultimi 2", .252, (.215, .288)), ("ultimi 4", .278, (.236, .321))]
    for xi, (lab, v, ci) in enumerate(rows):
        ax.bar(xi, v, color=PAL["gray"][1], alpha=0.88, width=0.62)
        ax.plot([xi, xi], ci, color=INK, lw=1.3)
        ax.text(xi, ci[1] + 0.008, "%.3f" % v, ha="center", fontsize=8.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 0.4)
    ax.set_title("layer mediati ($\\hat F=\\frac{1}{K}\\sum_k F^{(-k)}$)", fontsize=10.2)
    ax.set_xlabel("train intero (2495 query): ultimi 4 = 0.307", fontsize=9)
    fig.text(0.01, -0.04, "Il decoding soft aiuta solo con temperatura molto bassa e finestra larga; con $\\tau\\geq0.03$ e $r=12$ o con $\\tau=0.1$ peggiora rispetto all'argmax. La risoluzione e' monotona.", fontsize=9.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "F18_nb1_decodifica_risoluzione")


# --------------------------------------------------------------------------- f19
def f19():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.suptitle("Notebook 1 · Dai baseline all'adapter lineare (val 695 query)", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    ax = axs[0]
    rows = [("baseline: punto al centro", .047, (.022, .077), "gray"), ("baseline: posizione relativa", .022, (.010, .036), "gray"),
            ("baseline: media dei target", .016, (.003, .031), "gray"), ("NN argmax (B0)", .197, (.164, .233), None),
            ("soft-argmax (B1)", .229, (.193, .263), None), ("flip-TTA (E)", .229, (.195, .260), None),
            ("ensemble 448+672+copia (G)", .233, (.197, .266), None), ("ensemble + flip (G)", .240, (.205, .275), None),
            ("ri-ranking geometrico, migliore (I)", .240, (.204, .273), None), ("adapter lineare, soft (J)", .351, (.306, .397), "purple"),
            ("adapter + flip (J)", .374, (.328, .419), "purple"), ("ensemble + adapter + geom (J)", .331, (.288, .374), "purple")]
    ladder(ax, rows, xmax=0.5)
    ax.set_xlabel(r"$\mathrm{PCK}@0.10$")
    ax.set_title("riepilogo dei test (barre nere: IC 95%)", fontsize=10.5)
    ax = axs[1]
    lam = np.array([0, 0.25, 0.5, 1.0, 2.0])
    k3 = [0.233, 0.237, 0.237, 0.237, 0.237]
    k5 = [0.233, 0.240, 0.236, 0.235, 0.235]
    ax.fill_between([-0.1, 2.1], 0.197, 0.266, color="#E1F5EE", label="IC 95% di lambda = 0")
    ax.plot(lam, k3, "o-", color="#534AB7", label="$K=3$")
    ax.plot(lam, k5, "s--", color="#D85A30", label="$K=5$")
    ax.axhline(0.233, color="#9C9A92", ls=":", lw=1)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(0.19, 0.275)
    ax.set_xlabel(r"peso geometrico $\lambda$")
    ax.set_ylabel(r"$\mathrm{PCK}@0.10$")
    ax.set_title(r"ri-ranking geometrico: $\mathrm{score}(c)=S(c)-\lambda\,\frac{\|c-\hat p^{aff}\|}{\max(W_t,H_t)}$", fontsize=9.6)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.text(1.2, 0.255, "tutto dentro il rumore; la selezione\nsul train-split ha scelto $\\lambda=0$", ha="center", fontsize=9, color="#5F5E5A")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "F19_nb1_adapter_geometria")


# --------------------------------------------------------------------------- f20
def f20():
    fig = plt.figure(figsize=(14.2, 6.4))
    fig.suptitle("Notebook 2 · Fine-tuning di 2 blocchi e trucchi d'inferenza (val 695 query)", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.62, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    rows = [("Z0  zero-shot argmax", .245, (.204, .284), "gray"), ("Z1  zero-shot soft + flip + multiscala", .224, (.186, .262), "gray"),
            ("F0  fine-tuned argmax", .410, (.367, .456), "teal"), ("F1  + soft-argmax (migliore dei 9)", .432, (.389, .476), "teal"),
            ("F2  + flip-TTA", .420, (.376, .463), "teal"), ("F3  + multiscala 512/768", .455, (.404, .501), "teal"),
            ("F4  + allineamento di posa (margine 0.0)", .462, (.412, .507), "teal"), ("F5  + ri-ranking, migliore degli 8", .445, (.396, .488), "amber"),
            ("F6  + zoom (max_frac 0.7)", .488, (.440, .531), "teal")]
    ladder(ax, rows, xmax=0.62)
    ax.set_xlabel(r"$\mathrm{PCK}@0.10$")
    ax.set_title("ablazione cumulativa (il ri-ranking, in ambra, e' peggio)", fontsize=10.3)
    ax = fig.add_subplot(gs[0, 1])
    G = np.array([[0.427, 0.429, 0.429], [0.432, 0.426, 0.417], [0.416, 0.432, 0.423]])
    ax.imshow(G, cmap="RdYlGn", vmin=0.40, vmax=0.44)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%.3f" % G[i, j], ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["r = 3", "r = 6", "r = 10"], fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r"$\tau=0.01$", r"$\tau=0.03$", r"$\tau=0.1$"], fontsize=9)
    ax.set_title("F1: griglia soft-argmax (0.416-0.432)", fontsize=9.8)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax = fig.add_subplot(gs[1, 1])
    G = np.array([[0.462, 0.442, 0.436], [0.443, 0.445, 0.437], [0.420, 0.424, 0.423]])
    ax.imshow(G, cmap="RdYlGn", vmin=0.40, vmax=0.47)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%.3f" % G[i, j], ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([r"$\lambda_{cyc}=0$", r"$\lambda_{cyc}=0.5$", r"$\lambda_{cyc}=1$"], fontsize=8.6)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r"$\lambda=0$", r"$\lambda=0.5$", r"$\lambda=1$"], fontsize=9)
    ax.set_title("F5: ri-ranking (angolo = F4, nessun ri-ranking)", fontsize=9.8)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    save(fig, "F20_nb2_esperimenti")


# --------------------------------------------------------------------------- f21
def f21():
    rows = [("Z0  zero-shot argmax 512", .245, (.204, .284), "gray"),
            ("A0  A argmax 512", .471, (.425, .519), "purple"), ("B0  B argmax 512", .473, (.428, .519), "teal"),
            ("A1  A soft + multiscala", .499, (.452, .551), "purple"), ("B1x B argmax + multiscala", .496, (.452, .543), "teal"),
            ("B1  B soft + multiscala", .512, (.462, .563), "teal"),
            ("A2  A + zoom 2 giri", .509, (.466, .552), "purple"), ("B2  B + zoom 1 giro", .524, (.473, .573), "teal"),
            ("B3  B + zoom 2 giri (finale)", .532, (.480, .583), "teal"),
            ("C1  zoom 2 giri, crop (640,)", .527, (.475, .575), None), ("C1  zoom 2 giri, crop (768,)", .511, (.461, .556), None),
            ("C1  zoom 2 giri, crop (768, 1024)", .506, (.456, .556), None), ("C2  zoom 2 giri, max_frac 0.7", .521, (.467, .570), None),
            ("C3  zoom 3 giri", .518, (.465, .568), None), ("D1  B3 + flip-TTA", .531, (.484, .576), None),
            ("D2  ensemble A+B", .521, (.477, .565), None), ("D3  ensemble A+B + flip", .529, (.484, .575), None)]
    fig, ax = plt.subplots(figsize=(11.6, 7.6))
    ladder(ax, rows, xmax=0.68)
    ax.axhline(15.5, color="#D3D1C7", lw=0.8)
    ax.axhline(7.5, color="#D3D1C7", lw=0.8)
    ax.set_xlabel(r"$\mathrm{PCK}@0.10$")
    ax.set_title("Notebook 3 · A (immagini intere) contro B (con ritagli); poi zoom, ritagli, flip, ensemble (val 695 query)", fontsize=10.8, loc="left")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=PAL[c][1], alpha=0.88) for c in ("gray", "purple", "teal")] + [Rectangle((0, 0), 1, 1, fc="#B4B2A9")],
               labels=["zero-shot", "ricetta A", "ricetta B", "varianti sul metodo finale"], frameon=False, fontsize=8.8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.01, 0.005, "Tutte le differenze tra le ultime 12 righe (0.506-0.532) sono dentro l'IC di circa 5 punti.", fontsize=9.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "F21_nb3_esperimenti")


# --------------------------------------------------------------------------- f22
def f22():
    rows = [("Z0  L zero-shot argmax 512", .273, (.234, .311), "gray"), ("ZM  L zero-shot, media ultimi 4 layer", .253, (.214, .289), "gray"),
            ("Z1  L zero-shot argmax 768", .289, (.252, .325), "gray"),
            ("F0  F argmax 512", .522, (.473, .571), "coral"), ("F1  F soft + multiscala", .550, (.498, .601), "coral"), ("F2  F + zoom 1 giro (finale)", .604, (.553, .656), "coral"),
            ("M0  M argmax 512", .537, (.489, .588), "amber"), ("M1  M soft + multiscala", .548, (.494, .603), "amber"), ("M2  M + zoom 1 giro", .604, (.550, .655), "amber"),
            ("L0  LoRA argmax 512", .496, (.452, .538), "teal"), ("L1  LoRA soft + multiscala", .517, (.471, .558), "teal"), ("L2  LoRA + zoom 1 giro", .557, (.508, .607), "teal")]
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ladder(ax, rows, xmax=0.72)
    ax.axvline(0.535, color="#534AB7", ls="--", lw=1.2)
    ax.text(0.537, 11.55, "B (nb3), 1 giro: 0.535", fontsize=8.6, color="#534AB7", va="top")
    ax.set_xlabel(r"$\mathrm{PCK}@0.10$")
    ax.set_title("Notebook 5 · DINOv3-L: zero-shot e tre ricette (F pieno, M con mix dei layer, L con LoRA)", fontsize=10.8, loc="left")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=PAL[c][1], alpha=0.88) for c in ("gray", "coral", "amber", "teal")],
               labels=["zero-shot", "F: fine-tuning pieno", "M: + mix appreso", "L: LoRA r=16"], frameon=False, fontsize=8.8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "F22_nb5_esperimenti")


# --------------------------------------------------------------------------- f23
def f23():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    fig.suptitle("Il modello memorizza: PCK su train (sottoinsieme di 30 coppie) contro val", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    ax = axs[0]
    runs = [("nb3 (B)\n3 epoche", 0.891, 0.532, False), ("nb3 rieseguito\n24 epoche*", 0.934, 0.535, True), ("nb5 (L)\nricetta F", 0.973, 0.604, False)]
    for i, (lab, tr_, va, hat) in enumerate(runs):
        ax.bar(i - 0.18, tr_, width=0.36, color="#7F77DD", hatch=("//" if hat else None), edgecolor="white")
        ax.bar(i + 0.18, va, width=0.36, color="#0F6E56", hatch=("//" if hat else None), edgecolor="white")
        ax.text(i - 0.18, tr_ + 0.012, "%.3f" % tr_, ha="center", fontsize=8.8)
        ax.text(i + 0.18, va + 0.012, "%.3f" % va, ha="center", fontsize=8.8)
        ax.text(i, 1.06, "gap %.0f" % (100 * (tr_ - va)), ha="center", fontsize=9.4, color="#A32D2D")
    ax.set_xticks(range(3))
    ax.set_xticklabels([r[0] for r in runs], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(r"$\mathrm{PCK}@0.10$")
    ax.set_title("train (viola) e val (verde)", fontsize=10.4)
    ax = axs[1]
    thr = ["@0.05", "@0.10", "@0.20"]
    nb3 = ([0.732, 0.891, 0.940], [0.350, 0.532, 0.650])
    nb5 = ([0.913, 0.973, 0.978], [0.393, 0.604, 0.750])
    x = np.arange(3)
    wd = 0.2
    for k, (v, col, lab) in enumerate(((nb3[0], "#AFA9EC", "nb3 train"), (nb3[1], "#5DCAA5", "nb3 val"), (nb5[0], "#7F77DD", "nb5 train"), (nb5[1], "#0F6E56", "nb5 val"))):
        b = ax.bar(x + (k - 1.5) * wd, v, width=wd, color=col, label=lab)
        for b_, vv in zip(b, v):
            ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.01, "%.2f" % vv, ha="center", fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels(thr)
    ax.set_ylim(0, 1.12)
    ax.set_title("per soglia: il gap e' piu' piccolo a soglia larga", fontsize=10.4)
    ax.legend(frameon=False, fontsize=8.6, ncol=2, loc="upper left")
    fig.text(0.01, -0.04, "* dall'introduzione del nb5: nb3 rieseguito a 24 epoche, picco del val a meta' schedule (argmax 0.483) e poi calo di circa 5 punti. Quel run non e' tra i file analizzati.", fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "F23_gap_overfitting")


# --------------------------------------------------------------------------- f24
def f24():
    runs = [("nb1: adapter + flip", (0.158, 0.374, 0.624), "#B4B2A9"), ("nb2: finale (F6)", (0.296, 0.488, 0.678), "#5DCAA5"),
            ("nb3: finale (B3)", (0.350, 0.532, 0.650), "#AFA9EC"), ("nb5: finale (F2)", (0.393, 0.604, 0.750), "#D85A30")]
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    x = np.arange(3)
    wd = 0.2
    for k, (lab, v, col) in enumerate(runs):
        b = ax.bar(x + (k - 1.5) * wd, v, width=wd, color=col, label=lab)
        for b_, vv in zip(b, v):
            ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.01, "%.3f" % vv, ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\mathrm{PCK}@0.05$", r"$\mathrm{PCK}@0.10$", r"$\mathrm{PCK}@0.20$"], fontsize=10.5)
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    ax.set_title("Il miglior metodo di ogni notebook a tre soglie di tolleranza", fontsize=11, loc="left")
    fig.text(0.01, -0.02, r"Dal nb3 al nb5 il PCK@0.20 sale di 10 punti (0.650 $\to$ 0.750): per la prima volta si riducono gli errori semantici, non solo quelli di precisione.", fontsize=9.6)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "F24_soglie")


if __name__ == "__main__":
    for fn in (f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24):
        fn()
    print("fatto:", args.out)
