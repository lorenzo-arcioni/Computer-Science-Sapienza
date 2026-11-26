#!/usr/bin/env python3
"""
Genera tutte le figure SVG usate nella nota "AMLCV - Training Neural Networks".

Uso:   python genera_figure.py            (crea la cartella ./img con gli SVG)
Requisiti: numpy, matplotlib
Nota: gli esempi numerici (statistiche delle attivazioni) sono SIMULAZIONI reali
      con seed fisso, non valori copiati: riproducono i numeri mostrati nelle slide.
"""
import os
from math import erf, sqrt, pi

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "svg.fonttype": "path",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="svg", bbox_inches="tight")
    plt.close(fig)
    print("creato", name)


# ----------------------------------------------------------------------------
# 1. Funzioni di attivazione e derivate  (slide 4-19)
# ----------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gelu(x):
    return x * 0.5 * (1 + np.vectorize(erf)(x / sqrt(2)))


def gelu_grad(x):
    Phi = 0.5 * (1 + np.vectorize(erf)(x / sqrt(2)))
    phi = np.exp(-x ** 2 / 2) / sqrt(2 * pi)
    return Phi + x * phi


def fig_activations():
    x = np.linspace(-5, 5, 1001)
    a = 1.0  # alpha di ELU
    acts = {
        "Sigmoid": (sigmoid(x), sigmoid(x) * (1 - sigmoid(x))),
        "tanh": (np.tanh(x), 1 - np.tanh(x) ** 2),
        "ReLU": (np.maximum(0, x), (x > 0).astype(float)),
        "Leaky ReLU (0.1)": (np.maximum(0.1 * x, x), np.where(x > 0, 1.0, 0.1)),
        "ELU (α=1)": (np.where(x >= 0, x, a * (np.exp(x) - 1)), np.where(x >= 0, 1.0, a * np.exp(x))),
        "GELU": (gelu(x), gelu_grad(x)),
    }
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (name, (f, g)) in zip(axs.ravel(), acts.items()):
        ax.plot(x, f, color=BLUE, lw=2, label="f(x)")
        ax.plot(x, g, color=ORANGE, lw=1.8, ls="--", label="f'(x)")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_ylim(-1.6, 2.6)
        ax.set_xlim(-5, 5)
        ax.set_title(name)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Funzioni di attivazione (blu) e loro derivate (arancione tratteggiato)", y=1.0)
    fig.tight_layout()
    save(fig, "fig_activations.svg")


# ----------------------------------------------------------------------------
# 2. Saturazione del sigmoide (slide 6-8)
# ----------------------------------------------------------------------------
def fig_sigmoid_saturation():
    x = np.linspace(-10, 10, 1001)
    s = sigmoid(x)
    d = s * (1 - s)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    ax.plot(x, s, color=BLUE, lw=2)
    ax.axvspan(-10, -5, color=RED, alpha=0.12)
    ax.axvspan(5, 10, color=RED, alpha=0.12)
    ax.set_title("σ(x): le code sono 'piatte' (saturazione)")
    ax.set_xlabel("x")
    ax = axs[1]
    ax.plot(x, d, color=ORANGE, lw=2)
    ax.axvspan(-10, -5, color=RED, alpha=0.12)
    ax.axvspan(5, 10, color=RED, alpha=0.12)
    for xv in (-10, 0, 10):
        v = sigmoid(xv) * (1 - sigmoid(xv))
        ax.plot([xv], [v], "ko")
        ax.annotate(f"σ'({xv}) = {v:.2g}", (xv, v), textcoords="offset points",
                    xytext=(0 if xv == 0 else (25 if xv < 0 else -25), 10), ha="center", fontsize=9)
    ax.set_title("σ'(x) = σ(x)(1-σ(x)) ≤ 1/4: ≈0 in saturazione")
    ax.set_xlabel("x")
    fig.tight_layout()
    save(fig, "fig_sigmoid_saturation.svg")


# ----------------------------------------------------------------------------
# 3. Problema "non zero-centered" -> zig-zag (slide 9)
# ----------------------------------------------------------------------------
def fig_zigzag():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axs[0]
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.add_patch(Rectangle((0, 0), 2, 2, color=GREEN, alpha=0.18))
    ax.add_patch(Rectangle((-2, -2), 2, 2, color=GREEN, alpha=0.18))
    ax.add_patch(Rectangle((-2, 0), 2, 2, color=RED, alpha=0.12))
    ax.add_patch(Rectangle((0, -2), 2, 2, color=RED, alpha=0.12))
    ax.text(1, 1, "Δw1>0, Δw2>0\nPERMESSO", ha="center", va="center", color=GREEN)
    ax.text(-1, -1, "Δw1<0, Δw2<0\nPERMESSO", ha="center", va="center", color=GREEN)
    ax.text(-1, 1, "segni opposti\nVIETATO", ha="center", va="center", color=RED)
    ax.text(1, -1, "segni opposti\nVIETATO", ha="center", va="center", color=RED)
    ax.set_xlabel("Δw1"); ax.set_ylabel("Δw2")
    ax.set_title("Se tutti gli x_i > 0: ∂L/∂w_i = δ·x_i\nha lo stesso segno per ogni i")
    ax.grid(False)

    ax = axs[1]
    d1, d2 = np.array([1.0, 0.25]), np.array([-0.25, -1.0])
    p = np.array([0.0, 0.0]); pts = [p.copy()]
    for k in range(5):
        p = p + 0.55 * d1; pts.append(p.copy())
        p = p + 0.55 * d2; pts.append(p.copy())
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=BLUE, ms=4)
    ax.annotate("", xy=(pts[-1, 0], pts[-1, 1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2, ls="--"))
    ax.text(2.35, -1.3, "direzione ideale\n(+,−): non ottenibile\nin un solo passo", color=RED, fontsize=9)
    ax.set_title("Schema: per andare verso (+,−) servono\npassi alternati (+,+) e (−,−): zig-zag")
    ax.set_xlabel("w1"); ax.set_ylabel("w2")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-3.5, 1.2)
    fig.tight_layout()
    save(fig, "fig_zigzag.svg")


# ----------------------------------------------------------------------------
# 4. Data preprocessing (slide 22-24)
# ----------------------------------------------------------------------------
def fig_preprocessing():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((1500, 2))
    A = np.array([[2.2, 0.0], [1.6, 0.7]])
    X = z @ A.T + np.array([4.0, 3.0])
    Xc = X - X.mean(0)
    Xn = Xc / Xc.std(0)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, D, t, c in zip(axs, (X, Xc, Xn),
                           ("dati originali", "zero-centrati  X -= mean(X, axis=0)",
                            "normalizzati  X /= std(X, axis=0)"), ("#8b0000", GREEN, BLUE)):
        ax.scatter(D[:, 0], D[:, 1], s=3, alpha=0.5, color=c)
        ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="k", lw=0.7)
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_aspect("equal")
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_preprocessing.svg")

    # sensibilità dell'assegnazione di classe a una piccola rotazione del confine w
    rng = np.random.default_rng(1)
    P = rng.standard_normal((60, 2)) * 0.6
    far = P + np.array([5.0, 0.0])      # nuvola lontana dall'origine
    th = np.deg2rad(6)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, D, t in zip(axs, (far, P), ("prima: dati lontani dall'origine", "dopo: dati centrati")):
        side = lambda ang: (D[:, 1] * np.cos(ang) - D[:, 0] * np.sin(ang)) > 0   # lato della retta ruotata
        flips = (side(0.0) != side(th)).mean()
        ax.scatter(D[:, 0], D[:, 1], s=14, c=np.where(side(0.0) != side(th), RED, BLUE))
        xs = np.linspace(-3, 9, 10)
        ax.plot(xs, 0 * xs, color="k", lw=2)
        ax.plot(xs * np.cos(th), -xs * np.sin(th), color=RED, lw=2)
        ax.set_xlim(-3, 9); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.set_title(f"{t}\npunti che cambiano lato: {100*flips:.0f}%", fontsize=10)
    fig.suptitle("Stessa rotazione di 6° del confine (nero → rosso): cambia molto se i dati sono lontani dall'origine", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_preprocessing_sensitivity.svg")


# ----------------------------------------------------------------------------
# 5. Inizializzazione dei pesi (slide 26-34) - simulazioni reali
# ----------------------------------------------------------------------------
def forward_stats(kind, sigma=None, act="tanh", seed=0, n_layers=6, d=4096, n=16):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    outs = []
    for _ in range(n_layers):
        if kind == "const":
            W = sigma * rng.standard_normal((d, d))
        elif kind == "xavier":
            W = rng.standard_normal((d, d)) / np.sqrt(d)
        else:  # kaiming
            W = rng.standard_normal((d, d)) * np.sqrt(2.0 / d)
        s = x @ W
        x = np.tanh(s) if act == "tanh" else np.maximum(0, s)
        outs.append(x.copy())
    return outs


def fig_init():
    cases = [
        ("W = 0.01·randn, tanh", dict(kind="const", sigma=0.01, act="tanh")),
        ("W = 0.05·randn, tanh", dict(kind="const", sigma=0.05, act="tanh")),
        ("Xavier 1/√Din, tanh", dict(kind="xavier", act="tanh")),
        ("Xavier 1/√Din, ReLU", dict(kind="xavier", act="relu")),
        ("Kaiming √(2/Din), ReLU", dict(kind="kaiming", act="relu")),
    ]
    res = {name: forward_stats(**kw) for name, kw in cases}

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = [RED, "#9467bd", BLUE, ORANGE, GREEN]
    for (name, outs), c in zip(res.items(), cols):
        ax.plot(range(1, 7), [o.std() for o in outs], "-o", color=c, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("strato"); ax.set_ylabel("std delle attivazioni (scala log)")
    ax.set_title("Rete a 6 strati, 4096 unità: std delle attivazioni per strato")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_init_std.svg")

    rows = [("W = 0.01·randn, tanh", res["W = 0.01·randn, tanh"]),
            ("Xavier, tanh", res["Xavier 1/√Din, tanh"]),
            ("Kaiming, ReLU", res["Kaiming √(2/Din), ReLU"])]
    fig, axs = plt.subplots(3, 6, figsize=(14, 6.5))
    for r, (name, outs) in enumerate(rows):
        for c, o in enumerate(outs):
            ax = axs[r, c]
            ax.hist(o.ravel(), bins=60, range=(-1, 1), color=BLUE)
            ax.set_title(f"L{c+1}  mean={o.mean():.2f} std={o.std():.2f}", fontsize=8)
            ax.set_yticks([]); ax.set_xticks([-1, 0, 1]); ax.grid(False)
        axs[r, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("Istogrammi delle attivazioni per strato (asse x tagliato a [-1,1] come nelle slide)", y=1.0)
    fig.tight_layout()
    save(fig, "fig_init_hist.svg")


# ----------------------------------------------------------------------------
# 6. Normalizzazioni: quali elementi condividono media/varianza (slide 43-49)
# ----------------------------------------------------------------------------
def fig_norm_cubes():
    C, N, S = 6, 4, 4  # canali, campioni, posizioni spaziali (H*W compresso)
    cases = [
        ("Batch Norm\nmedia su (N, H, W) per canale", lambda fc: fc.__setitem__((2, slice(None), slice(None)), True)),
        ("Layer Norm\nmedia su (C, H, W) per campione", lambda fc: fc.__setitem__((slice(None), 1, slice(None)), True)),
        ("Instance Norm\nmedia su (H, W) per campione e canale", lambda fc: fc.__setitem__((2, 1, slice(None)), True)),
        ("Group Norm (G=3)\nmedia su (C/G, H, W) per campione e gruppo", lambda fc: fc.__setitem__((slice(2, 4), 1, slice(None)), True)),
    ]
    fig = plt.figure(figsize=(15, 4.2))
    for i, (title, mark) in enumerate(cases):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        filled = np.ones((C, N, S), dtype=bool)
        hl = np.zeros((C, N, S), dtype=bool)
        mark(hl)
        fc_hl = np.zeros((C, N, S, 4)); fc_hl[...] = (0.12, 0.47, 0.71, 1.0)
        fc_bg = np.zeros((C, N, S, 4)); fc_bg[...] = (0.9, 0.9, 0.9, 0.10)
        ax.voxels(~hl, facecolors=fc_bg, edgecolors=(0.55, 0.55, 0.55, 0.35), linewidth=0.4)
        ax.voxels(hl, facecolors=fc_hl, edgecolors=(0, 0, 0, 1), linewidth=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("C", labelpad=-8); ax.set_ylabel("N", labelpad=-8); ax.set_zlabel("H,W", labelpad=-8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=22, azim=-62)
        ax.grid(False)
    fig.suptitle("Celle azzurre = insieme su cui si calcolano UNA media e UNA varianza (tensore N×C×H×W)", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_norm_cubes.svg")


# ----------------------------------------------------------------------------
# 7. Dropout (slide 57-68)
# ----------------------------------------------------------------------------
def fig_dropout():
    rng = np.random.default_rng(3)
    layers = [4, 6, 6, 1]
    xs = [0, 1, 2, 3]
    pos = [[(x, (n - 1) / 2 - i) for i in range(n)] for x, n in zip(xs, layers)]
    keep = [np.ones(n, bool) for n in layers]
    for l in (1, 2):
        keep[l] = rng.random(layers[l]) < 0.5
    keep[0] = np.array([True, False, True, True])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, use in zip(axs, (False, True)):
        ax.set_axis_off()
        for l in range(3):
            for i, p in enumerate(pos[l]):
                for j, q in enumerate(pos[l + 1]):
                    if (not use) or (keep[l][i] and keep[l + 1][j]):
                        ax.plot([p[0], q[0]], [p[1], q[1]], color="#bbbbbb", lw=0.7, zorder=1)
        for l in range(4):
            for i, p in enumerate(pos[l]):
                alive = (not use) or keep[l][i]
                ax.add_patch(Circle(p, 0.13, fc="white" if alive else "#dddddd", ec="k", lw=1.2, zorder=3))
                if not alive:
                    ax.plot([p[0] - 0.09, p[0] + 0.09], [p[1] - 0.09, p[1] + 0.09], color=GRAY, lw=1.5, zorder=4)
                    ax.plot([p[0] - 0.09, p[0] + 0.09], [p[1] + 0.09, p[1] - 0.09], color=GRAY, lw=1.5, zorder=4)
        ax.set_xlim(-0.4, 3.4); ax.set_ylim(-3.3, 3.3); ax.set_aspect("equal")
        ax.set_title("rete completa" if not use else "forward pass con dropout (maschera casuale)")
    fig.tight_layout()
    save(fig, "fig_dropout.svg")

    # i 4 casi del singolo neurone a = w1 x + w2 y con p = 1/2
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = ["(x,y)\nw1x+w2y", "(x,0)\nw1x", "(0,y)\nw2y", "(0,0)\n0"]
    ax.bar(range(4), [1, 1, 1, 1], color=[BLUE, ORANGE, ORANGE, GRAY], alpha=0.0)
    for i, (lab, c) in enumerate(zip(labels, (BLUE, ORANGE, GREEN, GRAY))):
        ax.add_patch(Rectangle((i - 0.4, 0), 0.8, 0.25, color=c, alpha=0.7))
        ax.text(i, 0.31, lab, ha="center", va="bottom", fontsize=9)
        ax.text(i, 0.12, "prob 1/4", ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 0.9); ax.set_axis_off()
    ax.set_title("Neurone a = w1·x + w2·y con p(keep)=1/2: le 4 maschere\nE[a] = ¼(w1x+w2y) + ¼w1x + ¼w2y + 0 = ½(w1x+w2y)", fontsize=10)
    save(fig, "fig_dropout_masks.svg")


# ----------------------------------------------------------------------------
# 8. Curve di training / validation (slide 53, 83-85)  (schematiche)
# ----------------------------------------------------------------------------
def fig_curves():
    t = np.linspace(0, 1, 200)
    sat = lambda a: 1 - np.exp(-a * t)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))

    ax = axs[0]
    tr = 0.95 * sat(4); va = 0.7 * sat(5) - 0.12 * np.clip(t - 0.4, 0, None) ** 1.2 * 3
    ax.plot(t, tr, color=BLUE, lw=2, label="Train"); ax.plot(t, va, color=ORANGE, lw=2, label="Val")
    k = va.argmax(); ax.axvline(t[k], color="k", ls=":"); ax.annotate("early stopping:\nsalva questo snapshot", (t[k], va[k]), (t[k] + 0.05, va[k] - 0.35), arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_title("Early stopping")

    ax = axs[1]
    ax.plot(t, 0.75 * sat(2.5), color=BLUE, lw=2, label="Train"); ax.plot(t, 0.62 * sat(2.2), color=ORANGE, lw=2, label="Val")
    ax.set_title("Ancora in salita:\naddestra più a lungo")

    ax = axs[2]
    ax.plot(t, 0.98 * sat(6), color=BLUE, lw=2, label="Train")
    ax.plot(t, 0.65 * sat(8) - 0.25 * np.clip(t - 0.3, 0, None), color=ORANGE, lw=2, label="Val")
    ax.set_title("Gap grande = overfitting:\n+regolarizzazione, +dati")

    ax = axs[3]
    ax.plot(t, 0.6 * sat(4), color=BLUE, lw=2, label="Train"); ax.plot(t, 0.585 * sat(4), color=ORANGE, lw=2, label="Val")
    ax.set_title("Nessun gap = underfitting:\naddestra di più, modello più grande")

    for ax in axs:
        ax.set_xlabel("tempo (iterazioni/epoche)"); ax.set_ylim(-0.05, 1.05); ax.set_yticks([]); ax.grid(False)
    axs[0].set_ylabel("accuratezza"); axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Curve schematiche (non dati reali)", y=1.02, fontsize=9)
    fig.tight_layout()
    save(fig, "fig_curves.svg")


# ----------------------------------------------------------------------------
# 9. Data augmentation, cutout, mixup (slide 71-80) - immagini sintetiche
# ----------------------------------------------------------------------------
def make_toy(kind):
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n, 3))
    if kind == "A":   # "gatto" stilizzato, spostato a sinistra per mostrare flip/crop
        img[...] = np.array([0.55, 0.75, 0.95]) * (0.7 + 0.3 * yy[..., None] / n)
        img[((xx - 24) ** 2 + (yy - 36) ** 2) < 15 ** 2] = (0.95, 0.6, 0.2)
        for ex in (14, 34):
            tri = (yy < 26) & (yy > 14) & (np.abs(xx - ex) < (yy - 14) * 0.5 + 0.5) & (yy - 14 > 0)
            img[tri & (((xx - 24) ** 2 + (yy - 36) ** 2) >= 15 ** 2)] = (0.95, 0.6, 0.2)
        img[((xx - 19) ** 2 + (yy - 33) ** 2) < 9] = (0.1, 0.1, 0.1)
        img[((xx - 29) ** 2 + (yy - 33) ** 2) < 9] = (0.1, 0.1, 0.1)
    else:             # "cane" stilizzato
        img[...] = np.array([0.7, 0.9, 0.6]) * (0.7 + 0.3 * xx[..., None] / n)
        img[((xx - 40) ** 2 + (yy - 36) ** 2) < 16 ** 2] = (0.55, 0.35, 0.2)
        img[(xx > 22) & (xx < 32) & (yy > 30) & (yy < 56)] = (0.4, 0.25, 0.15)
        img[((xx - 36) ** 2 + (yy - 34) ** 2) < 9] = (1, 1, 1)
        img[((xx - 46) ** 2 + (yy - 34) ** 2) < 9] = (1, 1, 1)
    return np.clip(img, 0, 1)


def fig_augmentation():
    A, B = make_toy("A"), make_toy("B")
    flip = A[:, ::-1]
    crop = A[8:56, 8:56]
    jit = np.clip((A - 0.5) * 1.4 + 0.5 + 0.12, 0, 1)
    cut = A.copy(); cut[26:50, 6:30] = 0.5
    mix = 0.4 * A + 0.6 * B
    items = [(A, "originale"), (flip, "flip orizzontale"), (crop, "random crop 48×48\n(poi resize)"),
             (jit, "color jitter\n(contrasto×1.4, luminosità+0.12)"), (cut, "Cutout\n(regione a valore costante)"),
             (B, "altra immagine (cane)"), (mix, "Mixup  λ=0.4\ntarget: gatto 0.4, cane 0.6")]
    fig, axs = plt.subplots(2, 4, figsize=(11, 6))
    for ax in axs.ravel():
        ax.set_axis_off()
    for ax, (im, t) in zip(axs.ravel(), items):
        ax.imshow(im, interpolation="nearest"); ax.set_title(t, fontsize=9)
    fig.tight_layout()
    save(fig, "fig_augmentation.svg")


# ----------------------------------------------------------------------------
# 10. Transfer learning (slide 86-95)
# ----------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec="k", fs=9, tc="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def fig_transfer():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.set_axis_off()
    rows = [
        (5.1, "1) Pre-training su ImageNet (1000 classi, ~1M immagini)",
         [("CONV (feature)", "#cfe2f3"), ("FC 4096", "#cfe2f3"), ("FC 4096", "#cfe2f3"), ("FC 1000", "#cfe2f3")], "tutto addestrato"),
        (3.0, "2a) Pochi dati: feature extractor",
         [("CONV\nCONGELATO", "#d9d9d9"), ("FC 4096\nCONGELATO", "#d9d9d9"), ("FC nuovo\n(es. 10 classi)", "#fce5cd")], "si addestra solo il nuovo classificatore"),
        (0.9, "2b) Più dati: fine-tuning",
         [("CONV\nLR piccolo", "#cfe2f3"), ("FC 4096\nLR piccolo", "#cfe2f3"), ("FC nuovo\nLR normale", "#fce5cd")], "si addestra tutto, con LR più basso sul pre-addestrato"),
    ]
    for y, title, boxes, note in rows:
        ax.text(0.1, y + 1.05, title, fontsize=10, weight="bold")
        for i, (t, c) in enumerate(boxes):
            box(ax, 0.3 + i * 2.5, y, 2.1, 0.85, t, c)
            if i:
                ax.annotate("", xy=(0.3 + i * 2.5, y + 0.42), xytext=(0.3 + (i - 1) * 2.5 + 2.1, y + 0.42), arrowprops=dict(arrowstyle="->"))
        ax.text(0.3 + len(boxes) * 2.5 + 0.1, y + 0.42, note, fontsize=9, va="center", style="italic")
    ax.annotate("", xy=(3.5, 4.2), xytext=(3.5, 4.95), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(3.65, 4.55, "si copiano i pesi CONV; si scarta/sostituisce l'ultimo strato", color=RED, fontsize=9)
    save(fig, "fig_transfer.svg")


def main():
    fig_activations()
    fig_sigmoid_saturation()
    fig_zigzag()
    fig_preprocessing()
    fig_init()
    fig_norm_cubes()
    fig_dropout()
    fig_curves()
    fig_augmentation()
    fig_transfer()


if __name__ == "__main__":
    main()
