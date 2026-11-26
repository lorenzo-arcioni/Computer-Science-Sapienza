#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genera_figure_sequenze.py
Genera tutte le figure (SVG) usate nella nota "06 - Sequence Models I".
Uso:  python3 genera_figure_sequenze.py      (scrive gli .svg nella stessa cartella dello script)
Dipendenze: numpy, matplotlib.
Le figure con numeri (char-LM, campi recettivi, gradienti) sono CALCOLATE, non disegnate a mano.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon

plt.rcParams["svg.fonttype"] = "path"          # testo convertito in tracciati: si vede ovunque
plt.rcParams["font.family"] = "DejaVu Sans"
OUT = os.path.dirname(os.path.abspath(__file__))

C = dict(inp="#f4cccc", hid="#d9ead3", out="#c9daf8", loss="#d9d2e9", w="#fce5cd",
         fw="#cfcfcf", edge="#444444", red="#cc0000", blue="#1155cc", green="#38761d",
         orange="#e69138", gray="#999999", yellow="#fff2cc", magenta="#c800c8")


# ----------------------------------------------------------------------------- utilità
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="svg", bbox_inches="tight")
    if os.environ.get("PREVIEW"):
        os.makedirs("/home/claude/png", exist_ok=True)
        fig.savefig("/home/claude/png/" + name.replace(".svg", ".png"), dpi=70, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)


def newax(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, label="", fc="white", ec=None, fs=9, lw=1.0, bold=False,
        rounded=False, color="black", z=2):
    ec = ec or C["edge"]
    if rounded:
        p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=0,rounding_size=0.15", fc=fc, ec=ec, lw=lw, zorder=z)
    else:
        p = Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    if label:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=z + 1)
    return p


def arrow(ax, p, q, color=None, lw=1.3, rad=0.0, ls="-", ms=10, z=1):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color or C["edge"],
                        lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def txt(ax, x, y, s, fs=9, color="black", ha="center", va="center", bold=False, style="normal"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal", fontstyle=style)


# ----------------------------------------------------------------------------- 1. configurazioni I/O
def fig_seq_configs():
    cfgs = [("one to one", 1, [0], [0], "classificazione\nimmagine"),
            ("one to many", 3, [0], [0, 1, 2], "image captioning"),
            ("many to one", 3, [0, 1, 2], [2], "action prediction\n(video → classe)"),
            ("many to many\n(encoder–decoder)", 5, [0, 1, 2], [2, 3, 4],
             "machine translation,\nvideo captioning,\nforecasting"),
            ("many to many\n(sincrono)", 3, [0, 1, 2], [0, 1, 2], "classificazione video\na livello di frame")]
    fig, ax = newax(13, 4.6, (-0.8, 17.6), (-2.6, 5.4))
    x0 = 0.0
    for name, nh, ins, outs, ex in cfgs:
        wpanel = nh * 1.1 + 0.5
        box(ax, x0 + wpanel / 2 - 0.25, 1.9, wpanel, 5.0, fc="#f3f3f3", ec="#f3f3f3", rounded=True, z=0)
        for s in range(nh):
            cx = x0 + s * 1.1 + 0.3
            box(ax, cx, 1.9, 0.7, 1.0, fc=C["hid"])
            if s < nh - 1:
                arrow(ax, (cx + 0.37, 1.9), (cx + 1.1 - 0.37, 1.9), color=C["green"])
            if s in ins:
                box(ax, cx, 0.2, 0.7, 1.0, fc=C["inp"])
                arrow(ax, (cx, 0.72), (cx, 1.38), color=C["red"])
            if s in outs:
                box(ax, cx, 3.6, 0.7, 1.0, fc=C["out"])
                arrow(ax, (cx, 2.42), (cx, 3.08), color=C["blue"])
        txt(ax, x0 + wpanel / 2 - 0.25, -1.0, name, fs=9, bold=True)
        txt(ax, x0 + wpanel / 2 - 0.25, -2.0, ex, fs=8, color="#555555")
        x0 += wpanel + 0.35
    txt(ax, 8.4, 5.2, "rosa = input   |   verde = stato nascosto   |   azzurro = output", fs=9, color="#555555")
    save(fig, "fig_seq_configs.svg")


# ----------------------------------------------------------------------------- 2. forecasting
def fig_forecast():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    t = np.arange(8)
    xs = t * 1.0
    ys = 2.0 + 0.15 * t + 0.05 * np.sin(t)
    ax.plot(xs, ys, "o-", color=C["blue"], label="osservato  $x_1,\\dots,x_{T_o}$")
    h = np.arange(1, 7)
    xf = xs[-1] + h * 1.0
    ax.plot(np.r_[xs[-1], xf], np.r_[ys[-1], ys[-1] + 0.35 * h], "o--", color=C["green"],
            label="futuro plausibile A (gira a sinistra)")
    ax.plot(np.r_[xs[-1], xf], np.r_[ys[-1], ys[-1] - 0.30 * h], "o--", color=C["red"],
            label="futuro plausibile B (gira a destra)")
    ax.plot(np.r_[xs[-1], xf], np.r_[ys[-1], ys[-1] + 0.025 * h], "s-", color="black", lw=2,
            label="media delle due modalità (cosa predice una regressione MSE)")
    ax.add_patch(Rectangle((xf[2] - 0.3, ys[-1] - 0.35), 0.6, 0.7, fc="#bbbbbb", ec="k", zorder=3))
    ax.text(xf[2], ys[-1], "ostacolo", ha="center", va="center", fontsize=7, zorder=4)
    ax.axvline(xs[-1], color="gray", ls=":")
    ax.text(xs[-1] - 0.1, ax.get_ylim()[1] - 0.15, "adesso", ha="right", fontsize=8, color="gray")
    ax.set_xlabel("posizione x")
    ax.set_ylabel("posizione y")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Forecasting di traiettorie: il futuro è multimodale", fontsize=10)
    save(fig, "fig_forecast.svg")


# ----------------------------------------------------------------------------- 3. convoluzione 1D
def fig_conv1d():
    x = np.array([10, 50, 60, 10, 20, 40, 30], float)
    g = np.array([1, 1, 1]) / 3.0
    y = np.correlate(x, g, "valid")
    fig, ax = newax(9.5, 6.8, (-3.0, 12.0), (-5.6, 2.2))
    txt(ax, 3, 1.5, "sorgente  $x$ (tempo →)", fs=10, bold=True)
    for i in range(len(y)):
        yy = -i * 1.1
        for j in range(len(x)):
            inwin = i <= j < i + 3
            box(ax, j, yy, 0.85, 0.85, f"{x[j]:.0f}", fc=C["w"] if inwin else "white",
                ec=C["orange"] if inwin else C["gray"], lw=2 if inwin else 0.8, fs=9)
        s = "+".join(f"{x[i + k]:.0f}" for k in range(3))
        txt(ax, 7.0, yy, f"$y[{i}]=({s})/3={y[i]:.1f}$", fs=9, ha="left")
    txt(ax, 3, -5.3, "nucleo (kernel) $g=(\\frac{1}{3},\\frac{1}{3},\\frac{1}{3})$: la stessa finestra scorre lungo il tempo",
        fs=9, color="#555")
    save(fig, "fig_conv1d.svg")


# ----------------------------------------------------------------------------- 4. causale vs centrata
def fig_causal():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = ["t-3", "t-2", "t-1", "t", "t+1", "t+2", "t+3"]
    for ax, mode in zip(axs, ["centrata", "causale"]):
        ax.set_xlim(-1.2, 7.2)
        ax.set_ylim(-1.0, 3.2)
        ax.set_aspect("equal")
        ax.axis("off")
        used = [2, 3, 4] if mode == "centrata" else [1, 2, 3]
        for j in range(7):
            col = C["inp"]
            if j in used:
                col = "#ea9999" if (mode == "centrata" and j == 4) else C["w"]
            box(ax, j, 0, 0.8, 0.8, f"$x_{{{labels[j]}}}$", fc=col, fs=9,
                ec=C["orange"] if j in used else C["gray"], lw=2 if j in used else 0.8)
            box(ax, j, 2.0, 0.8, 0.8, "", fc=C["out"] if j == 3 else "white",
                ec=C["blue"] if j == 3 else "#cccccc", lw=2 if j == 3 else 0.8)
        for j in used:
            arrow(ax, (j, 0.42), (3, 1.58), color=C["blue"], lw=1.4)
        ax.text(3, 2.0, "$y_t$", ha="center", va="center", fontsize=10)
        if mode == "centrata":
            ax.set_title("convoluzione centrata (K=3): $y_t$ usa $x_{t+1}$ = FUTURO", fontsize=10)
            ax.text(4, -0.75, "futuro: inaccettabile\nin forecasting", color=C["red"], ha="center", fontsize=8)
        else:
            ax.set_title("convoluzione causale (K=3): $y_t$ usa solo $x_{t-2},x_{t-1},x_t$", fontsize=10)
            ax.text(2, -0.75, "si aggiungono K-1 zeri a sinistra\nper mantenere la lunghezza", color=C["green"],
                    ha="center", fontsize=8)
    save(fig, "fig_causal.svg")


# ----------------------------------------------------------------------------- 5. campo recettivo / dilatazione
def reach_sets(dils, T, K=3):
    L = len(dils)
    reach = [None] * (L + 1)
    reach[L] = {T - 1}
    for l in range(L, 0, -1):
        d = dils[l - 1]
        reach[l - 1] = {n - k * d for n in reach[l] for k in range(K) if n - k * d >= 0}
    return reach


def fig_dilated():
    T, K = 16, 3
    cases = [([1, 1, 1], "senza dilatazione: $d=1,1,1$"), ([1, 2, 4], "con dilatazione: $d=1,2,4$")]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8.6))
    for ax, (dils, title) in zip(axs, cases):
        L = len(dils)
        reach = reach_sets(dils, T, K)
        rf = len(reach[0])
        ax.set_xlim(-1.0, T + 1.2)
        ax.set_ylim(-0.9, L + 0.9)
        ax.axis("off")
        for l in range(L + 1):
            for n in range(T):
                inr = n in reach[l]
                ax.add_patch(Circle((n, l), 0.22, fc=C["orange"] if inr else "#eeeeee",
                                    ec=C["edge"] if inr else "#cccccc", lw=1, zorder=3))
        for l in range(L, 0, -1):
            d = dils[l - 1]
            for n in reach[l]:
                for k in range(K):
                    m = n - k * d
                    if m >= 0:
                        ax.plot([n, m], [l, l - 1], color=C["blue"], lw=1.1, zorder=1)
            ax.text(T + 0.5, l - 0.5, f"d={d}", fontsize=9, va="center")
        for n in (0, 1, T - 2, T - 1):
            ax.text(n, -0.55, f"$x_{{{n}}}$" if n < T - 1 else "$x_T$", ha="center", fontsize=8)
        ax.text(-0.9, L, "uscita", fontsize=8, va="center")
        ax.text(-0.9, 0, "input", fontsize=8, va="center")
        ax.set_title(f"{title}   →   campo recettivo = {rf} istanti"
                     f"  (formula: $1+(K-1)\\sum_\\ell d_\\ell={1 + (K - 1) * sum(dils)}$)", fontsize=10)
    save(fig, "fig_dilated.svg")


def fig_rf_growth():
    L = np.arange(1, 11)
    K = 3
    lin = 1 + L * (K - 1)
    dil = 1 + (K - 1) * (2 ** L - 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(L, lin, "o-", color=C["red"], label="$d=1$: $1+L(K-1)$  (lineare)")
    ax.semilogy(L, dil, "s-", color=C["green"], label="$d=2^{\\ell-1}$: $1+(K-1)(2^L-1)$  (esponenziale)")
    ax.axhline(1000, color="gray", ls=":")
    ax.text(1.1, 1150, "sequenza di T = 1000 istanti", fontsize=8, color="gray")
    ax.set_xlabel("numero di strati  L   (K = 3)")
    ax.set_ylabel("campo recettivo (scala log)")
    ax.legend(fontsize=8)
    ax.set_title("Campo recettivo: dilatazione ⇒ profondità logaritmica", fontsize=10)
    save(fig, "fig_rf_growth.svg")


# ----------------------------------------------------------------------------- 6. blocco residuo TCN
def fig_tcn_block():
    fig, ax = newax(6.2, 8.4, (-2.2, 5.0), (-1.2, 11.4))
    names = ["Dilated Causal Conv", "WeightNorm", "ReLU", "Dropout"] * 2
    cols = [C["hid"], C["yellow"], C["out"], C["loss"]] * 2
    y = 1.0
    arrow(ax, (0, -0.2), (0, 0.55), lw=1.5)
    txt(ax, 0, -0.6, "$z^{(i-1)}$  (ingresso del blocco)", fs=9)
    for n, c in zip(names, cols):
        box(ax, 0, y, 2.6, 0.6, n, fc=c, fs=9, rounded=True)
        if y < 8.0:
            arrow(ax, (0, y + 0.3), (0, y + 0.7), lw=1.3)
        y += 1.0
    # somma
    box(ax, 0, 9.4, 0.6, 0.6, "+", fc="white", fs=14, rounded=True)
    arrow(ax, (0, 8.3), (0, 9.1), lw=1.3)
    arrow(ax, (0, 9.7), (0, 10.4), lw=1.5)
    txt(ax, 0, 10.8, "$o=\\mathrm{Activation}(z^{(i-1)}+F(z^{(i-1)}))$", fs=9)
    # skip
    ax.plot([0, 0], [-0.0, 0.0], lw=0)
    ax.plot([0, 3.0, 3.0], [0.2, 0.2, 4.2], color=C["edge"], lw=1.4)
    box(ax, 3.0, 4.9, 1.7, 0.9, "1×1 Conv\n(opzionale)", fc=C["w"], fs=8, rounded=True)
    ax.plot([3.0, 3.0], [5.35, 9.4], color=C["edge"], lw=1.4)
    arrow(ax, (3.0, 9.4), (0.32, 9.4), lw=1.4)
    ax.add_patch(Circle((0, 0.2), 0.09, fc="k", zorder=5))
    txt(ax, -1.9, 4.5, "ramo\nprincipale\n$F(\\cdot)$", fs=9, color="#555")
    txt(ax, 4.3, 7.4, "ramo\nresiduo\n(skip)", fs=9, color="#555")
    save(fig, "fig_tcn_block.svg")


# ----------------------------------------------------------------------------- 7. RNN srotolata
def draw_unrolled(ax, T=4, backward=False):
    S = 3.1
    xs = [2.0 + i * S for i in range(T)]
    box(ax, 0.2, 0.0, 0.9, 1.0, "$h_0$", fc=C["hid"])
    prev = (0.65, 0.0)
    for i, x in enumerate(xs):
        box(ax, x, 0.0, 1.0, 1.0, "$f_W$", fc=C["fw"], fs=11)
        box(ax, x + 1.5, 0.0, 0.9, 1.0, f"$h_{i + 1}$", fc=C["hid"])
        box(ax, x, -2.1, 0.9, 1.0, f"$x_{i + 1}$", fc=C["inp"])
        box(ax, x + 1.5, 2.1, 0.9, 1.0, f"$y_{i + 1}$", fc=C["out"])
        arrow(ax, prev, (x - 0.5, 0.0))
        arrow(ax, (x + 0.5, 0.0), (x + 1.05, 0.0))
        arrow(ax, (x, -1.58), (x, -0.5))
        arrow(ax, (x + 1.5, 0.5), (x + 1.5, 1.58))
        prev = (x + 1.95, 0.0)
        if not backward:
            box(ax, x + 1.5, 4.0, 0.9, 0.9, f"$L_{i + 1}$", fc=C["loss"])
            arrow(ax, (x + 1.5, 2.62), (x + 1.5, 3.55))
            arrow(ax, (x + 1.5, 4.5), (xs[T // 2] + 0.5, 5.7), rad=0.0, color=C["gray"], lw=1)
        if backward:
            arrow(ax, (x + 1.05, -0.38), (x + 0.5, -0.38), color=C["red"], lw=1.6, z=4)
            arrow(ax, (x - 0.5, -0.38), (x - 0.95 if i else 0.7, -0.38), color=C["red"], lw=1.6, z=4)
    box(ax, xs[T // 2] + 0.5, 6.1, 1.0, 0.9, "$L$" if not backward else "", fc=C["loss"], fs=11) \
        if not backward else None
    box(ax, xs[0], -4.3, 0.9, 0.9, "$W$", fc=C["w"], fs=11)
    for i, x in enumerate(xs):
        arrow(ax, (xs[0] + 0.3, -3.85), (x + (0.2 if i else 0.0), -0.55), color=C["gray"], lw=1, rad=0.25 if i else 0.0)
    return xs


def fig_rnn_unrolled():
    fig, ax = newax(12, 8.4, (-1.0, 14.6), (-5.2, 6.8))
    draw_unrolled(ax, 4, backward=False)
    txt(ax, 8.4, -4.4, "$W$ è lo STESSO in ogni passo (grigio): $L=\\sum_t L_t$ e il gradiente su $W$ è la somma dei contributi",
        fs=9, color="#555")
    save(fig, "fig_rnn_unrolled.svg")


def fig_bptt():
    fig, ax = newax(12, 7.0, (-1.0, 14.6), (-5.2, 4.4))
    draw_unrolled(ax, 4, backward=True)
    for i in range(4):
        x = 2.0 + i * 3.1
        arrow(ax, (x + 1.5, 1.65), (x + 1.5, 0.55), color=C["red"], lw=1.6, z=4, rad=0.0)
    txt(ax, 6.6, 3.5, "forward: da sinistra a destra per TUTTA la sequenza → loss;  "
        "backward (rosso): da destra a sinistra per TUTTA la sequenza → gradienti",
        fs=9, color=C["red"])
    save(fig, "fig_bptt.svg")


# ----------------------------------------------------------------------------- 8. seq2seq
def fig_seq2seq():
    fig, ax = newax(14.4, 6.0, (-1.2, 16.4), (-3.4, 5.2))
    S = 2.2
    # encoder
    ex = [1.5 + i * S for i in range(3)]
    box(ax, -0.3, 0, 0.8, 0.9, "$h_0$", fc=C["hid"])
    prev = (0.12, 0)
    for i, x in enumerate(ex):
        box(ax, x, 0, 0.9, 0.9, "$f_{W_1}$", fc=C["fw"], fs=9)
        box(ax, x, -2.0, 0.8, 0.9, f"$x_{i + 1}$", fc=C["inp"])
        arrow(ax, (x, -1.52), (x, -0.45))
        arrow(ax, prev, (x - 0.45, 0))
        if i < 2:
            box(ax, x + 1.1, 0, 0.7, 0.9, f"$h_{i + 1}$", fc=C["hid"])
            arrow(ax, (x + 0.45, 0), (x + 0.75, 0))
            prev = (x + 1.45, 0)
    box(ax, ex[2] + 1.3, 0, 0.9, 1.0, "$h_T$", fc=C["yellow"], lw=2, fs=10)
    arrow(ax, (ex[2] + 0.45, 0), (ex[2] + 0.85, 0))
    txt(ax, ex[2] + 1.3, 1.1, "vettore di\ncontesto", fs=8, color="#7f6000")
    txt(ax, 3.7, -3.2, "ENCODER (many-to-one), pesi $W_1$", fs=10, bold=True)
    box(ax, 0.8, -3.2, 0.01, 0.01, "")
    # decoder
    dx = [9.6 + i * S for i in range(3)]
    prev = (ex[2] + 1.75, 0)
    ins = ["<START>", "$y_1$", "$y_2$"]
    for i, x in enumerate(dx):
        box(ax, x, 0, 0.9, 0.9, "$f_{W_2}$", fc=C["fw"], fs=9)
        box(ax, x + 1.1, 0, 0.7, 0.9, f"$s_{i + 1}$", fc=C["hid"])
        box(ax, x + 1.1, 2.0, 0.8, 0.9, f"$y_{i + 1}$", fc=C["out"])
        box(ax, x, -2.0, 0.95, 0.9, ins[i], fc=C["inp"], fs=8)
        arrow(ax, (x, -1.52), (x, -0.45))
        arrow(ax, prev, (x - 0.45, 0))
        arrow(ax, (x + 0.45, 0), (x + 0.75, 0))
        arrow(ax, (x + 1.1, 0.45), (x + 1.1, 1.55), color=C["blue"])
        if i < 2:
            arrow(ax, (x + 1.5, 2.0), (dx[i + 1] - 0.1, -1.5), color=C["gray"], rad=-0.45, ls="--", lw=1)
        prev = (x + 1.45, 0)
    txt(ax, 11.8, -3.2, "DECODER (one-to-many), pesi $W_2$", fs=10, bold=True)
    txt(ax, 7.0, 4.4, "frecce tratteggiate: l'uscita del passo precedente è l'ingresso del successivo",
        fs=8, color="#555")
    save(fig, "fig_seq2seq.svg")


# ----------------------------------------------------------------------------- 9. char-level LM
V = "helo"
LOGITS = np.array([[1.0, 2.2, -3.0, 4.1], [0.5, 0.3, -1.0, 1.2], [0.1, 0.5, 1.9, -1.1], [0.2, -1.5, -0.1, 2.2]])
HID = np.array([[0.3, -0.1, 0.9], [1.0, 0.3, 0.1], [0.1, -0.5, -0.3], [-0.3, 0.9, 0.7]])
INP = [0, 1, 2, 2]       # h e l l
TGT = [1, 2, 2, 3]       # e l l o


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def vec(ax, cx, ytop, vals, fc, hl=None, fmt="{:.1f}", bw=1.1, bh=0.5):
    for k, v in enumerate(vals):
        yy = ytop - k * bh
        hlk = (hl == k)
        box(ax, cx, yy, bw, bh, fmt.format(v), fc=fc, fs=8.5, bold=hlk,
            color=C["green"] if hlk else "black", lw=1.8 if hlk else 0.8,
            ec=C["green"] if hlk else C["edge"])


def fig_charlm():
    fig, ax = newax(12.5, 11.0, (-4.2, 12.6), (-1.6, 13.4))
    X = [0.5, 3.6, 6.7, 9.8]
    P = [softmax(l) for l in LOGITS]
    for t in range(4):
        x = X[t]
        txt(ax, x, -0.9, f"“{V[INP[t]]}”", fs=11, bold=True)
        oh = np.eye(4)[INP[t]]
        vec(ax, x, 1.15, oh, C["inp"], fmt="{:.0f}")
        vec(ax, x, 3.9, HID[t], C["hid"], hl=None)
        vec(ax, x, 6.55, LOGITS[t], C["out"], hl=TGT[t])
        vec(ax, x, 9.0, P[t], "#fde9d9", hl=TGT[t], fmt="{:.3f}")
        txt(ax, x, 10.9, f"target “{V[TGT[t]]}”", fs=10, bold=True)
        txt(ax, x, 11.8, f"$L_{t + 1}=-\\log p={-np.log(P[t][TGT[t]]):.2f}$", fs=9.5, color=C["red"])
        arrow(ax, (x, 1.5), (x, 2.6), color=C["red"])
        arrow(ax, (x, 4.2), (x, 4.75), color=C["blue"])
        arrow(ax, (x, 6.85), (x, 7.45), color="#b45f06")
        if t < 3:
            arrow(ax, (x + 0.65, 3.4), (X[t + 1] - 0.65, 3.4), color=C["green"])
            txt(ax, (x + X[t + 1]) / 2, 3.75, "$W_{hh}$", fs=8, color=C["green"])
    for t in range(4):
        txt(ax, X[t] + 0.95, 2.1, "$W_{xh}$", fs=7.5, color=C["red"], ha="left")
        txt(ax, X[t] + 0.95, 4.5, "$W_{hy}$", fs=7.5, color=C["blue"], ha="left")
    lab = [(0.7, "input (one-hot)\n[h e l o]"), (3.5, "strato nascosto\n$h_t$ (3 unità)"),
           (6.0, "strato di output\n(logit $z_t$)"), (8.4, "softmax $p_t$"),
           (10.9, "target"), (11.8, "loss")]
    for y, s in lab:
        txt(ax, -2.7, y, s, fs=9, color="#555")
    txt(ax, 5.2, -1.45, f"Loss totale $L=\\sum_t L_t={sum(-np.log(P[t][TGT[t]]) for t in range(4)):.2f}$"
        f"   (media {np.mean([-np.log(P[t][TGT[t]]) for t in range(4)]):.2f})", fs=10)
    save(fig, "fig_charlm.svg")


def fig_charlm_probs():
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    for t, ax in enumerate(axs):
        p = softmax(LOGITS[t])
        cols = [C["green"] if k == TGT[t] else C["gray"] for k in range(4)]
        ax.bar(range(4), p, color=cols)
        ax.set_xticks(range(4))
        ax.set_xticklabels(list(V))
        ax.set_title(f"t={t + 1}: input “{V[INP[t]]}”, target “{V[TGT[t]]}”\n"
                     f"$p_{{target}}={p[TGT[t]]:.3f}$,  $L_t={-np.log(p[TGT[t]]):.2f}$", fontsize=8.5)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("probabilità")
    fig.suptitle("Softmax calcolata dai logit mostrati nelle slide (verde = carattere corretto)", fontsize=9.5, y=1.07)
    save(fig, "fig_charlm_probs.svg")


def fig_teacher_forcing():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8.0))
    seqs = ["h", "e", "l", "l"]
    outs = ["e", "l", "l", "o"]
    for ax, mode in zip(axs, ["tf", "free"]):
        ax.set_xlim(-1.2, 12.0)
        ax.set_ylim(-1.9, 4.3)
        ax.set_aspect("equal")
        ax.axis("off")
        for t in range(4):
            x = 0.8 + t * 2.7
            box(ax, x, 1.2, 0.9, 0.9, "RNN", fc=C["hid"], fs=8)
            if mode == "tf":
                box(ax, x, -0.7, 0.9, 0.9, f"“{seqs[t]}”", fc=C["inp"])
            else:
                lab = seqs[0] if t == 0 else f"$\\hat y_{{{t}}}$"
                box(ax, x, -0.7, 0.9, 0.9, lab, fc=C["inp"])
            arrow(ax, (x, -0.22), (x, 0.7))
            box(ax, x, 3.1, 0.9, 0.9, f"“{outs[t]}”" if mode == "tf" else f"$\\hat y_{{{t + 1}}}$",
                fc=C["out"])
            arrow(ax, (x, 1.7), (x, 2.6), color=C["blue"])
            if t < 3:
                arrow(ax, (x + 0.5, 1.2), (x + 2.2, 1.2), color=C["green"])
                if mode == "free":
                    arrow(ax, (x + 0.45, 3.1), (x + 2.7, -0.3), color=C["red"], rad=-0.35, ls="--", lw=1.2)
        ax.set_title("TRAINING — teacher forcing (ground truth in ingresso, GT forcing)" if mode == "tf" else
                     "TEST — free running: si campiona e si rimanda in ingresso la PROPRIA predizione",
                     fontsize=10)
        if mode == "tf":
            ax.text(10.8, 1.2, "se sbaglia a t,\nl'ingresso a t+1\nrestà corretto", fontsize=8, color="#555", va="center")
        else:
            ax.text(10.8, 1.2, "un errore a t\ncambia l'ingresso\ndi t+1 e si propaga:\nexposure bias", fontsize=8,
                    color=C["red"], va="center")
    save(fig, "fig_teacher_forcing.svg")


# ----------------------------------------------------------------------------- 10. image captioning
def fig_captioning():
    fig, ax = newax(13, 6.0, (-1.6, 15.2), (-3.0, 5.4))
    box(ax, 0.2, 1.0, 1.7, 1.7, "immagine", fc="#e8e8e8", fs=8)
    arrow(ax, (1.1, 1.0), (1.9, 1.0))
    for i, (w, h) in enumerate([(1.0, 1.4), (0.8, 1.1), (0.6, 0.8)]):
        box(ax, 2.5 + i * 0.95, 1.0, w, h, "", fc="#f0f0f0")
    txt(ax, 3.4, 0.0 - 0.6, "CNN (VGG)\n senza FC-1000 e softmax", fs=8, color="#555")
    arrow(ax, (5.0, 1.0), (5.7, 1.0))
    box(ax, 6.4, 1.0, 1.3, 0.9, "FC-4096", fc=C["hid"], fs=8)
    txt(ax, 6.4, 1.75, "$v$", fs=11, bold=True)
    xs = [9.0, 11.2, 13.4]
    ins = ["<START>", "“straw”", "“hat”"]
    outs = ["“straw”", "“hat”", "<END>"]
    for i, x in enumerate(xs):
        box(ax, x, 0.8, 1.0, 1.0, f"$h_{i}$", fc=C["hid"], fs=10)
        box(ax, x, -1.5, 1.35, 0.9, ins[i], fc=C["inp"], fs=8)
        box(ax, x, 3.1, 1.2, 0.9, outs[i], fc=C["out"], fs=8)
        arrow(ax, (x, -1.04), (x, 0.3))
        arrow(ax, (x, 1.3), (x, 2.65), color=C["blue"])
        if i < 2:
            arrow(ax, (x + 0.52, 0.8), (xs[i + 1] - 0.52, 0.8), color=C["green"])
            arrow(ax, (x + 0.6, 3.1), (xs[i + 1] + 0.1, -1.0), color=C["gray"], rad=-0.45, ls="--", lw=1)
    arrow(ax, (7.05, 1.0), (8.48, 0.95), color=C["magenta"], lw=2)
    txt(ax, 7.8, 1.65, "$W_{ih}$", fs=10, color=C["magenta"])
    txt(ax, 11.2, 4.4, "$h_t=\\tanh(W_{hh}h_{t-1}+W_{xh}x_t+W_{ih}v)$", fs=10)
    txt(ax, 3.4, 3.6, "RETE CONVOLUZIONALE\n(riusata da ImageNet)", fs=9, color=C["red"], bold=True)
    txt(ax, 11.2, -2.7, "RETE RICORRENTE (genera la didascalia)", fs=9, color=C["green"], bold=True)
    save(fig, "fig_captioning.svg")


# ----------------------------------------------------------------------------- 11. RNN multistrato
def fig_multilayer():
    fig, ax = newax(8.4, 7.8, (-1.8, 9.6), (-1.6, 9.4))
    T, L = 5, 3
    for t in range(T):
        x = 1.0 + t * 1.7
        box(ax, x, 0.0, 0.8, 1.0, "", fc=C["inp"])
        arrow(ax, (x, 0.52), (x, 1.58))
        for l in range(L):
            y = 2.1 + l * 2.0
            box(ax, x, y, 0.9, 1.0, f"$h^{{{l + 1}}}_{t + 1}$", fc=C["hid"], fs=8)
            if t < T - 1:
                arrow(ax, (x + 0.47, y), (x + 1.7 - 0.47, y), color=C["green"])
            if l < L - 1:
                arrow(ax, (x, y + 0.52), (x, y + 1.58))
        arrow(ax, (x, 6.12), (x, 7.08), color=C["blue"])
        box(ax, x, 7.6, 0.8, 1.0, "", fc=C["out"])
    arrow(ax, (-1.2, -0.5), (-1.2, 7.8), color="#555")
    txt(ax, -1.45, 3.5, "profondità", fs=9, ha="center").set_rotation(90) if False else None
    ax.text(-1.55, 3.6, "profondità", rotation=90, fontsize=9, va="center")
    arrow(ax, (0.2, -1.1), (8.6, -1.1), color="#555")
    txt(ax, 4.4, -1.45, "tempo", fs=9)
    save(fig, "fig_multilayer.svg")


# ----------------------------------------------------------------------------- 12. geometria dei valori singolari
def fig_svd_geometry():
    th = np.linspace(0, 2 * np.pi, 400)
    circ = np.vstack([np.cos(th), np.sin(th)])
    R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cases = [((0.8, 0.5), "$\\sigma_1,\\sigma_2<1$: contrazione → gradiente svanisce"),
             ((1.3, 0.6), "$\\sigma_1>1>\\sigma_2$: una direzione esplode,\nle altre svaniscono"),
             ((1.3, 1.15), "$\\sigma_1,\\sigma_2>1$: espansione → gradiente esplode")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    Q = R(np.deg2rad(30))
    for ax, ((s1, s2), title) in zip(axs, cases):
        W = Q @ np.diag([s1, s2]) @ Q.T       # matrice simmetrica: valori singolari = autovalori
        ax.plot(circ[0], circ[1], "k--", lw=1, label="$\\|v\\|=1$")
        shades = ["#9fc5e8", "#3d85c6", "#073763"]
        for k in (1, 2, 3):
            e = np.linalg.matrix_power(W, k) @ circ
            ax.plot(e[0], e[1], color=shades[k - 1], lw=1.8, label=f"$W^{k}$ applicata al cerchio")
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal")
        ax.axhline(0, color="#dddddd", lw=0.6)
        ax.axvline(0, color="#dddddd", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Applicare $k$ volte la stessa matrice: il cerchio unitario diventa un'ellisse con semiassi $\\sigma_i^k$",
                 fontsize=10, y=1.02)
    save(fig, "fig_svd_geometry.svg")


# ----------------------------------------------------------------------------- 13. esperimenti sul flusso del gradiente
def jac_norms(gain, T=100, n=64, seed=0, sigma_x=1.0):
    rng = np.random.default_rng(seed)
    W = gain * rng.standard_normal((n, n)) / np.sqrt(n)
    U = rng.standard_normal((n, n)) / np.sqrt(n)
    h = np.zeros(n)
    P = np.eye(n)
    norms = []
    for t in range(T):
        x = sigma_x * rng.standard_normal(n)
        h = np.tanh(W @ h + U @ x)
        J = (1 - h ** 2)[:, None] * W
        P = J @ P
        norms.append(np.linalg.norm(P, 2))
    return np.array(norms)


def fig_grad_flow():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    k = np.arange(0, 101)
    for rho, col in [(0.9, C["red"]), (1.0, "black"), (1.1, C["green"])]:
        axs[0].semilogy(k, rho ** k, color=col, label=f"$\\|W^k\\|_2$ con $\\sigma_i=\\rho={rho}$")
    axs[0].set_xlabel("k = numero di passi indietro nel tempo")
    axs[0].set_ylabel("norma del prodotto (scala log)")
    axs[0].set_title("(a) caso lineare: $W=\\rho Q$, $Q$ ortogonale", fontsize=10)
    axs[0].legend(fontsize=8)
    T = 100
    for g, col in [(0.5, C["red"]), (0.9, C["orange"]), (1.0, "black"), (1.5, C["blue"]), (3.0, C["green"])]:
        axs[1].semilogy(np.arange(1, T + 1), jac_norms(g, T), color=col, label=f"guadagno $g={g}$")
    axs[1].set_xlabel("T − 1 = numero di passi indietro nel tempo")
    axs[1].set_ylabel("$\\|\\partial h_T/\\partial h_1\\|_2$ (scala log)")
    axs[1].set_title("(b) RNN tanh vera: $W_{hh}=g\\,N(0,1/n)$, $n=64$", fontsize=10)
    axs[1].legend(fontsize=8)
    save(fig, "fig_grad_flow.svg")
    for g in (0.5, 0.9, 1.0, 1.5, 3.0):
        nm = jac_norms(g, 100)
        print(f"   g={g}: ||dh_T/dh_1|| a T=10: {nm[9]:.2e}, T=50: {nm[49]:.2e}, T=100: {nm[99]:.2e}")


# ----------------------------------------------------------------------------- 14. cella LSTM
def fig_lstm_cell():
    fig, ax = newax(12.5, 7.4, (-1.2, 11.6), (-2.4, 5.6))
    box(ax, 5.2, 2.2, 9.6, 6.2, fc="#eaf4e4", ec="#93c47d", rounded=True, z=0, lw=1.5)
    # linea cella
    arrow(ax, (-1.0, 4.4), (5.0, 4.4), lw=2, color="#333")
    txt(ax, -0.9, 4.85, "$c_{t-1}$", fs=11, ha="left")
    box(ax, 5.5, 4.4, 0.6, 0.6, "⊙", fc="white", fs=12, rounded=True)
    arrow(ax, (5.8, 4.4), (6.9, 4.4), lw=2, color="#333")
    box(ax, 7.2, 4.4, 0.6, 0.6, "+", fc="white", fs=12, rounded=True)
    arrow(ax, (7.5, 4.4), (11.3, 4.4), lw=2, color="#333")
    txt(ax, 10.2, 4.85, "$c_t$", fs=11)
    # input
    box(ax, 0.6, 0.7, 1.1, 0.7, "stack", fc="white", fs=9)
    arrow(ax, (-1.0, 0.7), (0.05, 0.7))
    txt(ax, -0.9, 1.15, "$h_{t-1}$", fs=11, ha="left")
    arrow(ax, (0.6, -1.7), (0.6, 0.33))
    txt(ax, 0.6, -2.0, "$x_t$", fs=11)
    ax.add_patch(Circle((2.0, 0.7), 0.3, fc="#cccccc", ec="k", zorder=3))
    txt(ax, 2.0, 1.2, "$W$", fs=10)
    arrow(ax, (1.15, 0.7), (1.7, 0.7))
    gates = {"f": (3.7, 3.4, "σ", C["yellow"]), "i": (3.7, 2.4, "σ", "#fce5cd"),
             "g": (3.7, 1.4, "tanh", "#ead1dc"), "o": (3.7, -0.1, "σ", "#d0e0e3")}
    for n, (x, y, a, c) in gates.items():
        box(ax, x, y, 1.0, 0.7, f"{n}: {a}", fc=c, fs=9, rounded=True)
        arrow(ax, (2.3, 0.7), (x - 0.5, y), lw=1.0, color="#777")
    arrow(ax, (4.2, 3.4), (5.35, 4.15), lw=1.3)                    # f -> ⊙
    box(ax, 6.3, 2.0, 0.6, 0.6, "⊙", fc="white", fs=12, rounded=True)
    arrow(ax, (4.2, 2.4), (6.0, 2.1), lw=1.3)
    arrow(ax, (4.2, 1.4), (6.0, 1.9), lw=1.3)
    arrow(ax, (6.3, 2.32), (7.05, 4.1), lw=1.3)
    box(ax, 8.8, 2.8, 1.0, 0.6, "tanh", fc="white", fs=9, rounded=True)
    ax.plot([8.8, 8.8], [4.4, 3.12], color="#333", lw=1.6)
    arrow(ax, (8.8, 4.4), (8.8, 3.12), lw=1.6, color="#333")
    box(ax, 8.8, -0.1, 0.6, 0.6, "⊙", fc="white", fs=12, rounded=True)
    arrow(ax, (8.8, 2.48), (8.8, 0.22), lw=1.6, color="#333")
    arrow(ax, (4.2, -0.1), (8.48, -0.1), lw=1.3)
    arrow(ax, (9.12, -0.1), (11.3, -0.1), lw=2, color="#333")
    txt(ax, 10.3, 0.35, "$h_t$", fs=11)
    txt(ax, 5.2, -1.6,
        "$c_t=f\\odot c_{t-1}+i\\odot g$      $h_t=o\\odot\\tanh(c_t)$      "
        "$(i,f,o,g)=(\\sigma,\\sigma,\\sigma,\\tanh)\\,W\\,[h_{t-1};x_t]$", fs=9.5)
    save(fig, "fig_lstm_cell.svg")


# ----------------------------------------------------------------------------- 15. gradiente: LSTM vs RNN
def fig_lstm_grad():
    k = np.arange(0, 101)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.semilogy(k, 0.5 ** k, color=C["red"], lw=2,
                label="RNN vanilla: $\\prod(\\tanh'\\cdot\\sigma_{W})$ con fattore 0.5")
    for f, col in [(0.9, C["orange"]), (0.99, C["blue"]), (1.0, C["green"])]:
        ax.semilogy(k, f ** k, color=col, label=f"cella LSTM: $\\prod f_t$ con $f_t={f}$")
    ax.set_ylim(1e-32, 5)
    ax.set_xlabel("k = passi indietro nel tempo")
    ax.set_ylabel("fattore moltiplicativo sul gradiente")
    ax.set_title("Percorso diretto del gradiente: matrice+tanh' vs. solo forget gate", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig_lstm_grad.svg")


if __name__ == "__main__":
    fig_seq_configs()
    fig_forecast()
    fig_conv1d()
    fig_causal()
    fig_dilated()
    fig_rf_growth()
    fig_tcn_block()
    fig_rnn_unrolled()
    fig_bptt()
    fig_seq2seq()
    fig_charlm()
    fig_charlm_probs()
    fig_teacher_forcing()
    fig_captioning()
    fig_multilayer()
    fig_svd_geometry()
    fig_grad_flow()
    fig_lstm_cell()
    fig_lstm_grad()
