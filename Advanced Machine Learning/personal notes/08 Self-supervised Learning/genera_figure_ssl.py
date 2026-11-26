#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genera_figure_ssl.py
--------------------
Genera TUTTE le figure SVG usate nella nota 6 (Self-Supervised Learning).
Uso:   python3 genera_figure_ssl.py
       (scrive i file .svg nella STESSA cartella dello script e stampa i numeri calcolati)
Dipendenze: numpy, matplotlib.

Formule: tutte le etichette matematiche sono scritte in sintassi LaTeX ($...$) e rese da
matplotlib con il font Computer Modern ("mathtext.fontset = cm"), quindi NON serve alcuna
installazione di TeX. Comandi non supportati da mathtext (es. \\big, \\text) sono evitati.

Le figure con numeri (rotazioni, permutazioni del puzzle, maschere MAE, costo dell'encoder,
InfoNCE ottimizzata a mano su una circonferenza, matrice di affinita' di SimCLR, temperatura,
EMA di MoCo, collasso, centering/sharpening di DINO, matrici di Gram) sono CALCOLATE qui,
non disegnate a mano. Gli schemi (blocchi, frecce) usano le piccole utilita' qui sotto.
"""
import os, itertools, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
matplotlib.rcParams.update({
    "svg.fonttype": "path",            # le formule diventano tracciati: rendering identico ovunque
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",          # Computer Modern = look LaTeX
    "axes.formatter.use_mathtext": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

# ------------------------------------------------------------------ palette
C = dict(blue="#d6e6ff", blue_d="#2b5fb4", green="#d8f0d6", green_d="#2e7d32",
         red="#fbd5d5", red_d="#b3261e", yellow="#fff1bf", yellow_d="#9a7400",
         gray="#eceff1", gray_d="#455a64", purple="#e8dcf7", purple_d="#6a3fa0",
         orange="#ffe2bd", orange_d="#d84b00", teal="#d0f0ee", teal_d="#00796b",
         white="#ffffff", ink="#1b1b1b", pink="#fbdcec", pink_d="#b0206b")
PAL = [C["blue_d"], C["orange_d"], C["green_d"], C["purple_d"], C["pink_d"], C["teal_d"],
       C["yellow_d"], C["red_d"]]

# ------------------------------------------------------------------ utilita' di disegno
def canvas(W, H, title=None, scale=13.0):
    """Tela in 'unita' dati' W x H (13 unita' = 1 pollice). Ritorna fig, ax."""
    fig = plt.figure(figsize=(W / scale, H / scale))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    if title:
        ax.text(W / 2, H - 2.6, title, ha="center", va="top", fontsize=15, weight="bold", color=C["ink"])
    return fig, ax

def sub(ax, x, y, w, h, **kw):
    """Assi annidati in coordinate della tela (angolo in basso a sinistra, larghezza, altezza)."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return ax.inset_axes([(x - x0) / (x1 - x0), (y - y0) / (y1 - y0), w / (x1 - x0), h / (y1 - y0)], **kw)

def box(ax, x, y, w, h, txt="", fc=None, ec=None, fs=11, lw=1.5, r=1.3, ls="-", tc=None, weight="normal", alpha=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=C["gray"] if fc is None else fc, ec=ec or C["gray_d"], lw=lw, ls=ls, alpha=alpha, zorder=2)
    ax.add_patch(p)
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc or C["ink"],
                weight=weight, zorder=4, linespacing=1.25)
    return p

def arrow(ax, p0, p1, color=None, lw=1.8, style="-|>", ls="-", rad=0.0, ms=14, z=3):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color or C["gray_d"],
                        ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a); return a

def txt(ax, x, y, s, fs=11, color=None, ha="center", va="center", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=fs, color=color or C["ink"], ha=ha, va=va, weight=weight, zorder=5, **kw)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="svg", bbox_inches=None, facecolor="white")
    if os.environ.get("PREVIEW"):
        fig.savefig("/tmp/prev_" + name.replace(".svg", ".png"), dpi=60, facecolor="white")
    plt.close(fig); print("  scritto", name)

def snowflake(ax, x, y, s=2.0):   # simbolo "congelato" (pesi fissi)
    txt(ax, x, y, r"$\ast$", fs=26, color=C["blue_d"])

# ------------------------------------------------------------------ scena sintetica (per tutte le figure con immagini)
def toy_scene(n=56):
    """Immagine RGB asimmetrica (cielo, sole, collina, albero, casetta): la rotazione si 'vede'."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    img = np.zeros((n, n, 3))
    img[...] = np.array([0.55, 0.78, 0.98]) * (1 - 0.35 * yy[..., None]) + 0.05   # cielo (alto = chiaro)
    ground = yy > 0.68 + 0.06 * np.sin(5 * xx)
    img[ground] = np.array([0.36, 0.68, 0.30]) * (0.85 + 0.3 * xx[ground][:, None])
    sun = (xx - 0.80) ** 2 + (yy - 0.20) ** 2 < 0.012
    img[sun] = [1.0, 0.86, 0.15]
    trunk = (abs(xx - 0.25) < 0.025) & (yy > 0.45) & (yy < 0.78)
    img[trunk] = [0.45, 0.28, 0.12]
    crown = (xx - 0.25) ** 2 + (yy - 0.38) ** 2 < 0.02
    img[crown] = [0.12, 0.50, 0.18]
    house = (xx > 0.55) & (xx < 0.78) & (yy > 0.52) & (yy < 0.76)
    img[house] = [0.82, 0.30, 0.25]
    roof = (yy < 0.52) & (yy > 0.52 - (0.115 - abs(xx - 0.665))) & (abs(xx - 0.665) < 0.115)
    img[roof] = [0.35, 0.22, 0.20]
    door = (xx > 0.64) & (xx < 0.69) & (yy > 0.64) & (yy < 0.76)
    img[door] = [0.95, 0.85, 0.5]
    return np.clip(img, 0, 1)

SCENE = toy_scene(56)

def show_img(ax, img, x, y, w, h, ec=None, lw=1.2, cmap=None, vmin=None, vmax=None):
    a = sub(ax, x, y, w, h); a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest" if img.shape[0] < 20 else "bilinear")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(True); s.set_edgecolor(ec or C["gray_d"]); s.set_linewidth(lw)
    return a

def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True); e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

def l2n(z):
    return z / np.linalg.norm(z, axis=-1, keepdims=True)

# ==================================================================================== FIGURE 1
def fig_ssl_paradigm():
    fig, ax = canvas(130, 62, r"Self-supervised learning: pretext task $\rightarrow$ downstream task")
    # pretext
    box(ax, 3, 6, 60, 46, "", fc="#f7faff", ec=C["blue_d"], lw=1.2, ls="--")
    txt(ax, 33, 49, "PRETEXT (molti dati, nessuna etichetta umana)", fs=11, color=C["blue_d"], weight="bold")
    box(ax, 6, 22, 11, 14, "dati\n$x$\nnon etichettati", fc=C["gray"], fs=9)
    box(ax, 22, 22, 11, 14, "genera\n$(\\tilde{x},\\,y)$\n da $x$", fc=C["yellow"], ec=C["yellow_d"], fs=9)
    box(ax, 38, 22, 11, 14, "Encoder\n$f_\\theta$", fc=C["blue"], ec=C["blue_d"], fs=11, weight="bold")
    box(ax, 53, 26, 8, 6, "testa\n$h_\\phi$", fc=C["orange"], ec=C["orange_d"], fs=9)
    arrow(ax, (17, 29), (22, 29)); arrow(ax, (33, 29), (38, 29)); arrow(ax, (49, 29), (53, 29))
    txt(ax, 33, 14, r"$\mathcal{L}_{pretext}=\ell\left(h_\phi(f_\theta(\tilde{x})),\,y\right)$", fs=12)
    txt(ax, 33, 9, "etichette $y$ costruite dai dati stessi", fs=9, color=C["gray_d"])
    # trasferimento
    arrow(ax, (63, 29), (73, 29), lw=3, color=C["green_d"])
    txt(ax, 68, 36, "trasferisci\n$f_\\theta$", fs=10, color=C["green_d"], weight="bold")
    # downstream
    box(ax, 73, 6, 54, 46, "", fc="#f7fff7", ec=C["green_d"], lw=1.2, ls="--")
    txt(ax, 100, 49, "DOWNSTREAM (pochi dati etichettati)", fs=11, color=C["green_d"], weight="bold")
    box(ax, 77, 22, 12, 14, "dati\n$(x,\\,y)$\netichettati", fc=C["gray"], fs=9)
    box(ax, 93, 22, 12, 14, "Encoder\n$f_\\theta$", fc=C["blue"], ec=C["blue_d"], fs=11, weight="bold")
    box(ax, 109, 24, 14, 10, "classificatore,\nregressore, ...", fc=C["green"], ec=C["green_d"], fs=9)
    arrow(ax, (89, 29), (93, 29)); arrow(ax, (105, 29), (109, 29))
    txt(ax, 100, 14, "congelato (linear probing)  oppure  riaddestrato (fine-tuning)", fs=9, color=C["gray_d"])
    txt(ax, 100, 9, r"obiettivo: classificazione, detection, segmentazione...", fs=9, color=C["gray_d"])
    save(fig, "fig_ssl_paradigm.svg")

# ==================================================================================== FIGURE 2
def fig_probe_vs_finetune():
    fig, ax = canvas(130, 60, "Come si valuta una rappresentazione: linear probing vs fine-tuning")
    for k, (name, frozen, col) in enumerate([("Linear probing", True, C["blue_d"]), ("Full fine-tuning", False, C["orange_d"])]):
        x0 = 4 + k * 65
        box(ax, x0, 6, 58, 42, "", fc="white", ec=col, lw=1.4, ls="--")
        txt(ax, x0 + 29, 44, name, fs=13, color=col, weight="bold")
        box(ax, x0 + 3, 20, 10, 12, "$x$", fc=C["gray"], fs=12)
        eb = box(ax, x0 + 18, 17, 20, 18, "Encoder\n$f_\\theta$\n(pre-addestrato)", fc=C["blue"] if frozen else C["orange"],
                 ec=col, fs=10, lw=2.5 if frozen else 1.5)
        box(ax, x0 + 43, 20, 12, 12, "lineare\n$W\\,h+b$", fc=C["green"], ec=C["green_d"], fs=10)
        arrow(ax, (x0 + 13, 26), (x0 + 18, 26)); arrow(ax, (x0 + 38, 26), (x0 + 43, 26))
        if frozen:
            txt(ax, x0 + 28, 39.5, r"pesi $\theta$ FISSI", fs=10, color=C["blue_d"], weight="bold")
            txt(ax, x0 + 29, 12, "si addestra solo il livello lineare:\nmisura la qualita' della rappresentazione", fs=9, color=C["gray_d"])
        else:
            txt(ax, x0 + 28, 39.5, r"$\theta$ aggiornati (gradiente fino all'encoder)", fs=9, color=C["orange_d"], weight="bold")
            arrow(ax, (x0 + 49, 19.5), (x0 + 28, 17), color=C["orange_d"], ls="--", rad=-0.35)
            txt(ax, x0 + 29, 12, "si addestra tutto (anche layer non lineari):\nsfrutta il potenziale del modello sul nuovo task", fs=9, color=C["gray_d"])
    save(fig, "fig_probe_vs_finetune.svg")

# ==================================================================================== FIGURE 3
def fig_rotation():
    fig, ax = canvas(135, 78, r"Pretext: predire la rotazione (Gidaris et al.): classificazione a 4 vie")
    rots = [np.rot90(SCENE, -k) for k in range(4)]      # k=0:0°, 1:90° orario, ...
    names = [r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$270^\circ$"]
    for k in range(4):
        x = 4 + k * 22
        show_img(ax, rots[k], x, 44, 18, 18, ec=PAL[k], lw=2.5)
        txt(ax, x + 9, 40.5, rf"$\mathrm{{rot}}_{{{k}}}(x)$:  {names[k]}", fs=10, color=PAL[k])
        txt(ax, x + 9, 36.5, rf"$y={k}$", fs=11, color=PAL[k], weight="bold")
    txt(ax, 46, 67, "da UNA immagine $x$ senza etichette si ottengono 4 esempi con etichetta gratuita", fs=10.5)
    box(ax, 96, 40, 17, 22, "CNN\n$f_\\theta$", fc=C["blue"], ec=C["blue_d"], fs=14, weight="bold")
    box(ax, 117, 40, 14, 22, "softmax\n4 classi", fc=C["orange"], ec=C["orange_d"], fs=10)
    arrow(ax, (92, 51), (96, 51)); arrow(ax, (113, 51), (117, 51))
    txt(ax, 48, 30, r"$\mathcal{L}=-\dfrac{1}{4}\sum_{k=0}^{3}\log p_\theta\!\left(y=k\mid \mathrm{rot}_k(x)\right)$", fs=15)
    box(ax, 4, 4, 127, 18, "", fc=C["yellow"], ec=C["yellow_d"], lw=1.2)
    txt(ax, 67, 17, "Ipotesi (slide 18): il modello riconosce la rotazione solo se possiede il «senso comune visivo»", fs=10)
    txt(ax, 67, 11, "di come l'oggetto appare «non perturbato»: sole in alto, terreno in basso, albero in piedi.", fs=10)
    txt(ax, 67, 6.8, r"Chance level: $1/4=25\%$.  Gli oggetti dello sfondo ruotati danno indizi (e anche i bordi!): per questo i paper evitano scorciatoie.", fs=8, color=C["gray_d"])
    save(fig, "fig_rotation.svg")

# ==================================================================================== FIGURE 4
def greedy_max_hamming(K, rng):
    perms = list(itertools.permutations(range(9)))
    cand = np.array(perms)[rng.choice(len(perms), 4000, replace=False)]
    chosen = [cand[0]]
    for _ in range(K - 1):
        d = np.array([min((c != s).sum() for s in chosen) for c in cand])
        chosen.append(cand[int(np.argmax(d))])
    return np.array(chosen)

def fig_patch_tasks():
    rng = np.random.default_rng(0)
    fig, ax = canvas(140, 84, "Pretext su patch: posizione relativa e puzzle (jigsaw)")
    # -- sinistra: posizione relativa
    txt(ax, 34, 72, "Posizione relativa (Doersch et al., 2015)", fs=12, weight="bold", color=C["blue_d"])
    a = show_img(ax, SCENE, 8, 28, 36, 36)
    n = SCENE.shape[0]; t = n / 3.0
    for i in range(3):
        for j in range(3):
            a.add_patch(Rectangle((j * t - .5, i * t - .5), t, t, fc="none", ec="white", lw=1.2))
    a.add_patch(Rectangle((t - .5, t - .5), t, t, fc="none", ec=C["red_d"], lw=3))      # patch centrale
    a.add_patch(Rectangle((2 * t - .5, 0 * t - .5), t, t, fc="none", ec=C["green_d"], lw=3))  # vicino (in alto a dx)
    txt(ax, 26, 21, "patch di riferimento (rosso) + un vicino (verde)", fs=9)
    # le 8 classi
    bx, by = 53, 33
    for idx, (di, dj) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]):
        cls = idx + 1
        col = C["green"] if (di, dj) == (0, 2) else C["gray"]
        box(ax, bx + dj * 8.5, by + (2 - di) * 8.5 - 0, 7.5, 7.5, f"{cls}", fc=col, fs=10, r=0.8)
    box(ax, bx + 8.5, by + 8.5, 7.5, 7.5, "ref", fc=C["red"], ec=C["red_d"], fs=9, r=0.8)
    txt(ax, 66, 21, r"classe $y\in\{1,\dots,8\}$ (8 vicini)", fs=9.5)
    txt(ax, 34, 12, r"$\mathcal{L}=-\log p_\theta(y\mid f_\theta(P_c),f_\theta(P_n))$", fs=11)
    # -- destra: jigsaw
    txt(ax, 108, 72, "Jigsaw (Noroozi & Favaro, 2016)", fs=12, weight="bold", color=C["blue_d"])
    perm = np.array([3, 8, 1, 5, 0, 7, 2, 6, 4])
    tiles = [SCENE[(p // 3) * 18:(p // 3 + 1) * 18, (p % 3) * 18:(p % 3 + 1) * 18] for p in range(9)]
    for pos in range(9):
        i, j = divmod(pos, 3)
        a2 = show_img(ax, tiles[perm[pos]], 82 + j * 10.5, 37 + (2 - i) * 10.5, 9.8, 9.8, lw=0.8)
        txt(ax, 82 + j * 10.5 + 4.9, 37 + (2 - i) * 10.5 + 4.9, str(int(perm[pos])), fs=8, color="white", weight="bold")
    txt(ax, 99, 33, "9 tile mescolate (numerate come nell'originale)", fs=9)
    txt(ax, 99, 28.5, r"$\pi=("+",".join(map(str, perm))+r")$", fs=10.5, color=C["orange_d"])
    K = 64
    S = greedy_max_hamming(K, rng)
    dmin = min((S[i] != S[j]).sum() for i in range(K) for j in range(i + 1, K))
    print(f"  [jigsaw] 9! = {math.factorial(9)};  sottoinsieme greedy di {K} permutazioni: distanza di Hamming minima = {dmin}")
    txt(ax, 99, 23, rf"$9!={math.factorial(9):,}$ permutazioni: se ne usa un sottoinsieme di $K$", fs=9.5)
    txt(ax, 99, 19, rf"scelto con distanza di Hamming massima (qui $K={K}$, $d_{{min}}={dmin}$)", fs=9.5)
    txt(ax, 99, 13.5, r"la rete (condivisa sulle 9 tile) predice $\pi\in\{1,\dots,K\}$:  classificazione $K$-way", fs=9.5)
    txt(ax, 99, 8.3, r"$\mathcal{L}=-\log p_\theta(\pi\mid \mathrm{tile}_1,\dots,\mathrm{tile}_9)$", fs=11)
    txt(ax, 70, 2.5, "Entrambi i task richiedono di capire le PARTI e la loro disposizione: serve conoscere cosa sono gli oggetti.", fs=9.5, color=C["gray_d"])
    save(fig, "fig_patch_tasks.svg")

# ==================================================================================== FIGURE 5
def fig_inpainting():
    fig, ax = canvas(140, 76, "Pretext: inpainting (Context Encoders, Pathak et al., 2016)")
    n = SCENE.shape[0]
    M = np.zeros((n, n), bool); M[18:38, 18:38] = True
    masked = SCENE.copy(); masked[M] = [0.55, 0.55, 0.55]
    show_img(ax, masked, 4, 40, 22, 22, ec=C["gray_d"])
    txt(ax, 15, 66, r"$(\mathbf{1}-M)\odot x$", fs=11); txt(ax, 15, 36, "contesto", fs=9, color=C["gray_d"])
    box(ax, 32, 42, 16, 18, "Encoder\n$E$", fc=C["blue"], ec=C["blue_d"], fs=12, weight="bold")
    box(ax, 54, 46, 8, 10, "$z$", fc=C["yellow"], ec=C["yellow_d"], fs=12)
    box(ax, 68, 42, 16, 18, "Decoder\n$D$", fc=C["green"], ec=C["green_d"], fs=12, weight="bold")
    arrow(ax, (26, 51), (32, 51)); arrow(ax, (48, 51), (54, 51)); arrow(ax, (62, 51), (68, 51))
    show_img(ax, SCENE[18:38, 18:38], 92, 44, 14, 14, ec=C["green_d"]); txt(ax, 99, 40, r"$F(\cdot)$ (patch predetto)", fs=9)
    arrow(ax, (84, 51), (92, 51))
    show_img(ax, SCENE, 112, 40, 22, 22, ec=C["gray_d"]); txt(ax, 123, 66, r"$x$ (verita')", fs=11)
    txt(ax, 123, 36, "target = pixel mancanti", fs=9, color=C["gray_d"])
    # loss
    box(ax, 4, 4, 70, 26, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 39, 25.5, "Loss = ricostruzione + avversaria (slide 30)", fs=10.5, weight="bold", color=C["yellow_d"])
    txt(ax, 39, 19.5, r"$\mathcal{L}_{rec}=\left\|M\odot\left(x-F\left((\mathbf{1}-M)\odot x\right)\right)\right\|_2^2$", fs=12)
    txt(ax, 39, 12.5, r"$\mathcal{L}_{adv}=\max_{D_{adv}}\ \mathbb{E}\left[\log D_{adv}(x)+\log\left(1-D_{adv}(F(\cdot))\right)\right]$", fs=11)
    txt(ax, 39, 7, r"$\mathcal{L}=\lambda_{rec}\,\mathcal{L}_{rec}+\lambda_{adv}\,\mathcal{L}_{adv}$", fs=12)
    # perche' serve l'adversarial: L2 -> media
    a = sub(ax, 88, 7, 46, 17)
    xs = np.linspace(-0.5, 1.5, 400)
    pdf = 0.5 * np.exp(-(xs - 0) ** 2 / (2 * 0.06 ** 2)) + 0.5 * np.exp(-(xs - 1) ** 2 / (2 * 0.06 ** 2))
    a.fill_between(xs, pdf, color=C["blue"], ec=C["blue_d"], lw=1.4)
    a.axvline(0.5, color=C["red_d"], lw=2.2, ls="--"); a.set_yticks([]); a.set_xticks([0, 0.5, 1])
    a.set_xticklabels([r"$0$ (scuro)", r"$\mathbb{E}[x]=0.5$", r"$1$ (chiaro)"], fontsize=8)
    a.set_title(r"due completamenti plausibili: la L2 sceglie la MEDIA (sfocato)", fontsize=8.5, pad=4)
    a.annotate("la GAN sceglie un\nmodo (nitido)", xy=(1.0, 0.5), xycoords="data", xytext=(0.74, 0.62), textcoords="axes fraction",
               fontsize=8, color=C["green_d"], arrowprops=dict(arrowstyle="->", color=C["green_d"]))
    a.spines["left"].set_visible(False)
    print("  [inpainting] con due modi equiprobabili {0,1}: minimizzatore L2 = media = 0.5 (errore atteso 0.25 > 0)")
    save(fig, "fig_inpainting.svg")

# ==================================================================================== FIGURE 6
def rgb2lab(rgb):
    a = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = a @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

def fig_colorization():
    fig, ax = canvas(140, 86, r"Pretext: colorizzazione e Split-Brain Autoencoder (spazio $Lab$)")
    lab = rgb2lab(SCENE); L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    print(f"  [colorizzazione] range L=[{L.min():.0f},{L.max():.0f}]  a=[{A.min():.0f},{A.max():.0f}]  b=[{B.min():.0f},{B.max():.0f}]")
    show_img(ax, SCENE, 3, 52, 20, 20, ec=C["gray_d"]); txt(ax, 13, 75, "immagine", fs=10)
    arrow(ax, (24, 62), (30, 62)); txt(ax, 27, 66, "Lab", fs=8, color=C["gray_d"])
    show_img(ax, L, 31, 52, 20, 20, cmap="gray"); txt(ax, 41, 75, r"$L$ (luminanza)", fs=10)
    show_img(ax, A, 55, 52, 20, 20, cmap="RdYlGn_r"); txt(ax, 65, 75, r"canale $a$ (verde$\leftrightarrow$rosso)", fs=9)
    show_img(ax, B, 79, 52, 20, 20, cmap="YlGnBu_r"); txt(ax, 89, 75, r"canale $b$ (blu$\leftrightarrow$giallo)", fs=9)
    txt(ax, 118, 66, "Colorizzazione:\n" r"$\hat{ab}=F_\theta(L)$", fs=11, color=C["blue_d"])
    txt(ax, 118, 55, r"$\mathcal{L}=\ell\left(\hat{ab},ab\right)$", fs=11)
    txt(ax, 118, 49.5, "etichette = i canali $a,b$ dell'immagine stessa", fs=8, color=C["gray_d"])
    # split brain
    box(ax, 3, 3, 134, 40, "", fc="#fafcff", ec=C["blue_d"], lw=1.2, ls="--")
    txt(ax, 70, 40, "Split-brain autoencoder (Zhang et al., 2017): predizioni tra canali (slide 36)", fs=11, weight="bold", color=C["blue_d"])
    box(ax, 8, 22, 14, 11, r"$X_1=L$", fc=C["gray"], fs=11); box(ax, 8, 8, 14, 11, r"$X_2=ab$", fc=C["pink"], ec=C["pink_d"], fs=11)
    box(ax, 30, 22, 16, 11, r"rete $F_1$", fc=C["blue"], ec=C["blue_d"], fs=11); box(ax, 30, 8, 16, 11, r"rete $F_2$", fc=C["orange"], ec=C["orange_d"], fs=11)
    box(ax, 56, 22, 18, 11, r"$\hat{X}_2=\hat{ab}$", fc=C["pink"], ec=C["pink_d"], fs=11); box(ax, 56, 8, 18, 11, r"$\hat{X}_1=\hat{L}$", fc=C["gray"], fs=11)
    for y in (27.5, 13.5): arrow(ax, (22, y), (30, y)); arrow(ax, (46, y), (56, y))
    txt(ax, 91, 27.5, r"$\mathcal{L}_1=\ell(\hat{ab},ab)$", fs=11); txt(ax, 91, 13.5, r"$\mathcal{L}_2=\ell(\hat{L},L)$", fs=11)
    txt(ax, 110, 20.5, r"$\mathcal{L}=\mathcal{L}_1+\mathcal{L}_2$" "\n\n" r"feature finali:" "\n" r"$\left[F_1(X_1)\,;\,F_2(X_2)\right]$ concatenate", fs=9.5, color=C["gray_d"])
    txt(ax, 70, 5.2, r"idea: ogni sotto-rete impara una rappresentazione che spiega i canali che NON vede $\Rightarrow$ nessuna scorciatoia", fs=9)
    save(fig, "fig_colorization.svg")

# ==================================================================================== FIGURE 7
def fig_video_coloring():
    fig, ax = canvas(140, 84, "Colorizzazione video: l'attention come «puntatore» ai colori (Vondrick et al., 2018)")
    rng = np.random.default_rng(3)
    G = 6
    col = np.zeros((G, G, 3)); col[...] = [0.55, 0.78, 0.55]
    ref_obj = np.zeros((G, G), bool); ref_obj[1:4, 1:3] = True
    col[ref_obj] = [0.85, 0.20, 0.20]; col[4:, :] = [0.50, 0.70, 0.95]
    tgt_obj = np.zeros((G, G), bool); tgt_obj[1:4, 2:4] = True      # oggetto spostato di 1 a destra
    # feature: [classe(3 dim), posizione normalizzata] + rumore
    cls_vec = {"obj": np.array([1.0, 0, 0]), "bg": np.array([0, 1.0, 0]), "sky": np.array([0, 0, 1.0])}
    def feats(obj, shift):
        F = np.zeros((G, G, 5))
        for i in range(G):
            for j in range(G):
                c = "obj" if obj[i, j] else ("sky" if i >= 4 else "bg")
                F[i, j, :3] = 2.4 * cls_vec[c]; F[i, j, 3:] = 0.15 * np.array([i, j - shift])
        return F + 0.15 * rng.standard_normal(F.shape)
    Fr, Ft = feats(ref_obj, 0), feats(tgt_obj, 1)
    q = (2, 3)                                                    # pixel del target da colorare (dentro l'oggetto)
    fq = Ft[q]; fr = Fr.reshape(-1, 5)
    tau = 1.0
    attn = softmax(fr @ fq / tau); pred = attn @ col.reshape(-1, 3)
    gt = np.array([0.85, 0.20, 0.20])
    print(f"  [video] massa di attention sui pixel-oggetto della reference: {attn.reshape(G,G)[ref_obj].sum():.3f};  colore predetto = {np.round(pred,2)}  (vero {gt})")
    show_img(ax, col, 4, 36, 30, 30, ec=C["gray_d"], lw=2); txt(ax, 19, 69, "frame di riferimento (a colori)", fs=10)
    gray = np.full((G, G), 0.55); gray[tgt_obj] = 0.35; gray[4:, :] = 0.8
    a = show_img(ax, gray, 46, 36, 30, 30, cmap="gray", vmin=0, vmax=1, ec=C["gray_d"], lw=2)
    a.add_patch(Rectangle((q[1] - .5, q[0] - .5), 1, 1, fc="none", ec=C["orange_d"], lw=3))
    txt(ax, 61, 69, "frame target (grigio): pixel $j$", fs=10); 
    a = show_img(ax, attn.reshape(G, G), 88, 36, 30, 30, cmap="magma", ec=C["gray_d"], lw=2)
    txt(ax, 103, 69, r"attention $A_{ji}$ sul riferimento", fs=10)
    arrow(ax, (76, 51), (88, 51))
    box(ax, 122, 42, 14, 18, "", fc=pred.clip(0, 1), ec=C["ink"], lw=2); txt(ax, 129, 64, r"$\hat{c}_j$ predetto", fs=9)
    box(ax, 122, 20, 14, 14, "", fc=gt, ec=C["ink"], lw=1); txt(ax, 129, 37, "vero $c_j$", fs=9)
    arrow(ax, (118, 51), (122, 51))
    box(ax, 4, 3, 100, 26, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 54, 24.5, "Formule (slide 43-46)", fs=10.5, weight="bold", color=C["yellow_d"])
    txt(ax, 54, 18.3, r"$A_{ji}=\dfrac{\exp\left(f_j^{\top}f_i/\tau\right)}{\sum_{i'}\exp\left(f_j^{\top}f_{i'}/\tau\right)}$   (softmax sui pixel $i$ del riferimento)", fs=11)
    txt(ax, 54, 10.5, r"$\hat{c}_j=\sum_i A_{ji}\,c_i$  (somma pesata dei colori)     $\mathcal{L}=\sum_j \ell\left(\hat{c}_j,\,c_j\right)$", fs=11)
    txt(ax, 112, 10, "nessuna etichetta:\nil colore vero e' il\nvideo originale", fs=8.5, color=C["gray_d"])
    save(fig, "fig_video_coloring.svg")

# ==================================================================================== FIGURE 8
def fig_mae():
    rng = np.random.default_rng(7)
    fig, ax = canvas(150, 84, "Masked Autoencoder (MAE): maschera 75%, encoder solo sui patch visibili")
    P = 14; n = P * P; keep = int(round(0.25 * n))
    vis = np.zeros(n, bool); vis[rng.choice(n, keep, replace=False)] = True; vis = vis.reshape(P, P)
    print(f"  [MAE] griglia {P}x{P}={n} patch: visibili {vis.sum()} ({vis.mean():.0%}), mascherati {(~vis).sum()} ({(~vis).mean():.0%})")
    big = np.kron(SCENE[:P * 4:4, :P * 4:4], np.ones((1, 1, 1)))  # 14x14 reduction
    sc = toy_scene(P * 6)
    tile = lambda i, j: sc[i * 6:(i + 1) * 6, j * 6:(j + 1) * 6]
    masked = np.full_like(sc, 0.85)
    for i in range(P):
        for j in range(P):
            if vis[i, j]: masked[i * 6:(i + 1) * 6, j * 6:(j + 1) * 6] = tile(i, j)
    show_img(ax, sc, 3, 52, 22, 22); txt(ax, 14, 77, r"$x$ in $196$ patch", fs=10)
    a = show_img(ax, masked, 3, 24, 22, 22); txt(ax, 14, 20.5, r"visibili (25%)", fs=10, color=C["green_d"])
    for k in range(1, P):
        a.axhline(k * 6 - .5, color="white", lw=.3); a.axvline(k * 6 - .5, color="white", lw=.3)
    arrow(ax, (14, 51), (14, 47), color=C["red_d"]); txt(ax, 24, 49.5, "75% mascherati", fs=8, color=C["red_d"], ha="left")
    box(ax, 33, 28, 15, 14, "token\nvisibili\n$N_v=49$", fc=C["green"], ec=C["green_d"], fs=9)
    box(ax, 55, 24, 22, 22, "Encoder\n(ViT grande)\nsolo su $N_v$", fc=C["blue"], ec=C["blue_d"], fs=11, weight="bold")
    arrow(ax, (25, 35), (33, 35)); arrow(ax, (48, 35), (55, 35))
    # token + mask tokens
    box(ax, 85, 24, 8, 22, "", fc="white", ec=C["gray_d"], lw=1)
    for k in range(8):
        v = k in (0, 2, 3, 6)
        box(ax, 86, 25 + k * 2.6, 6, 2.2, "", fc=C["green"] if v else C["red"], ec=C["green_d"] if v else C["red_d"], lw=.8, r=.3)
    txt(ax, 89, 49, r"$196$ token", fs=8.5); txt(ax, 89, 20.5, "visibili +\n[MASK] condiviso", fs=8, color=C["red_d"])
    arrow(ax, (77, 35), (85, 35))
    box(ax, 99, 24, 20, 22, "Decoder\n(leggero)\n+ pos. enc.", fc=C["orange"], ec=C["orange_d"], fs=10.5, weight="bold")
    arrow(ax, (93, 35), (99, 35))
    show_img(ax, sc, 126, 28, 18, 18, ec=C["green_d"]); txt(ax, 135, 49, r"$\hat{x}$ (pixel)", fs=10)
    arrow(ax, (119, 35), (126, 37))
    # loss
    box(ax, 33, 4, 112, 14, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 89, 12.5, r"$\mathcal{L}=\dfrac{1}{|\mathcal{M}|}\sum_{p\in\mathcal{M}}\left\|\hat{x}_p-x_p\right\|_2^2$   solo sui patch MASCHERATI ($\mathcal{M}$)", fs=11.5)
    txt(ax, 89, 6.5, "il decoder serve solo in pre-training: dopo si butta e si tiene l'encoder (design asimmetrico)", fs=8.5, color=C["gray_d"])
    txt(ax, 118, 60, "Ratio alto $\\Rightarrow$ task difficile (non basta\ninterpolare i vicini) e encoder economico", fs=9.5, color=C["gray_d"])
    save(fig, "fig_mae.svg")

# ==================================================================================== FIGURE 9
def fig_mae_cost():
    fig = plt.figure(figsize=(11.5, 4.8))
    N, d = 196, 1024
    r = np.linspace(0, 0.95, 200); Nv = (1 - r) * N
    lin = Nv; att = Nv ** 2 / N                           # costo relativo per blocco (parte lineare ~ N, attention ~ N^2)
    flops = (12 * d * d * Nv + 2 * Nv ** 2 * d)
    rel = flops / flops[0]
    ax1 = fig.add_subplot(121)
    ax1.plot(r * 100, rel, color=C["blue_d"], lw=2.6)
    k = np.argmin(abs(r - 0.75)); ax1.scatter([75], [rel[k]], s=90, color=C["red_d"], zorder=5)
    ax1.annotate(rf"$r=75\%$: costo $\approx {rel[k]:.2f}\times$", (75, rel[k]), (50, 0.78), fontsize=12,
                 arrowprops=dict(arrowstyle="->", color=C["red_d"]), color=C["red_d"])
    ax1.set_xlabel(r"rapporto di masking $r$ [%]", fontsize=12); ax1.set_ylabel("costo relativo di un blocco dell'encoder", fontsize=11)
    ax1.set_title(r"costo $\approx 12\,d^2 N_v+2\,N_v^2 d$,  $N_v=(1-r)N$", fontsize=12); ax1.grid(alpha=.3)
    ax2 = fig.add_subplot(122)
    ax2.bar([0, 1], [N, Nv[k]], color=[C["gray"], C["green"]], ec=[C["gray_d"], C["green_d"]], lw=2, width=.55)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["ViT standard\n(tutti i token)", "MAE encoder\n(solo visibili)"], fontsize=11)
    for x, v in [(0, N), (1, Nv[k])]: ax2.text(x, v + 4, rf"$N={int(round(v))}$ token", ha="center", fontsize=12)
    ax2.set_ylabel("token processati dall'encoder", fontsize=11); ax2.set_ylim(0, 230)
    ax2.set_title(rf"attention: $\left(0.25\right)^2=\dfrac{{1}}{{16}}$ del costo quadratico", fontsize=12)
    fig.suptitle("Perche' l'encoder MAE puo' essere molto grande", fontsize=15, weight="bold", y=1.0)
    fig.tight_layout(); print(f"  [MAE costo] r=75%: token {int(round(Nv[k]))}/{N}; costo relativo del blocco = {rel[k]:.3f}")
    save(fig, "fig_mae_cost.svg")

# ==================================================================================== contrastive: utilita'
def nt_xent(Z, tau, pos_of):
    """NT-Xent / InfoNCE di SimCLR su Z (2N x d, normalizzata). pos_of[i] = indice del positivo di i."""
    S = Z @ Z.T / tau; n = len(Z)
    np.fill_diagonal(S, -np.inf)
    lse = np.log(np.exp(S).sum(1))
    return float(np.mean(lse - S[np.arange(n), pos_of]))

# ==================================================================================== FIGURE 10
def fig_contrastive_intuition():
    rng = np.random.default_rng(1)
    N = 5; tau = 0.5; pos_of = np.array([i ^ 1 for i in range(2 * N)])
    th = (rng.uniform(-0.55, 0.55, 2 * N) + 0.4)                     # tutte le viste partono in un arco stretto
    def Zf(t): return np.stack([np.cos(t), np.sin(t)], 1)
    def lossf(t): return nt_xent(Zf(t), tau, pos_of)
    hist = [lossf(th)]; th0 = th.copy()
    for step in range(400):
        g = np.zeros_like(th)
        for k in range(len(th)):
            e = np.zeros_like(th); e[k] = 1e-5
            g[k] = (lossf(th + e) - lossf(th - e)) / 2e-5
        th = th - 0.6 * g; hist.append(lossf(th))
    Z0, Z1 = Zf(th0), Zf(th)
    gap = lambda Z: float(np.mean([np.arccos(np.clip(Z[i] @ Z[i ^ 1], -1, 1)) for i in range(2 * N)]))
    print(f"  [InfoNCE 2D] loss iniziale {hist[0]:.3f} -> finale {hist[-1]:.3f}; angolo medio tra positivi {np.degrees(gap(Z0)):.1f} deg -> {np.degrees(gap(Z1)):.1f} deg; "
          f"riferimento 'a caso' log(2N-1)={math.log(2*N-1):.3f}")
    fig, ax = canvas(145, 76, "Apprendimento contrastivo: attrarre i positivi, respingere i negativi")
    # schema concettuale
    box(ax, 3, 41, 42, 29, "", fc="#fafcff", ec=C["blue_d"], lw=1.2, ls="--")
    txt(ax, 24, 66, "Idea (slide 65-69)", fs=10.5, weight="bold", color=C["blue_d"])
    ax.add_patch(Circle((24, 52), 4.0, fc=C["blue"], ec=C["blue_d"], lw=2, zorder=4)); txt(ax, 24, 52, "$x$", fs=11)
    ax.add_patch(Circle((36, 58), 3.4, fc=C["green"], ec=C["green_d"], lw=2, zorder=4)); txt(ax, 36, 58, "$x^+$", fs=10)
    ax.add_patch(Circle((12, 46), 3.4, fc=C["red"], ec=C["red_d"], lw=2, zorder=4)); txt(ax, 12, 46, "$x^-$", fs=10)
    ax.add_patch(Circle((32, 44.5), 3.4, fc=C["red"], ec=C["red_d"], lw=2, zorder=4)); txt(ax, 32, 44.5, "$x^-$", fs=10)
    arrow(ax, (31, 56), (28.2, 54), color=C["green_d"], lw=2.4); txt(ax, 36, 62.5, "attrai", fs=8.5, color=C["green_d"])
    arrow(ax, (20, 50), (15.5, 47.5), color=C["red_d"], lw=2.4, style="<|-"); arrow(ax, (27.5, 50), (30.5, 47.5), color=C["red_d"], lw=2.4, style="<|-")
    txt(ax, 10, 51.5, "respingi", fs=8.5, color=C["red_d"])
    txt(ax, 24, 43.0, "", fs=8)
    txt(ax, 24, 40.5 + 0, "", fs=8)
    # panels sul cerchio
    for k, (Z, ttl, xo) in enumerate([(Z0, r"inizio: tutto addensato", 50), (Z1, r"dopo 400 passi di gradiente", 92)]):
        a = sub(ax, xo, 36, 40, 33, aspect="equal") if False else sub(ax, xo, 36, 40, 33)
        a.set_aspect("equal"); a.axis("off")
        a.add_patch(Circle((0, 0), 1, fc="white", ec=C["gray_d"], lw=1.6))
        for i in range(N):
            c = PAL[i % len(PAL)]
            a.plot([Z[2 * i, 0], Z[2 * i + 1, 0]], [Z[2 * i, 1], Z[2 * i + 1, 1]], color=c, lw=1.4, alpha=.6)
            a.scatter(Z[2 * i:2 * i + 2, 0], Z[2 * i:2 * i + 2, 1], s=70, color=c, ec="white", zorder=5)
        a.set_xlim(-1.35, 1.35); a.set_ylim(-1.35, 1.35)
        a.set_title(ttl, fontsize=9.5)
    # curva loss
    a = sub(ax, 98 + 36 + 2 - 6 - 40, 3, 0.1, 0.1) if False else sub(ax, 52, 5, 36, 24)
    a.plot(hist, color=C["blue_d"], lw=2.2); a.axhline(math.log(2 * N - 1), color=C["gray_d"], ls=":", lw=1.3)
    a.text(150, math.log(2 * N - 1) + .04, r"$\log(2N-1)$: scelta a caso", fontsize=8, color=C["gray_d"])
    a.set_xlabel("passo", fontsize=9); a.set_ylabel(r"$\mathcal{L}$", fontsize=10); a.set_title("la loss scende", fontsize=9.5)
    a.tick_params(labelsize=8); a.grid(alpha=.3)
    box(ax, 94, 5, 48, 26, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 118, 26.5, "Cosa vediamo", fs=10, weight="bold", color=C["yellow_d"])
    txt(ax, 118, 17, "• ogni colore = una coppia di viste\n  (stessa immagine, 2 augmentation)\n• la loss avvicina i due punti dello stesso\n  colore e allontana quelli di colori diversi\n• i punti si dispongono sul cerchio (uniformita')\n  con coppie vicine (allineamento)", fs=8.2, ha="center")
    save(fig, "fig_contrastive_intuition.svg")

# ==================================================================================== FIGURE 11
def fig_infonce():
    fig = plt.figure(figsize=(13.2, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.34)
    s = np.array([2.0, 0.6, -0.4, 1.1, 0.2, -1.0, 0.7])           # score s(f(x), f(.)) ; indice 0 = positivo
    p = softmax(s); L = -np.log(p[0])
    names = [r"$x^+$"] + [rf"$x^-_{j}$" for j in range(1, len(s))]
    cols = [C["green_d"]] + [C["red_d"]] * (len(s) - 1)
    ax = fig.add_subplot(gs[0, 0]); ax.bar(range(len(s)), s, color=cols, alpha=.85); ax.set_xticks(range(len(s))); ax.set_xticklabels(names, fontsize=10)
    ax.axhline(0, color="k", lw=.8); ax.set_title(r"1) punteggi $s\left(f(x),f(\cdot)\right)$ (qui scelti a mano)", fontsize=11); ax.set_ylabel("score", fontsize=10)
    ax = fig.add_subplot(gs[0, 1]); ax.bar(range(len(s)), p, color=cols, alpha=.85); ax.set_xticks(range(len(s))); ax.set_xticklabels(names, fontsize=10)
    ax.set_title(r"2) softmax $\Rightarrow$ $p_k=e^{s_k}/\sum_{k'}e^{s_{k'}}$", fontsize=11)
    ax.text(0, p[0] + .015, rf"$p_+={p[0]:.2f}$", ha="center", fontsize=11, color=C["green_d"]); ax.set_ylim(0, p[0] + .12)
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.text(0.5, 0.88, "3) cross-entropy a $N$ classi", ha="center", fontsize=12, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.62, r"$\mathcal{L}=-\mathbb{E}\left[\log\dfrac{e^{s^+}}{e^{s^+}+\sum_{j=1}^{N-1}e^{s_j^-}}\right]$", ha="center", fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.30, rf"qui $\mathcal{{L}}=-\log p_+={L:.3f}$" "\n" r"il positivo e' l'unica classe «corretta»", ha="center", fontsize=11.5, transform=ax.transAxes, color=C["blue_d"])
    # bound
    Ns = np.logspace(0.3, 4, 200)
    ax = fig.add_subplot(gs[1, 0]); ax.semilogx(Ns, np.log(Ns), color=C["blue_d"], lw=2.5)
    for n in (8, 256, 8192): ax.scatter([n], [math.log(n)], s=45, color=C["red_d"], zorder=5); ax.text(n, math.log(n) - 0.9, f"{n}", ha="center", fontsize=9, color=C["red_d"])
    ax.set_xlabel(r"$N$ (positivo + negativi)", fontsize=10); ax.set_ylabel(r"$\log N$", fontsize=11)
    ax.set_title(r"4) bound: $I\left[f(x),f(x^+)\right]-\log N\geq -\mathcal{L}$", fontsize=10.5); ax.grid(alpha=.3)
    ax.text(0.04, 0.80, r"$I\geq \log N-\mathcal{L}$:" "\n" r"con $N$ piccolo il bound" "\n" r"non puo' superare $\log N$", transform=ax.transAxes, fontsize=9.5)
    # temperatura
    ax = fig.add_subplot(gs[1, 1]); cosv = np.array([0.9, 0.6, 0.5, 0.2, -0.1, -0.4, -0.7])
    for tau, c in zip([1.0, 0.5, 0.2, 0.05], [C["gray_d"], C["blue_d"], C["orange_d"], C["red_d"]]):
        ax.plot(range(len(cosv)), softmax(cosv / tau), "o-", color=c, lw=2, label=rf"$\tau={tau}$")
    ax.set_xticks(range(len(cosv))); ax.set_xticklabels([r"$x^+$"] + [rf"$x^-_{j}$" for j in range(1, len(cosv))], fontsize=9)
    ax.set_title(r"5) temperatura: $s=\cos(z_i,z_j)/\tau$", fontsize=11); ax.legend(fontsize=9); ax.set_ylabel(r"$p_k$", fontsize=10); ax.grid(alpha=.3)
    for tau in (1.0, 0.5, 0.2, 0.05): print(f"  [InfoNCE] tau={tau}: p_+={softmax(cosv/tau)[0]:.3f}, loss={-np.log(softmax(cosv/tau)[0]):.3f}")
    # loss casuale
    ax = fig.add_subplot(gs[1, 2]); Nb = np.array([2, 8, 32, 128, 512, 2048, 8192])
    ax.semilogx(Nb, np.log(2 * Nb - 1), "s-", color=C["purple_d"], lw=2.4)
    ax.set_xlabel(r"batch size $N$ (SimCLR: $2N$ viste)", fontsize=10); ax.set_ylabel(r"$\log(2N-1)$", fontsize=10); ax.grid(alpha=.3)
    ax.set_title("6) loss «a caso» al punto di partenza", fontsize=11)
    ax.text(0.04, 0.78, "con piu' negativi il task\ndiventa piu' difficile e\nil gradiente piu' informativo", transform=ax.transAxes, fontsize=9.5)
    fig.suptitle("InfoNCE = cross-entropy di un classificatore a $N$ vie (slide 69-74)", fontsize=15, weight="bold", y=0.985)
    save(fig, "fig_infonce.svg")

# ==================================================================================== FIGURE 12
def blur(img, k=3):
    pad = np.pad(img, ((k, k), (k, k), (0, 0)), mode="edge"); out = np.zeros_like(img)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): out += pad[k + dy:k + dy + img.shape[0], k + dx:k + dx + img.shape[1]]
    return out / (2 * k + 1) ** 2

def fig_simclr_pipeline():
    fig, ax = canvas(150, 80, "SimCLR: due viste augmentate della stessa immagine = coppia positiva")
    show_img(ax, SCENE, 3, 30, 20, 20); txt(ax, 13, 53, r"immagine $x$", fs=10)
    v1 = SCENE[6:46, 2:42]; v2 = np.clip(blur(SCENE[10:50, 14:54][:, ::-1]) * np.array([1.15, 0.85, 0.75]), 0, 1)
    show_img(ax, v1, 33, 44, 16, 16, ec=C["blue_d"], lw=2.2); txt(ax, 41, 63, r"$\tilde{x}_{2k-1}=t(x)$", fs=10, color=C["blue_d"])
    show_img(ax, v2, 33, 14, 16, 16, ec=C["orange_d"], lw=2.2); txt(ax, 41, 33, r"$\tilde{x}_{2k}=t'(x)$", fs=10, color=C["orange_d"])
    arrow(ax, (23, 44), (33, 52)); arrow(ax, (23, 36), (33, 22)); txt(ax, 24, 56, r"$t\sim\mathcal{T}$", fs=9, color=C["blue_d"]); txt(ax, 24, 21, r"$t'\sim\mathcal{T}$", fs=9, color=C["orange_d"])
    txt(ax, 13, 25, "crop casuale,\ncolor distortion,\nblur", fs=8, color=C["gray_d"])
    for y, col, cd, name in [(52, C["blue"], C["blue_d"], "1"), (22, C["orange"], C["orange_d"], "2")]:
        box(ax, 56, y - 5, 17, 10, r"$f(\cdot)$" "\nencoder", fc=col, ec=cd, fs=9.5); arrow(ax, (49, y), (56, y), color=cd)
        box(ax, 80, y - 5, 10, 10, rf"$h$", fc=C["gray"], fs=11); arrow(ax, (73, y), (80, y), color=cd)
        box(ax, 96, y - 5, 17, 10, r"$g(\cdot)$" "\nprojector", fc=C["purple"], ec=C["purple_d"], fs=9.5); arrow(ax, (90, y), (96, y), color=cd)
        box(ax, 119, y - 5, 10, 10, r"$z$", fc=C["yellow"], ec=C["yellow_d"], fs=11); arrow(ax, (113, y), (119, y), color=cd)
    txt(ax, 85, 59.5, "rappresentazione (si tiene)", fs=8, color=C["gray_d"]); txt(ax, 124, 59.5, "embedding", fs=8, color=C["gray_d"])
    arrow(ax, (124, 27), (124, 47), color=C["green_d"], style="<|-|>", lw=2.6)
    txt(ax, 140, 37, r"massimizza" "\n" r"$s_{2k-1,2k}$" "\n" r"$=\dfrac{z_{2k-1}^{\top}z_{2k}}{\|z_{2k-1}\|\|z_{2k}\|}$", fs=9.5, color=C["green_d"])
    box(ax, 3, 3, 144, 9, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 75, 7.5, r"dopo il pre-training si BUTTA $g(\cdot)$ e si usa $h=f(x)$ per il downstream (slide 88: $g$ conserva informazione utile in $h$)", fs=9.5)
    save(fig, "fig_simclr_pipeline.svg")

# ==================================================================================== FIGURE 13
def fig_simclr_matrix():
    rng = np.random.default_rng(5)
    N, d, tau = 4, 6, 0.5
    cen = l2n(rng.standard_normal((N, d)))
    Z = l2n(np.repeat(cen, 2, axis=0) + 0.35 * rng.standard_normal((2 * N, d)))
    S = Z @ Z.T; n = 2 * N; pos_of = np.array([i ^ 1 for i in range(n)])
    Sm = S / tau; Sm2 = Sm.copy(); np.fill_diagonal(Sm2, -np.inf); P = softmax(Sm2, axis=1)
    li = -np.log(P[np.arange(n), pos_of]); Lval = li.mean()
    print(f"  [SimCLR] N={N} (2N={n} viste), tau={tau}: loss per riga = {np.round(li,3).tolist()};  L = {Lval:.4f}")
    fig = plt.figure(figsize=(15.5, 6.4)); gs = fig.add_gridspec(1, 3, wspace=0.62)
    lab = [rf"${i}$" for i in range(n)]
    ax = fig.add_subplot(gs[0, 0]); im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n): ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(S[i, j]) > .6 else "k")
        ax.add_patch(Rectangle((pos_of[i] - .5, i - .5), 1, 1, fc="none", ec=C["green_d"], lw=3))
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fc="none", ec="k", lw=1.5, hatch="///"))
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    ax.set_title(r"matrice di affinita' $s_{i,j}=\cos(z_i,z_j)$" "\n(verde = positivo, tratteggio = auto-confronto escluso)", fontsize=10.5)
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i != j: ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if P[i, j] > .5 else "k")
        ax.add_patch(Rectangle((pos_of[i] - .5, i - .5), 1, 1, fc="none", ec=C["green_d"], lw=3))
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    ax.set_title(r"softmax per riga: $p_{i,j}=\dfrac{e^{s_{i,j}/\tau}}{\sum_{k\neq i}e^{s_{i,k}/\tau}}$", fontsize=10.5)
    ax = fig.add_subplot(gs[0, 2]); ax.barh(range(n), li, color=C["blue_d"], alpha=.85); ax.invert_yaxis()
    ax.set_yticks(range(n)); ax.set_yticklabels([rf"${i}\to{pos_of[i]}$" for i in range(n)], fontsize=10)
    for i in range(n): ax.text(li[i] + .01, i, f"{li[i]:.3f}", va="center", fontsize=9)
    ax.set_title(rf"$\ell(i,j)=-\log p_{{i,j}}$,  $\mathcal{{L}}=\dfrac{{1}}{{2N}}\sum\ell={Lval:.3f}$", fontsize=11)
    ax.set_xlabel("loss della riga   (etichetta  $i\\to$ indice del positivo)", fontsize=10)
    fig.suptitle(rf"SimCLR in mini-batch: $2N={n}$ viste, la label di ogni riga e' l'indice del suo positivo (slide 84-85)", fontsize=13.5, weight="bold", y=1.0)
    fig.subplots_adjust(top=0.80)
    save(fig, "fig_simclr_matrix.svg")

# ==================================================================================== FIGURE 14
def fig_moco():
    fig, ax = canvas(150, 92, "MoCo: encoder a momento e coda FIFO di negativi")
    show_img(ax, SCENE, 3, 52, 16, 16); txt(ax, 11, 71, "$x$", fs=11)
    show_img(ax, SCENE[6:46, 2:42], 26, 64, 13, 13, ec=C["blue_d"], lw=2); txt(ax, 32.5, 80, "$x_q$", fs=10)
    show_img(ax, np.clip(blur(SCENE[10:50, 14:54][:, ::-1]) * np.array([1.15, .85, .75]), 0, 1), 26, 43, 13, 13, ec=C["orange_d"], lw=2); txt(ax, 32.5, 39.5, "$x_k$", fs=10)
    arrow(ax, (19, 62), (26, 69)); arrow(ax, (19, 56), (26, 50))
    box(ax, 46, 63, 20, 16, r"$f_q$" "\n(query)", fc=C["blue"], ec=C["blue_d"], fs=11, weight="bold")
    box(ax, 46, 42, 20, 16, r"$f_k$" "\n(key)", fc=C["orange"], ec=C["orange_d"], fs=11, weight="bold")
    arrow(ax, (39, 70.5), (46, 71)); arrow(ax, (39, 50), (46, 50))
    txt(ax, 76, 71, r"$q$", fs=13, color=C["blue_d"]); arrow(ax, (66, 71), (73, 71)); 
    txt(ax, 76, 50, r"$k^+$", fs=13, color=C["orange_d"]); arrow(ax, (66, 50), (73, 50))
    txt(ax, 56, 36, "NO gradiente\n(stop-grad / detach)", fs=8.5, color=C["red_d"])
    # coda
    box(ax, 98, 28, 14, 46, "", fc="white", ec=C["gray_d"], lw=1.6)
    for k in range(9):
        box(ax, 99.5, 29.5 + k * 4.9, 11, 4.0, rf"$k_{{-{k+1}}}$", fc=C["red"], ec=C["red_d"], fs=7.5, lw=.8, r=.3)
    txt(ax, 105, 78, "coda FIFO\ndi $K$ chiavi", fs=9.5, weight="bold")
    arrow(ax, (82, 50), (97, 70), rad=-.1, color=C["orange_d"]); txt(ax, 90, 62, "enqueue", fs=8.5, color=C["orange_d"])
    arrow(ax, (105, 28), (105, 22), color=C["gray_d"]); txt(ax, 118, 24, "dequeue (le piu'\nvecchie escono)", fs=8.5, color=C["gray_d"])
    box(ax, 124, 40, 24, 26, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 136, 62, "logit", fs=9.5, weight="bold", color=C["yellow_d"])
    txt(ax, 136, 51, r"$\left[\,q^{\top}k^+\,;\,q^{\top}k_1^-,\dots,q^{\top}k_K^-\right]/\tau$", fs=8.3)
    txt(ax, 136, 43.5, r"$\Rightarrow$ cross-entropy, label $=0$", fs=8.8)
    arrow(ax, (112, 55), (124, 55), color=C["red_d"])
    arrow(ax, (80, 74), (132, 66.5), color=C["blue_d"], rad=-0.28); txt(ax, 112, 85, r"$q^{\top}k$ con tutte le chiavi", fs=9, color=C["blue_d"])
    # formule
    box(ax, 3, 3, 144, 16, "", fc="#f7faff", ec=C["blue_d"], lw=1.2)
    txt(ax, 75, 14, r"aggiornamento a momento:   $\theta_k\leftarrow m\,\theta_k+(1-m)\,\theta_q$     (solo $\theta_q$ riceve gradiente)", fs=12)
    txt(ax, 75, 7.2, r"i negativi sono le chiavi della coda: dimensione $K$ SEPARATA dal batch $\Rightarrow$ molti negativi senza batch enorme", fs=9.5, color=C["gray_d"])
    # EMA plot
    a = sub(ax, 77, 13, 0.1, 0.1) if False else None
    save(fig, "fig_moco.svg")

def fig_moco_momentum():
    rng = np.random.default_rng(2)
    T = 600; thq = np.cumsum(0.15 * rng.standard_normal(T)) + 0.002 * np.arange(T)   # parametro "query" che si muove a scatti (SGD rumoroso)
    fig = plt.figure(figsize=(11.5, 4.6)); ax = fig.add_subplot(121)
    ax.plot(thq, color=C["blue_d"], lw=1.2, alpha=.8, label=r"$\theta_q$ (aggiornato da SGD)")
    out = {}
    for m, c in zip([0.9, 0.99, 0.999], [C["green_d"], C["orange_d"], C["red_d"]]):
        thk = np.zeros(T); thk[0] = thq[0]
        for t in range(1, T): thk[t] = m * thk[t - 1] + (1 - m) * thq[t]
        out[m] = thk; ax.plot(thk, color=c, lw=2.2, label=rf"$\theta_k$, $m={m}$")
    ax.set_xlabel("passo di training", fontsize=11); ax.set_ylabel("un parametro", fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax.set_title(r"$\theta_k\leftarrow m\theta_k+(1-m)\theta_q$: media mobile esponenziale", fontsize=11.5)
    ax2 = fig.add_subplot(122)
    ms = np.array([0.9, 0.99, 0.999, 0.9999]); ax2.bar(range(4), 1 / (1 - ms), color=[C["green_d"], C["orange_d"], C["red_d"], C["purple_d"]], alpha=.85)
    ax2.set_yscale("log"); ax2.set_xticks(range(4)); ax2.set_xticklabels([rf"$m={m}$" for m in ms], fontsize=10)
    for i, m in enumerate(ms): ax2.text(i, 1 / (1 - m) * 1.15, rf"$\approx{int(round(1/(1-m)))}$ passi", ha="center", fontsize=10)
    ax2.set_ylabel(r"memoria $\approx 1/(1-m)$", fontsize=11); ax2.set_title("quanto «ricorda» l'encoder delle chiavi", fontsize=11.5); ax2.grid(alpha=.3, axis="y")
    ax2.set_ylim(1, 3e4)
    print("  [MoCo EMA] memoria effettiva 1/(1-m): " + ", ".join(f"m={m}: {1/(1-m):.0f}" for m in ms) +
          f";  dev.std. di theta_k attorno a theta_q: " + ", ".join(f"m={m}: {np.std(out[m]-thq):.2f}" for m in out))
    fig.suptitle("Perche' il momentum stabilizza le chiavi della coda", fontsize=14, weight="bold", y=1.0)
    fig.tight_layout(); save(fig, "fig_moco_momentum.svg")

# ==================================================================================== FIGURE 16
def _arch(ax, x0, kind, title, oy=17):
    """Schema compatto di un'architettura siamese: 2 rami, loss in alto (oy = offset verticale)."""
    box(ax, x0, oy, 33, 74, "", fc="white", ec=C["gray_d"], lw=1.2, ls="--")
    txt(ax, x0 + 16.5, oy + 70, title, fs=13, weight="bold", color=C["blue_d"])
    xl, xr = x0 + 3, x0 + 20
    w, h = 10, 7
    show_img(ax, SCENE[6:46, 2:42], xl, oy + 4, 10, 10, ec=C["blue_d"], lw=1.6)
    show_img(ax, np.clip(SCENE[10:50, 14:54][:, ::-1] * np.array([1.15, .85, .75]), 0, 1), xr, oy + 4, 10, 10, ec=C["orange_d"], lw=1.6)
    txt(ax, xl + 5, oy + 1.8, r"$x_1$", fs=10); txt(ax, xr + 5, oy + 1.8, r"$x_2$", fs=10)
    mom = kind in ("moco", "byol")
    fcR = C["orange"] if mom else C["blue"]; ecR = C["orange_d"] if mom else C["blue_d"]
    ey, gy, py, ly = oy + 20, oy + 32, oy + 44, oy + 57
    box(ax, xl, ey, w, h, r"$f$", fc=C["blue"], ec=C["blue_d"], fs=12); box(ax, xr, ey, w, h, r"$f_m$" if mom else r"$f$", fc=fcR, ec=ecR, fs=12)
    box(ax, xl, gy, w, h, r"$g$", fc=C["purple"], ec=C["purple_d"], fs=12); box(ax, xr, gy, w, h, r"$g_m$" if mom else r"$g$", fc=C["purple"], ec=C["purple_d"], fs=12)
    for xx in (xl, xr):
        arrow(ax, (xx + 5, oy + 14.2), (xx + 5, ey - .1), lw=1.4); arrow(ax, (xx + 5, ey + h), (xx + 5, gy - .1), lw=1.4)
    top_l = top_r = gy + h
    if kind in ("byol", "simsiam"):
        box(ax, xl, py, w, h, r"$p$", fc=C["pink"], ec=C["pink_d"], fs=12); arrow(ax, (xl + 5, top_l), (xl + 5, py - .1), lw=1.4); top_l = py + h
    if kind in ("moco", "byol", "simsiam"):
        box(ax, xr - 1.2, ey - 2, w + 2.4, gy + h + 2 - (ey - 2), "", fc="none", ec=C["red_d"], lw=1.3, ls=":")
        txt(ax, xr + 5, ey + h + 2.8, "sg" if kind == "simsiam" else "no grad", fs=7.5, color=C["red_d"], weight="bold", bbox=dict(fc="white", ec="none", pad=0.6))
    lbl = {"simclr": "InfoNCE\n(negativi = batch)", "moco": "InfoNCE\n(negativi = coda)", "byol": r"$\left\|\bar z_1-\bar h_2\right\|^2$" "\n(no negativi)", "simsiam": r"$\mathcal{D}(z_1,\mathrm{sg}(h_2))$" "\n(no negativi)"}[kind]
    box(ax, x0 + 3, ly + 2, 27, 10, lbl, fc=C["yellow"], ec=C["yellow_d"], fs=8.5, lw=1.2)
    arrow(ax, (xl + 5, top_l), (xl + 5, ly + 1.9), lw=1.4, color=C["green_d"]); arrow(ax, (xr + 5, top_r), (xr + 5, ly + 1.9), lw=1.4, color=C["green_d"])
    if kind == "moco":
        box(ax, xr, py, w, 8, "coda $K$", fc=C["red"], ec=C["red_d"], fs=8, r=.6)

def fig_architectures():
    fig, ax = canvas(160, 108, "Dal contrastivo al non-contrastivo: SimCLR, MoCo v2, BYOL, SimSiam")
    for k, (kind, ttl) in enumerate([("simclr", "SimCLR"), ("moco", "MoCo v2"), ("byol", "BYOL"), ("simsiam", "SimSiam")]):
        _arch(ax, 3 + k * 39, kind, ttl, oy=17)
    cols = ["negativi", "momentum", "predictor", "stop-grad"]
    M = {"SimCLR": [1, 0, 0, 0], "MoCo v2": [1, 1, 0, 1], "BYOL": [0, 1, 1, 1], "SimSiam": [0, 0, 1, 1]}
    for k, name in enumerate(M):
        x0 = 3 + k * 39
        for c, nm in enumerate(cols):
            on = M[name][c]
            box(ax, x0 + c * 8.3, 5, 7.7, 6, ("sì" if on else "no"), fc=C["green"] if on else C["gray"], ec=C["green_d"] if on else C["gray_d"], fs=8.5, r=.6, lw=1, weight="bold" if on else "normal")
            txt(ax, x0 + c * 8.3 + 3.85, 2.4, nm, fs=6.3, color=C["gray_d"])
    save(fig, "fig_architectures.svg")

# ==================================================================================== FIGURE 17
def _erank(Z):
    Zc = Z - Z.mean(0); ev = np.linalg.eigvalsh(Zc.T @ Zc / len(Z)); ev = np.clip(ev, 1e-12, None); p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def _train_collapse(use_neg, steps=1500, seed=0, lr=1.0, tau=0.5, n=128, d=8):
    """Encoder lineare con bias z = W [x;1]. Con solo positivi la soluzione banale e' z ~ b (costante)."""
    rng = np.random.default_rng(seed)
    sig = np.linspace(1.0, 0.35, d)
    W = 0.5 * rng.standard_normal((d, d + 1)); hist = []
    ones = np.ones((n, 1))
    for t in range(steps):
        x = rng.standard_normal((n, d)) * sig
        x1 = np.hstack([x + 0.5 * rng.standard_normal((n, d)), ones]); x2 = np.hstack([x + 0.5 * rng.standard_normal((n, d)), ones])
        X = np.vstack([x1, x2]); Z = X @ W.T; nz = np.linalg.norm(Z, axis=1, keepdims=True); Zh = Z / nz
        if use_neg:
            S = Zh @ Zh.T / tau; np.fill_diagonal(S, -np.inf); P = softmax(S, axis=1)
            pos = np.concatenate([np.arange(n, 2 * n), np.arange(0, n)]); D = P.copy(); D[np.arange(2 * n), pos] -= 1
            G = (D + D.T) @ Zh / tau / (2 * n)
        else:
            G = np.zeros_like(Zh); a, b = Zh[:n], Zh[n:]
            G[:n] = -b / n; G[n:] = -a / n                                   # d(-cos)/d zhat (media sul batch)
        dZ = (G - (G * Zh).sum(1, keepdims=True) * Zh) / nz; W = W - lr * dZ.T @ X
        if t % 25 == 0 or t == steps - 1:
            xt = np.hstack([rng.standard_normal((512, d)) * sig, np.ones((512, 1))]); Zt = xt @ W.T; Zt = Zt / np.linalg.norm(Zt, axis=1, keepdims=True)
            hist.append((t, _erank(Zt), float(Zt.std(0).mean()), float(np.linalg.norm(Zt.mean(0)))))
    return np.array(hist)

def fig_collapse():
    hA = _train_collapse(False); hB = _train_collapse(True)
    print(f"  [collasso] solo positivi: std {hA[0,2]:.3f}->{hA[-1,2]:.3f}, |media z| {hA[0,3]:.3f}->{hA[-1,3]:.3f};  "
          f"con negativi (InfoNCE): std {hB[0,2]:.3f}->{hB[-1,2]:.3f}, |media z| {hB[0,3]:.3f}->{hB[-1,3]:.3f}")
    fig = plt.figure(figsize=(12.5, 4.8)); ax = fig.add_subplot(121)
    ax.plot(hA[:, 0], hA[:, 2], color=C["red_d"], lw=2.8, label=r"solo positivi: $\mathcal{L}=-\cos(z_1,z_2)$")
    ax.plot(hB[:, 0], hB[:, 2], color=C["green_d"], lw=2.8, label="con negativi (InfoNCE)")
    ax.set_xlabel("passo di training", fontsize=11); ax.set_ylabel(r"dev. std. media di $z/\|z\|$ sul batch", fontsize=10.5)
    ax.set_title(r"se $\to 0$: tutti gli embedding coincidono", fontsize=11.5); ax.legend(fontsize=9.5); ax.grid(alpha=.3); ax.set_ylim(0, 0.42)
    ax = fig.add_subplot(122)
    ax.plot(hA[:, 0], hA[:, 3], color=C["red_d"], lw=2.8); ax.plot(hB[:, 0], hB[:, 3], color=C["green_d"], lw=2.8)
    ax.set_xlabel("passo di training", fontsize=11); ax.set_ylabel(r"$\left\|\mathbb{E}\left[z/\|z\|\right]\right\|$", fontsize=11)
    ax.set_title(r"$=1$: collasso totale su un solo punto;  $\approx0$: embedding ben distribuiti", fontsize=11); ax.grid(alpha=.3); ax.set_ylim(0, 1.05)
    fig.suptitle("Senza negativi la soluzione banale e' una costante (encoder lineare con bias, $d=8$; slide 100-102)", fontsize=13, weight="bold", y=1.0)
    fig.tight_layout(); save(fig, "fig_collapse.svg")

# ==================================================================================== FIGURE 18
def fig_byol_simsiam():
    fig, ax = canvas(150, 86, "BYOL e SimSiam: come evitare il collasso senza negativi")
    # sinistra: scenario banale
    box(ax, 3, 36, 66, 42, "", fc="#fff8f8", ec=C["red_d"], lw=1.2, ls="--")
    txt(ax, 36, 74, "Se togliamo solo i negativi (slide 101)", fs=11, weight="bold", color=C["red_d"])
    txt(ax, 36, 66, r"$f=f_m=\mathbf{0}\ \Rightarrow\ y_1=y_2=0$", fs=12)
    txt(ax, 36, 59, r"$g=g_m=\mathrm{random}\ \Rightarrow\ h_1=h_2=c$ (costante)", fs=12)
    txt(ax, 36, 52, r"$\mathcal{L}=\|h_1-h_2\|^2=0$   banale!", fs=12, color=C["red_d"])
    txt(ax, 36, 43.5, "la rete minimizza la loss ignorando completamente l'input:\nrappresentazioni inutili («bad representations»)", fs=9.5, color=C["gray_d"])
    # destra: simbolo
    box(ax, 76, 36, 71, 42, "", fc="#f7fff8", ec=C["green_d"], lw=1.2, ls="--")
    txt(ax, 111, 74, "Come BYOL lo evita (slide 102)", fs=11, weight="bold", color=C["green_d"])
    txt(ax, 111, 66, r"1) predictor $p$ (architettura ASIMMETRICA):  $z_1=p(h_1)$", fs=10.5)
    txt(ax, 111, 59, r"2) encoder e projector del target aggiornati a MOMENTO:", fs=10.5)
    txt(ax, 111, 52.5, r"$\xi\leftarrow m\,\xi+(1-m)\,\theta$", fs=12)
    txt(ax, 111, 44.5, "SimSiam: toglie il momentum e usa solo\nstop-gradient + predictor (slide 105-107)", fs=9.5, color=C["gray_d"])
    # formule loss
    box(ax, 3, 4, 144, 28, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 75, 27.5, "Notazione delle slide:  $y=f(x)$,  $h=g(y)$,  $z_1=p(h_1)$,  target $h_2=g_m(f_m(x_2))$", fs=10.5)
    txt(ax, 75, 19.2, r"BYOL:  $\mathcal{L}=\left\|\dfrac{z_1}{\|z_1\|}-\dfrac{h_2}{\|h_2\|}\right\|_2^2=2-2\,\dfrac{\langle z_1,h_2\rangle}{\|z_1\|\,\|h_2\|}$    (simmetrizzata scambiando $x_1,x_2$)", fs=11.5)
    txt(ax, 75, 9.2, r"SimSiam:  $\mathcal{L}=\dfrac{1}{2}\,\mathcal{D}(z_1,\mathrm{sg}(h_2))+\dfrac{1}{2}\,\mathcal{D}(z_2,\mathrm{sg}(h_1))$,   $\mathcal{D}(a,b)=-\dfrac{a}{\|a\|}\cdot\dfrac{b}{\|b\|}$", fs=11.5)
    save(fig, "fig_byol_simsiam.svg")

# ==================================================================================== FIGURE 19
def fig_dino():
    rng = np.random.default_rng(4)
    B, K = 6, 8
    bias = np.zeros(K); bias[0] = 3.0
    logits = bias + 1.0 * rng.standard_normal((B, K))
    def H(p): return float(-(p * np.log(p + 1e-12)).sum(-1).mean())
    cases = []
    c = (logits - 0) / 0.1;        cases.append(("1) nessun centering,\n" r"$\tau_t=0.1$ (sharp)", softmax(c)))
    cm = logits - logits.mean(0);  cases.append(("2) solo centering,\n" r"$\tau_t=1$ (no sharpening)", softmax(cm / 1.0)))
    cases.append(("3) centering + sharpening\n" r"$\tau_t=0.1$ (DINO)", softmax(cm / 0.1)))
    res = []
    for _, P in cases: res.append((H(P), float(-(P.mean(0) * np.log(P.mean(0) + 1e-12)).sum())))
    print("  [DINO] log K = %.3f;  (H per-campione, H marginale) -> " % math.log(K) + "; ".join(f"caso{i+1}: ({a:.2f}, {b:.2f})" for i, (a, b) in enumerate(res)))
    fig = plt.figure(figsize=(13.2, 8.4)); gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1], hspace=0.42, wspace=0.28)
    axd = fig.add_subplot(gs[0, :]); axd.set_xlim(0, 150); axd.set_ylim(0, 52); axd.axis("off")
    ax = axd
    box(ax, 2, 20, 8, 8, "$x$", fc=C["gray"], fs=12)
    box(ax, 17, 33, 12, 8, "view 1\n$x_1$", fc=C["blue"], ec=C["blue_d"], fs=9); box(ax, 17, 7, 12, 8, "view 2\n$x_2$", fc=C["orange"], ec=C["orange_d"], fs=9)
    arrow(ax, (10, 26), (17, 37)); arrow(ax, (10, 22), (17, 11))
    box(ax, 36, 31, 20, 12, r"student $g_{\theta_s}$", fc=C["blue"], ec=C["blue_d"], fs=10.5, weight="bold"); box(ax, 36, 5, 20, 12, r"teacher $g_{\theta_t}$", fc=C["orange"], ec=C["orange_d"], fs=10.5, weight="bold")
    arrow(ax, (29, 37), (36, 37)); arrow(ax, (29, 11), (36, 11))
    box(ax, 63, 31, 20, 12, r"softmax$(\cdot/\tau_s)$", fc=C["gray"], fs=9.5); box(ax, 63, 5, 20, 12, r"centering $-c$" "\n" r"softmax$(\cdot/\tau_t)$", fc=C["gray"], fs=9)
    arrow(ax, (56, 37), (63, 37)); arrow(ax, (56, 11), (63, 11))
    txt(ax, 86, 37, r"$P_s$", fs=13, color=C["blue_d"]); txt(ax, 86, 11, r"$P_t$", fs=13, color=C["orange_d"]); arrow(ax, (83, 37), (84.2, 37), lw=0); 
    arrow(ax, (46, 18.5), (46, 29.5), color=C["red_d"], style="<|-", lw=2.2, ls="--"); txt(ax, 51, 24, "EMA", fs=10, color=C["red_d"], weight="bold", ha="left")
    txt(ax, 46, 1.8, "stop-grad (sg)", fs=8.5, color=C["red_d"])
    box(ax, 95, 12, 52, 28, "", fc="#fffdf2", ec=C["yellow_d"], lw=1.2)
    txt(ax, 121, 36, r"$\mathcal{L}=-\sum_{k}P_t(x_2)_k\,\log P_s(x_1)_k$", fs=12)
    txt(ax, 121, 29.3, r"$P_s=\mathrm{softmax}(g_{\theta_s}(x_1)/\tau_s)$", fs=10)
    txt(ax, 121, 23.8, r"$P_t=\mathrm{softmax}\left((g_{\theta_t}(x_2)-c)/\tau_t\right)$", fs=10)
    txt(ax, 121, 18.3, r"$\theta_t\leftarrow\lambda\,\theta_t+(1-\lambda)\,\theta_s$,   $c\leftarrow m\,c+(1-m)\,\overline{g_{\theta_t}}$", fs=10)
    txt(ax, 121, 14.2, r"(loss simmetrizzata: $\frac{1}{2}[H(t_1,s_2)+H(t_2,s_1)]$)", fs=8.5, color=C["gray_d"])
    for k, (ttl, P) in enumerate(cases):
        a = fig.add_subplot(gs[1, k]); a.imshow(P, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        a.set_xticks(range(K)); a.set_xticklabels(range(K), fontsize=8); a.set_yticks(range(B)); a.set_yticklabels([rf"$x_{b+1}$" for b in range(B)], fontsize=8)
        a.set_xlabel("dimensione $k$ (prototipi)", fontsize=9)
        a.set_title(ttl + "\n" + rf"$H$ per-campione$={res[k][0]:.2f}$,  $H$ marginale$={res[k][1]:.2f}$" , fontsize=9.5)
    fig.suptitle(r"DINO: self-distillation senza etichette; centering + sharpening evitano il collasso", fontsize=14, weight="bold", y=0.985)
    fig.text(0.5, 0.005, r"1): tutti i campioni puntano alla stessa dimensione (collasso su una dimensione: $H_{marg}$ bassa).  2): tutte le uscite quasi uniformi (collasso uniforme, $H\approx\log K=%.2f$).  3): ognuno diverso e acuto." % math.log(K), ha="center", fontsize=9, color=C["gray_d"])
    save(fig, "fig_dino.svg")

# ==================================================================================== FIGURE 20
def fig_dinov3():
    rng = np.random.default_rng(11)
    n, d = 24, 12
    grp = np.repeat(np.arange(4), n // 4)
    cen = l2n(rng.standard_normal((4, d)))
    XG = l2n(cen[grp] + 0.30 * rng.standard_normal((n, d)))                 # teacher "di Gram": patch strutturati per oggetto
    XS = l2n(XG + 0.9 * rng.standard_normal((n, d)))                       # student dopo training lungo: patch rumorosi / locali degradati
    XA = l2n(XG + 0.25 * rng.standard_normal((n, d)))                      # student con Gram anchoring
    GG, GS, GA = XG @ XG.T, XS @ XS.T, XA @ XA.T
    LS, LA = np.linalg.norm(GS - GG) ** 2, np.linalg.norm(GA - GG) ** 2
    print(f"  [Gram] n={n} patch, d={d}:  L_Gram(student drifted)={LS:.2f}  ->  L_Gram(con ancoraggio)={LA:.2f}")
    fig = plt.figure(figsize=(14.5, 7.2)); gs = fig.add_gridspec(2, 4, height_ratios=[1.0, .7], hspace=.5, wspace=.25)
    for k, (G, ttl) in enumerate([(GG, r"$X_GX_G^{\top}$: Gram teacher" "\n(geometria di riferimento)"), (GS, r"$X_SX_S^{\top}$: dopo training lungo" "\n(struttura persa)"),
                                  (GA, r"$X_SX_S^{\top}$: con Gram anchoring" "\n(struttura preservata)")]):
        a = fig.add_subplot(gs[0, k]); a.imshow(G, cmap="RdBu_r", vmin=-1, vmax=1); a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=9.3)
        if k == 1: a.set_xlabel(rf"$\mathcal{{L}}_{{Gram}}=\|X_SX_S^\top-X_GX_G^\top\|_F^2={LS:.1f}$", fontsize=10, color=C["red_d"])
        if k == 2: a.set_xlabel(rf"$\mathcal{{L}}_{{Gram}}={LA:.1f}$", fontsize=10, color=C["green_d"])
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    ax.text(0.0, 0.95, "Idea (slide 115)", fontsize=11, weight="bold", va="top")
    ax.text(0.0, 0.80, r"patch L2-normalizzati" "\n" r"$\Rightarrow$ la matrice di Gram contiene" "\n" r"le sole SIMILARITA' coseno tra patch" "\n\n" r"si ancora la geometria, non le feature:" "\n" r"i patch possono ancora «muoversi»," "\n" r"ma le relazioni locali restano coerenti.", fontsize=9.5, va="top")
    ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 150); ax.set_ylim(0, 30); ax.axis("off")
    steps = [("1  Osserva", r"training lungo: accuracy globale $\uparrow$," "\n" r"coerenza locale dei patch $\downarrow$", C["red"], C["red_d"]),
             ("2  Ancora", r"$\mathcal{L}_{Gram}=\|X_SX_S^{\top}-X_GX_G^{\top}\|_F^2$" "\n" r"$X_S$: student, $X_G$: vecchio «Gram teacher»", C["yellow"], C["yellow_d"]),
             ("3  Ripara", r"ADE20K (lineare, alta res.): $50.3\to55.7$ mIoU" "\n" r"ImageNet-1k lineare: $88.2\to88.0$ (invariato)", C["green"], C["green_d"])]
    for k, (t, b, fc, ec) in enumerate(steps):
        x0 = 2 + k * 49; box(ax, x0, 3, 45, 24, "", fc=fc, ec=ec, lw=1.4)
        ax.text(x0 + 22.5, 22.5, t, ha="center", va="center", fontsize=11.5, weight="bold", color=ec); ax.text(x0 + 22.5, 11.5, b, ha="center", va="center", fontsize=9.3)
        if k < 2: ax.annotate("", (x0 + 48.5, 15), (x0 + 45.4, 15), arrowprops=dict(arrowstyle="-|>", color=C["gray_d"], lw=2))
    fig.suptitle("DINOv3: Gram anchoring preserva la geometria dei patch (slide 113-116)", fontsize=14, weight="bold", y=0.985)
    save(fig, "fig_dinov3.svg")

# ==================================================================================== FIGURE 21
def fig_landscape():
    rows = ["Rotation / Jigsaw /\nInpainting / Colorization", "MAE", "SimCLR", "MoCo v2", "BYOL", "SimSiam", "DINO"]
    cols = ["tipo di\nsegnale", "negativi", "momentum\n/ EMA", "predictor", "stop-grad", "batch\ngrande", "come evita\nil collasso"]
    T = [["pretext\n(trasformazione)", "—", "—", "—", "—", "no", "n/a (task con\nlabel proprie)"],
         ["ricostruzione\n(pixel mascherati)", "—", "—", "—", "—", "no", "n/a (target = pixel)"],
         ["contrastivo\n(InfoNCE)", "sì (batch)", "no", "no", "no", "sì", "negativi"],
         ["contrastivo\n(InfoNCE)", "sì (coda)", "sì", "no", "sì (chiavi)", "no", "negativi (coda)"],
         ["non contrastivo\n(regressione)", "no", "sì", "sì", "sì", "no", "predictor + momentum"],
         ["non contrastivo\n(regressione)", "no", "no", "sì", "sì", "no", "predictor + stop-grad"],
         ["self-distillation\n(cross-entropy)", "no", "sì (EMA)", "no", "sì", "n.d.", "centering + sharpening"]]
    fig = plt.figure(figsize=(14.5, 7.4)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.86]); ax.axis("off")
    ax.set_xlim(0, 9.75); ax.set_ylim(0, len(rows) + 1)
    colw = [1.8] + [1.0] * 5 + [1.25]; xs = np.cumsum([0] + colw) + 1.6
    for c, t in enumerate(cols): ax.text(xs[c] + colw[c] / 2, len(rows) + .45, t, ha="center", va="center", fontsize=10, weight="bold", color=C["blue_d"])
    fam = [C["yellow"], C["yellow"], C["blue"], C["blue"], C["green"], C["green"], C["purple"]]
    for r, name in enumerate(rows):
        y = len(rows) - 1 - r
        ax.add_patch(Rectangle((0, y + .06), 1.55, .88, fc=fam[r], ec="white", lw=2)); ax.text(.78, y + .5, name, ha="center", va="center", fontsize=9.5, weight="bold")
        for c in range(len(cols)):
            v = T[r][c]; on = v.startswith("sì"); off = v in ("no",); dash = v in ("—", "n.d.")
            fc = "#d8f0d6" if on else ("#eceff1" if (off or dash) else "#ffffff")
            if c in (0, 6): fc = "#ffffff"
            ax.add_patch(Rectangle((xs[c], y + .06), colw[c] - 0.05, .88, fc=fc, ec="#cfd8dc", lw=1.2))
            ax.text(xs[c] + colw[c] / 2 - .025, y + .5, v, ha="center", va="center", fontsize=9, color=C["green_d"] if on else C["ink"], weight="bold" if on else "normal")
    ax.text(0.78, len(rows) + .45, "famiglia", ha="center", va="center", fontsize=10, weight="bold", color=C["blue_d"])
    fig.suptitle("Mappa dei metodi SSL della lezione: ingredienti e meccanismo anti-collasso", fontsize=14.5, weight="bold", y=0.97)
    save(fig, "fig_landscape.svg")

if __name__ == "__main__":
    print("Genero le figure della nota 6 (SSL):")
    fig_ssl_paradigm(); fig_probe_vs_finetune(); fig_rotation(); fig_patch_tasks()
    fig_inpainting(); fig_colorization(); fig_video_coloring(); fig_mae(); fig_mae_cost()
    fig_contrastive_intuition(); fig_infonce(); fig_simclr_pipeline(); fig_simclr_matrix(); fig_moco(); fig_moco_momentum()
    fig_architectures(); fig_collapse(); fig_byol_simsiam(); fig_dino(); fig_dinov3(); fig_landscape()
