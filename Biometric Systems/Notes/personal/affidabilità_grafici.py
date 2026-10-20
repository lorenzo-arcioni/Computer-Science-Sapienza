#!/usr/bin/env python3
# =====================================================================
#  Grafici per "Biometria: Zoo, Decidability, SRR, Template update"
#  Stile unico (palette, font, griglia) + formule in LaTeX (mathtext).
#
#  Uso:
#     python grafici_biometria.py            # genera tutte le figure
#     python grafici_biometria.py fig8 fig11 # solo alcune
#     python grafici_biometria.py --list     # elenco figure
#  Output: cartella ./img/  (SVG + PNG)
#  Dipendenze: numpy, matplotlib (nessuna installazione LaTeX richiesta:
#  le formule usano il motore mathtext di matplotlib, font Computer Modern).
# =====================================================================
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

OUT = Path("img")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# PALETTE E TEMA (unici per tutte le figure)
# ---------------------------------------------------------------------
INK, MUTED, GRID, PAPER = "#111827", "#6b7280", "#e5e7eb", "#ffffff"
GEN = "#0d9488"      # genuini / affidabile      (teal)
IMP = "#4f46e5"      # impostori / informazione   (indigo)
FR = "#f59e0b"       # false rejection            (ambra)
FA = "#e11d48"       # false acceptance           (rosa-rosso)
SLATE = "#64748b"    # neutro
SKY = "#0ea5e9"
VIOLET = "#8b5cf6"
SOFT = {GEN: "#ccfbf1", IMP: "#e0e7ff", FR: "#fef3c7", FA: "#ffe4e6", SLATE: "#f1f5f9"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f8fafc", "#c7d2fe", "#6366f1", "#312e81"])
DIV = LinearSegmentedColormap.from_list("div", [FA, "#ffffff", IMP])

mpl.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 10,
    "legend.frameon": False, "lines.solid_capstyle": "round",
    "svg.fonttype": "path", "savefig.dpi": 200,
})

PHI = np.vectorize(lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2))))
PHIINV = None


def phi(x):
    return PHI(np.asarray(x, dtype=float)) if np.ndim(x) else float(PHI(x))


def phi_inv(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


REG = {}


def figure(name):
    def deco(fn):
        REG[name] = fn
        return fn
    return deco


def new_fig(w, h, title, subtitle=None, **kw):
    fig = plt.figure(figsize=(w, h), layout="constrained", **kw)
    head = 0.95 if subtitle else 0.65          # altezza intestazione in pollici
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - head / h), w_pad=0.04, h_pad=0.04)
    fig.text(0.012, 1 - 0.12 / h, title, fontsize=16, fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.012, 1 - 0.55 / h, subtitle, fontsize=10.5, color=MUTED, va="top", ha="left")
    return fig


def set_bottom(fig, b):
    w, h = fig.get_size_inches()
    top = fig.get_layout_engine().get()["rect"][3] if "rect" in fig.get_layout_engine().get() else 0.9
    fig.get_layout_engine().set(rect=(0, b, 1, top - b))


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("ok", name)


def clean(ax, grid="y", despine=True):
    ax.grid(False)
    if grid in ("y", "both"):
        ax.grid(axis="y")
    if grid in ("x", "both"):
        ax.grid(axis="x")
    return ax


def note(ax, x, y, s, color=INK, size=10.5, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va, **kw)


def pill(ax, x, y, s, color, size=10, ha="left", **kw):
    return ax.text(x, y, s, color=color, fontsize=size, fontweight="bold", ha=ha, va="center",
                   bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6", fc=SOFT.get(color, "#f1f5f9"), ec="none"), **kw)


def card(ax, x, y, w, h, text, color, size=10.5, align="center", coords="data", weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % (min(w, h) * 0.12),
                       fc=SOFT.get(color, "#f1f5f9"), ec=color, lw=1.4, transform=ax.transData if coords == "data" else ax.transAxes)
    ax.add_patch(p)
    ha = {"center": "center", "left": "left"}[align]
    tx = x + w / 2 if align == "center" else x + w * 0.05
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=INK, fontweight=weight, linespacing=1.5)


def canvas(fig, spec=None, W=100, H=100):
    """Asse 'libero' con coordinate 0..W, 0..H (y verso l'alto) per figure schematiche."""
    ax = fig.add_subplot(spec) if spec is not None else fig.add_subplot(111)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    return ax


def arrow(ax, p, q, color=INK, lw=1.8, style="-|>", ms=12):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def legend_handles(items, **kw):
    from matplotlib.patches import Patch
    return [Patch(fc=c, ec=c, alpha=0.9, label=l) for c, l in items]


# =====================================================================
# 20 — Matrice s(j,k): goat, lamb, wolf
# =====================================================================
@figure("fig10_matrice_lupi_agnelli")
def f10():
    rng = np.random.default_rng(7)
    K, GOAT, LAMB, WOLF = 7, 2, 4, 1
    M = np.where(np.eye(K, dtype=bool), rng.normal(0.76, 0.03, (K, K)), rng.normal(0.30, 0.04, (K, K)))
    M[GOAT, GOAT] = 0.40
    for j in range(K):
        if j != LAMB: M[j, LAMB] = rng.normal(0.60, 0.03)
    for k in range(K):
        if k != WOLF: M[WOLF, k] = rng.normal(0.58, 0.03)
    M[WOLF, WOLF] = 0.76
    fig = new_fig(10.5, 8, "Matrice degli score $s(j,k)$: goat, lamb, wolf",
                  "riga $j$ = probe dell'utente $j$ (attaccante)  ·  colonna $k$ = template dell'utente $k$ (vittima)  ·  diagonale = genuini $G_k$")
    ax = fig.add_subplot(111)
    im = ax.imshow(M, cmap=SEQ, vmin=0.15, vmax=0.85)
    for j in range(K):
        for k in range(K):
            ax.text(k, j, f"{M[j,k]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if M[j, k] > 0.55 else INK, fontweight="bold" if j == k else "normal")
    ax.set_xticks(range(K), [f"$U_{{{i+1}}}$" for i in range(K)], fontsize=12)
    ax.set_yticks(range(K), [f"$U_{{{i+1}}}$" for i in range(K)], fontsize=12)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel(r"template $k$ (vittima) $\rightarrow$", fontsize=12, labelpad=10)
    ax.set_ylabel(r"probe $j$ (attaccante)", fontsize=12)
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks(np.arange(-.5, K, 1), minor=True); ax.set_yticks(np.arange(-.5, K, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
    ax.add_patch(Rectangle((LAMB - .5, -.5), 1, K, fill=False, ec=FA, lw=2.6, ls=(0, (5, 3))))
    ax.add_patch(Rectangle((-.5, WOLF - .5), K, 1, fill=False, ec=FR, lw=2.6, ls=(0, (5, 3))))
    ax.add_patch(Rectangle((GOAT - .45, GOAT - .45), .9, .9, fill=False, ec=VIOLET, lw=3.2))
    ax.text(K - .35, WOLF, "Wolf $U_2$\n" + r"riga alta: $I_k^{\mathrm{att}}$" + "\ncausa FA verso gli altri", color=FR, fontsize=10.5, va="center", fontweight="bold", clip_on=False)
    ax.text(K - .35, GOAT, "Goat $U_3$\n" + r"diagonale bassa: $G_k$" + "\ngenuino mal riconosciuto\n(FRR alta)", color=VIOLET, fontsize=10.5, va="center", fontweight="bold", clip_on=False)
    ax.text(LAMB, K - .35, "Lamb $U_5$\n" + r"colonna alta: $I_k^{\mathrm{vit}}$ — subisce FA", color=FA, fontsize=10.5, ha="center", va="top", fontweight="bold", clip_on=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.20, location="left")
    cb.set_label("score di similarità $s$"); cb.outline.set_visible(False)
    save(fig, "fig10_matrice_lupi_agnelli")


# =====================================================================
# 21 — Piano (mu_G, mu_I): Yager e Dunstone
# =====================================================================
@figure("fig8_piano_animali")
def f8():
    rng = np.random.default_rng(11)
    MG, MI = 0.62, 0.38
    cl = {"Doves": (0.84, 0.16, 9, 0.035, SKY), "Chameleons": (0.84, 0.60, 8, 0.035, FR),
          "Phantoms": (0.40, 0.16, 8, 0.035, VIOLET), "Worms": (0.40, 0.60, 8, 0.035, FA),
          "Sheep": (MG, MI, 26, 0.045, SLATE)}
    fig = new_fig(10.5, 7.6, r"Il piano $(\mu_G,\ \mu_I)$: la classificazione di Yager e Dunstone",
                  r"ogni punto è un utente $k$  ·  le medie di popolazione dividono il piano in quattro quadranti")
    ax = fig.add_subplot(111)
    ax.set_xlim(0.25, 1.0); ax.set_ylim(0, 0.75); ax.grid(False)
    for (x0, x1, y0, y1, c) in [(0.25, MG, MI, .75, FA), (MG, 1, MI, .75, FR), (0.25, MG, 0, MI, VIOLET), (MG, 1, 0, MI, SKY)]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=c, alpha=0.08, ec="none"))
    ax.axvline(MG, color=INK, lw=1.3, ls=(0, (5, 4))); ax.axhline(MI, color=INK, lw=1.3, ls=(0, (5, 4)))
    for nm, (cx, cy, n, sp, c) in cl.items():
        ax.scatter(rng.normal(cx, sp, n), rng.normal(cy, sp * .9, n), s=70, color=c, alpha=.9, ec="white", lw=1.2, zorder=3)
    for nm, sub, x, y, ha, c in [("Worms", "pochi tratti distintivi, facili da impersonare", .27, .72, "left", FA),
                                 ("Chameleons", "pochi FR, ma tanti FA (tratti generici)", .98, .72, "right", FR),
                                 ("Phantoms", "molti FR, pochi FA (enrollment difficile)", .27, .025, "left", VIOLET),
                                 ("Doves", "i migliori: tratto molto distintivo", .98, .025, "right", SKY)]:
        va = "top" if y > .5 else "bottom"
        ax.text(x, y, nm, fontsize=14, fontweight="bold", color=c, ha=ha, va=va)
        ax.text(x, y + (-.032 if va == "top" else .032), sub, fontsize=9.5, color=MUTED, ha=ha, va=va)
    ax.text(MG + .015, MI - .035, "Sheep", fontsize=13, fontweight="bold", color=SLATE)
    ax.text(MG + .015, MI - .068, "comportamento medio", fontsize=9.5, color=MUTED)
    ax.set_xlabel(r"$\mu_G(k)$  $\rightarrow$  utente più facilmente riconosciuto", fontsize=12)
    ax.set_ylabel(r"$\mu_I(k)$  $\rightarrow$  utente più simile agli altri", fontsize=12)
    ax.text(MG, -0.045, r"$\mu_G$ popolazione", ha="center", fontsize=9.5, color=MUTED, clip_on=False)
    ax.text(0.252, MI + .012, r"$\mu_I$ popolazione", fontsize=9.5, color=MUTED)
    save(fig, "fig8_piano_animali")


# =====================================================================
# 22 — Distribuzioni degli score per animale
# =====================================================================
@figure("fig9_distribuzioni_animali")
def f9():
    T, S = 0.5, 0.09
    zoo = [("Sheep", .72, .30), ("Doves", .88, .15), ("Goats", .40, .30), ("Lambs", .72, .58),
           ("Wolves", .72, .58), ("Chameleons", .82, .60), ("Phantoms", .38, .18), ("Worms", .35, .62)]
    fig = new_fig(13, 8, "Distribuzioni degli score per tipo di utente",
                  rf"gaussiane con $\sigma={S}$  ·  soglia $t={T}$ (si accetta se $s>t$)  ·  verde = genuini, indaco = impostori")
    x = np.linspace(-.15, 1.15, 500)
    for i, (nm, mg, mi) in enumerate(zoo):
        ax = fig.add_subplot(2, 4, i + 1)
        frr, far, d = phi((T - mg) / S), 1 - phi((T - mi) / S), (mg - mi) / S
        ax.fill_between(x, npdf(x, mg, S), color=GEN, alpha=.12); ax.fill_between(x, npdf(x, mi, S), color=IMP, alpha=.12)
        ax.fill_between(x, npdf(x, mg, S), where=x <= T, color=FR, alpha=.9, lw=0)
        ax.fill_between(x, npdf(x, mi, S), where=x >= T, color=FA, alpha=.9, lw=0)
        ax.plot(x, npdf(x, mg, S), color=GEN, lw=2.2); ax.plot(x, npdf(x, mi, S), color=IMP, lw=2.2)
        ax.axvline(T, color=INK, lw=1.3, ls=(0, (4, 3)))
        ax.set_title(nm + {"Lambs": " (vittima, colonna)", "Wolves": " (attaccante, riga)"}.get(nm, ""), fontsize=13); ax.set_yticks([]); ax.set_xticks([0, .5, 1]); ax.set_xlim(-.15, 1.15); ax.grid(False)
        ax.spines["left"].set_visible(False)
        sub = rf"$\mu_G={mg:.2f}$   $\mu_I={mi:.2f}$   $d'={d:.1f}$"
        ax.text(.5, -.20, sub, transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED)
        ax.text(.03, -.36, rf"$FRR_k={frr:.2f}$", transform=ax.transAxes, color=FR, fontweight="bold", fontsize=11)
        ax.text(.97, -.36, rf"$FAR_k={far:.2f}$", transform=ax.transAxes, color=FA, fontweight="bold", fontsize=11, ha="right")
    fig.legend(handles=legend_handles([(FR, r"FR: genuini con $s\leq t$"), (FA, r"FA: impostori con $s>t$")]), loc="lower center", ncol=2, fontsize=11)
    set_bottom(fig, 0.05)
    save(fig, "fig9_distribuzioni_animali")


# =====================================================================
# N13 — z-norm
# =====================================================================
@figure("n13_znorm_per_utente")
def n13():
    users = [("U1 · agnello", .60, .08, FA), ("U2 · pecora", .45, .08, IMP), ("U3 · colomba", .30, .08, GEN)]
    T = 0.55; ZT = phi_inv(.90)
    fig = new_fig(13, 6.4, "Rimedio per lambs e wolves: normalizzazione degli score per utente (z-norm)",
                  r"distribuzioni degli score impostori contro il template di ciascun utente  ·  si accetta se $s>t$")
    axs = fig.subplots(1, 2)
    for ax, ttl, xmin, xmax, tv, mp, xl in [
        (axs[0], "Prima: score grezzi, soglia globale", 0, 1, T, lambda m, s: (m, s), r"score $s$"),
        (axs[1], r"Dopo: z-norm  $z=(s-\mu_I(k))\,/\,\sigma_I(k)$", -4, 4, ZT, lambda m, s: (0., 1.), r"score normalizzato $z$")]:
        x = np.linspace(xmin, xmax, 600)
        for nm, mu, sd, c in users:
            m, s = mp(mu, sd); fa = 1 - phi((tv - m) / s)
            ax.fill_between(x, npdf(x, m, s), where=x >= tv, color=c, alpha=.30, lw=0)
            ax.plot(x, npdf(x, m, s), color=c, lw=2.6, label=rf"{nm.split(' · ')[0]}   $FAR_k={fa:.1%}$".replace("%", r"\%"))
        ax.axvline(tv, color=INK, lw=1.8, ls=(0, (5, 4)))
        ax.text(tv, ax.get_ylim()[1] * 1.0, rf"  $t={tv:.2f}$", fontweight="bold", va="top")
        ax.set_title(ttl); ax.set_xlabel(xl); ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
        ax.legend(loc="upper left" if xmax < 2 else "upper right", fontsize=10.5)
    fig.text(0.5, 0.005, "Altri rimedi: t-norm, soglie utente-specifiche, enrollment multiplo (meno goats), fusione multimodale.  I genuini si spostano: la soglia va ritarata.", ha="center", fontsize=10, color=MUTED)
    save(fig, "n13_znorm_per_utente")


# =====================================================================
# 23 — Decidability d'
# =====================================================================
@figure("fig11_decidability")
def f11():
    SIG, MU = .08, .30
    dvals = [1.5, 3., 5.]
    fig = new_fig(13, 7.6, r"Decidability $d'$: separazione tra distanze intra-classe e inter-classe",
                  r"$D^I$ = stesso soggetto  ·  $D^E$ = soggetti diversi  ·  $d'=\dfrac{|\overline{D^E}-\overline{D^I}|}{\sqrt{(\sigma_I^2+\sigma_E^2)/2}}$")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    x = np.linspace(0, 1, 600)
    for r, dv in enumerate(dvals):
        ax = fig.add_subplot(gs[r, 0])
        mE = MU + dv * SIG; tm = (MU + mE) / 2
        ax.fill_between(x, npdf(x, MU, SIG), color=GEN, alpha=.12); ax.fill_between(x, npdf(x, mE, SIG), color=IMP, alpha=.12)
        ax.fill_between(x, npdf(x, MU, SIG), where=x >= tm, color=FR, alpha=.9, lw=0)
        ax.fill_between(x, npdf(x, mE, SIG), where=x <= tm, color=FA, alpha=.9, lw=0)
        ax.plot(x, npdf(x, MU, SIG), color=GEN, lw=2.4); ax.plot(x, npdf(x, mE, SIG), color=IMP, lw=2.4)
        ax.axvline(tm, color=INK, lw=1.3, ls=(0, (4, 3)))
        ax.annotate("", xy=(mE, -0.9), xytext=(MU, -0.9), arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.4), annotation_clip=False)
        ax.text((MU + mE) / 2, -2.2, rf"$\Delta\mu={dv:g}\sigma={dv*SIG:.2f}$", ha="center", fontsize=9.5, color=MUTED)
        ax.text(.99, .90, rf"$d'={dv:g}$", transform=ax.transAxes, ha="right", va="top", fontsize=14, fontweight="bold")
        ax.text(.99, .62, rf"$EER={phi(-dv/2):.2%}$".replace("%", r"\%"), transform=ax.transAxes, ha="right", color=FA, fontweight="bold", fontsize=11)
        if r == 0:
            ax.text(MU - .04, 4.3, r"$D^I$", color=GEN, fontsize=14, fontweight="bold", ha="right"); ax.text(mE + .04, 4.3, r"$D^E$", color=IMP, fontsize=14, fontweight="bold")
        ax.set_yticks([]); ax.set_xlim(0, 1); ax.grid(False); ax.spines["left"].set_visible(False)
        if r < 2: ax.set_xticklabels([])
    ax.set_xlabel(r"distanza $d$")
    ax = fig.add_subplot(gs[:, 1])
    d = np.linspace(.01, 9, 400)
    ax.plot(d, phi(-d / 2), color=FA, lw=3); ax.set_yscale("log"); ax.set_ylim(1e-6, .6)
    for dv in dvals:
        ax.scatter([dv], [phi(-dv / 2)], s=70, color=INK, ec="white", lw=2, zorder=4)
        ax.annotate(rf"$d'={dv:g}$", (dv, phi(-dv / 2)), xytext=(8, 8), textcoords="offset points", fontsize=10.5, fontweight="bold")
    ax.set_title(r"$EER=\Phi\!\left(-d'/2\right)$"); ax.set_xlabel(r"$d'$"); ax.set_ylabel("EER")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v*100:g}" + r"$\%$"))
    ax.text(.04, .06, r"$d'\uparrow\ \Rightarrow$ curve più separate $\Rightarrow$ EER $\downarrow$", transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.text(0.5, 0.005, r"Con $\sigma$ diverse si usa la media quadratica: es. $\sigma_I=0.06,\ \sigma_E=0.10\ \Rightarrow\ \sigma_{eff}=%.3f$.  $d'$ non dice nulla sulla soglia, solo sulla separabilità." % math.sqrt((.06**2 + .10**2) / 2), ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.03)
    save(fig, "fig11_decidability")


# =====================================================================
# 24 — Misure di qualità: distribuzioni + correlazione (simulato)
# =====================================================================
QCOL = {"SP": IMP, "SI": "#db2777", "SY": FA, "UIQI": SKY, "SE": INK}


@figure("fig12a_qualita_distribuzioni")
def f12a():
    rng = np.random.default_rng(12)
    PAR = {"FERET/fa": {"SP": (.82, .06), "SI": (.50, .17), "SY": (.18, .09), "SE": (.03, .015)},
           "LFW": {"SP": (.77, .07), "SI": (.65, .12), "SY": (.17, .08), "SE": (.025, .010)},
           "SCface": {"SP": (.82, .08), "SI": (.78, .09), "SY": (.25, .08), "SE": (.035, .012)}}
    LOAD = {"SP": .8, "SI": .75, "SY": .3, "SE": .05}
    n = 3000
    data = {}
    for ds in PAR:
        z = rng.normal(size=n)
        data[ds] = {m: np.clip(PAR[ds][m][0] + PAR[ds][m][1] * (a * z + math.sqrt(1 - a * a) * rng.normal(size=n)), 0, 1) for m, a in LOAD.items()}
    fig = new_fig(13, 7.6, "Come misurare la «qualità» di una misura di qualità (1): distribuzione dei valori",
                  "frazione di immagini per intervallo di valore  ·  tre dataset, cinque misure (dati simulati con la forma delle slide)")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
    for r, ds in enumerate(PAR):
        ax = fig.add_subplot(gs[r, 0])
        for m in ["SP", "SI", "SY", "SE"]:
            c, _ = np.histogram(data[ds][m], bins=25, range=(0, 1))
            ax.plot((np.arange(25) + .5) / 25, c / n, "-o", ms=3.5, lw=1.9, color=QCOL[m], label=m)
        ax.axvline(.5, color=QCOL["UIQI"], lw=3, label="UIQI")
        ax.set_ylim(0, .4); ax.set_xlim(0, 1); ax.set_title(ds, loc="right", fontsize=12)
        if r == 0: ax.legend(ncol=5, loc="upper left", fontsize=10)
        if r < 2: ax.set_xticklabels([])
    ax.set_xlabel("valore restituito dalla misura")
    names = ["SP", "SI", "SY", "SE"]
    C = np.corrcoef(np.array([data["FERET/fa"][m] for m in names]))
    ax = fig.add_subplot(gs[:, 1])
    ax.imshow(C, cmap=DIV, vmin=-1, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=12, fontweight="bold" if i == j else "normal", color="white" if abs(C[i, j]) > .6 else INK)
    ax.set_xticks(range(4), names, fontweight="bold"); ax.set_yticks(range(4), names, fontweight="bold")
    ax.xaxis.tick_top(); ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0)
    ax.set_title("Correlazione tra misure\n(FERET/fa simulato)", loc="center", fontsize=11.5, pad=28)
    ax.text(.5, -.12, "UIQI vale 0.5 su ogni immagine:\n" + r"$\sigma=0\ \Rightarrow$ correlazione non definita," + "\nla misura non discrimina.", transform=ax.transAxes, ha="center", va="top", color=SKY, fontweight="bold", fontsize=10)
    ax.text(.5, -.40, "Due misure molto correlate portano\nla stessa informazione: ne basta una.", transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=10)
    save(fig, "fig12a_qualita_distribuzioni")


# =====================================================================
# 25 — Qualità: tasso di immagini adeguate ed EER vs soglia (simulato)
# =====================================================================
def eer_emp(g, i):
    gs, is_ = np.sort(g), np.sort(i)
    ts = np.linspace(0, 1, 201)
    frr = np.searchsorted(gs, ts, "left") / len(gs)
    far = 1 - np.searchsorted(is_, ts, "left") / len(is_)
    k = np.argmin(np.abs(far - frr))
    return (far[k] + frr[k]) / 2


@figure("fig12b_qualita_tolleranza")
def f12b():
    rng = np.random.default_rng(16)
    n = 6000
    q = rng.normal(size=n)
    mix = lambda mu, sd, a: np.clip(mu + sd * (a * q + math.sqrt(1 - a * a) * rng.normal(size=n)), 0, 1)
    meas = {"SP": mix(.82, .06, .8), "SI": mix(.50, .17, .75), "SY": mix(.18, .09, .3),
            "UIQI": np.full(n, .5), "SE": np.clip(rng.normal(.03, .015, n), 0, 1)}
    gen = rng.normal(.60, .11, n) + .07 * q * 1.0
    imp = rng.normal(.35, .10, n)
    TH = np.linspace(0, 1, 51)
    rate, eer = {}, {}
    for m, v in meas.items():
        rate[m], eer[m] = [], []
        for th in TH:
            keep = v >= th; r = keep.mean(); rate[m].append(r)
            eer[m].append(eer_emp(gen[keep], imp[keep]) if r >= .30 else np.nan)
    base = eer["SP"][0]
    fig = new_fig(13, 6.4, "Come misurare la «qualità» di una misura di qualità (2): effetto sull'EER",
                  rf"si scartano le immagini con valore $<$ soglia di tolleranza  ·  EER senza scarti $={base:.1%}$ (dati simulati)".replace("%", r"\%"))
    a, b = fig.subplots(1, 2)
    for m in meas:
        if m == "UIQI": continue
        a.plot(TH, rate[m], color=QCOL[m], lw=2.6, label=m)
        r_ = 1 - np.array(rate[m]); e_ = np.array(eer[m]); ok = ~np.isnan(e_) & (r_ <= .7)
        b.plot(r_[ok], e_[ok], color=QCOL[m], lw=2.6, label=m)
    a.plot([.5, .5], [0, 1], color=QCOL["UIQI"], lw=3.2, label="UIQI")
    b.scatter([0], [base], color=QCOL["UIQI"], s=60, ec="white", lw=1.5, zorder=4)
    b.annotate("UIQI: nessuno scarto utile", (0, base), xytext=(10, 12), textcoords="offset points", color=SKY, fontweight="bold", fontsize=10)
    a.set_title("A · tasso di immagini adeguate"); a.set_xlabel("soglia di tolleranza (valore minimo accettato)"); a.set_ylabel("immagini adeguate")
    b.set_title("B · EER vs campioni scartati"); b.set_xlabel("campioni scartati"); b.set_ylabel("EER")
    b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0)); b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0))
    a.legend(ncol=2, fontsize=10)
    fig.text(.5, .005, "Una buona misura abbassa l'EER scartando poco. Con pochi campioni rimasti l'EER è stimato su insiemi piccoli: serve sempre il tasso di immagini adeguate.", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.04)
    save(fig, "fig12b_qualita_tolleranza")


# =====================================================================
# 26 — Qualità del campione e compromesso EER / scarti
# =====================================================================
@figure("fig12_qualita_eer")
def f12():
    rng = np.random.default_rng(21)
    n = 6000
    qual = rng.random(n)
    gen = rng.normal(.45 + .30 * qual, .08); imp = rng.normal(.35, .10, n)
    fr = np.arange(21) * .025
    meas = {"misura ideale (qualità vera)": (qual, GEN), "misura mediocre (qualità + rumore)": (qual + rng.normal(0, .35, n), FR),
            "nessuna misura (scarto casuale)": (rng.random(n), SLATE)}
    curves = {}
    for nm, (v, c) in meas.items():
        order = np.argsort(v)
        curves[nm] = [eer_emp(gen[order[int(f * n):]], imp[order[int(f * n):]]) for f in fr]
    fig = new_fig(13, 6.6, "Qualità del campione: quanto si guadagna in EER scartando i campioni peggiori",
                  "una buona misura di qualità riduce l'EER scartando poco  ·  una cattiva misura scarta ma non migliora")
    a, b = fig.subplots(1, 2)
    a.fill_between([0, 1], .35 - .2, .35 + .2, color=IMP, alpha=.10); a.axhline(.35, color=IMP, lw=2)
    a.text(.98, .35 + .03, r"impostori ($\mu\pm2\sigma$)", ha="right", color=IMP, fontweight="bold", fontsize=10)
    a.scatter(qual[:500], gen[:500], s=9, color=GEN, alpha=.5, lw=0)
    a.plot([0, 1], [.45, .75], color=GEN, lw=2.8); a.text(.03, .83, "genuini", color=GEN, fontweight="bold", fontsize=12)
    a.set_title("A · score genuino vs qualità"); a.set_xlabel(r"qualità del campione $q$"); a.set_ylabel(r"score di similarità $s$")
    for nm, (_, c) in meas.items():
        ys = curves[nm]
        b.plot(fr, ys, color=c, lw=3, label=rf"{nm}: {ys[0]:.1%} $\to$ {ys[8]:.1%}".replace("%", r"\%"))
        b.scatter([.2], [ys[8]], color=c, s=55, ec="white", lw=1.5, zorder=4)
    b.axvline(.2, color=INK, lw=1.2, ls=(0, (4, 3)))
    b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0)); b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0))
    b.set_title("B · EER vs frazione di campioni scartati"); b.set_xlabel("campioni scartati (peggiori secondo la misura)"); b.set_ylabel("EER")
    b.legend(loc="lower left", fontsize=9.5, title=r"EER $\to$ EER scartando il 20%", title_fontsize=9.5)
    fig.text(.5, .005, "Throughput: scartare il 20% dei campioni = chiedere una nuova acquisizione a 1 utente su 5 (Failure to Acquire controllata).", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.04)
    save(fig, "fig12_qualita_eer")


# =====================================================================
# 27 — Margini (Poh e Bengio)
# =====================================================================
@figure("fig13a_margine")
def f13a():
    MI_, SI_, MG_, SG_ = -1.5, .75, 1.2, .45
    far = lambda d: 1 - phi((d - MI_) / SI_); frr = lambda d: phi((d - MG_) / SG_)
    d = np.linspace(-6, 3, 901); m = np.abs(far(d) - frr(d)); de = d[np.argmin(m)]
    fig = new_fig(11.5, 7.8, "Margini basati sulla stima dell'errore (Poh e Bengio, 2004)",
                  r"$FAR(\Delta)=\dfrac{\mathrm{n.\ FA}(\Delta)}{\mathrm{n.\ impostori}}$  ·  $FRR(\Delta)=\dfrac{\mathrm{n.\ FR}(\Delta)}{\mathrm{n.\ client}}$  ·  $\mathcal{M}(\Delta)=|FAR(\Delta)-FRR(\Delta)|$")
    a, b = fig.subplots(2, 1, sharex=True)
    a.plot(d, far(d), color=FA, lw=3, ls=(0, (6, 3)), label=r"$FAR(\Delta)$"); a.plot(d, frr(d), color=IMP, lw=3, label=r"$FRR(\Delta)$")
    a.scatter([de], [far(de)], s=80, color="white", ec=INK, lw=2.4, zorder=4)
    a.annotate(rf"$EER={far(de):.1%}$".replace("%", r"\%") + "\n" + rf"$\Delta\approx{de:.2f}$", (de, far(de)), xytext=(14, 18), textcoords="offset points", fontweight="bold", fontsize=10.5)
    a.set_title(r"A · tassi di errore in funzione della soglia $\Delta$"); a.legend(loc="center left", fontsize=11)
    b.plot(d, m, color=GEN, lw=3, ls=(0, (6, 3))); b.fill_between(d, m, color=GEN, alpha=.10)
    b.scatter([de], [0], s=80, color="white", ec=INK, lw=2.4, zorder=4)
    b.annotate(r"$\mathcal{M}=0$: massima incertezza", (de, 0), xytext=(14, 24), textcoords="offset points", fontweight="bold", fontsize=10.5)
    b.text(-5.9, .86, r"$\mathcal{M}\approx1$: prevalgono i FA" + "\n(score basso → verdetto netto)", color=FA, fontsize=10, fontweight="bold", va="top")
    b.text(2.9, .55, r"$\mathcal{M}\approx1$: prevalgono i FR" + "\n(score alto → verdetto netto)", color=IMP, fontsize=10, fontweight="bold", ha="right", va="top")
    b.set_title(r"B · margine $\mathcal{M}(\Delta)=|FAR-FRR|$"); b.set_xlabel(r"score $\Delta$")
    for ax in (a, b): ax.axvline(de, color=INK, lw=1.1, ls=(0, (3, 4)))
    fig.text(.5, .005, "Dove il margine è piccolo FAR e FRR sono confrontabili e la decisione è incerta; dove è grande un solo errore domina e il verdetto è più affidabile.", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.03)
    save(fig, "fig13a_margine")


# =====================================================================
# 28 — SRR: la nuvola
# =====================================================================
def person(ax, x, y, c, s=1.0):
    ax.add_patch(Ellipse((x, y + 14 * s), 12 * s, 12 * s, fc=c, ec="none"))
    ax.add_patch(FancyBboxPatch((x - 9 * s, y - 14 * s), 18 * s, 24 * s, boxstyle="round,pad=0,rounding_size=%g" % (6 * s), fc=c, ec="none"))


@figure("fig13b_srr_folla")
def f13b():
    rng = np.random.default_rng(9)
    fig = new_fig(12, 7.4, "SRR: la nuvola attorno al soggetto restituito",
                  "più candidati sono vicini al primo, più la risposta è confondibile e quindi meno affidabile")
    ax = canvas(fig, W=920, H=540)
    few = [(-95, 15), (-45, 35), (70, -35), (105, 5)]
    many = []
    while len(many) < 17:
        dx, dy = rng.uniform(-125, 125), rng.uniform(-62, 62)
        if (dx / 150) ** 2 + (dy / 78) ** 2 <= 1 and (abs(dx) > 18 or abs(dy) > 22): many.append((dx, dy))
    for cy, others, col in [(400, few, GEN), (140, many, FA)]:
        cx = 250
        ax.add_patch(Ellipse((cx, cy), 340, 180, fc=SOFT[col], ec=col, lw=2))
        ax.add_patch(FancyBboxPatch((cx - 20, cy - 34), 40, 62, boxstyle="round,pad=0,rounding_size=8", fc=FA, ec="none", alpha=.9))
        for dx, dy in others: person(ax, cx + dx, cy + dy, "#334155", .85)
        person(ax, cx, cy, "white", 1.0)
    card(ax, 520, 360, 360, 90, r"Nube «poco affollata»" + "\npochi candidati vicini al primo\n" + r"$\rightarrow$ risposta più affidabile ($\varphi$ alto)", GEN, 11.5)
    card(ax, 520, 95, 360, 90, r"Nube «molto affollata»" + "\nmolti candidati vicini al primo\n" + r"$\rightarrow$ risposta meno affidabile ($\varphi$ basso)", FA, 11.5)
    ax.add_patch(FancyBboxPatch((520, 262), 20, 20, boxstyle="round,pad=0,rounding_size=4", fc=FA, ec="none"))
    ax.text(550, 272, "soggetto restituito (rango 1)", va="center", fontsize=11, color=MUTED)
    ax.text(460, 10, "Relative Distance e Density Ratio misurano la stessa idea di «confusione tra i candidati» in due modi diversi.", ha="center", fontsize=10, color=MUTED)
    ax.set_ylim(0, 540); ax.set_xlim(0, 920)
    save(fig, "fig13b_srr_folla")


# =====================================================================
# 29 — Esempi numerici delle slide: RD e DR
# =====================================================================
@figure("fig13c_srr_esempi")
def f13c():
    drd = [.15, .25, .28, .45]; ddr = [.15, .25, .31, .45]
    rd = (drd[1] - drd[0]) / drd[-1]; rest = ddr[1:]; nb = sum(x < 2 * ddr[0] for x in rest); dr = 1 - nb / len(rest)
    fig = new_fig(13, 5.8, "SRR: i due esempi numerici delle slide",
                  r"lista ordinata delle distanze $d(p,g_i)$ dei primi candidati  ·  A: Relative Distance  ·  B: Density Ratio")
    a, b = fig.subplots(1, 2)
    for ax in (a, b):
        ax.set_xlim(-.02, .52); ax.set_ylim(-1.6, 2.0); ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
        ax.axhline(0, color=INK, lw=1.5); ax.set_xlabel(r"distanza $d(p,g_i)$")
    a.set_title("A · Relative Distance")
    for k, v in enumerate(drd, 1):
        c = GEN if k == 1 else (SLATE if k < 4 else FA)
        a.scatter([v], [.35], s=170, color=c, ec="white", lw=2, zorder=3)
        a.text(v, .35 + (.35 if k != 3 else .65), rf"$g_{k}$  {v:.2f}", ha="center", fontweight="bold", fontsize=10.5)
    a.annotate("", xy=(drd[1], -.35), xytext=(drd[0], -.35), arrowprops=dict(arrowstyle="|-|", color=IMP, lw=1.8, mutation_scale=4))
    a.text((drd[0] + drd[1]) / 2, -.62, rf"num $={drd[1]:.2f}-{drd[0]:.2f}={drd[1]-drd[0]:.2f}$", color=IMP, ha="center", fontweight="bold", fontsize=10.5)
    a.annotate("", xy=(drd[-1], -1.0), xytext=(0, -1.0), arrowprops=dict(arrowstyle="|-|", color=FA, lw=1.8, mutation_scale=4))
    a.text(drd[-1] / 2, -1.27, rf"den $=$ distanza massima $={drd[-1]:.2f}$", color=FA, ha="center", fontweight="bold", fontsize=10.5)
    a.text(.25, 1.95, rf"$\varphi=\dfrac{{{drd[1]-drd[0]:.2f}}}{{{drd[-1]:.2f}}}={rd:.3f}$", ha="center", va="top", fontsize=15,
           bbox=dict(boxstyle="round,pad=.5", fc=SOFT[FR], ec=FR))
    b.set_title("B · Density Ratio")
    r_ = 2 * ddr[0]
    b.axvspan(0, r_, ymin=.5, ymax=.78, color=FR, alpha=.14); b.axvline(r_, ymin=.45, ymax=.82, color=FR, lw=2, ls=(0, (5, 3)))
    b.text(r_, .95 - .05, rf"$2d_1={r_:.2f}$", ha="center", color=FR, fontweight="bold", fontsize=10.5)
    for k, v in enumerate(ddr, 1):
        c = GEN if k == 1 else (FR if v < r_ else "#9ca3af")
        b.scatter([v], [.35], s=170, color=c, ec="white", lw=2, zorder=3); b.text(v, .7, f"{v:.2f}", ha="center", fontweight="bold", fontsize=10.5)
    b.text(.26, -.5, "verde = primo (escluso dal conteggio)  ·  ambra = in $N_b$  ·  grigio = fuori", ha="center", fontsize=9.5, color=MUTED)
    b.text(.26, -1.15, rf"$N_b=\{{0.25\}}\Rightarrow |N_b|={nb},\ |G|={len(rest)}$" + "\n" + rf"$\varphi=1-\dfrac{{{nb}}}{{{len(rest)}}}={dr:.3f}$", ha="center", va="center", fontsize=13,
           bbox=dict(boxstyle="round,pad=.5", fc=SOFT[GEN], ec=GEN))
    fig.text(.5, .005, r"Variante con il primo incluso: $1-2/4=0.5$ (non necessaria). Nessun parametro adattivo ha migliorato il fattore 2.", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.04)
    save(fig, "fig13c_srr_esempi")


# =====================================================================
# 30 — Density Ratio e Relative Distance su due probe (simulato)
# =====================================================================
def _make_probes():
    rng = np.random.default_rng(5)
    N = 40

    def make(d1, nc, lo, hi):
        d = [d1] + list(rng.uniform(lo, hi, nc))
        d += list(rng.uniform(.62, .95, N - len(d)))
        return np.sort(np.array(d))
    return make(.18, 0, 0, 0), make(.30, 12, .32, .58), N


def phi_density(d):
    rest = d[1:]; nb = int((rest < 2 * d[0]).sum()); return 1 - nb / len(rest), nb


def phi_relative(d):
    return (d[1] - d[0]) / d[-1]


@figure("fig13_density_ratio")
def f13():
    dA, dB, N = _make_probes()
    fig = new_fig(13, 6.2, "Density Ratio: quanto è affollata la nube attorno al primo candidato",
                  r"distanze $d(p,g_i)$ ordinate  ·  $N_b$ = template con $d<2d_1$  ·  $\varphi=1-|N_b|/|G|$ (primo escluso)")
    axs = fig.subplots(1, 2, sharey=True)
    for ax, nm, d, ok in [(axs[0], "Probe A · risposta affidabile", dA, True), (axs[1], "Probe B · risposta ambigua", dB, False)]:
        ph, nb = phi_density(d); k = np.arange(1, N + 1)
        ax.axhspan(0, 2 * d[0], color=FR, alpha=.12); ax.axhline(2 * d[0], color=FR, lw=1.8, ls=(0, (5, 3)))
        ax.text(N, 2 * d[0] + .02, rf"$2d_1={2*d[0]:.2f}$", ha="right", color=FR, fontweight="bold")
        cols = [GEN if i == 0 else (FR if v < 2 * d[0] else "#9ca3af") for i, v in enumerate(d)]
        ax.scatter(k, d, c=cols, s=[110 if i == 0 else 38 for i in range(N)], ec="white", lw=1.2, zorder=3)
        ax.annotate(r"$d_1$", (1, d[0]), xytext=(14, -22), textcoords="offset points", color=GEN, fontweight="bold", fontsize=12)
        ax.set_title(nm, color=GEN if ok else FA); ax.set_xlabel("rango $k$ nella lista ordinata"); ax.set_ylim(0, 1)
        ax.text(.03, .95, rf"$|N_b|={nb}$ su $|G|={N-1}$" + "\n" + rf"$\varphi={ph:.3f}$" + "\n" + rf"$RD=\dfrac{{d_2-d_1}}{{d_{{max}}}}={phi_relative(d):.3f}$", transform=ax.transAxes, va="top", fontsize=11.5,
                bbox=dict(boxstyle="round,pad=.5", fc=SOFT[GEN if ok else FA], ec=GEN if ok else FA))
    axs[0].set_ylabel(r"distanza $d(p,g_k)$")
    fig.text(.5, .005, "ambra = template nella nube (contano in $N_b$)  ·  grigio = lontani  ·  verde = primo candidato (escluso)  ·  $\\varphi$ alto = nube poco affollata", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.04)
    save(fig, "fig13_density_ratio")


# =====================================================================
# 31 — Valore critico phi_k
# =====================================================================
@figure("fig13d_phi_critico")
def f13d():
    rng = np.random.default_rng(17)
    cl = lambda v, a, b: np.clip(v, a, b)
    SETS = {"Density Ratio": (cl(rng.normal(.15, .09, 55), .01, .39), cl(rng.exponential(.035, 50), 0, .12), .4),
            "Relative Distance": (cl(rng.normal(.60, .26, 55), .05, 1.0), cl(rng.exponential(.14, 50), 0, .62), 1.0)}
    fig = new_fig(13, 8, r"Valore critico $\varphi_k$: separare i GA dai FA con la reliability",
                  r"cerchi = GA (genuini accettati)  ·  quadrati = FA (impostori accettati)  ·  $\varphi_k$ = soglia che massimizza la coerenza (dati simulati)")
    axs = fig.subplots(2, 2, gridspec_kw=dict(height_ratios=[1, 1.5]))
    for c, (nm, (ga, fa, smax)) in enumerate(SETS.items()):
        ths = np.linspace(0, smax, 201)
        rel = np.array([((ga >= t).sum() + (fa < t).sum()) / (len(ga) + len(fa)) for t in ths])
        kb = rel.argmax(); pk = ths[kb]
        a, b = axs[0, c], axs[1, c]
        a.plot(ths, rel, color=IMP, lw=2.8); a.scatter([pk], [rel[kb]], marker="X", s=130, color=FA, ec="white", lw=1.5, zorder=4)
        a.annotate(rf"$\varphi_k={pk:.3f}$", (pk, rel[kb]), xytext=(12, -22), textcoords="offset points", color=FA, fontweight="bold", fontsize=11.5)
        a.set_title(nm, fontsize=14); a.set_ylabel(r"reliability di $\varphi(p)$"); a.set_xlabel(r"soglia")
        idx = np.arange(len(ga) + len(fa)); rng.shuffle(idx)
        vals = np.concatenate([ga, fa]); isga = np.concatenate([np.ones(len(ga), bool), np.zeros(len(fa), bool)])
        v, g = vals[idx], isga[idx]
        b.scatter(idx[g] * 0 + np.where(g)[0], v[g], facecolors="none", edgecolors=FA, s=34, lw=1.5, label="GA")
        b.scatter(np.where(~g)[0], v[~g], marker="s", facecolors="none", edgecolors=IMP, s=34, lw=1.5, label="FA")
        b.axhline(pk, color=INK, lw=1.8); b.text(len(v), pk + smax * .02, rf"$\varphi_k={pk:.3f}$", ha="right", fontweight="bold")
        fa_hi = int((fa >= pk).sum()); ga_lo = int((ga < pk).sum())
        b.set_xlabel("probe"); b.set_ylabel(r"$\varphi(p)$"); b.legend(loc="upper left", ncol=2, fontsize=10)
        b.set_title(rf"FA sopra $\varphi_k$: {fa_hi}   ·   GA sotto $\varphi_k$: {ga_lo}", fontsize=10.5, color=MUTED, fontweight="normal")
    fig.text(.5, .005, r"Quadrati sopra $\varphi_k$ = FA confermati da un $\varphi$ alto; cerchi sotto $\varphi_k$ = GA non confermati per un $\varphi$ basso.", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.03)
    save(fig, "fig13d_phi_critico")


# =====================================================================
# 32 — Da phi a SRR e doppia soglia
# =====================================================================
@figure("fig14_srr_doppia_soglia")
def f14():
    PB, T = .80, .35
    dA, dB, _ = _make_probes()
    phiA, phiB = phi_density(dA)[0], phi_density(dB)[0]
    d1A, d1B = dA[0], dB[0]
    srr = lambda p, pb=PB: (p - pb) / ((1 - pb) if p > pb else pb)
    fig = new_fig(13, 6.6, r"Dal valore $\varphi$ all'indice SRR e decisione a doppia soglia",
                  r"$\bar\varphi=0.8$ (valore critico)  ·  $SRR=\dfrac{\varphi-\bar\varphi}{S(\varphi,\bar\varphi)}\in[-1,1]$  ·  $SRR>0$ = risposta ritenuta affidabile")
    a, b = fig.subplots(1, 2)
    p = np.linspace(0, 1, 401)
    a.axhspan(0, 1, color=GEN, alpha=.08); a.axhspan(-1, 0, color=FR, alpha=.10)
    a.plot(p, [srr(x) for x in p], color=IMP, lw=3.2); a.axhline(0, color=INK, lw=1.2); a.axvline(PB, color=INK, lw=1.3, ls=(0, (5, 3)))
    a.text(PB + .01, -.55, r"$\bar\varphi=0.8$", fontweight="bold")
    for nm, ph, c in [("A", phiA, GEN), ("B", phiB, FR)]:
        a.scatter([ph], [srr(ph)], s=110, color=c, ec="white", lw=2, zorder=4)
        a.annotate(rf"{nm}: $\varphi={ph:.2f}$" + "\n" + rf"$SRR={srr(ph):+.2f}$", (ph, srr(ph)), xytext=(-12, 16 if nm == "B" else -34), textcoords="offset points", ha="right", color=c, fontweight="bold", fontsize=10.5)
    a.text(.02, .92, "affidabile", color=GEN, fontweight="bold"); a.text(.98, -.93, "non affidabile", color=FR, fontweight="bold", ha="right")
    a.set_title("A · normalizzazione SRR"); a.set_xlabel(r"$\varphi(p)$"); a.set_ylabel("SRR"); a.set_ylim(-1.05, 1.05)
    b.set_xlim(0, .6); b.set_ylim(-1.05, 1.05)
    b.fill_between([0, T], 0, 1.05, color=GEN, alpha=.16); b.fill_between([0, T], -1.05, 0, color=FR, alpha=.18); b.fill_between([T, .6], -1.05, 1.05, color=SLATE, alpha=.14)
    b.axvline(T, color=INK, lw=1.8, ls=(0, (6, 4))); b.axhline(0, color=INK, lw=1.8, ls=(0, (6, 4)))
    b.text(T / 2, .62, "accettato\ne affidabile\n" + r"$\rightarrow$ output", ha="center", color=GEN, fontweight="bold", fontsize=11)
    b.text(T / 2, -.72, "accettato ma non affidabile\n" + r"$\rightarrow$ rifiutato dalla reliability", ha="center", color=FR, fontweight="bold", fontsize=10)
    b.text((T + .6) / 2, .85, "rifiutato\ndalla soglia $t$", ha="center", color=SLATE, fontweight="bold", fontsize=11)
    for nm, d1, s, c in [("A", d1A, srr(phiA), GEN), ("B", d1B, srr(phiB), FR), ("C", .48, .45, SLATE)]:
        b.scatter([d1], [s], s=120, color=c, ec="white", lw=2, zorder=4); b.text(d1 + .015, s, nm, color=c, fontweight="bold", fontsize=14, va="center")
    b.set_xticks([0, .1, .2, .3, T, .5, .6], ["0", "0.1", "0.2", "0.3", r"$t$", "0.5", "0.6"])
    b.set_title("B · due soglie: accettazione e reliability"); b.set_xlabel(r"distanza del primo candidato $d_1$"); b.set_ylabel("SRR")
    fig.text(.5, .005, r"A e B hanno $d_1$ sotto soglia, ma solo A è affidabile; C non supera nemmeno la soglia di accettazione.", ha="center", fontsize=10.5, color=MUTED)
    set_bottom(fig, 0.04)
    save(fig, "fig14_srr_doppia_soglia")


# =====================================================================
# 33 — Stima automatica della soglia th_i
# =====================================================================
@figure("fig15_soglia_reliability")
def f15():
    rng = np.random.default_rng(3)
    M = 30
    subs = [("Soggetto stabile", np.clip(rng.normal(.80, .05, M), 0, 1)), ("Soggetto instabile", np.clip(rng.normal(.75, .20, M), 0, 1))]
    fig = new_fig(12, 7.6, "Stima automatica della soglia di reliability dalle $M$ osservazioni",
                  r"$th_i=\left|\dfrac{E[\overline{S_i}]^2-\sigma[\overline{S_i}]}{E[\overline{S_i}]}\right|$  ·  la soglia si adatta a media e stabilità del soggetto")
    axs = fig.subplots(2, 1, sharex=True)
    for ax, (nm, v) in zip(axs, subs):
        E, sd = v.mean(), v.std(); th = abs(E ** 2 - sd) / E; m = np.arange(1, M + 1)
        ax.axhspan(max(0, E - sd), min(1, E + sd), color=IMP, alpha=.10); ax.axhline(E, color=IMP, lw=1.8)
        ax.axhline(th, color=INK, lw=2.2, ls=(0, (6, 4)))
        ax.plot(m, v, color="#9ca3af", lw=1.3, zorder=2)
        ax.scatter(m, v, c=[GEN if x >= th else FA for x in v], s=48, ec="white", lw=1.2, zorder=3)
        ax.text(M + .8, E, rf"$E={E:.2f}$", color=IMP, fontweight="bold", va="center")
        ax.text(M + .8, th + (-.06 if th < E else .05), rf"$th={th:.2f}$", fontweight="bold", va="center")
        ax.set_title(nm); ax.set_ylim(0, 1.05); ax.set_ylabel("reliability")
        ax.text(.01, .06, rf"$\sigma={sd:.2f}$  ·  {int((v>=th).sum())}/{M} osservazioni affidabili ($\geq th$)", transform=ax.transAxes, fontweight="bold", color=MUTED)
    axs[1].set_xlabel(r"osservazione $m$ (es. frame consecutivo)"); axs[1].set_xlim(.5, M + 5)
    fig.text(.5, .005, "verde = sopra $th$, rosso = sotto $th$  ·  banda = media $\\pm\\,\\sigma$  ·  soggetto instabile → soglia più bassa, altrimenti rifiuterebbe troppe risposte", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.03)
    save(fig, "fig15_soglia_reliability")


# =====================================================================
# 34 — Metrica: triangolare e simmetria
# =====================================================================
def _M6():
    rng = np.random.default_rng(15)
    P = rng.uniform(0, 1, (6, 2)); D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    N = rng.uniform(.85, 1.9, (6, 6)); N = (N + N.T) / 2; D = D * N; np.fill_diagonal(D, 0)
    return D


@figure("fig16_metrica")
def f16():
    M6 = _M6()
    bad = sum(any(d[i] > d[(i + 1) % 3] + d[(i + 2) % 3] + 1e-12 for i in range(3))
              for d in ([M6[a, b], M6[b, c], M6[a, c]] for a, b, c in combinations(range(6), 3)))
    tot = 20
    fig = new_fig(14, 6.8, "Proprietà di una metrica: disuguaglianza triangolare e simmetria",
                  r"$d(x,z)\leq d(x,y)+d(y,z)$  ·  $d(x,y)=d(y,x)$  ·  una misura che le viola è solo una semimetrica (o nemmeno)")
    axs = fig.subplots(1, 3)

    def tri(ax, ttl, sub, pos, dxy, dyz, dxz, ok):
        c = GEN if ok else FA
        ax.set_xlim(-.5, 6.5); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl, color=c, loc="center"); ax.text(3, 3.05, sub, ha="center", color=MUTED, fontsize=10)
        for a_, b_, lab, col, lw in [("x", "y", rf"$d(x,y)={dxy:.2f}$", SLATE, 1.8), ("y", "z", rf"$d(y,z)={dyz:.2f}$", SLATE, 1.8), ("x", "z", rf"$d(x,z)={dxz:.2f}$", c, 3)]:
            (x1, y1), (x2, y2) = pos[a_], pos[b_]
            ax.plot([x1, x2], [y1, y2], color=col, lw=lw, zorder=1)
            off = {"xy": (-.6, .35), "yz": (.6, .35), "xz": (0, -.5)}[a_ + b_]
            ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], lab, ha="center", fontsize=10, fontweight="bold")
        for k, (px, py) in pos.items():
            ax.scatter([px], [py], s=420, color=SOFT[IMP], ec=IMP, lw=2, zorder=3); ax.text(px, py, rf"${k}$", ha="center", va="center", color=IMP, fontweight="bold", fontsize=12, zorder=4)
        sc = 5.4 / max(dxz, dxy + dyz)
        ax.add_patch(Rectangle((.3, -1.55), dxz * sc, .5, color=c, alpha=.85)); ax.text(.4, -1.3, rf"$d(x,z)={dxz:.2f}$", color="white", fontweight="bold", va="center", fontsize=9.5)
        ax.add_patch(Rectangle((.3, -2.2), dxy * sc, .5, color=IMP, alpha=.85)); ax.add_patch(Rectangle((.3 + dxy * sc, -2.2), dyz * sc, .5, color="#818cf8", alpha=.95))
        ax.text(.4, -1.95, rf"$d(x,y)+d(y,z)={dxy+dyz:.2f}$", color="white", fontweight="bold", va="center", fontsize=9.5)
        ax.text(3, -3.05, (r"$d(x,z)\leq$ somma: rispettata" if ok else r"$d(x,z)>$ somma: VIOLATA"), ha="center", color=c, fontweight="bold", fontsize=11)

    pe = {"x": (0, 0), "y": (3, 2), "z": (6, 0)}
    de = lambda a, b: math.dist(pe[a], pe[b])
    tri(axs[0], "A · Distanza euclidea", "è una metrica", {"x": (.4, -.4), "y": (3, 1.9), "z": (5.6, -.4)}, de("x", "y"), de("y", "z"), de("x", "z"), True)
    tri(axs[1], "B · Misura elastica (tipo DTW)", "valori ipotetici: semimetrica", {"x": (.4, -.4), "y": (3, 1.9), "z": (5.6, -.4)}, .30, .30, .90, False)
    ax = axs[2]; ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal")
    ax.set_title("C · Asimmetria e simmetrizzazione", color=FR, loc="center")
    ax.text(3, 3.05, r"$d(A,B)\neq d(B,A)\ \Rightarrow$ si media nelle due direzioni", ha="center", color=MUTED, fontsize=9.5)
    for k, px in (("A", .8), ("B", 5.2)):
        ax.scatter([px], [.6], s=700, color=SOFT[IMP], ec=IMP, lw=2); ax.text(px, .6, rf"${k}$", ha="center", va="center", fontsize=15, color=IMP, fontweight="bold")
    arrow(ax, (1.3, 1.1), (4.7, 1.1), FR, 2.4); ax.text(3, 1.55, r"$d(A,B)=0.80$", ha="center", color=FR, fontweight="bold")
    arrow(ax, (4.7, .1), (1.3, .1), GEN, 2.4); ax.text(3, -.35, r"$d(B,A)=0.40$", ha="center", color=GEN, fontweight="bold")
    ax.text(3, -1.6, r"$d_{sym}(A,B)=\dfrac{d(A,B)+d(B,A)}{2}=\dfrac{0.80+0.40}{2}=0.60$", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=.5", fc=SOFT[FR], ec=FR))
    ax.text(3, -2.7, "simmetrica, ma resta una semimetrica\nse viola la disuguaglianza triangolare", ha="center", color=MUTED, fontsize=9.5)
    fig.text(.5, .005, rf"Test sulla matrice $M_6$ del §9: {bad} triple su {tot} violano la disuguaglianza triangolare $\rightarrow$ " + ("è una metrica" if bad == 0 else "è solo una semimetrica (simmetrica, diagonale nulla)") + ".", ha="center", fontsize=11, fontweight="bold")
    set_bottom(fig, 0.04)
    save(fig, "fig16_metrica")


# =====================================================================
# 35 — Confronto tra due sistemi
# =====================================================================
@figure("fig17_confronto_sistemi")
def f17():
    SYS = {"Sistema A": (.70, .10, .40, .10), "Sistema B": (.72, .20, .40, .06)}
    SC = {"Sistema A": IMP, "Sistema B": FA}
    far_s = lambda t, s: 1 - phi((t - s[2]) / s[3]); frr_s = lambda t, s: phi((t - s[0]) / s[1])
    auc = lambda s: phi((s[0] - s[2]) / math.sqrt(s[1] ** 2 + s[3] ** 2))
    frr_at = lambda f, s: frr_s(s[2] + phi_inv(1 - f) * s[3], s)
    fig = new_fig(13.5, 6.8, "Confrontare due sistemi: l'AUC non basta, conta il punto operativo",
                  "DET in scala log  ·  A: più separato in media  ·  B: impostori molto concentrati")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    t = np.linspace(-.3, 1.5, 4000)
    for nm, s in SYS.items():
        x, y = far_s(t, s), frr_s(t, s); ok = (x >= 1e-3) & (y >= 1e-3)
        a.plot(x[ok], y[ok], color=SC[nm], lw=3.2, label=rf"{nm}: $AUC={auc(s):.3f}$")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-3, 1); a.set_ylim(1e-3, 1)
    for ax_ in (a.xaxis, a.yaxis): ax_.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v*100:g}" + r"$\%$"))
    a.grid(True, which="major"); a.set_xlabel(r"FAR (scala log)"); a.set_ylabel(r"FRR (scala log)"); a.set_aspect("equal")
    fs = 10 ** np.linspace(-3, -.01, 2000)
    diff = np.array([frr_at(f, SYS["Sistema A"]) - frr_at(f, SYS["Sistema B"]) for f in fs])
    ix = np.where(diff[:-1] * diff[1:] < 0)[0]
    if len(ix):
        c = fs[ix[0]]; yc = frr_at(c, SYS["Sistema A"])
        a.scatter([c], [yc], s=110, color="white", ec=INK, lw=2.4, zorder=5)
        a.annotate(rf"incrocio: $FAR\approx{c:.1%}$".replace("%", r"\%"), (c, yc), xytext=(14, 14), textcoords="offset points", fontweight="bold")
    a.text(.0015, .6, "B migliore\n(sicurezza alta)", color=FA, fontweight="bold"); a.text(.3, .0045, "A migliore\n(comodità)", color=IMP, fontweight="bold", ha="right")
    a.text(.0013, .0013, r"$\swarrow$ migliore", color=GEN, fontweight="bold")
    a.legend(loc="upper right", fontsize=10.5)
    fars = [.1, .01, .001]; w = .36
    for j, (nm, s) in enumerate(SYS.items()):
        v = [frr_at(f, s) for f in fars]; xs = np.arange(3) + (j - .5) * w * 1.1
        bars = b.bar(xs, v, w, color=SC[nm], label=nm)
        for xx, vv in zip(xs, v): b.text(xx, vv + .015, f"{vv:.0%}".replace("%", r"$\%$"), ha="center", fontweight="bold", color=SC[nm], fontsize=10.5)
    b.set_xticks(range(3), [rf"$FAR={f:.1%}$".replace("%", r"\%") for f in fars]); b.set_ylim(0, 1.05)
    b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0)); b.set_title("FRR a FAR fissata"); b.grid(axis="x", visible=False); b.legend(fontsize=10)
    gap = frr_at(.001, SYS["Sistema A"]) - frr_at(.001, SYS["Sistema B"])
    fig.text(.5, .03, rf"A ha l'AUC più alta ma a $FAR=0.1\%$ ha {gap*100:.0f} punti di FRR in più di B: la scelta dipende dall'applicazione, non dall'AUC.", ha="center", fontsize=11, fontweight="bold")
    fig.text(.5, .003, "A parità di FAR vince chi ha FRR inferiore; a parità di FRR chi ha FAR inferiore; senza vincoli si confrontano AUC e curve.", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.07)
    save(fig, "fig17_confronto_sistemi")


# =====================================================================
# 36 — Dataset shift
# =====================================================================
@figure("fig18_shift_operativo")
def f18():
    LAB = (.70, .10, .35, .10); OPS = (.58, .13, .38, .11); T = (LAB[0] + LAB[2]) / 2
    fig = new_fig(12, 7.8, "Le prestazioni sono ex-post: la soglia scelta offline non vale in produzione",
                  rf"soglia fissata in laboratorio all'EER: $t={T:.3f}$ (si accetta se $s\geq t$)")
    axs = fig.subplots(2, 1, sharex=True)
    x = np.linspace(-.1, 1.1, 600)
    for ax, nm, (mg, sg, mi, si) in [(axs[0], "Valutazione offline (laboratorio)", LAB), (axs[1], "Uso reale (utenti nuovi, condizioni variabili)", OPS)]:
        fa, fr = 1 - phi((T - mi) / si), phi((T - mg) / sg)
        ax.fill_between(x, npdf(x, mi, si), color=IMP, alpha=.12); ax.fill_between(x, npdf(x, mg, sg), color=GEN, alpha=.12)
        ax.fill_between(x, npdf(x, mi, si), where=x >= T, color=FA, alpha=.9, lw=0); ax.fill_between(x, npdf(x, mg, sg), where=x <= T, color=FR, alpha=.9, lw=0)
        ax.plot(x, npdf(x, mi, si), color=IMP, lw=2.4); ax.plot(x, npdf(x, mg, sg), color=GEN, lw=2.4)
        ax.axvline(T, color=INK, lw=2, ls=(0, (6, 4)))
        ax.text(mi - .1, npdf(mi, mi, si) * .55, "impostori", color=IMP, fontweight="bold", fontsize=11.5, ha="center")
        ax.text(mg + .12, npdf(mg, mg, sg) * .65, "genuini", color=GEN, fontweight="bold", fontsize=11.5, ha="center")
        ax.text(.985, .90, rf"$FAR={fa:.1%}$".replace("%", r"\%") + "\n" + rf"$FRR={fr:.1%}$".replace("%", r"\%"), transform=ax.transAxes, ha="right", va="top", fontsize=13, fontweight="bold", color=FA, linespacing=1.5)
        ax.set_title(nm); ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
    axs[1].set_xlabel(r"score $s$")
    fig.text(.5, .005, "Stessa soglia e stessa tecnologia: medie più basse e code più larghe spostano FAR e FRR. Per questo si ri-valuta sul campo (valutazione di scenario/operativa).", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.03)
    save(fig, "fig18_shift_operativo")


# =====================================================================
# 37 — Dal sample al template
# =====================================================================
@figure("fig19a_template_schema")
def f19a():
    rng = np.random.default_rng(31)
    BASE = rng.uniform([.12, .1], [.88, .9], (16, 2))

    def acq(j=.025, drop=2, add=1):
        p = BASE + rng.normal(0, j, BASE.shape); p = np.delete(p, rng.choice(len(p), drop, replace=False), axis=0)
        return np.vstack([p, rng.uniform([.12, .1], [.88, .9], (add, 2))])

    def sample_icon(ax, x, y, w, h, tilt=0.):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", fc="#f8fafc", ec=SLATE, lw=1.6); ax.add_patch(box)
        for k in range(1, 9):
            e = Ellipse((x + w / 2 + tilt * w, y + h * .45), 2 * k * w * .075, 2 * k * h * .085, fc="none", ec="#94a3b8", lw=1.4); ax.add_patch(e); e.set_clip_path(box)

    def tmpl_icon(ax, x, y, w, h, pts):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", fc="white", ec=SLATE, lw=1.6))
        ax.scatter(x + pts[:, 0] * w, y + (1 - pts[:, 1]) * h, s=14, color=FA, zorder=3)

    fig = new_fig(13, 8.4, "Template: dal sample alla gallery",
                  "il matching usa il template (feature etichettate con l'identità), non il sample  ·  ogni acquisizione genera un template diverso")
    ax = canvas(fig, W=920, H=600)
    sample_icon(ax, 40, 380, 120, 150); ax.text(100, 355, "sample", ha="center", fontweight="bold", fontsize=12); ax.text(100, 335, "(es. impronta)", ha="center", color=MUTED, fontsize=10)
    arrow(ax, (170, 455), (238, 455), INK, 2.2); ax.text(204, 480, "estrazione\nfeature", ha="center", fontsize=10, fontweight="bold")
    tmpl_icon(ax, 246, 380, 120, 150, BASE); ax.text(306, 355, "template", ha="center", fontweight="bold", color=FA, fontsize=12); ax.text(306, 335, r"$\{(x,y,\theta)\ldots\}$ + identità", ha="center", color=MUTED, fontsize=10)
    card(ax, 40, 120, 330, 110, "Proprietà del template\n• sta in uno spazio ridotto:\n  facile da codificare e archiviare\n• non deve permettere di ricostruire\n  un sample valido", IMP, 10.5, align="left")
    ax.text(640, 560, "Stessa persona, tre acquisizioni", ha="center", fontweight="bold", fontsize=13)
    for j in range(3):
        x0 = 440 + j * 150
        sample_icon(ax, x0, 400, 110, 120, tilt=(j - 1) * .05); arrow(ax, (x0 + 55, 392), (x0 + 55, 358), INK, 1.8)
        tmpl_icon(ax, x0, 235, 110, 120, acq()); ax.text(x0 + 55, 215, rf"template $T_{j+1}$", ha="center", color=FA, fontweight="bold", fontsize=11.5)
    card(ax, 440, 100, 430, 80, r"$T_1\neq T_2\neq T_3$ anche per lo stesso soggetto" + "\n" + r"$\rightarrow$ la gallery può contenere più template per identità", FR, 11)
    ax.text(460, 25, "In esercizio arrivano molti più dati che all'enrollment: usarli per aggiornare i template risolve invecchiamento e qualità (§16).", ha="center", fontsize=10, color=MUTED)
    save(fig, "fig19a_template_schema")


# =====================================================================
# 38 — Tassonomia template updating
# =====================================================================
@figure("fig19b_template_tassonomia")
def f19b():
    fig = new_fig(13, 9, "Template updating: due scelte indipendenti",
                  "come si assegnano le etichette ai nuovi dati  ·  quando si scelgono i template più rappresentativi")
    ax = canvas(fig, W=920, H=640)
    ax.text(230, 600, "Assegnazione delle etichette", ha="center", fontweight="bold", fontsize=13.5)
    card(ax, 30, 490, 400, 90, "Supervised\nserve un supervisore che assegna le identità\ndi solito lavora offline", GEN, 11, weight="normal")
    card(ax, 30, 385, 400, 90, "Semi-supervised\nusa l'unione di dati etichettati e non etichettati\nlavora sia online sia offline", FR, 11)
    ax.text(690, 600, "Selezione dei template più rappresentativi", ha="center", fontweight="bold", fontsize=13.5)
    card(ax, 490, 490, 400, 90, "Online\nla selezione avviene appena il sistema\nacquisisce nuovi dati", IMP, 11)
    card(ax, 490, 385, 400, 90, "Offline\nla selezione avviene dopo aver raccolto dati\nin un certo intervallo di tempo", SLATE, 11)
    ax.text(30, 340, "Come cambia nel tempo", fontweight="bold", fontsize=13)
    acq = [190, 270, 340, 430, 520, 590, 680, 760]
    for r, (nm, col) in enumerate([("Online", IMP), ("Offline", SLATE)]):
        y = 260 - r * 120
        ax.text(30, y, nm, color=col, fontweight="bold", fontsize=12.5, va="center")
        if r == 1:
            for wx, ww, lab in [(170, 300, "finestra di raccolta"), (640, 220, "finestra successiva")]:
                ax.add_patch(FancyBboxPatch((wx, y - 20), ww, 40, boxstyle="round,pad=0,rounding_size=8", fc=SOFT[FR], ec=FR, lw=1.5, ls=(0, (5, 3))))
                ax.text(wx + ww / 2, y + 32, lab, ha="center", color=FR, fontweight="bold", fontsize=10)
        ax.plot([150, 860], [y, y], color="#94a3b8", lw=2, zorder=1); arrow(ax, (850, y), (872, y), "#94a3b8", 2)
        for x in acq:
            ax.scatter([x], [y], s=90, color=GEN, ec="white", lw=2, zorder=3)
            if r == 0: arrow(ax, (x, y - 10), (x, y - 38), col, 1.4, ms=9)
        if r == 0: ax.text(505, y - 58, "ogni nuova acquisizione può aggiornare subito la gallery", ha="center", color=MUTED, fontsize=10.5)
        else:
            arrow(ax, (470, y - 38), (470, y - 22), SLATE, 1.6, ms=9); ax.text(470, y - 52, "selezione dei template in blocco, alla fine di ogni finestra", ha="center", color=MUTED, fontsize=10.5)
    ax.text(30, 25, "Obiettivi:", fontweight="bold", va="center", fontsize=12)
    card(ax, 130, 5, 340, 40, "Template ageing: il tratto cambia nel tempo", FA, 10.5)
    card(ax, 490, 5, 390, 40, "Template enhancing: più dati → template migliori", GEN, 10.5)
    save(fig, "fig19b_template_tassonomia")


# =====================================================================
# 39 — Template update: invecchiamento e politiche
# =====================================================================
@figure("fig19_template_update")
def f19():
    rng = np.random.default_rng(8)
    NU, NS = 3000, 12
    TA, TU, SG = .55, .65, .08
    mu = lambda a: .80 - .05 * a
    def sim(policy):
        age = np.ones(NU); fr, mean = [], []
        for s in range(NS):
            sc = rng.normal(mu(age), SG); fr.append((sc < TA).mean()); mean.append(sc.mean())
            if policy == "none": age += 1
            elif policy == "supervised": age[:] = 1
            else: age = np.where(sc >= TU, 1, age + 1)
        return np.array(fr), np.array(mean)
    pol = [("none", "nessun aggiornamento", FA), ("semi", rf"semi-supervised ($s\geq{TU}$)", FR), ("supervised", "supervised (operatore)", GEN)]
    res = {k: sim(k) for k, _, _ in pol}
    pois = 1 - phi((TU - .35) / .10)
    fig = new_fig(13, 6.6, "Template update: l'invecchiamento del tratto e le politiche di aggiornamento",
                  rf"{NU} utenti simulati  ·  score genuino atteso $=0.80-0.05\cdot(\mathrm{{distanza\ dal\ template\ più\ recente}})$  ·  $t={TA}$".replace("\\mathrm{distanza", "\\mathrm{distanza"))
    a, b = fig.subplots(1, 2); s = np.arange(1, NS + 1)
    for k, nm, c in pol:
        a.plot(s, res[k][1], "-o", color=c, lw=2.8, ms=5, label=nm); b.plot(s, res[k][0], "-o", color=c, lw=2.8, ms=5, label=rf"{nm}: FRR$_{{12}}={res[k][0][-1]:.0%}$".replace("%", r"\%"))
    a.axhline(TA, color=INK, lw=1.6, ls=(0, (5, 3))); a.text(NS, TA + .015, rf"$t={TA}$", ha="right", fontweight="bold")
    a.set_title("A · score genuino medio"); b.set_title(r"B · $FRR$ (a soglia fissa)")
    b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0)); b.set_ylim(-.02, 1.02)
    for ax in (a, b): ax.set_xlabel("sessione dopo l'enrollment"); ax.set_xticks(s)
    a.set_ylabel(r"score medio $\bar s$"); b.set_ylabel("FRR"); a.legend(loc="lower left", fontsize=10); b.legend(loc="upper left", fontsize=9.5)
    fig.text(.5, .035, rf"Rischio del semi-supervised: un impostore con score $\geq{TU}$ entrerebbe in gallery (qui $p\approx{pois:.3%}$ per tentativo).".replace("%", r"\%"), ha="center", fontsize=10.5, color=MUTED)
    fig.text(.5, .005, "La soglia di update va tenuta più severa di quella di accettazione. I template vecchi si conservano: l'update aggiunge, non sostituisce.", ha="center", fontsize=10, color=MUTED)
    set_bottom(fig, 0.08)
    save(fig, "fig19_template_update")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        print("\n".join(REG)); sys.exit()
    todo = [n for n in REG if not args or any(n.startswith(a) for a in args)]
    for n in todo:
        REG[n]()
