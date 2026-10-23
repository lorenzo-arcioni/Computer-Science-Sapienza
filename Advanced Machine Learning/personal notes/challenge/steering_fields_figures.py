#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure e verifiche numeriche per la nota su
"Steering Fields: Adaptive Vector Fields for Safe Image Generation and Beyond"
(Facchiano et al., arXiv:2609.39573v1).

Uso:
    python3 steering_fields_figures.py [cartella_output]      (default: ./figs)

Cosa fa
-------
1. verify():  controlla NUMERICAMENTE le derivazioni dell'Appendice A
              (minimizzatore chiuso, decomposizione alpha/beta, caso blending,
              somma dei pesi = 1, caso mu = lambda  ->  steering classico).
2. Genera 14 figure SVG (testo mantenuto come testo, quindi nitide e selezionabili).

Il "modello" e' un TOY ANALITICO in 2D: ogni condizione c (src, tar, away) e' una
gaussiana isotropa N(m_c, s_c^2 I) e il campo di velocita' del Rectified Flow e'
noto in forma chiusa. Non e' FLUX/SD3.5: serve solo a rendere visibile la
geometria dell'idea. I numeri delle tabelle (fig 10-13) sono invece quelli
riportati nel paper.

Convenzione temporale usata ovunque qui (e nell'Algoritmo 1 del paper):
    t = 0  -> rumore,   t = 1 -> immagine pulita,   z_{i+1} = z_i + dt * v.
    z_t = (1 - t) * eps + t * x,   v_target = x - eps.
(L'Appendice B del paper usa la convenzione opposta: vedi la nota nel markdown.)

Dipendenze: numpy, matplotlib.
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# --------------------------------------------------------------------------- #
#  Stile globale
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "svg.fonttype": "none",          # testo vero nell'SVG
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# palette Okabe-Ito (leggibile anche con daltonismo)
C_SRC = "#0072B2"    # blu    -> source
C_TAR = "#009E73"    # verde  -> target
C_AWAY = "#D55E00"   # rosso-arancio -> away
C_STAR = "#CC79A7"   # rosa   -> v* (campo sterzato)
C_GREY = "#7a7a7a"
C_GOLD = "#E69F00"
C_INK = "#222222"

OUT = "figs"


def save(fig, name):
    path = os.path.join(OUT, name + ".svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print("  scritto", path)


# --------------------------------------------------------------------------- #
#  Toy model: campi di velocita' analitici del Rectified Flow
# --------------------------------------------------------------------------- #
# z_t = (1-t) eps + t x ,  eps~N(0,I), x~N(m, s^2 I).
# v(z,t) = E[x - eps | z_t = z] = m + k(t) (z - t m),
#   k(t) = (t s^2 - (1-t)) / ((1-t)^2 + t^2 s^2)        (condizionamento gaussiano)
M = {
    "src": np.array([2.4, 1.0]),
    "tar": np.array([3.1, -0.9]),
    "away": np.array([1.6, 2.5]),
}
S = {"src": 0.55, "tar": 0.30, "away": 0.85}


def vel(z, t, c):
    """Campo di velocita' esatto della condizione c in (z,t). z: (...,2)."""
    m, s = M[c], S[c]
    k = (t * s**2 - (1 - t)) / ((1 - t) ** 2 + t**2 * s**2)
    return m + k * (z - t * m)


def steer(vs, vt, va, mu, lam):
    """Eq. (4):  v* = (v_src + mu v_tar - lam v_away) / (1 + mu - lam)."""
    return (vs + mu * vt - lam * va) / (1.0 + mu - lam)


def integrate(z0, mode="none", mu=0.3, lam=0.3, n=40, t_start=0.0, const_r=None, const_a=None):
    """Euler da t_start a 1. mode in {none, blend, replace, const}."""
    ts = np.linspace(t_start, 1.0, n + 1)
    z = np.array(z0, float)
    path = [z.copy()]
    for i in range(n):
        t, dt = ts[i], ts[i + 1] - ts[i]
        vs = vel(z, t, "src")
        if mode == "none":
            v = vs
        elif mode == "blend":
            v = vs + mu / (1 + mu) * (vel(z, t, "tar") - vs)
        elif mode == "replace":
            v = steer(vs, vel(z, t, "tar"), vel(z, t, "away"), mu, lam)
        elif mode == "const":
            v = vs + const_a * const_r
        z = z + dt * v
        path.append(z.copy())
    return np.array(path)


# --------------------------------------------------------------------------- #
#  1. VERIFICHE NUMERICHE dell'Appendice A
# --------------------------------------------------------------------------- #
def verify():
    rng = np.random.default_rng(0)
    print("== VERIFICHE NUMERICHE (Appendice A) ==")
    ok_all = True
    for trial in range(5):
        d = 6
        vs, vt, va = rng.normal(size=(3, d))
        mu, lam = rng.uniform(0, 2), rng.uniform(0, 2)
        while 1 + mu - lam <= 0.05:
            mu, lam = rng.uniform(0, 2), rng.uniform(0, 2)

        L = lambda v: np.sum((v - vs) ** 2) + mu * np.sum((v - vt) ** 2) - lam * np.sum((v - va) ** 2)
        v_closed = steer(vs, vt, va, mu, lam)                            # Eq. 4

        # (a) gradiente nullo nel minimizzatore chiuso (A.2)
        grad = 2 * (v_closed - vs) + 2 * mu * (v_closed - vt) - 2 * lam * (v_closed - va)
        # (b) e' davvero un minimo: perturbazioni casuali aumentano L (stretta convessita')
        worse = all(L(v_closed + 0.1 * rng.normal(size=d)) > L(v_closed) for _ in range(200))
        # (c) decomposizione alpha/beta (A.3, Eq. 5)
        a, b = mu / (1 + mu - lam), lam / (1 + mu - lam)
        v_dec = vs + a * (vt - vs) - b * (va - vs)
        # (d) pesi affini: sommano a 1
        w = np.array([1, mu, -lam]) / (1 + mu - lam)
        # (e) minimizzazione numerica indipendente (gradient descent)
        v = np.zeros(d)
        for _ in range(4000):
            g = 2 * (1 + mu - lam) * v - 2 * vs - 2 * mu * vt + 2 * lam * va
            v -= 0.05 / (1 + mu - lam) * g
        ok = (np.linalg.norm(grad) < 1e-12 and worse and np.allclose(v_closed, v_dec)
              and abs(w.sum() - 1) < 1e-12 and np.allclose(v, v_closed, atol=1e-6))
        ok_all &= ok
        print(f"  trial {trial}: mu={mu:.3f} lam={lam:.3f}  |grad|={np.linalg.norm(grad):.1e}  "
              f"min={worse}  decomp_ok={np.allclose(v_closed, v_dec)}  sum_w={w.sum():.6f}  "
              f"GD_ok={np.allclose(v, v_closed, atol=1e-6)}  -> {'OK' if ok else 'FAIL'}")

    # blending (A.1): lambda = 0
    vs, vt = rng.normal(size=(2, 6))
    mu = 0.7
    a = mu / (1 + mu)
    assert np.allclose((vs + mu * vt) / (1 + mu), (1 - a) * vs + a * vt)
    assert np.allclose((1 - a) * vs + a * vt, vs + a * (vt - vs))
    print("  blending (lambda=0): (1-a)v_src + a v_tar == v_src + a(v_tar - v_src)  OK")

    # mu = lambda  ->  steering classico v_src + mu * (v_tar - v_away)
    vs, vt, va = rng.normal(size=(3, 6))
    m = 0.3
    assert np.allclose(steer(vs, vt, va, m, m), vs + m * (vt - va))
    print("  mu = lambda: v* = v_src + mu (v_tar - v_away)  (forma dello steering classico, Eq.1)  OK")

    # Soglia di convessita'
    print("  1 + mu - lambda > 0  <=>  Hessiana 2(1+mu-lam) I definita positiva (strettamente convessa)")
    print("  RISULTATO GLOBALE:", "TUTTO OK" if ok_all else "ERRORE")

    # iperparametri riportati nel paper -> alpha, beta
    print("\n  Iperparametri del paper (App. E.2) -> (alpha, beta, pesi su src/tar/away):")
    for name, mu, lam in [("FLUX (T2I safety)", .3, .3), ("SD3.5 (T2I safety)", .4, .4),
                          ("Editing (I2I)", .8, .8)]:
        den = 1 + mu - lam
        print(f"   {name:20s} mu={mu} lam={lam}  alpha={mu/den:.3f} beta={lam/den:.3f}  "
              f"pesi=({1/den:.3f}, {mu/den:.3f}, {-lam/den:.3f})")
    return ok_all


# --------------------------------------------------------------------------- #
#  FIG 1 - overview: campi e traiettorie (analogo del Fig.1-sinistra del paper)
# --------------------------------------------------------------------------- #
def fig01_overview():
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    # quiver di v_src a t = 0.5
    gx, gy = np.meshgrid(np.linspace(-1.2, 4.0, 17), np.linspace(-1.8, 3.6, 13))
    Z = np.stack([gx, gy], -1)
    V = vel(Z, 0.5, "src")
    ax.quiver(gx, gy, V[..., 0], V[..., 1], color="#c8c8c8", angles="xy", scale_units="xy",
              scale=7, width=0.0028, zorder=1)
    for c, col, lab in [("src", C_SRC, "distribuzione src"), ("tar", C_TAR, "distribuzione tar"),
                        ("away", C_AWAY, "distribuzione away")]:
        for r in (1, 2):
            ax.add_patch(Circle(M[c], r * S[c], fill=False, ec=col, lw=1.2, alpha=0.75 if r == 1 else 0.35, ls="-"))
        ax.plot(*M[c], "o", color=col, ms=6, zorder=5)
        ax.annotate(lab, M[c] + np.array([0.08, 0.12 + S[c]]), color=col, fontsize=9, fontweight="bold")

    z0 = np.array([-0.7, 0.5])
    p_none = integrate(z0, "none")
    p_rep = integrate(z0, "replace", 0.9, 0.9)   # forza didattica
    ax.plot(*p_none.T, color=C_SRC, lw=2.2, ls="--", label="traiettoria src (non sterzata)")
    ax.plot(*p_rep.T, color=C_STAR, lw=2.6, label="traiettoria sterzata (Steering Fields)")
    ax.plot(*z0, "ks", ms=9, zorder=6)
    ax.annotate("rumore $z$ (t=0)", z0, xytext=(-1.3, 0.75), fontsize=9)

    # frecce locali nei punti della traiettoria sterzata
    ts = np.linspace(0, 1, 41)
    for i in (4, 12, 22, 31):
        t, z = ts[i], p_rep[i]
        vs, vt, va = vel(z, t, "src"), vel(z, t, "tar"), vel(z, t, "away")
        vstar = steer(vs, vt, va, 0.9, 0.9)
        for v, col in [(vs, C_SRC), (vt, C_TAR), (va, C_AWAY), (vstar, C_STAR)]:
            ax.add_patch(FancyArrowPatch(z, z + 0.33 * v, arrowstyle="-|>", mutation_scale=9,
                                         color=col, lw=1.6, zorder=7))
    ax.plot(*p_none[-1], "o", color=C_SRC, mfc="white", ms=9, zorder=8)
    ax.plot(*p_rep[-1], "*", color=C_STAR, ms=15, zorder=8)
    ax.set_xlim(-1.4, 4.2)
    ax.set_ylim(-2.0, 3.9)
    ax.set_aspect("equal")
    ax.set_xlabel("dimensione latente 1 (toy 2D)")
    ax.set_ylabel("dimensione latente 2")
    ax.set_title("A ogni passo si stima di nuovo la direzione: $v_{src}$, $v_{tar}$, $v_{away}$ → $v^*$")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.text(3.95, -1.95, "frecce colorate = $v_{src}$ (blu), $v_{tar}$ (verde),\n$v_{away}$ (arancio), $v^*$ (rosa)",
            ha="right", va="bottom", fontsize=8, color=C_GREY)
    save(fig, "fig01_overview_traiettorie")


# --------------------------------------------------------------------------- #
#  FIG 2 - Flow matching: percorso lineare e Eulero
# --------------------------------------------------------------------------- #
def fig02_flow_matching():
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    # sinistra: percorso lineare coppia (eps, x)
    ax = axs[0]
    eps = np.array([-0.2, 0.2]); x = np.array([2.4, 1.0])
    for t in (0, 0.25, 0.5, 0.75, 1):
        z = (1 - t) * eps + t * x
        ax.plot(*z, "o", color=plt.cm.viridis(t), ms=9)
        ax.annotate(f"t={t:g}", z + np.array([0.0, -0.22]), ha="center", fontsize=8.5)
    ax.plot(*np.stack([eps, x]).T, color=C_GREY, lw=1.5, zorder=0)
    ax.add_patch(FancyArrowPatch(eps, eps + (x - eps) * 0.5, arrowstyle="-|>", mutation_scale=14,
                                 color=C_STAR, lw=2.2, shrinkA=0, shrinkB=0))
    ax.annotate(r"velocità costante  $x-\epsilon$", (1.0, 0.95), color=C_STAR, fontsize=9.5, fontweight="bold")
    ax.annotate("rumore ε", eps + np.array([-0.2, 0.25]), fontsize=9)
    ax.annotate("immagine x", x + np.array([-0.2, 0.25]), fontsize=9)
    ax.set_xlim(-0.8, 3.0); ax.set_ylim(-0.6, 1.8); ax.set_aspect("equal")
    ax.set_title(r"Rectified flow: $z_t=(1-t)\,\epsilon+t\,x$")
    ax.set_xticks([]); ax.set_yticks([])
    # destra: Eulero sull'ODE appresa (campo analitico src)
    ax = axs[1]
    gx, gy = np.meshgrid(np.linspace(-1, 3.5, 14), np.linspace(-1.2, 2.8, 10))
    Z = np.stack([gx, gy], -1)
    V = vel(Z, 0.5, "src")
    ax.quiver(gx, gy, V[..., 0], V[..., 1], color="#d0d0d0", angles="xy", scale_units="xy", scale=7, width=0.003)
    for k, N in enumerate((4, 10, 40)):
        p = integrate(np.array([-0.6, 0.2]), "none", n=N)
        ax.plot(*p.T, "-o" if N < 40 else "-", ms=4 if N < 40 else 0, lw=2 if N == 40 else 1.4,
                color=[C_GOLD, C_AWAY, C_SRC][k], label=f"Eulero, N = {N} passi")
    ax.plot(*M["src"], "x", color="k", ms=9)
    ax.set_title(r"Integrazione ODE: $z_{i+1}=z_i+\Delta t_i\,V(z_i,t_i\mid c)$")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlim(-1, 3.6); ax.set_ylim(-1.2, 2.8); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig02_flow_matching")


# --------------------------------------------------------------------------- #
#  FIG 3 - geometria della loss quadratica
# --------------------------------------------------------------------------- #
def fig03_loss_geometry():
    vs, vt, va = np.array([0.0, 0.0]), np.array([2.0, -1.2]), np.array([-1.0, 1.6])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    cases = [(0.0, 0.0, "μ=0, λ=0  →  $v^*=v_{src}$"),
             (0.5, 0.0, "μ=0.5, λ=0 (blending)  →  segmento"),
             (0.8, 0.5, "μ=0.8, λ=0.5 (replacing)  →  fuori dal segmento")]
    g = np.linspace(-3, 3.5, 300)
    X, Y = np.meshgrid(g, g)
    for ax, (mu, lam, ttl) in zip(axs, cases):
        sq = lambda c: (X - c[0]) ** 2 + (Y - c[1]) ** 2
        L = sq(vs) + mu * sq(vt) - lam * sq(va)
        ax.contour(X, Y, L, levels=14, colors="#aaaaaa", linewidths=0.8)
        ax.contourf(X, Y, L, levels=14, cmap="Blues", alpha=0.35)
        vstar = steer(vs, vt, va, mu, lam)
        for p, col, nm in [(vs, C_SRC, "$v_{src}$"), (vt, C_TAR, "$v_{tar}$"), (va, C_AWAY, "$v_{away}$")]:
            ax.plot(*p, "o", color=col, ms=9, zorder=5)
            ax.annotate(nm, p + np.array([0.12, 0.14]), color=col, fontsize=11, fontweight="bold")
        ax.plot(*vstar, "*", color=C_STAR, ms=18, zorder=6, mec="k", mew=0.6)
        ax.annotate("$v^*$", vstar + np.array([0.12, -0.35]), color=C_STAR, fontsize=12, fontweight="bold")
        ax.plot(*np.stack([vs, vt]).T, color=C_TAR, lw=1, ls=":")
        ax.plot(*np.stack([vs, va]).T, color=C_AWAY, lw=1, ls=":")
        ax.set_aspect("equal"); ax.set_xlim(-3, 3.5); ax.set_ylim(-3, 3.5)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(r"$L(v)=\|v-v_{src}\|^2+\mu\|v-v_{tar}\|^2-\lambda\|v-v_{away}\|^2$ : paraboloide con un solo minimo",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig03_geometria_loss")


# --------------------------------------------------------------------------- #
#  FIG 4 - convessita': 1+mu-lam > 0 (sezione 1D) e mappe di alpha, beta
# --------------------------------------------------------------------------- #
def fig04_convexity_and_maps():
    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.1])
    ax = fig.add_subplot(gs[0])
    v = np.linspace(-3, 4, 400)
    vs_, vt_, va_ = 0.0, 1.5, -1.0
    for mu, lam, col in [(0.3, 0.3, C_SRC), (0.5, 1.0, C_AWAY), (0.5, 1.5, C_GOLD)]:
        L = (v - vs_) ** 2 + mu * (v - vt_) ** 2 - lam * (v - va_) ** 2
        ax.plot(v, L, color=col, lw=2.2, label=f"μ={mu}, λ={lam}  →  1+μ−λ={1 + mu - lam:+.1f}")
    ax.set_ylim(-12, 25); ax.axhline(0, color="#ddd", lw=0.8)
    ax.set_xlabel("$v$ (sezione 1D)"); ax.set_ylabel(r"$L(v)$")
    ax.set_title("Convessa solo se 1+μ−λ>0")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    mu_g, lam_g = np.meshgrid(np.linspace(0, 2, 300), np.linspace(0, 2, 300))
    den = 1 + mu_g - lam_g
    valid = den > 0
    pts = [("FLUX", .3, .3), ("SD3.5", .4, .4), ("editing", .8, .8)]
    for i, (name, f) in enumerate([("α = μ/(1+μ−λ)", lambda m, l, d: m / d), ("β = λ/(1+μ−λ)", lambda m, l, d: l / d)]):
        ax = fig.add_subplot(gs[i + 1])
        F = np.where(valid, f(mu_g, lam_g, np.where(valid, den, 1)), np.nan)
        F = np.clip(F, 0, 3)
        im = ax.pcolormesh(mu_g, lam_g, F, cmap="viridis", shading="auto", vmin=0, vmax=3, rasterized=True)
        ax.plot([0, 2], [1, 3], color="white", lw=2)   # bordo λ = 1+μ
        ax.fill_between([0, 2], [1, 3], [2, 2], color="#e8e8e8")
        ax.text(0.25, 1.78, "regione non ammessa\n(1+μ−λ ≤ 0)", fontsize=8, color=C_GREY)
        for nm, m, l in pts:
            ax.plot(m, l, "o", color="white", mec="k", ms=8)
            ax.annotate(nm, (m, l), (m + 0.07, l - 0.16), fontsize=8, color="white", fontweight="bold")
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xlabel("μ (attrazione)"); ax.set_ylabel("λ (repulsione)")
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="saturato a 3")
    save(fig, "fig04_convessita_alpha_beta")


# --------------------------------------------------------------------------- #
#  FIG 5 - decomposizione e pesi affini
# --------------------------------------------------------------------------- #
def fig05_decomposition():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    vs, vt, va = np.array([0.0, 0.0]), np.array([2.6, -1.0]), np.array([-0.4, 2.0])
    mu, lam = 0.8, 0.5
    den = 1 + mu - lam
    a, b = mu / den, lam / den
    p1 = vs + a * (vt - vs)
    p2 = p1 - b * (va - vs)
    for p, col, nm, off in [(vs, C_SRC, "$v_{src}$", (-0.1, -0.35)), (vt, C_TAR, "$v_{tar}$", (0.1, 0.25)),
                            (va, C_AWAY, "$v_{away}$", (0.1, 0.1))]:
        ax.plot(*p, "o", color=col, ms=10)
        ax.annotate(nm, p + np.array(off), color=col, fontsize=12, fontweight="bold")
    arrow = lambda p, q, col, lw=2.2, ls="-": ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                                                                           color=col, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))
    arrow(vs, vt, C_TAR, 1.2, "--"); arrow(vs, va, C_AWAY, 1.2, "--")
    arrow(vs, p1, C_TAR, 3)
    ax.annotate(r"+ α ($v_{tar}-v_{src}$)", (p1 + vs) / 2 + np.array([-0.5, 0.35]), color=C_TAR, fontsize=10, fontweight="bold")
    arrow(p1, p2, C_AWAY, 3)
    ax.annotate(r"− β ($v_{away}-v_{src}$)", (p1 + p2) / 2 + np.array([0.3, -0.35]), color=C_AWAY, fontsize=10, fontweight="bold")
    ax.plot(*p2, "*", color=C_STAR, ms=20, mec="k", mew=0.6, zorder=6)
    ax.annotate("$v^*$", p2 + np.array([-0.55, -0.1]), color=C_STAR, fontsize=14, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xlim(-1.2, 3.4); ax.set_ylim(-2.2, 2.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"$v^*=v_{{src}}+\\alpha(v_{{tar}}-v_{{src}})-\\beta(v_{{away}}-v_{{src}})$   (μ={mu}, λ={lam}: α={a:.2f}, β={b:.2f})",
                 fontsize=9.5)
    ax = axs[1]
    cfgs = [("μ=λ=0.3\n(FLUX)", .3, .3), ("μ=λ=0.4\n(SD3.5)", .4, .4), ("μ=λ=0.8\n(editing)", .8, .8),
            ("μ=0.8, λ=0.5", .8, .5), ("μ=0.5, λ=0\n(blending)", .5, 0.0)]
    xs = np.arange(len(cfgs))
    w_s = np.array([1 / (1 + m - l) for _, m, l in cfgs])
    w_t = np.array([m / (1 + m - l) for _, m, l in cfgs])
    w_a = np.array([-l / (1 + m - l) for _, m, l in cfgs])
    ax.bar(xs, w_s, color=C_SRC, label="peso su $v_{src}$")
    ax.bar(xs, w_t, bottom=w_s, color=C_TAR, label="peso su $v_{tar}$")
    ax.bar(xs, w_a, color=C_AWAY, label="peso su $v_{away}$ (negativo)")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.text(len(cfgs) - 0.5, 1.02, "somma pesi = 1", ha="right", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cfgs], fontsize=8)
    ax.set_ylim(-0.9, 2.1)
    ax.set_title("Combinazione affine: i pesi sommano sempre a 1")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig05_decomposizione_pesi")


# --------------------------------------------------------------------------- #
#  FIG 6 - vettore costante vs campo
# --------------------------------------------------------------------------- #
def delta_field(z, t):
    return vel(z, t, "tar") - vel(z, t, "away")


def fig06_constant_vs_field():
    rng = np.random.default_rng(1)
    # "r" classico: media di v_tar - v_away su stati lungo molte traiettorie src
    zs, ts_all, dsamp = [], [], []
    ts = np.linspace(0, 1, 41)
    for _ in range(200):
        p = integrate(rng.normal(size=2), "none")
        for i in range(0, 40):
            dsamp.append(delta_field(p[i], ts[i]))
    r_bar = np.mean(dsamp, 0)
    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    gx, gy = np.meshgrid(np.linspace(-2, 4, 60), np.linspace(-2.5, 3.5, 60))
    Z = np.stack([gx, gy], -1)
    for j, t in enumerate((0.1, 0.5, 0.9)):
        ax = fig.add_subplot(gs[0, j])
        D = delta_field(Z, t)
        err = np.linalg.norm(D - r_bar, axis=-1)
        im = ax.pcolormesh(gx, gy, err, cmap="magma_r", vmin=0, vmax=3, shading="auto", rasterized=True)
        ax.quiver(gx[::5, ::5], gy[::5, ::5], D[::5, ::5, 0], D[::5, ::5, 1], color="white",
                  angles="xy", scale_units="xy", scale=9, width=0.004)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {t}:  $\\|v^\\Delta(z,t)-r\\|$")
        if j == 2:
            plt.colorbar(im, ax=ax, fraction=0.046, label="errore del vettore costante")
    ax = fig.add_subplot(gs[1, 0:2])
    tt = np.linspace(0.02, 0.98, 60)
    pth = integrate(np.array([-0.5, 0.4]), "none", n=60)[:-1]
    tg = np.linspace(0, 1, 61)[:-1]
    d_true = np.array([delta_field(pth[i], tg[i]) for i in range(len(tg))])
    ax.plot(tg, d_true[:, 0], color=C_STAR, lw=2.2, label="$v^\\Delta_x(z_t,t)$ (campo)")
    ax.plot(tg, d_true[:, 1], color=C_GOLD, lw=2.2, label="$v^\\Delta_y(z_t,t)$ (campo)")
    ax.axhline(r_bar[0], color=C_STAR, ls="--", lw=1.4, label="$r_x$ (costante)")
    ax.axhline(r_bar[1], color=C_GOLD, ls="--", lw=1.4, label="$r_y$ (costante)")
    ax.set_xlabel("tempo t (0 = rumore, 1 = immagine)")
    ax.set_ylabel("componenti")
    ax.set_title("Lungo UNA traiettoria la direzione corretta cambia: il vettore fisso è giusto solo 'in media'")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    ax = fig.add_subplot(gs[1, 2])
    errs = np.linalg.norm(d_true - r_bar, axis=1)
    ax.fill_between(tg, errs, color=C_AWAY, alpha=0.35)
    ax.plot(tg, errs, color=C_AWAY, lw=2)
    ax.set_xlabel("t"); ax.set_ylabel(r"$\|v^\Delta - r\|$")
    ax.set_title("Errore del vettore fisso")
    fig.tight_layout()
    print(f"  [fig06] r_bar (costante) = {np.round(r_bar, 3)};  errore medio lungo traiettoria = {errs.mean():.3f};  max = {errs.max():.3f}")
    save(fig, "fig06_costante_vs_campo")


# --------------------------------------------------------------------------- #
#  FIG 7 - flowchart algoritmo
# --------------------------------------------------------------------------- #
def fig07_algorithm():
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

    def box(x, y, w, h, txt, fc="#f4f4f4", ec="#555", fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arr(p, q, col="#444", txt=None, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=col, lw=1.6, linestyle=ls))
        if txt:
            ax.text((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2, txt, fontsize=8.5, color=col)

    box(0.3, 7.6, 3.6, 1.0, "T2I\n$z_0\\sim N(0,I)$,  s = 0", "#e3f0fa", C_SRC)
    box(4.3, 7.6, 4.0, 1.0, "I2I (editing)\n$z_1$=VAEenc($x_{img}$),  $z_s=(1-\\sigma_s)z_1+\\sigma_s z_0$", "#e8f6ef", C_TAR, 9)
    box(8.7, 7.6, 3.0, 1.0, "input: $c_{src}$, $c_{tar}$,\n($c_{away}$), μ_i, λ_i, modo", "#fff6e0", C_GOLD)
    box(3.8, 6.2, 4.4, 0.9, "for i = s … N−1", "#ffffff", "#333", bold=True)
    arr((2.1, 7.6), (4.5, 7.1)); arr((6.3, 7.6), (6.0, 7.1))
    box(0.3, 4.7, 3.2, 1.0, "$v_{src}=v_\\theta(z_i,t_i,c_{src})$", "#e3f0fa", C_SRC)
    box(4.4, 4.7, 3.2, 1.0, "$v_{tar}=v_\\theta(z_i,t_i,c_{tar})$", "#e8f6ef", C_TAR)
    box(8.5, 4.7, 3.2, 1.0, "$v_{away}=v_\\theta(z_i,t_i,c_{away})$\n(solo REPLACE)", "#fbe9df", C_AWAY, 9)
    for x in (1.9, 6.0, 10.1):
        arr((6.0, 6.2), (x, 5.7))
    box(0.3, 2.9, 5.4, 1.2, "BLEND:\n$v^*_i=v_{src}+\\frac{\\mu_i}{1+\\mu_i}(v_{tar}-v_{src})$", "#f9ecf3", C_STAR)
    box(6.3, 2.9, 5.4, 1.2, "REPLACE:\n$v^*_i=\\frac{v_{src}+\\mu_i v_{tar}-\\lambda_i v_{away}}{1+\\mu_i-\\lambda_i}$", "#f9ecf3", C_STAR)
    arr((1.9, 4.7), (2.4, 4.1)); arr((6.0, 4.7), (3.6, 4.1)); arr((6.0, 4.7), (8.4, 4.1)); arr((10.1, 4.7), (9.6, 4.1))
    arr((1.9, 4.7), (6.9, 4.1), ls=":", col="#999")
    box(3.4, 1.5, 5.2, 0.9, "$z_{i+1}=z_i+\\Delta t_i\\,v^*_i$   ($\\Delta t_i=t_{i+1}-t_i$)", "#ffffff", "#333", bold=True)
    arr((3.0, 2.9), (5.0, 2.4)); arr((9.0, 2.9), (7.0, 2.4))
    ax.add_patch(FancyArrowPatch((8.6, 1.95), (8.2, 6.65), connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                                 mutation_scale=13, color="#444", lw=1.4, linestyle="--"))
    ax.text(10.3, 4.0, "ripeti per\nogni passo", fontsize=8.5, color="#444")
    box(3.4, 0.2, 5.2, 0.8, "dopo l'ultimo passo:  $x=$VAEdecode($z_N$)", "#ffffff", "#333", bold=True)
    arr((6.0, 1.5), (6.0, 1.0))
    ax.text(6.0, 8.95, "Algoritmo 1 – Steering Fields (App. A.4)", ha="center", fontsize=12, fontweight="bold")
    ax.text(11.7, 0.15, "costo: BLEND = 2 forward/passo, REPLACE = 3\n(contro 1 del baseline)", ha="right", fontsize=8.5, color=C_GREY)
    save(fig, "fig07_algoritmo_flusso")


# --------------------------------------------------------------------------- #
#  FIG 8 - blending: effetto di alpha
# --------------------------------------------------------------------------- #
def fig08_blending():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    z0 = np.array([-0.6, 0.4])
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    cm = plt.cm.cool
    ends = []
    for k, a in enumerate(alphas):
        mu = a / (1 - a) if a < 1 else 1e6
        p = integrate(z0, "blend", mu=mu)
        ax.plot(*p.T, color=cm(k / 4), lw=2.2, label=f"α = {a:g}")
        ends.append(p[-1])
    for c, col in [("src", C_SRC), ("tar", C_TAR)]:
        ax.add_patch(Circle(M[c], S[c], fill=False, ec=col, lw=1.4))
        ax.annotate(c, M[c] + np.array([0.05, S[c] + 0.08]), color=col, fontweight="bold")
    ax.plot(*z0, "ks", ms=8)
    ax.set_aspect("equal"); ax.set_xlim(-1, 4); ax.set_ylim(-1.8, 2.2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title(r"Blending: $v^*=(1-\alpha)v_{src}+\alpha v_{tar}$, stesso rumore")
    ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1]
    aa = np.linspace(0, 0.99, 40)
    dist_s, dist_t = [], []
    for a in aa:
        p = integrate(z0, "blend", mu=a / (1 - a))[-1]
        dist_s.append(np.linalg.norm(p - M["src"])); dist_t.append(np.linalg.norm(p - M["tar"]))
    ax.plot(aa, dist_s, color=C_SRC, lw=2.2, label="distanza da media src")
    ax.plot(aa, dist_t, color=C_TAR, lw=2.2, label="distanza da media tar")
    ax.set_xlabel("α = μ/(1+μ)"); ax.set_ylabel("distanza del punto finale")
    ax.set_title("α è una manopola continua")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig08_blending_alpha")


# --------------------------------------------------------------------------- #
#  FIG 9 - I2I: livello di rumore sigma_s
# --------------------------------------------------------------------------- #
def fig09_i2i_noise():
    rng = np.random.default_rng(7)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    z1 = M["src"] + np.array([0.35, -0.25])                  # "immagine sorgente" codificata
    sigmas = [0.2, 0.5, 0.8, 1.0]
    ax = axs[0]
    eps0 = np.array([-0.4, 0.6])
    for k, sg in enumerate(sigmas):
        zs = (1 - sg) * z1 + sg * eps0
        p = integrate(zs, "replace", 0.6, 0.6, t_start=1 - sg, n=40)
        ax.plot(*p.T, lw=2, color=plt.cm.plasma(0.15 + 0.2 * k), label=f"σ_s = {sg}")
        ax.plot(*zs, "o", color=plt.cm.plasma(0.15 + 0.2 * k), ms=6)
    ax.plot(*z1, "k*", ms=15, label="latente sorgente $z_1$")
    for c, col in [("src", C_SRC), ("tar", C_TAR), ("away", C_AWAY)]:
        ax.add_patch(Circle(M[c], S[c], fill=False, ec=col, lw=1.2, alpha=0.7))
    ax.set_aspect("equal"); ax.set_xlim(-1, 4); ax.set_ylim(-1.8, 3.4)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("$z_s=(1-\\sigma_s)z_1+\\sigma_s z_0$, poi integrazione da $t_s=1-\\sigma_s$")
    ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1]
    ss = np.linspace(0.05, 1.0, 30)
    for a, col in [(0.3, C_GOLD), (0.8, C_STAR)]:
        d = []
        for sg in ss:
            vals = []
            for _ in range(60):
                e = rng.normal(size=2)
                zs = (1 - sg) * z1 + sg * e
                vals.append(np.linalg.norm(integrate(zs, "replace", a, a, t_start=1 - sg, n=30)[-1] - z1))
            d.append(np.mean(vals))
        ax.plot(ss, d, color=col, lw=2.2, label=f"μ=λ={a}")
    ax.set_xlabel("livello di rumore σ_s"); ax.set_ylabel("distanza finale dal latente sorgente")
    ax.set_title("Meno rumore ⇒ più struttura conservata")
    ax.legend(frameon=False)
    save(fig, "fig09_i2i_rumore")


# --------------------------------------------------------------------------- #
#  FIG 10-13 - risultati del paper (numeri trascritti dalle tabelle)
# --------------------------------------------------------------------------- #
def grouped_bars(ax, labels, series, colors, ylabel, title, ylim=None, fmt="{:.2f}", highlight_last=True):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(labels))
    for i, (nm, vals) in enumerate(series):
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=nm, color=colors[i])
        for xi, v in zip(x + (i - (n - 1) / 2) * w, vals):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    if ylim: ax.set_ylim(*ylim)


def fig10_table1():
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.4))
    lab_f = ["FLUX", "+UCE", "+ESD", "+EA", "+Steering\nFields"]
    col_f = ["#9a9a9a", "#56B4E9", "#E69F00", "#F0E442", C_STAR]
    # NudeNet FLUX
    def bars(ax, vals, labs, cols, title, ylab, ylim, fmt):
        b = ax.bar(range(len(vals)), vals, color=cols, edgecolor="k", linewidth=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylab); ax.set_ylim(*ylim)
        b[-1].set_linewidth(2.2)
    bars(axs[0, 0], [70.67, 72.25, 41.93, 62.98, 37.32], lab_f, col_f, "FLUX – Ring-a-Bell – NudeNet ↓", "% rilevato", (0, 85), "{:.2f}")
    bars(axs[0, 1], [55.67, 56.13, 30.97, 38.92, 24.43], lab_f, col_f, "FLUX – P4D – NudeNet ↓", "% rilevato", (0, 70), "{:.2f}")
    bars(axs[1, 0], [0.79, 0.74, 0.68, 0.74, 0.52], lab_f, col_f, "FLUX – Ring-a-Bell – VQAScore_detect ↓", "P('Yes')", (0, 0.95), "{:.2f}")
    bars(axs[1, 1], [0.64, 0.55, 0.57, 0.51, 0.33], lab_f, col_f, "FLUX – P4D – VQAScore_detect ↓", "P('Yes')", (0, 0.8), "{:.2f}")
    fig.suptitle("Tabella 1 (FLUX1): sicurezza. Più basso = meno contenuto NSFW", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig10_tabella1_flux_sicurezza")

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
    lab_s = ["SD3.5", "+SAFREE", "+Steering\nFields"]
    col_s = ["#9a9a9a", "#56B4E9", C_STAR]
    bars(axs[0], [47.33, 36.14, 29.02], lab_s, col_s, "SD3.5 – Ring-a-Bell – NudeNet ↓", "% rilevato", (0, 58), "{:.2f}")
    bars(axs[1], [44.27, 33.31, 23.39], lab_s, col_s, "SD3.5 – P4D – NudeNet ↓", "% rilevato", (0, 55), "{:.2f}")
    # retention COCO (FLUX): CLIP, VQAScore, FID
    ax = axs[2]
    meth = ["FLUX", "UCE", "ESD", "EA", "SF"]
    fid = [0.0, 38.31, 47.62, 28.54, 34.16]
    bars(ax, fid, meth, col_f, "FLUX – COCO-1k – FID ↓ (retention)", "FID", (0, 58), "{:.2f}")
    fig.suptitle("Tabella 1 (SD3.5 e retention): CLIP/VQAScore COCO sono praticamente invariati per tutti", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig11_tabella1_sd35_retention")


def fig12_table2_editing():
    meths = ["FLUX\nImg2Img", "RF-\nInversion", "Stable\nFlow", "FlowEdit", "Steering\nFields"]
    data = {
        "CLIP-txt ↑": [0.255, 0.259, 0.242, 0.260, 0.271],
        "CLIP-img ↑ (preserv.)": [0.812, 0.806, 0.919, 0.865, 0.809],
        "CLIP-dir ↑": [0.093, 0.108, 0.075, 0.116, 0.127],
        "VQAScore ↑": [0.673, 0.696, 0.592, 0.702, 0.770],
        "HPSv2 ↑": [0.257, 0.278, 0.252, 0.287, 0.284],
    }
    cols = ["#9a9a9a", "#56B4E9", "#E69F00", "#F0E442", C_STAR]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.9))
    for ax, (k, v) in zip(axs, data.items()):
        b = ax.bar(range(5), v, color=cols, edgecolor="k", lw=0.6)
        best = int(np.argmax(v))
        b[best].set_linewidth(2.4)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        lo = min(v) * 0.9
        ax.set_ylim(lo if k != "CLIP-dir ↑" else 0, max(v) * 1.15)
        ax.set_xticks(range(5)); ax.set_xticklabels(meths, fontsize=6.5)
        ax.set_title(k, fontsize=10)
    fig.suptitle("Tabella 2 – PieBench++ (bordo spesso = migliore). Steering Fields guida CLIP-txt, CLIP-dir, VQAScore; "
                 "StableFlow guida CLIP-img; FlowEdit guida HPSv2", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig12_tabella2_editing")


def fig13_ablation_activation():
    layers = {16: [(2, 65.03, 35.66), (3, 64.08, 35.67), (4, 62.97, 36.26), (5, 57.12, 37.21), (6, 48.26, 38.79), (7, 41.77, 40.57)],
              17: [(2, 69.94, 36.65), (3, 64.87, 37.28), (4, 63.77, 38.13), (5, 58.86, 39.14), (6, 53.32, 40.30), (7, 50.16, 41.37)]}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axs[0]
    for L, col in [(16, C_SRC), (17, C_GOLD)]:
        d = np.array(layers[L])
        ax.plot(d[:, 2], d[:, 1], "-o", color=col, lw=1.8, label=f"activation steering, layer {L}")
        for lam, nn, ff in d:
            ax.annotate(f"λ={int(lam)}", (ff, nn), (ff + 0.1, nn + 0.9), fontsize=7, color=col)
    ax.plot(34.16, 37.32, "*", color=C_STAR, ms=20, mec="k", label="Steering Fields")
    ax.annotate("meglio\n(in basso a sinistra)", (35.3, 38), (36.6, 33), fontsize=8, color=C_GREY,
                arrowprops=dict(arrowstyle="->", color=C_GREY))
    ax.set_xlabel("FID ↓"); ax.set_ylabel("NudeNet ↓")
    ax.set_title("Tabella 5: compromesso sicurezza vs. qualità")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axs[1]
    labs = ["FLUX", "UCE", "ESD", "EA", "SF"]
    cols = ["#9a9a9a", "#56B4E9", "#E69F00", "#F0E442", C_STAR]
    x = np.arange(2); w = 0.16
    rp = [0.80, 0.75, 0.63, 0.61, 0.55]
    sv = [0.65, 0.56, 0.40, 0.37, 0.34]
    for i in range(5):
        ax.bar(x + (i - 2) * w, [rp[i], sv[i]], w, color=cols[i], edgecolor="k", lw=0.6, label=labs[i])
        for xi, v in zip(x + (i - 2) * w, [rp[i], sv[i]]):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(["T2I-RiskyPrompt ↓", "T2I-SafetyViolence ↓"])
    ax.set_title("Tabella 4: categoria Violenza (FLUX)")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    save(fig, "fig13_ablation_violenza")


# --------------------------------------------------------------------------- #
#  FIG 14 - riepilogo: relazione tra i formalismi
# --------------------------------------------------------------------------- #
def fig14_relations():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    def arr(p, q, txt, off=(0, 0.15)):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color="#444", lw=1.6))
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], txt, ha="center", fontsize=8.5, color="#444")

    box(3.6, 3.5, 4.8, 1.2, "REPLACE (Eq. 4–5)\n$v^*=\\frac{v_{src}+\\mu v_{tar}-\\lambda v_{away}}{1+\\mu-\\lambda}$", "#f9ecf3", C_STAR, 10)
    box(0.2, 1.4, 3.6, 1.3, "BLEND (Eq. 6–7)\n$v^*=(1-\\alpha)v_{src}+\\alpha v_{tar}$", "#e8f6ef", C_TAR, 10)
    box(4.2, 0.2, 3.8, 1.3, "Steering classico (Eq. 1)\n$v^*=v_{src}+\\alpha\\,v^\\Delta$, $v^\\Delta=v_{tar}-v_{away}$", "#e3f0fa", C_SRC, 9.5)
    box(8.3, 1.4, 3.5, 1.3, "Editing I2I\nstesso operatore, ma\n$z_s$ parte dall'immagine rumorosa", "#fff6e0", C_GOLD, 9.5)
    arr((4.8, 3.5), (2.5, 2.7), "λ = 0", (-0.55, 0.25))
    arr((6.0, 3.5), (6.0, 1.5), "μ = λ  (α = β = μ)  e  $v^\\Delta$ fisso", (1.55, 0.2))
    arr((7.2, 3.5), (9.5, 2.7), "condizionare su $z_s$ invece\nche su rumore puro", (0.8, 0.35))
    save(fig, "fig14_relazioni_formalismi")


# --------------------------------------------------------------------------- #
def main():
    global OUT
    OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
    os.makedirs(OUT, exist_ok=True)
    verify()
    print("\n== FIGURE ==")
    for f in (fig01_overview, fig02_flow_matching, fig03_loss_geometry, fig04_convexity_and_maps,
              fig05_decomposition, fig06_constant_vs_field, fig07_algorithm, fig08_blending,
              fig09_i2i_noise, fig10_table1, fig12_table2_editing, fig13_ablation_activation, fig14_relations):
        f()
    print("fatto.")


if __name__ == "__main__":
    main()
