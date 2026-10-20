#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genera_figure_transformer.py
----------------------------
Genera TUTTE le figure SVG usate nella nota 4 (Sequence Models II: Transformer Networks).
Uso:   python3 genera_figure_transformer.py
       (scrive i file .svg nella STESSA cartella dello script e stampa i numeri calcolati)
Dipendenze: numpy, matplotlib.

Le figure con numeri (matrici di attention, codifiche posizionali, varianza del prodotto
scalare, complessita', non-local means, ...) sono CALCOLATE qui, non disegnate a mano.
Gli schemi (blocchi, frecce) sono disegnati con le piccole funzioni di utilita' qui sotto.
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["svg.fonttype"] = "none"          # testo come testo (non tracciati)
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"

# ------------------------------------------------------------------ palette
C = dict(blue="#cfe2ff", blue_d="#2b5fb4", green="#d6f0d4", green_d="#2e7d32",
         red="#f9d0d0", red_d="#b3261e", yellow="#fff3c4", yellow_d="#a67c00",
         gray="#eceff1", gray_d="#455a64", purple="#e6d9f5", purple_d="#6a3fa0",
         orange="#ffe0b2", orange_d="#e65100", white="#ffffff", ink="#1b1b1b",
         teal="#d0f0ee", teal_d="#00796b")

# ------------------------------------------------------------------ mini-markup testo
def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def mk(s):
    """Mini-markup: _x oppure _{xyz} = pedice;  ^x oppure ^{xyz} = apice."""
    out, buf, i = [], "", 0
    def flush():
        nonlocal buf
        if buf:
            out.append(_esc(buf)); buf = ""
    while i < len(s):
        ch = s[i]
        if ch in "_^" and i + 1 < len(s):
            if s[i + 1] == "{":
                j = s.index("}", i); sub = s[i + 2:j]; i = j + 1
            else:
                sub = s[i + 1]; i += 2
            flush()
            dy = 4 if ch == "_" else -6
            out.append(f'<tspan font-size="70%" dy="{dy}">{_esc(sub)}</tspan><tspan dy="{-dy}">\u200b</tspan>')
            continue
        buf += ch; i += 1
    flush()
    return "".join(out)

def lerp_color(c0, c1, t):
    t = max(0.0, min(1.0, float(t)))
    return "#%02x%02x%02x" % tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))

def hex2rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

# ------------------------------------------------------------------ classe SVG
class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.e, self.markers = w, h, [], set()
        self.e.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>')

    def _m(self, color):
        self.markers.add(color); return "url(#m_%s)" % color.lstrip("#")

    def rect(self, x, y, w, h, fill="#fff", stroke="#37474f", sw=1.2, rx=4, dash=None, op=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{op}"' if op is not None else ""
        self.e.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def circle(self, x, y, r, fill="#fff", stroke="#37474f", sw=1.2):
        self.e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=13, anchor="middle", fill=C["ink"], weight="normal", italic=False, rot=None):
        st = ' font-style="italic"' if italic else ""
        r = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ""
        lines = s.split("\n")
        for k, ln in enumerate(lines):
            self.e.append(f'<text x="{x:.1f}" y="{y + k * size * 1.25:.1f}" font-family="{FONT}" font-size="{size}" '
                          f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"{st}{r}>{mk(ln)}</text>')

    def line(self, x1, y1, x2, y2, color="#37474f", sw=1.3, dash=None, op=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{op}"' if op is not None else ""
        self.e.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}{o}/>')

    def arrow(self, x1, y1, x2, y2, color="#37474f", sw=1.5, dash=None, op=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{op}"' if op is not None else ""
        self.e.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" marker-end="{self._m(color)}"{d}{o}/>')

    def path(self, d, color="#37474f", sw=1.5, arrow=False, dash=None, fill="none", op=None):
        m = f' marker-end="{self._m(color)}"' if arrow else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{op}"' if op is not None else ""
        self.e.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{m}{da}{o}/>')

    def box(self, x, y, w, h, label, fill="#fff", stroke="#37474f", size=13, sw=1.2, rx=5, dash=None, weight="normal", tcolor=None):
        self.rect(x, y, w, h, fill, stroke, sw, rx, dash)
        n = label.count("\n") + 1
        self.text(x + w / 2, y + h / 2 + size * 0.35 - (n - 1) * size * 0.625, label, size, fill=tcolor or C["ink"], weight=weight)

    def badge(self, x, y, n, color=None):
        self.circle(x, y, 9, color or C["orange"], C["orange_d"], 1.2)
        self.text(x, y + 4, str(n), 11, weight="bold", fill=C["orange_d"])

    def heat(self, M, x, y, cw, ch, vmin=None, vmax=None, fmt="{:.2f}", base="#2b5fb4", size=11,
             rows=None, cols=None, label_size=11, rot_cols=False, text_thresh=0.55, show=True, gray=False, mask_label=None):
        """Disegna una matrice come heat-map con i numeri dentro le celle."""
        M = np.asarray(M, dtype=float); R, K = M.shape
        finite = M[np.isfinite(M)]
        lo = finite.min() if vmin is None else vmin
        hi = finite.max() if vmax is None else vmax
        rgb = hex2rgb(base)
        for r in range(R):
            for c in range(K):
                v = M[r, c]
                if not np.isfinite(v):
                    t = 0.0; fill = "#f3f3f3"; lab = mask_label if mask_label is not None else "-∞"
                else:
                    t = (v - lo) / (hi - lo + 1e-12)
                    fill = lerp_color((0, 0, 0), (255, 255, 255), t) if gray else lerp_color((255, 255, 255), rgb, t)
                    lab = fmt.format(v)
                self.rect(x + c * cw, y + r * ch, cw, ch, fill, "#90a4ae", 0.8, 0)
                if show:
                    dark = (t > text_thresh) and np.isfinite(v)
                    self.text(x + c * cw + cw / 2, y + r * ch + ch / 2 + size * 0.35, lab, size,
                              fill="#fff" if dark else C["ink"])
        if rows:
            for r, lab in enumerate(rows):
                self.text(x - 6, y + r * ch + ch / 2 + label_size * 0.35, lab, label_size, anchor="end")
        if cols:
            for c, lab in enumerate(cols):
                if rot_cols:
                    self.text(x + c * cw + cw / 2 + 4, y - 6, lab, label_size, anchor="start", rot=-60)
                else:
                    self.text(x + c * cw + cw / 2, y - 6, lab, label_size)

    def save(self, name):
        defs = "".join(
            f'<marker id="m_{c.lstrip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in sorted(self.markers))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">'
               f'<defs>{defs}</defs>' + "".join(self.e) + "</svg>")
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(svg)
        print("scritto", name)

def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

NUMS = {}   # numeri calcolati, stampati a fine esecuzione e riusati nella nota

# =====================================================================================
# FIG 1 - seq2seq con RNN e collo di bottiglia (slide 6-12)
# =====================================================================================
def fig_s2s_bottleneck():
    s = Svg(860, 340)
    s.text(430, 24, "Seq2seq con RNN: tutta la frase passa da un solo vettore c", 15, weight="bold")
    xs = [30, 115, 200, 285]; words = ["we", "are", "eating", "bread"]
    s.text(190, 118, "Encoder:  h_t = f_W(x_t, h_{t-1})", 13, fill=C["green_d"], weight="bold")
    for i, (x, wd) in enumerate(zip(xs, words)):
        s.box(x, 160, 60, 36, f"h_{i+1}", C["green"], C["green_d"])
        s.box(x, 232, 60, 30, wd, C["red"], C["red_d"], 12)
        s.arrow(x + 30, 232, x + 30, 197, C["red_d"])
        if i: s.arrow(xs[i - 1] + 60, 178, x - 1, 178, C["green_d"])
    s.box(385, 160, 70, 36, "c = h_T", C["red"], C["red_d"], 13, dash="5,3", sw=1.8)
    s.arrow(346, 178, 384, 178, C["green_d"])
    s.text(420, 222, "collo di bottiglia:\nun vettore di taglia\nfissa; e se T = 1000?", 11, fill=C["red_d"])
    ys = ["estamos", "comiendo", "pan", "[STOP]"]; ins = ["[START]", "estamos", "comiendo", "pan"]
    dx = [520, 600, 680, 760]
    s.text(650, 48, "Decoder:  s_t = g_U(y_{t-1}, s_{t-1}, c)", 13, fill=C["blue_d"], weight="bold")
    for i, x in enumerate(dx):
        s.box(x, 160, 60, 36, f"s_{i+1}", C["blue"], C["blue_d"])
        s.box(x, 70, 60, 30, ys[i], C["yellow"], C["yellow_d"], 11)
        s.arrow(x + 36, 160, x + 36, 101, C["blue_d"])
        s.box(x, 232, 60, 30, ins[i], C["red"], C["red_d"], 11)
        s.arrow(x + 30, 232, x + 30, 197, C["red_d"])
        s.path(f"M420,160 L420,138 L{x+18},138 L{x+18},159", C["red_d"], 1.1, arrow=True, dash="4,3")
        if i: s.arrow(dx[i - 1] + 60, 178, x - 1, 178, C["blue_d"])
    s.arrow(456, 178, 519, 178, C["red_d"])
    s.text(430, 300, "c (tratteggiato rosso) entra in OGNI passo del decoder e inizializza s_0 (slide 7-11).", 12, fill=C["gray_d"])
    s.text(430, 320, "Idea (slide 12): «refer to the inputs while we decode» → attention", 12, fill=C["gray_d"], italic=True)
    s.save("fig_s2s_bottleneck.svg")

# =====================================================================================
# FIG 2 - un passo di attention alla Bahdanau (slide 13-22)
# =====================================================================================
def fig_attention_step():
    s = Svg(900, 450)
    s.text(450, 24, "Un passo di decoder con attention (Bahdanau): dal calcolo di e_{1,i} a c_1", 15, weight="bold")
    xs = [50, 140, 230, 320]; words = ["we", "are", "eating", "bread"]
    a = [0.45, 0.45, 0.05, 0.05]
    for i, x in enumerate(xs):
        s.box(x, 250, 70, 34, f"h_{i+1}", C["green"], C["green_d"])
        s.box(x, 312, 70, 28, words[i], C["red"], C["red_d"], 12)
        s.arrow(x + 35, 312, x + 35, 285, C["red_d"])
        if i: s.arrow(xs[i - 1] + 70, 267, x - 1, 267, C["green_d"], 1.2)
        s.box(x, 185, 70, 28, f"e_{{1,{i+1}}}", C["yellow"], C["yellow_d"], 12)
        s.arrow(x + 22, 250, x + 22, 214, C["green_d"], 1.2)
        s.arrow(x + 50, 232, x + 50, 214, C["purple_d"], 1.2)
        v = a[i]
        s.rect(x, 75, 70, 30, lerp_color((255, 255, 255), (106, 63, 160), v / 0.5), C["purple_d"], 1.2, 4)
        s.text(x + 35, 95, f"a_{{1,{i+1}}}={v:.2f}", 11, fill="#fff" if v > 0.3 else C["ink"])
        s.arrow(x + 35, 135, x + 35, 106, C["purple_d"], 1.2)
        s.arrow(x + 35, 185, x + 35, 162, C["yellow_d"], 1.2)
    s.rect(50, 136, 340, 26, C["purple"], C["purple_d"], 1.2, 5)
    s.text(220, 154, "softmax  (0 < a_{1,i} < 1,  Σ_i a_{1,i} = 1)", 12, fill=C["purple_d"], weight="bold")
    s.box(520, 250, 70, 34, "s_0", C["blue"], C["blue_d"])
    s.path("M555,250 L555,232 L98,232", C["purple_d"], 1.2, dash="4,3")
    s.circle(470, 90, 14, "#fff", C["gray_d"], 1.5); s.text(470, 95, "Σ", 15, weight="bold")
    s.arrow(390, 90, 455, 90, C["purple_d"], 1.5)
    s.arrow(484, 90, 520, 90, C["gray_d"], 1.5)
    s.box(520, 75, 60, 30, "c_1", C["red"], C["red_d"], 13)
    s.path("M580,90 L690,90 L690,162", C["red_d"], 1.5, arrow=True)
    s.box(655, 163, 70, 34, "s_1", C["blue"], C["blue_d"])
    s.path("M590,267 L610,267 L610,180 L654,180", C["blue_d"], 1.5, arrow=True)
    s.box(655, 250, 70, 34, "[START]", C["red"], C["red_d"], 11)
    s.arrow(690, 250, 690, 198, C["red_d"], 1.5)
    s.arrow(725, 180, 760, 180, C["blue_d"], 1.5)
    s.box(760, 165, 90, 30, "estamos", C["yellow"], C["yellow_d"], 12)
    s.text(690, 306, "y_0", 11, fill=C["red_d"]); s.text(806, 212, "y_1", 11, fill=C["yellow_d"])
    s.badge(32, 199, 1); s.badge(32, 149, 2); s.badge(466, 66, 3); s.badge(640, 150, 4)
    L = ["① e_{1,i} = f_{att}(s_0, h_i)   (f_{att} = piccola MLP, slide 14)",
         "② a_{1,i} = softmax over i of e_{1,i}   (slide 15)",
         "③ c_1 = Σ_i a_{1,i} h_i   (slide 16)",
         "④ s_1 = g_U(y_0, s_0, c_1)  →  y_1 = «estamos»   (slide 17)"]
    for k, t in enumerate(L):
        s.text(420, 336 + k * 20, t, 12, anchor="start", fill=C["gray_d"])
    s.text(30, 438, "Numeri a_{1,i} = quelli «intuitivi» della slide 17: «estamos» = «we are» ⇒ a_{11}=a_{12}=0.45, a_{13}=a_{14}=0.05.", 11, anchor="start", fill=C["gray_d"], italic=True)
    s.save("fig_attention_step.svg")

# =====================================================================================
# FIG 3 - mappe di attention: esempio giocattolo (slide 17-21) e struttura EN->FR (slide 24-26)
# =====================================================================================
def fig_attention_maps():
    s = Svg(1000, 480)
    s.text(500, 24, "Pesi di attention a_{t,i}: esempio giocattolo e struttura reale (EN→FR)", 15, weight="bold")
    src = ["we", "are", "eating", "bread"]; tgt = ["estamos", "comiendo", "pan", "[STOP]"]
    A = np.array([[.45, .45, .05, .05], [.05, .10, .80, .05], [.03, .02, .05, .90], [.25, .25, .25, .25]])
    s.text(215, 60, "decoder (righe) guarda l'encoder (colonne)", 12, fill=C["gray_d"])
    s.heat(A, 120, 110, 62, 44, 0, 1, "{:.2f}", "#6a3fa0", 12, rows=tgt, cols=src, label_size=12)
    for r in range(4): s.text(120 + 4 * 62 + 28, 110 + r * 44 + 27, f"Σ={A[r].sum():.2f}", 11, anchor="start", fill=C["gray_d"])
    s.text(215, 310, "righe 1-2: valori «intuitivi» delle slide 17 e 21\n(a_{11}=a_{12}=0.45; a_{21}=a_{24}=0.05, a_{22}=0.1, a_{23}=0.8)\nrighe 3-4: aggiunte per completare l'esempio (illustrative)", 11, fill=C["gray_d"])
    en = ["The", "agreement", "on", "the", "European", "Economic", "Area", "was", "signed", "in", "August", "1992", ".", "<end>"]
    fr = ["L'", "accord", "sur", "la", "zone", "économique", "européenne", "a", "été", "signé", "en", "août", "1992", ".", "<end>"]
    main = {0: [0], 1: [1], 2: [2], 3: [3], 4: [6], 5: [5], 6: [4], 7: [7], 8: [7, 8], 9: [8], 10: [9], 11: [10], 12: [11], 13: [12], 14: [13]}
    B = np.full((15, 14), 0.01)
    for r, cs in main.items():
        for c in cs: B[r, c] += 0.8 / len(cs)
    B /= B.sum(axis=1, keepdims=True)
    s.text(790, 60, "schema EN→FR (slide 24-26): struttura, non pesi reali", 12, fill=C["gray_d"])
    s.heat(B, 705, 140, 17, 17, 0, 0.6, base="#000000", rows=fr, cols=en, label_size=10, rot_cols=True, show=False, gray=True)
    s.rect(705, 140, 14 * 17, 15 * 17, "none", C["gray_d"], 1, 0)
    s.rect(705, 140, 4 * 17, 4 * 17, "none", "#1e88e5", 2.2, 0)       # L'accord sur la
    s.rect(705 + 4 * 17, 140 + 4 * 17, 3 * 17, 3 * 17, "none", C["red_d"], 2.2, 0)   # anti-diagonale
    s.rect(705 + 9 * 17, 140 + 10 * 17, 5 * 17, 5 * 17, "none", "#1e88e5", 2.2, 0)   # in August 1992
    s.text(790, 425, "bordo blu: diagonale = parole nello stesso ordine\nbordo rosso: «zone économique européenne» ↔\n«European Economic Area» (ordine invertito)", 11, fill=C["gray_d"])
    s.save("fig_attention_maps.svg")

# =====================================================================================
# FIG 4 - captioning con attention spaziale (slide 28-38)
# =====================================================================================
def fig_caption_attention():
    s = Svg(1000, 470)
    s.text(500, 24, "Image captioning con attention (Show, Attend and Tell): un c_t diverso a ogni parola", 15, weight="bold")
    e = {1: np.array([[.2, .4, .1], [.9, 2.4, .8], [.3, 1.0, .2]]),
         3: np.array([[.3, 1.9, 2.6], [.1, .5, .7], [-.2, 0., .1]])}
    words = {1: "person", 3: "hat"}
    s.text(95, 62, "feature spaziali z", 12, weight="bold")
    s.text(95, 78, "(H×W×D = 3×3×D)", 11, fill=C["gray_d"])
    for i in range(3):
        for j in range(3):
            s.rect(30 + j * 42, 92 + i * 42, 42, 42, lerp_color((255, 255, 255), (46, 125, 50), 0.15 + 0.1 * ((i * 3 + j) % 4)), "#90a4ae", 0.8, 0)
            s.text(30 + j * 42 + 21, 92 + i * 42 + 25, f"z_{{{i},{j}}}", 11)
    s.box(30, 240, 126, 40, "CNN\npre-addestrata", C["red"], C["red_d"], 11)
    s.arrow(93, 240, 93, 222, C["red_d"])
    s.text(93, 296, "(immagine in ingresso)", 11, fill=C["gray_d"])
    NUMS["cap_a"] = {}
    for row, t in enumerate([1, 3]):
        y0 = 70 + row * 190
        a = softmax(e[t].ravel()).reshape(3, 3)
        NUMS["cap_a"][t] = a.round(3).tolist()
        s.text(285, y0 - 10, f"t = {t}", 12, weight="bold", fill=C["purple_d"])
        s.text(285, y0 + 4, "e_{t,i,j}", 12, fill=C["yellow_d"])
        s.heat(e[t], 230, y0 + 12, 42, 36, base="#a67c00", size=11, fmt="{:.1f}")
        s.arrow(362, y0 + 66, 410, y0 + 66, C["purple_d"]); s.text(386, y0 + 58, "softmax", 10, fill=C["purple_d"])
        s.text(500, y0 + 4, "a_{t,i,j}  (Σ = 1)", 12, fill=C["purple_d"])
        s.heat(a, 430, y0 + 12, 42, 36, 0, 0.6, "{:.2f}", "#6a3fa0", 11)
        s.arrow(562, y0 + 66, 620, y0 + 66, C["gray_d"]); s.text(590, y0 + 58, "Σ a·z", 11)
        s.box(622, y0 + 46, 70, 40, f"c_{t}", C["red"], C["red_d"], 13)
        s.arrow(692, y0 + 66, 735, y0 + 66, C["gray_d"])
        s.box(736, y0 + 46, 80, 40, f"h_{t}", C["blue"], C["blue_d"], 13)
        s.arrow(816, y0 + 66, 850, y0 + 66, C["gray_d"])
        s.box(851, y0 + 46, 120, 40, f"y_{t} = «{words[t]}»", C["yellow"], C["yellow_d"], 12)
    s.text(500, 448, "c_t = Σ_{i,j} a_{t,i,j} z_{i,j}   ·   decoder: y_t = g_V(y_{t-1}, h_{t-1}, c_t)   ·   tutto differenziabile, nessuna supervisione sui pesi a", 12, fill=C["gray_d"])
    s.save("fig_caption_attention.svg")

# =====================================================================================
# Self-attention numerica (usata da FIG 5, 9, 11)
# =====================================================================================
EX = dict(
    X=np.array([[1, 0, 1, 0], [0, 2, 0, 1], [1, 1, 1, 1]], float),
    Wq=np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float),
    Wk=np.array([[0, 1], [1, 0], [0, 1], [1, 0]], float),
    Wv=np.array([[1, 0], [0, 1], [1, 1], [0, 0]], float))

def sa_forward(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    E = Q @ K.T / np.sqrt(Wq.shape[1])
    A = softmax(E, axis=1)
    return Q, K, V, E, A, A @ V

def fig_selfattn_example():
    Q, K, V, E, A, Y = sa_forward(**EX)
    NUMS["sa"] = dict(Q=Q.tolist(), K=K.tolist(), V=V.tolist(), E=E.round(3).tolist(), A=A.round(3).tolist(), Y=Y.round(3).tolist())
    s = Svg(1090, 440)
    s.text(545, 24, "Self-attention su 3 vettori (N=3, D=4, D_k=D_v=2): tutti i passaggi con numeri", 15, weight="bold")
    s.text(108, 80, "X  (N×D)", 12, weight="bold"); s.heat(EX["X"], 40, 92, 34, 30, fmt="{:.0f}", base="#2e7d32", rows=["x_0", "x_1", "x_2"])
    ys = [58, 168, 278]; names = ["Q = X W_q", "K = X W_k", "V = X W_v"]; mats = [Q, K, V]; cols = ["#b3261e", "#2b5fb4", "#2e7d32"]
    for y, nm, M, cl in zip(ys, names, mats, cols):
        s.text(296, y - 8, nm + "  (N×2)", 12, weight="bold", fill=cl)
        s.heat(M, 258, y, 38, 30, fmt="{:.0f}", base=cl)
        s.arrow(178, 140, 250, y + 45, C["gray_d"], 1.2)
    s.text(584, 150, "E = Q Kᵀ / √D_k", 12, weight="bold"); s.text(584, 166, "(riga j = query j, colonna i = chiave i)", 10, fill=C["gray_d"])
    s.heat(E, 500, 186, 56, 36, fmt="{:.2f}", base="#a67c00", rows=["q_0", "q_1", "q_2"], cols=["k_0", "k_1", "k_2"])
    s.arrow(338, 103, 466, 192, C["gray_d"], 1.2); s.arrow(338, 213, 466, 213, C["gray_d"], 1.2)
    s.text(804, 150, "A = softmax per riga", 12, weight="bold", fill=C["purple_d"])
    s.heat(A, 720, 186, 56, 36, 0, 1, "{:.2f}", "#6a3fa0", cols=["k_0", "k_1", "k_2"])
    s.text(804, 306, "(ogni riga somma a 1)", 10, fill=C["gray_d"])
    s.arrow(672, 204, 706, 204, C["purple_d"], 1.5)
    s.text(1010, 150, "Y = A V  (N×2)", 12, weight="bold", fill=C["green_d"])
    s.heat(Y, 962, 186, 48, 36, fmt="{:.2f}", base="#2e7d32", rows=["y_0", "y_1", "y_2"])
    s.arrow(902, 204, 930, 204, C["green_d"], 1.5)
    s.path("M338,323 L640,323 L640,376 L1010,376 L1010,297", C["green_d"], 1.2, arrow=True, dash="4,3")
    s.text(825, 394, "y_j = Σ_i A_{j,i} v_i   (media pesata dei valori V)", 11, fill=C["green_d"])
    s.text(545, 424, "Esempio: q_1 = (0,3) dà lo stesso punteggio (6/√2 = 4.24) a k_0 e k_2 e 0 a k_1 ⇒ A_1 ≈ (0.50, 0.01, 0.50).", 11, fill=C["gray_d"])
    s.save("fig_selfattn_example.svg")

# =====================================================================================
# FIG 6 - perche' dividere per sqrt(d_k) (slide 39-40, formula e_ij = q_j . k_i / sqrt(D))
# =====================================================================================
def fig_scale_sqrt():
    rng = np.random.default_rng(0)
    dks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    std_raw, std_sc, maxw_raw, maxw_sc = [], [], [], []
    for dk in dks:
        q = rng.standard_normal((4000, dk)); k = rng.standard_normal((4000, dk))
        dots = (q * k).sum(1)
        std_raw.append(dots.std()); std_sc.append((dots / np.sqrt(dk)).std())
        Kmat = rng.standard_normal((4000, 10, dk)); qv = rng.standard_normal((4000, dk))
        logit = np.einsum("bnd,bd->bn", Kmat, qv)
        maxw_raw.append(softmax(logit, axis=1).max(1).mean())
        maxw_sc.append(softmax(logit / np.sqrt(dk), axis=1).max(1).mean())
    NUMS["scale"] = dict(dk=dks, std_raw=np.round(std_raw, 2).tolist(), std_sc=np.round(std_sc, 2).tolist(),
                         maxw_raw=np.round(maxw_raw, 2).tolist(), maxw_sc=np.round(maxw_sc, 2).tolist())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ax[0].plot(dks, std_raw, "o-", color="#b3261e", label="deviazione standard di q·k (misurata)")
    ax[0].plot(dks, np.sqrt(dks), "--", color="#555", label="√d_k (teoria)")
    ax[0].plot(dks, std_sc, "s-", color="#2e7d32", label="di q·k/√d_k (misurata)")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel("d_k"); ax[0].set_ylabel("deviazione standard del punteggio"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("(a) q, k con componenti i.i.d. N(0,1)", fontsize=10)
    ax[1].plot(dks, maxw_raw, "o-", color="#b3261e", label="senza scala")
    ax[1].plot(dks, maxw_sc, "s-", color="#2e7d32", label="con scala 1/√d_k")
    ax[1].axhline(0.1, color="#888", ls=":", label="uniforme (n=10 chiavi)")
    ax[1].set_xscale("log", base=2); ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("d_k"); ax[1].set_ylabel("peso massimo medio  max_i a_i")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("(b) la softmax senza scala diventa one-hot", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scale_sqrt.svg")); plt.close(fig); print("scritto fig_scale_sqrt.svg")

# =====================================================================================
# FIG 7 - lunghezza massima del cammino: RNN vs self-attention (slide 43)
# =====================================================================================
def fig_path_length():
    s = Svg(940, 575)
    s.text(470, 24, "Maximum path length tra «like» e «好き»: RNN contro self-attention", 15, weight="bold")
    en = ["I", "like", "cats", "more", "than", "dogs"]; jp = ["私", "は", "犬", "より", "猫", "が", "好き"]
    ex = [60 + i * 70 for i in range(6)]; dx = [500 + j * 62 for j in range(7)]
    s.text(120, 52, "RNN (encoder → decoder)", 13, anchor="start", weight="bold", fill=C["orange_d"])
    for i, x in enumerate(ex):
        s.box(x - 25, 120, 50, 30, "RNN", C["orange"], C["orange_d"], 11)
        s.box(x - 25, 175, 50, 26, en[i], C["blue"], C["blue_d"], 12); s.arrow(x, 175, x, 151, C["gray_d"], 1)
        if i: s.arrow(ex[i - 1] + 25, 135, x - 26, 135, C["gray_d"], 1)
    for j, x in enumerate(dx):
        s.box(x - 25, 120, 50, 30, "RNN", C["blue"], C["blue_d"], 11)
        s.box(x - 25, 68, 50, 28, jp[j], C["blue"], C["blue_d"], 13); s.arrow(x, 119, x, 97, C["gray_d"], 1)
        if j: s.arrow(dx[j - 1] + 25, 135, x - 26, 135, C["gray_d"], 1)
    s.arrow(ex[5] + 25, 135, dx[0] - 26, 135, C["gray_d"], 1)
    red = C["red_d"]; hops = 0
    for i in range(1, 5): s.arrow(ex[i] + 25, 129, ex[i + 1] - 26, 129, red, 3); hops += 1
    s.arrow(ex[5] + 25, 129, dx[0] - 26, 129, red, 3); hops += 1
    for j in range(6): s.arrow(dx[j] + 25, 129, dx[j + 1] - 26, 129, red, 3); hops += 1
    s.rect(ex[1] - 27, 173, 54, 30, "none", red, 2.5, 5); s.rect(dx[6] - 27, 66, 54, 32, "none", red, 2.5, 5)
    NUMS["rnn_hops"] = hops
    s.text(470, 232, f"cammino rosso = {hops} passi di ricorrenza  ⇒  lunghezza O(n)", 13, fill=red, weight="bold")
    o = 50
    s.text(120, 272, "Transformer (multi-head attention, schema semplificato)", 13, anchor="start", weight="bold", fill=C["blue_d"])
    tx = [60 + i * 70 for i in range(6)]
    for i, x in enumerate(tx):
        s.box(x - 25, 425 + o, 50, 26, en[i], C["blue"], C["blue_d"], 12)
        s.line(x, 424 + o, x, 395 + o, C["gray_d"], 1, op=.5)
    s.rect(36, 366 + o, 410, 28, C["blue"], C["blue_d"], 1.5, 6); s.text(241, 385 + o, "Multi-Head Attention (self, encoder)", 12, weight="bold", fill=C["blue_d"])
    s.rect(36, 308 + o, 840, 42, C["orange"], C["orange_d"], 1.5, 6); s.text(456, 341 + o, "Multi-Head Attention (decoder ↔ encoder)", 12, weight="bold", fill=C["orange_d"])
    for i, x in enumerate(tx): s.line(x, 366 + o, x, 351 + o, C["gray_d"], 1, op=.5)
    for j, x in enumerate(dx):
        s.box(x - 25, 270 + o, 50, 28, jp[j], C["blue"], C["blue_d"], 13)
        s.line(x, 308 + o, x, 299 + o, C["gray_d"], 1, op=.5)
    s.arrow(tx[1], 424 + o, tx[1], 396 + o, red, 3); s.arrow(tx[1], 366 + o, tx[1], 352 + o, red, 3)
    s.path(f"M{tx[1]},{351 + o} L{tx[1]},{318 + o} L{dx[6]},{318 + o} L{dx[6]},{300 + o}", red, 3, arrow=True)
    s.rect(tx[1] - 27, 423 + o, 54, 30, "none", red, 2.5, 5); s.rect(dx[6] - 27, 268 + o, 54, 32, "none", red, 2.5, 5)
    s.text(470, 540 + o, "tra due posizioni qualsiasi bastano 3 salti (1 per strato di attention) ⇒ lunghezza O(1)", 13, fill=red, weight="bold")
    s.save("fig_path_length.svg")

# =====================================================================================
# FIG 8 - tabella della slide 45 (complessita' per strato, cammino) in grafico
# =====================================================================================
def fig_complexity():
    n = np.logspace(1, 4, 200); d = 512; k = 3
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ax[0].loglog(n, n ** 2 * d, color="#b3261e", label="self-attention  O(n²·d)")
    ax[0].loglog(n, n * d ** 2, color="#2e7d32", label="ricorrente  O(n·d²)")
    ax[0].loglog(n, k * n * d ** 2, color="#2b5fb4", ls="--", label="convoluzionale  O(k·n·d²), k=3")
    ax[0].axvline(d, color="#888", ls=":"); ax[0].text(d * 1.06, 3e5, "n = d = 512", fontsize=8, color="#555", rotation=90)
    ax[0].set_xlabel("lunghezza della sequenza n"); ax[0].set_ylabel("operazioni per strato (ordine di grandezza)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both"); ax[0].set_title("(a) costo per strato (d = 512)", fontsize=10)
    ax[1].semilogx(n, n, color="#b3261e", label="ricorrente  O(n)")
    ax[1].semilogx(n, np.log(n) / np.log(k), color="#2b5fb4", ls="--", label="convoluzionale dilatata  O(log_k n)")
    ax[1].semilogx(n, np.ones_like(n), color="#2e7d32", label="self-attention  O(1)")
    ax[1].set_yscale("symlog", linthresh=10); ax[1].set_ylim(0, 1e4)
    ax[1].set_xlabel("lunghezza della sequenza n"); ax[1].set_ylabel("lunghezza massima del cammino")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both"); ax[1].set_title("(b) distanza tra due posizioni", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_complexity.svg")); plt.close(fig); print("scritto fig_complexity.svg")
    tab = {}
    for nn in (100, 512, 1000, 4096):
        tab[nn] = dict(sa=nn ** 2 * d, rnn=nn * d ** 2, conv=k * nn * d ** 2)
    NUMS["complexity"] = tab

# =====================================================================================
# FIG 9 - self-attention e' permutation-equivariant (slide 48-49)
# =====================================================================================
def fig_perm_equiv():
    X = EX["X"]; perm = [1, 0, 2]
    Q, K, V, E, A, Y = sa_forward(**EX)
    Q2, K2, V2, E2, A2, Y2 = sa_forward(X[perm], EX["Wq"], EX["Wk"], EX["Wv"])
    ok = np.allclose(Y2, Y[perm]); NUMS["perm_ok"] = bool(ok)
    assert ok
    cols = [(C["red"], C["red_d"]), (C["green"], C["green_d"]), (C["blue"], C["blue_d"])]
    s = Svg(900, 400)
    s.text(450, 24, "Self-attention è permutation-equivariant: Y(P·X) = P·Y(X)", 15, weight="bold")
    def group(x0, order, Ym, title):
        s.text(x0 + 150, 120, title, 13, weight="bold")
        for pos, idx in enumerate(order):
            x = x0 + pos * 105
            f, st = cols[idx]
            s.box(x, 300, 90, 32, "[" + " ".join(f"{v:.0f}" for v in X[idx]) + "]", f, st, 11)
            s.text(x + 45, 352, f"x_{idx}", 12, fill=st, weight="bold")
            s.arrow(x + 45, 299, x + 45, 232, C["gray_d"], 1.3)
            s.box(x, 175, 90, 32, f"[{Ym[pos][0]:.2f} {Ym[pos][1]:.2f}]", f, st, 11)
            s.text(x + 45, 160, f"y_{idx}", 12, fill=st, weight="bold")
            s.arrow(x + 45, 232, x + 45, 208, C["gray_d"], 1.3)
        s.box(x0 - 10, 225, 330, 40, "self-attention (stessi W_q, W_k, W_v)", C["gray"], C["gray_d"], 12)
    group(30, [0, 1, 2], Y, "ordine originale  (x_0, x_1, x_2)")
    group(500, perm, Y2, "ordine permutato  (x_1, x_0, x_2)")
    s.text(450, 385, "Le uscite sono le stesse, solo permutate: l'ordine delle parole non entra da nessuna parte ⇒ servono le codifiche posizionali.", 11, fill=C["gray_d"])
    s.save("fig_perm_equiv.svg")

# =====================================================================================
# FIG 10 - codifica posizionale sinusoidale (slide 50-54)
# =====================================================================================
def pos_enc(T, d):
    P = np.zeros((T, d)); t = np.arange(T)[:, None]
    wk = 1.0 / (10000 ** (2 * np.arange(d // 2) / d))
    P[:, 0::2] = np.sin(t * wk); P[:, 1::2] = np.cos(t * wk)
    return P

def fig_posenc():
    d = 64; P = pos_enc(64, d)
    fig = plt.figure(figsize=(12.5, 4.2))
    ax = fig.add_subplot(1, 3, 1)
    im = ax.pcolormesh(P, cmap="RdBu_r", vmin=-1, vmax=1, rasterized=True); ax.invert_yaxis()
    ax.set_xlabel("dimensione i (0..63)"); ax.set_ylabel("posizione t"); ax.set_title("(a) p(t): sin/cos alternati", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046)
    ax = fig.add_subplot(1, 3, 2)
    tt = np.arange(0, 100); d2 = 128
    for kk, col in [(0, "#b3261e"), (8, "#2b5fb4"), (24, "#2e7d32"), (48, "#6a3fa0")]:
        w = 1 / 10000 ** (2 * kk / d2)
        ax.plot(tt, np.sin(w * tt), color=col, label=f"k={kk}: ω={w:.4f}, λ={2*np.pi/w:.0f}")
    ax.set_xlabel("posizione t"); ax.set_ylabel(r"$\sin(\omega_k t)$"); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=.3)
    ax.set_title("(b) frequenze: da veloci a lentissime", fontsize=10)
    ax = fig.add_subplot(1, 3, 3)
    P2 = pos_enc(100, 128); Sm = P2 @ P2.T / 64.0
    im = ax.pcolormesh(Sm, cmap="viridis", vmin=0, vmax=1, rasterized=True); ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("posizione t'"); ax.set_ylabel("posizione t"); ax.set_title("(c) p(t)·p(t')/64 dipende solo da |t−t'|", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_posenc.svg")); plt.close(fig); print("scritto fig_posenc.svg")
    # numeri di controllo
    dots = [float(P2[10] @ P2[10 + dl] / 64) for dl in (0, 1, 5, 20)]
    dots2 = [float(P2[60] @ P2[60 + dl] / 64) for dl in (0, 1, 5, 20)]
    NUMS["pe_dot_t10"] = np.round(dots, 4).tolist(); NUMS["pe_dot_t60"] = np.round(dots2, 4).tolist()
    # verifica della rotazione p(t+D) = R(D) p(t)
    dd = 8; t0 = 5; dl = 3; wk = 1 / 10000 ** (2 * np.arange(dd // 2) / dd)
    p = pos_enc(20, dd)
    R = np.zeros((dd, dd))
    for kk in range(dd // 2):
        c_, s_ = np.cos(wk[kk] * dl), np.sin(wk[kk] * dl)
        R[2 * kk:2 * kk + 2, 2 * kk:2 * kk + 2] = [[c_, s_], [-s_, c_]]
    NUMS["pe_rot_ok"] = bool(np.allclose(R @ p[t0], p[t0 + dl]))
    NUMS["pe_bin_rows"] = [format(i, "04b") for i in range(16)]

# =====================================================================================
# FIG 11 - masked self-attention (slide 56)
# =====================================================================================
def fig_masked():
    E = np.array([[1.2, 0.3, -0.5, 0.8], [0.4, 1.5, 0.2, -0.3], [-0.2, 0.9, 1.1, 0.5], [0.6, -0.4, 0.7, 1.3]])
    N = 4
    mask = np.triu(np.ones((N, N), bool), k=1)       # chiave i > query j  => futuro
    Em = E.copy(); Em[mask] = -np.inf
    Af = softmax(E, axis=1); Am = softmax(Em, axis=1)
    NUMS["mask"] = dict(Af=Af.round(2).tolist(), Am=Am.round(3).tolist())
    toks = ["x_0", "x_1", "x_2", "x_3"]
    s = Svg(1110, 360)
    s.text(555, 24, "Masked self-attention: ogni query vede solo il passato (e se stessa)", 15, weight="bold")
    def blk(x, M, title, **kw):
        s.text(x + 2 * 52, 70, title, 12, weight="bold")
        s.heat(M, x, 100, 52, 40, rows=["q_0", "q_1", "q_2", "q_3"], cols=["k_0", "k_1", "k_2", "k_3"], **kw)
    blk(60, E, "E = QKᵀ/√D_k  (punteggi)", fmt="{:.1f}", base="#a67c00")
    blk(350, Em, "E con maschera (futuro = −∞)", fmt="{:.1f}", base="#a67c00", vmin=-0.5, vmax=1.5)
    blk(640, Am, "A = softmax per riga", fmt="{:.2f}", base="#6a3fa0", vmin=0, vmax=1)
    s.text(1008, 70, "senza maschera: A", 12, weight="bold")
    s.heat(Af, 920, 100, 44, 40, 0, 1, "{:.2f}", "#9e9e9e", 10, cols=["k_0", "k_1", "k_2", "k_3"])
    s.arrow(274, 180, 330, 180, C["gray_d"], 1.5); s.arrow(566, 180, 622, 180, C["gray_d"], 1.5)
    s.text(302, 172, "mask", 10, fill=C["gray_d"]); s.text(594, 172, "softmax", 10, fill=C["gray_d"])
    s.text(555, 290, "e^{−∞} = 0  ⇒  i pesi sul futuro sono ESATTAMENTE 0 e le righe restano normalizzate (la diagonale non è mai mascherata).", 12, fill=C["gray_d"])
    s.text(555, 312, "Si calcola tutta la matrice in un colpo (parallelo su t): non serve ottenere prima i context vector dei passi precedenti.", 12, fill=C["gray_d"])
    s.text(555, 334, "È la stessa idea della convoluzione causale della nota 3 (sez. 3.4): niente futuro.", 12, fill=C["gray_d"], italic=True)
    s.save("fig_masked.svg")

# =====================================================================================
# FIG 12 - multi-head self-attention (slide 58-59)
# =====================================================================================
def fig_multihead():
    s = Svg(960, 430)
    s.text(480, 24, "Multi-head self-attention: H teste in parallelo, poi concatenazione", 15, weight="bold")
    s.box(25, 150, 60, 110, "X\nN×d", C["red"], C["red_d"], 13)
    pats = [np.eye(4) * 0.7 + 0.1, np.roll(np.eye(4), -1, axis=1) * 0.7 + 0.1, np.tile(np.array([[.7, .1, .1, .1]]), (4, 1))]
    names = ["testa 0", "testa 1", "testa H−1"]; rows = [50, 150, 300]
    cols = [(C["blue"], C["blue_d"]), (C["green"], C["green_d"]), (C["orange"], C["orange_d"])]
    s.box(630, 50, 60, 330, "concat\n(N×d)", C["gray"], C["gray_d"], 13)
    for r, (y, nm, pat, (f, st)) in enumerate(zip(rows, names, pats, cols)):
        s.text(250, y - 2, nm, 12, weight="bold", fill=st)
        s.box(165, y + 8, 170, 50, "W_q^h, W_k^h, W_v^h\nd → d/H", f, st, 11)
        s.arrow(86, 205, 163, y + 33, C["gray_d"], 1.3)
        s.arrow(336, y + 33, 396, y + 33, st, 1.3)
        s.heat(pat, 400, y + 5, 14, 14, 0, 1, show=False, base=st)
        s.text(428, y + 74, "A^h  (N×N)", 10, fill=st)
        s.arrow(458, y + 33, 496, y + 33, st, 1.3)
        s.box(498, y + 13, 90, 40, "Y^h  (N×d/H)", f, st, 11)
        s.arrow(590, y + 33, 628, y + 33, st, 1.3)
    s.text(250, 250, "⋮", 22, weight="bold", fill=C["gray_d"])
    s.text(540, 250, "⋮", 22, weight="bold", fill=C["gray_d"])
    s.arrow(692, 215, 740, 215, C["gray_d"], 1.5)
    s.box(742, 190, 70, 50, "W_O\nd×d", C["purple"], C["purple_d"], 12)
    s.arrow(813, 215, 850, 215, C["gray_d"], 1.5)
    s.box(852, 170, 80, 90, "Y\nN×d", C["green"], C["green_d"], 13)
    s.text(480, 405, "Ogni testa ha le proprie W_q^h, W_k^h, W_v^h e una propria matrice di attention A^h: può specializzarsi in una relazione diversa", 11, fill=C["gray_d"])
    s.text(480, 422, "(qui: stessa posizione, posizione precedente, primo token). Con d_k = d/H il costo totale è lo stesso di una testa sola.", 11, fill=C["gray_d"])
    s.save("fig_multihead.svg")

# =====================================================================================
# FIG 13 - blocco encoder e blocco decoder (slide 67-70)
# =====================================================================================
def fig_transformer_blocks():
    s = Svg(1000, 740)
    s.text(500, 24, "Blocchi del Transformer (Vaswani et al. 2017): encoder a sinistra, decoder a destra", 15, weight="bold")
    W, H = 230, 34
    def stack(xc, items, title):
        s.text(xc, 56, title, 14, weight="bold")
        pos = {}
        for key, y, lab, fill, st in items:
            s.box(xc - W / 2, y, W, H, lab, fill, st, 11)
            pos[key] = y
        return pos
    def up(xc, y_from, y_to, color=C["gray_d"]):
        s.arrow(xc, y_from, xc, y_to, color, 1.4)
    def residual(xc, y_sub_bottom, y_add_center, side=-1):
        xe = xc + side * W / 2
        s.path(f"M{xe},{y_sub_bottom + 9} L{xe + side * 20},{y_sub_bottom + 9} L{xe + side * 20},{y_add_center} L{xe},{y_add_center}", C["gray_d"], 1.2, arrow=True, dash="4,3")
    # ---------------- encoder
    xe = 230
    E = [("in", 680, "input: feature z_{i,j} / embedding", C["red"], C["red_d"]),
         ("pe", 620, "+ codifica posizionale p_t", C["purple"], C["purple_d"]),
         ("att", 545, "Multi-head self-attention", C["orange"], C["orange_d"]),
         ("n1", 485, "+ residuo → Layer norm", C["gray"], C["gray_d"]),
         ("mlp", 410, "MLP (d → 4d → d), per posizione", C["blue"], C["blue_d"]),
         ("n2", 350, "+ residuo → Layer norm", C["gray"], C["gray_d"]),
         ("out", 260, "output c (stessa forma dell'input)", C["green"], C["green_d"])]
    stack(xe, E, "Encoder block  (×N)")
    up(xe, 680, 655); up(xe, 620, 580); up(xe, 545, 520); up(xe, 485, 445); up(xe, 410, 385); up(xe, 350, 295)
    residual(xe, 580 - 0, 485 + 17); residual(xe, 445, 350 + 17)
    s.rect(xe - 150, 330, 300, 280, "none", C["gray_d"], 1.2, 12, dash="6,4")
    s.text(xe + 138, 350, "×N", 13, anchor="end", weight="bold", fill=C["gray_d"])
    s.text(xe, 727, "N = 6, d_{model} = 512 (Vaswani et al.)", 11, fill=C["gray_d"])
    # ---------------- decoder
    xd = 730
    D = [("in", 680, "input: [START] y_1 y_2 … (spostati a dx)", C["red"], C["red_d"]),
         ("pe", 625, "+ codifica posizionale p_t", C["purple"], C["purple_d"]),
         ("att", 555, "Masked multi-head self-attention", C["orange"], C["orange_d"]),
         ("n1", 500, "+ residuo → Layer norm", C["gray"], C["gray_d"]),
         ("x", 425, "Cross-attention (q: decoder; k, v: c)", C["teal"], C["teal_d"]),
         ("n2", 370, "+ residuo → Layer norm", C["gray"], C["gray_d"]),
         ("mlp", 300, "MLP (d → 4d → d), per posizione", C["blue"], C["blue_d"]),
         ("n3", 245, "+ residuo → Layer norm", C["gray"], C["gray_d"]),
         ("fc", 170, "FC (+ softmax sul vocabolario)", C["yellow"], C["yellow_d"]),
         ("out", 105, "probabilità y_t", C["green"], C["green_d"])]
    stack(xd, D, "Decoder block  (×N)")
    up(xd, 680, 660); up(xd, 625, 590); up(xd, 555, 535); up(xd, 500, 460); up(xd, 425, 405); up(xd, 370, 335); up(xd, 300, 280); up(xd, 245, 205); up(xd, 170, 140)
    residual(xd, 590, 500 + 17, +1); residual(xd, 460, 370 + 17, +1); residual(xd, 335, 245 + 17, +1)
    s.rect(xd - 150, 225, 300, 405, "none", C["gray_d"], 1.2, 12, dash="6,4")
    s.text(xd + 138, 612, "×N", 13, anchor="end", weight="bold", fill=C["gray_d"])
    # c encoder -> cross attention
    s.path(f"M{xe + W / 2},277 L520,277 L520,442 L{xd - W / 2 - 1},442", C["teal_d"], 2, arrow=True)
    s.text(528, 360, "c: serve da\nchiavi e valori", 11, anchor="start", fill=C["teal_d"], weight="bold")
    s.text(xd, 727, "il blocco centrale NON è self-attention: attende sull'output dell'encoder", 11, fill=C["teal_d"])
    s.save("fig_transformer_blocks.svg")

# =====================================================================================
# FIG 14 - image captioning con Transformer (slide 63-66, 84)
# =====================================================================================
def fig_caption_transformer():
    s = Svg(1040, 360)
    s.text(490, 24, "Image captioning con Transformer: nessuna ricorrenza", 15, weight="bold")
    for i in range(3):
        for j in range(3):
            s.rect(30 + j * 36, 100 + i * 36, 36, 36, lerp_color((255, 255, 255), (46, 125, 50), 0.15 + 0.1 * ((i * 3 + j) % 4)), "#90a4ae", .8, 0)
            s.text(30 + j * 36 + 18, 100 + i * 36 + 22, f"z_{{{i}{j}}}", 10)
    s.text(84, 90, "z = CNN(I)", 12, weight="bold")
    s.box(30, 230, 108, 36, "CNN\npre-addestrata", C["red"], C["red_d"], 11); s.arrow(84, 230, 84, 210, C["red_d"])
    s.arrow(142, 154, 190, 154, C["gray_d"], 1.5)
    s.box(192, 110, 150, 90, "Transformer\nencoder T_W\n(self-attention\nsu 9 posizioni)", C["blue"], C["blue_d"], 11)
    s.arrow(343, 154, 390, 154, C["gray_d"], 1.5)
    s.box(392, 124, 110, 60, "c = T_W(z)\n(c_{00} … c_{22})", C["green"], C["green_d"], 11)
    s.path("M502,154 L560,154 L560,160 L596,160", C["teal_d"], 2, arrow=True)
    s.text(548, 138, "k, v", 11, fill=C["teal_d"], weight="bold")
    s.box(598, 105, 220, 110, "Transformer decoder T_D\nmasked self-attention\n+ attention su c\n+ MLP", C["orange"], C["orange_d"], 11)
    words_out = ["person", "wearing", "hat", "[END]"]; words_in = ["[START]", "person", "wearing", "hat"]
    for k in range(4):
        x = 604 + k * 54
        s.box(x - 2, 40, 52, 28, words_out[k], C["yellow"], C["yellow_d"], 10)
        s.arrow(x + 24, 104, x + 24, 69, C["orange_d"], 1.2)
        s.box(x - 2, 250, 52, 28, words_in[k], C["red"], C["red_d"], 10)
        s.arrow(x + 24, 249, x + 24, 216, C["red_d"], 1.2)
    s.text(828, 58, "← y_t = T_D(y_{0:t-1}, c)", 11, anchor="start", fill=C["orange_d"], weight="bold")
    s.text(840, 270, "← ingresso del decoder:\nparole precedenti", 11, anchor="start", fill=C["red_d"])
    s.text(490, 318, "Training: tutte le posizioni t in parallelo (teacher forcing + maschera). Inferenza: una parola alla volta (autoregressivo).", 12, fill=C["gray_d"])
    s.text(490, 338, "Rispetto alla versione con RNN (slide 28-37): al posto di h_t e di un c_t ricorrente c'è attention su tutto c a ogni posizione.", 12, fill=C["gray_d"])
    s.save("fig_caption_transformer.svg")

# =====================================================================================
# FIG 15 - self-similarity e non-local means (slide 74-78)
# =====================================================================================
def nlm_demo(seed=3):
    rng = np.random.default_rng(seed); L = 300; t = np.arange(L)
    n = np.arange(30)
    motif = np.exp(-((n - 9) / 3.0) ** 2) - 0.7 * np.exp(-((n - 20) / 4.0) ** 2) + 0.3 * np.sin(n / 3)
    clean = np.zeros(L)
    for st in (20, 130, 230): clean[st:st + 30] += motif
    clean += 0.5 * np.exp(-((t - 80) / 6.0) ** 2) + 0.5 * np.exp(-((t - 185) / 10.0) ** 2)
    sig = 0.15; noisy = clean + sig * rng.standard_normal(L)
    hw = 7
    pad = np.pad(noisy, (hw, hw), mode="reflect")
    patches = np.stack([pad[i:i + 2 * hw + 1] for i in range(L)])
    d2 = ((patches[:, None, :] - patches[None, :, :]) ** 2).mean(-1)
    h2 = 0.04
    W = np.exp(-np.maximum(d2 - 2 * sig ** 2, 0) / h2); W /= W.sum(1, keepdims=True)
    den = W @ noisy
    return t, clean, noisy, den, W

def fig_nlm():
    t, clean, noisy, den, W = nlm_demo()
    p = 29
    mse_n = float(np.mean((noisy - clean) ** 2)); mse_d = float(np.mean((den - clean) ** 2))
    NUMS["nlm"] = dict(mse_noisy=round(mse_n, 4), mse_denoised=round(mse_d, 4),
                       w_motif=[round(float(W[p, p + 110]), 3), round(float(W[p, p + 210]), 3)], w_self=round(float(W[p, p]), 3),
                       mass_on_motifs=round(float(sum(W[p, s_:s_ + 30].sum() for s_ in (20, 130, 230))), 3))
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    ax[0].plot(t, noisy, color="#999", lw=.8, label="segnale rumoroso"); ax[0].plot(t, clean, color="#2b5fb4", lw=1.4, label="segnale pulito")
    for st in (20, 130, 230): ax[0].axvspan(st, st + 30, color="#ffe0b2", alpha=.6)
    ax[0].axvline(p, color="#b3261e", ls="--"); ax[0].text(p + 2, ax[0].get_ylim()[1] * .8, "pixel p", color="#b3261e", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper right"); ax[0].set_title("(a) lo stesso «motivo» compare 3 volte (aree arancio), come nella musica (slide 76)", fontsize=10)
    ax[1].bar(t, W[p], color="#6a3fa0", width=1.0)
    for st in (20, 130, 230): ax[1].axvspan(st, st + 30, color="#ffe0b2", alpha=.6)
    ax[1].axvline(p, color="#b3261e", ls="--"); ax[1].set_ylabel("w(p, q)")
    ax[1].set_title("(b) pesi non-locali w(p,q) ∝ exp(−‖N(p)−N(q)‖²/h²): picchi dove il vicinato è simile, anche lontano", fontsize=10)
    ax[2].plot(t, noisy, color="#ccc", lw=.8, label="rumoroso"); ax[2].plot(t, den, color="#2e7d32", lw=1.4, label="non-local means  Σ_q w(p,q) I(q)")
    ax[2].plot(t, clean, color="#2b5fb4", lw=.8, ls=":", label="pulito")
    ax[2].legend(fontsize=8, loc="upper right"); ax[2].set_xlabel("posizione")
    ax[2].set_title(f"(c) MSE: rumoroso {mse_n:.4f} → denoised {mse_d:.4f}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_nlm.svg")); plt.close(fig); print("scritto fig_nlm.svg")

# =====================================================================================
# FIG 16 - attention globale vs locale su immagini (slide 80-81, 45)
# =====================================================================================
def fig_local_attention():
    s = Svg(960, 470)
    s.text(480, 24, "Perché sulle immagini serve attention locale (Image Transformer)", 15, weight="bold")
    G, cs = 16, 14
    def grid(x0, y0, mode):
        for i in range(G):
            for j in range(G):
                fill = "#ffffff"
                if mode == "global": fill = C["blue"]
                if mode == "local" and 4 <= i <= 10 and 4 <= j <= 10: fill = C["blue"]
                if (i, j) == (7, 7) or (mode == "local" and i in (7, 8) and j in (7, 8)): fill = C["red"] if mode == "local" else C["red"]
                s.rect(x0 + j * cs, y0 + i * cs, cs, cs, fill, "#b0bec5", .5, 0)
        s.rect(x0, y0, G * cs, G * cs, "none", C["gray_d"], 1.2, 0)
    grid(60, 70, "global"); grid(430, 70, "local")
    s.text(172, 62, "globale: ogni query vede tutti gli n pixel", 12, weight="bold")
    s.text(542, 62, "locale: blocco di query (rosso) → memoria vicina (blu)", 12, weight="bold")
    s.text(172, 316, "n² coppie (query, chiave)", 12, fill=C["blue_d"], weight="bold")
    s.text(542, 316, "≈ n·r coppie  (r = pixel nel vicinato)", 12, fill=C["blue_d"], weight="bold")
    r = 49; rows = []
    for H_ in (32, 256):
        n = H_ * H_; rows.append((H_, n, n * n, n * r, n * n / (n * r)))
    NUMS["local"] = rows
    y = 345
    s.text(480, y, "r = 7×7 = 49 pixel nel vicinato:", 12, weight="bold")
    for k, (H_, n, g, l, ratio) in enumerate(rows):
        s.text(480, y + 22 + 20 * k, f"immagine {H_}×{H_}:  n = {n:,}   globale n² = {g:,}   locale n·r = {l:,}   (≈ {ratio:,.0f}× meno)".replace(",", "."), 12, fill=C["gray_d"])
    s.text(480, y + 82, "È la «self-attention (restricted)» della tabella di slide 45: O(r·n·d), cammino O(n/r).", 12, fill=C["gray_d"], italic=True)
    s.text(700, 140, "rosso = query", 11, anchor="start", fill=C["red_d"])
    s.save("fig_local_attention.svg")

# =====================================================================================
# FIG 17 - relative positional encoding (slide 82-83)
# =====================================================================================
def fig_relpos():
    n = 7; kmax = 2
    s = Svg(940, 400)
    s.text(470, 24, "Codifica posizionale relativa (Shaw et al.): l'embedding dipende solo da i − j", 15, weight="bold")
    pal = {-2: "#f9d0d0", -1: "#ffe0b2", 0: "#fff3c4", 1: "#d6f0d4", 2: "#cfe2ff"}
    def mat(x0, y0, clip):
        for i in range(n):
            for j in range(n):
                d = i - j; v = max(-kmax, min(kmax, d)) if clip else d
                fill = pal[v] if clip else lerp_color((255, 255, 255), (43, 95, 180), abs(d) / 6) if d != 0 else "#fff3c4"
                s.rect(x0 + j * 40, y0 + i * 34, 40, 34, fill, "#90a4ae", .8, 0)
                s.text(x0 + j * 40 + 20, y0 + i * 34 + 22, (f"w_{{{v}}}" if clip else f"{d}"), 11, fill="#1b1b1b" if (clip or abs(d) < 4) else "#fff")
        for j in range(n): s.text(x0 + j * 40 + 20, y0 - 6, f"j={j}", 10)
        for i in range(n): s.text(x0 - 6, y0 + i * 34 + 22, f"i={i}", 10, anchor="end")
    s.text(175, 62, "(a) distanza relativa i − j", 12, weight="bold"); mat(60, 90, False)
    s.text(655, 62, f"(b) clip(i − j, −{kmax}, {kmax}) → {2*kmax+1} embedding appresi", 12, weight="bold"); mat(520, 90, True)
    s.text(470, 360, "e_{ij} = (x_i W^Q)(x_j W^K + a^K_{ij})ᵀ / √d_k   ;   z_i = Σ_j α_{ij} (x_j W^V + a^V_{ij})", 13, fill=C["ink"], weight="bold")
    s.text(470, 382, "a^K_{ij}, a^V_{ij}: vettori appresi, funzione della sola distanza relativa (stessi valori lungo ogni diagonale: matrice di Toeplitz).", 11, fill=C["gray_d"])
    s.save("fig_relpos.svg")

# =====================================================================================
# FIG 18 - Vision Transformer (slide 84-88)
# =====================================================================================
def fig_vit():
    s = Svg(1000, 400)
    s.text(500, 24, "Vision Transformer: «An Image is Worth 16×16 Words» (slide 86-87)", 15, weight="bold")
    G, cs = 4, 34
    n = 1
    for i in range(G):
        for j in range(G):
            s.rect(30 + j * cs, 90 + i * cs, cs, cs, lerp_color((255, 255, 255), (43, 95, 180), 0.15 + 0.07 * ((i * 3 + j * 2) % 7)), "#fff", 1.5, 0)
            s.text(30 + j * cs + cs / 2, 90 + i * cs + 22, str(n), 11); n += 1
    s.rect(30, 90, G * cs, G * cs, "none", C["gray_d"], 1.2, 0)
    s.text(98, 76, "immagine 224×224×3", 12, weight="bold"); s.text(98, 244, "→ 14×14 = 196 patch\n16×16×3 = 768 valori", 11, fill=C["gray_d"])
    s.arrow(170, 158, 214, 158, C["gray_d"], 1.5); s.text(192, 148, "flatten", 10)
    labs = ["[CLS]", "p_1", "p_2", "p_3", "…", "p_{196}"]
    for k, lb in enumerate(labs):
        x = 220 + k * 62
        s.box(x, 130, 54, 30, lb, C["yellow"] if k == 0 else C["red"], C["yellow_d"] if k == 0 else C["red_d"], 11)
        s.arrow(x + 27, 128, x + 27, 108, C["gray_d"], 1.2) if False else None
    s.text(400, 112, "patch appiattiti (vettori di 768)", 11, fill=C["red_d"])
    s.box(220, 180, 366, 32, "proiezione lineare E (768 → d)   [stessa E per ogni patch]", C["blue"], C["blue_d"], 11)
    s.box(220, 228, 366, 32, "+ codifica posizionale (appresa), una per posizione", C["purple"], C["purple_d"], 11)
    for k in range(6):
        x = 220 + k * 62 + 27
        s.arrow(x, 162, x, 179, C["gray_d"], 1.1); s.arrow(x, 213, x, 227, C["gray_d"], 1.1)
    s.arrow(403, 262, 403, 290, C["gray_d"], 1.5)
    s.box(220, 292, 366, 50, "Transformer encoder (×L)\nself-attention globale tra le 197 posizioni", C["orange"], C["orange_d"], 12)
    s.arrow(586, 317, 640, 317, C["gray_d"], 1.5)
    s.box(642, 297, 130, 40, "output [CLS]", C["yellow"], C["yellow_d"], 12)
    s.arrow(773, 317, 810, 317, C["gray_d"], 1.5)
    s.box(812, 297, 160, 40, "MLP head → classe", C["green"], C["green_d"], 12)
    s.text(790, 120, "Per il captioning (slide 86):\nl'uscita dell'encoder\nalimenta un Transformer\ndecoder (cross-attention),\nsenza alcuna CNN.", 12, anchor="start", fill=C["gray_d"])
    s.text(500, 380, "Nessun bias induttivo di località/traslazione come nelle CNN (nota 1): lo impara dai dati ⇒ serve pre-addestramento su molti dati.", 11, fill=C["gray_d"])
    s.save("fig_vit.svg")

# =====================================================================================
# FIG 19 - BERT: masked LM, maschere di attention, "le parole vedono se stesse" (slide 91-93)
# =====================================================================================
def fig_bert_masks():
    s = Svg(1020, 470)
    s.text(510, 24, "BERT: perché serve la maschera sulle parole (masked LM)", 15, weight="bold")
    toks = ["the", "man", "went", "to", "the", "[MASK]", "to", "buy", "a", "[MASK]", "of", "milk"]
    x0 = 24
    for i, w in enumerate(toks):
        m = w == "[MASK]"
        s.box(x0 + i * 82, 100, 76, 28, w, C["red"] if m else C["blue"], C["red_d"] if m else C["blue_d"], 11)
    for i, lab in ((5, "store"), (9, "gallon")):
        s.box(x0 + i * 82, 44, 76, 28, lab, C["yellow"], C["yellow_d"], 11)
        s.arrow(x0 + i * 82 + 38, 98, x0 + i * 82 + 38, 74, C["yellow_d"], 1.5)
    s.text(300, 62, "k = 15% delle parole mascherate →", 12, anchor="end", fill=C["gray_d"])
    s.text(510, 156, "la loss conta solo le posizioni mascherate:  L = − Σ_{i∈M} log p(w_i | parole non mascherate)", 12, fill=C["gray_d"])
    n = 6; cs = 22
    def mask(x, y, kind, title):
        s.text(x + n * cs / 2, y - 14, title, 11, weight="bold")
        for i in range(n):
            for j in range(n):
                ok = (j <= i) if kind == "l2r" else (j >= i) if kind == "r2l" else True
                s.rect(x + j * cs, y + i * cs, cs, cs, C["blue"] if ok else "#f3f3f3", "#b0bec5", .6, 0)
        s.rect(x, y, n * cs, n * cs, "none", C["gray_d"], 1.1, 0)
        s.text(x - 8, y + n * cs / 2, "query i", 10, anchor="end", rot=-90)
        s.text(x + n * cs / 2, y + n * cs + 14, "chiave j", 10)
    mask(70, 240, "l2r", "solo sinistra (GPT)"); mask(260, 240, "r2l", "solo destra"); mask(450, 240, "full", "bidirezionale (BERT)")
    s.text(20, 420, "blu = posizione visibile.  LM unidirezionali: la distribuzione p(w_t | w_{<t}) è ben formata (regola della catena).", 11, anchor="start", fill=C["gray_d"])
    s.text(20, 438, "Con attention bidirezionale a più strati la parola vedrebbe sé stessa (grafo a destra) → si maschera.", 11, anchor="start", fill=C["gray_d"])
    gx = [760, 830, 900]; gy = [260, 320, 380]
    s.text(830, 224, "«le parole vedono sé stesse»", 11, weight="bold", fill=C["red_d"])
    for li in range(3):
        for k in range(3):
            s.circle(gx[k], gy[li], 12, C["blue"], C["blue_d"])
            s.text(gx[k], gy[li] + 4, f"w{k+1}" if li == 0 else "", 10)
    for li in range(2):
        for a in range(3):
            for b in range(3):
                s.line(gx[a], gy[li] - 12, gx[b], gy[li + 1] + 12, "#b0bec5", 1) if False else s.line(gx[a], gy[li + 1] + 12 if False else gy[li] - 12, gx[b], gy[li + 1] + 12, "#cfd8dc", 1)
    s.line(gx[1], gy[0] - 12, gx[0], gy[1] + 12, C["red_d"], 2.5); s.line(gx[0], gy[1] - 12, gx[1], gy[2] + 12, C["red_d"], 2.5)
    s.circle(gx[1], gy[0], 12, C["red"], C["red_d"], 2); s.circle(gx[1], gy[2], 12, C["red"], C["red_d"], 2)
    s.text(gx[1], gy[0] + 4, "w2", 10); s.text(gx[1], gy[2] + 30, "uscita che predice w2", 10, fill=C["red_d"])
    s.text(830, 440, "w2 → posizione 1 → posizione 2: l'informazione\ndi w2 torna a w2 in 2 strati", 10, fill=C["red_d"])
    s.save("fig_bert_masks.svg")

# =====================================================================================
# FIG 20 - pre-training e fine-tuning di BERT (slide 94-95)
# =====================================================================================
def fig_bert_finetune():
    s = Svg(1040, 500)
    s.text(520, 24, "BERT: un'architettura, un pre-training, tante teste di fine-tuning", 15, weight="bold")
    s.text(20, 56, "Pre-training (testo non etichettato)", 13, anchor="start", weight="bold", fill=C["purple_d"])
    s.box(20, 380, 220, 34, "[CLS] frase A mascherata [SEP] frase B", C["red"], C["red_d"], 10)
    s.arrow(130, 380, 130, 330, C["gray_d"], 1.5)
    s.box(20, 270, 220, 60, "BERT\n(Transformer encoder)", C["blue"], C["blue_d"], 13)
    s.arrow(60, 270, 60, 200, C["purple_d"], 1.5); s.arrow(190, 270, 190, 200, C["purple_d"], 1.5)
    s.box(20, 160, 90, 40, "NSP\n(da C = [CLS])", C["purple"], C["purple_d"], 10)
    s.box(140, 160, 100, 40, "Mask LM\n(sui [MASK])", C["purple"], C["purple_d"], 10)
    s.text(130, 130, "2 obiettivi insieme", 11, fill=C["gray_d"])
    s.arrow(250, 300, 300, 300, C["gray_d"], 2, dash="5,4"); s.text(275, 290, "pesi", 10)
    tasks = [("(a) coppia di frasi", "MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG", "[CLS] frase 1 [SEP] frase 2", "etichetta di classe da C", "cls"),
             ("(b) frase singola", "SST-2, CoLA", "[CLS] frase", "etichetta di classe da C", "cls"),
             ("(c) domanda-risposta", "SQuAD v1.1", "[CLS] domanda [SEP] paragrafo", "inizio / fine dello span", "span"),
             ("(d) etichettatura token", "CoNLL-2003 NER", "[CLS] frase", "un'etichetta per token (O, B-PER, …)", "tag")]
    for k, (ti, ds, inp, out, kind) in enumerate(tasks):
        x = 320 + (k % 2) * 355; y = 50 + (k // 2) * 220
        s.rect(x, y, 335, 205, C["gray"], C["gray_d"], 1, 10)
        s.text(x + 167, y + 20, ti, 12, weight="bold"); s.text(x + 167, y + 36, ds, 9, fill=C["gray_d"])
        s.box(x + 20, y + 150, 295, 28, inp, C["red"], C["red_d"], 10)
        s.arrow(x + 167, y + 150, x + 167, y + 125, C["gray_d"], 1.3)
        s.box(x + 20, y + 85, 295, 40, "BERT (fine-tuning di tutti i pesi)", C["blue"], C["blue_d"], 11)
        for q in range(5):
            xx = x + 40 + q * 60
            hi = (kind == "cls" and q == 0) or (kind == "span" and q in (3, 4)) or kind == "tag"
            s.box(xx - 8, y + 50, 32, 22, ["C", "T1", "T2", "T3", "T4"][q], C["yellow"] if hi else "#ffffff", C["yellow_d"] if hi else "#b0bec5", 9)
            if hi: s.arrow(xx + 8, y + 84, xx + 8, y + 73, C["red_d"], 1.6)
        s.text(x + 167, y + 195, "→ " + out, 10, fill=C["red_d"])
    s.text(520, 488, "Si cambia solo lo strato di uscita (e i dati): è il transfer learning della nota 2 (sez. 9) applicato al linguaggio.", 11, fill=C["gray_d"])
    s.save("fig_bert_finetune.svg")

# =====================================================================================
# FIG 21 - ELECTRA: replaced-token detection (slide 96-97)
# =====================================================================================
def fig_electra():
    s = Svg(1000, 380)
    s.text(500, 24, "ELECTRA: il discriminatore giudica OGNI token (original / replaced)", 15, weight="bold")
    orig = ["the", "chef", "cooked", "the", "meal"]
    masked = ["[MASK]", "chef", "[MASK]", "the", "meal"]
    sampled = ["the", "chef", "ate", "the", "meal"]
    lab = ["original", "original", "replaced", "original", "original"]
    ys = [70 + i * 52 for i in range(5)]
    s.text(80, 52, "testo reale", 11, fill=C["gray_d"]); s.text(200, 52, "input al generatore", 11, fill=C["gray_d"])
    for i, y in enumerate(ys):
        s.box(30, y, 100, 30, orig[i], C["gray"], C["gray_d"], 12)
        m = masked[i] == "[MASK]"
        s.box(160, y, 100, 30, masked[i], C["red"] if m else C["blue"], C["red_d"] if m else C["blue_d"], 12)
        rep = sampled[i] != orig[i]
        s.box(540, y, 100, 30, sampled[i], C["red"] if rep else C["blue"], C["red_d"] if rep else C["blue_d"], 12)
        s.arrow(262, y + 15, 320, y + 15, C["gray_d"], 1.2)
        s.arrow(642, y + 15, 700, y + 15, C["gray_d"], 1.2)
        bad = lab[i] == "replaced"
        s.box(870, y, 110, 30, lab[i], C["red"] if bad else C["green"], C["red_d"] if bad else C["green_d"], 11)
        s.arrow(792, y + 15, 868, y + 15, C["gray_d"], 1.2)
        if m: s.arrow(472, y + 15, 538, y + 15, C["yellow_d"], 1.6, dash="5,3")
        else: s.line(472, y + 15, 538, y + 15, "#cfd8dc", 1.2, dash="3,3")
    s.box(320, 120, 150, 150, "Generatore\n(piccolo MLM)\ncampiona i token\nmascherati", C["yellow"], C["yellow_d"], 12)
    s.box(700, 120, 92, 150, "Discri-\nminatore\n(ELECTRA)", C["orange"], C["orange_d"], 12)
    s.text(590, 52, "testo «corrotto»", 11, fill=C["gray_d"]); s.text(925, 52, "etichetta per token", 11, fill=C["gray_d"])
    s.text(500, 346, "«ate» sostituisce «cooked»: plausibile localmente ma non reale. Il campione «the» coincide con l'originale ⇒ resta «original».", 11, fill=C["gray_d"])
    s.text(500, 366, "La loss è su tutti i token (non solo sul 15%) ⇒ più segnale per esempio, stessi risultati con meno calcolo (slide 97).", 11, fill=C["gray_d"])
    s.save("fig_electra.svg")

# =====================================================================================
# FIG 22 - distillazione: schema (slide 99)
# =====================================================================================
def fig_distill_scheme():
    s = Svg(1000, 300)
    s.text(500, 24, "Distillation (model compression): teacher → etichette → student", 15, weight="bold")
    s.box(30, 80, 200, 90, "1. Teacher\nSOTA pre-training\n+ fine-tuning\n(massima accuratezza)", C["blue"], C["blue_d"], 12)
    s.text(130, 64, "50k esempi etichettati", 11, fill=C["gray_d"])
    s.arrow(232, 125, 290, 125, C["gray_d"], 1.8)
    s.box(292, 80, 200, 90, "2. Etichetta molti\nesempi NON etichettati\ncon il teacher\n(es. 8M)", C["yellow"], C["yellow_d"], 12)
    s.arrow(494, 125, 552, 125, C["gray_d"], 1.8)
    s.box(554, 80, 200, 90, "3. Student\nmodello molto più piccolo\n(es. 50× meno parametri)\nimita l'uscita del teacher", C["green"], C["green_d"], 12)
    s.arrow(756, 125, 800, 125, C["gray_d"], 1.8)
    s.box(802, 95, 170, 60, "obiettivo dello student:\nMSE oppure cross-entropy\nrispetto al teacher", C["orange"], C["orange_d"], 11)
    s.text(500, 230, "Riferimenti nella slide: Bucila et al. 2006, Hinton et al. 2015, Belagiannis-Farshad-Galasso 2018.", 12, fill=C["gray_d"])
    s.text(500, 252, "Lo student impara da molti più esempi (etichettati dal teacher) di quelli che avrebbe con le sole etichette vere.", 12, fill=C["gray_d"])
    s.save("fig_distill_scheme.svg")

# =====================================================================================
# FIG 23 - distillazione: temperatura (Hinton) e risultati di Turc et al. (slide 100)
# =====================================================================================
def fig_distill_results():
    z = np.array([8.0, 5.0, 2.0, 0.0, -1.0]); cls = ["A", "B", "C", "D", "E"]
    p1 = softmax(z); p4 = softmax(z / 4)
    NUMS["distill_T"] = dict(p_T1=p1.round(3).tolist(), p_T4=p4.round(3).tolist())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.arange(5)
    ax[0].bar(x - .2, p1, .4, color="#b3261e", label="T = 1")
    ax[0].bar(x + .2, p4, .4, color="#2b5fb4", label="T = 4")
    ax[0].set_xticks(x); ax[0].set_xticklabels(cls); ax[0].set_ylabel("probabilità")
    ax[0].set_title("(a) soft targets del teacher: softmax(z/T), z = (8,5,2,0,−1)", fontsize=10); ax[0].legend(fontsize=9)
    sizes = ["Tiny\n2L 128H", "Mini\n4L 256H", "Small\n4L 512H", "Medium\n8L 512H", "Base\n12L 768H"]
    dist = [86.3, 88.1, 88.0, 88.3, 88.5]; ft = [81.1, 82.7, 83.5, 85.1, 85.7]
    ax[1].axhline(88.0, color="#888", ls="--", label="teacher")
    ax[1].plot(range(5), dist, "s-", color="#2e7d32", label="pre-trained distillation")
    ax[1].plot(range(5), ft, "d-", color="#2b5fb4", label="pre-training + fine-tuning")
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels(sizes, fontsize=8); ax[1].set_ylim(78, 90)
    ax[1].set_ylabel("accuratezza (Amazon Book Reviews)"); ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("(b) Turc et al. 2020 (valori letti dalla slide 100, ≈ ±0.3)", fontsize=10); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_distill_results.svg")); plt.close(fig); print("scritto fig_distill_results.svg")

# =====================================================================================
ALL = [fig_s2s_bottleneck, fig_attention_step, fig_attention_maps, fig_caption_attention, fig_selfattn_example,
       fig_scale_sqrt, fig_path_length, fig_complexity, fig_perm_equiv, fig_posenc, fig_masked, fig_multihead,
       fig_transformer_blocks, fig_caption_transformer, fig_nlm, fig_local_attention, fig_relpos, fig_vit,
       fig_bert_masks, fig_bert_finetune, fig_electra, fig_distill_scheme, fig_distill_results]

if __name__ == "__main__":
    import json
    for f in ALL: f()
    print("\n=== numeri calcolati ===")
    print(json.dumps(NUMS, indent=1, default=str))
