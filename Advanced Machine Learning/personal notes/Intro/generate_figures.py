#!/usr/bin/env python3
"""
Genera le figure SVG usate in `AMLCV_formalizzazione.md`.

Uso:
    python generate_figures.py

Non servono librerie esterne (solo la standard library).
Le immagini vengono salvate nella cartella `figures/` accanto a questo file.
"""
import math
import os
import random
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Palette e font
# ----------------------------------------------------------------------------
INK = "#1e293b"
MUTED = "#64748b"
LIGHT = "#94a3b8"
BORDER = "#e2e8f0"
PANEL = "#f8fafc"
BLUE = "#3b82f6"
INDIGO = "#6366f1"
GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
VIOLET = "#8b5cf6"
PINK = "#ec4899"
TEAL = "#14b8a6"

SANS = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MATH = "'STIX Two Math', 'Cambria Math', 'Times New Roman', serif"

ARROW_COLORS = {
    "ink": INK, "blue": BLUE, "red": RED, "green": GREEN,
    "amber": AMBER, "muted": MUTED, "violet": VIOLET,
}


# ----------------------------------------------------------------------------
# Primitive di disegno
# ----------------------------------------------------------------------------
def T(x, y, s, size=14, fill=INK, weight=400, anchor="middle",
      family=SANS, italic=False, rotate=None, opacity=None):
    st = ' font-style="italic"' if italic else ""
    rot = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    op = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'font-family="{family}"{st}{rot}{op}>{escape(str(s))}</text>')


def M(x, y, s, size=16, fill=INK, anchor="middle", weight=400):
    """Testo matematico (serif, corsivo)."""
    return T(x, y, s, size=size, fill=fill, anchor=anchor, family=MATH,
             italic=True, weight=weight)


def R(x, y, w, h, fill="#fff", stroke=None, rx=8, sw=1.5, extra=""):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}"{st} {extra}/>')


def L(x1, y1, x2, y2, stroke=INK, sw=2, dash=None, marker=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#a_{marker})"' if marker else ""
    op = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="round"{d}{mk}{op}/>')


def C(cx, cy, r, fill="#fff", stroke=None, sw=2, extra=""):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{st} {extra}/>'


def PATH(d, fill="none", stroke=INK, sw=2, dash=None, marker=None, opacity=None):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#a_{marker})"' if marker else ""
    op = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dd}{mk}{op}/>')


def card(x, y, w, h, fill="#ffffff", stroke=BORDER, rx=14, sw=1.5):
    return R(x, y, w, h, fill=fill, stroke=stroke, rx=rx, sw=sw,
             extra='filter="url(#sh)"')


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def defs():
    mk = "".join(
        f'<marker id="a_{n}" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{c}"/></marker>'
        for n, c in ARROW_COLORS.items())
    return (
        "<defs>"
        '<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f1f5f9"/>'
        "</linearGradient>"
        '<filter id="sh" x="-10%" y="-10%" width="120%" height="135%">'
        '<feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" '
        'flood-opacity="0.10"/></filter>'
        + mk + "</defs>")


def svg(w, h, body, title=None, subtitle=None):
    head = ""
    if title:
        head += T(36, 40, title, size=21, weight=700, anchor="start")
    if subtitle:
        head += T(36, 63, subtitle, size=13.5, fill=MUTED, anchor="start")
    accent = R(36, 16, 38, 4, fill=INDIGO, rx=2) if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img">'
        + defs()
        + f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="20" fill="url(#bg)" '
          f'stroke="{BORDER}" stroke-width="2"/>'
        + accent + head + "".join(body) + "</svg>")


def save(name, content):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  ✓", name)


def grid(x, y, rows, cols, cell, values, fill_fn, fmt=str, size=15,
         stroke="#ffffff", txt_fn=None, rx=6):
    out = []
    for i in range(rows):
        for j in range(cols):
            v = values[i][j]
            out.append(R(x + j * cell, y + i * cell, cell, cell,
                         fill=fill_fn(i, j, v), stroke=stroke, rx=rx, sw=2))
            col = txt_fn(i, j, v) if txt_fn else INK
            out.append(T(x + j * cell + cell / 2, y + i * cell + cell / 2 + size * 0.35,
                         fmt(v), size=size, weight=600, fill=col))
    return out


# ----------------------------------------------------------------------------
# FIG 01 - Immagine come tensore
# ----------------------------------------------------------------------------
def fig01():
    b = []
    cs, n = 16, 8
    ox, oy = 90, 180
    chans = [("B", (40, 90, 230), 44), ("G", (30, 160, 90), 22), ("R", (220, 50, 60), 0)]
    for name, (r, g, bl), off in chans:
        for i in range(n):
            for j in range(n):
                v = (math.sin(i * .9 + off * .1) + math.cos(j * .7) + 2) / 4
                col = f"rgb({int(255-(255-r)*v)},{int(255-(255-g)*v)},{int(255-(255-bl)*v)})"
                b.append(R(ox + off + j * cs, oy - off + i * cs, cs, cs, col,
                           stroke="#fff", rx=2, sw=1))
        b.append(T(ox + n * cs + off + 10, oy - off + 12, name, size=13, weight=700,
                   fill=f"rgb({r},{g},{bl})", anchor="start"))
    b.append(T(ox + 64, oy + n * cs + 28, "W = 32", size=14, fill=MUTED))
    b.append(T(ox - 12, oy + 68, "H = 32", size=14, fill=MUTED, anchor="end"))
    b.append(T(ox + 80, oy - 62, "3 canali", size=14, fill=MUTED, weight=600))

    b.append(L(300, 240, 430, 240, stroke=INDIGO, sw=3, marker="blue"))
    b.append(T(365, 226, "srotola", size=13, fill=INDIGO, weight=600))
    b.append(T(365, 262, "(flatten)", size=12, fill=MUTED))

    vals = [231, 98, 17, 204, 66, 143]
    vx, vy = 480, 112
    for i, v in enumerate(vals):
        b.append(R(vx, vy + i * 26, 60, 24, fill="#eef2ff", stroke="#c7d2fe", rx=5, sw=1.5))
        b.append(T(vx + 30, vy + i * 26 + 17, v, size=13, weight=600, fill=INDIGO))
    b.append(T(vx + 30, vy + 6 * 26 + 18, "⋮", size=22, fill=MUTED))
    for k, v in enumerate([88, 175]):
        yy = vy + 6 * 26 + 30 + k * 26
        b.append(R(vx, yy, 60, 24, fill="#eef2ff", stroke="#c7d2fe", rx=5, sw=1.5))
        b.append(T(vx + 30, yy + 17, v, size=13, weight=600, fill=INDIGO))
    yb = vy + 6 * 26 + 30 + 2 * 26
    b.append(PATH(f"M552 {vy} q10 0 10 10 v{yb-vy-20} q0 10 -10 10", stroke=MUTED, sw=2))
    b.append(T(572, (vy + yb) / 2 + 5, "3072", size=15, weight=700, fill=INK, anchor="start"))
    b.append(M(vx + 30, yb + 34, "x ∈ ℝ³⁰⁷²", size=18))

    b.append(card(650, 118, 220, 190))
    b.append(T(670, 148, "Digitalizzazione", size=14, weight=700, anchor="start", fill=INDIGO))
    b.append(T(670, 176, "campionamento +", size=13, anchor="start", fill=MUTED))
    b.append(T(670, 194, "quantizzazione (8 bit)", size=13, anchor="start", fill=MUTED))
    b.append(M(670, 232, "I ∈ {0,…,255}", size=16, anchor="start"))
    b.append(M(670, 256, "^{H×W×3}", size=14, anchor="start", fill=MUTED))
    b.append(T(670, 288, "32·32·3 = 3072 numeri", size=13, weight=600, anchor="start", fill=GREEN))
    save("fig01_image_tensor.svg",
         svg(900, 420, b, "Un'immagine è un tensore H×W×3",
             "Tre matrici di intensità (R, G, B) che, srotolate, diventano un vettore"))


# ----------------------------------------------------------------------------
# FIG 02 - Classificatore lineare
# ----------------------------------------------------------------------------
def fig02():
    b = []
    y0 = 120
    # W
    b.append(R(50, y0, 300, 100, fill="#eff6ff", stroke=BLUE, rx=8, sw=2))
    for k in range(1, 10):
        b.append(L(50, y0 + k * 10, 350, y0 + k * 10, stroke="#bfdbfe", sw=1))
    b.append(R(50, y0 + 40, 300, 10, fill=AMBER, rx=0, extra='opacity=".85"'))
    b.append(M(200, y0 + 200 - 100 + 16, "", size=1))
    b.append(M(200, y0 + 128, "W", size=24, fill=BLUE))
    b.append(T(200, y0 + 150, "10 × 3072", size=13, fill=MUTED, weight=600))
    # x
    b.append(R(420, y0, 36, 300, fill="#f5f3ff", stroke=VIOLET, rx=8, sw=2))
    b.append(M(438, y0 + 150, "x", size=24, fill=VIOLET))
    b.append(T(438, y0 + 328, "3072 × 1", size=13, fill=MUTED, weight=600))
    # b
    b.append(R(520, y0, 36, 100, fill="#ecfdf5", stroke=GREEN, rx=8, sw=2))
    b.append(R(520, y0 + 40, 36, 10, fill=AMBER, rx=0, extra='opacity=".85"'))
    b.append(M(538, y0 + 74, "b", size=22, fill=GREEN))
    b.append(T(538, y0 + 128, "10 × 1", size=13, fill=MUTED, weight=600))
    # s
    b.append(R(640, y0, 36, 100, fill="#fff7ed", stroke=AMBER, rx=8, sw=2))
    b.append(R(640, y0 + 40, 36, 10, fill=AMBER, rx=0, extra='opacity=".85"'))
    b.append(M(658, y0 + 74, "s", size=22, fill=AMBER))
    b.append(T(658, y0 + 128, "10 × 1", size=13, fill=MUTED, weight=600))
    # operatori
    b.append(T(390, y0 + 60, "·", size=34, fill=MUTED))
    b.append(T(488, y0 + 60, "+", size=30, fill=MUTED))
    b.append(T(598, y0 + 62, "=", size=30, fill=MUTED))
    # annotazione riga k
    b.append(PATH("M350 165 C 380 160, 395 120, 420 120", stroke=AMBER, sw=0))
    b.append(card(60, 300, 330, 120, fill="#fffbeb", stroke="#fde68a"))
    b.append(T(78, 328, "Riga k di W = template della classe k", size=14, weight=700,
               anchor="start", fill="#b45309"))
    b.append(M(78, 360, "sₖ = wₖᵀ x + bₖ", size=18, anchor="start"))
    b.append(T(78, 388, "prodotto scalare immagine · template", size=13, fill=MUTED, anchor="start"))
    b.append(card(720, 120, 190, 130, fill="#ffffff"))
    b.append(T(815, 150, "Parametri", size=14, weight=700, fill=INDIGO))
    b.append(T(815, 180, "10 · 3072 + 10", size=15, weight=600))
    b.append(T(815, 212, "= 30 730", size=22, weight=800, fill=GREEN))
    save("fig02_linear_classifier.svg",
         svg(960, 490, b, "Classificatore lineare  f(x, W) = Wx + b",
             "Ogni classe ha una riga di pesi: lo score è il prodotto scalare con l'immagine"))


# ----------------------------------------------------------------------------
# FIG 03 - Softmax
# ----------------------------------------------------------------------------
def fig03():
    b = []
    names = ["cat", "car", "frog"]
    cols = [AMBER, BLUE, VIOLET]
    base = 300
    maxh = 160
    panels = [
        ("logit  s", [3.2, 5.1, -1.7], 5.1, lambda v: f"{v:g}"),
        ("exp(s)", [24.5, 164.0, 0.18], 164.0, lambda v: f"{v:g}"),
        ("softmax  p", [0.13, 0.87, 0.001], 1.0, lambda v: "0.00" if v < .005 else f"{v:.2f}"),
        ("target  q", [1.0, 0.0, 0.0], 1.0, lambda v: f"{v:.2f}"),
    ]
    for pi, (title, vals, vmax, fmt) in enumerate(panels):
        px = 40 + pi * 235
        b.append(card(px, 84, 215, 330))
        b.append(T(px + 107, 112, title, size=15, weight=700, fill=INK, family=SANS))
        b.append(L(px + 18, base, px + 197, base, stroke=LIGHT, sw=1.5))
        for k, v in enumerate(vals):
            bx = px + 28 + k * 56
            h = v / vmax * (maxh - 20)
            if v >= 0:
                b.append(R(bx, base - h, 40, max(h, 1.5), fill=cols[k], rx=5))
                b.append(T(bx + 20, base - h - 8, fmt(v), size=12.5, weight=700, fill=INK))
            else:
                b.append(R(bx, base, 40, -h, fill=cols[k], rx=5))
                b.append(T(bx + 20, base - h + 16, fmt(v), size=12.5, weight=700, fill=INK))
            b.append(T(bx + 20, 392, names[k], size=13, fill=MUTED, weight=600))
    for pi, lab in enumerate(["exp", "÷ 188.7", "confronta"]):
        ax = 40 + pi * 235 + 215
        b.append(L(ax + 2, 250, ax + 18, 250, stroke=INDIGO, sw=2.5, marker="blue"))
        b.append(T(ax + 10, 236, lab, size=10.5, fill=INDIGO, weight=700))
    b.append(R(250, 424, 480, 40, fill="#fef2f2", stroke="#fecaca", rx=12, sw=1.5))
    b.append(M(490, 450, "Lᵢ = −ln(0.13) ≈ 2.04      (immagine = gatto, ma p(cat) è bassa)", size=16,
               fill="#b91c1c"))
    save("fig03_softmax_pipeline.svg",
         svg(980, 480, b, "Dai logit alle probabilità: softmax",
             "Esempio delle slide: logit (3.2, 5.1, −1.7) → probabilità (0.13, 0.87, 0.00)"))


# ----------------------------------------------------------------------------
# FIG 04 - Curva -log p
# ----------------------------------------------------------------------------
def fig04():
    b = []
    x0, x1, y0, ytop = 100, 700, 420, 120
    sx = lambda p: x0 + p * (x1 - x0)
    sy = lambda v: y0 - v * 75
    for v in range(0, 5):
        b.append(L(x0, sy(v), x1, sy(v), stroke=BORDER, sw=1))
        b.append(T(x0 - 12, sy(v) + 4, v, size=12, fill=MUTED, anchor="end"))
    for p in [0, .25, .5, .75, 1]:
        b.append(T(sx(p), y0 + 22, f"{p:g}", size=12, fill=MUTED))
    pts = []
    p = 0.0183
    while p <= 1.0001:
        pts.append((sx(p), sy(-math.log(p))))
        p += 0.004
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    b.append(PATH(d + f" L{x1} {y0} L{pts[0][0]:.1f} {y0} Z", fill="#6366f1", stroke="none",
                  sw=0, opacity=0.08))
    b.append(PATH(d, stroke=INDIGO, sw=4))
    b.append(L(x0, y0, x1 + 10, y0, stroke=INK, sw=2, marker="ink"))
    b.append(L(x0, y0, x0, ytop - 10, stroke=INK, sw=2, marker="ink"))
    b.append(T((x0 + x1) / 2, y0 + 50, "probabilità della classe corretta  p", size=14, fill=MUTED))
    b.append(T(38, 270, "loss  −log p", size=14, fill=MUTED, rotate=-90))
    # punto 0.13
    px, py = sx(0.13), sy(-math.log(0.13))
    b.append(L(px, py, px, y0, stroke=RED, sw=1.5, dash="5 4"))
    b.append(L(px, py, x0, py, stroke=RED, sw=1.5, dash="5 4"))
    b.append(C(px, py, 8, fill=RED, stroke="#fff", sw=3))
    b.append(card(px + 28, py - 28, 190, 56, fill="#fef2f2", stroke="#fecaca"))
    b.append(T(px + 123, py - 6, "p = 0.13  →  L ≈ 2.04", size=14, weight=700, fill="#b91c1c"))
    b.append(T(px + 123, py + 14, "esempio gatto delle slide", size=11.5, fill=MUTED))
    # punto p=1
    b.append(C(sx(1), sy(0), 7, fill=GREEN, stroke="#fff", sw=3))
    b.append(T(sx(1) - 12, sy(0) - 16, "p = 1 → L = 0", size=13, weight=700, fill="#047857", anchor="end"))
    b.append(T(sx(0.05) + 14, sy(3.7) + 4, "p → 0  ⇒  L → ∞", size=13, weight=700, fill=INDIGO, anchor="start"))
    save("fig04_cross_entropy_curve.svg",
         svg(780, 480, b, "Cross-entropy: L = −log p",
             "Penalizza in modo esplosivo le predizioni sicure ma sbagliate"))


# ----------------------------------------------------------------------------
# FIG 05 - Regolarizzazione L2 vs L1
# ----------------------------------------------------------------------------
def fig05():
    b = []
    cards = [
        (40, "w₁ = (1, 0, 0, 0)", [1, 0, 0, 0], "1", "1", False),
        (460, "w₂ = (¼, ¼, ¼, ¼)", [.25] * 4, "0.25", "1", True),
    ]
    b.append(T(450, 100, "x = (1, 1, 1, 1)    ⇒    w₁ᵀx = w₂ᵀx = 1  (stessa data loss)",
               size=15, weight=600, fill=MUTED))
    for x, title, w, r2, r1, best in cards:
        stroke = GREEN if best else BORDER
        b.append(card(x, 125, 400, 250, stroke=stroke, sw=3 if best else 1.5))
        b.append(T(x + 24, 160, title, size=16, weight=700, anchor="start"))
        for i, v in enumerate(w):
            col = lerp_color("#eff6ff", BLUE, abs(v))
            b.append(R(x + 24 + i * 84, 180, 72, 72, fill=col, stroke="#fff", rx=10, sw=2))
            b.append(T(x + 24 + i * 84 + 36, 224, f"{v:g}", size=20, weight=700,
                       fill="#fff" if abs(v) > .6 else INK))
        b.append(T(x + 24, 292, "R_L2 = Σ w²", size=14, anchor="start", fill=MUTED, weight=600))
        b.append(T(x + 376, 292, r2, size=20, weight=800, anchor="end",
                   fill=GREEN if best else RED))
        b.append(T(x + 24, 330, "R_L1 = Σ |w|", size=14, anchor="start", fill=MUTED, weight=600))
        b.append(T(x + 376, 330, r1, size=20, weight=800, anchor="end", fill=INK))
        if best:
            b.append(R(x + 232, 138, 150, 26, fill=GREEN, rx=13))
            b.append(T(x + 307, 156, "preferito da L2", size=12.5, weight=700, fill="#fff"))
    save("fig05_regularization.svg",
         svg(900, 400, b, "Perché L2 preferisce pesi distribuiti",
             "Stessa predizione, penalità diversa: L2 sceglie il vettore che usa tutte le feature"))


# ----------------------------------------------------------------------------
# FIG 06 - GD vs SGD
# ----------------------------------------------------------------------------
def fig06():
    b = []
    cx, cy, sc = 430, 300, 62
    a, bq, lr = 1.0, 5.0, 0.15
    levels = [8, 5, 3, 1.6, 0.6, 0.15]
    cols = ["#e0e7ff", "#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5"]
    for c, col in zip(levels, cols):
        b.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{math.sqrt(2*c/a)*sc:.1f}" '
                 f'ry="{math.sqrt(2*c/bq)*sc:.1f}" fill="{col}" opacity=".75" '
                 f'stroke="#fff" stroke-width="2"/>')
    rng = random.Random(11)

    def run(noise):
        x, y = -3.0, 1.4
        pts = [(x, y)]
        for _ in range(26):
            gx, gy = a * x, bq * y
            if noise:
                gx += rng.gauss(0, 1.1)
                gy += rng.gauss(0, 2.2)
            x -= lr * gx
            y -= lr * gy
            pts.append((x, y))
        return pts

    for pts, col, sw, dash in [(run(False), GREEN, 3.5, None), (run(True), RED, 2.5, None)]:
        P = [(cx + x * sc, min(max(cy + y * sc, 120), 480)) for x, y in pts]
        b.append(PATH("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in P), stroke=col, sw=sw,
                      opacity=0.95))
        for x, y in P:
            b.append(C(x, y, 3.5, fill=col, stroke="#fff", sw=1.2))
    b.append(C(cx + -3 * sc, cy + 1.4 * sc, 8, fill=AMBER, stroke="#fff", sw=3))
    b.append(T(cx - 3 * sc - 14, cy + 1.4 * sc - 14, "start", size=12.5, weight=700,
               fill="#b45309", anchor="end"))
    b.append(C(cx, cy, 9, fill=INK, stroke="#fff", sw=3))
    b.append(T(cx, cy + 32, "minimo", size=12.5, weight=700, fill=INK))
    b.append(card(640, 90, 200, 90))
    b.append(L(656, 120, 690, 120, stroke=GREEN, sw=4))
    b.append(T(700, 125, "gradient descent", size=13, anchor="start", weight=600))
    b.append(L(656, 152, 690, 152, stroke=RED, sw=3))
    b.append(T(700, 157, "SGD (minibatch)", size=13, anchor="start", weight=600))
    b.append(M(430, 515, "W ← W − η ∇̂L", size=18))
    save("fig06_sgd_vs_gd.svg",
         svg(860, 540, b, "Stochastic Gradient Descent",
             "Il gradiente di minibatch è non distorto ma rumoroso: percorso a zig-zag verso il minimo"))


# ----------------------------------------------------------------------------
# FIG 07 - Grafo computazionale (x+y)z
# ----------------------------------------------------------------------------
def edge_labels(p1, p2, fwd, bwd, b):
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy)
    nx, ny = -dy / ln, dx / ln
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    b.append(T(mx - nx * 17, my - ny * 17 + 5, fwd, size=15, weight=800, fill="#047857"))
    b.append(T(mx + nx * 20, my + ny * 20 + 5, bwd, size=15, weight=800, fill="#b91c1c"))


def fig07():
    b = []
    nodes = {"x": (110, 110, "−2"), "y": (110, 230, "5"), "z": (110, 360, "−4")}
    plus, mul, f = (330, 170), (560, 265), (790, 265)
    edges = [
        ((144, 112), (304, 160), "−2", "−4"),
        ((144, 228), (304, 182), "5", "−4"),
        ((358, 176), (533, 252), "3", "−4"),
        ((144, 358), (533, 280), "−4", "3"),
        ((590, 265), (754, 265), "−12", "1"),
    ]
    for p1, p2, fw, bw in edges:
        b.append(L(*p1, *p2, stroke=MUTED, sw=2.5, marker="muted"))
        edge_labels(p1, p2, fw, bw, b)
    for nm, (x, y, v) in nodes.items():
        b.append(C(x, y, 34, fill="#eef2ff", stroke=INDIGO, sw=3, extra='filter="url(#sh)"'))
        b.append(M(x, y - 2, nm, size=20, fill=INDIGO))
        b.append(T(x, y + 18, v, size=12, fill=MUTED, weight=600))
    b.append(C(*plus, 30, fill="#ecfdf5", stroke=GREEN, sw=3, extra='filter="url(#sh)"'))
    b.append(T(plus[0], plus[1] + 10, "+", size=30, weight=700, fill=GREEN))
    b.append(M(plus[0], plus[1] - 44, "q = x + y", size=15, fill=MUTED))
    b.append(C(*mul, 30, fill="#fff7ed", stroke=AMBER, sw=3, extra='filter="url(#sh)"'))
    b.append(T(mul[0], mul[1] + 10, "×", size=30, weight=700, fill=AMBER))
    b.append(C(*f, 36, fill="#fef2f2", stroke=RED, sw=3, extra='filter="url(#sh)"'))
    b.append(M(f[0], f[1] - 2, "f", size=22, fill=RED))
    b.append(T(f[0], f[1] + 18, "−12", size=12, fill=MUTED, weight=600))
    b.append(L(560, 50, 590, 50, stroke="#047857", sw=3))
    b.append(T(598, 55, "forward", size=13, anchor="start", fill="#047857", weight=700))
    b.append(L(670, 50, 700, 50, stroke="#b91c1c", sw=3))
    b.append(T(708, 55, "gradiente  ∂f/∂·  (backward)", size=13, anchor="start",
               fill="#b91c1c", weight=700))
    b.append(card(300, 395, 560, 70, fill="#ffffff"))
    b.append(M(580, 422, "∂f/∂y = (∂f/∂q)(∂q/∂y) = (−4)(1) = −4", size=17))
    b.append(T(580, 448, "gradiente upstream × gradiente locale", size=12.5, fill=MUTED))
    save("fig07_backprop_graph.svg",
         svg(900, 490, b, "Backpropagation su f(x, y, z) = (x + y) z",
             "x = −2, y = 5, z = −4: forward in verde, gradienti in rosso"))


# ----------------------------------------------------------------------------
# FIG 08 - Pattern dei gate
# ----------------------------------------------------------------------------
def gate_card(x, y, title, sub, sym, color, left, right):
    b = [card(x, y, 420, 200)]
    b.append(T(x + 20, y + 30, title, size=15, weight=800, anchor="start", fill=color))
    b.append(T(x + 20, y + 50, sub, size=12.5, anchor="start", fill=MUTED))
    cx, cy = x + 210, y + 125
    b.append(C(cx, cy, 25, fill="#fff", stroke=color, sw=3))
    b.append(T(cx, cy + (9 if len(sym) == 1 else 5), sym, size=24 if len(sym) == 1 else 13, weight=800, fill=color))

    def ys(n):
        return [cy] if n == 1 else [cy - 32, cy + 32] if n == 2 else [cy]

    for (fw, bw), yy in zip(left, ys(len(left))):
        ey = cy + (yy - cy) * 0.35
        b.append(L(x + 28, yy, cx - 23, ey, stroke=MUTED, sw=2.2, marker="muted"))
        b.append(T(x + 66, yy - 16, fw, size=14, weight=800, fill="#047857"))
        b.append(T(x + 66, yy + 30, bw, size=14, weight=800, fill="#b91c1c"))
    for (fw, bw), yy in zip(right, ys(len(right))):
        ey = cy + (yy - cy) * 0.35
        b.append(L(cx + 23, ey, x + 392, yy, stroke=MUTED, sw=2.2, marker="muted"))
        b.append(T(x + 345, yy - 14, fw, size=14, weight=800, fill="#047857"))
        b.append(T(x + 345, yy + 26, bw, size=14, weight=800, fill="#b91c1c"))
    return b


def fig08():
    b = []
    b += gate_card(30, 90, "add gate", "distributore di gradiente", "+", GREEN,
                   [("3", "2"), ("4", "2")], [("7", "2")])
    b += gate_card(450, 90, "mul gate", "scambia i moltiplicatori", "×", AMBER,
                   [("2", "5·3 = 15"), ("3", "5·2 = 10")], [("6", "5")])
    b += gate_card(30, 310, "copy gate", "somma i gradienti", "copy", BLUE,
                   [("3", "4+2 = 6")], [("3", "4"), ("3", "2")])
    b += gate_card(450, 310, "max gate", "router del gradiente", "max", VIOLET,
                   [("4", "0"), ("5", "9")], [("5", "9")])
    b.append(L(40, 76, 68, 76, stroke="#047857", sw=3))
    b.append(T(76, 81, "forward", size=13, anchor="start", fill="#047857", weight=700))
    b.append(L(150, 76, 178, 76, stroke="#b91c1c", sw=3))
    b.append(T(186, 81, "backward", size=13, anchor="start", fill="#b91c1c", weight=700))
    save("fig08_gradient_gates.svg",
         svg(900, 530, b, "Pattern nel flusso del gradiente",
             "Quattro gate elementari e come instradano il gradiente upstream"))


# ----------------------------------------------------------------------------
# FIG 09 - Funzioni di attivazione
# ----------------------------------------------------------------------------
def fig09():
    b = []
    sig = lambda x: 1 / (1 + math.exp(-x))
    elu = lambda x: x if x >= 0 else math.exp(x) - 1
    m1, m2 = (lambda x: -0.6 * x + 0.3), (lambda x: x - 0.2)
    specs = [
        ("Sigmoid", "σ(x) = 1 / (1 + e⁻ˣ)", sig, (-6, 6), (-0.15, 1.15), BLUE, "satura → vanishing gradient"),
        ("tanh", "tanh(x) = 2σ(2x) − 1", math.tanh, (-3, 3), (-1.3, 1.3), VIOLET, "centrata, ma satura"),
        ("ReLU", "max(0, x)", lambda x: max(0, x), (-3, 3), (-0.6, 3), GREEN, "buon default"),
        ("Leaky ReLU", "max(0.1x, x)", lambda x: max(0.1 * x, x), (-3, 3), (-0.6, 3), TEAL, "niente neuroni morti"),
        ("ELU", "x  (x≥0),  α(eˣ−1)  (x<0)", elu, (-3, 3), (-1.3, 3), AMBER, "continua in 0"),
        ("Maxout", "max(w₁ᵀx+b₁, w₂ᵀx+b₂)", lambda x: max(m1(x), m2(x)), (-3, 3), (-0.6, 3), PINK,
         "generalizza ReLU"),
    ]
    for idx, (name, formula, fn, xr, yr, col, note) in enumerate(specs):
        px, py = 30 + (idx % 3) * 315, 88 + (idx // 3) * 265
        b.append(card(px, py, 300, 245))
        b.append(T(px + 18, py + 28, name, size=16, weight=800, anchor="start", fill=col))
        ax, ay, aw, ah = px + 24, py + 48, 252, 128
        mx = lambda x: ax + (x - xr[0]) / (xr[1] - xr[0]) * aw
        my = lambda y: ay + ah - (min(max(y, yr[0]), yr[1]) - yr[0]) / (yr[1] - yr[0]) * ah
        b.append(R(ax, ay, aw, ah, fill=PANEL, stroke=BORDER, rx=6, sw=1))
        b.append(L(ax, my(0), ax + aw, my(0), stroke=LIGHT, sw=1.2))
        b.append(L(mx(0), ay, mx(0), ay + ah, stroke=LIGHT, sw=1.2))
        if name == "Maxout":
            for g in (m1, m2):
                pts = [(mx(xr[0] + i * (xr[1] - xr[0]) / 80), my(g(xr[0] + i * (xr[1] - xr[0]) / 80)))
                       for i in range(81)]
                b.append(PATH("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts),
                              stroke=col, sw=1.6, dash="4 4", opacity=.5))
        pts = [(mx(xr[0] + i * (xr[1] - xr[0]) / 120), my(fn(xr[0] + i * (xr[1] - xr[0]) / 120)))
               for i in range(121)]
        b.append(PATH("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts), stroke=col, sw=3.5))
        b.append(M(px + 150, py + 200, formula, size=14.5))
        b.append(T(px + 150, py + 226, note, size=12, fill=MUTED))
    save("fig09_activation_functions.svg",
         svg(980, 620, b, "Funzioni di attivazione", "Forma, formula e caratteristica principale"))


# ----------------------------------------------------------------------------
# FIG 10 - Parametri FC vs locale vs conv
# ----------------------------------------------------------------------------
def fig10():
    b = []
    rows = [
        ("Fully connected", "10¹²", 12, RED, "ogni unità vede tutti i 10⁶ pixel"),
        ("Localmente connessa", "10⁸", 8, AMBER, "filtro 10×10 per unità"),
        ("Convoluzionale", "10⁴", 4, GREEN, "100 filtri 10×10, pesi condivisi"),
    ]
    for i, (name, val, ex, col, note) in enumerate(rows):
        y = 110 + i * 82
        b.append(T(40, y + 22, name, size=15, weight=700, anchor="start"))
        b.append(T(40, y + 42, note, size=12, fill=MUTED, anchor="start"))
        w = ex * 40
        b.append(R(280, y, 500, 46, fill="#f1f5f9", rx=10))
        b.append(R(280, y, w, 46, fill=col, rx=10))
        b.append(T(280 + w - 14 if w > 140 else 280 + w + 14, y + 31, val + " parametri", size=16,
                   weight=800, fill="#fff" if w > 140 else INK,
                   anchor="end" if w > 140 else "start"))
    b.append(T(530, 355, "scala logaritmica (lunghezza ∝ esponente)", size=12.5, fill=MUTED))
    b.append(T(530, 90, "immagine 1000×1000, 10⁶ unità nascoste", size=13, fill=MUTED, weight=600))
    save("fig10_parameter_count.svg",
         svg(840, 380, b, "Località + condivisione dei pesi",
             "Da 10¹² a 10⁴ parametri: quattro ordini di grandezza in meno a ogni passo"))


# ----------------------------------------------------------------------------
# FIG 11 - Convoluzione 2D
# ----------------------------------------------------------------------------
def fig11():
    b = []
    I = [[8, 5, 2], [7, 5, 3], [9, 4, 1]]
    K = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    F = [[-10, 10, 10], [-14, 18, 14], [-9, 12, 9]]
    cell = 58
    b.append(T(60 + 87, 98, "immagine I", size=14, weight=700, fill=BLUE))
    b += grid(60, 110, 3, 3, cell, I, lambda i, j, v: lerp_color("#eff6ff", BLUE, v / 10), size=18,
              txt_fn=lambda i, j, v: "#fff" if v > 6 else INK)
    b.append(T(285, 205, "⊗", size=34, fill=MUTED))
    b.append(T(340 + 87, 98, "kernel (ribaltato)", size=14, weight=700, fill=VIOLET))
    b += grid(340, 110, 3, 3, cell, K,
              lambda i, j, v: "#ddd6fe" if v > 0 else "#fecdd3" if v < 0 else "#f1f5f9", size=18)
    b.append(T(565, 205, "=", size=34, fill=MUTED))
    b.append(card(610, 110, 270, 174, fill="#fffbeb", stroke="#fde68a"))
    b.append(T(745, 140, "pixel centrale f[1,1]", size=14, weight=700, fill="#b45309"))
    b.append(M(745, 178, "(8−2) + (7−3) + (9−1)", size=17))
    b.append(M(745, 212, "= 6 + 4 + 8", size=17))
    b.append(R(690, 228, 110, 42, fill=AMBER, rx=12))
    b.append(T(745, 258, "18", size=24, weight=800, fill="#fff"))
    # output
    b.append(T(60 + 87, 330, "output f  (stride 1, padding 1)", size=14, weight=700, fill=INK))
    def ofill(i, j, v):
        return lerp_color("#f8fafc", BLUE if v > 0 else RED, abs(v) / 18 * .9)
    b += grid(60, 342, 3, 3, cell, F, ofill, size=18,
              txt_fn=lambda i, j, v: "#fff" if abs(v) > 11 else INK)
    b.append(R(60 + cell, 342 + cell, cell, cell, fill="none", stroke=AMBER, rx=6, sw=4))
    b.append(card(340, 342, 540, 174, fill="#ffffff"))
    b.append(T(360, 374, "Kernel (−1, 0, 1) per riga = rilevatore di bordi verticali", size=14,
               weight=700, anchor="start", fill=VIOLET))
    b.append(T(360, 402, "• valori alti in modulo dove l'intensità cambia da sinistra a destra",
               size=13, anchor="start", fill=MUTED))
    b.append(M(360, 440, "W_out = (3 + 2·1 − 3)/1 + 1 = 3", size=16, anchor="start"))
    b.append(T(360, 474, "• blu = risposta positiva, rosso = negativa", size=13, anchor="start",
               fill=MUTED))
    save("fig11_convolution_2d.svg",
         svg(940, 540, b, "Convoluzione 2D discreta",
             "Ogni pixel diventa una combinazione lineare dei suoi vicini"))


# ----------------------------------------------------------------------------
# FIG 12 - Volumi convoluzionali
# ----------------------------------------------------------------------------
def cuboid(x, y, w, h, dx, dy, fc, tc, sc):
    b = [
        f'<polygon points="{x},{y} {x+dx},{y-dy} {x+w+dx},{y-dy} {x+w},{y}" fill="{tc}" stroke="#fff" stroke-width="1.5"/>',
        f'<polygon points="{x+w},{y} {x+w+dx},{y-dy} {x+w+dx},{y+h-dy} {x+w},{y+h}" fill="{sc}" stroke="#fff" stroke-width="1.5"/>',
        R(x, y, w, h, fill=fc, stroke="#fff", rx=0, sw=1.5),
    ]
    return b


def fig12():
    b = []
    cyc = 225
    specs = [
        (50, 130, 18, "#bfdbfe", "#93c5fd", "#60a5fa", "32×32×3", "entrata"),
        (390, 114, 36, "#bbf7d0", "#86efac", "#4ade80", "28×28×6", "32 − 5 + 1 = 28"),
        (730, 98, 60, "#e9d5ff", "#d8b4fe", "#c084fc", "24×24×10", "28 − 5 + 1 = 24"),
    ]
    for x, s, d, fc, tc, sc, lab, sub in specs:
        b += cuboid(x, cyc - s / 2, s, s, d, d, fc, tc, sc)
        b.append(T(x + s / 2 + d / 2, cyc + s / 2 + 34, lab, size=16, weight=800))
        b.append(T(x + s / 2 + d / 2, cyc + s / 2 + 56, sub, size=12.5, fill=MUTED))
    # filtro sull'input
    b += cuboid(70, cyc - 50, 36, 36, 18, 18, "#fde68a", "#fcd34d", "#f59e0b")
    b.append(T(100, cyc - 92, "filtro 5×5×3", size=12, weight=700, fill="#b45309"))
    # frecce
    for x1, x2, l1, l2 in [(215, 380, "CONV  6 filtri 5×5×3", "ReLU · 456 param."),
                           (560, 720, "CONV  10 filtri 5×5×6", "ReLU · 1 510 param.")]:
        b.append(L(x1, cyc, x2, cyc, stroke=INDIGO, sw=3.5, marker="blue"))
        b.append(T((x1 + x2) / 2, cyc - 22, l1, size=13, weight=700, fill=INDIGO))
        b.append(T((x1 + x2) / 2, cyc + 28, l2, size=12, fill=MUTED))
    b.append(card(210, 345, 580, 62, fill="#ffffff"))
    b.append(M(500, 372, "W_out = (W_in + 2P − F) / S + 1        parametri = C_out (C_in·F·F + 1)", size=16))
    b.append(T(500, 396, "la profondità dei filtri coincide con i canali dell'input (3, poi 6)",
               size=12.5, fill=MUTED))
    save("fig12_conv_volumes.svg",
         svg(960, 430, b, "ConvNet: una sequenza di CONV + ReLU",
             "Ogni filtro produce una mappa di attivazione; K filtri → K canali in uscita"))


# ----------------------------------------------------------------------------
# FIG 13 - Max pooling
# ----------------------------------------------------------------------------
def fig13():
    b = []
    A = [[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]]
    quad = ["#dbeafe", "#dcfce7", "#fef3c7", "#fae8ff"]
    qdark = [BLUE, GREEN, AMBER, VIOLET]
    cell = 58
    ox, oy = 70, 120
    for i in range(4):
        for j in range(4):
            q = (i // 2) * 2 + j // 2
            v = A[i][j]
            qs = [A[r][c] for r in range((i // 2) * 2, (i // 2) * 2 + 2)
                  for c in range((j // 2) * 2, (j // 2) * 2 + 2)]
            ismax = v == max(qs)
            b.append(R(ox + j * cell, oy + i * cell, cell, cell, fill=quad[q],
                       stroke="#fff", rx=4, sw=2))
            if ismax:
                b.append(R(ox + j * cell + 5, oy + i * cell + 5, cell - 10, cell - 10, fill="none",
                           stroke=qdark[q], rx=8, sw=3))
            b.append(T(ox + j * cell + cell / 2, oy + i * cell + cell / 2 + 7, v, size=19,
                       weight=800 if ismax else 500, fill=INK if ismax else MUTED))
    b.append(L(ox + 2 * cell, oy - 8, ox + 2 * cell, oy + 4 * cell + 8, stroke="#fff", sw=5))
    b.append(L(ox - 8, oy + 2 * cell, ox + 4 * cell + 8, oy + 2 * cell, stroke="#fff", sw=5))
    b.append(T(ox + 2 * cell, oy - 18, "input 4×4", size=14, weight=700))
    b.append(L(340, 236, 450, 236, stroke=INDIGO, sw=3.5, marker="blue"))
    b.append(T(395, 220, "max pool", size=13, weight=700, fill=INDIGO))
    b.append(T(395, 262, "2×2, stride 2", size=12, fill=MUTED))
    O = [[6, 8], [3, 4]]
    oc = 78
    for i in range(2):
        for j in range(2):
            q = i * 2 + j
            b.append(R(480 + j * oc, 158 + i * oc, oc, oc, fill=quad[q], stroke=qdark[q], rx=10, sw=3))
            b.append(T(480 + j * oc + oc / 2, 158 + i * oc + oc / 2 + 9, O[i][j], size=26, weight=800))
    b.append(T(480 + oc, 138, "output 2×2", size=14, weight=700))
    b.append(card(670, 130, 230, 160, fill="#ffffff"))
    b.append(T(688, 160, "Proprietà", size=14, weight=800, anchor="start", fill=INDIGO))
    for k, s in enumerate(["nessun parametro", "canali indipendenti", "224×224×64 → 112×112×64",
                           "backprop: solo al massimo"]):
        b.append(T(688, 190 + k * 26, "• " + s, size=12.5, anchor="start", fill=MUTED))
    save("fig13_max_pooling.svg",
         svg(930, 400, b, "Max pooling", "Si prende il massimo su ogni finestra, canale per canale"))


# ----------------------------------------------------------------------------
# FIG 14 - Receptive field
# ----------------------------------------------------------------------------
def fig14():
    b = []
    cx = 400
    rows = [(1, 110, "Strato 3", 7), (3, 215, "Strato 2", 5), (5, 320, "Strato 1", 3), (7, 425, "Input", 1)]
    sp = 72

    def pos(n, i, y):
        return cx + (i - (n - 1) / 2) * sp, y
    # sfondo piramide
    b.append(f'<polygon points="{cx},{rows[0][1]-30} {cx+3.5*sp+26},{rows[3][1]+30} {cx-3.5*sp-26},{rows[3][1]+30}" '
             f'fill="{INDIGO}" opacity="0.07"/>')
    for r in range(3):
        n_up, y_up = rows[r][0], rows[r][1]
        n_dn, y_dn = rows[r + 1][0], rows[r + 1][1]
        for i in range(n_up):
            for k in range(3):
                x1, y1 = pos(n_up, i, y_up)
                x2, y2 = pos(n_dn, i + k, y_dn)
                b.append(L(x1, y1 + 18, x2, y2 - 18, stroke=INDIGO, sw=1.8, opacity=.5))
    for r, (n, y, lab, rf) in enumerate(rows):
        for i in range(n):
            x, yy = pos(n, i, y)
            if r == 0:
                b.append(C(x, yy, 18, fill=AMBER, stroke="#fff", sw=3))
            elif r == 3:
                b.append(C(x, yy, 18, fill=BLUE, stroke="#fff", sw=3))
            else:
                b.append(C(x, yy, 18, fill="#c7d2fe", stroke="#fff", sw=3))
        b.append(T(60, y + 5, lab, size=14, weight=700, anchor="start"))
        b.append(T(690, y + 5, f"r = {rf}" if r else "r = 7 pixel", size=15, weight=800,
                   anchor="start", fill=AMBER if r == 0 else INDIGO))
    b.append(card(560, 140, 290, 70, fill="#fff"))
    b.append(M(705, 170, "r = 1 + L (K − 1)", size=19))
    b.append(T(705, 196, "K = 3, stride 1", size=12.5, fill=MUTED))
    save("fig14_receptive_field.svg",
         svg(880, 480, b, "Receptive field con convoluzioni 3×3 successive",
             "Ogni strato aggiunge K − 1 = 2 pixel: 1 → 3 → 5 → 7"))


# ----------------------------------------------------------------------------
# FIG 15 - Errore ImageNet
# ----------------------------------------------------------------------------
def fig15():
    b = []
    data = [
        ("2010", "shallow", None, 28.2), ("2011", "shallow", None, 25.8),
        ("2012", "AlexNet", 8, 16.4), ("2013", "—", None, 11.7),
        ("2014", "VGG", 19, 7.3), ("2014", "GoogLeNet", 22, 6.7),
        ("2015", "ResNet", 152, 3.57),
    ]
    base, sc = 410, 9.2
    for g in range(0, 31, 10):
        b.append(L(80, base - g * sc, 920, base - g * sc, stroke=BORDER, sw=1.2))
        b.append(T(68, base - g * sc + 4, f"{g}%", size=12, fill=MUTED, anchor="end"))
    for i, (yr, name, layers, err) in enumerate(data):
        x = 100 + i * 118
        h = err * sc
        col = LIGHT if layers is None else lerp_color(BLUE, GREEN, i / 6)
        if i == 6:
            col = GREEN
        b.append(R(x, base - h, 84, h, fill=col, rx=10))
        b.append(T(x + 42, base - h - 10, f"{err:g}", size=16, weight=800))
        b.append(T(x + 42, base + 24, yr, size=13, weight=700))
        b.append(T(x + 42, base + 43, name, size=12, fill=MUTED))
        if layers:
            b.append(R(x + 12, base - 30, 60, 22, fill="#ffffff", rx=11, extra='opacity=".9"'))
            b.append(T(x + 42, base - 14, f"{layers} strati", size=11, weight=700, fill=INK))
    b.append(T(32, 250, "errore top-5 (%)", size=13, fill=MUTED, rotate=-90))
    save("fig15_imagenet_error.svg",
         svg(960, 480, b, "ImageNet: più profonde, meno errore",
             "Errore top-5 dal 2010 al 2015: da 28.2% a 3.57%"))


# ----------------------------------------------------------------------------
# FIG 16 - LeNet-5
# ----------------------------------------------------------------------------
def stack(x, cy, size, n, off, color, edge):
    out = []
    for i in range(n - 1, -1, -1):
        out.append(R(x + i * off, cy - size / 2 - i * off, size, size, fill=color, stroke=edge,
                     rx=3, sw=1.5))
    return out


def fig16():
    b = []
    cy = 215
    b += stack(30, cy, 96, 1, 0, "#dbeafe", BLUE)
    b += stack(180, cy, 84, 6, 7, "#dcfce7", GREEN)
    b += stack(340, cy, 42, 6, 7, "#fef3c7", AMBER)
    b += stack(450, cy, 30, 8, 5, "#dcfce7", GREEN)
    b += stack(600, cy, 16, 8, 5, "#fef3c7", AMBER)
    for x, h, col, edge in [(710, 112, "#e0e7ff", INDIGO), (810, 84, "#e0e7ff", INDIGO),
                            (910, 40, "#fce7f3", PINK)]:
        b.append(R(x, cy - h / 2, 22, h, fill=col, stroke=edge, rx=6, sw=2))
    labels = [
        (78, "INPUT", "32×32"), (240, "C1", "6@28×28"), (375, "S2", "6@14×14"),
        (480, "C3", "16@10×10"), (630, "S4", "16@5×5"), (721, "C5", "120"),
        (821, "F6", "84"), (921, "OUT", "10"),
    ]
    for x, n, d in labels:
        b.append(T(x, 335, n, size=14, weight=800))
        b.append(T(x, 355, d, size=12, fill=MUTED))
    ops = [(128, 180, "conv 5×5"), (285, 340, "pool 2×2"), (415, 450, "conv 5×5"),
           (545, 600, "pool 2×2"), (655, 705, "conv 5×5")]
    for x1, x2, s in ops:
        b.append(L(x1, 130, x2, 130, stroke=INDIGO, sw=2.5, marker="blue"))
        b.append(T((x1 + x2) / 2, 118, s, size=11.5, weight=700, fill=INDIGO))
    b.append(L(745, 130, 800, 130, stroke=MUTED, sw=2.5, marker="muted"))
    b.append(L(845, 130, 900, 130, stroke=MUTED, sw=2.5, marker="muted"))
    b.append(T(773, 118, "FC", size=11.5, weight=700, fill=MUTED))
    b.append(T(873, 118, "FC", size=11.5, weight=700, fill=MUTED))
    b.append(card(150, 385, 760, 56, fill="#fff"))
    b.append(T(530, 410, "convoluzione → non linearità → pooling → feature map",
               size=14, weight=700))
    b.append(T(530, 430, "parametri: C1 = 156 · C5 = 48 120 · F6 = 10 164", size=12.5, fill=MUTED))
    save("fig16_lenet5.svg",
         svg(1000, 460, b, "LeNet-5 (LeCun et al., 1998)",
             "Alternanza di convoluzione e sotto-campionamento, poi strati fully connected"))


# ----------------------------------------------------------------------------
# FIG 17 - Connessione residua
# ----------------------------------------------------------------------------
def fig17():
    b = []
    y = 200
    b.append(L(40, y, 130, y, stroke=INK, sw=2.5, marker="ink"))
    b.append(M(30, y + 5, "x", size=20, anchor="end"))
    for x, lab, col, fill in [(140, "conv", BLUE, "#eff6ff"), (270, "ReLU", GREEN, "#ecfdf5"),
                              (400, "conv", BLUE, "#eff6ff")]:
        b.append(R(x, y - 30, 90, 60, fill=fill, stroke=col, rx=12, sw=2.5, extra='filter="url(#sh)"'))
        b.append(T(x + 45, y + 6, lab, size=15, weight=700, fill=col))
    b.append(L(230, y, 268, y, stroke=INK, sw=2.5, marker="ink"))
    b.append(L(360, y, 398, y, stroke=INK, sw=2.5, marker="ink"))
    b.append(L(490, y, 575, y, stroke=INK, sw=2.5, marker="ink"))
    b.append(M(330, y - 56, "F(x)", size=18, fill=INDIGO))
    b.append(PATH(f"M110 {y} C 110 80, 600 80, 600 {y-30}", stroke=AMBER, sw=3.5, marker="amber"))
    b.append(T(355, 104, "identity: x", size=14, weight=700, fill="#b45309"))
    b.append(C(600, y, 24, fill="#fff7ed", stroke=AMBER, sw=3))
    b.append(T(600, y + 10, "+", size=28, weight=800, fill=AMBER))
    b.append(L(624, y, 690, y, stroke=INK, sw=2.5, marker="ink"))
    b.append(M(704, y + 6, "y = F(x) + x", size=17, anchor="start"))
    b.append(card(120, 285, 560, 80, fill="#fff"))
    b.append(M(400, 318, "∂L/∂x = ∂L/∂y · (∂F/∂x + I)", size=19))
    b.append(T(400, 346, "il termine I è un cammino diretto per il gradiente attraverso centinaia di strati",
               size=12.5, fill=MUTED))
    save("fig17_residual_block.svg",
         svg(900, 390, b, "Connessione residua (ResNet)", "Il gradiente ha sempre una scorciatoia"))


# ----------------------------------------------------------------------------
# FIG 18 - Rete a 2 strati
# ----------------------------------------------------------------------------
def fig18():
    b = []
    layers = [(4, 150, BLUE, "#dbeafe", "input  x ∈ ℝᴰ"),
              (6, 450, GREEN, "#dcfce7", "hidden  max(0, W₁x)"),
              (3, 750, PINK, "#fce7f3", "scores  s = W₂h")]
    cy = 240
    sp = 52
    pos = []
    for n, x, *_ in layers:
        pos.append([(x, cy + (i - (n - 1) / 2) * sp) for i in range(n)])
    for a in range(2):
        for (x1, y1) in pos[a]:
            for (x2, y2) in pos[a + 1]:
                b.append(L(x1, y1, x2, y2, stroke=INDIGO, sw=1.1, opacity=.28))
    for (n, x, col, fill, lab), pts in zip(layers, pos):
        for (px, py) in pts:
            b.append(C(px, py, 16, fill=fill, stroke=col, sw=3))
        b.append(T(x, 392, lab, size=14, weight=700, fill=col))
    b.append(T(300, 110, "W₁  [H × D]", size=14, weight=800, fill=INDIGO))
    b.append(T(600, 110, "W₂  [C × H]", size=14, weight=800, fill=INDIGO))
    b.append(card(210, 410, 480, 46, fill="#fff"))
    b.append(M(450, 440, "f = W₂ max(0, W₁ x)", size=19))
    save("fig18_two_layer_network.svg",
         svg(900, 480, b, "Rete neurale a 2 strati",
             "Ogni neurone è collegato a tutti quelli dello strato precedente (con bias)"))


if __name__ == "__main__":
    print("Generazione figure in:", OUT)
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09,
               fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18):
        fn()
    print("Fatto.")
