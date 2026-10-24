#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera tutte le figure SVG della nota 5 (Detection and Segmentation).
Uso:  python3 genera_figure_detection_segmentation.py
Scrive i file fig_*.svg nella stessa cartella dello script e stampa i numeri calcolati.
Dipendenze: solo numpy.
"""
import os, math, itertools
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREEN, ORANGE, GRAY, PURP = '#2b6cb0', '#c53030', '#2f855a', '#dd6b20', '#718096', '#6b46c1'
LBLUE, LRED, LGREEN, LORANGE, LGRAY, YEL, LPURP = '#bee3f8', '#fed7d7', '#c6f6d5', '#feebc8', '#edf2f7', '#fefcbf', '#e9d8fd'


def esc(t):
    return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Svg:
    def __init__(self, w, h, title=None):
        self.w, self.h, self.e = w, h, []
        if title:
            self.text(w / 2, 26, title, 16, anchor='middle', bold=True)

    def rect(self, x, y, w, h, fill='#fff', stroke='#333', sw=1.2, rx=3, dash=None, op=1):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.e.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                      f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')

    def text(self, x, y, t, size=12, color='#222', anchor='start', bold=False, italic=False):
        b = ' font-weight="bold"' if bold else ''
        i = ' font-style="italic"' if italic else ''
        self.e.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{b}{i}>{esc(t)}</text>')

    def lines(self, x, y, ls, size=12, color='#222', anchor='start', dy=None, bold=False):
        dy = dy or size * 1.45
        for k, l in enumerate(ls):
            self.text(x, y + k * dy, l, size, color, anchor, bold)

    def line(self, x1, y1, x2, y2, color='#333', sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.e.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>')

    def arrow(self, x1, y1, x2, y2, color='#333', sw=1.6, dash=None):
        self.line(x1, y1, x2, y2, color, sw, dash)
        a = math.atan2(y2 - y1, x2 - x1)
        l = 9
        p = [(x2, y2), (x2 - l * math.cos(a - 0.4), y2 - l * math.sin(a - 0.4)), (x2 - l * math.cos(a + 0.4), y2 - l * math.sin(a + 0.4))]
        self.e.append('<polygon points="' + ' '.join(f'{px:.1f},{py:.1f}' for px, py in p) + f'" fill="{color}"/>')

    def circle(self, x, y, r, fill='#fff', stroke='#333', sw=1.2):
        self.e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def poly(self, pts, stroke='#333', sw=2, fill='none'):
        self.e.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts) + f'" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def box(self, x, y, w, h, label, fill=LGRAY, stroke='#333', size=12, color='#222'):
        self.rect(x, y, w, h, fill, stroke)
        ls = label if isinstance(label, list) else [label]
        y0 = y + h / 2 - (len(ls) - 1) * size * 0.7 + size * 0.35
        for k, l in enumerate(ls):
            self.text(x + w / 2, y0 + k * size * 1.4, l, size, color, 'middle')

    def mat(self, x, y, M, cell=32, fills=None, fs=12, fmt='{:g}', stroke='#444', bold=None):
        for i, row in enumerate(M):
            for j, v in enumerate(row):
                f = fills[i][j] if fills is not None else '#fff'
                self.rect(x + j * cell, y + i * cell, cell, cell, f, stroke, 1, 0)
                t = v if isinstance(v, str) else fmt.format(v)
                self.text(x + j * cell + cell / 2, y + i * cell + cell / 2 + fs * 0.35, t, fs, anchor='middle',
                          bold=bool(bold and bold[i][j]))

    def save(self, name):
        s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
             f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>' + ''.join(self.e) + '</svg>')
        with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
            f.write(s)
        print('scritto', name)


def P(*a):
    print(*a)


# ----------------------------------------------------------------------------------------------
# 1. I quattro compiti
# ----------------------------------------------------------------------------------------------
def fig_tasks():
    R, C = 7, 12
    cls = np.zeros((R, C), int); inst = np.zeros((R, C), int)
    cls[3:, :] = 1
    cls[3:6, 1:4] = 2; inst[3:6, 1:4] = 1
    cls[4:7, 4:7] = 2; inst[4:7, 4:7] = 2
    cls[3:6, 8:11] = 3; inst[3:6, 8:11] = 3
    ccol = {0: LBLUE, 1: LGREEN, 2: '#f6ad55', 3: '#b794f4'}
    icol = {1: '#f6ad55', 2: '#c05621', 3: '#b794f4'}
    cn = {1: 'DOG', 2: 'DOG', 3: 'CAT'}
    s = Svg(900, 330, 'I quattro compiti: da "una etichetta" a "una etichetta per istanza e per pixel"')
    cell = 14
    titles = ['Classificazione', 'Segmentazione semantica', 'Object detection', 'Instance segmentation']
    outs = [['output: CAT', '(nessuna estensione spaziale)'], ['output: classe per pixel', 'GRASS, DOG, CAT, SKY'],
            ['output: {(classe, box)}', 'DOG, DOG, CAT (box)'], ['output: {(classe, maschera)}', 'DOG, DOG, CAT (pixel)']]
    for p in range(4):
        x0 = 20 + p * 220; y0 = 70
        s.text(x0, 55, titles[p], 13, bold=True)
        for i in range(R):
            for j in range(C):
                if p == 0 or p == 2:
                    col = '#f7fafc'
                    if cls[i, j] == 0: col = '#ebf8ff'
                    if cls[i, j] == 1: col = '#f0fff4'
                    if cls[i, j] >= 2: col = '#e2e8f0'
                elif p == 1:
                    col = ccol[cls[i, j]]
                else:
                    col = icol[inst[i, j]] if inst[i, j] else ccol[cls[i, j]]
                s.rect(x0 + j * cell, y0 + i * cell, cell, cell, col, '#cbd5e0', 0.5, 0)
        if p == 2:
            for k in (1, 2, 3):
                ii, jj = np.where(inst == k)
                bx, by = x0 + jj.min() * cell, y0 + ii.min() * cell
                bw, bh = (jj.max() - jj.min() + 1) * cell, (ii.max() - ii.min() + 1) * cell
                s.rect(bx, by, bw, bh, 'none', RED, 2, 0)
                s.text(bx, by - 3, cn[k], 9, RED, bold=True)
        if p == 0:
            s.text(x0 + C * cell / 2, y0 + R * cell + 22, 'CAT', 16, RED, 'middle', True)
        s.lines(x0, y0 + R * cell + 50 if p else y0 + R * cell + 50, outs[p], 11, GRAY)
    s.text(20, 215, 'Colori: cielo = azzurro, erba = verde, cani = arancio, gatto = viola. In (4) i due cani hanno colori diversi (istanze).', 11, GRAY)
    s.rect(20, 235, 860, 80, LGRAY, '#cbd5e0')
    s.lines(32, 256, ['THINGS (persona, gatto, cavallo): forma vincolata, istanze con identita propria -> servono oggetti (detection / instance).',
                      'STUFF (strada, erba, cielo): amorfo, nessuna nozione di istanza, spesso basta il livello del pixel ("texture") -> segmentazione semantica.'], 11.5)
    s.save('fig_tasks.svg')


# ----------------------------------------------------------------------------------------------
# 2. Sliding window vs fully convolutional (+ costo)
# ----------------------------------------------------------------------------------------------
def fig_sliding_fcn():
    H = W = 256; p = 65
    n_patch = (H - p + 1) * (W - p + 1)
    ratio = n_patch * p * p / (H * W)
    P('[sliding] patch', p, 'n_patch', n_patch, 'rapporto di pixel elaborati', round(ratio, 1))
    D = 64
    fl = {}
    for name, f in (('H x W', 1), ('H/2 x W/2', 2), ('H/4 x W/4', 4)):
        h = 224 // f
        fl[name] = 2 * 9 * D * D * h * h / 1e9
    P('[sliding] GFLOPs di un conv 3x3 64->64 (224x224):', {k: round(v, 3) for k, v in fl.items()})
    s = Svg(900, 400, 'Sliding window contro rete completamente convoluzionale')
    s.text(20, 55, 'Sliding window: una CNN per ogni patch', 13, bold=True)
    cell = 22
    for i in range(10):
        for j in range(10):
            s.rect(30 + j * cell, 70 + i * cell, cell, cell, '#f7fafc', '#cbd5e0', 0.5, 0)
    s.rect(30 + 1 * cell, 70 + 2 * cell, 5 * cell, 5 * cell, LBLUE, BLUE, 2, 0, op=0.55)
    s.rect(30 + 2 * cell, 70 + 2 * cell, 5 * cell, 5 * cell, LRED, RED, 2, 0, op=0.55)
    s.text(30 + 3.5 * cell, 70 + 4.6 * cell, 'overlap', 12, '#222', 'middle', True)
    s.lines(30, 70 + 10 * cell + 20, ['Due patch 5x5 con stride 1 condividono 20/25 = 80% dei pixel:',
                                      'le stesse feature vengono ricalcolate quasi da capo.',
                                      f'Immagine {H}x{W}, patch {p}x{p}: {n_patch:,} patch,',
                                      f'~{ratio:,.0f} volte i pixel dell\'immagine elaborati.'], 11.5)
    s.text(470, 55, 'Fully convolutional: tutti i pixel in un solo passaggio', 13, bold=True)
    blocks = [('3xHxW', 0, 'input'), ('D1xH/2xW/2', 1, 'conv + down'), ('D2xH/4xW/4', 2, 'conv + down'),
              ('D3xH/4xW/4', 2, 'conv (dilatata/denso)'), ('D2xH/4xW/4', 2, 'conv'), ('D1xH/2xW/2', 1, 'up'), ('CxHxW', 0, 'score')]
    x = 470
    for k, (lab, lev, sub) in enumerate(blocks):
        hh = [90, 66, 46][lev]
        s.rect(x, 130 - hh / 2 + 25, 52, hh, [LBLUE, LGREEN, LORANGE][lev], '#333')
        s.text(x + 26, 130 + 25 + hh / 2 + 14, lab, 8.5, anchor='middle')
        if k < len(blocks) - 1: s.arrow(x + 52, 155, x + 62, 155)
        x += 62
    s.text(470, 250, 'argmax sui C canali -> mappa H x W di etichette', 11.5, GRAY)
    s.rect(470, 270, 410, 110, YEL, '#d69e2e')
    s.lines(480, 290, ['Costo di un conv 3x3, 64->64 canali, su un\'immagine 224x224:',
                       f'  a risoluzione piena      : {fl["H x W"]:.2f} GFLOPs',
                       f'  a meta risoluzione (/2)  : {fl["H/2 x W/2"]:.2f} GFLOPs  (4x meno)',
                       f'  a un quarto (/4)         : {fl["H/4 x W/4"]:.2f} GFLOPs  (16x meno)',
                       'Downsampling dentro la rete => convoluzioni economiche e campo recettivo ampio.'], 11)
    s.save('fig_sliding_fcn.svg')


# ----------------------------------------------------------------------------------------------
# 3. Unpooling
# ----------------------------------------------------------------------------------------------
def fig_unpool():
    x = np.array([[1, 2], [3, 4]])
    near = np.kron(x, np.ones((2, 2), int))
    bed = np.zeros((4, 4), int); bed[::2, ::2] = x
    A = np.array([[1, 2, 6, 3], [3, 5, 2, 1], [1, 2, 2, 1], [7, 3, 4, 8]])
    pooled = np.zeros((2, 2), int); pos = {}
    for bi in range(2):
        for bj in range(2):
            blk = A[2 * bi:2 * bi + 2, 2 * bj:2 * bj + 2]
            k = np.unravel_index(np.argmax(blk), blk.shape)
            pooled[bi, bj] = blk[k]; pos[(bi, bj)] = (2 * bi + k[0], 2 * bj + k[1])
    mu = np.zeros((4, 4), int)
    for (bi, bj), (r, c) in pos.items():
        mu[r, c] = x[bi, bj]
    P('[unpool] nearest\n', near, '\n bed of nails\n', bed, '\n maxpool\n', pooled, '\n posizioni', pos, '\n max-unpool\n', mu)
    s = Svg(980, 380, 'Upsampling senza parametri: nearest neighbor, "bed of nails", max unpooling')
    c = 30
    s.text(20, 52, 'Nearest neighbor', 13, bold=True)
    s.mat(30, 70, x.tolist(), c, [[LBLUE] * 2] * 2); s.arrow(105, 100, 140, 100)
    s.mat(150, 55, near.tolist(), c, [[LGREEN] * 4] * 4)
    s.text(330, 52, '"Bed of nails"', 13, bold=True)
    s.mat(340, 70, x.tolist(), c, [[LBLUE] * 2] * 2); s.arrow(415, 100, 450, 100)
    fills = [[LORANGE if bed[i, j] else '#fff' for j in range(4)] for i in range(4)]
    s.mat(460, 55, bed.tolist(), c, fills)
    s.text(20, 215, 'Max pooling (a sinistra) memorizza dove era il massimo; max unpooling (a destra) rimette i valori li', 13, bold=True)
    fills = [[LORANGE if (i, j) in pos.values() else '#fff' for j in range(4)] for i in range(4)]
    s.mat(30, 235, A.tolist(), c, fills)
    s.arrow(155, 295, 195, 295); s.text(175, 285, 'max pool', 10, anchor='middle')
    s.mat(205, 265, pooled.tolist(), c, [[LORANGE] * 2] * 2)
    s.text(285, 300, '... resto della rete ...', 11, GRAY)
    s.mat(430, 265, x.tolist(), c, [[LBLUE] * 2] * 2)
    s.arrow(505, 295, 545, 295); s.text(525, 285, 'unpool', 10, anchor='middle')
    fills = [[LORANGE if mu[i, j] else '#fff' for j in range(4)] for i in range(4)]
    s.mat(555, 235, mu.tolist(), c, fills)
    s.lines(700, 255, ['Le posizioni dei massimi (arancio) vengono', 'riusate: coppie di strati down/up', 'corrispondenti. Il resto e zero.'], 11.5)
    s.save('fig_unpool.svg')


# ----------------------------------------------------------------------------------------------
# 4. Convoluzione trasposta
# ----------------------------------------------------------------------------------------------
def tconv1d(x, w, stride):
    n, k = len(x), len(w)
    out = np.zeros((n - 1) * stride + k)
    for i, xi in enumerate(x):
        out[i * stride:i * stride + k] += xi * np.array(w)
    return out


def fig_tconv():
    x = [1, 2]; w = [1, 2, 3]
    out = tconv1d(x, w, 2)
    P('[tconv] 1D x=', x, 'w=', w, '-> out', out.tolist())
    cov = {}
    for k in (3, 4):
        cov[k] = tconv1d([1] * 5, [1] * k, 2).astype(int).tolist()
    P('[tconv] conteggi di sovrapposizione n=5 stride 2:', cov)
    s = Svg(960, 420, 'Convoluzione trasposta: copie del filtro pesate dall\'input, sommate dove si sovrappongono')
    s.text(20, 55, 'Esempio 1D (stride 2, k=3)', 13, bold=True)
    s.text(20, 85, 'input (a,b) = (1,2)', 12); s.text(20, 105, 'filtro (x,y,z) = (1,2,3)', 12)
    cw = 38
    for t, (lab, v, off, col) in enumerate([('a*(x,y,z)', 1, 0, LBLUE), ('b*(x,y,z)', 2, 2, LORANGE)]):
        for k in range(3):
            s.rect(30 + (off + k) * cw, 125 + t * 40, cw, 32, col, '#333')
            s.text(30 + (off + k) * cw + cw / 2, 146 + t * 40, f'{v * w[k]}', 12, anchor='middle')
        s.text(30 + 5 * cw + 10, 146 + t * 40, lab, 11, GRAY)
    for j, v in enumerate(out):
        s.rect(30 + j * cw, 215, cw, 32, LGREEN, '#333'); s.text(30 + j * cw + cw / 2, 236, f'{v:g}', 12, anchor='middle', bold=True)
    s.text(30, 270, 'somma: (ax, ay, az+bx, by, bz) = (1, 2, 5, 4, 6)', 12)
    s.text(30, 290, 'lunghezza (n-1)s + k = 5; per ottenere 2n = 4 si taglia 1 elemento', 11, GRAY)
    s.text(500, 55, 'Perche compaiono artefatti a scacchiera', 13, bold=True)
    s.text(500, 78, 'Quante volte ogni posizione di uscita riceve un contributo (n=5, s=2):', 11.5)
    for r, k in enumerate((3, 4)):
        s.text(500, 112 + r * 70, f'k={k}', 12, bold=True)
        for j, v in enumerate(cov[k]):
            s.rect(540 + j * 32, 95 + r * 70, 32, 26, [LGREEN, LORANGE, LRED][min(v, 3) - 1] if True else '#fff', '#333', 1, 0)
            s.text(540 + j * 32 + 16, 113 + r * 70, str(v), 12, anchor='middle')
    s.lines(500, 245, ['k=3, s=2: la copertura alterna 1,2,1,2,... => uscita non uniforme', '(artefatto a scacchiera, "checkerboard").',
                       'k=4, s=2 (k multiplo di s): copertura 2 uniforme all\'interno.'], 11.5)
    s.rect(20, 320, 920, 85, YEL, '#d69e2e')
    s.lines(32, 342, ['Dimensioni:  convoluzione  n_out = floor((n + 2p - k)/s) + 1  ;  trasposta  n_out = (n - 1) s - 2p + k + output_padding.',
                      'Esempio delle slide: ingresso 2x2, k=3, s=2, p=1  =>  (2-1)*2 - 2 + 3 = 3; con output_padding=1 si ottiene 4x4 (= "tagliare" un pixel).',
                      'Nome: "stride" = rapporto tra spostamento in uscita e in ingresso (all\'opposto della convoluzione normale).'], 11.5)
    s.save('fig_tconv.svg')


# ----------------------------------------------------------------------------------------------
# 5. Convoluzione come prodotto di matrici
# ----------------------------------------------------------------------------------------------
def fig_convmatrix():
    n, k, st, pd = 6, 3, 2, 1
    m = (n + 2 * pd - k) // st + 1
    C = [['0'] * n for _ in range(m)]
    num = np.zeros((m, n))
    wv = [1, 2, 3]
    for i in range(m):
        for kk in range(k):
            j = st * i - pd + kk
            if 0 <= j < n:
                C[i][j] = f'w{kk + 1}'; num[i, j] = wv[kk]
    xv = np.arange(1, n + 1)
    y = num @ xv
    xr = num.T @ y
    P('[convmatrix] C=\n', num, '\n y=Cx =', y, '\n C^T y =', xr)
    s = Svg(960, 400, 'Convoluzione come prodotto matrice-vettore (1D, k=3, stride 2, padding 1)')
    s.text(30, 55, f'C  ({m} x {n}):  y = C x   (n={n} -> {m})', 13, bold=True)
    fills = [[LBLUE if C[i][j] != '0' else '#fff' for j in range(n)] for i in range(m)]
    s.mat(30, 70, [[('.' if v == '0' else v) for v in r] for r in C], 42, fills)
    s.text(30, 70 + m * 42 + 22, 'riga i = filtro posto in 2i-1 (il padding "taglia" w1 nella riga 0)', 11, GRAY)
    s.text(500, 55, f'C^T  ({n} x {m}):  x\' = C^T y   (m={m} -> {n})', 13, bold=True)
    CT = [[C[i][j] for i in range(m)] for j in range(n)]
    fills = [[LORANGE if CT[i][j] != '0' else '#fff' for j in range(m)] for i in range(n)]
    s.mat(500, 70, [[('.' if v == '0' else v) for v in r] for r in CT], 42, fills)
    s.text(500, 70 + n * 42 + 22, 'ogni colonna = copia del filtro: e la convoluzione trasposta', 11, GRAY)
    s.rect(30, 230, 440, 150, YEL, '#d69e2e')
    s.lines(42, 252, ['Esempio numerico: w=(1,2,3), x=(1,...,6)', f'  y = C x      = {y.astype(int).tolist()}',
                      f'  C^T y        = {xr.astype(int).tolist()}', 'C^T NON e l\'inversa di C: riporta le dimensioni',
                      'originali (3 -> 6) ma non i valori. E un operatore di', 'upsampling con kernel APPRESO.'], 11.5)
    s.rect(500, 330, 440, 55, LGREEN, '#38a169')
    s.lines(512, 350, ['Nel backward di una convoluzione si moltiplica per C^T:', 'la conv trasposta è quindi il "gradiente della convoluzione".'], 11.5)
    s.save('fig_convmatrix.svg')


# ----------------------------------------------------------------------------------------------
# 6. Architettura encoder-decoder e U-Net
# ----------------------------------------------------------------------------------------------
def fig_arch():
    s = Svg(960, 470, 'Encoder-decoder (FCN) e U-Net: le skip connection restituiscono il dettaglio spaziale')
    ch = [64, 128, 256, 512]
    sp = ['H x W', 'H/2 x W/2', 'H/4 x W/4', 'H/8 x W/8']
    for l in range(4):
        hh = 120 - 20 * l
        xe, ye = 40 + l * 110, 60 + l * 55
        s.box(xe, ye, 70, hh, [f'{ch[l]}', sp[l]], LBLUE, size=10.5)
        xd = 880 - 70 - l * 110
        s.box(xd, ye, 70, hh, [f'{ch[l]}+{ch[l]}', f'->{ch[l]}'], LGREEN, size=10.5)
        s.arrow(xe + 70, ye + hh / 2, xd, ye + hh / 2, ORANGE, 1.6, '6,4')
        s.text((xe + 70 + xd) / 2, ye + hh / 2 - 6, 'concatenazione (skip)', 10, ORANGE, 'middle')
        if l < 3:
            s.arrow(xe + 35, ye + hh, xe + 35 + 110 - 0, 60 + (l + 1) * 55 + (120 - 20 * (l + 1)) / 2 - 20, BLUE)
            s.arrow(880 - 70 - (l + 1) * 110 + 35, 60 + (l + 1) * 55 + (120 - 20 * (l + 1)) / 2 - 20 + 8, 880 - 70 - l * 110 + 35, ye + hh, GREEN)
    s.box(40 + 3 * 110 + 110 - 20, 60 + 3 * 55 + 5, 120, 60, ['1024', 'H/16 x W/16'], LORANGE, size=10.5)
    s.text(40, 52, 'Discesa: conv + pooling/stride (campo recettivo cresce, si perde dettaglio spaziale)', 11.5, BLUE, bold=True)
    s.text(500, 52, 'Risalita: up-conv 2x2 + concat + conv (si ricostruisce la mappa ad alta risoluzione)', 11.5, GREEN, bold=True)
    s.rect(30, 400, 900, 55, YEL, '#d69e2e')
    s.lines(42, 420, ['Uscita: conv 1x1 -> C x H x W (C classi), poi argmax per pixel. Esempio H=W=256, C=21 (PASCAL VOC con sfondo): 21x256x256 = 1 376 256 punteggi.',
                      'La skip connection concatena (non somma) le feature ad alta risoluzione dell\'encoder: i canali del decoder raddoppiano e poi una conv li riporta a ch.'], 11)
    P('[arch] punteggi in uscita', 21 * 256 * 256)
    s.save('fig_arch.svg')


# ----------------------------------------------------------------------------------------------
# 7. Metriche di segmentazione
# ----------------------------------------------------------------------------------------------
def fig_metrics():
    M = np.array([[70, 3, 2], [4, 12, 1], [2, 1, 5]])  # righe = ground truth, colonne = predizione
    names = ['sfondo', 'gatto', 'cane']
    tp = np.diag(M); gt = M.sum(1); pr = M.sum(0)
    iou = tp / (gt + pr - tp)
    acc_pix = tp.sum() / M.sum()
    acc_cls = tp / gt
    triv = np.array([[75, 0, 0], [17, 0, 0], [8, 0, 0]])
    tiou = np.diag(triv) / (triv.sum(1) + triv.sum(0) - np.diag(triv))
    P('[metrics] acc pixel', acc_pix, 'IoU', iou.round(4), 'mIoU', iou.mean().round(4), 'acc per classe', acc_cls.round(4), 'media', acc_cls.mean().round(4))
    P('[metrics] predittore banale: acc', 75 / 100, 'IoU', tiou, 'mIoU', tiou.mean())
    s = Svg(960, 430, 'Metriche per segmentazione: l\'accuratezza di pixel inganna, l\'IoU no')
    s.text(30, 55, 'Matrice di confusione (100 pixel; righe = verita, colonne = predetto)', 12.5, bold=True)
    fills = [[(LGREEN if i == j else (LRED if M[i, j] else '#fff')) for j in range(3)] for i in range(3)]
    s.mat(110, 90, M.tolist(), 56, fills, 14, bold=[[i == j for j in range(3)] for i in range(3)])
    for j in range(3):
        s.text(110 + j * 56 + 28, 82, names[j], 11, anchor='middle')
        s.text(100, 90 + j * 56 + 32, names[j], 11, anchor='end')
    s.text(30, 280, 'Per la classe c:   IoU_c = TP / (TP + FP + FN)   (Jaccard)', 12)
    s.text(30, 300, 'accuratezza di pixel = sum_c TP_c / N ;   accuratezza per classe = media_c TP_c / (TP_c + FN_c)', 11.5)
    s.rect(430, 60, 500, 190, YEL, '#d69e2e')
    s.lines(442, 82, [f'Pixel corretti: 70+12+5 = 87 => accuratezza di pixel = {acc_pix:.2f}',
                      f'IoU sfondo = 70/(70+6+5)  = {iou[0]:.3f}', f'IoU gatto   = 12/(12+4+5)  = {iou[1]:.3f}',
                      f'IoU cane    = 5/(5+3+3)    = {iou[2]:.3f}', f'mIoU = media = {iou.mean():.3f}',
                      f'Acc. per classe: {acc_cls[0]:.3f}, {acc_cls[1]:.3f}, {acc_cls[2]:.3f} -> media {acc_cls.mean():.3f}'], 12)
    s.rect(30, 330, 900, 85, LRED, RED)
    s.lines(42, 352, ['Predittore banale "tutto sfondo": accuratezza di pixel = 75/100 = 0.75, ma IoU = (0.75, 0, 0) e mIoU = 0.25.',
                      'Per questo, con classi molto sbilanciate (heavy tails), si usa la IoU (media sulle classi e sulle immagini) e non l\'accuratezza di pixel.'], 12)
    s.save('fig_metrics.svg')


# ----------------------------------------------------------------------------------------------
# 8. IoU e NMS
# ----------------------------------------------------------------------------------------------
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua, inter, ua


def nms(boxes, scores, thr):
    order = list(np.argsort(scores)[::-1]); keep = []; log = []
    while order:
        i = order.pop(0); keep.append(i); rem = []
        for j in order:
            v = iou(boxes[i], boxes[j])[0]
            if v > thr:
                log.append((i, j, v, 'soppresso'))
            else:
                rem.append(j)
        order = rem
    return keep, log


def fig_nms():
    boxes = [(20, 40, 100, 120), (30, 45, 110, 125), (25, 60, 95, 130), (150, 50, 220, 120), (155, 55, 225, 128)]
    scores = [0.95, 0.90, 0.60, 0.85, 0.55]
    names = 'ABCDE'
    I = np.array([[iou(a, b)[0] for b in boxes] for a in boxes])
    keep, log = nms(boxes, scores, 0.5)
    P('[nms] IoU\n', I.round(3), '\n tenuti', [names[i] for i in keep], '\n log', [(names[a], names[b], round(v, 3)) for a, b, v, _ in log])
    a, b = boxes[0], boxes[1]
    v, inter, ua = iou(a, b)
    P('[nms] IoU(A,B)=', v, 'inter', inter, 'unione', ua)
    s = Svg(960, 470, 'IoU e Non-Maximum Suppression')
    cols = [BLUE, RED, GREEN, ORANGE, PURP]
    for panel in range(2):
        x0 = 30 + panel * 330
        s.rect(x0, 50, 260, 220, '#f7fafc', '#cbd5e0', 1, 0)
        s.text(x0, 44, ['Tutte le predizioni (classe: cane)', 'Dopo NMS (soglia 0.5)'][panel], 12, bold=True)
        for i, bx in enumerate(boxes):
            if panel == 1 and i not in keep:
                continue
            s.rect(x0 + bx[0] + 10, 50 + bx[1] - 20, bx[2] - bx[0], bx[3] - bx[1], 'none', cols[i], 2.2, 0)
            s.text(x0 + bx[0] + 14, 50 + bx[1] - 6, f'{names[i]} {scores[i]:.2f}', 10, cols[i], bold=True)
    s.text(690, 60, 'IoU = area(A n B) / area(A u B)', 12.5, bold=True)
    s.lines(690, 82, [f'A=(20,40,100,120), B=(30,45,110,125)', f'intersezione = {inter:.0f}, unione = {ua:.0f}', f'IoU(A,B) = {inter:.0f}/{ua:.0f} = {v:.3f}'], 11.5)
    M = [[f'{x:.2f}' for x in r] for r in I]
    fills = [[(LRED if (i != j and I[i, j] > 0.5) else '#fff') for j in range(5)] for i in range(5)]
    s.mat(690, 150, M, 44, fills, 10.5)
    for j in range(5):
        s.text(690 + j * 44 + 22, 144, names[j], 11, anchor='middle'); s.text(682, 150 + j * 44 + 26, names[j], 11, anchor='end')
    s.rect(30, 290, 640, 165, YEL, '#d69e2e')
    s.lines(42, 312, ['Algoritmo (greedy): ordina per punteggio -> prendi il migliore -> elimina i box con IoU > 0.5 con esso -> ripeti.',
                      f'Ordine: A(0.95), B(0.90), D(0.85), C(0.60), E(0.55).',
                      f'1) tengo A; IoU(A,B) = {I[0, 1]:.2f} > 0.5 e IoU(A,C) = {I[0, 2]:.2f} > 0.5: B e C soppressi.',
                      f'2) tengo D; IoU(D,E) = {I[3, 4]:.2f} > 0.5: E soppresso.', f'Risultato: {[names[i] for i in keep]}.',
                      'Limite: due oggetti veri molto sovrapposti (folla) possono essere soppressi per errore.'], 11.5)
    s.save('fig_nms.svg')


# ----------------------------------------------------------------------------------------------
# 9. RoI pool vs RoI align
# ----------------------------------------------------------------------------------------------
def bilinear(F, x, y):
    fx, fy = x - 0.5, y - 0.5
    fx = min(max(fx, 0), F.shape[1] - 1); fy = min(max(fy, 0), F.shape[0] - 1)
    x1, y1 = int(math.floor(fx)), int(math.floor(fy)); x2, y2 = min(x1 + 1, F.shape[1] - 1), min(y1 + 1, F.shape[0] - 1)
    dx, dy = fx - x1, fy - y1
    return (F[y1, x1] * (1 - dx) * (1 - dy) + F[y1, x2] * dx * (1 - dy) + F[y2, x1] * (1 - dx) * dy + F[y2, x2] * dx * dy)


def fig_roi():
    F = np.array([[2, 4, 3, 1, 5, 2, 6, 3], [3, 8, 4, 2, 6, 7, 1, 2], [1, 5, 9, 6, 3, 4, 2, 5],
                  [4, 2, 7, 8, 5, 6, 3, 1], [6, 3, 4, 5, 9, 2, 4, 6], [2, 5, 1, 3, 4, 7, 8, 2]], float)
    x0, y0, x1, y1 = 1.3, 0.8, 5.7, 4.6
    cx0, cx1, cy0, cy1 = int(math.floor(x0)), int(math.ceil(x1)), int(math.floor(y0)), int(math.ceil(y1))
    sub = F[cy0:cy1, cx0:cx1]
    rs = np.array_split(np.arange(sub.shape[0]), 2); cs = np.array_split(np.arange(sub.shape[1]), 2)
    pool = np.array([[sub[np.ix_(r, c)].max() for c in cs] for r in rs])
    xe = np.linspace(x0, x1, 3); ye = np.linspace(y0, y1, 3)
    al = np.zeros((2, 2)); al_max = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            vals = []
            for a in (0.25, 0.75):
                for b in (0.25, 0.75):
                    vals.append(bilinear(F, xe[j] + a * (xe[j + 1] - xe[j]), ye[i] + b * (ye[i + 1] - ye[i])))
            al[i, j] = np.mean(vals); al_max[i, j] = np.max(vals)
    P('[roi] snap: colonne', cx0, cx1 - 1, 'righe', cy0, cy1 - 1, '\n RoI pool\n', pool, '\n RoI align (media)\n', al.round(3), '\n RoI align (max)\n', al_max.round(3))
    P('[roi] esempio bilinear in (3.5,2.0):', round(float(bilinear(F, 3.5, 2.0)), 3))
    cell = 42
    s = Svg(980, 470, 'Ritaglio di feature: RoI Pool ("snapping") contro RoI Align (interpolazione bilineare)')
    gx, gy = 30, 70
    fills = [[LGRAY] * 8 for _ in range(6)]
    for i in range(cy0, cy1):
        for j in range(cx0, cx1):
            fills[i][j] = LORANGE
    s.mat(gx, gy, F.astype(int).tolist(), cell, fills, 13)
    s.rect(gx + x0 * cell, gy + y0 * cell, (x1 - x0) * cell, (y1 - y0) * cell, 'none', BLUE, 3, 0)
    s.rect(gx + cx0 * cell, gy + cy0 * cell, (cx1 - cx0) * cell, (cy1 - cy0) * cell, 'none', RED, 2, 0, '6,4')
    s.text(gx, 56, 'Mappa di feature 6x8 (un canale)', 12, bold=True)
    s.text(gx, gy + 6 * cell + 22, 'blu: proposta reale (1.3, 0.8)-(5.7, 4.6)   rosso tratteggiato: dopo lo snapping (colonne 1-5, righe 0-4)', 11, GRAY)
    s.text(500, 70, 'RoI Pool', 13, bold=True)
    s.text(500, 90, 'snap + griglia 2x2 + max', 11, GRAY)
    s.mat(500, 105, pool.astype(int).tolist(), 54, [[LRED] * 2] * 2, 15)
    s.text(680, 70, 'RoI Align', 13, bold=True)
    s.text(680, 90, '4 campioni bilineari per cella, media', 11, GRAY)
    s.mat(680, 105, [[f'{v:.2f}' for v in r] for r in al], 54, [[LGREEN] * 2] * 2, 13)
    s.rect(500, 235, 450, 200, YEL, '#d69e2e')
    s.lines(512, 257, ['Bilineare: f(x,y) = (1-dx)(1-dy) f11 + dx(1-dy) f21 + (1-dx) dy f12 + dx dy f22',
                       '(f_ij = feature dei 4 vicini, dx,dy = distanza frazionaria).',
                       'RoI Pool arrotonda le coordinate a celle intere: la regione',
                       'ritagliata e leggermente spostata e di taglia diversa da quella vera',
                       '(0.3 celle di errore su ogni lato = ~5 pixel con stride 16),',
                       'e il max e un\'operazione discreta. RoI Align non arrotonda mai:',
                       'e differenziabile rispetto ai bordi e mantiene l\'allineamento',
                       '(essenziale per le maschere di Mask R-CNN).'], 11.5)
    s.save('fig_roi.svg')


# ----------------------------------------------------------------------------------------------
# 10. RPN
# ----------------------------------------------------------------------------------------------
def fig_rpn():
    scales = [128, 256, 512]; ratios = [0.5, 1.0, 2.0]
    anchors = []
    for sc in scales:
        for r in ratios:
            w = sc * math.sqrt(r); h = sc / math.sqrt(r); anchors.append((w, h))
    K = len(anchors); Hf, Wf = 15, 20
    n_anchors = K * Hf * Wf
    P('[rpn] K', K, 'ancore totali', n_anchors, 'canali cls', K, 'canali reg', 4 * K)
    xa, ya, wa, ha = 100., 100., 128., 128.
    t = (0.1, -0.2, 0.3, -0.1)
    x = xa + t[0] * wa; y = ya + t[1] * ha; w = wa * math.exp(t[2]); h = ha * math.exp(t[3])
    P('[rpn] decode: centro', round(x, 2), round(y, 2), 'w', round(w, 2), 'h', round(h, 2))
    s = Svg(960, 470, 'Region Proposal Network: ancore + due teste convoluzionali')
    s.text(30, 55, f'K = {K} ancore per posizione (3 scale x 3 rapporti)', 12.5, bold=True)
    cx, cy = 190, 190
    cols = [BLUE, GREEN, RED]
    for i, (w_, h_) in enumerate(anchors):
        sc = 0.55
        s.rect(cx - w_ * sc / 2, cy - h_ * sc / 2, w_ * sc, h_ * sc, 'none', cols[i // 3], 1.6, 0, ['none', '5,3', '2,3'][i % 3] if i % 3 else None)
    s.circle(cx, cy, 3, '#000', '#000')
    s.text(30, 380, 'blu/verde/rosso = scala 128/256/512; continuo / tratteggio / punteggiato = rapporto 1:2, 1:1, 2:1', 10.5, GRAY)
    s.text(430, 55, 'Teste sulla feature map (es. 512 x 15 x 20)', 12.5, bold=True)
    s.box(430, 75, 130, 50, ['feature map', '512 x 15 x 20'], LBLUE, size=11)
    s.box(600, 75, 90, 50, 'conv 3x3', LORANGE, size=11)
    s.arrow(560, 100, 600, 100); s.arrow(690, 100, 730, 70); s.arrow(690, 100, 730, 135)
    s.box(730, 45, 200, 50, [f'objectness: K x 15 x 20', f'= {K} x 15 x 20'], LGREEN, size=11)
    s.box(730, 110, 200, 50, [f'trasformazioni: 4K x 15 x 20', f'= {4 * K} x 15 x 20'], LGREEN, size=11)
    s.rect(430, 190, 500, 215, YEL, '#d69e2e')
    s.lines(442, 212, [f'Ancore totali = K x 15 x 20 = {n_anchors}. Si ordinano per objectness, NMS, si tengono le ~300 migliori.',
                       'Parametrizzazione del box (ancora (xa,ya,wa,ha) -> box (x,y,w,h)):',
                       '  tx=(x-xa)/wa, ty=(y-ya)/ha, tw=log(w/wa), th=log(h/ha)',
                       f'Esempio: ancora (100,100,128,128), t=(0.1,-0.2,0.3,-0.1):',
                       f'  x = 100+0.1*128 = {x:.1f},  y = 100-0.2*128 = {y:.1f},',
                       f'  w = 128*e^0.3 = {w:.1f},  h = 128*e^-0.1 = {h:.1f}.',
                       'Le 4 loss di Faster R-CNN: objectness (RPN), box (RPN), classe, box (finale).'], 11.2)
    s.save('fig_rpn.svg')


# ----------------------------------------------------------------------------------------------
# 11. R-CNN -> Fast -> Faster
# ----------------------------------------------------------------------------------------------
def fig_pipeline():
    s = Svg(980, 460, 'Da R-CNN a Faster R-CNN: dove si condivide il calcolo')
    rows = [('R-CNN', [('immagine', LGRAY), ('~2000 proposte (Selective Search)', LORANGE), ('crop + warp 2000 volte', LORANGE), ('CNN x 2000', LRED), ('SVM + regressione box', LBLUE)], '2000 passaggi'),
            ('Fast R-CNN', [('immagine', LGRAY), ('CNN x 1 (tutta l\'immagine)', LGREEN), ('proposte (Selective Search)', LORANGE), ('RoI Pool su feature', LGREEN), ('FC: classe + box', LBLUE)], '1 passaggio (+ proposte)'),
            ('Faster R-CNN', [('immagine', LGRAY), ('CNN x 1 (backbone)', LGREEN), ('RPN (proposte dalla CNN)', LGREEN), ('RoI Pool/Align', LGREEN), ('classe + box', LBLUE)], '1 passaggio, tutto nella rete')]
    for r, (nm, bl, note) in enumerate(rows):
        y = 60 + r * 105
        s.text(20, y + 30, nm, 13, bold=True)
        x = 130
        for k, (lab, col) in enumerate(bl):
            w = 135
            s.box(x, y, w, 55, lab if len(lab) < 20 else [lab[:19], lab[19:]], col, size=10)
            if k < len(bl) - 1: s.arrow(x + w, y + 27, x + w + 12, y + 27)
            x += w + 12
        s.text(130, y + 75, 'costo del backbone: ' + note, 11, GRAY)
    s.rect(20, 380, 940, 70, YEL, '#d69e2e')
    s.lines(32, 400, ['Idea chiave di Fast R-CNN: scambiare convoluzione e ritaglio ("process image before cropping") => backbone una sola volta per immagine.',
                      'Idea chiave di Faster R-CNN: eliminare anche le proposte esterne (CPU) con la RPN; il collo di bottiglia diventava il tempo delle proposte.',
                      'Ordini di grandezza del tempo per immagine riportati in letteratura: ~49 s, ~2.3 s, ~0.2 s (valori indicativi, non nel testo delle slide).'], 11)
    s.save('fig_pipeline.svg')


# ----------------------------------------------------------------------------------------------
# 12. YOLO
# ----------------------------------------------------------------------------------------------
def fig_yolo():
    S, B, Cn = 7, 3, 20
    tot = S * S * (5 * B + Cn)
    P('[yolo] output', S, S, 5 * B + Cn, '=', tot, '; YOLOv1 (B=2,C=20):', S * S * (5 * 2 + 20), '; box candidati', S * S * B)
    cell = 40
    cxp, cyp = 170., 130.
    ci, cj = int(cyp // cell), int(cxp // cell)
    ox, oy = cxp / cell - cj, cyp / cell - ci
    P('[yolo] centro', (cxp, cyp), 'cella (riga,col)', (ci, cj), 'offset', (ox, oy))
    s = Svg(980, 440, 'Single-stage: YOLO (griglia 7x7, B=3 box base per cella, C=20 classi)')
    gx, gy = 30, 60
    for i in range(S):
        for j in range(S):
            s.rect(gx + j * cell, gy + i * cell, cell, cell, YEL if (i, j) == (ci, cj) else '#f7fafc', '#cbd5e0', 0.8, 0)
    s.rect(gx + 110, gy + 70, 130, 120, 'none', RED, 2.5, 0)
    s.circle(gx + cxp, gy + cyp, 4, RED, RED)
    s.rect(gx + cxp - 40, gy + cyp - 40, 80, 80, 'none', BLUE, 1.4, 0, '5,3')
    s.rect(gx + cxp - 55, gy + cyp - 35, 110, 70, 'none', GREEN, 1.4, 0, '5,3')
    s.rect(gx + cxp - 35, gy + cyp - 60, 70, 120, 'none', ORANGE, 1.4, 0, '5,3')
    s.text(gx, gy + S * cell + 20, 'rosso = oggetto vero; punto = suo centro, che cade nella cella gialla (riga 3, colonna 4);', 11, GRAY)
    s.text(gx, gy + S * cell + 36, 'tratteggiati = B=3 box base di quella cella, da cui si regredisce (dx,dy,dw,dh,conf).', 11, GRAY)
    s.text(400, 60, 'Vettore di una cella (35 numeri)', 13, bold=True)
    x = 400
    for b in range(B):
        s.box(x, 80, 55, 46, [f'box {b + 1}', '5'], [LBLUE, LGREEN, LORANGE][b], size=10.5)
        s.text(x + 27, 142, 'dx dy dw dh conf', 8, GRAY, 'middle'); x += 60
    s.box(x, 80, 180, 46, ['C = 20 punteggi di classe', '(incluso sfondo come classe)'], LRED, size=10.5)
    s.text(400, 168, 'Uscita: 7 x 7 x (5B + C) = 7 x 7 x 35', 12, bold=True)
    s.rect(400, 190, 550, 215, YEL, '#d69e2e')
    s.lines(412, 212, [f'Numeri totali = 49 x 35 = {tot}. Box candidati = 49 x 3 = {S * S * B}.',
                       f'Centro (170,130) px, cella {cell} px: colonna {cj}, riga {ci}; offset nella cella = ({ox:.2f}, {oy:.2f}).',
                       'Una cella e "responsabile" dell\'oggetto se ne contiene il centro.',
                       'Si assomiglia alla RPN (ancore + regressione) ma e specifica per categoria:',
                       'classifica e localizza in un solo passaggio, senza secondo stadio.',
                       'Poi NMS per eliminare i duplicati. Perdita: somma di errori quadratici su',
                       'coordinate, confidenza e classi (YOLOv1); RetinaNet usa la focal loss.',
                       'YOLOv1 originale: B=2, C=20 => 7x7x30 = 1470 numeri (le slide usano B=3).'], 11.2)
    s.save('fig_yolo.svg')


# ----------------------------------------------------------------------------------------------
# 13. DETR - bipartite matching
# ----------------------------------------------------------------------------------------------
def fig_detr():
    classes = ['cane', 'gatto', 'nessun oggetto']
    pc = np.array([[0.80, 0.10, 0.10], [0.30, 0.60, 0.10], [0.10, 0.10, 0.80], [0.50, 0.20, 0.30]])
    pb = np.array([[0.30, 0.40, 0.20, 0.30], [0.70, 0.60, 0.20, 0.20], [0.50, 0.50, 0.10, 0.10], [0.33, 0.43, 0.22, 0.28]])
    gb = np.array([[0.30, 0.40, 0.20, 0.30], [0.70, 0.60, 0.20, 0.20]]); gc = [0, 1]
    lam = 5.0
    cost = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            cost[i, j] = -pc[i, gc[j]] + lam * np.abs(pb[i] - gb[j]).sum()
    best, bp = 1e9, None
    for perm in itertools.permutations(range(4), 2):
        c = sum(cost[perm[j], j] for j in range(2))
        if c < best: best, bp = c, perm
    P('[detr] costo\n', cost.round(3), '\n assegnamento (query per GT)', bp, 'costo totale', round(best, 3))
    s = Svg(960, 440, 'DETR: insieme di box predetto + abbinamento bipartito con i box veri')
    s.text(30, 55, 'Matrice dei costi  C[i,j] = -p_i(c_j) + 5 ||b_i - g_j||_1', 12.5, bold=True)
    fills = [[(LGREEN if (i in bp and bp.index(i) == j) else '#fff') for j in range(2)] for i in range(4)]
    s.mat(150, 95, [[f'{v:.2f}' for v in r] for r in cost], 70, fills, 13, bold=[[(i in bp and bp.index(i) == j) for j in range(2)] for i in range(4)])
    for j, g in enumerate(['GT 1: cane', 'GT 2: gatto']):
        s.text(150 + j * 70 + 35, 87, g, 10, anchor='middle')
    for i in range(4):
        s.text(140, 95 + i * 70 + 40, f'query {i + 1}', 11, anchor='end')
    s.text(30, 395, f'Assegnamento ottimo (verde): query {bp[0] + 1} -> cane, query {bp[1] + 1} -> gatto; costo totale {best:.2f}.', 12)
    s.rect(450, 70, 490, 330, YEL, '#d69e2e')
    s.lines(462, 92, ['N_q query (qui 4) producono ciascuna (classe, box) in parallelo (decoder del', 'Transformer, query apprese), senza ancore e senza NMS.', '',
                      'Abbinamento: sigma* = argmin_sigma sum_j C[sigma(j), j]  (algoritmo ungherese;', 'qui si enumerano le 4*3 = 12 permutazioni).', '',
                      'Loss (Hungarian loss) sulle coppie abbinate:  -log p_sigma(j)(c_j) + lambda_L1 ||b - g||_1', '   + lambda_giou L_GIoU(b, g).',
                      'Le query NON abbinate (query 3 e 4) devono predire "nessun oggetto" (classe vuota),', 'tipicamente con peso ridotto perche sono la maggioranza.', '',
                      'Query 4 e un duplicato di query 1 (stesso cane): proprio perche l\'abbinamento e', 'uno-a-uno, il duplicato viene penalizzato => niente NMS.'], 11.2)
    s.save('fig_detr.svg')


# ----------------------------------------------------------------------------------------------
# 14. Precision-recall e AP
# ----------------------------------------------------------------------------------------------
def fig_ap():
    tpfp = [1, 1, 0, 1, 0, 0, 1, 0]; ngt = 5
    tp = np.cumsum(tpfp); fp = np.cumsum([1 - t for t in tpfp])
    prec = tp / (tp + fp); rec = tp / ngt
    pint = prec.copy()
    for i in range(len(pint) - 2, -1, -1):
        pint[i] = max(pint[i], pint[i + 1])
    rr = np.concatenate([[0], rec]); ap = sum((rr[i + 1] - rr[i]) * pint[i] for i in range(len(rec)))
    P('[ap] precisione', prec.round(3), 'recall', rec.round(3), 'prec. interpolata', pint.round(3), 'AP', round(ap, 4))
    s = Svg(960, 450, 'Valutazione della detection: curva precisione-richiamo e Average Precision')
    ox, oy, W, H = 60, 70, 300, 280
    s.rect(ox, oy, W, H, '#fff', '#333', 1, 0)
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        s.line(ox + t * W, oy + H, ox + t * W, oy + H + 4); s.text(ox + t * W, oy + H + 18, f'{t:.1f}', 10, anchor='middle')
        s.line(ox - 4, oy + H - t * H, ox, oy + H - t * H); s.text(ox - 8, oy + H - t * H + 4, f'{t:.1f}', 10, anchor='end')
    s.text(ox + W / 2, oy + H + 38, 'richiamo (recall)', 11, anchor='middle'); s.text(ox - 40, oy + 10, 'precisione', 11)
    pts = [(ox, oy + H - prec[0] * H)] + [(ox + r * W, oy + H - p * H) for r, p in zip(rec, prec)]
    s.poly(pts, BLUE, 2)
    for r, p in zip(rec, prec): s.circle(ox + r * W, oy + H - p * H, 3.5, BLUE, BLUE)
    step = [(ox, oy + H - pint[0] * H)]
    for i in range(len(rec)):
        step.append((ox + rec[i] * W, oy + H - pint[i] * H))
        if i + 1 < len(rec): step.append((ox + rec[i] * W, oy + H - pint[i + 1] * H))
    s.poly(step, RED, 2.2)
    s.text(ox + 140, oy + 30, 'blu: curva P-R', 11, BLUE); s.text(ox + 140, oy + 46, 'rosso: precisione interpolata', 11, RED)
    s.text(430, 70, 'Detezioni ordinate per confidenza (5 oggetti veri)', 12.5, bold=True)
    hdr = ['#', 'TP/FP', 'prec.', 'recall']
    for j, hd in enumerate(hdr): s.text(430 + j * 70, 95, hd, 11, bold=True)
    for i in range(8):
        y = 115 + i * 22
        s.rect(425, y - 15, 290, 20, LGREEN if tpfp[i] else LRED, '#cbd5e0', 0.5, 0)
        for j, v in enumerate([str(i + 1), 'TP' if tpfp[i] else 'FP', f'{prec[i]:.3f}', f'{rec[i]:.2f}']):
            s.text(430 + j * 70, y, v, 11)
    s.rect(430, 310, 510, 120, YEL, '#d69e2e')
    s.lines(442, 330, ['Una detezione e TP se IoU >= soglia (es. 0.5) con un oggetto non ancora abbinato e classe giusta; altrimenti FP.',
                       'precisione = TP/(TP+FP), richiamo = TP/(# oggetti veri).  AP = area sotto la curva P-R interpolata',
                       f'(precisione = massimo a destra): qui AP = {ap:.3f}.  mAP = media di AP sulle classi.',
                       'COCO: AP mediata anche su soglie IoU 0.50, 0.55, ..., 0.95 (premia la localizzazione precisa).'], 11)
    s.save('fig_ap.svg')


# ----------------------------------------------------------------------------------------------
# 15. Mask R-CNN
# ----------------------------------------------------------------------------------------------
def fig_maskrcnn():
    G = np.zeros((16, 16), int)
    yy, xx = np.mgrid[0:16, 0:16]
    G[((yy - 8) ** 2 / 25 + (xx - 8) ** 2 / 16) <= 1] = 1
    r0, r1, c0, c1 = 2, 14, 3, 14
    crop = G[r0:r1, c0:c1]
    ri = (np.arange(8) * crop.shape[0] / 8 + crop.shape[0] / 16).astype(int)
    ci = (np.arange(8) * crop.shape[1] / 8 + crop.shape[1] / 16).astype(int)
    target = crop[np.ix_(ri, ci)]
    P('[mask] ritaglio', crop.shape, '-> bersaglio 8x8\n', target)
    s = Svg(980, 470, 'Mask R-CNN: un ramo che predice una maschera binaria per ogni RoI')
    xs = 30
    for lab, col, w in [('immagine', LGRAY, 80), ('CNN + RPN', LBLUE, 90), ('RoIAlign', LGREEN, 90), ('256x14x14', LORANGE, 90), ('conv', LORANGE, 60), ('256x14x14', LORANGE, 90), ('deconv 2x2, s=2', LORANGE, 110), ('conv 1x1', LORANGE, 80)]:
        s.box(xs, 60, w, 46, lab, col, size=10); s.arrow(xs + w, 83, xs + w + 8, 83); xs += w + 8
    s.box(xs, 60, 100, 46, ['C x 28 x 28', '(una per classe)'], LRED, size=10)
    s.text(30, 135, 'Ramo classificazione/box (come Faster R-CNN): classi C, box 4C.  Ramo maschera: C maschere 28x28 per RoI.', 11.5, GRAY)
    s.text(30, 175, 'Costruzione del bersaglio (esempio ridotto: maschera vera 16x16, proposta = righe 2-13, colonne 3-13, uscita 8x8)', 12, bold=True)
    c = 10
    s.mat(30, 195, G.tolist(), c, [[LBLUE if G[i, j] else '#fff' for j in range(16)] for i in range(16)], 1, '')
    s.rect(30 + c0 * c, 195 + r0 * c, (c1 - c0) * c, (r1 - r0) * c, 'none', RED, 2.5, 0)
    s.text(30, 375, 'maschera vera + proposta (rosso)', 11, GRAY)
    s.arrow(205, 275, 260, 275); s.text(232, 265, 'ritaglia', 10, anchor='middle')
    s.mat(270, 215, target.tolist(), 18, [[LGREEN if target[i, j] else '#fff' for j in range(8)] for i in range(8)], 10, '{:g}')
    s.text(270, 375, 'dopo il ritaglio e il ridimensionamento a 8x8 (28x28 in pratica)', 11, GRAY)
    s.rect(470, 190, 480, 250, YEL, '#d69e2e')
    s.lines(482, 212, ['Loss della maschera: entropia incrociata binaria per pixel, calcolata SOLO', 'sul canale della classe vera k:  L_mask = media_(u,v) BCE(sigma(m_k(u,v)), t(u,v)).',
                       '(Le altre C-1 maschere non contribuiscono: nessuna competizione tra classi,', 'si decide "che cosa" con il ramo di classificazione, "dove" con la maschera.)', '',
                       'Loss totale:  L = L_cls + L_box + L_mask.', 'RoIAlign (non RoIPool) e necessario: con lo snapping la maschera risulterebbe', 'disallineata di diversi pixel rispetto all\'oggetto.', '',
                       'Le slide: "predice una maschera binaria 28x28 per ognuna delle C classi" e', 'mostrano anche la stima della posa come estensione.'], 11.2)
    s.save('fig_maskrcnn.svg')


# ----------------------------------------------------------------------------------------------
# 16. Focal loss
# ----------------------------------------------------------------------------------------------
def fig_focal():
    ps = np.linspace(0.02, 0.999, 120)
    gs = [(0, BLUE), (1, GREEN), (2, RED), (5, PURP)]
    for g, _ in gs:
        P('[focal] gamma', g, ' p_t=0.9 ->', round(-(0.1) ** g * math.log(0.9), 5), ' p_t=0.1 ->', round(-(0.9) ** g * math.log(0.1), 4))
    P('[focal] rapporto (gamma=2) tra difficile 0.1 e facile 0.9:', round((-(0.9) ** 2 * math.log(0.1)) / (-(0.1) ** 2 * math.log(0.9)), 1))
    s = Svg(900, 420, 'Focal loss FL(p_t) = -(1 - p_t)^gamma log p_t (RetinaNet): ridurre il peso degli esempi facili')
    ox, oy, W, H = 70, 60, 420, 300
    s.rect(ox, oy, W, H, '#fff', '#333', 1, 0)
    ymax = 4.0
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        s.text(ox + t * W, oy + H + 16, f'{t:.1f}', 10, anchor='middle')
    for v in (1, 2, 3, 4):
        s.text(ox - 8, oy + H - v / ymax * H + 4, str(v), 10, anchor='end')
    s.text(ox + W / 2, oy + H + 36, 'p_t (probabilita della classe giusta)', 11, anchor='middle'); s.text(ox - 40, oy + 12, 'loss', 11)
    for g, col in gs:
        pts = [(ox + p * W, oy + H - min(-(1 - p) ** g * math.log(p), ymax) / ymax * H) for p in ps]
        s.poly(pts, col, 2.2)
    for k, (g, col) in enumerate(gs):
        s.line(ox + 250, oy + 30 + k * 20, ox + 280, oy + 30 + k * 20, col, 2.5); s.text(ox + 288, oy + 34 + k * 20, f'gamma = {g}' + (' (cross-entropy)' if g == 0 else ''), 11)
    s.rect(520, 70, 360, 290, YEL, '#d69e2e')
    s.lines(532, 92, ['Detector a singolo stadio: ~10^4-10^5 candidati per immagine,', 'quasi tutti sfondo facile => la loss e dominata dai facili.', '',
                      'Con gamma = 2:', '  esempio facile p_t = 0.9: (0.1)^2 * 0.105 = 0.00105', '  esempio difficile p_t = 0.1: (0.9)^2 * 2.303 = 1.865',
                      '  rapporto difficile/facile = 1770 (con CE: 21.9)', '', 'Il fattore (1 - p_t)^gamma "spegne" gli esempi gia', 'classificati bene e concentra il gradiente sui difficili.',
                      '(Di solito si aggiunge un peso alpha_t di bilanciamento.)'], 11.5)
    s.save('fig_focal.svg')


# ----------------------------------------------------------------------------------------------
# 17. Oltre il 2D
# ----------------------------------------------------------------------------------------------
def fig_3d():
    V, F = 2562, 5120
    P('[3d] voxel 32^3', 32 ** 3, '64^3', 64 ** 3, '128^3', 128 ** 3, 'nuvola 2048x3', 2048 * 3, 'mesh', V * 3, F * 3)
    f, X, Z = 500., 1.0, 5.0
    P('[3d] proiezione u = f X/Z =', f * X / Z, ' (X,Z)=(2,10) ->', f * 2 / 10)
    s = Svg(980, 430, 'Oltre il 2D: da un raggio a una scatola 3D, e rappresentazioni di forme 3D')
    s.text(30, 55, 'Modello a foro stenopeico: un punto 2D e un raggio 3D', 12.5, bold=True)
    s.circle(60, 200, 5, '#000', '#000'); s.text(60, 225, 'camera', 11, anchor='middle')
    s.line(260, 100, 260, 300, GRAY, 3); s.text(260, 92, 'piano immagine', 11, anchor='middle')
    s.arrow(60, 200, 440, 120, GRAY, 1.2, '4,3')
    for k, (px, py, lab) in enumerate([(160, 184, 'Z=1'), (260, 168, ''), (360, 136, 'Z=3'), (440, 120, 'Z=5')]):
        pass
    s.circle(260, 168, 5, RED, RED); s.text(272, 160, 'pixel (u,v)', 11, RED)
    for px, py, col in [(150, 186, BLUE), (350, 138, GREEN), (430, 122, ORANGE)]:
        s.circle(px, py, 6, col, col)
    s.text(150, 214, 'stesso pixel,', 10, BLUE, 'middle'); s.text(150, 227, 'profondita diverse', 10, BLUE, 'middle')
    s.text(30, 290, 'u = f X / Z + c_x ,   v = f Y / Z + c_y   (da 3D a 2D si perde Z)', 11.5)
    s.text(30, 310, 'Da una sola immagine (monoculare) la profondita e ambigua: si regredisce', 11.5)
    s.text(30, 328, 'box 3D = (x,y,z,w,h,l,theta) con un prior (es. Faster R-CNN + testa 3D).', 11.5)
    s.text(520, 55, 'Rappresentazioni di forme 3D (esempio numerico)', 12.5, bold=True)
    items = [('Nuvola di punti', 'V x 3 float', '2048 x 3 = 6144 numeri', LBLUE), ('Voxel', 'D x D x D binario', '32^3 = 32 768;  128^3 = 2 097 152', LORANGE),
             ('Mesh', 'V x 3 float + F x 3 int', f'V={V}: {V * 3} + F={F}: {F * 3} numeri', LGREEN)]
    for k, (nm, sh, ex, col) in enumerate(items):
        s.box(520, 75 + k * 70, 150, 55, [nm, sh], col, size=11); s.text(685, 107 + k * 70, ex, 11.5)
    s.rect(520, 295, 440, 115, YEL, '#d69e2e')
    s.lines(532, 315, ['Per confrontare due nuvole di punti A, B si usa la distanza di Chamfer:', '  d_CD(A,B) = (1/|A|) sum_a min_b ||a-b||^2 + (1/|B|) sum_b min_a ||a-b||^2.',
                       'Mesh: F ~ 2V (formula di Eulero), quindi 3V + 3F ~ 9V numeri.', 'Mesh R-CNN: Mask R-CNN + voxel grossolani + raffinamento della mesh.'], 11)
    s.save('fig_3d.svg')


if __name__ == '__main__':
    for fn in (fig_tasks, fig_sliding_fcn, fig_unpool, fig_tconv, fig_convmatrix, fig_arch, fig_metrics, fig_nms, fig_roi,
               fig_rpn, fig_pipeline, fig_yolo, fig_detr, fig_ap, fig_maskrcnn, fig_focal, fig_3d):
        fn()
