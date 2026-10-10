# Performance dei Sistemi Biometrici — Riassunto Completo

```python
# =====================================================================
# 00 — Helper extra per i nuovi grafici  (richiede 01, 05)
# Da eseguire subito dopo il blocco 05.
# =====================================================================
def person(o, cx, cy, color, s=1.0):
    o.append(f'<circle cx="{cx:.1f}" cy="{cy-17*s:.1f}" r="{7.5*s:.1f}" fill="{color}"/>')
    o.append(f'<rect x="{cx-11*s:.1f}" y="{cy-7*s:.1f}" width="{22*s:.1f}" height="{28*s:.1f}" rx="{7*s:.1f}" fill="{color}"/>')

def badge(o, x, y, n, color, r=11):
    o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
    txt(o, x, y+4.5, str(n), 13, 700, "#fff", halo=False)

def polyline(o, pts, color, sw=2, dash=None, fill="none"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o.append(f'<path d="M ' + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="{fill}" stroke="{color}" stroke-width="{sw}"{d}/>')
```

---

## 1. Fonti di errore nei sistemi biometrici

### 1.1 Intra-class variation
Variazioni **all'interno della stessa classe** (stessa persona): posa, espressione, occhiali, illuminazione. L'immagine ideale è frontale, con illuminazione omogenea ed espressione neutra.

### 1.2 Inter-class variation (piccola)
Somiglianza **tra soggetti diversi** (es. gemelli, padre/figlio), che può creare confusione soprattutto in certe condizioni (espressione simile, stessa illuminazione).

### 1.3 Acquisizioni rumorose/distorte
Qualità del campione scarsa (es. impronte di lavoratori manuali, pelle secca). Si possono applicare tecniche di normalizzazione dell'illuminazione.

**Normalizzazione dell'illuminazione.** Stesso pattern con illuminazione non uniforme: prima il livello medio cresce da sinistra a destra (due acquisizioni della stessa persona risultano lontane), dopo la normalizzazione resta solo il pattern.

```python
# =====================================================================
# N15 — Normalizzazione dell'illuminazione  (richiede 01, 05)
# =====================================================================
W, H = 920, 560
o = new_svg(W, H)
header(o, W, "Rumore e illuminazione: normalizzare prima di confrontare",
       "Stesso pattern, illuminazione non uniforme: dopo la normalizzazione le differenze spurie scompaiono")
NX, NY, CS = 48, 10, 8
def pat(x, y):
    return 0.5+0.38*math.sin(x*0.55+y*0.25)*math.cos(y*0.35)
illum = lambda x: 0.22+0.78*(x/(NX-1))
def gray(v):
    g = int(max(0, min(1, v))*255)
    return f"#{g:02x}{g:02x}{g:02x}"
def panel(x0, title, fn, prof):
    txt(o, x0+NX*CS/2, 98, title, 14.5, 700, "#111", halo=False)
    for y in range(NY):
        for x in range(NX):
            o.append(f'<rect x="{x0+x*CS}" y="{120+y*CS}" width="{CS}" height="{CS}" fill="{gray(fn(x, y))}"/>')
    o.append(f'<rect x="{x0}" y="120" width="{NX*CS}" height="{NY*CS}" fill="none" stroke="#9ca3af"/>')
    # profilo di intensità lungo x (riga centrale)
    BT, BB = 250, 400
    o.append(f'<line x1="{x0}" y1="{BB}" x2="{x0+NX*CS}" y2="{BB}" stroke="#222"/>')
    o.append(f'<line x1="{x0}" y1="{BT}" x2="{x0}" y2="{BB}" stroke="#222"/>')
    polyline(o, [(x0+x*CS+CS/2, BB-fn(x, 5)*(BB-BT)) for x in range(NX)], C_I, 2.4)
    polyline(o, [(x0+x*CS+CS/2, BB-prof(x)*(BB-BT)) for x in range(NX)], C_FA, 2, "5,4")
    txt(o, x0+NX*CS/2, BB+22, "posizione x", 12, None, "#333", halo=False)
    txt(o, x0-6, BT+4, "1", 11, None, "#555", "end", halo=False)
    txt(o, x0-6, BB+4, "0", 11, None, "#555", "end", halo=False)
    txt(o, x0+NX*CS/2, 238, "profilo di intensità (blu) · livello medio locale (rosso)", 11.5, None, "#555", halo=False)
panel(30, "Prima: illuminazione non omogenea", lambda x, y: illum(x)*pat(x, y), lambda x: illum(x)*0.5)
panel(470, "Dopo: normalizzazione dell'illuminazione", lambda x, y: pat(x, y), lambda x: 0.5)
arrow(o, 418, 160, 462, 160, "#111", 2.4)
txt(o, 440, 148, "norm.", 11.5, 700, "#111")
txt(o, W/2, 470, "Prima: il livello medio sale da sinistra a destra, quindi due immagini della stessa persona con luce diversa risultano 'lontane' (intra-class ↑).", 12, None, "#333", halo=False)
txt(o, W/2, 490, "Dopo: si divide per il livello medio locale (stima dell'illuminazione): resta il pattern, la distanza intra-class si riduce.", 12, None, "#333", halo=False)
txt(o, W/2, 520, "Vale anche per campioni rumorosi/distorti (§1.3): filtraggio e normalizzazione migliorano la qualità del campione prima dell'estrazione.", 11.5, None, "#666", halo=False)
save(o, 'n15_normalizzazione_illuminazione.svg')
```
<img src="./img/n15_normalizzazione_illuminazione.svg" alt="Normalizzazione dell'illuminazione" style="display:block; margin:1.5em auto; max-width:100%;">


### 1.4 Non universalità
Una parte della popolazione non può essere riconosciuta da un certo tratto (es. ~4% ha impronte di scarsa qualità).

**Da dove nascono gli errori.** A sinistra: persone diverse come cluster nello spazio delle feature; B ha un'alta variazione intra-class, A e C (es. gemelli) sono vicini. A destra: le due distribuzioni delle distanze si sovrappongono, e quella zona è la fonte degli errori una volta fissata la soglia.

```python
# =====================================================================
# N02 — Intra-class vs inter-class  (richiede 01, 05)
# =====================================================================
random.seed(5)
W, H = 920, 580
o = new_svg(W, H)
header(o, W, "Da dove nascono gli errori: variazione intra-class e inter-class",
       "Prima ancora di soglia, FAR e FRR: i campioni di persone diverse (o della stessa) si sovrappongono")
rect(o, 20, 78, 430, 480, "#fff", "#e5e7eb", 1.5, 14)
txt(o, 235, 104, "Spazio delle feature (proiezione 2D)", 14, 700, "#111", halo=False)
cl = [("A", (115, 280), 12, C_I), ("C", (158, 305), 12, "#db2777"), ("B", (330, 250), 36, C_G)]
for nm, (cx, cy), sd, col in cl:
    o.append(f'<circle cx="{cx}" cy="{cy}" r="{2*sd}" fill="{col}" fill-opacity="0.08" stroke="{col}" stroke-dasharray="5,4" stroke-width="1.5"/>')
    for _ in range(12):
        o.append(f'<circle cx="{random.gauss(cx, sd):.1f}" cy="{random.gauss(cy, sd):.1f}" r="5.5" fill="{col}" fill-opacity="0.85" stroke="#fff"/>')
txt(o, 115, 280-2*12-10, "A", 15, 700, C_I)
txt(o, 158, 305+2*12+20, "C", 15, 700, "#db2777")
txt(o, 330, 250-72-12, "B", 15, 700, C_G_T)
txt(o, 330, 250+72+22, "B: alta variazione intra-class", 12.5, 700, C_G_T)
txt(o, 330, 250+72+38, "(posa, espressione, occhiali, luce)", 11.5, None, "#444")
leader(o, [(136, 420), (136, 292)], "#111")
txt(o, 136, 438, "A e C vicini: inter-class piccola", 12.5, 700, "#111")
txt(o, 136, 454, "(gemelli, padre/figlio)", 11.5, None, "#444")
txt(o, 235, 535, "punto = campione · colore = persona · tratteggio = spread intra-class", 11.5, None, "#555", halo=False)

rect(o, 470, 78, 430, 480, "#fff", "#e5e7eb", 1.5, 14)
txt(o, 685, 104, "Distribuzioni delle distanze", 14, 700, "#111", halo=False)
L, R, TOP, BOT = 500, 880, 150, 380
XMIN, XMAX, YMAX = -0.1, 1.1, 3.8
sx = lambda x: L+(x-XMIN)/(XMAX-XMIN)*(R-L)
sy = lambda y: BOT-y/YMAX*(BOT-TOP)
mi, si, me, se = 0.30, 0.11, 0.58, 0.14
n = 220
xsr = [XMIN+(XMAX-XMIN)*i/n for i in range(n+1)]
ov = [(sx(x), sy(min(pdf(x, mi, si), pdf(x, me, se)))) for x in xsr]
o.append(f'<path d="M {sx(XMIN):.1f},{BOT} ' + " ".join(f"L {x:.1f},{y:.1f}" for x, y in ov) + f' L {sx(XMAX):.1f},{BOT} Z" fill="{C_FA}" fill-opacity="0.45"/>')
o.append(f'<path d="{g_line(mi,si,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_G}" stroke-width="2.6"/>')
o.append(f'<path d="{g_line(me,se,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_I}" stroke-width="2.6"/>')
frame(o, L, R, TOP, BOT, "distanza tra due campioni", None)
o.append(f'<text transform="translate(488,{(TOP+BOT)/2}) rotate(-90)" text-anchor="middle" font-size="13" fill="#222">Densità</text>')
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{sx(v):.1f}" y1="{BOT}" x2="{sx(v):.1f}" y2="{BOT+5}" stroke="#222"/>')
    txt(o, sx(v), BOT+20, f"{v:.1f}", 12, None, "#333", halo=False)
txt(o, sx(0.02), sy(2.7), "intra-class", 14, 700, C_G_T)
txt(o, sx(0.02), sy(2.7)+15, "(stessa persona)", 11.5, None, "#333")
txt(o, sx(0.97), sy(2.9), "inter-class", 14, 700, C_I)
txt(o, sx(0.97), sy(2.9)+15, "(persone diverse)", 11.5, None, "#333")
xo = sx(0.44)
leader(o, [(xo, 441), (xo, sy(min(pdf(0.44, mi, si), pdf(0.44, me, se)))+6)], C_FA_T)
txt(o, xo, 458, "zona di sovrapposizione", 12.5, 700, C_FA_T)
txt(o, 685, 490, "Più la variazione intra-class è grande e la inter-class piccola,", 12, None, "#333")
txt(o, 685, 507, "più l'area rossa cresce: è qui che nasceranno FA e FR", 12, None, "#333")
txt(o, 685, 524, "una volta scelta la soglia (vedi grafico successivo).", 12, None, "#333")
save(o, 'n02_intra_inter_class.svg')
```
<img src="./img/n02_intra_inter_class.svg" alt="Variazione intra-class e inter-class" style="display:block; margin:1.5em auto; max-width:100%;">

### 1.5 Attacchi di spoofing (architettura del sistema)
Punti di attacco della pipeline **Sensore → Feature Extractor → Matcher → Stored Templates → Application Device**:
1. Biometria falsa al sensore
2. Replay di dati vecchi
3. Override del feature extractor
4. Vettore di feature sintetico iniettato
5. Override del matcher
6. Modifica dei template salvati
7. Intercettazione del canale
8. Override della decisione finale

**Gli 8 punti sulla pipeline.** I colori distinguono presentazione al sensore (1), canali (2, 4, 7), override di moduli (3, 5, 8) e database (6).

```python
# =====================================================================
# N01 — Pipeline con gli 8 punti di attacco  (richiede 01, 05, 00)
# =====================================================================
W, H = 920, 610
o = new_svg(W, H)
header(o, W, "Pipeline di un sistema biometrico e i suoi 8 punti di attacco",
       "Sensore → Feature Extractor → Matcher → Decisione → Application Device · gli Stored Templates alimentano il Matcher")
C_PRES, C_CH, C_MOD, C_DB = "#7c3aed", "#2563eb", "#dc2626", "#0f766e"
for i, (c, t) in enumerate([(C_PRES, "presentazione (1)"), (C_CH, "canale / dati in transito (2, 4, 7)"),
                            (C_MOD, "override di un modulo (3, 5, 8)"), (C_DB, "database (6)")]):
    xx = [60, 230, 505, 745][i]
    o.append(f'<circle cx="{xx}" cy="86" r="7" fill="{c}"/>')
    txt(o, xx+13, 90, t, 12.5, None, "#222", "start", halo=False)

cy, bw, bh = 215, 130, 70
xs = [105, 280, 455, 630, 805]
names = [["Sensore", "acquisisce"], ["Feature Extractor", "estrae le feature"], ["Matcher", "confronta"],
         ["Decisione", "soglia t"], ["Application Device", "es. serratura"]]
for x, nm in zip(xs, names):
    box(o, x, cy, bw, bh, nm, BLU_BG, C_I, 12.5)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(o, a+bw/2, cy, b-bw/2, cy)
# template store
box(o, 455, 395, 170, 60, ["Stored Templates", "gallery"], "#ecfdf5", C_DB, 12.5)
arrow(o, 455, 395-30, 455, cy+bh/2, "#222")
# badge sui moduli (angolo in alto a destra)
badge(o, xs[0]+bw/2-4, cy-bh/2+2, 1, C_PRES)
badge(o, xs[1]+bw/2-4, cy-bh/2+2, 3, C_MOD)
badge(o, xs[2]+bw/2-4, cy-bh/2+2, 5, C_MOD)
badge(o, xs[3]+bw/2-4, cy-bh/2+2, 8, C_MOD)
badge(o, 455+85-4, 395-30+2, 6, C_DB)
# badge sui canali (a metà freccia)
badge(o, (xs[0]+xs[1])/2, cy-22, 2, C_CH)
badge(o, (xs[1]+xs[2])/2, cy-22, 4, C_CH)
badge(o, 455, (395-30+cy+bh/2)/2, 7, C_CH)
txt(o, 105, cy-bh/2-16, "dito / volto falso", 11.5, None, C_PRES, halo=False)

items = ["Biometria falsa presentata al sensore (spoofing)", "Replay di dati vecchi (sensore → estrazione)",
         "Override del feature extractor", "Vettore di feature sintetico iniettato",
         "Override del matcher", "Modifica dei template salvati",
         "Intercettazione del canale (templates → matcher)", "Override della decisione finale"]
cols = [C_PRES, C_CH, C_MOD, C_CH, C_MOD, C_DB, C_CH, C_MOD]
txt(o, 60, 470, "I punti di attacco", 14, 700, "#111", "start", halo=False)
for k, (s, c) in enumerate(zip(items, cols)):
    x0 = 60 if k < 4 else 490
    y0 = 500 + (k % 4)*27
    badge(o, x0+11, y0-4, k+1, c, 10)
    txt(o, x0+28, y0, s, 12.5, None, "#222", "start", halo=False)
save(o, 'n01_attacchi_pipeline.svg')
```
<img src="./img/n01_attacchi_pipeline.svg" alt="Pipeline e 8 punti di attacco" style="display:block; margin:1.5em auto; max-width:100%;">


### 1.6 Cosa viene confrontato, e come (tipi di template e misure di similarità)

**Dal sample al template.** Il sample passa dall'estrazione di feature e diventa un template; la forma del template decide la misura di confronto.

```python
# =====================================================================
# N03 — Tipi di template e confronto  (richiede 01, 05)
# =====================================================================
random.seed(3)
W, H = 920, 600
o = new_svg(W, H)
header(o, W, "Dal sample al template: cosa si confronta e come",
       "Il matching non avviene sui dati grezzi ma sui template · la misura dipende dalla forma del template")
cy = 135
box(o, 120, cy, 180, 64, ["Sample", "dato grezzo del sensore", "(immagine, audio, scansione)"], GREY_BG, "#6b7280", 11.5)
box(o, 360, cy, 180, 64, ["Feature extraction", "algoritmo o rete neurale"], BLU_BG, C_I, 11.5)
box(o, 600, cy, 190, 64, ["Template", "rappresentazione di feature"], C_OK_BG, C_G_T, 11.5)
box(o, 815, cy, 150, 64, ["Confronto", "→ score (o distanza)"], C_GT_BG, C_GT, 11.5)
for a, b in [(210, 270), (450, 505), (695, 740)]:
    arrow(o, a, cy, b, cy)
txt(o, 460, 205, "Cinque forme tipiche di template", 14, 700, "#111", halo=False)

CW, CH, Y0 = 172, 340, 225
cards = []
def card(i, title, sub, drawfn, cmp_lines):
    x0 = 15 + i*(CW+10)
    rect(o, x0, Y0, CW, CH, "#fff", "#d1d5db", 1.5, 12)
    txt(o, x0+CW/2, Y0+26, title, 13.5, 700, "#111", halo=False)
    txt(o, x0+CW/2, Y0+44, sub, 11, None, "#555", halo=False)
    drawfn(x0+14, Y0+62, CW-28, 130)
    o.append(f'<line x1="{x0+14}" y1="{Y0+212}" x2="{x0+CW-14}" y2="{Y0+212}" stroke="#e5e7eb"/>')
    txt(o, x0+CW/2, Y0+232, "Confronto", 11, 700, C_GT, halo=False)
    for k, s in enumerate(cmp_lines):
        txt(o, x0+CW/2, Y0+252+k*17, s, 11.5, 700 if k == 0 else None, "#222", halo=False)

def d_vec(x, y, w, h):
    vals = [0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.45]
    for k, v in enumerate(vals):
        bw = w/len(vals)
        o.append(f'<rect x="{x+k*bw+3:.1f}" y="{y+h-v*h:.1f}" width="{bw-6:.1f}" height="{v*h:.1f}" fill="{C_I}" fill-opacity="0.75"/>')
    txt(o, x+w/2, y+h+14, "[v1, v2, …, vn]", 11, None, "#555", halo=False)
def d_hist(x, y, w, h):
    a = [0.2, 0.5, 0.9, 0.7, 0.4, 0.2, 0.1, 0.05]
    b = [0.1, 0.3, 0.7, 0.9, 0.6, 0.3, 0.15, 0.1]
    bw = w/len(a)
    for k in range(len(a)):
        o.append(f'<rect x="{x+k*bw+2:.1f}" y="{y+h-a[k]*h:.1f}" width="{bw-4:.1f}" height="{a[k]*h:.1f}" fill="{C_I}" fill-opacity="0.55"/>')
        o.append(f'<rect x="{x+k*bw+2:.1f}" y="{y+h-b[k]*h:.1f}" width="{bw-4:.1f}" height="{b[k]*h:.1f}" fill="none" stroke="{C_FA}" stroke-width="1.8"/>')
    txt(o, x+w/2, y+h+14, "due istogrammi sovrapposti", 11, None, "#555", halo=False)
def d_ts(x, y, w, h):
    n = 40
    pa = [(x+w*i/(n-1), y+h*0.30+math.sin(i/n*6.28*1.5)*h*0.22) for i in range(n)]
    pb = [(x+w*i/(n-1), y+h*0.72+math.sin((i/n)**1.2*6.28*1.5)*h*0.22) for i in range(n)]
    polyline(o, pa, C_I, 2.2); polyline(o, pb, C_FA, 2.2)
    txt(o, x+w/2, y+h+14, "serie temporali (es. andatura)", 11, None, "#555", halo=False)
def d_pts(x, y, w, h):
    for _ in range(9):
        px, py, th = x+random.uniform(8, w-8), y+random.uniform(10, h-10), random.uniform(0, 6.28)
        o.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px+11*math.cos(th):.1f}" y2="{py+11*math.sin(th):.1f}" stroke="{C_FA}" stroke-width="2"/>')
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{C_I}"/>')
    txt(o, x+w/2, y+h+14, "minuzie {(x, y, θ)}", 11, None, "#555", halo=False)
def d_net(x, y, w, h):
    layers = [3, 4, 3, 2]
    pos = []
    for li, n_ in enumerate(layers):
        lx = x+10+li*(w-20)/3
        pos.append([(lx, y+10+(k+0.5)*(h-20)/n_) for k in range(n_)])
    for li in range(3):
        for p in pos[li]:
            for q in pos[li+1]:
                o.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="#d1d5db" stroke-width="1" {"stroke-dasharray=\"3,3\"" if li == 2 else ""}/>')
    for li, ps in enumerate(pos):
        for p in ps:
            col = "#9ca3af" if li == 3 else (C_G if li == 2 else C_I)
            o.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="6" fill="{col}"/>')
    lx = pos[3][0][0]
    o.append(f'<line x1="{lx-12:.1f}" y1="{y+4}" x2="{lx+12:.1f}" y2="{y+h-4}" stroke="{C_FA}" stroke-width="3"/>')
    o.append(f'<line x1="{lx+12:.1f}" y1="{y+4}" x2="{lx-12:.1f}" y2="{y+h-4}" stroke="{C_FA}" stroke-width="3"/>')
    txt(o, pos[2][1][0], y+h+14, "embedding", 11, 700, C_G_T, halo=False)
    
card(0, "Vettore di valori", "feature numeriche", d_vec, ["Distanza euclidea", "oppure", "cosine similarity"])
card(1, "Istogramma", "distribuzione di valori", d_hist, ["Correlazione (Pearson)", "oppure", "dist. di Bhattacharyya"])
card(2, "Serie temporale", "segnale nel tempo", d_ts, ["Dynamic Time Warping", "(DTW)", "allinea nel tempo"])
card(3, "Insieme di punti", "triplette (x, y, θ)", d_pts, ["Point-pattern matching", "cerca l'accoppiamento", "migliore, misura l'accordo"])
card(4, "Embedding deep", "uscita di una rete", d_net, ["Si toglie l'ultimo layer", "di classificazione e si", "confrontano gli embedding"])
txt(o, W/2, H-12, "Una volta ottenuta una similarità o una distanza, la si confronta con la soglia t: il ragionamento è simmetrico, cambia solo il verso della disuguaglianza.", 11.5, None, "#555", halo=False)
save(o, 'n03_sample_template_confronto.svg')
```
<img src="./img/n03_sample_template_confronto.svg" alt="Tipi di template e misure di confronto" style="display:block; margin:1.5em auto; max-width:100%;">


Il matching non avviene mai sui dati grezzi (il **sample**, cioè il dato acquisito dal sensore: un'immagine, una registrazione vocale, una scansione), ma sui **template** — la rappresentazione di feature estratta dal sample. La tecnica di confronto dipende dalla forma del template:

| Tipo di template | Confronto tipico |
|---|---|
| Vettore di valori | Distanza euclidea o cosine similarity |
| Istogramma | Correlazione (di Pearson), oppure distanza di Bhattacharyya |
| Serie temporale (es. accelerometro, andatura) | **Dynamic Time Warping (DTW)** |
| Insieme di punti/triplette (es. minuzie delle impronte $\{(x,y,\theta)\}$) | Point-pattern matching (si cerca l'accoppiamento migliore, poi si misura l'accordo) |
| Embedding di deep learning | Si rimuove l'ultimo layer di classificazione e si confrontano i vettori di embedding risultanti come normali vettori di feature |

**I due confronti meno intuitivi.** A sinistra il **DTW** allinea due serie di lunghezza e ritmo diversi (le linee grigie sono gli accoppiamenti scelti dal cammino ottimo, calcolato dal codice). A destra il **point-pattern matching**: si cerca l'allineamento migliore tra due insiemi di minuzie, poi si misura l'accordo.

```python
# =====================================================================
# N03b — DTW e point-pattern matching  (richiede 01, 05, 00)
# =====================================================================
random.seed(21)
W, H = 920, 560
o = new_svg(W, H)
header(o, W, "I due confronti meno intuitivi: DTW e point-pattern matching",
       "A sinistra serie temporali di lunghezza/ritmo diversi · a destra due insiemi di minuzie")
# ---------------- DTW ----------------
rect(o, 20, 78, 430, 450, "#fff", "#e5e7eb", 1.5, 14)
txt(o, 235, 104, "Dynamic Time Warping", 15, 700, "#111", halo=False)
na, nb = 18, 24
a = [math.sin(i/(na-1)*6.28*1.25)+0.4*math.sin(i/(na-1)*6.28*3) for i in range(na)]
b = [math.sin((i/(nb-1))**1.25*6.28*1.25)+0.4*math.sin((i/(nb-1))**1.25*6.28*3) for i in range(nb)]
INF = 1e9
D = [[INF]*(nb+1) for _ in range(na+1)]; D[0][0] = 0
for i in range(1, na+1):
    for j in range(1, nb+1):
        c = abs(a[i-1]-b[j-1])
        D[i][j] = c+min(D[i-1][j], D[i][j-1], D[i-1][j-1])
i, j, path = na, nb, []
while i > 0 and j > 0:
    path.append((i-1, j-1))
    k = min((D[i-1][j-1], 0), (D[i-1][j], 1), (D[i][j-1], 2))[1]
    if k == 0: i, j = i-1, j-1
    elif k == 1: i -= 1
    else: j -= 1
path.reverse()
XA, XB, WW = 50, 50, 370
ya = lambda v: 190-v*34
yb = lambda v: 400-v*34
pa = [(XA+WW*i/(na-1), ya(a[i])) for i in range(na)]
pb = [(XB+WW*j/(nb-1), yb(b[j])) for j in range(nb)]
for i, j in path:
    o.append(f'<line x1="{pa[i][0]:.1f}" y1="{pa[i][1]:.1f}" x2="{pb[j][0]:.1f}" y2="{pb[j][1]:.1f}" stroke="#9ca3af" stroke-width="1"/>')
polyline(o, pa, C_I, 2.6); polyline(o, pb, C_FA, 2.6)
for p in pa: o.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.5" fill="{C_I}"/>')
for p in pb: o.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.5" fill="{C_FA}"/>')
txt(o, 52, 126, f"serie A ({na} campioni)", 12, 700, C_I, "start", halo=False)
txt(o, 52, 450, f"serie B ({nb} campioni)", 12, 700, C_FA_T, "start", halo=False)
txt(o, 235, 478, f"distanza DTW = {D[na][nb]:.2f}  (somma lungo il cammino di allineamento)", 12, 700, "#111", halo=False)
txt(o, 235, 497, "le linee grigie sono gli accoppiamenti scelti: un punto può", 11.5, None, "#444", halo=False)
txt(o, 235, 513, "corrispondere a più punti, così si assorbono ritmi diversi.", 11.5, None, "#444", halo=False)
# ---------------- minuzie ----------------
rect(o, 470, 78, 430, 450, "#fff", "#e5e7eb", 1.5, 14)
txt(o, 685, 104, "Point-pattern matching", 15, 700, "#111", halo=False)
BW, BH = 170, 230
GX, GY, PX, PY = 495, 160, 705, 160
for x0, y0, lab in [(GX, GY, "template (gallery)"), (PX, PY, "probe")]:
    rect(o, x0, y0, BW, BH, "#f9fafb", "#9ca3af", 1.2, 8)
    txt(o, x0+BW/2, y0-10, lab, 12.5, 700, "#333", halo=False)
A = [(random.uniform(18, BW-18), random.uniform(18, BH-18), random.uniform(0, 6.28)) for _ in range(12)]
rot, tx, ty = 0.22, 8, -6
cxm, cym = BW/2, BH/2
def tf(p):
    x, y, th = p
    dx, dy = x-cxm, y-cym
    return (cxm+dx*math.cos(rot)-dy*math.sin(rot)+tx+random.gauss(0, 2), cym+dx*math.sin(rot)+dy*math.cos(rot)+ty+random.gauss(0, 2), th+rot)
drop = {3, 8}
B = {i: tf(p) for i, p in enumerate(A) if i not in drop}
spur = [(40, 40, 1.0), (BW-35, BH-30, 2.2)]
def dot(x0, y0, p, col):
    x, y, th = p
    o.append(f'<line x1="{x0+x:.1f}" y1="{y0+y:.1f}" x2="{x0+x+11*math.cos(th):.1f}" y2="{y0+y+11*math.sin(th):.1f}" stroke="{col}" stroke-width="2"/>')
    o.append(f'<circle cx="{x0+x:.1f}" cy="{y0+y:.1f}" r="4.2" fill="{col}"/>')
for i, q in B.items():
    o.append(f'<line x1="{GX+A[i][0]:.1f}" y1="{GY+A[i][1]:.1f}" x2="{PX+q[0]:.1f}" y2="{PY+q[1]:.1f}" stroke="{C_G}" stroke-width="1.3" stroke-opacity="0.85"/>')
for i, p in enumerate(A): dot(GX, GY, p, C_G_T if i not in drop else "#9ca3af")
for i, q in B.items(): dot(PX, PY, q, C_G_T)
for q in spur: dot(PX, PY, q, "#9ca3af")
txt(o, 685, 440, f"coppie accordate: {len(B)} su 12 minuzie del template", 12, 700, "#111", halo=False)
txt(o, 685, 458, "grigio = minuzie senza corrispondente (mancanti o spurie)", 11.5, None, "#444", halo=False)
txt(o, 685, 482, "1. si cerca l'allineamento (rotazione + traslazione) migliore", 11.5, None, "#444", halo=False)
txt(o, 685, 498, "2. si conta/pesa l'accordo → score di similarità", 11.5, None, "#444", halo=False)
save(o, 'n03b_dtw_e_minuzie.svg')
```
<img src="./img/n03b_dtw_e_minuzie.svg" alt="DTW e minuzie" style="display:block; margin:1.5em auto; max-width:100%;">

Una volta ottenuta una **similarità** (più alta = più simile) o una **distanza** (più bassa = più simile), il valore viene confrontato con una **soglia di accettazione**: tutto il ragionamento sulle soglie è perfettamente simmetrico che si lavori con similarità o con distanze — cambia solo il verso della disuguaglianza.

---

## 3. Verifica (Verification): i 4 possibili esiti

Nella verifica l'utente **dichiara un'identità**; il sistema confronta il campione con il template della gallery corrispondente a quell'identità.

| Esito | Nome | Significato |
|---|---|---|
| Identità vera, accettato | **GA** (Genuine Acceptance / Genuine Match) | Esito positivo |
| Identità vera, rifiutato | **FR** (False Rejection / False Non Match, errore tipo I) | Punteggio insufficiente |
| Identità falsa, rifiutato | **GR** (Genuine Reject / Genuine Non Match) | Esito positivo |
| Identità falsa, accettato | **FA** (False Acceptance / False Match, errore tipo II) | Il più critico in ambito security |

**La stessa tabella in forma di griglia.** Righe: la rivendicazione è vera o falsa; colonne: decisione del sistema. Sotto, come ciascun esito entra nei tassi.

```python
# =====================================================================
# N04 — Matrice 2×2 degli esiti della verifica  (richiede 01, 05)
# =====================================================================
W, H = 920, 600
o = new_svg(W, H)
header(o, W, "Verifica: i 4 possibili esiti", "Righe = la rivendicazione è vera o falsa (ground truth) · colonne = decisione del sistema")
XL, CW, YT, RH = 290, 285, 120, 135
cols = [("Accettato", "punteggio sopra la soglia"), ("Rifiutato", "punteggio sotto la soglia")]
for c, (a, b) in enumerate(cols):
    txt(o, XL+c*CW+CW/2, 96, a, 15, 700, "#111", halo=False)
    txt(o, XL+c*CW+CW/2, 112, b, 12, None, "#555", halo=False)
rows = [("Identità dichiarata VERA", "rivendicazione genuina", "TG = GA + FR"),
        ("Identità dichiarata FALSA", "rivendicazione impostore", "TI = FA + GR")]
for r, (a, b, c) in enumerate(rows):
    y = YT+r*RH
    txt(o, XL-14, y+RH/2-14, a, 14, 700, "#111", "end", halo=False)
    txt(o, XL-14, y+RH/2+4, b, 12, None, "#555", "end", halo=False)
    txt(o, XL-14, y+RH/2+24, c, 12.5, 700, C_GT, "end", halo=False)
cells = {(0, 0): ("GA", "Genuine Acceptance / Genuine Match", "esito positivo", C_OK_BG, C_G_T),
         (0, 1): ("FR", "False Rejection / False Non Match", "errore tipo I · utente bloccato", C_FR_BG, C_FR_T),
         (1, 0): ("FA", "False Acceptance / False Match", "errore tipo II · il più critico", C_FA_BG, C_FA_T),
         (1, 1): ("GR", "Genuine Reject / Genuine Non Match", "esito positivo", C_OK_BG, C_G_T)}
for (r, c), (nm, full, note, bg, fg) in cells.items():
    x, y = XL+c*CW+8, YT+r*RH+8
    rect(o, x, y, CW-16, RH-16, bg, fg, 2, 10)
    txt(o, x+(CW-16)/2, y+42, nm, 28, 700, fg, halo=False)
    txt(o, x+(CW-16)/2, y+66, full, 11.5, None, "#333", halo=False)
    txt(o, x+(CW-16)/2, y+84, note, 11.5, 700, fg, halo=False)
yb = YT+2*RH+30
txt(o, 40, yb, "Come entrano nei tassi (ognuno normalizzato sulla propria riga)", 13.5, 700, "#111", "start", halo=False)
items = [("GAR = GA / TG", C_G_T), ("FRR = FR / TG = 1 − GAR", C_FR_T), ("FAR = FA / TI", C_FA_T), ("GRR = GR / TI = 1 − FAR", C_G_T)]
for i, (s, col) in enumerate(items):
    txt(o, 40+(i % 2)*340, yb+28+(i//2)*24, s, 13, 700 if i in (1, 2) else None, col, "start", halo=False)
txt(o, 40, yb+86, "FA e FR sono in compromesso: alzando la soglia FAR scende e FRR sale.", 12, None, "#555", "start", halo=False)
save(o, 'n04_matrice_esiti_verifica.svg')
```
<img src="./img/n04_matrice_esiti_verifica.svg" alt="Esiti della verifica" style="display:block; margin:1.5em auto; max-width:100%;">


> **FA vs FR:** FA è generalmente il più critico in applicazioni di sicurezza (es. accesso di un impostore), FR crea problemi di usabilità (utenti legittimi bloccati). È un compromesso: non si possono minimizzare entrambi contemporaneamente.

---

## 4. Tassi di errore (Rates)

Le **rate** normalizzano il numero di errori rispetto alla popolazione corretta (non al numero totale di prove), per ottenere una misura indipendente dalla dimensione del probe set.

### 4.1 FAR e FRR — definizione probabilistica

Ipotesi:
- **H₀**: la persona è diversa da quella dichiarata
- **H₁**: la persona è la stessa dichiarata

Decisioni: **D₀** = persone diverse, **D₁** = stessa persona

$$
FAR = P(D_1 \mid H_0 = \text{vero})
$$
$$
FRR = P(D_0 \mid H_1 = \text{vero})
$$

- **FAR** = probabilità di accettare un impostore (calcolata sul numero di **rivendicazioni false**)
- **FRR** = probabilità di rifiutare un utente genuino (calcolata sul numero di **rivendicazioni genuine**)

> **I due scenari di rivendicazione falsa (impostor claim):** una rivendicazione impostore può nascere in due modi, calcolati in modo identico ai fini del FAR: (1) il probe appartiene a qualcuno **non iscritto** in gallery ($P_N$); (2) il probe appartiene a qualcun altro **già iscritto** in gallery, che sta mentendo sulla propria identità. Le rivendicazioni genuine, invece, provengono sempre e solo da $P_G$ (i probe dei soggetti effettivamente iscritti).

**Tre rivendicazioni.** Una genuina (da $P_G$) e due false: un non iscritto ($P_N$) e un iscritto che mente. Le due false finiscono nel denominatore del FAR.

```python
# =====================================================================
# N06 — Rivendicazioni genuine e impostore  (richiede 01, 05, 00)
# =====================================================================
W, H = 920, 600
o = new_svg(W, H)
header(o, W, "Le rivendicazioni nella verifica: genuine e i due tipi di impostore",
       "Il FAR si calcola su tutte le rivendicazioni false, qualunque sia la loro origine")
GXL, GW = 470, 170
rect(o, GXL-20, 100, GW+40, 440, "#f0fdf4", C_DB if False else "#0f766e", 1.8, 14, "6,4")
txt(o, GXL+GW/2, 124, "Gallery (iscritti)", 13.5, 700, "#0f766e", halo=False)
rows = [(190, "Luca", "Luca (iscritto)", C_G, "Luca", "genuina", C_OK_BG, C_G_T,
         ["Rivendicazione GENUINA", "p_j ∈ P_G, dichiara la propria identità", "→ denominatore del FRR (TG)"]),
        (330, "Mario", "Z (non iscritto)", "#6b7280", "Mario", "falsa", C_FA_BG, C_FA_T,
         ["Rivendicazione FALSA · scenario 1", "p_j ∈ P_N: persona non in gallery", "→ denominatore del FAR (TI)"]),
        (470, "Anna", "Luca (iscritto)", C_I, "Anna", "falsa", C_FA_BG, C_FA_T,
         ["Rivendicazione FALSA · scenario 2", "p_j ∈ P_G ma mente sull'identità", "→ denominatore del FAR (TI)"])]
for y, gid, plab, pcol, claim, kind, bg, fg, tag in rows:
    box(o, GXL+GW/2, y, GW, 56, [f"template di {gid}", "g_" + gid.lower()], "#fff", "#0f766e", 12.5)
    person(o, 80, y, pcol, 1.5)
    txt(o, 80, y+42, plab, 12, 700, "#222", halo=False)
    col = C_G if kind == "genuina" else C_FA
    arrow(o, 125, y, GXL-4, y, col, 2.2)
    txt(o, (125+GXL)/2, y-10, f"dichiara «{claim}»", 12.5, 700, fg, halo=True)
    rect(o, 690, y-46, 215, 92, bg, fg, 1.8, 10)
    for k, s in enumerate(tag):
        txt(o, 698, y-24+k*22, s, 11 if k else 12, 700 if k in (0, 2) else None, fg if k != 1 else "#333", "start", halo=False)
txt(o, W/2, 575, "In entrambi i casi di rivendicazione falsa il sistema deve rifiutare: se accetta, è un FA.", 12.5, None, "#444", halo=False)
save(o, 'n06_impostor_claim.svg')
```
<img src="./img/n06_impostor_claim.svg" alt="Rivendicazioni nella verifica" style="display:block; margin:1.5em auto; max-width:100%;">


> FAR e FRR sono complementari a GAR e GRR: **FRR + GAR = 1**, **FAR + GRR = 1**

### 4.2 Formule esplicite con soglia t

Sia `s(gₓ, pⱼ)` la similarità tra il template gallery gₓ e la probe pⱼ; `id()` restituisce l'identità vera.

**FRR** (calcolato solo sulle rivendicazioni genuine, insieme Pɢ):

$$
FRR(t) = \frac{|\{p_j : s_{xj} \le t,\ id(g_x) = id(p_j)\}|}{|\{p_j : id(g_x) = id(p_j)\}|}
$$

**FAR** (calcolato solo sulle rivendicazioni false):

$$
FAR(t) = \frac{|\{p_j : s_{xj} \ge t \ \wedge\ id(g_x) \ne id(p_j)\}|}{|\{p_j : id(g_x) \ne id(p_j)\}|}
$$

> ⚠️ **Attenzione classica:** FAR e FRR vanno sempre normalizzati rispetto al numero di tentativi genuini/impostori **reali**, non al numero totale di probe. Esempio: 100 probe, 10 FA e 10 FR.

**Ogni quadratino è una probe.** Normalizzando sul totale (100) si ottiene FAR = FRR = 0.10 e il problema scompare; normalizzando per categoria si vede che FRR = 10/90 ≈ 0.11 ma FAR = 10/10 = 1.

```python
# =====================================================================
# N05 — Esempio 100 probe  (richiede 01, 05)
# =====================================================================
W, H = 920, 560
o = new_svg(W, H)
header(o, W, "Perché normalizzare sulla categoria giusta: 100 probe, 10 FA e 10 FR",
       "Ogni quadratino è una probe · 90 rivendicazioni genuine, 10 da impostori")
GX, GY, CS = 80, 100, 30
for r in range(10):
    for c in range(10):
        if r < 8: fill, tag = C_G, "GA"
        elif r == 8: fill, tag = C_FR, "FR"
        else: fill, tag = C_FA, "FA"
        o.append(f'<rect x="{GX+c*CS+1}" y="{GY+r*CS+1}" width="{CS-2}" height="{CS-2}" rx="4" fill="{fill}" fill-opacity="{0.55 if r<8 else 0.9}"/>')
# parentesi sinistra
def brace(y1, y2, label, col):
    o.append(f'<path d="M {GX-10} {y1+2} L {GX-16} {y1+2} L {GX-16} {y2-2} L {GX-10} {y2-2}" fill="none" stroke="{col}" stroke-width="2"/>')
    o.append(f'<text transform="translate({GX-26},{(y1+y2)/2}) rotate(-90)" text-anchor="middle" font-size="12.5" font-weight="700" fill="{col}">{label}</text>')
brace(GY, GY+9*CS, "90 genuini (TG)", C_G_T)
brace(GY+9*CS, GY+10*CS, "10 imp. (TI)", C_FA_T)
xr = GX+10*CS+14
txt(o, xr, GY+4*CS, "80 GA", 13, 700, C_G_T, "start", halo=False)
txt(o, xr, GY+8*CS+CS/2+4, "10 FR", 13, 700, C_FR_T, "start", halo=False)
txt(o, xr, GY+9*CS+CS/2+4, "10 FA", 13, 700, C_FA_T, "start", halo=False)
txt(o, xr, GY+9*CS+CS/2+20, "(GR = 0)", 11.5, None, "#555", "start", halo=False)
# pannelli destri
PX, PW = 500, 390
def bar(y, label, v, col, val_txt):
    txt(o, PX, y-6, label, 12.5, 700, col, "start", halo=False)
    o.append(f'<rect x="{PX}" y="{y}" width="{PW-40}" height="18" fill="#f3f4f6" stroke="#d1d5db"/>')
    o.append(f'<rect x="{PX}" y="{y}" width="{(PW-40)*v:.1f}" height="18" fill="{col}" fill-opacity="0.85"/>')
    txt(o, PX+PW-34, y+14, val_txt, 12.5, 700, col, "start", halo=False)
rect(o, PX-16, 88, PW+32, 175, "#fff", "#fecaca", 2, 12)
txt(o, PX, 112, "Errato: dividere per il totale delle probe (100)", 13.5, 700, C_FA_T, "start", halo=False)
bar(150, "FRR = 10/100", 0.10, C_FR, "0.10")
bar(198, "FAR = 10/100", 0.10, C_FA, "0.10")
txt(o, PX, 244, "sembra un buon sistema: l'errore è nascosto", 12, None, "#555", "start", halo=False)
rect(o, PX-16, 285, PW+32, 205, "#fff", "#bbf7d0", 2, 12)
txt(o, PX, 309, "Corretto: ogni tasso sulla propria categoria", 13.5, 700, C_G_T, "start", halo=False)
bar(347, "FRR = FR/TG = 10/90", 10/90, C_FR, "0.11")
bar(395, "FAR = FA/TI = 10/10", 1.0, C_FA, "1.00")
txt(o, PX, 442, "tutti gli impostori sono stati accettati!", 12.5, 700, C_FA_T, "start", halo=False)
txt(o, PX, 460, "FAR e FRR separano genuini e impostori: nulla viene mascherato", 11.5, None, "#555", "start", halo=False)
for i, (c, t) in enumerate([(C_G, "GA genuino accettato"), (C_FR, "FR genuino rifiutato"), (C_FA, "FA impostore accettato")]):
    o.append(f'<rect x="80" y="{GY+10*CS+28+i*22}" width="16" height="16" rx="3" fill="{c}"/>')
    txt(o, 104, GY+10*CS+41+i*22, t, 12, None, "#222", "start", halo=False)
save(o, 'n05_esempio_100_probe.svg')
```
<img src="./img/n05_esempio_100_probe.svg" alt="Esempio 100 probe" style="display:block; margin:1.5em auto; max-width:100%;">


> - Se impostori reali = 10, genuini = 90 → **FRR = 10/90 ≈ 0.11**, **FAR = 10/10 = 1** (!) → tutti gli impostori sono stati accettati!
> - Calcolare erroneamente FAR = FRR = 10/100 nasconde completamente questo problema.

### 4.3 Score genuini e impostori

- **Score genuino (authentic):** confronto tra due campioni della **stessa** persona enrollata.
- **Score impostore:** confronto con il campione di una persona **non enrollata** (o con identità diversa dichiarata).

Le distribuzioni (tipicamente gaussiane) di score genuini e impostori si sovrappongono in parte: in quella zona di intersezione nascono gli errori (FA/FR), a seconda della soglia scelta.

### 4.4 Equal Error Rate (EER)
Punto in cui FAR(t) = FRR(t):

$$
EER = \{x : FRR(t) = x \ \wedge\ FAR(t) = x\}
$$

Non è una soglia, ma il **valore** di errore comune raggiunto a quella soglia.

### 4.5 Altri punti operativi
- **ZeroFAR** (Zero False Match Rate): valore di FRR quando FAR = 0.
- **ZeroFRR** (Zero False Non Match Rate): valore di FAR quando FRR = 0.
- Non è mai possibile avere realmente FAR = 0 o FRR = 0 esatti: sono punti concettuali di riferimento.

```python
# =====================================================================
# 01 — Distribuzioni degli score  (ESEGUIRE PER PRIMO)
# Definisce parametri, funzioni e helper SVG riusati dagli script 02-04.
# =====================================================================
import math, random
from statistics import NormalDist

OUT = '/mnt/user-data/outputs'

# --- parametri (distribuzioni gaussiane degli score) ---
MU_I, SD_I = 0.35, 0.12   # impostori
MU_G, SD_G = 0.65, 0.12   # genuini
T = 0.55                  # soglia di esempio

# --- funzioni condivise ---
pdf = lambda x, m, s: math.exp(-0.5*((x-m)/s)**2)/(s*math.sqrt(2*math.pi))
cdf = lambda x, m, s: 0.5*(1+math.erf((x-m)/(s*math.sqrt(2))))
far = lambda t: 0.5*math.erfc((t-MU_I)/(SD_I*math.sqrt(2)))   # = 1 - cdf, stabile nelle code
frr = lambda t: cdf(t, MU_G, SD_G)
gar = lambda t: 1-frr(t)

FAR, FRR = far(T), frr(T)
GRR, GAR = 1-FAR, 1-FRR
T_EER = (MU_I+MU_G)/2                         # intersezione delle curve (sigma uguali)
EER = far(T_EER)
T_1 = NormalDist(MU_I, SD_I).inv_cdf(0.99)    # soglia con FAR = 1%

# --- colori condivisi ---
C_I, C_G = "#2563eb", "#16a34a"       # impostori / genuini
C_FA, C_FR = "#dc2626", "#f59e0b"     # errori
C_FA_T, C_FR_T, C_G_T = "#b91c1c", "#b45309", "#15803d"   # varianti scure per il testo

# --- helper SVG condivisi ---
def new_svg(W, H):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

def txt(o, x, y, s, size=12, weight=None, fill="#222", anchor="middle", halo=True, extra=""):
    """Testo; con halo=True sotto viene disegnata una copia con contorno bianco
    che separa il testo da linee e curve (funziona in qualsiasi renderer SVG)."""
    w = f' font-weight="{weight}"' if weight else ""
    base = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"{w}{extra}'
    if halo:
        o.append(f'<text {base} fill="#fff" stroke="#fff" stroke-width="4" stroke-linejoin="round">{s}</text>')
    o.append(f'<text {base} fill="{fill}">{s}</text>')

def header(o, W, title, subtitle):
    txt(o, W/2, 32, title, 20, 700, "#111", halo=False)
    txt(o, W/2, 54, subtitle, 13, None, "#444", halo=False)

def frame(o, L, R, TOP, BOT, xlabel=None, ylabel=None):
    o.append(f'<line x1="{L}" y1="{BOT}" x2="{R}" y2="{BOT}" stroke="#222" stroke-width="1.5"/>')
    o.append(f'<line x1="{L}" y1="{TOP}" x2="{L}" y2="{BOT}" stroke="#222" stroke-width="1.5"/>')
    if xlabel: txt(o, (L+R)/2, BOT+44, xlabel, 13, None, "#222", halo=False)
    if ylabel: o.append(f'<text transform="translate(26,{(TOP+BOT)/2}) rotate(-90)" text-anchor="middle" font-size="13" fill="#222">{ylabel}</text>')

def leader(o, pts, color):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    o.append(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3.2" fill="{color}" stroke="#fff" stroke-width="1"/>')

def save(o, name):
    o.append('</svg>')
    open(f'{OUT}/{name}', 'w', encoding='utf-8').write("\n".join(o))

# --- verifica empirica con campione simulato ---
random.seed(42)
gen = [random.gauss(MU_G, SD_G) for _ in range(10000)]
imp = [random.gauss(MU_I, SD_I) for _ in range(10000)]
print(f"FAR teorico={FAR:.4f} empirico={sum(s>=T for s in imp)/len(imp):.4f} | "
      f"FRR teorico={FRR:.4f} empirico={sum(s<=T for s in gen)/len(gen):.4f}")

# --- punti operativi ---
T_ZFRR = min(gen) - 1e-6    # FRR = 0 sul campione
T_ZFAR = max(imp) + 1e-6    # FAR = 0 sul campione
OPS = [
    (1, "#7c3aed", "Zero FRR", T_ZFRR, "FRR = 0%", f"FAR = {far(T_ZFRR):.1%}"),
    (2, "#db2777", "1% FAR",   T_1,    "FAR = 1%", f"FRR = {frr(T_1):.1%}"),
    (3, "#0f766e", "Zero FAR", T_ZFAR, "FAR = 0%", f"FRR = {frr(T_ZFAR):.1%}"),
]
for n, c, name, t, a, b in OPS:
    print(n, name, round(t, 4), a, b)

# punti evidenziati anche nelle figure ROC e DET (script 03 e 04)
OP_PTS = [(f"t = {T}", T, C_I), (f"EER (t = {T_EER:.2f})", T_EER, "#111111"), ("FAR = 1%", T_1, "#db2777")]

# =====================================================================
# Grafico 1
# =====================================================================
W, H = 920, 710
L, R, TOP, BOT = 70, 880, 105, 455
XMIN, XMAX, YMAX = -0.1, 1.1, 4.4
sx = lambda x: L + (x-XMIN)/(XMAX-XMIN)*(R-L)
sy = lambda y: BOT - y/YMAX*(BOT-TOP)

def curve(m, s, a, b, n=200):
    return [(sx(a+(b-a)*i/n), sy(pdf(a+(b-a)*i/n, m, s))) for i in range(n+1)]
def area(m, s, a, b):
    pts = curve(m, s, a, b)
    return f"M {sx(a):.1f},{BOT} " + " ".join(f"L {x:.1f},{y:.1f}" for x, y in pts) + f" L {sx(b):.1f},{BOT} Z"
def line(m, s):
    return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in curve(m, s, XMIN, XMAX))

o = new_svg(W, H)
header(o, W, "Distribuzioni degli score: genuini vs impostori",
       f"Genuini ~ N({MU_G}, {SD_G}²) · Impostori ~ N({MU_I}, {SD_I}²) · soglia t = {T}")

# aree e curve
o.append(f'<path d="{area(MU_I,SD_I,XMIN,T)}" fill="{C_I}" fill-opacity="0.18"/>')
o.append(f'<path d="{area(MU_G,SD_G,T,XMAX)}" fill="{C_G}" fill-opacity="0.18"/>')
o.append(f'<path d="{area(MU_I,SD_I,T,XMAX)}" fill="{C_FA}" fill-opacity="0.75"/>')
o.append(f'<path d="{area(MU_G,SD_G,XMIN,T)}" fill="{C_FR}" fill-opacity="0.75"/>')
o.append(f'<path d="{line(MU_I,SD_I)}" fill="none" stroke="{C_I}" stroke-width="2.5"/>')
o.append(f'<path d="{line(MU_G,SD_G)}" fill="none" stroke="{C_G}" stroke-width="2.5"/>')

# assi
frame(o, L, R, TOP, BOT, "Score di similarità s", "Densità di probabilità")
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{sx(v):.1f}" y1="{BOT}" x2="{sx(v):.1f}" y2="{BOT+5}" stroke="#222"/>')
    txt(o, sx(v), BOT+20, f"{v:.1f}", 12, None, "#333", halo=False)

# soglia
o.append(f'<line x1="{sx(T):.1f}" y1="{TOP+6}" x2="{sx(T):.1f}" y2="{BOT}" stroke="#111" stroke-width="2" stroke-dasharray="7,5"/>')
txt(o, sx(T), TOP-24, f"soglia t = {T}", 14, 700, "#111", halo=False)
txt(o, sx(T)-8, TOP-4, "← rifiuto (D0)", 12, None, "#111", "end", halo=False)
txt(o, sx(T)+8, TOP-4, "accetto (D1) →", 12, None, "#111", "start", halo=False)

# punti operativi
for n, c, name, t, a, b in OPS:
    o.append(f'<line x1="{sx(t):.1f}" y1="{TOP+46}" x2="{sx(t):.1f}" y2="{BOT}" stroke="{c}" stroke-width="2" stroke-dasharray="2,4" stroke-linecap="round"/>')
for n, c, name, t, a, b in OPS:
    o.append(f'<circle cx="{sx(t):.1f}" cy="{TOP+34}" r="10" fill="{c}"/>')
    txt(o, sx(t), TOP+39, str(n), 13, 700, "#fff", halo=False)

# nomi delle distribuzioni
txt(o, sx(MU_I), sy(pdf(MU_I,MU_I,SD_I))-12, "Impostori", 15, 700, C_I)
txt(o, sx(MU_G)+36, sy(pdf(MU_G,MU_G,SD_G))-12, "Genuini", 15, 700, C_G)

# intersezione (tag breve, spiegazione in legenda)
yx = pdf(T_EER, MU_I, SD_I)
o.append(f'<circle cx="{sx(T_EER):.1f}" cy="{sy(yx):.1f}" r="5" fill="#fff" stroke="#111" stroke-width="2"/>')
txt(o, sx(T_EER), sy(yx)-12, "EER", 12, 700, "#111")

# metriche: GRR/GAR dentro le aree grandi; FRR/FAR (aree sottili) con richiamo lungo l'asse
Y_LEAD = BOT-15
xg = sx(0.30)
txt(o, xg, sy(0.95), f"GRR = {GRR:.1%}", 14, 700, C_I)
txt(o, xg, sy(0.95)+15, "impostori", 11.5, None, "#333")
txt(o, xg, sy(0.95)+28, "correttamente rifiutati", 11.5, None, "#333")
xr = (sx(T_1)+sx(T_ZFAR))/2
txt(o, xr, sy(0.80), f"GAR = {GAR:.1%}", 14, 700, C_G_T)
for i, s in enumerate(["genuini", "correttamente", "accettati"]):
    txt(o, xr, sy(0.80)+15+13*i, s, 11.5, None, "#333")
yl = sy(1.45)
txt(o, 88, yl, f"FRR = {FRR:.1%}", 14, 700, C_FR_T, "start")
txt(o, 88, yl+15, "genuini rifiutati", 11.5, None, "#333", "start")
txt(o, 88, yl+28, "(s ≤ t)", 11.5, None, "#333", "start")
leader(o, [(100, yl+36), (100, Y_LEAD), (sx(0.50), Y_LEAD)], C_FR_T)
txt(o, 868, yl, f"FAR = {FAR:.1%}", 14, 700, C_FA_T, "end")
txt(o, 868, yl+15, "impostori accettati", 11.5, None, "#333", "end")
txt(o, 868, yl+28, "(s ≥ t)", 11.5, None, "#333", "end")
leader(o, [(850, yl+36), (850, Y_LEAD), (sx(0.575), Y_LEAD)], C_FA_T)

# legenda
ly = BOT+68
x0 = L
for c, t, op in [(C_FA,"FA – impostore accettato",0.75),(C_FR,"FR – genuino rifiutato",0.75),
                 (C_I,"GRR – impostore rifiutato",0.18),(C_G,"GAR – genuino accettato",0.18)]:
    o.append(f'<rect x="{x0}" y="{ly-11}" width="16" height="16" fill="{c}" fill-opacity="{op}" stroke="{c}"/>')
    txt(o, x0+22, ly+2, t, 12.5, None, "#222", "start", halo=False)
    x0 += 205
o.append(f'<circle cx="{L+8}" cy="{ly+28}" r="5" fill="#fff" stroke="#111" stroke-width="2"/>')
txt(o, L+22, ly+32, f"EER – intersezione delle curve (s = {T_EER:.2f})", 12.5, None, "#222", "start", halo=False)
txt(o, L, ly+56, f"FRR + GAR = 1 ({FRR:.3f} + {GAR:.3f}) · FAR + GRR = 1 ({FAR:.3f} + {GRR:.3f})", 12, None, "#555", "start", halo=False)
ly2 = ly+86
txt(o, L, ly2, f"Punti operativi (soglie alternative a t = {T})", 13, 700, "#111", "start", halo=False)
for i, (n, c, name, t, a, b) in enumerate(OPS):
    yy = ly2 + 24 + i*24
    o.append(f'<circle cx="{L+9}" cy="{yy-4}" r="9" fill="{c}"/>')
    txt(o, L+9, yy, str(n), 12, 700, "#fff", halo=False)
    o.append(f'<text x="{L+28}" y="{yy}" font-size="12.5" fill="#222"><tspan font-weight="700">{name}</tspan>: t = {t:.3f} → {a}, {b}</text>')
txt(o, L, ly2+24+3*24+4, "Zero FRR / Zero FAR: soglie ricavate dal campione simulato (min dei genuini, max degli impostori); 1% FAR: percentile 99 degli impostori.", 11.5, None, "#666", "start", halo=False)
save(o, 'far_frr_distribuzioni.svg')
```

<img src="./img/far_frr_distribuzioni.svg" width="100%" style="align: center;"/>

---

## 5. Curve di prestazione (Verifica)

### 5.1 Curva FAR/FRR vs soglia
FAR e FRR hanno **andamento opposto** rispetto alla soglia t: aumentando t (più selettivo), FAR diminuisce e FRR aumenta.

```python
# =====================================================================
# 02 — FAR/FRR vs soglia + margin  (richiede 01_distribuzioni.py)
# Usa: far, frr, T, T_EER, EER, C_FA, C_FR, helper SVG
# =====================================================================
margin = lambda t: abs(far(t)-frr(t))
C_M = "#7c3aed"

W, H = 920, 730
L, R = 80, 880
A_TOP, A_BOT = 95, 400      # pannello A: FAR/FRR
B_TOP, B_BOT = 490, 660     # pannello B: margin
sx = lambda t: L + t*(R-L)
syA = lambda y: A_BOT - y*(A_BOT-A_TOP)
syB = lambda y: B_BOT - y*(B_BOT-B_TOP)
ts = [i/400 for i in range(401)]
path = lambda f, sy: "M " + " L ".join(f"{sx(t):.1f},{sy(f(t)):.1f}" for t in ts)

def grid(o, top, bot):
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        y = bot - v*(bot-top)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        txt(o, L-8, y+4, f"{v:.0%}", 12, None, "#333", "end", halo=False)
    for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        o.append(f'<line x1="{sx(v):.1f}" y1="{bot}" x2="{sx(v):.1f}" y2="{bot+5}" stroke="#222"/>')
        txt(o, sx(v), bot+20, f"{v:.1f}", 12, None, "#333", halo=False)

o = new_svg(W, H)
header(o, W, "FAR e FRR in funzione della soglia t",
       f"Genuini ~ N({MU_G}, {SD_G}²) · Impostori ~ N({MU_I}, {SD_I}²)")

# ---------- pannello A ----------
txt(o, L, A_TOP-14, "A · Tassi di errore", 14, 700, "#111", "start", halo=False)
grid(o, A_TOP, A_BOT)
frame(o, L, R, A_TOP, A_BOT, ylabel="Tasso di errore")
o.append(f'<path d="{path(far, syA)}" fill="none" stroke="{C_FA}" stroke-width="3"/>')
o.append(f'<path d="{path(frr, syA)}" fill="none" stroke="{C_FR}" stroke-width="3"/>')
txt(o, sx(0.30)+8, syA(far(0.30))-9, "FAR(t)", 15, 700, C_FA_T, "start")
txt(o, sx(0.70)-8, syA(frr(0.70))-9, "FRR(t)", 15, 700, C_FR_T, "end")

# soglia di esempio (linea su entrambi i pannelli)
o.append(f'<line x1="{sx(T):.1f}" y1="{A_TOP}" x2="{sx(T):.1f}" y2="{B_BOT}" stroke="#111" stroke-width="1.5" stroke-dasharray="7,5"/>')
txt(o, sx(T)+6, A_TOP+14, f"t = {T}", 12, 700, "#111", "start")
for f, c in ((far, C_FA), (frr, C_FR)):
    o.append(f'<circle cx="{sx(T):.1f}" cy="{syA(f(T)):.1f}" r="5" fill="{c}" stroke="#fff" stroke-width="1.5"/>')
txt(o, sx(T)+12, syA(frr(T))+17, f"FRR = {frr(T):.1%}", 12, 700, C_FR_T, "start")
txt(o, sx(T)+12, syA(far(T))-9, f"FAR = {far(T):.1%}", 12, 700, C_FA_T, "start")

# EER: etichetta nello spazio libero tra le due curve, sopra il punto
yE = syA(EER)
o.append(f'<circle cx="{sx(T_EER):.1f}" cy="{yE:.1f}" r="6" fill="#fff" stroke="#111" stroke-width="2.5"/>')
txt(o, sx(T_EER)-10, yE-62, f"EER = {EER:.1%}", 12.5, 700, "#111")
txt(o, sx(T_EER)-10, yE-48, f"t = {T_EER:.2f}", 11.5, None, "#333")

txt(o, R, A_BOT-24, "t ↑ → più selettivo: FAR ↓, FRR ↑", 12, None, "#555", "end")

# ---------- pannello B ----------
txt(o, L, B_TOP-14, "B · Margin(t) = |FAR(t) − FRR(t)|", 14, 700, "#111", "start", halo=False)
grid(o, B_TOP, B_BOT)
frame(o, L, R, B_TOP, B_BOT, xlabel="Soglia t", ylabel="Margin")
o.append(f'<path d="{path(margin, syB)}" fill="none" stroke="{C_M}" stroke-width="3"/>')
o.append(f'<circle cx="{sx(T_EER):.1f}" cy="{syB(0):.1f}" r="6" fill="#fff" stroke="#111" stroke-width="2.5"/>')
txt(o, sx(T_EER)-20, syB(0)-62, "margin = 0", 12, 700, "#111")
txt(o, sx(T_EER)-20, syB(0)-48, "(soglia EER)", 11.5, None, "#333")
o.append(f'<circle cx="{sx(T):.1f}" cy="{syB(margin(T)):.1f}" r="5" fill="{C_M}" stroke="#fff" stroke-width="1.5"/>')
txt(o, sx(T)+12, syB(margin(T))+17, f"margin({T}) = {margin(T):.1%}", 12, 700, C_M, "start")
save(o, 'far_frr_vs_soglia.svg')
print(f"EER = {EER:.4f} a t = {T_EER}; margin({T}) = {margin(T):.4f}")
```

<img src="./img/far_frr_vs_soglia.svg" width="100%" style="align: center;"/>

### 5.2 ROC (Receiver Operating Characteristic)
Asse x = FAR, asse y = **1 − FRR** (= GAR). Più la curva è vicina all'angolo in alto a sinistra, migliore è il sistema. Poiché confrontare due curve visivamente può essere ambiguo, si usa una metrica sintetica: l'**AUC (Area Under the Curve)** — l'area sotto la curva ROC. Un'AUC vicina a 1 indica prestazioni eccellenti; un'AUC di 0.5 indica un sistema che si comporta come una scelta casuale (non discriminante).

```python
# =====================================================================
# 03 — Curva ROC con AUC  (richiede 01_distribuzioni.py)
# Usa: far, gar, MU_*, SD_*, OP_PTS, helper SVG
# =====================================================================
ts = [1.3 - 1.8*i/1500 for i in range(1501)]          # t decrescente -> FAR crescente
roc = [(far(t), gar(t)) for t in ts]
auc = sum((roc[i+1][0]-roc[i][0])*(roc[i+1][1]+roc[i][1])/2 for i in range(len(roc)-1))
auc_th = NormalDist().cdf(((MU_G-MU_I)/SD_I)/math.sqrt(2))   # AUC teorica (sigma uguali)
print(f"AUC numerica = {auc:.4f}, teorica = {auc_th:.4f}")

W, H = 780, 800
L, R, TOP, BOT = 95, 715, 85, 705
sx = lambda x: L + x*(R-L)
sy = lambda y: BOT - y*(BOT-TOP)
C_ROC = "#2563eb"

o = new_svg(W, H)
header(o, W, "Curva ROC (Receiver Operating Characteristic)",
       "x = FAR · y = GAR = 1 − FRR · ogni punto corrisponde a una soglia t")

for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#e5e7eb"/>')
    o.append(f'<line x1="{sx(v):.1f}" y1="{TOP}" x2="{sx(v):.1f}" y2="{BOT}" stroke="#e5e7eb"/>')
    txt(o, L-8, sy(v)+4, f"{v:.1f}", 12, None, "#333", "end", halo=False)
    txt(o, sx(v), BOT+20, f"{v:.1f}", 12, None, "#333", halo=False)
frame(o, L, R, TOP, BOT, "FAR (False Acceptance Rate)", "GAR = 1 − FRR")

# area AUC, diagonale (sistema casuale), curva
area = f"M {sx(0):.1f},{BOT} " + " ".join(f"L {sx(x):.1f},{sy(y):.1f}" for x, y in roc) + f" L {sx(1):.1f},{BOT} Z"
o.append(f'<path d="{area}" fill="{C_ROC}" fill-opacity="0.12"/>')
o.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="#6b7280" stroke-width="1.8" stroke-dasharray="6,5"/>')
rx, ry = sx(0.60), sy(0.60)+20
txt(o, rx, ry, "sistema casuale (AUC = 0.5)", 12, None, "#6b7280", "start", extra=f' transform="rotate(-45 {rx:.1f} {ry:.1f})"')
o.append(f'<path d="M {" L ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in roc)}" fill="none" stroke="{C_ROC}" stroke-width="3.2"/>')

txt(o, L+14, TOP+22, "↖ migliore", 13, 700, C_G_T, "start")
txt(o, sx(0.55), sy(0.30), f"AUC = {auc:.3f}", 18, 700, "#1d4ed8")
txt(o, sx(0.55), sy(0.30)+18, "area sotto la curva", 12, None, "#333")

# punti operativi: etichette sotto-destra del punto (dentro l'area, lontano dalla curva)
for name, t, c in OP_PTS:
    x, y = far(t), gar(t)
    o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="6.5" fill="{c}" stroke="#fff" stroke-width="2"/>')
    txt(o, sx(x)+14, sy(y)+22, name, 12.5, 700, c, "start")
    txt(o, sx(x)+14, sy(y)+36, f"FAR = {x:.1%}, GAR = {y:.1%}", 11.5, None, "#333", "start")
save(o, 'roc.svg')
```

<img src="./img/roc.svg" width="100%" style="align: center;"/>

### 5.3 DET (Detection Error Tradeoff)
Asse x = FAR, asse y = FRR (scala logaritmica). Qui **più bassa è la curva, migliore è il sistema** (interpretazione opposta rispetto a ROC).

```python
# =====================================================================
# 04 — Curva DET (scala log)  (richiede 01_distribuzioni.py)
# Usa: far, frr, OP_PTS, helper SVG
# =====================================================================
XMIN, XMAX = 1e-3, 1.0
YMIN, YMAX = 1e-3, 1.0

det = [(far(-0.2+1.4*i/3000), frr(-0.2+1.4*i/3000)) for i in range(3001)]
det = [(x, y) for x, y in det if XMIN <= x <= XMAX and YMIN <= y <= YMAX]

W, H = 780, 800
L, R, TOP, BOT = 95, 715, 85, 705
lg = lambda v, a, b: (math.log10(v)-math.log10(a))/(math.log10(b)-math.log10(a))
sx = lambda x: L + lg(x, XMIN, XMAX)*(R-L)
sy = lambda y: BOT - lg(y, YMIN, YMAX)*(BOT-TOP)

o = new_svg(W, H)
header(o, W, "Curva DET (Detection Error Tradeoff)",
       "x = FAR · y = FRR · scale logaritmiche · più bassa è la curva, migliore è il sistema")

# griglia log
labels = {1e-3: "0.1%", 1e-2: "1%", 1e-1: "10%", 1.0: "100%"}
for e in (-3, -2, -1):
    for k in range(2, 10):
        v = k*10**e
        o.append(f'<line x1="{sx(v):.1f}" y1="{TOP}" x2="{sx(v):.1f}" y2="{BOT}" stroke="#f1f5f9"/>')
        o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#f1f5f9"/>')
for v, lab in labels.items():
    o.append(f'<line x1="{sx(v):.1f}" y1="{TOP}" x2="{sx(v):.1f}" y2="{BOT}" stroke="#d1d5db"/>')
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#d1d5db"/>')
    txt(o, sx(v), BOT+20, lab, 12, None, "#333", halo=False)
    txt(o, L-8, sy(v)+4, lab, 12, None, "#333", "end", halo=False)
frame(o, L, R, TOP, BOT, "FAR (scala log)", "FRR (scala log)")

# diagonale FAR = FRR e curva
o.append(f'<line x1="{sx(XMIN)}" y1="{sy(YMIN)}" x2="{sx(XMAX)}" y2="{sy(YMAX)}" stroke="#6b7280" stroke-width="1.8" stroke-dasharray="6,5"/>')
dx, dy = sx(0.012), sy(0.012)-8
txt(o, dx, dy, "FAR = FRR", 12, None, "#6b7280", "start", extra=f' transform="rotate(-45 {dx:.1f} {dy:.1f})"')
o.append(f'<path d="M {" L ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in det)}" fill="none" stroke="#2563eb" stroke-width="3.2"/>')
txt(o, L+90, BOT-14, "↙ migliore", 13, 700, C_G_T, "start")

# punti operativi: etichette negli spazi liberi (sotto-sinistra della curva; EER nel cuneo a destra)
place = [("end", -14, 22), ("start", 35, -6), ("end", -14, 22)]
for (name, t, c), (anc, ddx, ddy) in zip(OP_PTS, place):
    x, y = far(t), frr(t)
    o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="6.5" fill="{c}" stroke="#fff" stroke-width="2"/>')
    txt(o, sx(x)+ddx, sy(y)+ddy, name, 12.5, 700, c, anc)
    txt(o, sx(x)+ddx, sy(y)+ddy+14, f"FAR = {x:.1%}, FRR = {y:.1%}", 11.5, None, "#333", anc)
save(o, 'det.svg')
```

<img src="./img/det.svg" width="100%" style="align: center;"/>

---

## 6. Identificazione Open Set (Watchlist)

A differenza della verifica, **non c'è rivendicazione di identità**: il probe viene confrontato con **tutta** la gallery (1-a-N), e il sistema deve decidere da solo due cose: *se* il soggetto è noto e, in caso affermativo, *chi* è.

### 6.0 Notazione

| Simbolo | Significato |
|---|---|
| $G = \{g_1, \dots, g_N\}$ | Gallery: template delle $N$ identità registrate |
| $id(g_i)$ | Identità associata al template $g_i$ |
| $p_j$ | Probe (campione da identificare) |
| $s_{ij} = sim(p_j, g_i)$ | Punteggio di similarità tra probe $j$ e template $i$ (più alto = più simile) |
| $P_G$ | Insieme dei probe di persone **presenti** in gallery (*genuine*, "noti") |
| $P_N$ | Insieme dei probe di persone **non presenti** in gallery (*impostori*, "ignoti") |
| $t$ | Soglia di accettazione |
| $rank(p_j)$ | Posizione, nella lista dei template ordinata per punteggio decrescente, del template della vera identità di $p_j$ |

### 6.1 Procedura del sistema

Dato un probe $p_j$:

1. **Confronto 1:N**: si calcolano tutti i punteggi $s_{1j}, \dots, s_{Nj}$.
2. **Ordinamento**: si ordinano i template per punteggio decrescente. Il template al rank 1 è $g_{i^*}$ con $i^* = \arg\max_i s_{ij}$.
3. **Test di soglia**: si controlla se $s_{i^*j} \ge t$.
4. **Decisione**:
   - se $s_{i^*j} \ge t$ → il sistema accetta e restituisce l'identità $id(g_{i^*})$;
   - se $s_{i^*j} < t$ → il sistema rifiuta ("soggetto non in watchlist").

Il sistema vede **solo** questo. Se la persona fosse davvero in gallery, o se $id(g_{i^*})$ fosse davvero la sua identità, lo sa solo chi valuta il sistema (ground truth). Gli esiti si definiscono confrontando la decisione con la ground truth.

**Il flusso decisionale in una figura.** Il sistema esegue sempre gli stessi quattro passi: confronto 1:N, ordinamento, test di soglia su $s^*$, decisione. Tutto ciò che sta sotto il rombo (riquadri viola tratteggiati) il sistema **non lo vede**: è la ground truth, usata da chi valuta per etichettare ogni decisione come A, B, C, D o E. Sono due domande indipendenti — *la persona è in gallery?* e *il rank 1 è la sua identità?* — e per questo i casi corretti sono due (A e D) e gli errori tre (B, C, E).

> Questo primo blocco definisce anche gli helper (`rect`, `box`, `arrow`, `diamond`, `mat_cell`, `g_line`, `g_area`) usati da tutti i blocchi successivi.

```python
# =====================================================================
# 05 — Helper per diagrammi + flusso decisionale open set  (richiede 01)
# Definisce rect/box/arrow/diamond/mat_cell/g_line/g_area riusati da 06-22.
# Produce: 01_flusso_decisionale.svg
# =====================================================================
C_OK_BG, C_FR_BG, C_FA_BG = "#dcfce7", "#fef3c7", "#fee2e2"   # sfondi esito: corretto / falso rifiuto / falso allarme
C_GT, C_GT_BG = "#7c3aed", "#f5f3ff"                           # ground truth (la conosce solo chi valuta)
BLU_BG, GREY_BG = "#eff6ff", "#f3f4f6"

def rect(o, x, y, w, h, fill="#fff", stroke="#222", sw=1.5, rx=8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def box(o, cx, cy, w, h, lines, fill="#fff", stroke="#222", size=12.5, color="#222", dash=None, sw=1.5):
    """Riquadro centrato; la prima riga è in grassetto."""
    rect(o, cx-w/2, cy-h/2, w, h, fill, stroke, sw, dash=dash)
    lh = size + 3.5
    y0 = cy - (len(lines)-1)*lh/2 + size*0.35
    for i, s in enumerate(lines):
        txt(o, cx, y0+i*lh, s, size, 700 if i == 0 else None, color, halo=False)

def arrow(o, x1, y1, x2, y2, color="#222", sw=1.8, head=9, dash=None):
    ang = math.atan2(y2-y1, x2-x1)
    bx, by = x2-head*math.cos(ang), y2-head*math.sin(ang)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>')
    px, py = -math.sin(ang)*head*0.45, math.cos(ang)*head*0.45
    o.append(f'<polygon points="{x2:.1f},{y2:.1f} {bx+px:.1f},{by+py:.1f} {bx-px:.1f},{by-py:.1f}" fill="{color}"/>')

def diamond(o, cx, cy, w, h, lines, fill="#fff", stroke="#222", size=13):
    o.append(f'<polygon points="{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    lh = size + 3.5
    y0 = cy - (len(lines)-1)*lh/2 + size*0.35
    for i, s in enumerate(lines):
        txt(o, cx, y0+i*lh, s, size, 700, "#111", halo=False)

def mat_cell(o, x, y, w, h, fill, text=None, size=12, tcolor="#222", bold=False, stroke="#9ca3af", sw=1):
    o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    if text is not None:
        txt(o, x+w/2, y+h/2+size*0.36, text, size, 700 if bold else None, tcolor, halo=False)

def g_curve(m, s, a, b, sx, sy, n=160):
    return [(sx(a+(b-a)*i/n), sy(pdf(a+(b-a)*i/n, m, s))) for i in range(n+1)]

def g_line(m, s, a, b, sx, sy):
    return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in g_curve(m, s, a, b, sx, sy))

def g_area(m, s, a, b, sx, sy, base):
    pts = g_curve(m, s, a, b, sx, sy)
    return f"M {sx(a):.1f},{base} " + " ".join(f"L {x:.1f},{y:.1f}" for x, y in pts) + f" L {sx(b):.1f},{base} Z"

# ------------------------- figura -------------------------
W, H = 920, 770
o = new_svg(W, H)
header(o, W, "Identificazione open set: flusso decisionale",
       "Il sistema vede solo la soglia · gli esiti si definiscono con la ground truth (riquadri viola tratteggiati)")
cx = 460

# --- parte "sistema" (bordo pieno) ---
box(o, cx, 92, 300, 40, ["Probe p_j"], BLU_BG, C_I)
box(o, cx, 155, 380, 46, ["Confronto 1:N", "s_1j, ..., s_Nj con tutti i template della gallery"], BLU_BG, C_I)
box(o, cx, 222, 380, 46, ["Ordinamento decrescente", "rank 1 = argmax s_ij  ·  s* = max s_ij"], BLU_BG, C_I)
diamond(o, cx, 322, 230, 88, ["s* ≥ t ?"], "#fff", "#111", 15)
arrow(o, cx, 112, cx, 132); arrow(o, cx, 178, cx, 199); arrow(o, cx, 245, cx, 278)

box(o, 190, 322, 240, 52, ["Il sistema RIFIUTA", "«soggetto non in watchlist»"], GREY_BG, "#111")
box(o, 730, 322, 240, 52, ["Il sistema ACCETTA", "restituisce id(g_rank1)"], GREY_BG, "#111")
arrow(o, cx-115, 322, 312, 322); txt(o, 350, 312, "no", 12, 700, "#111")
arrow(o, cx+115, 322, 608, 322); txt(o, 570, 312, "sì", 12, 700, "#111")

# --- parte "ground truth" (bordo viola tratteggiato) ---
gt = dict(fill=C_GT_BG, stroke=C_GT, dash="6,4")
box(o, 190, 435, 250, 46, ["Ground truth", "la persona è in gallery?"], **gt)
box(o, 700, 435, 250, 46, ["Ground truth", "la persona è in gallery?"], **gt)
arrow(o, 190, 348, 190, 412, "#555"); arrow(o, 730, 348, 700, 412, "#555")

# rifiuto
box(o, 95, 545, 170, 66, ["D · Genuine Reject", "persona non in gallery", "esito corretto"], C_OK_BG, C_G_T, 12)
box(o, 285, 545, 170, 66, ["C · False Rejection", "persona in gallery", "noto non riconosciuto"], C_FR_BG, C_FR_T, 12)
arrow(o, 160, 458, 100, 511, C_GT); txt(o, 112, 486, "no", 12, 700, C_GT)
arrow(o, 220, 458, 280, 511, C_GT); txt(o, 268, 486, "sì", 12, 700, C_GT)

# accettazione
box(o, 835, 545, 150, 66, ["E · False Alarm", "persona non in gallery", "impostore accettato"], C_FA_BG, C_FA_T, 12)
box(o, 610, 545, 210, 50, ["Ground truth", "rank 1 = identità vera?"], **gt)
arrow(o, 745, 458, 820, 511, C_GT); txt(o, 806, 486, "no", 12, 700, C_GT)
arrow(o, 640, 458, 625, 520, C_GT); txt(o, 616, 488, "sì", 12, 700, C_GT)
box(o, 520, 665, 190, 66, ["A · Correct Detect", "and Identify", "rank 1"], C_OK_BG, C_G_T, 12)
box(o, 730, 665, 200, 66, ["B · False Rejection", "(misidentification)", "rank 1 sbagliato"], C_FR_BG, C_FR_T, 12)
arrow(o, 580, 570, 535, 632, C_GT); txt(o, 541, 603, "sì", 12, 700, C_GT)
arrow(o, 650, 570, 715, 632, C_GT); txt(o, 698, 603, "no", 12, 700, C_GT)

# --- che cosa misurano ---
txt(o, 40, 640, "Come entrano nelle metriche", 13, 700, "#111", "start", halo=False)
for i, s in enumerate(["A → DIR(t,1)", "B + C → FNIR(t) = 1 − DIR(t,1)", "E → FPIR(t)", "D → 1 − FPIR(t)"]):
    txt(o, 40, 662+i*20, s, 12.5, None, "#333", "start", halo=False)

# --- legenda ---
rect(o, 40, 745, 22, 14, "#fff", "#111", 1.5, 3)
txt(o, 70, 757, "decisione del sistema", 12, None, "#333", "start", halo=False)
rect(o, 230, 745, 22, 14, C_GT_BG, C_GT, 1.5, 3, "4,3")
txt(o, 260, 757, "ground truth (nota solo a chi valuta)", 12, None, "#333", "start", halo=False)
save(o, '01_flusso_decisionale.svg')
```

<img src="./img/01_flusso_decisionale.svg" width="100%" style="align: center;"/>

> **Come leggere lo schema.** C'è un solo rombo decisionale (la soglia). Il resto è la ground truth: a sinistra il sistema ha rifiutato, a destra ha accettato. Dentro ogni ramo ci sono gli esiti, che dipendono da due fatti che solo chi valuta conosce: la persona è in gallery? E il rank 1 è la sua identità?

> **Perché l'Identification Open Set è più difficile della Verification?** Nella verifica il sistema deve soddisfare un solo vincolo: confrontare il probe con il template dell'identità dichiarata e verificare che il punteggio superi la soglia (confronto 1:1). Nell'identificazione open set invece bisogna soddisfare **due vincoli contemporaneamente**: (1) effettuare un confronto 1:N con l'intera gallery, ordinare tutti i punteggi e verificare che il migliore superi la soglia di accettazione (per stabilire se il soggetto è noto al sistema); (2) verificare che l'identità associata a quel punteggio massimo sia effettivamente quella corretta. Questo doppio vincolo (soglia + correttezza del match 1:N tra molti candidati) rende l'identificazione open set intrinsecamente più complessa e soggetta a tassi di errore più elevati rispetto alla verifica 1:1.

#### Confronto tra i tre scenari

| | Verifica | Identificazione closed set | Identificazione open set |
|---|---|---|---|
| Domanda | "Sei chi dici di essere?" | "Chi sei, tra gli N noti?" | "Sei uno degli N noti? E se sì, chi?" |
| Confronto | 1:1 | 1:N | 1:N |
| Il probe è sempre in gallery? | Sì (dichiara un'identità) | **Sì** (ipotesi chiusa) | **No** |
| Soglia | Sì | No (si prende il rank 1) | Sì |
| Errori possibili | FAR, FRR | rank 1 sbagliato | falso allarme, falso rifiuto, rank 1 sbagliato |
| Metrica tipica | ROC / DET | CMC (rank-$k$) | DIR e FPIR al variare di $t$ |

### 6.2 Template ordinati, soglia e identità assegnata

Lo schema seguente mostra cosa succede *dentro* il sistema per un singolo probe: i template della gallery vengono ordinati per punteggio, la soglia $t$ divide la lista in due zone, e **solo il template al rank 1** (se sopra soglia) viene restituito come identità.

**Un solo probe, visto dall'interno del sistema.** I template sono ordinati per punteggio e la soglia divide la lista in due zone. Qui tre template superano $t$, ma il sistema ne restituisce uno solo: il rank 1 ($g_3$). Il template vero (★) è $g_7$, al rank 2: la persona è in gallery ma viene scambiata, quindi è il caso B (falso rifiuto per la persona vera). Per vedere il caso A basta impostare `TRUE_ID = "g3"` nello script.

```python
# =====================================================================
# 06 — Template ordinati, soglia, identità assegnata  (richiede 01, 05)
# Produce: 02_gallery_ordinata.svg
# Cambia TRUE_ID in "g3" per vedere il caso A (rank 1 corretto).
# =====================================================================
t_ex = 0.70
gal = [("g3", 0.91), ("g7", 0.84), ("g1", 0.77), ("g5", 0.58),
       ("g2", 0.49), ("g8", 0.41), ("g6", 0.33), ("g4", 0.22)]
TRUE_ID = "g7"                                   # template della vera identità (ground truth)

W, H = 920, 610
L, R, TOP, RH = 190, 560, 125, 44
sx = lambda v: L + v*(R-L)
n_up = sum(s >= t_ex for _, s in gal)
rank_true = [g for g, _ in gal].index(TRUE_ID) + 1
yb = TOP + n_up*RH - 6                           # confine tra le due zone

o = new_svg(W, H)
header(o, W, "Template ordinati per punteggio, soglia e identità assegnata",
       f"Un probe p_j confrontato con N = {len(gal)} template · soglia t = {t_ex}")
o.append(f'<rect x="20" y="{TOP-14}" width="880" height="{yb-(TOP-14)}" fill="{C_OK_BG}" fill-opacity="0.6"/>')
o.append(f'<rect x="20" y="{yb}" width="880" height="{TOP+len(gal)*RH-6-yb}" fill="{GREY_BG}"/>')
o.append(f'<line x1="20" y1="{yb}" x2="900" y2="{yb}" stroke="#111" stroke-width="1.5" stroke-dasharray="7,5"/>')

for i, (g, s) in enumerate(gal):
    y = TOP + i*RH
    up = s >= t_ex
    is_true = (g == TRUE_ID)
    o.append(f'<rect x="{L}" y="{y+6}" width="{sx(s)-L:.1f}" height="28" rx="3" fill="{"#16a34a" if up else "#9ca3af"}"/>')
    lab = f"rank {i+1} · {g}" + (" ★" if is_true else "")
    txt(o, L-12, y+25, lab, 13, 700 if i == 0 else None, C_GT if is_true else "#222", "end", halo=False)
    txt(o, sx(s)+8, y+25, f"{s:.2f}", 12, None, "#333", "start", halo=False)
    if i == 0:
        note, col = f"restituito: id({g})" + ("  ★ vero" if is_true else ""), C_G_T
    elif up:
        note, col = "candidato scartato" + ("  ★ template vero" if is_true else ""), C_GT if is_true else "#555"
    else:
        note, col = "sotto soglia" + ("  ★ template vero" if is_true else ""), C_GT if is_true else "#777"
    txt(o, 640, y+25, note, 12.5, 700 if i == 0 or is_true else None, col, "start", halo=False)

xt = sx(t_ex)
yend = TOP + len(gal)*RH - 6
o.append(f'<line x1="{xt:.1f}" y1="{TOP-14}" x2="{xt:.1f}" y2="{yend}" stroke="#111" stroke-width="2" stroke-dasharray="7,5"/>')
txt(o, xt, TOP-22, f"t = {t_ex}", 14, 700, "#111", halo=False)
txt(o, 30, yb-8, "sopra soglia", 11.5, 700, C_G_T, "start", halo=False)
txt(o, 30, yb+16, "sotto soglia", 11.5, 700, "#555", "start", halo=False)

yo = yend + 52
if rank_true == 1:
    box(o, 460, yo, 760, 62, [f"Output: id({gal[0][0]}) ✓ identità corretta", "rank(p_j) = 1 e sopra soglia: caso A (Correct Detect and Identify)"], C_OK_BG, C_G_T, 13)
else:
    box(o, 460, yo, 760, 62, [f"Output: id({gal[0][0]}) ✗ non è la vera identità",
        f"il template vero è al rank {rank_true}: caso B (falso rifiuto per la persona vera)"], C_FR_BG, C_FR_T, 13)
txt(o, 460, H-12, "★ = template della vera identità: il rank lo calcola chi valuta (ground truth), il sistema vede solo l'ordinamento", 11.5, None, "#555", halo=False)
save(o, '02_gallery_ordinata.svg')
```

<img src="./img/02_gallery_ordinata.svg" width="100%" style="align: center;"/>

Punti da notare:

- **Il rank non è un'etichetta del sistema.** È la posizione che il template *vero* occupa nella lista. Il sistema conosce solo l'ordinamento, non quale sia quello vero; il rank lo calcola chi valuta.
- **Più template possono stare sopra soglia** (qui $g_3$, $g_7$, $g_1$), ma il sistema ne restituisce uno solo: il rank 1. Gli altri sopra soglia sono candidati scartati.
- **La soglia è un filtro, non un giudice di correttezza.** Superare $t$ significa "abbastanza simile", non "identità giusta". Per questo servono entrambi i vincoli.

### 6.3 Possibili esiti

I cinque casi si distinguono incrociando *dove sta la persona* (in gallery o no) con *cosa fa il sistema*:

**La stessa classificazione in forma di griglia.** Incrociando le tre righe (cosa fa il sistema) con le due colonne (dove sta davvero la persona) si ottengono sei celle, ma una è impossibile: un ignoto non ha un'identità corretta da restituire, quindi "accetta con rank 1 corretto" non può accadere. Le altre cinque sono i casi A–E. In basso, a quale metrica contribuisce ciascun caso.

```python
# =====================================================================
# 07 — I cinque esiti dell'open set  (richiede 01, 05)
# Produce: 03_cinque_esiti.svg
# =====================================================================
W, H = 920, 640
o = new_svg(W, H)
header(o, W, "Identificazione open set: i cinque esiti",
       "Righe = cosa fa il sistema · colonne = dove si trova davvero la persona (ground truth)")
XL, CW, YT, RHt = 270, 300, 150, 125
cols = ["Persona IN gallery", "Persona NON in gallery"]
subs = ["p_j ∈ P_G (noto)", "p_j ∈ P_N (impostore)"]
rows = [("Il sistema rifiuta", "s* &lt; t"),
        ("Accetta · rank 1 corretto", "s* ≥ t  e  id(g_rank1) = id(p_j)"),
        ("Accetta · rank 1 sbagliato", "s* ≥ t  e  id(g_rank1) ≠ id(p_j)")]
for c in range(2):
    txt(o, XL+c*CW+CW/2, 98, cols[c], 15, 700, C_GT, halo=False)
    txt(o, XL+c*CW+CW/2, 118, subs[c], 12.5, None, "#555", halo=False)
for r, (a, b) in enumerate(rows):
    y = YT + r*RHt
    txt(o, XL-14, y+RHt/2-4, a, 14, 700, "#111", "end", halo=False)
    txt(o, XL-14, y+RHt/2+14, b, 11.5, None, "#555", "end", halo=False)
cells = {
    (0, 0): ("C · False Rejection", "noto non riconosciuto", C_FR_BG, C_FR_T),
    (0, 1): ("D · Genuine Reject", "impostore rifiutato: corretto", C_OK_BG, C_G_T),
    (1, 0): ("A · Correct Detect and Identify", "noto riconosciuto: corretto", C_OK_BG, C_G_T),
    (1, 1): ("impossibile", "un ignoto non ha identità corretta", GREY_BG, "#6b7280"),
    (2, 0): ("B · False Rejection", "noto accettato ma scambiato (misidentification)", C_FR_BG, C_FR_T),
    (2, 1): ("E · False Alarm", "impostore accettato (False Acceptance)", C_FA_BG, C_FA_T),
}
for (r, c), (t1, t2, bg, fg) in cells.items():
    x, y = XL + c*CW + 8, YT + r*RHt + 8
    rect(o, x, y, CW-16, RHt-16, bg, fg, 2, 10, "5,4" if t1 == "impossibile" else None)
    txt(o, x+(CW-16)/2, y+(RHt-16)/2-2, t1, 14.5, 700, fg, halo=False)
    txt(o, x+(CW-16)/2, y+(RHt-16)/2+19, t2, 11.5, None, "#333", halo=False)

txt(o, 40, 560, "Come entrano nelle metriche", 13, 700, "#111", "start", halo=False)
items = [("A", "DIR(t,1) = A / |P_G|", C_G_T), ("B + C", "FNIR(t) = (B + C) / |P_G| = 1 − DIR(t,1)", C_FR_T),
         ("E", "FPIR(t) = E / |P_N|", C_FA_T), ("D", "1 − FPIR(t) = D / |P_N|", C_G_T)]
for i, (k, v, col) in enumerate(items):
    xx = 40 + (i % 2)*430
    yy = 585 + (i // 2)*24
    txt(o, xx, yy, k, 13, 700, col, "start", halo=False)
    txt(o, xx+62, yy, v, 12.5, None, "#333", "start", halo=False)
save(o, '03_cinque_esiti.svg')
```

<img src="./img/03_cinque_esiti.svg" width="100%" style="align: center;"/>

| Caso | Situazione | Esito |
|---|---|---|
| D | Nessun valore sopra soglia, persona non in gallery | Genuine Reject |
| C | Nessun valore sopra soglia, persona in gallery | False Rejection |
| A | Valori sopra soglia, il primo è quello corretto | **Correct Detect and Identify** |
| B | Valori sopra soglia, ma il primo NON è quello corretto | False Rejection (per l'identità corretta) |
| E | Persona non in gallery ma un valore supera la soglia | False Acceptance (**False Alarm**) |

Nel caso B il sistema ha *accettato* qualcuno, ma ha restituito l'identità sbagliata: per la persona vera è comunque un fallimento, quindi conta come falso rifiuto. In altri testi lo trovi anche come *misidentification*.

### 6.4 Rank e Detection and Identification Rate (DIR)

Il **rank** è la posizione nella lista ordinata in cui compare il template dell'identità corretta.

$$
DIR(t,k) = \frac{|\{p_j : rank(p_j) \le k,\ s_{ij} \ge t,\ id(g_i) = id(p_j)\}|}{|P_G|} \quad \forall p_j \in P_G
$$

$$
FRR(t) = 1 - DIR(t,1)
$$

$$
FAR(t) = \frac{|\{p_j : \max_i s_{ij} \ge t\}|}{|P_N|} \quad \forall p_j \in P_N,\ \forall g_i \in G
$$

$$
EER = \{t : FRR(t) = FAR(t)\}
$$

> **Nomenclatura standard ISO nell'open set:** l'equivalente del FRR è la **FNIR (False Negative Identification Rate)** = $1 - DIR(t,1)$; l'equivalente del FAR è la **FPIR (False Positive Identification Rate)**, cioè la probabilità che un impostore ($p_j \in P_N$) venga accettato per errore:
> $$FPIR(t) = \frac{|\{p_j \in P_N : \max_i s_{ij} \ge t\}|}{|P_N|}$$
> Applicazioni tipiche del DIR/open set: videosorveglianza, ricerca persone in grandi database, riconoscimento facciale open-set, applicazioni di *law enforcement*, dove non si sa a priori se il soggetto ripreso sia registrato nel sistema.

### 6.5 Come si derivano le formule

L'idea è una sola: **per ogni tipo di probe si identifica l'evento "errore" o "successo" e se ne calcola la frequenza relativa**. I due tipi di probe ($P_G$ e $P_N$) hanno denominatori diversi perché possono sbagliare in modi diversi.

#### Passo 0: la regola di decisione

Per un probe $p_j$ sia $i^* = \arg\max_i s_{ij}$ e $s^*_j = s_{i^*j} = \max_i s_{ij}$. Il sistema produce

$$
D_t(p_j) = \begin{cases} id(g_{i^*}) & \text{se } s^*_j \ge t \\ \text{rifiuto} & \text{se } s^*_j < t \end{cases}
$$

La soglia $t$ è l'unico parametro. Ogni formula sotto è la frequenza di un evento definito da $D_t$.

#### Passo 1: probe di impostori ($P_N$) → FPIR (il "FAR" dell'open set)

Un impostore non ha un'identità corretta da restituire, quindi qualunque accettazione è un errore. L'evento errore è:

$$
E_N = \{ D_t(p_j) \neq \text{rifiuto} \} = \{ s^*_j \ge t \} = \{\max_i s_{ij} \ge t\}
$$

La frequenza relativa su $P_N$ è la formula FPIR/FAR:

$$
FPIR(t) = \frac{|\{p_j \in P_N : \max_i s_{ij} \ge t\}|}{|P_N|}
$$

Il denominatore è $|P_N|$ perché un falso allarme esiste solo per chi non è in gallery. In termini probabilistici, $FPIR(t) = P(s^* \ge t \mid p \in P_N)$.

**Perché cresce con la dimensione della gallery.** Se per un impostore i punteggi con i singoli template sono indipendenti e ciascuno supera $t$ con probabilità $FAR_{1:1}(t)$ (il FAR della verifica), la probabilità che *almeno uno* dei $N$ lo faccia è

$$
FPIR(t) = 1 - \bigl(1 - FAR_{1:1}(t)\bigr)^N \approx N \cdot FAR_{1:1}(t) \quad (\text{se } N \cdot FAR_{1:1} \ll 1)
$$

Un sistema con $FAR_{1:1} = 0.1\%$ su gallery da $N = 1000$ ha $FPIR \approx 1 - 0.999^{1000} \approx 63\%$. È il motivo quantitativo per cui l'open set è più difficile della verifica e per cui le soglie vanno alzate molto al crescere di $N$. (L'indipendenza è un'assunzione di modello, utile come ordine di grandezza; i punteggi reali sono correlati.)

**La FPIR cresce con N.** Le curve sono $1-(1-FAR_{1:1})^N$ per tre valori di FAR; il tratteggio è l'approssimazione $N\cdot FAR_{1:1}$, che satura a 1 mentre la curva vera no. Il punto evidenziato è l'esempio del testo (FAR 0.1%, N = 1000 → 63%).

```python
# =====================================================================
# N07 — FPIR vs N  (richiede 01, 05, 00)
# =====================================================================
W, H = 920, 640
L, R, TOP, BOT = 95, 760, 90, 520
o = new_svg(W, H)
header(o, W, "L'open set peggiora con la gallery: FPIR in funzione di N",
       "FPIR(t) = 1 − (1 − FAR(1:1))^N  ≈  N · FAR(1:1)  solo finché N · FAR(1:1) è molto minore di 1")
lx = lambda n: math.log10(n)
sx = lambda n: L+lx(n)/5*(R-L)
sy = lambda v: BOT-v*(BOT-TOP)
for v in [0, 0.25, 0.5, 0.75, 1.0]:
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#e5e7eb"/>')
    txt(o, L-8, sy(v)+4, f"{v:.0%}", 12, None, "#333", "end", halo=False)
for e, lab in enumerate(["1", "10", "100", "1k", "10k", "100k"]):
    o.append(f'<line x1="{sx(10**e):.1f}" y1="{TOP}" x2="{sx(10**e):.1f}" y2="{BOT}" stroke="#e5e7eb"/>')
    txt(o, sx(10**e), BOT+20, lab, 12, None, "#333", halo=False)
frame(o, L, R, TOP, BOT, "N = numero di identità in gallery (scala log)", "FPIR")
ns = [10**(5*i/400) for i in range(401)]
exact = lambda n, f: 1-math.exp(n*math.log1p(-f))
curves = [(1e-3, "#2563eb", "FAR(1:1) = 0.1%"), (1e-4, "#16a34a", "FAR(1:1) = 0.01%"), (1e-5, "#7c3aed", "FAR(1:1) = 0.001%")]
polyline(o, [(sx(n), sy(min(1, n*1e-3))) for n in ns], "#6b7280", 2, "6,5")
for f, col, lab in curves:
    polyline(o, [(sx(n), sy(exact(n, f))) for n in ns], col, 3.2)
    n50 = math.log(2)/f
    txt(o, sx(n50)+12, sy(0.5)+22, lab, 12.5, 700, col, "start")
# punto N=1000
v = exact(1000, 1e-3)
o.append(f'<line x1="{sx(1000):.1f}" y1="{sy(v):.1f}" x2="{sx(1000):.1f}" y2="{BOT}" stroke="#111" stroke-dasharray="4,4"/>')
o.append(f'<circle cx="{sx(1000):.1f}" cy="{sy(v):.1f}" r="7" fill="#fff" stroke="#111" stroke-width="2.5"/>')
txt(o, sx(1000)-14, sy(v)+24, f"N = 1000 → FPIR = {v:.0%}", 13, 700, "#111", "end")
txt(o, sx(1000)-14, sy(v)+40, "con FAR(1:1) = 0.1%", 11.5, None, "#333", "end")
# approssimazione
txt(o, sx(1.3), sy(0.90), "Tratteggio grigio = approssimazione N · FAR:", 12, 700, "#6b7280", "start")
txt(o, sx(1.3), sy(0.90)+16, "a N = 1000 darebbe 100%, la curva vera dà 63%.", 12, None, "#444", "start")
txt(o, sx(1.3), sy(0.90)+32, "Oltre N · FAR ≈ 1 l'approssimazione non vale.", 12, None, "#444", "start")
txt(o, L, H-40, "Per mantenere la FPIR accettabile al crescere di N la soglia va alzata molto (e la FNIR sale di conseguenza).", 12.5, None, "#444", "start", halo=False)
txt(o, L, H-20, "L'indipendenza dei punteggi è un'assunzione di modello: i punteggi reali sono correlati, quindi è un ordine di grandezza.", 11.5, None, "#666", "start", halo=False)
save(o, 'n07_fpir_vs_N.svg')
```
<img src="./img/n07_fpir_vs_N.svg" alt="FPIR in funzione di N" style="display:block; margin:1.5em auto; max-width:100%;">

#### Passo 2: probe di persone note ($P_G$) → DIR

Per un probe noto il successo richiede **tre condizioni insieme**:

1. il template vero compare entro i primi $k$ posti: $rank(p_j) \le k$;
2. quel template ha punteggio sopra soglia: $s_{ij} \ge t$;
3. l'identità coincide: $id(g_i) = id(p_j)$ (è la definizione stessa di "template vero").

L'evento successo è quindi $S_k = \{rank(p_j) \le k \ \wedge\ s_{ij} \ge t \ \wedge\ id(g_i) = id(p_j)\}$ e la sua frequenza relativa su $P_G$ è la formula del DIR:

$$
DIR(t,k) = \frac{|\{p_j : rank(p_j) \le k,\ s_{ij} \ge t,\ id(g_i) = id(p_j)\}|}{|P_G|}
$$

Il denominatore è $|P_G|$: solo chi è in gallery può essere "detected and identified". Per $k = 1$ si chiede che il template vero sia **proprio** quello restituito dal sistema, cioè il caso A. Per $k > 1$ si ammette una shortlist (utile se poi decide un operatore).

**Scomposizione utile.** Per $k=1$, condizionando al fatto che il rank 1 sia corretto:

$$
DIR(t,1) = P(\text{rank 1 corretto} \mid p \in P_G) \cdot P(s^* \ge t \mid \text{rank 1 corretto})
$$

Il primo fattore è il *rank-1 identification rate* del closed set (indipendente da $t$); il secondo è la frazione di genuine con punteggio sopra soglia. Cambiando $t$ si agisce solo sul secondo. Quindi $DIR(t,1)$ **non può superare il rank-1 closed set**, e questo spiega perché la FNIR non va a zero nemmeno con $t$ molto basso.

#### Passo 3: FRR = FNIR

Per un probe noto gli esiti possibili sono mutuamente esclusivi ed esaustivi:

- A: rank 1 giusto e sopra soglia → successo;
- B: rank 1 sbagliato → fallimento;
- C: vero template sotto soglia (e quindi rank 1 sotto soglia o sbagliato) → fallimento.

Quindi la probabilità di fallire è il complemento di quella di riuscire:

$$
FRR(t) = FNIR(t) = 1 - DIR(t,1)
$$

Attenzione: questo complemento vale per $k=1$. Con $k>1$ "non successo entro rank $k$" non coincide con l'errore del sistema, perché il sistema restituisce comunque solo il rank 1.

#### Passo 4: andamento con $t$ ed EER

- Alzando $t$, meno impostori superano la soglia → **FPIR scende** (monotona non crescente).
- Alzando $t$, meno noti superano la soglia → $DIR(t,1)$ scende → **FNIR sale** (monotona non decrescente).

Una funzione che scende e una che sale si incrociano: l'**EER** è il valore di $t$ (e del tasso) in cui

$$
FNIR(t^*) = FPIR(t^*)
$$

Nei dati reali, con insiemi finiti, le curve sono a gradini e l'uguaglianza esatta può non esistere: si prende il $t$ che minimizza $|FNIR(t) - FPIR(t)|$. La curva $\bigl(FPIR(t),\, DIR(t,1)\bigr)$ al variare di $t$ è la **ROC dell'open set**; con $DIR(0,k)$ al variare di $k$ si ottiene invece la **CMC**, che è il caso closed set.

**ROC dell'open set.** La curva $(FPIR(t), DIR(t,1))$ non arriva a 1: il limite è il Recognition Rate del closed set, perché $DIR(t,1)\le RR$ per ogni $t$.

```python
# =====================================================================
# N08 — ROC open set  (richiede 01, 05, 00)
# =====================================================================
MU_N, SD_N, MU_K, SD_K, RR = 0.45, 0.10, 0.72, 0.12, 0.85
fpir = lambda t: 0.5*math.erfc((t-MU_N)/(SD_N*math.sqrt(2)))
dir1 = lambda t: RR*(1-cdf(t, MU_K, SD_K))
W, H = 780, 800
L, R, TOP, BOT = 95, 715, 85, 705
sx = lambda x: L+x*(R-L); sy = lambda y: BOT-y*(BOT-TOP)
o = new_svg(W, H)
header(o, W, "ROC dell'open set: DIR(t,1) contro FPIR(t)",
       f"ogni punto = una soglia t · rank-1 corretto per il {RR:.0%} dei noti (Recognition Rate del closed set)")
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#e5e7eb"/>')
    o.append(f'<line x1="{sx(v):.1f}" y1="{TOP}" x2="{sx(v):.1f}" y2="{BOT}" stroke="#e5e7eb"/>')
    txt(o, L-8, sy(v)+4, f"{v:.1f}", 12, None, "#333", "end", halo=False)
    txt(o, sx(v), BOT+20, f"{v:.1f}", 12, None, "#333", halo=False)
frame(o, L, R, TOP, BOT, "FPIR(t) = falsi allarmi sugli ignoti P_N", "DIR(t,1) = 1 − FNIR(t)")
ts = [1.4-1.8*i/1500 for i in range(1501)]
roc = [(fpir(t), dir1(t)) for t in ts]
area = f"M {sx(0):.1f},{BOT} " + " ".join(f"L {sx(x):.1f},{sy(y):.1f}" for x, y in roc) + f" L {sx(1):.1f},{BOT} Z"
o.append(f'<path d="{area}" fill="{C_I}" fill-opacity="0.10"/>')
# regione inaccessibile sopra RR
o.append(f'<rect x="{L}" y="{TOP}" width="{R-L}" height="{sy(RR)-TOP:.1f}" fill="#f3f4f6" fill-opacity="0.9"/>')
o.append(f'<line x1="{L}" y1="{sy(RR):.1f}" x2="{R}" y2="{sy(RR):.1f}" stroke="{C_FA}" stroke-width="2.2" stroke-dasharray="7,5"/>')
txt(o, R-8, sy(RR)-10, f"limite RR = {RR:.2f}  (DIR(t,1) ≤ RR per ogni t)", 12.5, 700, C_FA_T, "end")
txt(o, (L+R)/2, (TOP+sy(RR))/2+4, "zona irraggiungibile, qualunque sia t", 12, None, "#6b7280")
txt(o, (L+R)/2, (TOP+sy(RR))/2+20, "un noto con rank 1 sbagliato non si recupera abbassando t", 12, None, "#6b7280")
polyline(o, [(sx(x), sy(y)) for x, y in roc], C_I, 3.4)
t0 = 0.62
x0, y0 = fpir(t0), dir1(t0)
o.append(f'<circle cx="{sx(x0):.1f}" cy="{sy(y0):.1f}" r="7" fill="#111" stroke="#fff" stroke-width="2"/>')
txt(o, sx(x0)+16, sy(y0)+24, f"t = {t0}", 12.5, 700, "#111", "start")
txt(o, sx(x0)+16, sy(y0)+40, f"FPIR = {x0:.1%}, DIR = {y0:.1%}", 11.5, None, "#333", "start")
txt(o, sx(x0)+16, sy(y0)+56, f"FNIR = {1-y0:.1%}", 11.5, None, "#333", "start")
o.append(f'<circle cx="{sx(1):.1f}" cy="{sy(RR):.1f}" r="6" fill="{C_FA}" stroke="#fff" stroke-width="2"/>')
txt(o, sx(1)-10, sy(RR)+22, "t = t_min: CMS(1) = RR", 12, 700, C_FA_T, "end")
txt(o, sx(0.52), sy(0.22), "la curva non arriva a 1", 13, 700, "#1d4ed8")
txt(o, sx(0.52), sy(0.22)+17, "perché alcuni noti hanno il rank 1 sbagliato", 12, None, "#444")
save(o, 'n08_roc_open_set.svg')
```
<img src="./img/n08_roc_open_set.svg" alt="ROC dell'open set" style="display:block; margin:1.5em auto; max-width:100%;">


**Le due distribuzioni di $s^*$.** Per ogni probe conta solo il punteggio massimo contro la gallery. Gli ignoti ($P_N$) hanno $s^*$ più alto di quanto avrebbero in una verifica 1:1, perché è il massimo su $N$ template: è il motivo per cui la FPIR cresce con $N$. L'area arancione a destra di $t$ è la FPIR; l'area verde a sinistra è la quota di noti persi per punteggio troppo basso. Alla FNIR si aggiunge poi l'errore di rank, che in questo grafico non compare.

```python
# =====================================================================
# 08 — Distribuzioni di s* e soglia (FPIR e noti persi)  (richiede 01, 05)
# Produce: 04_distribuzioni_soglia.svg
# =====================================================================
MU_N, SD_N = 0.45, 0.10        # s* = max_i s_ij per gli ignoti (P_N): il massimo su N template è spostato in alto
MU_K, SD_K = 0.72, 0.12        # s* per i noti (P_G)
T_OS = 0.62
C_FPIR, C_LOST = "#f97316", "#16a34a"

W, H = 920, 560
L, R, TOP, BOT = 70, 880, 110, 400
XMIN, XMAX, YMAX = 0.0, 1.1, 4.4
sx = lambda x: L + (x-XMIN)/(XMAX-XMIN)*(R-L)
sy = lambda y: BOT - y/YMAX*(BOT-TOP)

fpir_v = 0.5*math.erfc((T_OS-MU_N)/(SD_N*math.sqrt(2)))
lost_v = cdf(T_OS, MU_K, SD_K)

o = new_svg(W, H)
header(o, W, "Open set: distribuzione di s* = max_i s_ij e soglia t",
       f"Ignoti (P_N) ~ N({MU_N}, {SD_N}²) · Noti (P_G) ~ N({MU_K}, {SD_K}²) · soglia t = {T_OS}")
o.append(f'<path d="{g_area(MU_N,SD_N,XMIN,T_OS,sx,sy,BOT)}" fill="{C_I}" fill-opacity="0.15"/>')
o.append(f'<path d="{g_area(MU_K,SD_K,T_OS,XMAX,sx,sy,BOT)}" fill="{C_G}" fill-opacity="0.15"/>')
o.append(f'<path d="{g_area(MU_N,SD_N,T_OS,XMAX,sx,sy,BOT)}" fill="{C_FPIR}" fill-opacity="0.8"/>')
o.append(f'<path d="{g_area(MU_K,SD_K,XMIN,T_OS,sx,sy,BOT)}" fill="{C_LOST}" fill-opacity="0.8"/>')
o.append(f'<path d="{g_line(MU_N,SD_N,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_I}" stroke-width="2.5"/>')
o.append(f'<path d="{g_line(MU_K,SD_K,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_G}" stroke-width="2.5"/>')
frame(o, L, R, TOP, BOT, "s* = punteggio massimo del probe contro la gallery", "Densità")
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{sx(v):.1f}" y1="{BOT}" x2="{sx(v):.1f}" y2="{BOT+5}" stroke="#222"/>')
    txt(o, sx(v), BOT+20, f"{v:.1f}", 12, None, "#333", halo=False)
o.append(f'<line x1="{sx(T_OS):.1f}" y1="{TOP-6}" x2="{sx(T_OS):.1f}" y2="{BOT}" stroke="#111" stroke-width="2" stroke-dasharray="7,5"/>')
txt(o, sx(T_OS), TOP-18, f"soglia t = {T_OS}", 14, 700, "#111", halo=False)
txt(o, sx(T_OS)-8, TOP-1, "← rifiuto", 12, None, "#111", "end", halo=False)
txt(o, sx(T_OS)+8, TOP-1, "accetto →", 12, None, "#111", "start", halo=False)
txt(o, sx(MU_N)-12, sy(pdf(MU_N, MU_N, SD_N))-10, "Ignoti (P_N)", 15, 700, C_I)
txt(o, sx(MU_K)+40, sy(pdf(MU_K, MU_K, SD_K))-10, "Noti (P_G)", 15, 700, C_G_T)
yl = BOT-18
txt(o, sx(0.99), sy(1.55), f"FPIR = {fpir_v:.1%}", 14, 700, "#c2410c")
txt(o, sx(0.99), sy(1.55)+15, "ignoti accettati", 11.5, None, "#333")
leader(o, [(sx(0.99), sy(1.55)+22), (sx(0.99), yl), (sx(0.665), yl)], "#c2410c")
txt(o, sx(0.17), sy(1.55), f"noti persi = {lost_v:.1%}", 14, 700, C_G_T)
txt(o, sx(0.17), sy(1.55)+15, "punteggio troppo basso", 11.5, None, "#333")
leader(o, [(sx(0.17), sy(1.55)+22), (sx(0.17), yl), (sx(0.52), yl)], C_G_T)
ly = BOT+68
for i, (c, t, op) in enumerate([(C_FPIR, "FPIR: ignoti con s* ≥ t", 0.8), (C_LOST, "noti con s* &lt; t (contributo a FNIR)", 0.8),
                                (C_I, "ignoti rifiutati", 0.15), (C_G, "noti con s* ≥ t", 0.15)]):
    xx = L + (i % 2)*330
    yy = ly + (i // 2)*24
    o.append(f'<rect x="{xx}" y="{yy-11}" width="16" height="16" fill="{c}" fill-opacity="{op}" stroke="{c}"/>')
    txt(o, xx+22, yy+2, t, 12.5, None, "#222", "start", halo=False)
txt(o, L, ly+62, "Nota: l'errore di rank (rank 1 sbagliato, caso B) non si vede in questo grafico, perché riguarda quale template è primo e non il suo punteggio.", 11.5, None, "#666", "start", halo=False)
save(o, '04_distribuzioni_soglia.svg')
```

<img src="./img/04_distribuzioni_soglia.svg" alt="Distribuzioni di s* e soglia" style="width: 100%; height: 100%; align: center">

> Nel grafico l'area arancione a destra di $t$ è la FPIR (impostori accettati); l'area verde a sinistra di $t$ è la quota di noti persi per punteggio troppo basso. A questa si somma l'errore di rank, che nel grafico non si vede perché riguarda *quale* template è al primo posto, non il suo punteggio.

#### Esempio numerico

Si valuta un sistema con $|P_G| = 200$ probe noti e $|P_N| = 300$ probe di impostori, con una certa soglia $t$.

- Tra i 200 noti: 150 hanno il rank 1 corretto e sopra soglia (caso A); 30 hanno il template vero sotto soglia (caso C); 20 hanno un rank 1 sbagliato (caso B).
- Tra i 300 impostori: 24 hanno $\max_i s_{ij} \ge t$ (caso E).

Allora

$$
DIR(t,1) = \frac{150}{200} = 0.75, \qquad FNIR(t) = 1 - 0.75 = 0.25 = \frac{30 + 20}{200}, \qquad FPIR(t) = \frac{24}{300} = 0.08
$$

Con una soglia più alta, per esempio, la FPIR potrebbe scendere verso 0.02 ma la FNIR salirebbe oltre 0.25: questo è il compromesso che si legge sulla curva ROC.

### 6.6 Cinque aree operative (scelta soglia watchlist)

La soglia non è un valore "giusto" in assoluto: dipende dal costo relativo dei due errori nell'applicazione. Le cinque aree corrispondono a cinque posizioni sulla curva ROC dell'open set:

| # | Area operativa | Soglia | Quando ha senso |
|---|---|---|---|
| 1 | Falso allarme estremamente basso (es. sorveglianza pubblica) | $t$ molto alta | Ogni falso allarme ferma o disturba una persona innocente: meglio perdere qualche sospetto |
| 2 | Probabilità di detect/identify estremamente alta (falsi allarmi secondari) | $t$ molto bassa | Non si vuole perdere nessun sospetto; i match vengono poi verificati da un operatore |
| 3 | Basso falso allarme e basso detect/identify | $t$ alta | Sistema prudente: pochi allarmi, ma ne riconosce pochi |
| 4 | Alto falso allarme e alto detect/identify | $t$ bassa | Sistema permissivo: riconosce molto, ma genera molti allarmi |
| 5 | Nessuna soglia: si vogliono tutti i risultati con relativa confidenza | assente | Decide l'utente guardando la lista ordinata (ricerca investigativa) |

**Le cinque aree sulla ROC** (FPIR in scala log). Le aree 1 e 3 sono nella zona a bassi falsi allarmi, 4 e 2 a falsi allarmi crescenti, 5 è l'assenza di soglia (lista ordinata, cioè CMC).

```python
# =====================================================================
# N09 — Cinque aree operative  (richiede 01, 05, 00)
# =====================================================================
MU_N, SD_N, MU_K, SD_K, RR = 0.45, 0.10, 0.72, 0.12, 0.85
nd = NormalDist(MU_N, SD_N)
dir1 = lambda t: RR*(1-cdf(t, MU_K, SD_K))
tof = lambda fp: nd.inv_cdf(1-fp)
W, H = 920, 700
L, R, TOP, BOT = 95, 630, 85, 600
XMIN, XMAX = 1e-3, 1.0
lg = lambda v: (math.log10(v)-math.log10(XMIN))/(math.log10(XMAX)-math.log10(XMIN))
sx = lambda x: L+lg(x)*(R-L); sy = lambda y: BOT-y*(BOT-TOP)
o = new_svg(W, H)
header(o, W, "Le cinque aree operative sulla ROC dell'open set", "FPIR in scala logaritmica · ogni area è una scelta di soglia diversa")
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#e5e7eb"/>')
    txt(o, L-8, sy(v)+4, f"{v:.1f}", 12, None, "#333", "end", halo=False)
for v, lab in [(1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%"), (1.0, "100%")]:
    o.append(f'<line x1="{sx(v):.1f}" y1="{TOP}" x2="{sx(v):.1f}" y2="{BOT}" stroke="#e5e7eb"/>')
    txt(o, sx(v), BOT+20, lab, 12, None, "#333", halo=False)
frame(o, L, R, TOP, BOT, "FPIR (scala log)", "DIR(t,1)")
pts = []
for i in range(1, 600):
    fp = 10**(-3+3*i/600)
    pts.append((sx(fp), sy(dir1(tof(fp)))))
polyline(o, pts, C_I, 3.4)
o.append(f'<line x1="{L}" y1="{sy(RR):.1f}" x2="{R}" y2="{sy(RR):.1f}" stroke="{C_FA}" stroke-dasharray="7,5" stroke-width="1.8"/>')
txt(o, L+8, sy(RR)-8, f"RR = {RR:.2f}", 12, 700, C_FA_T, "start")
areas = [(1, "#0f766e", 0.0015, "Falso allarme estremamente basso", "t molto alta"),
         (3, "#7c3aed", 0.02, "Basso FA e basso detect/identify", "t alta"),
         (4, "#db2777", 0.12, "Alto FA e alto detect/identify", "t bassa"),
         (2, "#ea580c", 0.55, "Detect/identify estremamente alta", "t molto bassa"),
         (5, "#374151", 1.0, "Nessuna soglia: lista ordinata", "decide l'utente")]
for n, col, fp, name, sub in areas:
    y = RR if n == 5 else dir1(tof(fp))
    o.append(f'<circle cx="{sx(fp):.1f}" cy="{sy(y):.1f}" r="19" fill="{col}" fill-opacity="0.14" stroke="{col}" stroke-dasharray="4,3"/>')
    badge(o, sx(fp), sy(y), n, col, 10)
txt(o, sx(1.0)+2, sy(RR)+38, "t assente: CMC", 11.5, 700, "#374151", "end")
# legenda a destra
xl = 665
txt(o, xl, 110, "Quando ha senso", 14, 700, "#111", "start", halo=False)
whens = {1: "ogni falso allarme disturba un innocente (sorveglianza pubblica)",
         2: "non si vuole perdere nessun sospetto; poi verifica l'operatore",
         3: "sistema prudente: pochi allarmi, ne riconosce pochi",
         4: "sistema permissivo: riconosce molto, molti allarmi",
         5: "ricerca investigativa: si guarda la lista ordinata"}
y = 150
for n, col, fp, name, sub in sorted(areas, key=lambda a: a[0]):
    badge(o, xl+10, y-4, n, col, 10)
    txt(o, xl+28, y, name, 11.5, 700, "#111", "start", halo=False)
    txt(o, xl+28, y+15, f"soglia: {sub}", 11, None, col, "start", halo=False)
    words = whens[n].split(); lines, cur = [], ""
    for w_ in words:
        if len(cur)+len(w_) > 34: lines.append(cur); cur = w_
        else: cur = (cur+" "+w_).strip()
    lines.append(cur)
    for k, s in enumerate(lines):
        txt(o, xl+28, y+30+k*14, s, 11, None, "#444", "start", halo=False)
    y += 30+len(lines)*14+24
save(o, 'n09_cinque_aree_operative.svg')
```
<img src="./img/n09_cinque_aree_operative.svg" alt="Cinque aree operative" style="display:block; margin:1.5em auto; max-width:100%;">


### 6.7 Riepilogo rapido

- **Verifica**: 1:1, un solo vincolo (soglia).
- **Open set**: 1:N, due vincoli (soglia **e** rank 1 corretto).
- **FPIR** si misura sugli impostori ($P_N$): conta chi supera la soglia con *qualunque* template. Cresce circa come $N \cdot FAR_{1:1}$.
- **DIR(t,1)** si misura sui noti ($P_G$): conta chi ha il rank 1 corretto **e** sopra soglia.
- **FNIR = 1 - DIR(t,1)**.
- Alzare $t$ → meno falsi allarmi, più falsi rifiuti. L'**EER** è il punto in cui si equivalgono.

---

## 7. Identificazione Closed Set

Caso speciale: si assume che **ogni probe appartenga a un soggetto enrollato** (non realistico, ma utile per valutare le prestazioni perché più semplice da calcolare).

- **Non c'è soglia.**
- L'unica domanda è: *chi è questa persona?*
- L'unico errore possibile è la **False Rejection** (identità corretta non al primo posto); **non esiste False Acceptance**.
- Il FRR del closed set si calcola come complemento del **Recognition Rate** (= CMS al rango 1):

$$
FRR_{closed} = 1 - CMS(1) = 1 - RR
$$

### 7.0 Notazione e procedura

| Simbolo | Significato |
|---|---|
| $G = \{g_1, \dots, g_N\}$ | Gallery con $N$ identità enrollate |
| $P$ | Insieme dei probe; nel closed set **tutti** hanno l'identità in $G$ (quindi $P = P_G$ e $P_N = \emptyset$) |
| $s_{ij} = sim(p_j, g_i)$ | Punteggio di similarità tra probe $j$ e template $i$ |
| $rango(p_j)$ | Posizione del template della vera identità di $p_j$ nella lista ordinata per punteggio decrescente ($1 \le rango \le N$) |

Procedura per un probe $p_j$:

1. si calcolano i punteggi con **tutti** i template della gallery (confronto 1:N);
2. si ordinano i template per punteggio decrescente;
3. si restituisce l'identità del template al **rank 1**, **senza alcun test di soglia**.

Il sistema risponde quindi sempre con un'identità. Chi valuta, conoscendo la ground truth, guarda a che rank compare il template vero.

**Closed set: nessun rombo di soglia.** Il sistema restituisce sempre l'identità del rank 1. A sinistra il template vero è primo (successo), a destra è al rank 3: l'identità restituita è sbagliata, ma lo stesso probe contribuisce a $CMS(k)$ per ogni $k \ge 3$. È questa la differenza tra "sbagliare" e "essere in una shortlist".

```python
# =====================================================================
# 09 — Identificazione closed set: flusso e due casi  (richiede 01, 05)
# Produce: 05_closed_set_flusso.svg
# =====================================================================
W, H = 920, 610
o = new_svg(W, H)
header(o, W, "Identificazione closed set: nessuna soglia, risponde sempre il rank 1",
       "★ = template della vera identità (ground truth: la conosce solo chi valuta)")

def closed_panel(x, ids, true_idx, title, ok):
    col = C_G_T if ok else C_FR_T
    rect(o, x, 78, 420, 480, "#fff", "#e5e7eb", 1.5, 14)
    txt(o, x+210, 106, title, 15, 700, col, halo=False)
    box(o, x+58, 200, 96, 40, ["probe p_j"], BLU_BG, C_I)
    arrow(o, x+108, 200, x+132, 200); txt(o, x+120, 190, "1:N", 11, 700, "#111")
    for i in range(len(ids)):
        y = 135 + i*36
        is_t = (i == true_idx)
        rect(o, x+134, y, 150, 30, C_GT_BG if is_t else "#f9fafb", C_GT if is_t else "#9ca3af", 1.6, 6, "5,3" if is_t else None)
        txt(o, x+209, y+20, f"{i+1}  ·  {ids[i]}" + ("  ★" if is_t else ""), 13, 700 if i == 0 else None, C_GT if is_t else "#222", halo=False)
        if is_t:
            txt(o, x+296, y+20, f"rango = {i+1}", 12.5, 700, C_GT, "start", halo=False)
    txt(o, x+209, 135+len(ids)*36+14, "⋮", 16, None, "#666", halo=False)
    arrow(o, x+209, 312, x+209, 342, "#111")
    box(o, x+209, 378, 330, 56, [f"Output: identità del rank 1 ({ids[0]})", "nessuna soglia: il sistema risponde sempre"], GREY_BG, "#111", 12.5)
    arrow(o, x+209, 406, x+209, 438, "#111")
    if ok:
        box(o, x+209, 480, 360, 66, ["✓ identificazione corretta", "il template vero è al rank 1", "conta in CMS(k) per ogni k ≥ 1"], C_OK_BG, C_G_T, 12.5)
    else:
        box(o, x+209, 480, 360, 66, ["✗ identità sbagliata → False Rejection", f"il template vero è al rank {true_idx+1}", f"conta in CMS(k) solo per k ≥ {true_idx+1}"], C_FR_BG, C_FR_T, 12.5)

closed_panel(30, ["g4", "g9", "g2", "g7"], 0, "Caso A · template vero al rank 1", True)
closed_panel(470, ["g9", "g2", "g4", "g7"], 2, "Caso B · template vero al rank 3", False)
txt(o, W/2, H-18, "Tutti i probe appartengono a soggetti enrollati: nessun impostore, quindi nessuna False Acceptance", 12, None, "#555", halo=False)
save(o, '05_closed_set_flusso.svg')
```

<img src="./img/05_closed_set_flusso.svg" width="100%" style="align: center;"/>

Lo schema mostra due probe:

- **Caso A**: il template vero è al rank 1 → identificazione corretta.
- **Caso B**: il template vero è al rank 3 → il sistema restituisce l'identità sbagliata, quindi per la persona vera è un falso rifiuto. Il probe comunque "conta" nel CMS per ogni $k \ge 3$ (vedi 7.1).

#### Come si derivano RR e FRR

Per un probe sono possibili solo due eventi, mutuamente esclusivi ed esaustivi:

- successo: $rango(p_j) = 1$;
- errore: $rango(p_j) \ge 2$.

La frequenza relativa del successo su $P$ è il Recognition Rate:

$$
RR = \frac{|\{p_j : rango(p_j) = 1\}|}{|P|}
$$

Poiché l'errore è il complemento del successo:

$$
FRR_{closed} = \frac{|\{p_j : rango(p_j) \ge 2\}|}{|P|} = 1 - RR
$$

Non esiste una FAR perché non esistono impostori: non c'è nessuno da "accettare per errore". Nota sulla terminologia: in senso stretto l'errore del closed set è una *misidentification* (il sistema restituisce un'identità, ma sbagliata); lo si chiama False Rejection per coerenza con l'open set, dove lo stesso evento ha lo stesso nome.

#### Relazione con l'open set

Il closed set è l'open set **senza soglia** e senza impostori:

$$
CMS(k) = DIR(t_{min},\,k) \qquad \text{e quindi} \qquad RR = DIR(t_{min},\,1)
$$

dove $t_{min}$ è una soglia così bassa da accettare sempre il rank 1. Poiché alzare $t$ può solo ridurre il DIR, vale

$$
DIR(t,1) \le RR \quad \forall t
$$

cioè il Recognition Rate del closed set è un **limite superiore** per il DIR dell'open set. Per questo il closed set è utile come stima ottimistica (e facile da calcolare) delle prestazioni, ma non basta per valutare un sistema di watchlist.

| | Closed set | Open set |
|---|---|---|
| Il probe è sempre in gallery? | Sì | No |
| Soglia | No | Sì |
| Errori | solo rank 1 sbagliato | falso allarme, falso rifiuto, rank 1 sbagliato |
| Metrica | CMC, RR | DIR e FPIR al variare di $t$ |

### 7.1 Cumulative Match Characteristic (CMC) curve

**CMS (Cumulative Match Score) a rango k** = probabilità che l'identità corretta sia tra le prime k posizioni della lista ordinata.

- CMS a rango 1 = **Recognition Rate** (probabilità che sia esattamente al primo posto).
- La curva CMC raggiunge sempre probabilità 1 (perché tutti sono in gallery, quindi prima o poi compare).
- Area sotto la curva massima = dimensione della gallery; si può normalizzare dividendo per il massimo.
- Si usano spesso CMS a rango 1, 5, 10 come indicatori sintetici.

**La CMC come cumulata dell'istogramma dei rank.** Le barre sono $h(r)$, la linea è $CMS(k)=\sum_{r\le k}h(r)$. La curva parte da $RR=CMS(1)$ (non da zero), non decresce e arriva a 1. La diagonale tratteggiata è il sistema che ordina a caso. Sotto l'asse ci sono i conteggi dell'esempio: da qui si leggono direttamente $RR=0.62$, $CMS(5)=0.92$ e l'AUC normalizzata $0.893$.

```python
# =====================================================================
# 10 — Curva CMC: dall'istogramma dei rank alla cumulata  (richiede 01, 05)
# Produce: 06_cmc.svg
# =====================================================================
cnt = [620, 140, 80, 50, 30, 30, 20, 15, 10, 5]          # n. probe con rango = r
NPR, NG = sum(cnt), len(cnt)
hr = [c/NPR for c in cnt]
cms = [sum(hr[:k+1]) for k in range(NG)]
auc_n = sum(cms)/NG
auc_rnd = (NG+1)/(2*NG)

W, H = 920, 540
L, R, TOP, BOT = 100, 860, 100, 420
cw = (R-L)/NG
sxk = lambda k: L + (k-0.5)*cw
sy = lambda v: BOT - v*(BOT-TOP)

o = new_svg(W, H)
header(o, W, "Curva CMC (closed set)",
       f"N = {NG} template · |P| = {NPR} probe · barre = h(r) istogramma dei rank · linea = CMS(k) cumulata")
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#e5e7eb"/>')
    txt(o, L-8, sy(v)+4, f"{v:.1f}", 12, None, "#333", "end", halo=False)
frame(o, L, R, TOP, BOT, None, "Probabilità")
txt(o, (L+R)/2, BOT+34, "rango k", 13, None, "#222", halo=False)
for k in range(1, NG+1):
    o.append(f'<rect x="{sxk(k)-cw*0.25:.1f}" y="{sy(hr[k-1]):.1f}" width="{cw*0.5:.1f}" height="{BOT-sy(hr[k-1]):.1f}" fill="#93c5fd" fill-opacity="0.8"/>')
    txt(o, sxk(k), BOT+18, str(k), 12.5, None, "#333", halo=False)
    txt(o, sxk(k), BOT+58, str(cnt[k-1]), 12, None, "#333", halo=False)
    txt(o, sxk(k), BOT+78, f"{hr[k-1]:.3f}".rstrip("0").rstrip("."), 12, None, "#333", halo=False)
txt(o, L-8, BOT+58, "n. probe", 12, 700, "#555", "end", halo=False)
txt(o, L-8, BOT+78, "h(r)", 12, 700, "#555", "end", halo=False)

# area sotto CMC, baseline casuale, curva
pts = [(sxk(k), sy(cms[k-1])) for k in range(1, NG+1)]
o.append(f'<path d="M {sxk(1):.1f},{BOT} ' + " ".join(f"L {x:.1f},{y:.1f}" for x, y in pts) + f' L {sxk(NG):.1f},{BOT} Z" fill="{C_G}" fill-opacity="0.10"/>')
o.append(f'<path d="M {sxk(1):.1f},{sy(1/NG):.1f} L {sxk(NG):.1f},{sy(1):.1f}" fill="none" stroke="#6b7280" stroke-width="1.8" stroke-dasharray="6,5"/>')
txt(o, sxk(7)+6, sy(7/NG)+24, "ordinamento casuale: CMS(k) = k/N", 12, None, "#6b7280", "start")
o.append('<path d="M ' + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{C_G}" stroke-width="3.2"/>')
for k in range(1, NG+1):
    o.append(f'<circle cx="{sxk(k):.1f}" cy="{sy(cms[k-1]):.1f}" r="5" fill="{C_G}" stroke="#fff" stroke-width="1.5"/>')
    if k not in (1, 5):
        txt(o, sxk(k), sy(cms[k-1])-11, f"{cms[k-1]:.3f}".rstrip("0").rstrip("."), 12, None, C_G_T)
# punti evidenziati: RR = CMS(1) e CMS(5)
for k, name, dx, dy in [(1, f"RR = CMS(1) = {cms[0]:.2f}", 30, 36), (5, f"CMS(5) = {cms[4]:.2f}", -10, -18)]:
    o.append(f'<circle cx="{sxk(k):.1f}" cy="{sy(cms[k-1]):.1f}" r="9" fill="none" stroke="#111" stroke-width="2"/>')
    txt(o, sxk(k)+dx, sy(cms[k-1])+dy, name, 13, 700, "#111", "start" if dx > 0 else "end")
txt(o, sxk(6)+10, sy(0.40), f"AUC normalizzata = Σ CMS(k) / N = {auc_n:.3f}", 13.5, 700, "#1d4ed8", "start")
txt(o, sxk(6)+10, sy(0.40)+18, f"(sistema casuale: {auc_rnd:.2f} · sistema perfetto: 1)", 12, None, "#444", "start")
txt(o, L+6, TOP+18, "↖ migliore", 13, 700, C_G_T, "start", halo=False)
save(o, '06_cmc.svg')
```

<img src="./img/06_cmc.svg" width="100%" style="align: center;"/>

#### Come si deriva la CMC

**1. Istogramma dei rank.** Per ogni probe si registra il rank del template vero. La frazione di probe con rango esattamente $r$ è

$$
h(r) = P(rango = r) = \frac{|\{p_j : rango(p_j) = r\}|}{|P|}, \qquad \sum_{r=1}^{N} h(r) = 1
$$

La somma vale 1 perché ogni probe ha un rango in $\{1, \dots, N\}$.

**2. Cumulata.** Il CMS a rango $k$ è la frazione di probe con rango **al più** $k$:

$$
CMS(k) = \frac{|\{p_j : rango(p_j) \le k\}|}{|P|} = \sum_{r=1}^{k} h(r)
$$

La CMC è quindi la **funzione di distribuzione cumulata** del rango del template vero.

**3. Proprietà (seguono dalla definizione).**

- $CMS(1) = h(1) = RR$: la curva parte dal Recognition Rate, non da zero.
- **Non decrescente**: l'insieme $\{rango \le k\}$ può solo crescere con $k$, quindi $CMS(k+1) \ge CMS(k)$.
- $CMS(N) = 1$: nel closed set il template vero è sempre in gallery, quindi ha rango $\le N$. Nell'open set questo non vale, perché gli impostori non hanno un template vero.
- Un sistema che ordina a caso ha $CMS(k) = k/N$ (retta), che serve da riferimento minimo.

**4. Area sotto la curva.** Sommando su tutti i rank il massimo possibile si ottiene $CMS(k) = 1$ per ogni $k$, cioè un'area pari a $N$. Dividendo per $N$ si ha l'area normalizzata, compresa tra circa $0.5$ (sistema casuale) e $1$ (sistema perfetto):

$$
AUC_{norm} = \frac{1}{N} \sum_{k=1}^{N} CMS(k)
$$

Per un sistema casuale $AUC_{norm} = \frac{1}{N}\sum_{k=1}^{N}\frac{k}{N} = \frac{N+1}{2N}$, che per $N=10$ vale $0.55$.

#### Esempio numerico (quello della figura)

Gallery con $N = 10$ template e $|P| = 1000$ probe. I rank osservati sono:

| rango $r$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| n. probe | 620 | 140 | 80 | 50 | 30 | 30 | 20 | 15 | 10 | 5 |
| $h(r)$ | 0.62 | 0.14 | 0.08 | 0.05 | 0.03 | 0.03 | 0.02 | 0.015 | 0.01 | 0.005 |
| $CMS(r)$ | 0.62 | 0.76 | 0.84 | 0.89 | 0.92 | 0.95 | 0.97 | 0.985 | 0.995 | 1 |

Quindi $RR = CMS(1) = 0.62$, $FRR_{closed} = 0.38$, $CMS(5) = 0.92$ e $AUC_{norm} = 8.93 / 10 = 0.893$.

**Come si leggono gli indicatori sintetici.** $CMS(1)$ dice quanto spesso il sistema "ci azzecca" da solo. $CMS(5)$ e $CMS(10)$ sono interessanti quando l'output è una **shortlist** rivista da un operatore: con $CMS(5) = 0.92$ il template giusto è tra i primi cinque candidati nel 92% dei casi, anche se solo nel 62% è il primo.

### 7.2 Identificazione vs Re-Identificazione

- **Identificazione**: il sistema riceve una probe e deve stabilire **chi è**, confrontandola con l'intera gallery. Orizzonte temporale medio/lungo; può essere open o closed set; l'output è un'**identità precisa** (un nome/ID).
- **Re-Identificazione (Re-ID)**: il sistema deve ritrovare **la stessa persona** in immagini/frame diversi (tipicamente provenienti da telecamere diverse), su un orizzonte temporale breve, **senza necessariamente conoscerne l'identità reale**. Non cerca un nome, ma una corrispondenza tra apparizioni della stessa persona — tipica della videosorveglianza e del tracking multi-camera; è tipicamente closed set, e l'output è "stessa persona / persona diversa" anziché un'identità.
- La Re-ID si valuta con la **CMC** (quando c'è una sola immagine di gallery per query) o con la **mAP (mean Average Precision)**, quando più immagini di gallery corrispondono alla stessa query.
**Stessa pipeline, due domande.** A sinistra il probe è confrontato con persone *enrolled* e l'output è un'identità (ID 42). A destra non c'è nessun nome: si cerca soltanto, tra le apparizioni di un'altra telecamera, quella che corrisponde alla query. Per questo cambiano orizzonte temporale, tipo di gallery e metriche (CMC per un solo match, mAP per più match).

```python
# =====================================================================
# 11 — Identificazione vs Re-identificazione  (richiede 01, 05)
# Produce: 07_identificazione_vs_reid.svg
# =====================================================================
def person(o, cx, cy, color, s=1.0):
    o.append(f'<circle cx="{cx:.1f}" cy="{cy-17*s:.1f}" r="{7.5*s:.1f}" fill="{color}"/>')
    o.append(f'<rect x="{cx-11*s:.1f}" y="{cy-7*s:.1f}" width="{22*s:.1f}" height="{28*s:.1f}" rx="{7*s:.1f}" fill="{color}"/>')

W, H = 920, 600
o = new_svg(W, H)
header(o, W, "Identificazione vs Re-identificazione", "Stessa pipeline di confronto, domande e output diversi")
for x0, ttl, sub in [(20, "Identificazione", "«Chi è questa persona?»"), (470, "Re-identificazione", "«È la stessa persona vista altrove?»")]:
    rect(o, x0, 76, 430, 506, "#fff", "#e5e7eb", 1.5, 14)
    txt(o, x0+215, 106, ttl, 18, 700, "#111", halo=False)
    txt(o, x0+215, 126, sub, 13, None, "#555", halo=False)

# --- identificazione ---
person(o, 68, 235, C_I, 1.3); txt(o, 68, 275, "probe", 12, None, "#333", halo=False)
arrow(o, 100, 225, 150, 225); txt(o, 125, 214, "1:N", 11.5, 700, "#111")
rect(o, 150, 150, 180, 190, BLU_BG, C_I, 1.8, 10)
txt(o, 240, 168, "gallery (enrolled)", 12.5, 700, C_I, halo=False)
ids_g = ["ID 07", "ID 42", "ID 18", "ID 31", "ID 55", "ID 09"]
cols_g = ["#9ca3af", C_G, "#9ca3af", "#9ca3af", "#9ca3af", "#9ca3af"]
for i, (nm, c) in enumerate(zip(ids_g, cols_g)):
    gx, gy = 190 + (i % 3)*52, 215 + (i // 3)*68
    person(o, gx, gy, c, 0.85); txt(o, gx, gy+36, nm, 10.5, 700 if c == C_G else None, "#333", halo=False)
arrow(o, 330, 235, 360, 235)
box(o, 402, 235, 76, 56, ["Output", "ID 42"], C_OK_BG, C_G_T, 12)
for i, s in enumerate(["• confronto con persone note (enrollment)", "• orizzonte temporale medio/lungo",
                       "• open set oppure closed set", "• output: un'identità precisa (ID)", "• metriche: DIR / FPIR, CMC"]):
    txt(o, 44, 372+i*26, s, 13, None, "#222", "start", halo=False)

# --- re-identificazione ---
rect(o, 490, 150, 110, 150, GREY_BG, "#6b7280", 1.8, 10)
txt(o, 545, 168, "Camera 1", 12.5, 700, "#374151", halo=False)
person(o, 545, 232, C_I, 1.4); txt(o, 545, 286, "query", 12, None, "#333", halo=False)
arrow(o, 602, 225, 640, 225)
rect(o, 642, 150, 238, 170, GREY_BG, "#6b7280", 1.8, 10)
txt(o, 761, 168, "Camera 2 · candidati", 12.5, 700, "#374151", halo=False)
cand = [("#f59e0b", "?"), (C_I, "stessa ✓"), ("#a855f7", "?")]
for i, (c, lab) in enumerate(cand):
    gx = 685 + i*76
    if c == C_I:
        rect(o, gx-30, 190, 60, 106, C_OK_BG, C_G, 2.2, 8)
    person(o, gx, 235, c, 1.1)
    txt(o, gx, 282, lab, 11.5, 700 if c == C_I else None, C_G_T if c == C_I else "#555", halo=False)
arrow(o, 761, 322, 761, 345)
box(o, 700, 378, 360, 46, ["Output: stessa persona / persona diversa", "nessun nome: solo corrispondenza tra apparizioni"], C_OK_BG, C_G_T, 12)
for i, s in enumerate(["• confronto tra apparizioni di telecamere diverse", "• orizzonte breve, identità reale non necessaria",
                       "• tipicamente closed set", "• output: stessa persona sì/no", "• metriche: CMC (1 match) · mAP (più match)"]):
    txt(o, 494, 440+i*26, s, 13, None, "#222", "start", halo=False)
save(o, '07_identificazione_vs_reid.svg')
```

<img src="./img/07_identificazione_vs_reid.svg" width="100%" style="align: center;"/>

#### Perché la mAP e non solo la CMC

Nella CMC conta **solo la posizione del primo match corretto**: se per una query il primo è al rank 1, il resto della lista è irrilevante. Questo va bene quando c'è una sola immagine corretta in gallery. In Re-ID, invece, la stessa persona compare di solito in **più** frame di gallery, e un buon sistema deve metterli tutti in alto: la CMC non lo misura, la mAP sì.

#### Come si deriva la mAP

**1. Precisione al rango $k$** (per una query $q$): frazione di immagini corrette tra le prime $k$.

$$
P(k) = \frac{\text{n. di immagini corrette nei primi } k}{k}
$$

**2. Average Precision.** Si media $P(k)$ **solo nei rank dove compare un'immagine corretta**, usando $rel(k) = 1$ se l'immagine al rank $k$ è la stessa persona e $0$ altrimenti:

$$
AP(q) = \frac{1}{|R_q|} \sum_{k=1}^{n} P(k)\, rel(k)
$$

dove $R_q$ è l'insieme delle immagini corrette per $q$ e $n$ la lunghezza della lista. Il fattore $\frac{1}{|R_q|}$ normalizza: $AP = 1$ solo se **tutte** le immagini corrette sono ai primi posti, e ogni corretta messa in basso abbassa il valore.

**3. Media sulle query.**

$$
mAP = \frac{1}{|Q|} \sum_{q \in Q} AP(q)
$$

**Average Precision passo per passo.** In alto i risultati ordinati (corretti ai rank 1, 3, 6). Al centro $P(k)$ a ogni rank: solo le barre verdi, cioè quelle dove compare un'immagine corretta, entrano nella media. In basso lo stesso insieme di immagini corrette messo in cima o in fondo alla lista: l'AP passa da 1.00 a 0.28, mentre la CMC non se ne accorgerebbe.

```python
# =====================================================================
# 12 — Average Precision  (richiede 01, 05)
# Produce: 08_average_precision.svg
# =====================================================================
def prec_at(rel):
    hits, out = 0, []
    for k, r in enumerate(rel, 1):
        hits += r
        out.append(hits/k)
    return out

def ap_of(rel):
    p = prec_at(rel)
    return sum(pk for pk, r in zip(p, rel) if r)/sum(rel)

REL = [1, 0, 1, 0, 0, 1, 0, 0]                 # rank 1, 3, 6 corretti (esempio del testo)
P = prec_at(REL); AP = ap_of(REL)
n = len(REL)

W, H = 920, 700
L, R = 120, 860
cw = (R-L)/n
sxk = lambda k: L + (k-0.5)*cw
o = new_svg(W, H)
header(o, W, "Average Precision (AP) per una query", "Si media P(k) solo nei rank dove compare un'immagine corretta")

# riga dei risultati
txt(o, L-14, 118, "risultati", 12.5, 700, "#555", "end", halo=False)
for k in range(1, n+1):
    ok = REL[k-1]
    rect(o, sxk(k)-26, 90, 52, 52, C_OK_BG if ok else GREY_BG, C_G if ok else "#9ca3af", 2.2 if ok else 1.2, 8)
    txt(o, sxk(k), 123, "✓" if ok else "✗", 22, 700, C_G_T if ok else "#9ca3af", halo=False)
    txt(o, sxk(k), 80, f"rank {k}", 12.5, 700, "#333", halo=False)

# precisione P(k)
BT, BB = 215, 395
sy = lambda v: BB - v*(BB-BT)
txt(o, L-14, BT-12, "P(k)", 13, 700, "#555", "end", halo=False)
for v in [0, 0.5, 1.0]:
    o.append(f'<line x1="{L-6}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#e5e7eb"/>')
    txt(o, L-12, sy(v)+4, f"{v:.1f}", 12, None, "#333", "end", halo=False)
hits = 0
for k in range(1, n+1):
    ok = REL[k-1]; hits += ok
    col = C_G if ok else "#d1d5db"
    o.append(f'<rect x="{sxk(k)-26:.1f}" y="{sy(P[k-1]):.1f}" width="52" height="{BB-sy(P[k-1]):.1f}" fill="{col}" fill-opacity="{0.9 if ok else 0.6}"/>')
    txt(o, sxk(k), sy(P[k-1])-8, f"{hits}/{k} = {P[k-1]:.2f}", 12, 700 if ok else None, C_G_T if ok else "#777")
txt(o, L-14, BB+22, "verde = contano nell'AP · grigio = P(k) in un rank sbagliato, ignorata", 11.5, None, "#555", "start", halo=False)

# formula
terms = " + ".join(f"{P[k]:.2f}" for k in range(n) if REL[k])
box(o, W/2, 455, 700, 46, [f"AP = ( {terms} ) / {sum(REL)} = {AP:.3f}", "|R_q| = 3 immagini corrette → il fattore 1/|R_q| penalizza ogni corretta messa in basso"], "#eff6ff", C_I, 13)

# confronto con casi estremi
txt(o, L-14, 520, "Stessa query, risultati diversi", 13, 700, "#111", "start", halo=False)
for j, (name, rel) in enumerate([("corrette in cima (rank 1, 2, 3)", [1, 1, 1, 0, 0, 0, 0, 0]),
                                 ("esempio (rank 1, 3, 6)", REL),
                                 ("corrette in fondo (rank 6, 7, 8)", [0, 0, 0, 0, 0, 1, 1, 1])]):
    y = 548 + j*44
    for k in range(1, n+1):
        ok = rel[k-1]
        rect(o, L+ (k-1)*34, y, 28, 28, C_OK_BG if ok else GREY_BG, C_G if ok else "#9ca3af", 1.6 if ok else 1, 5)
    txt(o, L+n*34+16, y+19, f"AP = {ap_of(rel):.2f}", 14, 700, "#1d4ed8", "start", halo=False)
    txt(o, L+n*34+110, y+19, name, 12.5, None, "#444", "start", halo=False)
save(o, '08_average_precision.svg')
```

<img src="./img/08_average_precision.svg" width="100%" style="align: center;"/>

**Esempio (quello della figura).** Le immagini corrette sono ai rank 1, 3 e 6, quindi $|R_q| = 3$ e

$$
AP = \frac{1}{3}\left(\frac{1}{1} + \frac{2}{3} + \frac{3}{6}\right) \approx 0.72
$$

Se le tre corrette fossero ai rank 1, 2, 3 si avrebbe $AP = \frac{1}{3}(1 + 1 + 1) = 1$; se fossero ai rank 6, 7, 8, $AP = \frac{1}{3}(\frac{1}{6} + \frac{2}{7} + \frac{3}{8}) \approx 0.28$.

**CMC e mAP a confronto.**

| | CMC / CMS($k$) | mAP |
|---|---|---|
| Cosa guarda | posizione del **primo** match corretto | posizione di **tutti** i match corretti |
| Quando usarla | una sola immagine corretta in gallery per query | più immagini corrette per query |
| Risposta | "il match giusto è nei primi $k$?" | "quanto in alto stanno tutti i match giusti?" |

### 7.3 Riepilogo rapido

- **Closed set**: nessuna soglia, nessun impostore; l'unico errore è il rank 1 sbagliato.
- $RR = CMS(1)$ e $FRR_{closed} = 1 - RR$.
- $CMS(k) = \sum_{r=1}^{k} h(r)$: la CMC è la cumulata dell'istogramma dei rank; parte da $RR$, non decresce e arriva a 1.
- $DIR(t,1) \le RR$: il closed set è un limite superiore per l'open set.
- **Re-ID** = ritrovare la stessa persona su più telecamere, senza nome; si valuta con CMC (un solo match) o mAP (più match).

---

## 8. Progettazione sperimentale (Offline Evaluation)

La valutazione statistica avviene **offline**, su dataset con **ground truth** noto (`id(template)` restituisce l'identità vera). In produzione questo non è disponibile (es. attacco "zero effort").

### 8.1 Tre scelte di partizionamento del dataset

1. **Training vs Testing (TR/TS)**
   - Necessario per approcci machine learning.
   - **Nessuna sovrapposizione** tra campioni di training e testing.
   - Il training deve includere campioni di qualità/condizioni varie per garantire **generalizzabilità**.
   - Partizionamento per **soggetti** (subject mai visti = never seen subjects) o per **campioni** (stesso soggetto in TR e TS ma campioni diversi).

2. **Probe vs Gallery (P/G)**
   - Scelta 1: campioni migliori in gallery, condizioni variabili nel probe.
   - Scelta 2: condizioni variabili anche in gallery (per riconoscere meglio in diverse condizioni).
   - Sempre nessuna sovrapposizione tra probe e gallery.

3. **Open set vs Closed set**
   - Closed set: tutte le probe appartengono a soggetti in gallery.
   - Open set: probe = PG ∪ PN (soggetti in gallery + soggetti fuori gallery).
   - Non influisce sulla verifica (dipende solo dal claim), ma è cruciale nell'identificazione open set.

**Le tre scelte.** Training vs Testing (per soggetti o per campioni), Probe vs Gallery (gallery con il campione migliore o variabile), closed vs open set (con probe $P_N$ fuori gallery).

```python
# =====================================================================
# N10 — Partizionamento del dataset  (richiede 01, 05)
# =====================================================================
W, H = 920, 890
o = new_svg(W, H)
header(o, W, "Progettazione sperimentale: le tre scelte di partizionamento",
       "Ogni cella = un campione (riga = soggetto, colonna = campione) · nessuna sovrapposizione tra insiemi")
C_TR, C_TS, C_GAL, C_PRB, C_UNK = "#93c5fd", "#fdba74", "#86efac", "#c4b5fd", "#fca5a5"
def cell_grid(x, y, rows, cols, cs, fn, rlab=None, clab=None):
    for r in range(rows):
        if rlab: txt(o, x-8, y+r*cs+cs/2+4, rlab[r], 11, None, "#444", "end", halo=False)
        for c in range(cols):
            f, t = fn(r, c)
            mat_cell(o, x+c*cs, y+r*cs, cs, cs, f if f else "#fff", t, 10, "#111", False, "#ffffff" if f else "#d1d5db", 1.5)
    if clab:
        for c in range(cols): txt(o, x+c*cs+cs/2, y-6, clab[c], 10, None, "#444", halo=False)
def legend(x, y, items):
    for i, (c, t) in enumerate(items):
        o.append(f'<rect x="{x}" y="{y+i*24-11}" width="16" height="16" rx="3" fill="{c}"/>')
        txt(o, x+24, y+i*24+2, t, 12, None, "#222", "start", halo=False)
subs = [f"S{i+1}" for i in range(6)]
# ---- 1 TR/TS ----
txt(o, 30, 90, "1 · Training vs Testing", 15, 700, "#111", "start", halo=False)
txt(o, 150, 118, "per SOGGETTI", 12.5, 700, "#1d4ed8", "start", halo=False)
cell_grid(110, 130, 6, 6, 26, lambda r, c: (C_TR if r < 4 else C_TS, None), subs)
txt(o, 110, 306, "soggetti del TS mai visti in training", 11.5, None, "#555", "start", halo=False)
txt(o, 480, 118, "per CAMPIONI", 12.5, 700, "#1d4ed8", "start", halo=False)
cell_grid(440, 130, 6, 6, 26, lambda r, c: (C_TR if c < 4 else C_TS, None), subs)
txt(o, 440, 306, "stessi soggetti, campioni diversi", 11.5, None, "#555", "start", halo=False)
legend(690, 160, [(C_TR, "Training (TR)"), (C_TS, "Testing (TS)")])
txt(o, 690, 225, "per il ML: TR vario per", 11.5, None, "#555", "start", halo=False)
txt(o, 690, 241, "qualità e condizioni", 11.5, None, "#555", "start", halo=False)
# ---- 2 P/G ----
txt(o, 30, 350, "2 · Probe vs Gallery", 15, 700, "#111", "start", halo=False)
cond = ["ideale", "luce", "posa", "espr.", "occh.", "rumore"]
txt(o, 150, 378, "Scelta 1: gallery = campione migliore", 12.5, 700, "#1d4ed8", "start", halo=False)
cell_grid(110, 402, 6, 6, 30, lambda r, c: (C_GAL if c == 0 else C_PRB, "G" if c == 0 else "P"), subs, cond)
txt(o, 110, 598, "probe in condizioni variabili", 11.5, None, "#555", "start", halo=False)
txt(o, 500, 378, "Scelta 2: gallery varia", 12.5, 700, "#1d4ed8", "start", halo=False)
cell_grid(440, 402, 6, 6, 30, lambda r, c: (C_GAL if c in (0, 3) else C_PRB, "G" if c in (0, 3) else "P"), subs, cond)
txt(o, 440, 598, "riconosce meglio in condizioni diverse", 11.5, None, "#555", "start", halo=False)
legend(690, 440, [(C_GAL, "G · gallery"), (C_PRB, "P · probe")])
txt(o, 690, 505, "mai lo stesso campione", 11.5, None, "#555", "start", halo=False)
txt(o, 690, 521, "in probe e gallery", 11.5, None, "#555", "start", halo=False)
# ---- 3 closed/open ----
txt(o, 30, 640, "3 · Closed set vs Open set", 15, 700, "#111", "start", halo=False)
names = ["A", "B", "C", "D", "E", "F"]
txt(o, 150, 656, "CLOSED: tutti i probe in gallery", 12.5, 700, "#1d4ed8", "start", halo=False)
cell_grid(110, 694, 4, 4, 26, lambda r, c: (C_GAL if c == 0 else C_PRB, "g" if c == 0 else "p"), names[:4], ["gall.", "p1", "p2", "p3"])
txt(o, 110, 818, "P = P_G", 11.5, 700, C_G_T, "start", halo=False)
txt(o, 470, 656, "OPEN: P = P_G ∪ P_N", 12.5, 700, "#1d4ed8", "start", halo=False)
cell_grid(440, 694, 6, 4, 26, lambda r, c: (C_GAL if (c == 0 and r < 4) else (C_PRB if r < 4 else (None if c == 0 else C_UNK)), ("g" if c == 0 else "p") if (r < 4 or c > 0) else "—"), names, ["gall.", "p1", "p2", "p3"])
txt(o, 440, 868, "E, F non sono in gallery: impostori (P_N)", 11.5, 700, C_FA_T, "start", halo=False)
legend(690, 712, [(C_GAL, "g · in gallery"), (C_PRB, "p · probe di un iscritto (P_G)"), (C_UNK, "p · probe di un ignoto (P_N)")])
txt(o, 690, 802, "non incide sulla verifica,", 11.5, None, "#555", "start", halo=False)
txt(o, 690, 818, "è cruciale nell'open set", 11.5, None, "#555", "start", halo=False)
save(o, 'n10_partizionamento_dataset.svg')
```
<img src="./img/n10_partizionamento_dataset.svg" alt="Partizionamento del dataset" style="display:block; margin:1.5em auto; max-width:100%;">

### 8.2 K-Fold Cross Validation

Il dataset è diviso in k sottoinsiemi; si ripete k volte l'addestramento usando k−1 sottoinsiemi come training e 1 come test, ruotando. L'errore finale è la **media** sui k trial. Tipicamente **k = 5 o 10**.

**K-fold con k = 5.** A ogni trial un fold diverso è il test e gli altri quattro sono il training; l'errore finale è la media dei k errori.

```python
# =====================================================================
# N10b — K-Fold  (richiede 01, 05)
# =====================================================================
K = 5
W, H = 920, 470
o = new_svg(W, H)
header(o, W, f"K-Fold Cross Validation (k = {K})", "Il dataset è diviso in k sottoinsiemi: a turno uno è il test, gli altri k − 1 sono il training")
X0, BW, BH, Y0, GAP = 150, 100, 40, 110, 56
for c in range(K):
    txt(o, X0+c*BW+BW/2, Y0-10, f"fold {c+1}", 12, 700, "#555", halo=False)
errs = [0.08, 0.11, 0.07, 0.10, 0.09]
for r in range(K):
    y = Y0+r*GAP
    txt(o, X0-14, y+BH/2+4, f"trial {r+1}", 12.5, 700, "#222", "end", halo=False)
    for c in range(K):
        test = (c == r)
        rect(o, X0+c*BW+2, y, BW-4, BH, "#fed7aa" if test else "#bfdbfe", "#ea580c" if test else "#60a5fa", 1.8 if test else 1.2, 6)
        txt(o, X0+c*BW+BW/2, y+BH/2+4, "test" if test else "training", 12, 700 if test else None, "#9a3412" if test else "#1e3a8a", halo=False)
    txt(o, X0+K*BW+20, y+BH/2+5, f"errore e{r+1} = {errs[r]:.2f}", 12.5, None, "#222", "start", halo=False)
yb = Y0+K*GAP+16
o.append(f'<line x1="{X0+K*BW+14}" y1="{Y0}" x2="{X0+K*BW+14}" y2="{yb-14}" stroke="#9ca3af"/>')
box(o, 460, yb+30, 560, 46, [f"errore finale = media sui k trial = {sum(errs)/K:.3f}", "tipicamente k = 5 o k = 10 · il training non contiene mai il fold di test"], C_GT_BG, C_GT, 12.5)
save(o, 'n10b_kfold.svg')
```
<img src="./img/n10b_kfold.svg" alt="K-fold" style="display:block; margin:1.5em auto; max-width:100%;">

---

## 9. Strategia All-Against-All

Invece di simulare **una sola rivendicazione d'identità per ogni probe** (es. "sono Mario"), si valutano **tutte le coppie possibili**: ogni probe $i$ viene confrontato con ogni campione $j$ della galleria, come se potesse dichiarare *qualsiasi* identità.

Grazie al **ground truth** (si conosce $\mathrm{label}(\cdot)$ di ogni campione) non serve ripetere gli esperimenti: si calcola **una sola volta** la matrice $M$ delle distanze (o similarità) e poi ogni cella viene classificata come *genuina* o *impostore*. Ogni riga della matrice rappresenta quindi **più esperimenti**.

### 9.1 Matrice delle distanze

$$
M \in \mathbb{R}^{|P|\times|G|}, \qquad M[i,j] = d\big(\text{probe}_i,\ \text{gallery}_j\big)
$$

$$
\text{la cella } (i,j) \text{ è }
\begin{cases}
\textbf{genuina} & \mathrm{label}(i)=\mathrm{label}(j)\\
\textbf{impostore} & \mathrm{label}(i)\neq\mathrm{label}(j)
\end{cases}
$$

**Anatomia della matrice.** Con $N=3$ soggetti e $S=2$ template ciascuno, la matrice è $6\times6$. I blocchi viola sulla diagonale contengono le celle genuine, tutto il resto è impostore; la diagonale vera e propria ($i=j$) è lo stesso campione confrontato con sé stesso e viene esclusa. Ogni riga è già un insieme di esperimenti: 1 genuino e 4 impostori.

```python
# =====================================================================
# 13 — All-against-all: struttura della matrice  (richiede 01, 05)
# Produce: fig1_matrice.svg
# =====================================================================
N_, S_ = 3, 2
LAB = [f"{c}{s}" for c in "ABC" for s in range(1, S_+1)]      # A1 A2 B1 B2 C1 C2
nG = N_*S_
CS, X0, Y0 = 62, 150, 150
C_GEN_BG = "#ede9fe"

W, H = 920, 560
o = new_svg(W, H)
header(o, W, "Matrice all-against-all", f"N = {N_} soggetti · S = {S_} template per soggetto · |G| = S·N = {nG}")
txt(o, X0+nG*CS/2, Y0-46, "galleria j  →", 13, 700, "#555", halo=False)
o.append(f'<text transform="translate(52,{Y0+nG*CS/2}) rotate(-90)" text-anchor="middle" font-size="13" font-weight="700" fill="#555">probe i  →</text>')
for k, nm in enumerate(LAB):
    txt(o, X0+k*CS+CS/2, Y0-12, nm, 13, 700, "#222", halo=False)
    txt(o, X0-12, Y0+k*CS+CS/2+5, nm, 13, 700, "#222", "end", halo=False)
for i in range(nG):
    for j in range(nG):
        if i == j:      mat_cell(o, X0+j*CS, Y0+i*CS, CS, CS, "#d1d5db", "×", 16, "#6b7280")
        elif i//S_ == j//S_: mat_cell(o, X0+j*CS, Y0+i*CS, CS, CS, C_GEN_BG, "G", 15, C_GT, True)
        else:           mat_cell(o, X0+j*CS, Y0+i*CS, CS, CS, GREY_BG, "imp", 12, "#6b7280")
for b in range(N_):                                          # blocchi genuini sulla diagonale
    o.append(f'<rect x="{X0+b*S_*CS}" y="{Y0+b*S_*CS}" width="{S_*CS}" height="{S_*CS}" fill="none" stroke="{C_GT}" stroke-width="3"/>')
yr = Y0 + 1*CS                                               # riga evidenziata (A2)
o.append(f'<rect x="{X0-4}" y="{yr-3}" width="{nG*CS+8}" height="{CS+6}" rx="6" fill="none" stroke="#111" stroke-width="2.5" stroke-dasharray="6,4"/>')

# pannello a destra
xr = 570
txt(o, xr, 150, "M[i,j] = d(probe_i, gallery_j)", 15, 700, "#111", "start", halo=False)
leg = [(C_GEN_BG, C_GT, "G", "cella genuina: label(i) = label(j)"), (GREY_BG, "#6b7280", "imp", "cella impostore: label(i) ≠ label(j)"),
       ("#d1d5db", "#6b7280", "×", "diagonale i = j: stesso campione, esclusa")]
for q, (bg, fg, ch, tx) in enumerate(leg):
    mat_cell(o, xr, 175+q*36, 34, 26, bg, ch, 12, fg, True)
    txt(o, xr+46, 193+q*36, tx, 12.5, None, "#222", "start", halo=False)
txt(o, xr, 310, "Una riga = più esperimenti", 13.5, 700, "#111", "start", halo=False)
txt(o, xr+196, 310, "(riga tratteggiata)", 12, None, "#555", "start", halo=False)
txt(o, xr, 334, f"• celle genuine: S − 1 = {S_-1}", 13, None, "#222", "start", halo=False)
txt(o, xr, 356, f"• celle impostore: (N − 1)·S = {(N_-1)*S_}", 13, None, "#222", "start", halo=False)
txt(o, xr, 400, "Totali sulla matrice (diagonale esclusa)", 13.5, 700, "#111", "start", halo=False)
txt(o, xr, 424, f"TG = |G|·(S − 1) = {nG}·{S_-1} = {nG*(S_-1)}", 13, None, "#222", "start", halo=False)
txt(o, xr, 446, f"TI = |G|·(N − 1)·S = {nG}·{N_-1}·{S_} = {nG*(N_-1)*S_}", 13, None, "#222", "start", halo=False)
txt(o, xr, 488, "Gli impostori sono molto più numerosi dei genuini:", 12.5, None, "#555", "start", halo=False)
txt(o, xr, 506, "il sistema viene «stressato» di più.", 12.5, None, "#555", "start", halo=False)
save(o, 'fig1_matrice.svg')
```

<img src="./img/fig1_matrice.svg" alt="Struttura della matrice all-against-all" style="display:block; margin:1.5em auto; max-width:100%;">

**Vantaggi**

- Facile da programmare; calcola una "media" su tutte le possibili distribuzioni genuino/impostore.
- Gli impostori sono molto più numerosi dei genuini (rapporto $\frac{(N-1)S}{S-1}$ per riga) → si può **stressare molto** il sistema.

**Svantaggi**

- Tempo computazionale elevato: $O(|G|^2)$ confronti.
- Non permette di analizzare distribuzioni genuino/impostore *specifiche* (es. solo certi soggetti).
- Non adatto se il dataset ha **sessioni** temporalmente separate: campioni della stessa sessione sono più simili tra loro, quindi i genuini risultano troppo vicini e le prestazioni **ottimisticamente falsate**. In tal caso si usa la variante **All-Against-All Probe vs Gallery** (§10): una sessione = galleria, un'altra = probe.

### 9.2 Notazione

| Simbolo | Significato |
|---|---|
| $N$ | numero di soggetti |
| $S$ | numero di template per soggetto |
| $\lvert G\rvert = S\cdot N$ | cardinalità della galleria (campioni totali) |
| $i$ | indice di riga (probe) |
| $j$ | indice di colonna (galleria) |
| $\mathrm{label}(i),\ \mathrm{label}(j)$ | identità associate |
| $t$ | soglia di decisione: si **accetta** se $M[i,j]\le t$ |
| $TG,\ TI$ | numero totale di tentativi genuini / impostori |
| $GA,\ FR,\ FA,\ GR$ | genuine accept / false reject / false accept / genuine reject |

### 9.3 Come la soglia classifica le celle

Fissata $t$, ogni esperimento ha uno di quattro esiti:

$$
\begin{array}{c|cc}
 & M[i,j]\le t \ (\text{accettato}) & M[i,j]> t \ (\text{rifiutato})\\ \hline
\text{genuino} & GA & FR\\
\text{impostore} & FA & GR
\end{array}
$$

con le relazioni

$$
GA+FR = TG, \qquad FA+GR = TI
$$

$$
GAR(t)=\frac{GA}{TG},\quad FRR(t)=\frac{FR}{TG}=1-GAR(t),\quad
FAR(t)=\frac{FA}{TI},\quad GRR(t)=\frac{GR}{TI}=1-FAR(t)
$$

**La soglia taglia le due distribuzioni di distanza.** Qui si lavora con distanze, quindi si accetta a sinistra di $t$. Le quattro aree sono i quattro esiti GA, FR, FA, GR, e i tassi si ottengono dividendo per il numero di tentativi della *propria* categoria (TG per i genuini, TI per gli impostori), non per il totale.

```python
# =====================================================================
# 14 — Soglia sulle distanze: GA, FR, FA, GR  (richiede 01, 05)
# Produce: fig2_soglia.svg
# =====================================================================
MU_GD, SD_GD = 0.30, 0.10      # distanze genuine
MU_ID, SD_ID = 0.62, 0.13      # distanze impostori
T_D = 0.40                     # si accetta se d ≤ t
ndg, ndi = NormalDist(MU_GD, SD_GD), NormalDist(MU_ID, SD_ID)
GAR_D, FRR_D = ndg.cdf(T_D), 1-ndg.cdf(T_D)
FAR_D, GRR_D = ndi.cdf(T_D), 1-ndi.cdf(T_D)

W, H = 920, 580
L, R, TOP, BOT = 70, 880, 100, 420
XMIN, XMAX, YMAX = -0.1, 1.1, 4.4
sx = lambda x: L + (x-XMIN)/(XMAX-XMIN)*(R-L)
sy = lambda y: BOT - y/YMAX*(BOT-TOP)
o = new_svg(W, H)
header(o, W, "Genuini e impostori: la soglia classifica le celle di M",
       f"Distanze: genuini ~ N({MU_GD}, {SD_GD}²) · impostori ~ N({MU_ID}, {SD_ID}²) · si accetta se M[i,j] ≤ t = {T_D}")
o.append(f'<path d="{g_area(MU_GD,SD_GD,XMIN,T_D,sx,sy,BOT)}" fill="{C_G}" fill-opacity="0.35"/>')
o.append(f'<path d="{g_area(MU_GD,SD_GD,T_D,XMAX,sx,sy,BOT)}" fill="{C_FR}" fill-opacity="0.85"/>')
o.append(f'<path d="{g_area(MU_ID,SD_ID,XMIN,T_D,sx,sy,BOT)}" fill="{C_FA}" fill-opacity="0.85"/>')
o.append(f'<path d="{g_area(MU_ID,SD_ID,T_D,XMAX,sx,sy,BOT)}" fill="{C_I}" fill-opacity="0.18"/>')
o.append(f'<path d="{g_line(MU_GD,SD_GD,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_G}" stroke-width="2.5"/>')
o.append(f'<path d="{g_line(MU_ID,SD_ID,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_I}" stroke-width="2.5"/>')
frame(o, L, R, TOP, BOT, "distanza M[i,j]", "Densità")
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{sx(v):.1f}" y1="{BOT}" x2="{sx(v):.1f}" y2="{BOT+5}" stroke="#222"/>')
    txt(o, sx(v), BOT+20, f"{v:.1f}", 12, None, "#333", halo=False)
o.append(f'<line x1="{sx(T_D):.1f}" y1="{TOP-4}" x2="{sx(T_D):.1f}" y2="{BOT}" stroke="#111" stroke-width="2" stroke-dasharray="7,5"/>')
txt(o, sx(T_D), TOP-14, f"soglia t = {T_D}", 14, 700, "#111", halo=False)
txt(o, sx(T_D)-8, TOP+6, "← accetto", 12, None, "#111", "end", halo=False)
txt(o, sx(T_D)+8, TOP+6, "rifiuto →", 12, None, "#111", "start", halo=False)
txt(o, sx(MU_GD)-30, sy(pdf(MU_GD, MU_GD, SD_GD))-10, "Genuini", 15, 700, C_G_T)
txt(o, sx(MU_ID)+10, sy(pdf(MU_ID, MU_ID, SD_ID))-10, "Impostori", 15, 700, C_I)
txt(o, sx(0.20), sy(1.2), "GA", 15, 700, C_G_T)
txt(o, sx(0.79), sy(0.7), "GR", 15, 700, C_I)
leader(o, [(sx(0.50), sy(2.2)), (sx(0.50), sy(0.35))], C_FR_T); txt(o, sx(0.50), sy(2.2)-8, "FR", 14, 700, C_FR_T)
leader(o, [(sx(0.12), sy(1.5)), (sx(0.12), sy(0.12)), (sx(0.35), sy(0.12))], C_FA_T); txt(o, sx(0.12), sy(1.5)-8, "FA", 14, 700, C_FA_T)
ly = BOT+64
for i, (c, op, name, val) in enumerate([(C_G, 0.35, "GA", f"GAR = GA/TG = {GAR_D:.1%}"), (C_FR, 0.85, "FR", f"FRR = FR/TG = {FRR_D:.1%}"),
                                         (C_FA, 0.85, "FA", f"FAR = FA/TI = {FAR_D:.1%}"), (C_I, 0.18, "GR", f"GRR = GR/TI = {GRR_D:.1%}")]):
    xx = L + (i % 2)*400
    yy = ly + (i // 2)*26
    o.append(f'<rect x="{xx}" y="{yy-12}" width="16" height="16" fill="{c}" fill-opacity="{op}" stroke="{c}"/>')
    txt(o, xx+24, yy+1, f"{name}  →  {val}", 13, None, "#222", "start", halo=False)
txt(o, L, ly+66, "GA + FR = TG · FA + GR = TI · spostando t a destra aumentano GAR e FAR, a sinistra il contrario (da qui nascono ROC e DET).", 12, None, "#555", "start", halo=False)
save(o, 'fig2_soglia.svg')
```

<img src="./img/fig2_soglia.svg" alt="Distribuzioni genuini e impostori con soglia t" style="display:block; margin:1.5em auto; max-width:100%;">

> Spostare $t$ a destra **aumenta** $GAR$ ma anche $FAR$; spostarla a sinistra fa il contrario. Variando $t$ si ottengono le curve ROC / DET.

---

### 9.4 Verifica — Single Template

Ogni campione della galleria è confrontato **singolarmente**. Per ogni riga $i$ (escludendo la diagonale $i=j$):

- celle genuine: gli altri $S-1$ campioni della stessa identità;
- celle impostore: gli $(N-1)\cdot S$ campioni delle altre identità.

$$
\boxed{TG = |G|\cdot(S-1)} \qquad \boxed{TI = |G|\cdot(N-1)\cdot S}
$$

```
for each threshold t
    for each cell M[i,j] con i ≠ j
        if M[i,j] ≤ t then
            if label(i) = label(j) then GA++
            else FA++
        else
            if label(i) = label(j) then FR++
            else GR++
    GAR(t) = GA/TG ;  FAR(t) = FA/TI
    FRR(t) = FR/TG ;  GRR(t) = GR/TI
```

**Esempio numerico** ($N=3,\ S=2,\ |G|=6$, $t=0.40$): $TG=6\cdot1=6$, $TI=6\cdot2\cdot2=24$. Le celle con contorno verde sono quelle accettate ($M[i,j]\le t$).

**L'esempio numerico, cella per cella.** Ogni cella fuori diagonale è un esperimento ed è etichettata GA/FR/FA/GR. Il codice conta gli esiti e verifica con un `assert` che i totali siano quelli del testo ($GA=4$, $FR=2$, $FA=4$, $GR=20$). Questa stessa matrice `M6` viene riusata nel blocco successivo.

```python
# =====================================================================
# 15 — Single template: esempio numerico 6×6  (richiede 01, 05)
# Definisce la matrice M6 (riusata dal blocco 16). Produce: fig3_single_template.svg
# =====================================================================
LAB6 = ["A1", "A2", "B1", "B2", "C1", "C2"]
sub6 = [l[0] for l in LAB6]
_pairs = {(0, 1): 0.22, (2, 3): 0.31, (4, 5): 0.55,                      # genuine
          (0, 2): 0.62, (0, 3): 0.71, (0, 4): 0.84, (0, 5): 0.78,        # impostori
          (1, 2): 0.36, (1, 3): 0.66, (1, 4): 0.73, (1, 5): 0.69,
          (2, 4): 0.59, (2, 5): 0.67, (3, 4): 0.38, (3, 5): 0.74}
M6 = [[0.0]*6 for _ in range(6)]
for (a, b), v in _pairs.items():
    M6[a][b] = M6[b][a] = v
T_M = 0.40

GA = FR = FA = GR = 0
for i in range(6):
    for j in range(6):
        if i == j: continue
        gen, acc = sub6[i] == sub6[j], M6[i][j] <= T_M
        GA += gen and acc; FR += gen and not acc; FA += (not gen) and acc; GR += (not gen) and not acc
TG, TI = GA+FR, FA+GR
assert (GA, FR, FA, GR, TG, TI) == (4, 2, 4, 20, 6, 24)

CS, X0, Y0 = 66, 130, 150
W, H = 920, 640
o = new_svg(W, H)
header(o, W, "Verifica single template: N = 3, S = 2, |G| = 6",
       f"Si accetta se M[i,j] ≤ t = {T_M} · contorno verde = cella accettata · la diagonale è esclusa")
for k, nm in enumerate(LAB6):
    txt(o, X0+k*CS+CS/2, Y0-12, nm, 13, 700, "#222", halo=False)
    txt(o, X0-12, Y0+k*CS+CS/2+5, nm, 13, 700, "#222", "end", halo=False)
for i in range(6):
    for j in range(6):
        x, y = X0+j*CS, Y0+i*CS
        if i == j:
            mat_cell(o, x, y, CS, CS, "#d1d5db", "×", 16, "#6b7280"); continue
        gen, acc = sub6[i] == sub6[j], M6[i][j] <= T_M
        mat_cell(o, x, y, CS, CS, "#ede9fe" if gen else GREY_BG, f"{M6[i][j]:.2f}", 14, "#111", acc)
        tag = ("GA" if acc else "FR") if gen else ("FA" if acc else "GR")
        txt(o, x+5, y+13, tag, 10, 700, {"GA": C_G_T, "FR": C_FR_T, "FA": C_FA_T, "GR": "#6b7280"}[tag], "start", halo=False)
for i in range(6):
    for j in range(6):
        if i != j and M6[i][j] <= T_M:
            o.append(f'<rect x="{X0+j*CS+2}" y="{Y0+i*CS+2}" width="{CS-4}" height="{CS-4}" fill="none" stroke="{C_G}" stroke-width="3.5"/>')
xr = 590
txt(o, xr, 150, "Conteggi (diagonale esclusa)", 15, 700, "#111", "start", halo=False)
rows_ = [("GA", GA, C_G_T, "genuine accept"), ("FR", FR, C_FR_T, "false reject"), ("FA", FA, C_FA_T, "false accept"), ("GR", GR, "#6b7280", "genuine reject")]
for q, (nm, v, c, desc) in enumerate(rows_):
    txt(o, xr, 182+q*26, f"{nm} = {v}", 14, 700, c, "start", halo=False)
    txt(o, xr+78, 182+q*26, desc, 12.5, None, "#444", "start", halo=False)
txt(o, xr, 310, f"TG = |G|·(S−1) = 6·1 = {TG}  (GA+FR = {GA}+{FR})", 13, None, "#222", "start", halo=False)
txt(o, xr, 334, f"TI = |G|·(N−1)·S = 6·2·2 = {TI}  (FA+GR = {FA}+{GR})", 13, None, "#222", "start", halo=False)
for q, s in enumerate([f"GAR = GA/TG = {GA}/{TG} = {GA/TG:.3f}", f"FRR = FR/TG = {FR}/{TG} = {FR/TG:.3f}",
                       f"FAR = FA/TI = {FA}/{TI} = {FA/TI:.3f}", f"GRR = GR/TI = {GR}/{TI} = {GR/TI:.3f}"]):
    txt(o, xr, 378+q*24, s, 13, None, "#222", "start", halo=False)
txt(o, X0, Y0+6*CS+36, "viola = genuina (stesso soggetto) · grigio = impostore · ogni cella è un esperimento", 12, None, "#555", "start", halo=False)
save(o, 'fig3_single_template.svg')
```

<img src="./img/fig3_single_template.svg" alt="Esempio numerico single template" style="display:block; margin:1.5em auto; max-width:100%;">

Si verifica che $GA+FR=4+2=6=TG$ e $FA+GR=4+20=24=TI$.

---

### 9.5 Verifica — Multiple Template

Quando la galleria ha $S$ template per soggetto, la decisione è **per identità**, non per campione: per ogni riga $i$ le colonne si raggruppano per label,

$$
\mathcal{M}_X = \{\, M[i,j] \;:\; \mathrm{label}(j)=X,\ j\neq i \,\}
$$

e di ogni gruppo si tiene il **miglior match** (distanza minima):

$$
d_X = \min \mathcal{M}_X
$$

Poiché ogni riga produce **un esperimento per identità** ($N$ in tutto, di cui 1 genuino e $N-1$ impostori):

$$
\boxed{TG = |G|} \qquad \boxed{TI = |G|\cdot(N-1)}
$$

```
for each threshold t
    for each row i
        for each gruppo M_label di celle M[i,j] con stessa label(j), escludendo M[i,i]
            diff = min(M_label)
            if diff ≤ t then
                if label(i) = label(M_label) then GA++
                else FA++
            else
                if label(i) = label(M_label) then FR++
                else GR++
    GAR(t) = GA/TG ; FAR(t) = FA/TI ; FRR(t) = FR/TG ; GRR(t) = GR/TI
```


**Da $6\times6$ a $6\times3$.** Per ogni riga, le colonne vengono raggruppate per identità e si tiene solo il minimo del gruppo (il pallino colorato indica quale cella vince). A parità di matrice e soglia, il FRR resta $2/6$ ma il FAR sale da $4/24$ a $4/12$: è l'effetto di $S$ descritto nel riquadro sotto.

```python
# =====================================================================
# 16 — Multiple template: riduzione a minimo per gruppo  (richiede 01, 05, 15)
# Produce: fig4_multiple_template.svg
# =====================================================================
groups = ["A", "B", "C"]
D6 = []                                   # D6[i][g] = min delle distanze della riga i verso il gruppo g (esclusa la diagonale)
ARG = []                                  # colonna che realizza il minimo
for i in range(6):
    rowD, rowA = [], []
    for g in groups:
        cand = [(M6[i][j], j) for j in range(6) if sub6[j] == g and j != i]
        v, j = min(cand)
        rowD.append(v); rowA.append(j)
    D6.append(rowD); ARG.append(rowA)

GA = FR = FA = GR = 0
for i in range(6):
    for gi, g in enumerate(groups):
        gen, acc = sub6[i] == g, D6[i][gi] <= T_M
        GA += gen and acc; FR += gen and not acc; FA += (not gen) and acc; GR += (not gen) and not acc
TG, TI = GA+FR, FA+GR
assert (GA, FR, FA, GR, TG, TI) == (4, 2, 4, 8, 6, 12)

W, H = 920, 600
o = new_svg(W, H)
header(o, W, "Verifica multiple template: minimo per gruppo",
       f"d_X = min {{ M[i,j] : label(j) = X, j ≠ i }} · matrice 6×6 → 6×3 · t = {T_M}")
# matrice di sinistra
CSL, XL0, YL0 = 46, 90, 150
txt(o, XL0+3*CSL, YL0-56, "M  (|G| × |G|)", 14, 700, "#111", halo=False)
gcol = {"A": "#2563eb", "B": "#db2777", "C": "#0f766e"}
for k, nm in enumerate(LAB6):
    txt(o, XL0+k*CSL+CSL/2, YL0-10, nm, 12, 700, "#222", halo=False)
    txt(o, XL0-8, YL0+k*CSL+CSL/2+4, nm, 12, 700, "#222", "end", halo=False)
for gi, g in enumerate(groups):
    o.append(f'<line x1="{XL0+gi*2*CSL+4}" y1="{YL0-30}" x2="{XL0+(gi+1)*2*CSL-4}" y2="{YL0-30}" stroke="{gcol[g]}" stroke-width="4"/>')
for i in range(6):
    for j in range(6):
        x, y = XL0+j*CSL, YL0+i*CSL
        if i == j: mat_cell(o, x, y, CSL, CSL, "#d1d5db", "×", 14, "#6b7280"); continue
        gen = sub6[i] == sub6[j]
        mat_cell(o, x, y, CSL, CSL, "#ede9fe" if gen else GREY_BG, f"{M6[i][j]:.2f}", 11.5, "#111", ARG[i][groups.index(sub6[j])] == j)
        if ARG[i][groups.index(sub6[j])] == j:
            o.append(f'<circle cx="{x+CSL-9}" cy="{y+9}" r="4" fill="{gcol[sub6[j]]}"/>')
txt(o, XL0, YL0+6*CSL+22, "● = cella che realizza il minimo del gruppo", 11.5, None, "#555", "start", halo=False)

arrow(o, XL0+6*CSL+16, YL0+3*CSL, 440, YL0+3*CSL, "#111", 2.2)
txt(o, (XL0+6*CSL+16+440)/2, YL0+3*CSL-24, "min per", 12.5, 700, "#111")
txt(o, (XL0+6*CSL+16+440)/2, YL0+3*CSL-9, "gruppo", 12.5, 700, "#111")

# matrice ridotta
CSR, XR0 = 74, 470
txt(o, XR0+1.5*CSR, YL0-56, "D  (|G| × N)", 14, 700, "#111", halo=False)
for gi, g in enumerate(groups):
    txt(o, XR0+gi*CSR+CSR/2, YL0-10, f"d_{g}", 13, 700, gcol[g], halo=False)
    o.append(f'<line x1="{XR0+gi*CSR+4}" y1="{YL0-30}" x2="{XR0+(gi+1)*CSR-4}" y2="{YL0-30}" stroke="{gcol[g]}" stroke-width="4"/>')
for i in range(6):
    txt(o, XR0-8, YL0+i*CSL+CSL/2+4, LAB6[i], 12, 700, "#222", "end", halo=False)
    for gi, g in enumerate(groups):
        gen, acc = sub6[i] == g, D6[i][gi] <= T_M
        x, y = XR0+gi*CSR, YL0+i*CSL
        mat_cell(o, x, y, CSR, CSL, "#ede9fe" if gen else GREY_BG, f"{D6[i][gi]:.2f}", 13.5, "#111", acc)
        tag = ("GA" if acc else "FR") if gen else ("FA" if acc else "GR")
        txt(o, x+5, y+12, tag, 9.5, 700, {"GA": C_G_T, "FR": C_FR_T, "FA": C_FA_T, "GR": "#6b7280"}[tag], "start", halo=False)
        if acc:
            o.append(f'<rect x="{x+2}" y="{y+2}" width="{CSR-4}" height="{CSL-4}" fill="none" stroke="{C_G}" stroke-width="3.5"/>')

# conteggi
xr = 730
txt(o, xr, 150, "Conteggi", 15, 700, "#111", "start", halo=False)
for q, (nm, v, c) in enumerate([("GA", GA, C_G_T), ("FR", FR, C_FR_T), ("FA", FA, C_FA_T), ("GR", GR, "#6b7280")]):
    txt(o, xr, 178+q*24, f"{nm} = {v}", 14, 700, c, "start", halo=False)
txt(o, xr, 290, f"TG = |G| = {TG}", 13, None, "#222", "start", halo=False)
txt(o, xr, 312, f"TI = |G|(N−1) = {TI}", 13, None, "#222", "start", halo=False)
txt(o, xr, 346, f"FRR = {FR}/{TG} = {FR/TG:.3f}", 12.5, None, "#222", "start", halo=False)
txt(o, xr, 368, f"FAR = {FA}/{TI} = {FA/TI:.3f}", 12.5, None, "#222", "start", halo=False)
txt(o, 90, 500, "Ogni riga produce un esperimento per identità: N = 3 (1 genuino + 2 impostori).", 13, None, "#222", "start", halo=False)
txt(o, 90, 524, "Il minimo su S valori è non crescente in S: più template in galleria → FRR ↓ ma FAR può ↑.", 13, None, "#222", "start", halo=False)
txt(o, 90, 560, "Confronto con single template (stessa matrice, stessa soglia): FRR = 2/6 e FAR = 4/24 = 0.167 → qui FAR = 4/12 = 0.333", 12.5, None, "#555", "start", halo=False)
save(o, 'fig4_multiple_template.svg')
```

<img src="./img/fig4_multiple_template.svg" alt="Riduzione a minimo per gruppo (multiple template)" style="display:block; margin:1.5em auto; max-width:100%;">

Nell'esempio la matrice $6\times6$ si riduce a una matrice $6\times3$ ($|G|\times N$); con $t=0.40$ si ottiene $GA=4,\ FR=2$ (somma $=TG=6$) e $FA=4,\ GR=8$ (somma $=TI=12$).

> **Effetto di $S$ sul compromesso errori.**
> Più campioni in galleria per soggetto → **diminuisce $FRR$** (più occasioni di trovare un match corretto, perché il minimo di più valori è più piccolo) ma **può aumentare $FAR$** (più occasioni per un impostore di assomigliare a un template).
> In formula: $\min$ su $S$ valori è non crescente in $S$, quindi sia i genuini sia gli impostori si spostano verso distanze minori.

**Distribuzione del minimo di S distanze.** Con più template per soggetto sia i genuini sia gli impostori si spostano verso distanze minori: a parità di soglia FRR scende e FAR sale. Sotto, i valori per S da 1 a 5, calcolati assumendo campioni indipendenti.

```python
# =====================================================================
# N12 — Effetto di S  (richiede 01, 05)
# =====================================================================
MG, SG, MI, SI = 0.30, 0.10, 0.62, 0.13
T_S = 0.40
ng, ni = NormalDist(MG, SG), NormalDist(MI, SI)
def pmin(x, nd, S):
    return S*nd.pdf(x)*(1-nd.cdf(x))**(S-1)
W, H = 920, 650
o = new_svg(W, H)
header(o, W, "Effetto di S (template per soggetto) sulle distribuzioni di distanza",
       f"Con S template si tiene il minimo di S distanze · genuini ~ N({MG}, {SG}²), impostori ~ N({MI}, {SI}²) · t = {T_S}")
XMIN, XMAX, YMAX = -0.1, 1.1, 7.0
def panel(x0, S, title):
    L, R, TOP, BOT = x0, x0+400, 120, 360
    sx = lambda x: L+(x-XMIN)/(XMAX-XMIN)*(R-L)
    sy = lambda y: BOT-min(y, YMAX)/YMAX*(BOT-TOP)
    n = 220
    xs = [XMIN+(XMAX-XMIN)*i/n for i in range(n+1)]
    def area(nd, a, b, col, op):
        pts = [(sx(x), sy(pmin(x, nd, S))) for x in xs if a <= x <= b]
        o.append(f'<path d="M {pts[0][0]:.1f},{BOT} ' + " ".join(f"L {x:.1f},{y:.1f}" for x, y in pts) + f' L {pts[-1][0]:.1f},{BOT} Z" fill="{col}" fill-opacity="{op}"/>')
    area(ng, T_S, XMAX, C_FR, 0.85)      # FR: genuini sopra soglia
    area(ni, XMIN, T_S, C_FA, 0.85)      # FA: impostori sotto soglia
    polyline(o, [(sx(x), sy(pmin(x, ng, S))) for x in xs], C_G, 2.8)
    polyline(o, [(sx(x), sy(pmin(x, ni, S))) for x in xs], C_I, 2.8)
    frame(o, L, R, TOP, BOT, "distanza (minimo del gruppo)", None)
    for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        o.append(f'<line x1="{sx(v):.1f}" y1="{BOT}" x2="{sx(v):.1f}" y2="{BOT+5}" stroke="#222"/>')
        txt(o, sx(v), BOT+20, f"{v:.1f}", 11.5, None, "#333", halo=False)
    o.append(f'<line x1="{sx(T_S):.1f}" y1="{TOP-6}" x2="{sx(T_S):.1f}" y2="{BOT}" stroke="#111" stroke-width="2" stroke-dasharray="7,5"/>')
    txt(o, (L+R)/2, 98, title, 14.5, 700, "#111", halo=False)
    fr = (1-ng.cdf(T_S))**S
    fa = 1-(1-ni.cdf(T_S))**S
    txt(o, L+4, TOP+8, "genuini", 12.5, 700, C_G_T, "start")
    txt(o, R-4, TOP+8, "impostori", 12.5, 700, C_I, "end")
    txt(o, (L+R)/2, BOT+74, f"FRR = {fr:.1%}", 13.5, 700, C_FR_T)
    txt(o, (L+R)/2, BOT+94, f"FAR = {fa:.1%}", 13.5, 700, C_FA_T)
panel(50, 1, "S = 1")
panel(490, 4, "S = 4")
txt(o, W/2, 490, "FRR = (1 − P(genuino ≤ t))^S  ·  FAR = 1 − (1 − P(impostore ≤ t))^S   (campioni indipendenti)", 12.5, None, "#333", halo=False)
# tabella S
txt(o, 50, 530, "Con la stessa soglia", 13.5, 700, "#111", "start", halo=False)
Ss = [1, 2, 3, 4, 5]
for k, S in enumerate(Ss):
    x = 50+k*165
    fr = (1-ng.cdf(T_S))**S; fa = 1-(1-ni.cdf(T_S))**S
    rect(o, x, 545, 155, 70, "#f9fafb", "#d1d5db", 1.2, 8)
    txt(o, x+77, 565, f"S = {S}", 12.5, 700, "#111", halo=False)
    txt(o, x+77, 585, f"FRR {fr:.1%}", 12.5, 700, C_FR_T, halo=False)
    txt(o, x+77, 603, f"FAR {fa:.1%}", 12.5, 700, C_FA_T, halo=False)
txt(o, W/2, 638, "più template → FRR ↓ (più occasioni di match corretto) ma FAR ↑ (più occasioni per un impostore)", 12, None, "#555", halo=False)
save(o, 'n12_effetto_S_template.svg')
```
<img src="./img/n12_effetto_S_template.svg" alt="Effetto di S" style="display:block; margin:1.5em auto; max-width:100%;">

---

## 10. All-Against-All — Probe vs Gallery (sessioni separate)

Probe e galleria provengono da **sessioni diverse** e **non condividono campioni**: non esiste diagonale da escludere. Si ha $|P|=|G|=S\cdot N$ e tutte le $|P|\cdot|G|$ celle sono utilizzabili.

**Due sessioni, nessuna diagonale.** Righe e colonne provengono da sessioni diverse: ogni cella è utilizzabile e ogni riga ha $S$ genuini (non $S-1$). È la configurazione da preferire quando i campioni della stessa sessione sono troppo simili tra loro.

```python
# =====================================================================
# 17 — Probe vs Gallery (sessioni separate)  (richiede 01, 05, 13)
# Produce: fig5_probe_gallery.svg
# =====================================================================
N_, S_ = 3, 2
nG = N_*S_
CS, X0, Y0 = 62, 170, 170
W, H = 920, 560
o = new_svg(W, H)
header(o, W, "All-against-all probe vs gallery: sessioni separate",
       "Nessun campione condiviso → nessuna diagonale da escludere · |P| = |G| = S·N")
txt(o, X0+nG*CS/2, Y0-52, "galleria (sessione 1)  →", 13, 700, C_I, halo=False)
o.append(f'<text transform="translate(52,{Y0+nG*CS/2}) rotate(-90)" text-anchor="middle" font-size="13" font-weight="700" fill="{C_FA_T}">probe (sessione 2)</text>')
for k, nm in enumerate(LAB):
    txt(o, X0+k*CS+CS/2, Y0-12, "g_"+nm, 12.5, 700, C_I, halo=False)
    txt(o, X0-12, Y0+k*CS+CS/2+5, "p_"+nm, 12.5, 700, C_FA_T, "end", halo=False)
for i in range(nG):
    for j in range(nG):
        gen = i//S_ == j//S_
        mat_cell(o, X0+j*CS, Y0+i*CS, CS, CS, C_GEN_BG if gen else GREY_BG, "G" if gen else "imp", 15 if gen else 12, C_GT if gen else "#6b7280", gen)
for b in range(N_):
    o.append(f'<rect x="{X0+b*S_*CS}" y="{Y0+b*S_*CS}" width="{S_*CS}" height="{S_*CS}" fill="none" stroke="{C_GT}" stroke-width="3"/>')
yr = Y0 + CS
o.append(f'<rect x="{X0-4}" y="{yr-3}" width="{nG*CS+8}" height="{CS+6}" rx="6" fill="none" stroke="#111" stroke-width="2.5" stroke-dasharray="6,4"/>')
xr = 600
txt(o, xr, 170, "Una riga (probe p_A2)", 14, 700, "#111", "start", halo=False)
txt(o, xr, 194, f"• celle genuine: S = {S_}   (non S − 1)", 13, None, "#222", "start", halo=False)
txt(o, xr, 216, f"• celle impostore: (N − 1)·S = {(N_-1)*S_}", 13, None, "#222", "start", halo=False)
txt(o, xr, 262, "Single template (ogni cella è un esperimento)", 13.5, 700, "#111", "start", halo=False)
txt(o, xr, 286, f"TG = |P|·S = {nG*S_}", 13, None, "#222", "start", halo=False)
txt(o, xr, 308, f"TI = |P|·(N−1)·S = {nG*(N_-1)*S_}", 13, None, "#222", "start", halo=False)
txt(o, xr, 354, "Multiple template (minimo per gruppo)", 13.5, 700, "#111", "start", halo=False)
txt(o, xr, 378, f"TG = |P| = {nG}", 13, None, "#222", "start", halo=False)
txt(o, xr, 400, f"TI = |P|·(N−1) = {nG*(N_-1)}", 13, None, "#222", "start", halo=False)
txt(o, xr, 446, "Perché due sessioni?", 13.5, 700, "#111", "start", halo=False)
txt(o, xr, 470, "Campioni della stessa sessione sono più simili", 12.5, None, "#444", "start", halo=False)
txt(o, xr, 488, "tra loro: i genuini risulterebbero troppo vicini", 12.5, None, "#444", "start", halo=False)
txt(o, xr, 506, "e le prestazioni ottimisticamente falsate.", 12.5, None, "#444", "start", halo=False)
save(o, 'fig5_probe_gallery.svg')
```

<img src="./img/fig5_probe_gallery.svg" alt="Probe vs gallery con sessioni separate" style="display:block; margin:1.5em auto; max-width:100%;">

### 10.1 Verifica

**Single template** (ogni cella è un esperimento):

$$
TG = |P|\cdot S \qquad TI = |P|\cdot(N-1)\cdot S
$$

**Multiple template** (min per gruppo, un esperimento per identità):

$$
TG = |P| \qquad TI = |P|\cdot(N-1)
$$

Confronto con il caso §9 (con diagonale): al posto di $S-1$ genuini per riga se ne hanno $S$, perché il campione di probe non è presente in galleria.

### 10.2 Identificazione Open Set (multiple template)

Qui **non c'è un claim**: il sistema deve decidere *chi* è il probe, oppure rispondere "sconosciuto". Per questo ogni riga rappresenta **2 esperimenti**, uno per ciascuno scenario:

| Scenario | Situazione | Esito |
|---|---|---|
| **Genuino** | l'identità del probe **è** in galleria | valutiamo se viene riconosciuta al rango $k$ |
| **Impostore** | l'identità del probe **non è** in galleria (si rimuove) | valutiamo se viene erroneamente accettata |

$$
\boxed{TG = |G|} \qquad \boxed{TI = |G|}
$$

Per ogni riga $i$ si costruisce la lista $L[i]$ delle distanze (una per identità, minimo per gruppo) **ordinata in modo crescente**, e $L[i,k]$ è il $k$-esimo elemento.

$$
L[i,1]\le L[i,2]\le\dots
$$

**Due esperimenti per riga.** Nello scenario genuino si cerca a che rango compare l'identità del probe e si controlla che la sua distanza sia $\le t$. Nello scenario impostore la stessa identità viene rimossa dalla galleria: qualunque elemento con distanza $\le t$ è un'accettazione errata (FA), altrimenti è un GR.

```python
# =====================================================================
# 18 — Open set all-against-all: scenari genuino e impostore  (richiede 01, 05)
# Produce: fig6_open_set.svg
# =====================================================================
T_OSA = 0.35
ID_PROBE = "A"
L_gen = [("C", 0.21), ("A", 0.27), ("D", 0.44), ("B", 0.58), ("E", 0.71)]   # L[i]: una distanza per identità (min per gruppo), crescente
L_imp = [(g, d) for g, d in L_gen if g != ID_PROBE]                          # identità del probe rimossa
k_star = [g for g, _ in L_gen].index(ID_PROBE) + 1
assert L_gen[k_star-1][1] <= T_OSA and L_imp[0][1] <= T_OSA

def strip(o, y, lst, title, subtitle, mark_true):
    BW, BH, X0 = 120, 66, 120
    txt(o, 40, y-34, title, 14.5, 700, "#111", "start", halo=False)
    txt(o, 40, y-14, subtitle, 12.5, None, "#555", "start", halo=False)
    for k, (g, d) in enumerate(lst, 1):
        x = X0 + (k-1)*(BW+12)
        acc = d <= T_OSA
        is_t = mark_true and g == ID_PROBE
        rect(o, x, y, BW, BH, C_OK_BG if acc else GREY_BG, C_GT if is_t else (C_G if acc else "#9ca3af"), 3 if is_t else 1.6, 8, "6,3" if is_t else None)
        txt(o, x+BW/2, y+17, f"k = {k}", 11.5, None, "#555", halo=False)
        txt(o, x+BW/2, y+40, f"id {g}" + (" ★" if is_t else ""), 15, 700, C_GT if is_t else "#111", halo=False)
        txt(o, x+BW/2, y+58, f"L[i,{k}] = {d:.2f}", 12, None, C_G_T if acc else "#6b7280", halo=False)
    na = sum(d <= T_OSA for _, d in lst)
    xs = X0 + na*(BW+12) - 6
    o.append(f'<line x1="{xs}" y1="{y-6}" x2="{xs}" y2="{y+BH+8}" stroke="#111" stroke-width="2" stroke-dasharray="6,4"/>')
    txt(o, xs, y+BH+24, f"t = {T_OSA}", 12.5, 700, "#111", halo=False)

W, H = 920, 540
o = new_svg(W, H)
header(o, W, "Identificazione open set (multiple template): due scenari per ogni riga",
       f"Lista L[i] = una distanza per identità (minimo per gruppo), ordinata in modo crescente · t = {T_OSA}")
strip(o, 120, L_gen, "Scenario genuino · l'identità del probe (A) è in galleria",
      f"k* = {k_star}: prima occorrenza di A nella lista", True)
box(o, 460, 248, 760, 44, [f"L[i,k*] = L[i,{k_star}] = {L_gen[k_star-1][1]:.2f} ≤ t   →   DI(t, {k_star}) += 1",
                           f"contribuisce a DIR(t,k) per ogni k ≥ {k_star}; DIR(t,1) non cresce perché il rank 1 è un'altra identità"], C_OK_BG, C_G_T, 12.5)
strip(o, 350, L_imp, "Scenario impostore · l'identità A viene rimossa dalla galleria",
      "nessuna risposta corretta esiste: qualunque accettazione è un errore", False)
box(o, 460, 468, 760, 44, [f"L[i,1] = {L_imp[0][1]:.2f} ≤ t   →   FA += 1", "altrimenti (L[i,1] > t) → GR += 1"], C_FA_BG, C_FA_T, 12.5)
txt(o, W/2, H-14, "TG = |G| e TI = |G|: ogni riga conta una volta come genuina e una volta come impostore", 12, None, "#555", halo=False)
save(o, 'fig6_open_set.svg')
```

<img src="./img/fig6_open_set.svg" alt="Identificazione open set: scenari genuino e impostore" style="display:block; margin:1.5em auto; max-width:100%;">

**Scenario genuino.** Sia $k^*$ il rango della prima occorrenza con $\mathrm{label}=\mathrm{label}(i)$. Si ha un'identificazione corretta al rango $k^*$ se

$$
L[i,k^*]\le t \quad\Longrightarrow\quad DI(t,k^*)\mathrel{+}=1
$$

**Scenario impostore.** Rimossa l'identità del probe, qualunque elemento $L[i,1]\le t$ è un'accettazione errata:

$$
L[i,1]\le t \Longrightarrow FA\mathrel{+}=1, \qquad \text{altrimenti } GR\mathrel{+}=1
$$

**Metriche.**

$$
DIR(t,1)=\frac{DI(t,1)}{TG}, \qquad FRR(t)=1-DIR(t,1)
$$

$$
DIR(t,k)=\frac{DI(t,k)}{TG}+DIR(t,k-1) \quad (k\ge2)
$$

$$
FAR(t)=\frac{FA}{TI}, \qquad GRR(t)=\frac{GR}{TI}
$$

$DIR(t,k)$ è il **Detection & Identification Rate** al rango $k$: frazione di probe genuini correttamente riconosciuti *entro* i primi $k$ candidati e con distanza $\le t$.

Pseudocodice concettuale:

```
for each threshold t
    for each row i
        L[i] = lista ordinata crescente delle distanze M[i,j] (esclusa diagonale)
        if L[i,1] ≤ t then                          # potenziale accettazione
            if label(i) = label(L[i,1]) then DI(t,1)++     # caso genuino
            cerca il primo L[i,k] con label diversa e ≤ t → se esiste: FA++   # caso impostore
            altrimenti cerca il primo L[i,k] con stessa label e ≤ t
                → se esiste: DI(t,k)++ ; FA++
        else GR++
    DIR(t,1) = DI(t,1)/TG ;  FRR(t) = 1 - DIR(t,1)
    FAR(t) = FA/TI ;  GRR(t) = GR/TI
    # per ranghi superiori:
    DIR(t,k) = DI(t,k)/TG + DIR(t,k-1)
```

### 10.3 Identificazione Closed Set

Si assume che il probe **sia sempre in galleria**: non esistono impostori e **non serve soglia**. Si misura solo *a che rango* compare l'identità corretta.

$$
TA = |P|
$$

Per ogni riga si ordina $L[i]$ e si cerca il primo $k$ tale che $\mathrm{label}(i)=\mathrm{label}(L[i,k])$; si incrementa $CMS(k)$ (Cumulative Match Score). Poi:

$$
CMS(1)=\frac{n_1}{TA}=RR, \qquad
CMS(k)=\frac{n_k}{TA}+CMS(k-1)\quad (k=2,\dots,K)
$$

dove $n_k$ è il numero di probe la cui identità corretta compare **per la prima volta** al rango $k$, e $K=|G|-1$ (caso con diagonale) oppure $K=|G|$ (probe vs gallery).

**CMS con la strategia all-against-all.** $n_k$ conta i probe la cui identità corretta compare per la prima volta al rango $k$; la CMS è la somma cumulata di $n_k/TA$. Nell'esempio, 60 probe: $RR=30/60=0.50$ e la curva arriva a 1 al rango $K=9$.

```python
# =====================================================================
# 19 — Closed set all-against-all: CMS  (richiede 01, 05)
# Produce: fig7_cms.svg
# =====================================================================
nk = [30, 10, 6, 4, 3, 2, 2, 2, 1]                 # n_k: probe la cui identità corretta compare per la prima volta al rango k
TA = sum(nk); K = len(nk)
cms_k = [sum(nk[:k+1])/TA for k in range(K)]

W, H = 920, 540
L, R, TOP, BOT = 100, 860, 100, 410
cw = (R-L)/K
sxk = lambda k: L + (k-0.5)*cw
sy = lambda v: BOT - v*(BOT-TOP)
o = new_svg(W, H)
header(o, W, "CMS closed set con la strategia all-against-all",
       f"TA = |P| = {TA} probe · K = {K} ranghi · n_k = probe con identità corretta al rango k per la prima volta")
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="#e5e7eb"/>')
    txt(o, L-8, sy(v)+4, f"{v:.1f}", 12, None, "#333", "end", halo=False)
frame(o, L, R, TOP, BOT, None, "CMS(k)")
txt(o, (L+R)/2, BOT+34, "rango k", 13, None, "#222", halo=False)
for k in range(1, K+1):
    o.append(f'<rect x="{sxk(k)-cw*0.25:.1f}" y="{sy(nk[k-1]/TA):.1f}" width="{cw*0.5:.1f}" height="{BOT-sy(nk[k-1]/TA):.1f}" fill="#93c5fd" fill-opacity="0.8"/>')
    txt(o, sxk(k), BOT+18, str(k), 12.5, None, "#333", halo=False)
    txt(o, sxk(k), BOT+58, str(nk[k-1]), 12, None, "#333", halo=False)
txt(o, L-8, BOT+58, "n_k", 12, 700, "#555", "end", halo=False)
pts = [(sxk(k), sy(cms_k[k-1])) for k in range(1, K+1)]
o.append('<path d="M ' + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{C_G}" stroke-width="3.2"/>')
for k in range(1, K+1):
    o.append(f'<circle cx="{sxk(k):.1f}" cy="{sy(cms_k[k-1]):.1f}" r="5" fill="{C_G}" stroke="#fff" stroke-width="1.5"/>')
    if k > 1:
        txt(o, sxk(k), sy(cms_k[k-1])-11, f"{cms_k[k-1]:.2f}", 12, None, C_G_T)
o.append(f'<circle cx="{sxk(1):.1f}" cy="{sy(cms_k[0]):.1f}" r="9" fill="none" stroke="#111" stroke-width="2"/>')
txt(o, sxk(1)+16, sy(cms_k[0])-14, f"CMS(1) = n_1/TA = RR = {cms_k[0]:.2f}", 13, 700, "#111", "start")
txt(o, sxk(K)-6, sy(1)+20, f"CMS(K) = 1", 13, 700, C_G_T, "end")
txt(o, sxk(4), sy(0.35), "CMS(k) = n_k/TA + CMS(k−1)", 14, 700, "#1d4ed8", "start")
txt(o, sxk(4), sy(0.35)+20, "non decrescente, parte da RR e arriva a 1", 12, None, "#444", "start")
txt(o, L, H-16, "Ogni riga della matrice P×G contribuisce una sola volta, al rango in cui compare per la prima volta la sua identità.", 12, None, "#555", "start", halo=False)
save(o, 'fig7_cms.svg')
```

<img src="./img/fig7_cms.svg" alt="Curva CMS closed set" style="display:block; margin:1.5em auto; max-width:100%;">

Proprietà utili:

- $CMS(k)$ è **non decrescente** e $CMS(K)=1$ (la risposta giusta è sempre da qualche parte).
- $RR = CMS(1)$ è il **Recognition Rate** (rango-1).

```
for each row i
    L[i] = lista ordinata crescente delle distanze M[i,j]
    trova il primo L[i,k] tale che label(i) = label(L[i,k])
    CMS(k)++
CMS(1) = CMS(1)/TA ;  RR = CMS(1)
for k = 2 to |G|-1
    CMS(k) = CMS(k)/TA + CMS(k-1)
```
**Mappa delle strategie.** Per ogni strategia, i totali di esperimenti genuini e impostori con la diagonale (§9) e con due sessioni (§10), con l'esempio N = 3, S = 2.

```python
# =====================================================================
# N14 — Mappa delle strategie  (richiede 01, 05)
# =====================================================================
W, H = 920, 560
o = new_svg(W, H)
header(o, W, "Mappa delle strategie all-against-all: quanti esperimenti (TG, TI)",
       "Esempio numerico con N = 3 soggetti, S = 2 template, |G| = |P| = 6")
cols = [("Strategia", 40, 200), ("§9 · stessa matrice (con diagonale)", 245, 270), ("§10 · probe vs gallery (due sessioni)", 520, 270), ("Esempio (TG · TI)", 795, 110)]
Y0, RH = 90, 74
for name, x, w in cols:
    mat_cell(o, x, Y0, w, 44, "#e5e7eb", None, stroke="#9ca3af")
    txt(o, x+w/2, Y0+27, name, 12, 700, "#111", halo=False)
rows = [
 ("Verifica", "single template", "#dbeafe", "TG = |G|·(S−1)", "TI = |G|·(N−1)·S", "TG = |P|·S", "TI = |P|·(N−1)·S", "6 · 24 / 12 · 24"),
 ("Verifica", "multiple template (min per gruppo)", "#e0e7ff", "TG = |G|", "TI = |G|·(N−1)", "TG = |P|", "TI = |P|·(N−1)", "6 · 12 / 6 · 12"),
 ("Identificazione", "open set (multiple template)", "#fef3c7", "—", "—", "TG = |G|", "TI = |G|", "6 · 6"),
 ("Identificazione", "closed set (CMS, nessuna soglia)", "#dcfce7", "TA = |P|", "K = |G|−1", "TA = |P|", "K = |G|", "TA = 6"),
]
for r, (a, b, bg, g1, g2, p1, p2, ex) in enumerate(rows):
    y = Y0+44+r*RH
    mat_cell(o, 40, y, 200, RH, bg, None)
    txt(o, 140, y+RH/2-4, a, 13.5, 700, "#111", halo=False)
    txt(o, 140, y+RH/2+14, b, 11, None, "#333", halo=False)
    for (x, w, l1, l2) in [(245, 270, g1, g2), (520, 270, p1, p2)]:
        mat_cell(o, x, y, w, RH, "#fff", None)
        txt(o, x+w/2, y+RH/2-4, l1, 13, 700 if l1 != "—" else None, "#111" if l1 != "—" else "#9ca3af", halo=False)
        if l2 != "—": txt(o, x+w/2, y+RH/2+16, l2, 13, 700, "#111", halo=False)
    mat_cell(o, 795, y, 110, RH, "#f9fafb", None)
    parts = ex.split(" / ")
    if len(parts) == 2:
        txt(o, 850, y+RH/2-4, "§9: "+parts[0], 12, None, "#333", halo=False)
        txt(o, 850, y+RH/2+16, "§10: "+parts[1], 12, None, "#333", halo=False)
    else:
        txt(o, 850, y+RH/2+4, ex, 12, None, "#333", halo=False)
yb = Y0+44+4*RH+30
txt(o, 40, yb, "Come leggerla", 13.5, 700, "#111", "start", halo=False)
notes = ["Single template: ogni cella è un esperimento. Multiple template: un esperimento per identità (minimo del gruppo).",
         "Con la diagonale (§9) i genuini per riga sono S−1; con due sessioni (§10) sono S, perché il probe non è in gallery.",
         "Open set: ogni riga conta due volte (scenario genuino + scenario impostore con l'identità rimossa): TG = TI = |G|.",
         "Esempio: valori (TG · TI) calcolati per la matrice 6×6. Il closed set conta solo a che rango compare l'identità giusta."]
for k, s in enumerate(notes):
    txt(o, 40, yb+24+k*20, s, 12, None, "#444", "start", halo=False)
save(o, 'n14_mappa_strategie_TG_TI.svg')
```
<img src="./img/n14_mappa_strategie_TG_TI.svg" alt="Mappa delle strategie" style="display:block; margin:1.5em auto; max-width:100%;">


---

## 11. Doddington Zoo (e Biometric Menagerie)

### 11.0 Idea di fondo

Un sistema biometrico non è "ugualmente bravo" con tutti gli utenti: tipicamente gli errori sono **concentrati su pochi individui**. Doddington (riconoscimento vocale) ha classificato gli utenti in base al comportamento medio dei loro score, collegando ogni tipologia di utente a 4 animali; Yager e Dunstone hanno poi esteso la classificazione con quattro nuovi animali, basati sul piano (media genuini, media impostori).

> **Convenzione.** Gli score $s$ sono **similarità**: più alto = più simile, si accetta se $s>t$. Con le distanze (come in §9–10) tutte le disuguaglianze si invertono.

### 11.1 Definizioni formali

Sia $s(j,k)$ lo score del probe $j$ confrontato con il template $k$ (matrice della §9). Per ogni utente $k$:

$$
G_k=\{\,s(k,k)\,\} \qquad
I_k^{\mathrm{vit}}=\{\,s(j,k)\,\}_{j\neq k} \qquad
I_k^{\mathrm{att}}=\{\,s(k,j)\,\}_{j\neq k}
$$

$$
I_k = I_k^{\mathrm{vit}}\cup I_k^{\mathrm{att}}
$$

- $G_k$ = score **genuini** di $k$ (diagonale);
- $I_k^{\mathrm{vit}}$ = score ottenuti dagli altri quando **attaccano $k$** (colonna: $k$ è la *vittima*);
- $I_k^{\mathrm{att}}$ = score ottenuti da $k$ quando **attacca gli altri** (riga: $k$ è l'*attaccante*).

**Dove si leggono goat, lamb e wolf.** In questa matrice di similarità (dati simulati con seed fisso) una *goat* è una cella della diagonale più bassa delle altre; un *lamb* è una colonna alta (tanti utenti ottengono score elevati contro il suo template); un *wolf* è una riga alta (il suo probe ottiene score elevati contro molti template). Colonna = vittima, riga = attaccante.

```python
# =====================================================================
# 20 — Matrice s(j,k): goat, lamb, wolf  (richiede 01, 05)
# Produce: fig10_matrice_lupi_agnelli.svg
# =====================================================================
random.seed(7)
KU = 7
GOAT, LAMB, WOLF = 2, 4, 1                          # indici utente (0-based)
Ms = [[0.0]*KU for _ in range(KU)]
for j in range(KU):                                 # j = probe/attaccante, k = template/vittima
    for k in range(KU):
        if j == k:   Ms[j][k] = random.gauss(0.76, 0.03)
        else:        Ms[j][k] = random.gauss(0.30, 0.04)
Ms[GOAT][GOAT] = 0.40                               # capra: genuino basso
for j in range(KU):
    if j != LAMB: Ms[j][LAMB] = random.gauss(0.60, 0.03)      # agnello: colonna alta (vittima facile)
for k in range(KU):
    if k != WOLF: Ms[WOLF][k] = random.gauss(0.58, 0.03)      # lupo: riga alta (attaccante efficace)
Ms[WOLF][WOLF] = 0.76

def heat(v):                                        # bianco → blu
    a = max(0.0, min(1.0, (v-0.15)/0.7))
    r, g, b = [int(c0 + (c1-c0)*a) for c0, c1 in zip((248, 250, 252), (29, 78, 216))]
    return f"#{r:02x}{g:02x}{b:02x}", ("#fff" if a > 0.55 else "#111")

CS, X0, Y0 = 62, 140, 170
W, H = 920, 690
o = new_svg(W, H)
header(o, W, "Matrice degli score s(j,k): goat, lamb, wolf",
       "Riga j = probe dell'utente j (attaccante) · colonna k = template dell'utente k (vittima) · diagonale = genuini G_k")
for k in range(KU):
    txt(o, X0+k*CS+CS/2, Y0-12, f"U{k+1}", 13, 700, "#222", halo=False)
    txt(o, X0-12, Y0+k*CS+CS/2+5, f"U{k+1}", 13, 700, "#222", "end", halo=False)
txt(o, X0+KU*CS/2, Y0-40, "template k  (vittima)  →", 13, 700, "#555", halo=False)
o.append(f'<text transform="translate(62,{Y0+KU*CS/2}) rotate(-90)" text-anchor="middle" font-size="13" font-weight="700" fill="#555">probe j  (attaccante)</text>')
for j in range(KU):
    for k in range(KU):
        bg, fg = heat(Ms[j][k])
        mat_cell(o, X0+k*CS, Y0+j*CS, CS, CS, bg, f"{Ms[j][k]:.2f}", 13, fg, j == k, "#ffffff", 1.5)
# evidenziazioni
o.append(f'<rect x="{X0+LAMB*CS-2}" y="{Y0-4}" width="{CS+4}" height="{KU*CS+8}" fill="none" stroke="{C_FA}" stroke-width="3" stroke-dasharray="7,4"/>')
o.append(f'<rect x="{X0-4}" y="{Y0+WOLF*CS-2}" width="{KU*CS+8}" height="{CS+4}" fill="none" stroke="{C_GT}" stroke-width="3" stroke-dasharray="7,4"/>')
o.append(f'<rect x="{X0+GOAT*CS+2}" y="{Y0+GOAT*CS+2}" width="{CS-4}" height="{CS-4}" fill="none" stroke="{C_FR}" stroke-width="4"/>')
# etichette
xr = X0 + KU*CS + 24
txt(o, xr, Y0+WOLF*CS+CS/2-4, "Wolf U2", 14, 700, C_GT, "start", halo=False)
txt(o, xr, Y0+WOLF*CS+CS/2+13, "riga alta: I_k^att", 12, None, "#444", "start", halo=False)
txt(o, xr, Y0+WOLF*CS+CS/2+29, "causa FA verso gli altri", 12, None, "#444", "start", halo=False)
txt(o, X0+LAMB*CS+CS/2, Y0+KU*CS+24, "Lamb U5", 14, 700, C_FA_T, halo=False)
txt(o, X0+LAMB*CS+CS/2, Y0+KU*CS+41, "colonna alta: I_k^vit", 12, None, "#444", halo=False)
txt(o, X0+LAMB*CS+CS/2, Y0+KU*CS+57, "subisce FA", 12, None, "#444", halo=False)
txt(o, xr, Y0+GOAT*CS+CS/2-4, "Goat U3", 14, 700, C_FR_T, "start", halo=False)
txt(o, xr, Y0+GOAT*CS+CS/2+13, "diagonale bassa: G_k", 12, None, "#444", "start", halo=False)
txt(o, xr, Y0+GOAT*CS+CS/2+29, "genuino mal riconosciuto: FRR alta", 12, None, "#444", "start", halo=False)
# barra colori
for q in range(5):
    mat_cell(o, xr+q*34, Y0-28, 34, 12, heat(0.15 + q*0.175)[0], None, stroke="#ffffff")
txt(o, xr, Y0-33, "score basso", 11, None, "#555", "start", halo=False)
txt(o, xr+170, Y0-33, "alto", 11, None, "#555", "end", halo=False)
save(o, 'fig10_matrice_lupi_agnelli.svg')
```

<img src="./img/fig10_matrice_lupi_agnelli.svg" alt="Matrice s(j,k) con goat, lamb e wolf" style="display:block; margin:1.5em auto; max-width:100%;">

Per ogni utente si calcolano medie e deviazioni standard, e si confrontano con quelle della popolazione ($\mu_G,\sigma_G,\mu_I,\sigma_I$):

$$
\mu_G(k)=\mathbb{E}[G_k],\quad \mu_I(k)=\mathbb{E}[I_k],\quad
z_G(k)=\frac{\mu_G(k)-\mu_G}{\sigma_G},\quad
z_I(k)=\frac{\mu_I(k)-\mu_I}{\sigma_I}
$$

Un utente è "anomalo" quando $|z|$ supera una soglia $\tau$ (tipicamente un percentile, es. 5–10% estremo).

### 11.2 Il piano $(\mu_G,\ \mu_I)$ — Yager e Dunstone

Ogni utente è un punto $(\mu_G(k),\mu_I(k))$ nel piano; le medie di popolazione dividono il piano in quattro quadranti.

**Il piano $(\mu_G,\mu_I)$.** Ogni punto è un utente (dati simulati). Le due medie di popolazione dividono il piano in quattro quadranti: doves in basso a destra, chameleons in alto a destra, phantoms in basso a sinistra, worms in alto a sinistra; le sheep stanno vicino al centro.

```python
# =====================================================================
# 21 — Piano (μ_G, μ_I): Yager e Dunstone  (richiede 01, 05)
# Produce: fig8_piano_animali.svg
# =====================================================================
random.seed(11)
MG_POP, MI_POP = 0.62, 0.38                 # medie di popolazione
clusters = {                                # (centro μ_G, centro μ_I, n, spread, colore)
    "Doves":      (0.84, 0.16, 9, 0.035, "#0ea5e9"),
    "Chameleons": (0.84, 0.60, 8, 0.035, "#f59e0b"),
    "Phantoms":   (0.40, 0.16, 8, 0.035, "#8b5cf6"),
    "Worms":      (0.40, 0.60, 8, 0.035, "#dc2626"),
    "Sheep":      (MG_POP, MI_POP, 26, 0.045, "#6b7280"),
}
W, H = 920, 660
L, R, TOP, BOT = 100, 840, 100, 560
XMIN, XMAX, YMIN, YMAX = 0.25, 1.0, 0.0, 0.75
sx = lambda v: L + (v-XMIN)/(XMAX-XMIN)*(R-L)
sy = lambda v: BOT - (v-YMIN)/(YMAX-YMIN)*(BOT-TOP)
xm, ym = sx(MG_POP), sy(MI_POP)
o = new_svg(W, H)
header(o, W, "Il piano (μ_G, μ_I): la classificazione di Yager e Dunstone",
       "Ogni punto è un utente k · le medie di popolazione dividono il piano in quattro quadranti")
quad = [(L, TOP, xm-L, ym-TOP, "#fee2e2"), (xm, TOP, R-xm, ym-TOP, "#fef3c7"),
        (L, ym, xm-L, BOT-ym, "#ede9fe"), (xm, ym, R-xm, BOT-ym, "#e0f2fe")]
for x, y, w, h, c in quad:
    o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{c}" fill-opacity="0.7"/>')
frame(o, L, R, TOP, BOT, "μ_G(k)  →  utente più facilmente riconosciuto", "μ_I(k)  →  utente più simile agli altri")
o.append(f'<line x1="{xm:.1f}" y1="{TOP}" x2="{xm:.1f}" y2="{BOT}" stroke="#111" stroke-width="1.6" stroke-dasharray="7,5"/>')
o.append(f'<line x1="{L}" y1="{ym:.1f}" x2="{R}" y2="{ym:.1f}" stroke="#111" stroke-width="1.6" stroke-dasharray="7,5"/>')
txt(o, xm, BOT+18, "μ_G popolazione", 11.5, None, "#111", halo=False)
txt(o, L+6, ym-6, "μ_I popolazione", 11.5, None, "#111", "start", halo=False)
for name, (cx0, cy0, n, sp, col) in clusters.items():
    for _ in range(n):
        px, py = random.gauss(cx0, sp), random.gauss(cy0, sp*0.9)
        o.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="5.5" fill="{col}" fill-opacity="0.85" stroke="#fff" stroke-width="1"/>')
labels_q = [("Worms", "pochi tratti distintivi, facili da impersonare", L+10, TOP+22, "start", "#b91c1c"),
            ("Chameleons", "pochi FR, ma tanti FA (tratti generici)", R-10, TOP+22, "end", "#b45309"),
            ("Phantoms", "molti FR, pochi FA (enrollment difficile)", L+10, BOT-26, "start", "#6d28d9"),
            ("Doves", "i migliori: tratto molto distintivo", R-10, BOT-26, "end", "#0369a1")]
for nm, sub, x, y, anc, col in labels_q:
    txt(o, x, y, nm, 15, 700, col, anc, halo=False)
    txt(o, x, y+16, sub, 11.5, None, "#333", anc, halo=False)
txt(o, xm+14, ym+34, "Sheep", 14, 700, "#374151", "start")
txt(o, xm+14, ym+50, "comportamento medio", 11.5, None, "#333", "start")
save(o, 'fig8_piano_animali.svg')
```

<img src="./img/fig8_piano_animali.svg" alt="Piano di Yager-Dunstone con gli animali" style="display:block; margin:1.5em auto; max-width:100%;">

| Categoria | $\mu_G(k)$ | $\mu_I(k)$ | Condizione | Effetto |
|---|---|---|---|---|
| **Doves** (colombe) | alto | basso | $z_G>\tau,\ z_I<-\tau$ | I migliori: tratto molto distintivo, raramente causano errori |
| **Chameleons** | alto | alto | $z_G>\tau,\ z_I>\tau$ | Raramente causano FR, ma facilmente causano FA (tratti generici) |
| **Phantoms** | basso | basso | $z_G<-\tau,\ z_I<-\tau$ | Causano FR, raramente FA (difficoltà di enrollment/estrazione feature) |
| **Worms** (vermi) | basso | alto | $z_G<-\tau,\ z_I>\tau$ | I peggiori: pochi tratti distintivi, facili da impersonare |
| **Sheep** (pecore) | ≈ media | ≈ media | $\lvert z\rvert\le\tau$ | Comportamento normale/buono: la maggioranza della popolazione |

### 11.3 Gli animali di Doddington basati sull'asimmetria

Questi tre animali non guardano solo $\mu_I$ complessivo, ma **quale parte** di $I_k$ è anomala:

| Categoria | Parte anomala | Condizione | Effetto |
|---|---|---|---|
| **Goats** (capre) | $G_k$ (diagonale) | $z_G(k)<-\tau$ | $FRR$ più alta della media: sono mal riconosciuti |
| **Lambs** (agnelli) | $I_k^{\mathrm{vit}}$ (colonna) | $\mu(I_k^{\mathrm{vit}})$ alto | Facilmente impersonabili → più $FA$ **contro di loro** |
| **Wolves** (lupi) | $I_k^{\mathrm{att}}$ (riga) | $\mu(I_k^{\mathrm{att}})$ alto | Bravi a impersonare → causano $FA$ **verso gli altri** |

> Doddington descrive quattro categorie: **sheep** (la maggioranza, comportamento normale), **goats**, **lambs** e **wolves**. Le sheep sono il riferimento rispetto al quale le altre tre risultano anomale.

Perché la distinzione è importante: un agnello è una **vittima** (la colonna $k$ della matrice è chiara), un lupo è un **attaccante** (la riga $k$ è chiara). Lo stesso utente può essere entrambe le cose, o nessuna delle due.

### 11.4 Impatto sulle prestazioni

Per una soglia $t$, gli errori *per utente* sono (accettazione se $s>t$):

$$
FRR_k(t)=P\big(s\le t \mid s\in G_k\big)\qquad
FAR_k(t)=P\big(s> t \mid s\in I_k\big)
$$

Sotto un'approssimazione gaussiana $G_k\sim\mathcal N(\mu_G(k),\sigma^2)$, $I_k\sim\mathcal N(\mu_I(k),\sigma^2)$:

$$
FRR_k(t)=\Phi\!\left(\frac{t-\mu_G(k)}{\sigma}\right) \qquad
FAR_k(t)=1-\Phi\!\left(\frac{t-\mu_I(k)}{\sigma}\right)
$$

dove $\Phi$ è la CDF della normale standard. La separabilità dell'utente è sintetizzata da

$$
d'_k=\frac{\mu_G(k)-\mu_I(k)}{\sigma}
$$

e le prestazioni **globali** sono la media pesata di quelle individuali:

$$
FRR(t)=\frac{1}{N}\sum_{k}FRR_k(t),\qquad
FAR(t)=\frac{1}{N}\sum_{k}FAR_k(t)
$$

Questo è il punto chiave: un **piccolo numero di utenti** (capre, vermi, camaleonti) può dominare gli errori medi.

**Le stesse $t$ e $\sigma$, otto utenti diversi.** Le aree colorate sono gli errori per utente: arancione = FR (genuini sotto soglia), rosso = FA (impostori sopra soglia). I valori di $FRR_k$ e $FAR_k$ sono calcolati dal codice e coincidono con la tabella del testo. Lambs e wolves hanno gli stessi numeri: cambia solo da quale lato della matrice li si osserva.

```python
# =====================================================================
# 22 — Distribuzioni degli score e tassi di errore per animale  (richiede 01, 05)
# Produce: fig9_distribuzioni_animali.svg
# =====================================================================
T_A, SIG = 0.5, 0.09
zoo = [("Sheep", 0.72, 0.30), ("Doves", 0.88, 0.15), ("Goats", 0.40, 0.30), ("Lambs", 0.72, 0.58),
       ("Wolves", 0.72, 0.58), ("Chameleons", 0.82, 0.60), ("Phantoms", 0.38, 0.18), ("Worms", 0.35, 0.62)]
Phi = NormalDist().cdf

W, H = 920, 720
PW, PH, GAPX, GAPY = 205, 190, 15, 40
X00, Y00 = 30, 90
XMIN, XMAX, YMAX = -0.15, 1.15, 4.7
o = new_svg(W, H)
header(o, W, "Distribuzioni degli score per tipo di utente",
       f"Gaussiane con σ = {SIG} · soglia t = {T_A} (si accetta se s > t) · verde = genuini, blu = impostori")
for idx, (name, mg, mi) in enumerate(zoo):
    px = X00 + (idx % 4)*(PW+GAPX)
    py = Y00 + (idx // 4)*(PH+GAPY+60)
    sx = lambda x, px=px: px + (x-XMIN)/(XMAX-XMIN)*PW
    sy = lambda y, py=py: py + PH - y/YMAX*(PH-30)
    base = py + PH
    frr_k = Phi((T_A-mg)/SIG)
    far_k = 1 - Phi((T_A-mi)/SIG)
    dprime = (mg-mi)/SIG
    rect(o, px-4, py-4, PW+8, PH+90, "#fff", "#e5e7eb", 1.2, 10)
    txt(o, px+PW/2, py+14, name, 14.5, 700, "#111", halo=False)
    o.append(f'<path d="{g_area(mg,SIG,XMIN,T_A,sx,sy,base)}" fill="{C_FR}" fill-opacity="0.85"/>')
    o.append(f'<path d="{g_area(mi,SIG,T_A,XMAX,sx,sy,base)}" fill="{C_FA}" fill-opacity="0.85"/>')
    o.append(f'<path d="{g_line(mg,SIG,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_G}" stroke-width="2.2"/>')
    o.append(f'<path d="{g_line(mi,SIG,XMIN,XMAX,sx,sy)}" fill="none" stroke="{C_I}" stroke-width="2.2"/>')
    o.append(f'<line x1="{px}" y1="{base}" x2="{px+PW}" y2="{base}" stroke="#222" stroke-width="1.2"/>')
    o.append(f'<line x1="{sx(T_A):.1f}" y1="{py+22}" x2="{sx(T_A):.1f}" y2="{base}" stroke="#111" stroke-width="1.6" stroke-dasharray="5,4"/>')
    for v in (0, 0.5, 1.0):
        txt(o, sx(v), base+14, f"{v:g}", 10.5, None, "#555", halo=False)
    txt(o, px+8, base+36, f"μ_G = {mg:.2f}   μ_I = {mi:.2f}   d' = {dprime:.1f}", 11, None, "#333", "start", halo=False)
    txt(o, px+8, base+54, f"FRR_k = {frr_k:.2f}", 12, 700, C_FR_T, "start", halo=False)
    txt(o, px+PW-8, base+54, f"FAR_k = {far_k:.2f}", 12, 700, C_FA_T, "end", halo=False)
    if name in ("Lambs", "Wolves"):
        txt(o, px+PW/2, base+72, "vittima (colonna)" if name == "Lambs" else "attaccante (riga)", 11, None, "#555", halo=False)
ly = H-18
for i, (c, t) in enumerate([(C_FR, "FR: genuini con s ≤ t"), (C_FA, "FA: impostori con s > t")]):
    xx = 270 + i*230
    o.append(f'<rect x="{xx}" y="{ly-11}" width="16" height="16" fill="{c}" fill-opacity="0.85" stroke="{c}"/>')
    txt(o, xx+22, ly+2, t, 12.5, None, "#222", "start", halo=False)
save(o, 'fig9_distribuzioni_animali.svg')
```

<img src="./img/fig9_distribuzioni_animali.svg" alt="Distribuzioni degli score e tassi di errore per animale" style="display:block; margin:1.5em auto; max-width:100%;">

Esempio con $t=0.5$, $\sigma=0.09$ e valori illustrativi:

| Animale | $\mu_G$ | $\mu_I$ | $d'$ | $FRR_k$ | $FAR_k$ | Lettura |
|---|---|---|---|---|---|---|
| Sheep | 0.72 | 0.30 | 4.7 | 0.01 | 0.01 | tutto nella norma |
| Doves | 0.88 | 0.15 | 8.1 | ≈ 0 | ≈ 0 | quasi nessun errore |
| Goats | 0.40 | 0.30 | 1.1 | **0.87** | 0.01 | molti FR, FA normali |
| Lambs | 0.72 | 0.58 (vittima) | 1.6 | 0.01 | **0.81** | molti FA *contro* di loro |
| Wolves | 0.72 | 0.58 (attaccante) | 1.6 | 0.01 | **0.81** | causano FA verso gli altri |
| Chameleons | 0.82 | 0.60 | 2.4 | ≈ 0 | **0.87** | niente FR, ma tanti FA |
| Phantoms | 0.38 | 0.18 | 2.2 | **0.91** | ≈ 0 | tutti i genuini respinti, nessun FA |
| Worms | 0.35 | 0.62 | −3.0 | **0.95** | **0.91** | falliscono su entrambi i fronti |

> Nota: i valori di $FAR_k$ per Lambs e Wolves sono *per interazione*: il lamb subisce gli attacchi, il wolf li produce. Nel conteggio globale i due contribuiscono alla stessa cella della matrice, ma si individuano guardando **colonna** vs **riga**.

### 11.5 Cosa cambia in pratica

- **Aggiungere la soglia non risolve il problema**: abbassare $t$ aiuta capre e phantoms (meno FR) ma peggiora lambs, wolves e chameleons (più FA).
- **Gli animali "peggiori" sono pochi ma pesano molto**: per questo si riportano anche le prestazioni per utente (o percentili), non solo la media.
- **Rimedi tipici**: normalizzazione degli score *per utente* (z-norm, t-norm), soglie utente-specifiche, enrollment multiplo (più template → meno capre), fusione multimodale.

**Prima e dopo la z-norm.** Con una soglia globale l'agnello subisce molti più FA della colomba; dopo $z=(s-\mu_I(k))/\sigma_I(k)$ le distribuzioni impostori coincidono e ogni utente ha lo stesso $FAR_k$. Anche i genuini si spostano, quindi la soglia va ritarata sul nuovo asse.

```python
# =====================================================================
# N13 — z-norm  (richiede 01, 05)
# =====================================================================
random.seed(1)
users = [("U1 · agnello", 0.60, 0.08, "#dc2626"), ("U2 · pecora", 0.45, 0.08, "#2563eb"), ("U3 · colomba", 0.30, 0.08, "#16a34a")]
T_Z = 0.55
ZT = NormalDist().inv_cdf(0.90)
W, H = 920, 620
o = new_svg(W, H)
header(o, W, "Rimedio per lambs e wolves: normalizzazione degli score per utente (z-norm)",
       "Distribuzioni degli score impostori contro il template di ciascun utente · si accetta se s > t")
def panel(x0, title, xmin, xmax, tval, mapf, ticks, xlabel):
    L, R, TOP, BOT = x0, x0+400, 140, 380
    ymax = 5.3 if xmax < 2 else 0.5
    sx = lambda x: L+(x-xmin)/(xmax-xmin)*(R-L)
    sy = lambda y: BOT-y/ymax*(BOT-TOP)
    txt(o, (L+R)/2, 100, title, 14.5, 700, "#111", halo=False)
    for nm, mu, sd, col in users:
        m, s = mapf(mu, sd)
        o.append(f'<path d="{g_area(m, s, tval, xmax, sx, sy, BOT)}" fill="{col}" fill-opacity="0.35"/>')
        o.append(f'<path d="{g_line(m, s, xmin, xmax, sx, sy)}" fill="none" stroke="{col}" stroke-width="2.6"/>')
    frame(o, L, R, TOP, BOT, xlabel, None)
    for v in ticks:
        o.append(f'<line x1="{sx(v):.1f}" y1="{BOT}" x2="{sx(v):.1f}" y2="{BOT+5}" stroke="#222"/>')
        txt(o, sx(v), BOT+20, f"{v:g}", 11.5, None, "#333", halo=False)
    o.append(f'<line x1="{sx(tval):.1f}" y1="{TOP-6}" x2="{sx(tval):.1f}" y2="{BOT}" stroke="#111" stroke-width="2" stroke-dasharray="7,5"/>')
    txt(o, sx(tval), TOP-14, f"soglia = {tval:.2f}", 12.5, 700, "#111", halo=False)
    for k, (nm, mu, sd, col) in enumerate(users):
        m, s = mapf(mu, sd)
        fa = 1-NormalDist(m, s).cdf(tval)
        o.append(f'<rect x="{L+4+k*134}" y="{BOT+62}" width="12" height="12" fill="{col}"/>')
        txt(o, L+22+k*134, BOT+72, nm.split(" · ")[0], 12, 700, col, "start", halo=False)
        txt(o, L+22+k*134, BOT+90, f"FAR_k = {fa:.1%}", 12.5, 700, "#111", "start", halo=False)
panel(40, "Prima: score grezzi, soglia globale", 0, 1.0, T_Z, lambda m, s: (m, s), [0, 0.2, 0.4, 0.6, 0.8, 1.0], "score s")
panel(480, "Dopo: z-norm  z = (s − μ_I(k)) / σ_I(k)", -4, 4, ZT, lambda m, s: (0.0, 1.0), [-4, -2, 0, 2, 4], "score normalizzato z")
txt(o, 240, 540, "L'agnello (U1) subisce tanti FA, la colomba quasi nessuno:", 12.5, None, "#333", halo=False)
txt(o, 240, 558, "una sola soglia non è equa per tutti.", 12.5, None, "#333", halo=False)
txt(o, 680, 540, "Le tre distribuzioni impostori coincidono: ogni utente ha lo", 12.5, None, "#333", halo=False)
txt(o, 680, 558, "stesso FAR_k. (anche i genuini si spostano: la soglia va ritarata)", 12.5, None, "#333", halo=False)
txt(o, W/2, 600, "Altri rimedi: t-norm, soglie utente-specifiche, enrollment multiplo (meno capre), fusione multimodale.", 12, None, "#555", halo=False)
save(o, 'n13_znorm_per_utente.svg')
```
<img src="./img/n13_znorm_per_utente.svg" alt="z-norm per utente" style="display:block; margin:1.5em auto; max-width:100%;">
  
---

## 12. Decidability Value

Misura alternativa (usata nelle competizioni di iris recognition) che tiene conto insieme dei contributi di FA e FR.

Si costruiscono due insiemi di distanze: **D^I** (intra-class, stesso soggetto) e **D^E** (inter-class, soggetti diversi).

$$
\text{Decidability} = \frac{\overline{D^E} - \overline{D^I}}{\sigma}
$$

dove σ è una misura di deviazione standard normalizzante calcolata sui due insiemi.

---

## 13. Reliability of an Identification System (SRR)

La **reliability** di una singola risposta è diversa dall'accuratezza globale del sistema (FAR/FRR/CMS): riguarda **quanto ci si può fidare di una specifica decisione**.

### 13.1 Catena logica
1. Valutare la **qualità della probe** prima del riconoscimento (pre-processing).
2. Eseguire il riconoscimento.
3. Valutare la **reliability della risposta** dopo il riconoscimento.

### 13.2 Misure di qualità dell'immagine (esempio volto)

- **SP (Score Pose):**
$$
SP = \alpha\cdot(1-\text{roll}) + \beta\cdot(1-\text{yaw}) + \gamma\cdot(1-\text{pitch})
$$
  - **Roll** = rotazione attorno all'asse x (facile da correggere: angolo tra i centri degli occhi).
  - **Yaw** = rotazione attorno all'asse z (si perde parte del volto; corregibile solo se moderato, tramite simmetria dₗ vs dᵣ).
  - **Pitch** = rotazione attorno all'asse y (difficile da correggere, basata sul rapporto tra distanza occhi-naso e naso-mento).

- **SI (omogeneità dei livelli di grigio):**
$$
SI = 1 - F(\text{std}(mc))
$$

- **SY (simmetria del volto):**
$$
SY = \sum_{(i,j)\in X} sym(P_i, P_j)
$$

Una buona misura di qualità deve **ridurre l'EER scartando il minor numero possibile** di campioni (bilanciare accuratezza e throughput del sistema).

### 13.3 System Response Reliability (SRR)

Indice **srr ∈ [0,1]** che misura la capacità di separare genuini da impostori **su base di singola probe**, sfruttando la nozione di "confusione" tra i candidati nella lista ordinata.

**Relative Distance:**

$$
\varphi(p) = \frac{F\big(d(p,g_{i_2})\big) - F\big(d(p,g_{i_1})\big)}{F\big(d(p,g_{i_{|G|}})\big)}
$$

- Numeratore: differenza tra la prima e la seconda distanza (quanto sono vicini i primi due candidati).
- Denominatore: distanza massima nella lista.
- **Più basso è φ, peggiore è l'affidabilità** (i primi due candidati sono troppo simili tra loro rispetto alla distanza massima).

**Density Ratio** (meno sensibile agli outlier, generalmente migliore della Relative Distance):

$$
\varphi(p) = 1 - \frac{|N_b|}{|N|}, \qquad N_b = \{g_{i_k} \in G \mid F(d(p,g_{i_k})) < 2\cdot F(d(p,g_{i_1}))\}
$$

Conta quanti template hanno distanza dalla probe inferiore al doppio della prima distanza (nube di candidati "vicini" al primo). **Nube meno affollata → maggiore affidabilità → φ più alto è meglio.**

**Valore critico φₖ:** soglia (analoga concettualmente all'EER) che separa risposte affidabili da non affidabili, minimizzando le stime errate di φ.

**Normalizzazione finale (SRR):**

$$
S(\varphi(p), \overline{\varphi}) =
\begin{cases}
1 - \overline{\varphi} & \text{se } \varphi(p) > \overline{\varphi} \\
\overline{\varphi} & \text{altrimenti}
\end{cases}
$$

$$
SRR = \frac{\varphi(p) - \overline{\varphi}}{S(\overline{\varphi})}
$$

Con **due soglie** finali nel sistema: una per l'accettazione (FAR/FRR) e una per la reliability della risposta — quest'ultima permette di **rifiutare un'identificazione anche se formalmente accettata**, se non ritenuta affidabile.

### 13.4 Stima automatica della soglia di reliability (th)

Il **reliability threshold** ($th$) può essere stimato automaticamente sfruttando un certo numero $M$ di osservazioni successive dello stesso soggetto (es. $M$ frame consecutivi di un video, o $M$ acquisizioni ripetute). Si vuole una soglia che rifletta un compromesso tra due esigenze:

- **Media alta** dei valori di SRR osservati → il sistema è generalmente affidabile.
- **Varianza bassa** → il sistema è stabile (le risposte non oscillano troppo tra affidabili e non affidabili).

Per l'i-esimo soggetto/serie di osservazioni $S_i$, la soglia si stima come:

$$
th_i = \left|\frac{\, E[\overline{S_i}]^2 - \sigma[\overline{S_i}] \,}{E[\overline{S_i}]}\right|
$$

dove:
- $E[\overline{S_i}]$ è la **media** dei valori di reliability osservati sulle $M$ osservazioni;
- $\sigma[\overline{S_i}]$ è la loro **deviazione standard (varianza)**.

> Intuizione: il numeratore penalizza sia una media bassa (poco affidabile) sia un'alta variabilità (poco stabile); dividere per la media normalizza il risultato. Una soglia $th_i$ così calcolata si adatta automaticamente al comportamento tipico di quel soggetto/sistema, invece di usare un valore fisso globale.

---

## 14. Metrica delle distanze — proprietà formali

Una **metrica** su un insieme X è una funzione `d: X × X → ℝ` che soddisfa, per ogni x, y, z ∈ X:

1. **Non negatività (separazione):** d(x,y) ≥ 0
2. **Identità degli indiscernibili:** d(x,y) = 0 ⟺ x = y
3. **Simmetria:** d(x,y) = d(y,x)
4. **Disuguaglianza triangolare:** d(x,z) ≤ d(x,y) + d(y,z)

Una **semimetrica** soddisfa solo le prime 3 proprietà (non necessariamente la disuguaglianza triangolare) — è il caso tipico delle matrici di distanza biometriche, che sono **simmetriche con diagonale nulla**.

Una misura non simmetrica può essere resa simmetrica calcolando la media nelle due direzioni:
$$
d_{sym}(A,B) = \frac{d(A,B) + d(B,A)}{2}
$$

---

## 15. Considerazioni per un confronto affidabile tra sistemi

Per confrontare sistemi in modo equo bisogna considerare:
- Numero e caratteristiche dei dataset usati
- Dimensione delle immagini (risoluzione)
- Dimensione relativa di probe e gallery
- Quantità/qualità delle variazioni tollerate
- Interoperabilità (generalizzazione cross-dataset)

> I dataset pubblici sono stati fondamentali per il progresso della ricerca (misurare e confrontare algoritmi in modo equo), ma rischiano di diventare "mondi chiusi" che restringono la ricerca a battere un singolo numero di benchmark, perdendo di vista lo scopo originale (citando Torralba).

I FoM (**Figures of Merit**: FAR, FRR, EER, ROC, CMS, ecc.) sono misure **ex-post**, legate al dataset/ground truth usato: nel mondo reale il contesto operativo può cambiare (utenti non familiari col sistema, condizioni diverse), quindi la distribuzione degli score può variare.

---

## 16. Aggiornamento del template (Template Update)

Per migliorare l'affidabilità nel tempo:
- Aggiungere nuovi template in gallery quando il riconoscimento è affidabile (mantenendo anche i vecchi → utile contro le intra-class variation).
- Necessario in caso di **invecchiamento** del tratto biometrico.
- Gestione delle nuove tecnologie (es. sensori a risoluzione più alta).
- Modalità: **Supervised** (un operatore conferma) o **Semi-Supervised** (automatica, tramite confronto statistico).
- Selezione dei template più rappresentativi: **Online** (appena arrivano nuovi dati) o **Offline** (dopo un certo periodo di raccolta).

---

## 17. Approfondimenti e domande frequenti aggiuntive

### 17.1 Come scegliere il sistema migliore tra due candidati?

- **A parità di FAR:** si sceglie il sistema con il **FRR inferiore** a quella stessa soglia (più utenti genuini vengono accettati, GAR più alto).
- **A parità di FRR:** si sceglie il sistema con il **FAR inferiore** (maggiore sicurezza contro gli impostori).
- **In senso globale (nessun vincolo di parità):** si confronta l'**AUC** delle rispettive curve ROC. Si seleziona il sistema con l'AUC maggiore, perché indica prestazioni complessivamente migliori su tutto il range di soglie possibili.

### 17.2 Detection Rate vs Identification Rate

- **Detection Rate (DR):** probabilità che un soggetto genuino venga **rilevato come presente** in gallery (il sistema capisce "è uno dei nostri"). Nell'open set coincide con $DIR(t,1)$ considerato come semplice "detezione" (senza guardare se l'identità restituita è esatta).
- **Identification Rate (IR):** probabilità che un soggetto genuino venga **correttamente identificato**, cioè che il sistema scelga proprio l'identità giusta. Nel closed set corrisponde al Rank-1 Identification Rate, cioè $CMS(1)$.

> La distinzione è utile perché nel closed set il problema è solo "trovare chi è" (esiste solo detezione implicita, dato che tutti sono in gallery), mentre nell'open set c'è anche il problema preliminare di "riconoscere se il soggetto appartiene al sistema" prima ancora di provare a identificarlo.

### 17.3 Perché l'accuracy "classica" del Machine Learning non basta?

Se si calcolassero i tassi di errore dividendo semplicemente per il **numero totale di probe** (come fa l'accuracy standard in ML), si otterrebbe un valore che **nasconde comportamenti critici asimmetrici**. 

**Esempio:** un sistema può mostrare un'accuracy dell'80% pur accettando il 100% degli impostori (FAR = 1, gravissimo per la sicurezza) oppure rifiutando il 100% degli utenti legittimi (FRR = 1, sistema inutilizzabile). L'accuracy aggrega tutto in un unico numero medio e può mascherare uno dei due errori. Le metriche biometriche FAR/FRR, invece, **separano sempre genuini e impostori**, rendendo visibile ogni comportamento anomalo.

### 17.4 FAR/FRR vs Precision/Recall (confronto con le metriche ML)

Le metriche ML classiche si basano sui **risultati corretti** (veri positivi):

$$
Precision = \frac{TP}{TP+FP} = \frac{GA}{GA+FA}
\qquad\qquad
Recall = \frac{TP}{TP+FN} = \frac{GA}{GA+FR}
$$

Le metriche biometriche si basano invece sugli **errori**, sempre separati per categoria di utente:

$$
FRR = \frac{FR}{GA+FR} = \frac{FN}{TP+FN}
\qquad\qquad
FAR = \frac{FA}{FA+GR} = \frac{FP}{FP+TN}
$$

> Le due coppie partono da prospettive opposte e complementari: Precision/Recall valutano **quante risposte positive sono corrette**; FAR/FRR valutano **quanto spesso il sistema sbaglia** su ciascuna categoria di utenti (genuini vs impostori). Per questo in ambito biometrico si preferisce sempre riportare FAR/FRR (o la coppia EER/ROC) piuttosto che una singola accuracy aggregata.

### 17.5 Relazione tra CMC, ROC, FAR e FRR

Bolle et al. (2005) hanno dimostrato un risultato importante: quando un matcher 1:1 viene usato per ordinare i candidati (cioè per costruire la lista ordinata usata nel closed/open set), la **curva CMC è direttamente derivabile da FAR e FRR** — non aggiunge quindi nuova informazione statistica rispetto alla curva ROC, che già mostra il trade-off FAR/FRR al variare della soglia.

> In sintesi: **CMC, ROC, FAR e FRR sono tutte rappresentazioni diverse della stessa informazione**, contenuta a monte nella Distance/Similarity Matrix (DM) calcolata tra probe e gallery. Cambia solo il modo in cui questa informazione viene "letta" e visualizzata (per rango vs per soglia).

### 17.6 Regola pratica sulla soglia di accettazione

La soglia $t$ decide se accettare o respingere un match, e la regola dipende dal tipo di score usato:

- se si usa una **distanza** → si accetta se $\text{distanza} \le t$ (più bassa = più simile);
- se si usa una **similarità** → si accetta se $\text{similarità} \ge t$ (più alta = più simile).

Cambiando $t$, FAR e FRR si muovono sempre in direzioni opposte: soglia più **alta/restrittiva** → FAR minore ma FRR maggiore; soglia più **bassa/permissiva** → FRR minore ma FAR maggiore. Non esiste una soglia "oggettivamente migliore": si sceglie il punto operativo in base al tipo di applicazione (sicurezza vs comodità utente), valutando le prestazioni su una griglia di soglie tramite le curve ROC/DET.

### 17.7 Tabella riassuntiva: Verification vs Closed Set vs Open Set

| Aspetto | Verification (1:1) | Closed Set (1:N) | Open Set (1:N) |
|---|---|---|---|
| Obiettivo | Validare l'identità dichiarata | Trovare l'identità corretta (soggetto sempre registrato) | Rilevare se il soggetto è registrato e, in caso, identificarlo |
| Rivendicazione (claim) | Sì | No | No |
| Impostori possibili | Sì | No | Sì |
| Soglia di accettazione | Sì | No | Sì |
| Errori possibili | GA, FR, GR, FA | Solo False Rejection (rango > 1) | Correct D&I, False Rejection, False Acceptance, Genuine Reject |
| Metriche principali | FAR, FRR, EER, ROC, DET | CMS, CMC, Recognition Rate (RR) | DIR al rango k, FAR, FRR, FNIR/FPIR |

---

## 18. Riepilogo — mappa concettuale delle formule chiave

$$
\boxed{FAR(t) = \frac{FA}{TI}} \qquad
\boxed{FRR(t) = \frac{FR}{TG}} \qquad
\boxed{GAR = \frac{GA}{TG} = 1-FRR} \qquad
\boxed{GRR = \frac{GR}{TI} = 1-FAR}
$$

$$
\boxed{EER: \ FAR(t^*) = FRR(t^*)}
$$

$$
\boxed{DIR(t,1) = \frac{\text{correct detect \& identify a rango 1}}{|P_G|}} \qquad
\boxed{FRR(t) = 1 - DIR(t,1)}
$$

$$
\boxed{CMS(k) = P(\text{identità corretta nei primi } k \text{ posti})}, \quad CMS(1) = \text{Recognition Rate}
$$

**Schema dei task e loro errori possibili:**

| Task | Claim identità | Soglia | Errori possibili |
|---|---|---|---|
| **Verifica** | Sì | Sì | GA, FR, GR, FA |
| **Identificazione Open Set** | No | Sì | Correct D&I, False Rejection, False Acceptance, Genuine Reject |
| **Identificazione Closed Set** | No | No | Solo False Rejection (nessuna FA) |

