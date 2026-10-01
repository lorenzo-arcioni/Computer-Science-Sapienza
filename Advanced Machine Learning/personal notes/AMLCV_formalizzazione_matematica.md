# Advanced Machine Learning and Computer Vision — Formalizzazione matematica

**Contenuto:** *02 – Recap on Image Classification and Neural Networks* e *03 – Recap on Convolutional Neural Networks* (Prof. F. Galasso, Sapienza).
**Scopo:** formalizzare ogni concetto delle slide, esplicitare tutti i passaggi e aggiungere le derivazioni che nelle slide sono solo accennate.

> Le parti marcate con **[Aggiunta]** non sono nelle slide ma servono a completare il ragionamento.
> Le parti marcate con **[Nota]** segnalano imprecisioni o convenzioni delle slide.

---

## 0. Notazione

| Simbolo | Significato |
|---|---|
| $x \in \mathbb{R}^D$ | input (immagine "srotolata" in un vettore) |
| $y_i \in \{1,\dots,C\}$ | etichetta della classe corretta dell'esempio $i$ |
| $W$, $b$ | pesi e bias |
| $s = f(x;W)$ | vettore degli *score* (logit), $s\in\mathbb{R}^C$ |
| $p$ | vettore delle probabilità predette (softmax di $s$) |
| $L_i$ | loss del singolo esempio; $L$ loss totale |
| $N$ | numero di esempi di training |
| $\odot$ | prodotto elementwise (Hadamard) |
| $\mathbb{1}[\cdot]$ | funzione indicatrice |
| $\otimes$ | convoluzione (slide) |

---

# PARTE I — Classificazione di immagini e reti neurali (PDF 02)

## 1. Machine Learning

**Arthur Samuel (1959).** Il ML è il campo di studio che dà ai computer la capacità di apprendere senza essere programmati esplicitamente.

**Tom Mitchell (1998), problema di apprendimento ben posto.** Un programma impara dall'esperienza $E$ rispetto a un compito $T$ e a una misura di prestazione $P$ se la sua prestazione su $T$, misurata da $P$, migliora con $E$:

$$
P\big(T \mid E_2\big) > P\big(T \mid E_1\big)\quad\text{se } E_2 \text{ contiene più (o migliore) esperienza di } E_1 .
$$

Nel nostro caso:

- $T$ = classificare immagini;
- $E$ = dataset $\{(x_i,y_i)\}_{i=1}^N$;
- $P$ = accuratezza (o, come surrogato differenziabile, l'opposto della loss).

## 2. Computer Vision e immagini digitali

Le slide distinguono tre prospettive: **scienza** (modello computazionale della visione umana, come nella gerarchia V1→V2→V4→PIT→AIT), **ingegneria** (sistemi che percepiscono il mondo, ad es. rilevare pedoni) e **applicazioni** (imaging medico, sorveglianza, intrattenimento, grafica, automotive).

**Processo di formazione dell'immagine** (slide "Digital Images"): mondo → camera (modello pinhole) → digitalizzatore → immagine digitale.

**[Aggiunta] Modello pinhole.** Un punto 3D $(X,Y,Z)$ è proiettato sul piano immagine a distanza focale $f$:

$$
u = f\,\frac{X}{Z},\qquad v = f\,\frac{Y}{Z}.
$$

**Digitalizzazione = campionamento + quantizzazione.** Il piano immagine viene campionato su una griglia $H\times W$ e ogni valore d'intensità viene quantizzato (tipicamente 8 bit):

$$
I \in \{0,\dots,255\}^{H\times W\times 3}.
$$

Un'immagine a colori è quindi un **tensore** di ordine 3 (altezza × larghezza × canali). Per CIFAR-like $32\times32\times3$: $32\cdot32\cdot3 = 3072$ numeri.

## 3. Classificazione di immagini

Dato un insieme finito di etichette $\mathcal{Y}=\{\text{dog},\text{cat},\text{truck},\text{plane},\dots\}$, si cerca una funzione

$$
F:\ \mathbb{R}^{H\times W\times 3}\to\mathcal{Y}.
$$

Non si scrive $F$ a mano: si parametrizza uno *score function* $f(x;W)$ e si impara $W$ dai dati.

## 4. Classificatore lineare

$$
\boxed{\,f(x,W)=Wx+b\,}
$$

Dimensioni (slide, immagine $32\times32\times3$, 10 classi):

$$
\underbrace{f}_{10\times1}=\underbrace{W}_{10\times3072}\ \underbrace{x}_{3072\times1}+\underbrace{b}_{10\times1}.
$$

**Numero di parametri:** $10\cdot3072+10=30\,730$.

**Interpretazione.** La riga $k$ di $W$, $w_k^\top$, è un *template* per la classe $k$: lo score è $s_k=w_k^\top x+b_k$, un prodotto scalare fra immagine e template.

## 5. Classificatore Softmax (regressione logistica multinomiale)

### 5.1 Da score a probabilità

Vogliamo interpretare $s=f(x_i;W)$ come **log-probabilità non normalizzate** (*logit*). Servono due proprietà:

1. probabilità $\ge 0$ → si applica $\exp$ (mappa $\mathbb{R}\to\mathbb{R}_{>0}$);
2. somma $=1$ → si normalizza dividendo per la somma.

$$
\boxed{P(Y=k\mid X=x_i)=\frac{e^{s_k}}{\sum_{j=1}^{C}e^{s_j}}}\qquad\text{(funzione softmax)}
$$

### 5.2 Esempio numerico delle slide

Classi (cat, car, frog), logit $s=(3.2,\;5.1,\;-1.7)$.

| Passo | cat | car | frog |
|---|---|---|---|
| logit $s_k$ | 3.2 | 5.1 | −1.7 |
| $\exp(s_k)$ | 24.5 | 164.0 | 0.18 |
| somma | | 188.7 | |
| $p_k=\exp(s_k)/188.7$ | **0.13** | 0.87 | 0.00 (≈0.001) |
| distribuzione corretta $q$ (one-hot) | 1.00 | 0.00 | 0.00 |

L'immagine è un gatto ma il modello dà 0.13 al gatto e 0.87 all'auto: la loss sarà alta.

### 5.3 Loss di cross-entropy

Per l'esempio $i$ con classe corretta $y_i$:

$$
\boxed{L_i=-\log P(Y=y_i\mid X=x_i)}
$$

**Perché.** Minimizzare $L_i$ equivale a massimizzare la log-verosimiglianza della classe corretta (*maximum likelihood*).

Nell'esempio: $L_i=-\ln(0.13)\approx 2.04$.

**Legame con la cross-entropy fra distribuzioni.** Sia $Q$ la distribuzione "vera" (one-hot su $y_i$) e $P$ quella predetta (come nella slide: "compare"):

$$
H(Q,P)=-\sum_k Q_k\log P_k = \underbrace{-\sum_k Q_k\log Q_k}_{H(Q)}+\underbrace{\sum_k Q_k\log\frac{Q_k}{P_k}}_{D_{KL}(Q\|P)} .
$$

*Passaggio:* si somma e sottrae $\sum_k Q_k\log Q_k$. Con $Q$ one-hot, $H(Q)=0$, quindi
$H(Q,P)=D_{KL}(Q\|P)=-\log P_{y_i}=L_i$. Minimizzare la cross-entropy = avvicinare la distribuzione predetta a quella vera in senso KL.

Espandendo il logaritmo si ottiene la forma numericamente utile:

$$
L_i=-s_{y_i}+\log\sum_j e^{s_j}.
$$

**[Aggiunta] Stabilità numerica.** $e^{s}$ può andare in overflow; poiché il softmax è invariante a traslazioni ($s\to s-c$), si usa $c=\max_j s_j$.

### 5.4 [Aggiunta] Gradiente di softmax + cross-entropy

Sia $p_k=e^{s_k}/Z$, $Z=\sum_j e^{s_j}$. Da $L_i=-s_{y}+\log Z$:

$$
\frac{\partial L_i}{\partial s_k}=-\mathbb{1}[k=y]+\frac{1}{Z}\frac{\partial Z}{\partial s_k}=-\mathbb{1}[k=y]+\frac{e^{s_k}}{Z}
$$

$$
\boxed{\ \nabla_s L_i = p-e_{y}\ }
$$

dove $e_y$ è il vettore one-hot. Nell'esempio: $p-e_y=(0.13-1,\,0.87,\,0.00)=(-0.87,\,0.87,\,0.00)$. Il gradiente spinge verso l'alto lo score della classe corretta e verso il basso quello delle altre.

Per il classificatore lineare, con $s=Wx+b$ e la regola della catena (sez. 9):

$$
\nabla_W L_i=(p-e_y)\,x^\top,\qquad \nabla_b L_i=p-e_y .
$$

## 6. Regolarizzazione

$$
\boxed{L(W)=\underbrace{\frac1N\sum_{i=1}^{N}L_i\big(f(x_i,W),y_i\big)}_{\text{data loss}}+\lambda\,\underbrace{R(W)}_{\text{regolarizzazione}}}
$$

- **Data loss:** le predizioni devono combaciare con i dati di training.
- **Regolarizzazione:** impedisce al modello di andare *troppo* bene sul training (overfitting).
- $\lambda$ è un iperparametro (forza della regolarizzazione).

| Nome | $R(W)$ | Gradiente |
|---|---|---|
| L2 | $\sum_k\sum_l W_{k,l}^2$ | $2W$ (*weight decay*) |
| L1 | $\sum_k\sum_l \lvert W_{k,l}\rvert$ | $\operatorname{sign}(W)$ |
| Elastic net | $\sum_k\sum_l \big(\beta W_{k,l}^2+\lvert W_{k,l}\rvert\big)$ | $2\beta W+\operatorname{sign}(W)$ |

Metodi più complessi (citati): Dropout, Batch Normalization, stochastic depth, fractional pooling. Indicazione della slide: **preferire modelli grandi, fortemente regolarizzati.**

**[Aggiunta] Esempio: perché L2 preferisce pesi distribuiti.** Sia $x=(1,1,1,1)$, $w_1=(1,0,0,0)$, $w_2=(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$. Entrambi danno $w^\top x=1$ (stessa data loss), ma $R_{L2}(w_1)=1$ e $R_{L2}(w_2)=4\cdot\tfrac1{16}=0.25$: L2 sceglie $w_2$, che usa *tutte* le feature. L1 (con $\|w_1\|_1=\|w_2\|_1=1$) è indifferente, e tende in generale a soluzioni sparse.

## 7. Stochastic Gradient Descent (SGD)

Loss e gradiente completi:

$$
L(W)=\frac1N\sum_{i=1}^N L_i(x_i,y_i,W)+\lambda R(W),\qquad
\nabla_W L(W)=\frac1N\sum_{i=1}^N \nabla_W L_i(x_i,y_i,W)+\lambda\nabla_W R(W).
$$

La somma completa è costosa per $N$ grande. Si approssima con un **minibatch** $B\subset\{1..N\}$ di dimensione $m$ (32/64/128 comuni):

$$
\widehat{\nabla}_W L=\frac1m\sum_{i\in B}\nabla_W L_i+\lambda\nabla_W R .
$$

**[Aggiunta] Lo stimatore è non distorto.** Se $B$ è campionato uniformemente,
$\mathbb{E}\big[\frac1m\sum_{i\in B}\nabla L_i\big]=\frac1m\sum_{i\in B}\mathbb{E}[\nabla L_i]=\frac1m\cdot m\cdot\frac1N\sum_{i=1}^N\nabla L_i=\frac1N\sum_i\nabla L_i$.
La varianza scala come $1/m$.

**Aggiornamento** (codice della slide, `weights += -step_size * weights_grad`):

$$
W\leftarrow W-\eta\,\widehat{\nabla}_W L ,
$$

con $\eta$ = *step size* / learning rate. (Il codice della slide campiona 256 esempi; il testo indica 32/64/128 come valori comuni: è solo un esempio.)

## 8. Grafi computazionali

Una funzione complessa si scompone in operazioni elementari (nodi), collegate da archi che trasportano valori.

**Grafo del classificatore lineare** (slide "Computational graphs + Backpropagation"):

$$
x,W\ \xrightarrow{\ *\ }\ s\ \xrightarrow{\text{cross-entropy}}\ L_{\text{data}}\ \xrightarrow{\ +\ }\ L,\qquad W\xrightarrow{R}R(W)\to(+).
$$

$W$ ha due usi (nel prodotto e nella regolarizzazione), quindi il suo gradiente sarà la **somma** dei contributi dei due cammini (cfr. *copy gate*, sez. 9.4).

**[Nota]** Nella slide il riquadro rosso riporta $L_i=\frac1N\sum_i L_i(\dots)$: la quantità a sinistra è in realtà la *data loss media*, non la loss del singolo esempio.

## 9. Backpropagation

### 9.1 Esempio semplice: $f(x,y,z)=(x+y)z$, con $x=-2,\ y=5,\ z=-4$

**Forward pass.** Si introduce $q=x+y$, quindi $f=qz$:

$$
q=-2+5=3,\qquad f=3\cdot(-4)=-12 .
$$

**Derivate locali.**

$$
q=x+y:\ \ \frac{\partial q}{\partial x}=1,\ \frac{\partial q}{\partial y}=1;\qquad
f=qz:\ \ \frac{\partial f}{\partial q}=z,\ \frac{\partial f}{\partial z}=q .
$$

**Backward pass** (dal fondo verso l'inizio; si vogliono $\partial f/\partial x,\partial f/\partial y,\partial f/\partial z$):

| Passo | Calcolo | Valore |
|---|---|---|
| 1 | $\dfrac{\partial f}{\partial f}$ | $1$ |
| 2 | $\dfrac{\partial f}{\partial z}=q$ | $3$ |
| 3 | $\dfrac{\partial f}{\partial q}=z$ | $-4$ |
| 4 | $\dfrac{\partial f}{\partial y}=\dfrac{\partial f}{\partial q}\dfrac{\partial q}{\partial y}=(-4)(1)$ | $-4$ |
| 5 | $\dfrac{\partial f}{\partial x}=\dfrac{\partial f}{\partial q}\dfrac{\partial q}{\partial x}=(-4)(1)$ | $-4$ |

**Regola della catena:**

$$
\boxed{\ \frac{\partial f}{\partial y}=\underbrace{\frac{\partial f}{\partial q}}_{\text{gradiente upstream}}\ \underbrace{\frac{\partial q}{\partial y}}_{\text{gradiente locale}}\ }
$$

### 9.2 Forma generale di un nodo

Un nodo $f$ riceve $x,y$ e produce $z=f(x,y)$. Dal lato di uscita arriva il **gradiente upstream** $\partial L/\partial z$. Il nodo calcola i **gradienti locali** $\partial z/\partial x$, $\partial z/\partial y$ e restituisce i **gradienti downstream**:

$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial x},\qquad
\frac{\partial L}{\partial y}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial y}.
$$

Ogni nodo è quindi un'unità *locale*: non serve conoscere il resto del grafo.

### 9.3 Esempio con sigmoide ("flat code")

Valori: $w_0=2,\ x_0=-1,\ w_1=-3,\ x_1=-2,\ w_2=-3$.

**Forward:**

$$
s_0=w_0x_0=-2,\quad s_1=w_1x_1=6,\quad s_2=s_0+s_1=4,\quad s_3=s_2+w_2=1,\quad L=\sigma(s_3)=\frac1{1+e^{-1}}\approx0.73 .
$$

**Derivata della sigmoide** **[Aggiunta]:**

$$
\sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}=\sigma(x)\big(1-\sigma(x)\big).
$$

**Backward** (corrisponde riga per riga al codice della slide):

| Codice | Formula | Valore |
|---|---|---|
| `grad_L = 1.0` | $\partial L/\partial L$ | 1 |
| `grad_s3 = grad_L*(1-L)*L` | $1\cdot 0.73\cdot0.27$ | $\approx 0.20$ |
| `grad_w2 = grad_s3` | add gate | 0.20 |
| `grad_s2 = grad_s3` | add gate | 0.20 |
| `grad_s0 = grad_s2` | add gate | 0.20 |
| `grad_s1 = grad_s2` | add gate | 0.20 |
| `grad_w1 = grad_s1*x1` | $0.20\cdot(-2)$ | $-0.40$ |
| `grad_x1 = grad_s1*w1` | $0.20\cdot(-3)$ | $-0.60$ |
| `grad_w0 = grad_s0*x0` | $0.20\cdot(-1)$ | $-0.20$ |
| `grad_x0 = grad_s0*w0` | $0.20\cdot 2$ | $0.40$ |

### 9.4 Pattern nel flusso del gradiente

| Gate | Forward | Backward | Esempio slide |
|---|---|---|---|
| **add** ($z=a+b$) | somma | **distributore**: $\partial z/\partial a=\partial z/\partial b=1$, ogni ingresso riceve l'upstream | inputs 3,4 → 7; upstream 2 → 2 e 2 |
| **mul** ($z=ab$) | prodotto | **"swap multiplier"**: $\partial L/\partial a=\frac{\partial L}{\partial z}\,b$ e viceversa | inputs 2,3 → 6; upstream 5 → $5\cdot3=15$ e $5\cdot2=10$ |
| **copy** ($a\to a,a$) | duplica | **sommatore**: i gradienti delle uscite si sommano | upstream 4 e 2 → $4+2=6$ |
| **max** ($z=\max(a,b)$) | massimo | **router**: tutto l'upstream va all'ingresso maggiore, 0 all'altro | inputs 4,5; upstream 9 → 0 e 9 |

**Perché il copy gate somma.** Se una variabile $a$ è usata in due rami, $L=L(g_1(a),g_2(a))$; per la regola della catena multivariata
$\frac{\partial L}{\partial a}=\frac{\partial L}{\partial g_1}\frac{\partial g_1}{\partial a}+\frac{\partial L}{\partial g_2}\frac{\partial g_2}{\partial a}$.

**Perché il max fa da router.** $\max(a,b)=a$ se $a>b$, quindi $\partial z/\partial a=\mathbb{1}[a>b]$, $\partial z/\partial b=\mathbb{1}[b>a]$.

### 9.5 Implementazione modulare

Un oggetto `ComputationalGraph`:

- `forward`: itera i nodi in **ordine topologico** e chiama `gate.forward()`; l'ultimo nodo produce la loss;
- `backward`: itera in **ordine topologico inverso** e chiama `gate.backward()` (un pezzetto di backprop = regola della catena applicata al nodo).

L'ordine topologico garantisce che quando si calcola un nodo, tutti i suoi ingressi (forward) o tutti i suoi gradienti upstream (backward) siano già disponibili.

### 9.6 Backprop con vettori

Sia $x\in\mathbb{R}^{D_x}$, $y\in\mathbb{R}^{D_y}$, $z=f(x,y)\in\mathbb{R}^{D_z}$; la loss $L$ resta **scalare**.

- Gradiente upstream $\dfrac{\partial L}{\partial z}\in\mathbb{R}^{D_z}$: per ogni componente di $z$, quanto influenza $L$.
- Gradienti locali = **matrici Jacobiane**.

**[Nota – convenzione]** La slide scrive le Jacobiane come $[D_x\times D_z]$ e $[D_y\times D_z]$, cioè nel *denominator layout*. Nella convenzione standard $J_x=\partial z/\partial x\in\mathbb{R}^{D_z\times D_x}$ e

$$
\boxed{\ \frac{\partial L}{\partial x}=J_x^\top\,\frac{\partial L}{\partial z}\ }\quad\Longleftrightarrow\quad(\text{slide})\ \ [D_x\times D_z]\cdot[D_z]=[D_x].
$$

Il prodotto è un **matrice-vettore** e il risultato ha la stessa forma di $x$ ($D_x$), come deve essere (il gradiente ha sempre la stessa forma della variabile).

**Esempio ReLU elementwise** $y=\max(0,x)$, $x\in\mathbb{R}^4$:

$$
x=(1,-2,3,-1)^\top\ \Rightarrow\ y=(1,0,3,0)^\top,\qquad
\frac{\partial L}{\partial y}=(4,-1,5,9)^\top .
$$

La Jacobiana è diagonale: $J=\operatorname{diag}\big(\mathbb{1}[x_j>0]\big)=\operatorname{diag}(1,0,1,0)$. Quindi

$$
\frac{\partial L}{\partial x}=J^\top\frac{\partial L}{\partial y}=(4,\,0,\,5,\,0)^\top .
$$

**Jacobiana sparsa → non formarla mai esplicitamente.** Per funzioni elementwise si usa la moltiplicazione *implicita*:

$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\odot\mathbb{1}[x>0].
$$

(Costo $O(D)$ invece di $O(D^2)$.)

### 9.7 Backprop con matrici

Prodotto di matrici $y=xw$ con $x\in\mathbb{R}^{N\times D}$, $w\in\mathbb{R}^{D\times M}$, $y\in\mathbb{R}^{N\times M}$:

$$
y_{n,m}=\sum_{d=1}^{D}x_{n,d}\,w_{d,m}.
$$

**Derivazione [Aggiunta].** Data l'upstream $\partial L/\partial y_{n,m}$, e poiché $x_{n,d}$ influenza solo la riga $n$ di $y$:

$$
\frac{\partial L}{\partial x_{n,d}}=\sum_{m}\frac{\partial L}{\partial y_{n,m}}\frac{\partial y_{n,m}}{\partial x_{n,d}}=\sum_m\frac{\partial L}{\partial y_{n,m}}\,w_{d,m}
\ \Rightarrow\ \boxed{\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\,w^\top}\quad[N\times M][M\times D]=[N\times D]
$$

$$
\frac{\partial L}{\partial w_{d,m}}=\sum_{n}\frac{\partial L}{\partial y_{n,m}}\,x_{n,d}
\ \Rightarrow\ \boxed{\frac{\partial L}{\partial w}=x^\top\frac{\partial L}{\partial y}}\quad[D\times N][N\times M]=[D\times M]
$$

**Trucco mnemonico (slide):** queste formule "sono l'unico modo di far combaciare le dimensioni".

**Esempio numerico.** Con gli $x,w$ della slide:

$$
x=\begin{pmatrix}2&1&-3\\-3&4&2\end{pmatrix},\quad
w=\begin{pmatrix}3&2&1&-1\\2&1&3&2\\3&2&1&-2\end{pmatrix},\quad
\frac{\partial L}{\partial y}=\begin{pmatrix}2&3&-3&9\\-8&1&4&6\end{pmatrix}.
$$

**[Nota]** I valori di $y$ mostrati nella slide ($13,9,-2,-6\,/\,5,2,17,1$) sono illustrativi e **non** coincidono con $xw$. Il prodotto corretto è

$$
y=xw=\begin{pmatrix}-1&-1&2&6\\5&2&11&7\end{pmatrix}
$$

(es. $y_{1,1}=2\cdot3+1\cdot2+(-3)\cdot3=-1$). Ciò che la slide vuole mostrare è la struttura delle dipendenze: $x_{1,2}$ influenza solo la riga 1 di $y$, mentre $w_{2,3}$ influenza la colonna 3.

Gradienti con l'upstream dato:

$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}w^\top=\begin{pmatrix}0&16&-9\\-24&9&-30\end{pmatrix},\qquad
\frac{\partial L}{\partial w}=x^\top\frac{\partial L}{\partial y}=\begin{pmatrix}28&3&-18&0\\-30&7&13&33\\-22&-7&17&-15\end{pmatrix}.
$$

*Esempio di un elemento:* $\partial L/\partial x_{1,2}=2\cdot2+3\cdot1+(-3)\cdot3+9\cdot2=16$.

## 10. Reti neurali profonde

| | Funzione |
|---|---|
| Prima (lineare) | $f=Wx$ |
| Rete a 2 strati | $f=W_2\max(0,W_1x)$ |
| Rete a 3 strati | $f=W_3\max(0,W_2\max(0,W_1x))$ |

Dimensioni: $x\in\mathbb{R}^D,\ W_1\in\mathbb{R}^{H_1\times D},\ W_2\in\mathbb{R}^{H_2\times H_1},\ W_3\in\mathbb{R}^{C\times H_2}$ (nella rete a 2 strati $W_2\in\mathbb{R}^{C\times H}$). In pratica si aggiunge un bias a ogni strato.

**[Aggiunta] Perché serve la non linearità.** Senza $\max(0,\cdot)$, $W_2(W_1x)=(W_2W_1)x=W'x$: la composizione di mappe lineari è lineare, quindi nessuna profondità aumenterebbe l'espressività. La non linearità rende la rete un approssimatore universale di funzioni.

**[Aggiunta] Parametri** (con bias, ad es. $D=3072$, $H=100$, $C=10$): $100\cdot3072+100+10\cdot100+10=308\,310$.

## 11. Funzioni di attivazione

| Nome | Formula | Derivata | Note |
|---|---|---|---|
| Sigmoid | $\sigma(x)=\dfrac1{1+e^{-x}}$ | $\sigma(1-\sigma)\le\tfrac14$ | range $(0,1)$; satura → *vanishing gradient*; non centrata sullo zero |
| tanh | $\tanh(x)=2\sigma(2x)-1$ | $1-\tanh^2(x)$ | range $(-1,1)$; centrata, ma satura |
| **ReLU** | $\max(0,x)$ | $\mathbb{1}[x>0]$ | **buon default per la maggior parte dei problemi**; non satura per $x>0$; economica; può "morire" (gradiente 0 per $x<0$) |
| Leaky ReLU | $\max(0.1x,\,x)$ | $1$ se $x>0$, $0.1$ altrimenti | evita neuroni morti |
| Maxout | $\max(w_1^\top x+b_1,\ w_2^\top x+b_2)$ | gradiente sul ramo massimo | generalizza ReLU e Leaky ReLU; raddoppia i parametri |
| ELU | $\begin{cases}x&x\ge0\\\alpha(e^x-1)&x<0\end{cases}$ | $1$ se $x\ge0$, $\alpha e^x$ altrimenti | continua in 0 (perché $\alpha(e^0-1)=0$); asintoto $-\alpha$ |

**Derivazione $\tanh$–sigmoide [Aggiunta]:**
$\tanh x=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=\frac{1-e^{-2x}}{1+e^{-2x}}=\frac{2}{1+e^{-2x}}-1=2\sigma(2x)-1.$

**Vanishing gradient con sigmoide [Aggiunta].** In una catena di $L$ strati sigmoidali il gradiente contiene un prodotto di $L$ fattori $\sigma'\le0.25$, quindi decade come $0.25^L$.

## 12. Rete a 2 strati in ~20 righe di NumPy

Codice della slide: $N=64$, $D_{in}=1000$, $H=100$, $D_{out}=10$, 2000 iterazioni, $\eta=10^{-4}$.

**Modello (forward), senza bias:**

$$
h=\sigma(xw_1),\qquad \hat y=h\,w_2,\qquad L=\sum_{n,m}(\hat y_{n,m}-y_{n,m})^2 .
$$

Dimensioni: $x:64\times1000$, $w_1:1000\times100$, $h:64\times100$, $w_2:100\times10$, $\hat y:64\times10$.

**Backward (derivazione riga per riga):**

1. `grad_y_pred = 2.0*(y_pred - y)` — da $\partial (\hat y-y)^2/\partial\hat y=2(\hat y-y)$.
2. `grad_w2 = h.T.dot(grad_y_pred)` — regola matrice: $\partial L/\partial w_2=h^\top\,\partial L/\partial\hat y$ $[100\times64][64\times10]$.
3. `grad_h = grad_y_pred.dot(w2.T)` — $\partial L/\partial h=\frac{\partial L}{\partial\hat y}w_2^\top$ $[64\times10][10\times100]$.
4. Attraverso la sigmoide (elementwise, $\sigma'=h(1-h)$): $\partial L/\partial a=\partial L/\partial h\odot h\odot(1-h)$, con $a=xw_1$.
5. `grad_w1 = x.T.dot(grad_h*h*(1-h))` — $\partial L/\partial w_1=x^\top\,\partial L/\partial a$ $[1000\times64][64\times100]$.
6. Aggiornamento: `w1 -= 1e-4*grad_w1`, `w2 -= 1e-4*grad_w2`.

Parametri: $1000\cdot100+100\cdot10=101\,000$.

---

# PARTE II — Reti convoluzionali (PDF 03)

## 13. Dal fully connected alla convoluzione

Esempio: immagine $1000\times1000$ (in scala di grigi), $10^6$ unità nascoste.

| Architettura | Connessioni per unità | Parametri totali |
|---|---|---|
| **Fully connected** | $10^6$ (tutti i pixel) | $10^6\cdot10^6=\mathbf{10^{12}}$ |
| **Localmente connessa** (filtro $10\times10$) | $10\cdot10=100$ | $10^6\cdot100=\mathbf{10^{8}}$ (100M) |
| **Convoluzionale** (100 filtri $10\times10$, pesi condivisi) | — | $100\cdot(10\cdot10)=\mathbf{10^4}$ (10K) |

**Due ipotesi sulle immagini:**

1. **Località:** la correlazione spaziale è locale → ogni unità guarda solo una piccola finestra (*better to put resources elsewhere*).
2. **Stazionarietà:** le statistiche sono simili in posizioni diverse → si **condividono gli stessi parametri** in tutte le posizioni. Il calcolo diventa una **convoluzione con kernel appreso**.

Per apprendere feature diverse si usano **più filtri**; ciascuno produce una mappa di attivazione. Con bias, i parametri sarebbero $100\cdot(100+1)=10\,100$.

**[Aggiunta] Lettura algebrica.** Un layer localmente connesso è $s=Wx$ con $W$ molto sparsa (una riga per unità, poche entrate non nulle). La convoluzione impone in più che le righe siano la stessa riga traslata (matrice di Toeplitz/Toeplitz a blocchi): $W$ ha pochi gradi di libertà.

## 14. Convoluzione 2D discreta

Immagine $I[m,n]$, kernel $g[k,l]$, immagine filtrata $f[m,n]$:

$$
\boxed{\,f[m,n]=(I\otimes g)[m,n]=\sum_{k,l}I[m-k,\,n-l]\;g[k,l]\,}
$$

Ogni pixel è sostituito da una **combinazione lineare dei suoi vicini**.

**Esempio della slide (pixel centrale).**
$I=\begin{pmatrix}8&5&2\\7&5&3\\9&4&1\end{pmatrix}$, $g=\begin{pmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{pmatrix}$.

Nella convoluzione vera l'indice $m-k$ ribalta il kernel (in entrambe le direzioni); l'effetto è che il kernel è applicato come $\begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}$:

$$
f[1,1]=(8-2)+(7-3)+(9-1)=6+4+8=18 .
$$

(Senza ribaltamento, cioè con la *cross-correlazione*, si otterrebbe $-18$.)

**[Aggiunta] Convoluzione vs cross-correlazione.** Le librerie di deep learning implementano in realtà la cross-correlazione $f[m,n]=\sum_{k,l}I[m+k,n+l]g[k,l]$. Poiché il kernel è appreso, la differenza è irrilevante: il kernel appreso sarà semplicemente il ribaltato.

### Esempio con stride 1 e padding 1 (zero-padding)

Con $g$ e $I$ come sopra, l'output è $3\times3$ (dimensione: $(3+2\cdot1-3)/1+1=3$). Per ogni finestra $3\times3$ del $I$ imbottito di zeri, il risultato è (somma della colonna sinistra) − (somma della colonna destra):

$$
f=\begin{pmatrix}-10&10&10\\-14&18&14\\-9&12&9\end{pmatrix}.
$$

*Verifica di due elementi:* $f[0,0]$: colonna sinistra $=0+0+0$, colonna destra $=0+5+5$ → $-10$. $f[1,2]$: sinistra $=5+5+4=14$, destra $=0$ → $14$.

Il kernel $(−1,0,1)$ per riga è un rilevatore di **bordi verticali** (derivata orizzontale): valori grandi in modulo dove l'intensità cambia da sinistra a destra.

## 15. Sistemi lineari

Un sistema $T$ è **lineare** se vale la **sovrapposizione**:

| Proprietà | Definizione |
|---|---|
| Omogeneità | $T[aX]=aT[X]$ |
| Additività | $T[X_1+X_2]=T[X_1]+T[X_2]$ |
| Sovrapposizione | $T[aX_1+bX_2]=aT[X_1]+bT[X_2]$ |

Lineare $\iff$ sovrapposizione (omogeneità + additività ⇒ sovrapposizione, e viceversa ponendo $b=0$ oppure $a=b=1$).

Esempi: operazioni matriciali e **convoluzioni**.

**Dimostrazione che la convoluzione è lineare [Aggiunta].**

$$
\big((aI_1+bI_2)\otimes g\big)[m,n]=\sum_{k,l}\big(aI_1+bI_2\big)[m-k,n-l]\,g[k,l]
=a\sum_{k,l}I_1[m-k,n-l]g[k,l]+b\sum_{k,l}I_2[m-k,n-l]g[k,l].
$$

Quindi $=a(I_1\otimes g)+b(I_2\otimes g)$. ∎

**[Aggiunta] Equivarianza alla traslazione.** Se $I'[m,n]=I[m-a,n-b]$, allora $(I'\otimes g)[m,n]=(I\otimes g)[m-a,n-b]$: traslare l'input equivale a traslare l'output. È la conseguenza matematica della condivisione dei pesi.

**[Aggiunta] Backprop attraverso una convoluzione.** Da $f[m,n]=\sum_{k,l}I[m-k,n-l]g[k,l]$:

$$
\frac{\partial f[m,n]}{\partial g[k,l]}=I[m-k,n-l]\ \Rightarrow\ \frac{\partial L}{\partial g[k,l]}=\sum_{m,n}\frac{\partial L}{\partial f[m,n]}\,I[m-k,n-l],
$$

$$
\frac{\partial L}{\partial I[p,q]}=\sum_{m,n}\frac{\partial L}{\partial f[m,n]}\,g[m-p,\,n-q].
$$

Il gradiente rispetto al kernel è una **somma su tutte le posizioni** in cui il kernel è stato usato (è il *copy gate* della sez. 9.4: lo stesso peso è copiato in ogni posizione).

## 16. Layer convoluzionale su volumi

### 16.1 Un filtro

Input $32\times32\times3$, filtro $5\times5\times3$. Il filtro attraversa **tutta la profondità** dell'input. Ogni valore di output è un prodotto scalare di $5\cdot5\cdot3=75$ numeri più un bias:

$$
a[m,n]=b+\sum_{c=1}^{3}\sum_{k=0}^{4}\sum_{l=0}^{4}W[c,k,l]\;x[c,\,m+k,\,n+l].
$$

Facendo scorrere il filtro su tutte le posizioni spaziali si ottiene **una mappa di attivazione** $28\times28\times1$.

### 16.2 Più filtri

Con $K$ filtri si ottengono $K$ mappe, impilate in un "nuovo volume". Con 6 filtri $5\times5$: $32\times32\times3\to28\times28\times6$.

In generale, con $C_{in}$ canali in ingresso e $C_{out}$ filtri di lato $F$:

$$
y[c',m,n]=b_{c'}+\sum_{c=1}^{C_{in}}\sum_{k,l=0}^{F-1}W[c',c,k,l]\,x[c,m+k,n+l],\qquad
\#\text{parametri}=C_{out}\,(C_{in}F^2+1).
$$

### 16.3 Dimensione dell'output

Per input di lato $W_{in}$, filtro $F$, padding $P$, stride $S$:

$$
\boxed{\,W_{out}=\frac{W_{in}+2P-F}{S}+1\,}
$$

*Derivazione [Aggiunta]:* dopo il padding il lato è $W_{in}+2P$; il primo filtro occupa le posizioni $0..F-1$; ogni salto avanza di $S$; il filtro sta nella posizione $j$ se $jS+F\le W_{in}+2P$, cioè $j\le (W_{in}+2P-F)/S$; con $j$ da 0 il numero di posizioni è $\lfloor(W_{in}+2P-F)/S\rfloor+1$.

Casi:

- senza padding, $S=1$: $32-5+1=28$;
- "same": con $S=1$ serve $P=(F-1)/2$ (per $F=5$, $P=2$).

### 16.4 ConvNet = sequenza di CONV + attivazioni

$$
32\times32\times3\xrightarrow[\text{6 filtri }5\times5\times3]{\text{CONV, ReLU}}28\times28\times6\xrightarrow[\text{10 filtri }5\times5\times\mathbf{6}]{\text{CONV, ReLU}}24\times24\times10\to\cdots
$$

Il numero di canali dei filtri deve coincidere con la profondità dell'input (3, poi 6).

| Strato | Parametri |
|---|---|
| CONV1 (6 filtri $5\times5\times3$) | $6\,(75+1)=456$ |
| CONV2 (10 filtri $5\times5\times6$) | $10\,(150+1)=1\,510$ |

### 16.5 Esercizi delle slide

**Dimensione dell'output.** Input $32\times32\times3$, 10 filtri $5\times5$, stride 1, pad 2:

$$
\frac{32+2\cdot2-5}{1}+1=32\ \Rightarrow\ \text{output }32\times32\times10 .
$$

**Numero di parametri.** Ogni filtro: $5\cdot5\cdot3+1=76$ (il $+1$ è il bias); totale $76\cdot10=\mathbf{760}$.

**[Aggiunta] Costo computazionale** (moltiplicazioni-somme): $32\cdot32\cdot10\cdot75=768\,000$. Notare che il costo cresce con la risoluzione spaziale, i parametri no.

## 17. Cenni storici

**Mappatura topografica nella corteccia.** Cellule vicine della corteccia visiva rappresentano regioni vicine del campo visivo (retinotopia; aree V1, V2, V3, hV4, VO1). Questo motiva connessioni **locali**.

**LeNet-5** (LeCun, Bottou, Bengio, Haffner, 1998): riconoscimento di documenti (cifre) con apprendimento basato sul gradiente.

Flusso: INPUT $32\times32$ → C1: $6@28\times28$ → S2: $6@14\times14$ → C3: $16@10\times10$ → S4: $16@5\times5$ → C5: 120 → F6: 84 → OUTPUT 10.

*Verifica delle dimensioni:* $32-5+1=28$; pooling $2\times2$ → 14; $14-5+1=10$; pooling → 5; $5-5+1=1$ (quindi C5 è di fatto un layer fully connected da $16\cdot25=400$ ingressi a 120).

*Parametri:* C1 $=6(25+1)=156$; C5 $=120(400+1)=48\,120$; F6 $=84(120+1)=10\,164$.

**Caratteristiche di LeNet** (slide): feed-forward con la sequenza *convoluzione (appresa) → non linearità (ReLU) → pooling (massimo locale = sotto-campionamento) → feature map*; **supervisionata**; i filtri sono addestrati **retropropagando l'errore di classificazione**.

## 18. Pooling

- Rende le rappresentazioni più piccole e gestibili;
- opera su ogni mappa di attivazione **indipendentemente** (non mescola i canali).

Esempio: $224\times224\times64\to112\times112\times64$. Con finestra $F=2$ e stride $S=2$: $(224-2)/2+1=112$.

### Max pooling

Su ogni finestra si prende il massimo. Esempio della slide (finestra $2\times2$, stride 2):

$$
\begin{pmatrix}1&1&2&4\\5&6&7&8\\3&2&1&0\\1&2&3&4\end{pmatrix}\ \longrightarrow\ \begin{pmatrix}6&8\\3&4\end{pmatrix}.
$$

*Calcolo:* $\max(1,1,5,6)=6$; $\max(2,4,7,8)=8$; $\max(3,2,1,2)=3$; $\max(1,0,3,4)=4$.

**Proprietà [Aggiunta].**

- **Nessun parametro** da apprendere.
- **Backprop:** è un *max gate* (sez. 9.4): il gradiente upstream va solo alla posizione del massimo, $\partial y/\partial x_{ij}=\mathbb{1}[(i,j)=\arg\max]$; le altre ricevono 0.
- **Invarianza locale a piccole traslazioni:** spostando l'input di un pixel dentro la finestra, il massimo spesso non cambia.
- Riduce il costo dei layer successivi di un fattore $S^2$ in area.

## 19. AlexNet (Krizhevsky, Sutskever, Hinton, NIPS 2012)

Rispetto a LeNet-98:

| | LeNet-98 | AlexNet |
|---|---|---|
| Modello | piccolo | più grande (8 strati) |
| Dati | $10^3$ immagini | $10^6$ immagini |
| Calcolo | CPU | GPU (speedup 50×) |
| Regolarizzazione | — | **Dropout** |

Numeri: 7 strati nascosti, 650 000 neuroni, **60 000 000 parametri**, addestrata su 2 GPU per una settimana.

Struttura (dalla figura): 5 strati convoluzionali (con max-pooling dopo alcuni) + 3 fully connected (2048+2048 per GPU → 1000 classi). Primo strato: filtri $11\times11$, stride 4 → mappe $55\times55$.

**[Nota]** Con $W_{in}=224$ si avrebbe $(224-11)/4+1=54.25$: non intero. Nell'implementazione reale l'input è $227\times227$, che dà $(227-11)/4+1=55$.

**Dropout [Aggiunta].** In training, per ogni unità $h_j$ si campiona $m_j\sim\text{Bernoulli}(p)$ e si usa $\tilde h_j=m_jh_j$. In test si usano tutte le unità scalando le attivazioni di $p$ (oppure, nella variante *inverted dropout*, si divide per $p$ in training). Effetto: evita la co-adattazione dei neuroni e approssima un ensemble di $2^n$ sotto-reti che condividono i pesi.

## 20. Progressi che hanno reso possibili le DNN

| Contributo | Riferimento | Idea (formalizzazione) |
|---|---|---|
| Cognitron/Neocognitron | Fukushima 1971–1982 | gerarchia di celle semplici (feature) e complesse (invarianza) |
| Pooling | Riesenhuber & Poggio 1999 | massimo su regioni locali |
| Convnet | LeCun et al. 1989 | convoluzione + backprop |
| Non linearità (ReLU) | Nair & Hinton 2010 | $\max(0,x)$ |
| DropOut | Krizhevsky et al. 2012 | maschere casuali (sez. 19) |
| Batch Normalization | Ioffe & Szegedy 2015 | vedi sotto |
| Identity mapping | He et al. 2015 | connessioni residue |
| Attention | Bengio et al. 2015 | pesatura softmax degli ingressi |

**Batch Normalization [Aggiunta].** Per un minibatch $B$ e ogni feature:

$$
\mu_B=\frac1m\sum_{i\in B}x_i,\quad \sigma_B^2=\frac1m\sum_{i\in B}(x_i-\mu_B)^2,\quad \hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},\quad y_i=\gamma\hat x_i+\beta,
$$

con $\gamma,\beta$ appresi.

**Connessione residua [Aggiunta].** $y=F(x)+x$. Il gradiente è $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial y}\Big(\dfrac{\partial F}{\partial x}+I\Big)$: il termine $I$ garantisce un cammino diretto per il gradiente attraverso decine o centinaia di strati.

**Attention [Aggiunta].** Dati punteggi $e_j$ per gli ingressi $v_j$: $\alpha_j=\dfrac{e^{e_j}}{\sum_k e^{e_k}}$, uscita $c=\sum_j\alpha_jv_j$ (media pesata con pesi softmax).

**Figura della slide (rete di LeCun 1989):** 256 ingressi ($16\times16$); H1 $=12\times64=768$ unità (12 mappe $8\times8$, kernel $5\times5$); H2 $=12\times16=192$ unità (12 mappe $4\times4$, kernel $5\times5\times8$); H3 30 unità; 10 uscite.
*Verifica dei collegamenti:* H1: $768\cdot25=19\,200\approx20\,000$; H2: $192\cdot(5\cdot5\cdot8)=38\,400\approx40\,000$; H3: $192\cdot30=5\,760\approx6\,000$; uscita: $30\cdot10=300$. ✓.

## 21. Receptive field

Con un kernel di lato $K$, ogni elemento dell'output dipende da una regione $K\times K$ dell'input (*receptive field*).

**Convoluzioni successive (stride 1).** Ogni convoluzione aggiunge $K-1$ al receptive field:

$$
\boxed{\,r_L=1+L\,(K-1)\,}
$$

**Dimostrazione per induzione [Aggiunta].**
- $L=0$: $r_0=1$ (un pixel).
- Se un elemento dello strato $L-1$ vede $r_{L-1}$ pixel per lato, l'elemento dello strato $L$ combina $K$ elementi adiacenti del livello $L-1$: il loro insieme copre $r_{L-1}+(K-1)$ pixel (i $K$ campi si sovrappongono e si spostano di 1 ciascuno). Quindi $r_L=r_{L-1}+K-1$, da cui la formula.

Esempi con $K=3$: $L=1\to3$, $L=2\to5$, $L=3\to7$.

**[Aggiunta] Formula generale con stride.** Con $j_0=1$ (salto fra elementi adiacenti in unità di pixel di input) e $r_0=1$:

$$
r_l=r_{l-1}+(K_l-1)\,j_{l-1},\qquad j_l=j_{l-1}\,S_l .
$$

Lo stride e il pooling fanno crescere $j$ e quindi il receptive field cresce più che linearmente.

**Attenzione (slide):** distinguere "receptive field nell'**input**" da "receptive field nello **strato precedente**" (nel secondo caso è sempre $K\times K$).

## 22. Tendenza: filtri più piccoli e reti più profonde

Le ConvNet impilano strati di convoluzione, non linearità e pooling. La tendenza è verso **filtri più piccoli e architetture più profonde**; i filtri piccoli sono più facili da apprendere di quelli grandi.

**[Aggiunta] Perché.** $L$ strati $3\times3$ hanno receptive field $2L+1$. Due strati $3\times3$ vedono $5\times5$ come un filtro $5\times5$, ma con $C$ canali costano $2\cdot9C^2=18C^2$ parametri contro $25C^2$, e includono una non linearità in più.

**Errore top-5 su ImageNet (figura della slide):**

| Anno | Modello | Strati | Errore (%) |
|---|---|---|---|
| 2010 | shallow | — | 28.2 |
| 2011 | shallow | — | 25.8 |
| 2012 | AlexNet | 8 | 16.4 |
| 2013 | — | — | 11.7 |
| 2014 | VGG | 19 | 7.3 |
| 2014 | GoogLeNet | 22 | 6.7 |
| 2015 | ResNet | 152 | **3.57** |

L'errore cala mentre la profondità cresce da 8 a 152 strati.

## 23. Letture consigliate (metodi di spiegabilità)

Le letture riguardano le spiegazioni visive delle predizioni di una CNN. Riassunto formale di alcune:

- **Grad-CAM (Selvaraju et al.).** Per la classe $c$ con score $y^c$ e mappe $A^k$ dell'ultimo strato convoluzionale:
  $$
  \alpha_k^c=\frac1Z\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}},\qquad
  L^c_{\text{Grad-CAM}}=\operatorname{ReLU}\Big(\sum_k\alpha_k^cA^k\Big).
  $$
  I pesi $\alpha_k^c$ sono il gradiente mediato spazialmente (importanza del canale $k$); la ReLU tiene solo le regioni che aumentano lo score.
- **Grad-CAM++** (Chattopadhyay et al.): versione con pesi basati su derivate di ordine superiore, per localizzare meglio più istanze dello stesso oggetto.
- **RISE** (Petsiuk et al.): metodo *black-box*; si applicano maschere casuali all'input e si pesano le maschere con lo score ottenuto.
- **LRP** (Bach et al.): propagazione a ritroso della *rilevanza* con conservazione, $\sum_i R_i^{(l)}=\sum_j R_j^{(l+1)}$.
- **Pixel-level Certified Explanations via Randomized Smoothing** (Anani et al., 2025): fornisce garanzie certificate sulla stabilità delle spiegazioni a livello di pixel.

---

## Riepilogo delle formule fondamentali

| Concetto | Formula |
|---|---|
| Classificatore lineare | $f=Wx+b$ |
| Softmax | $p_k=e^{s_k}/\sum_je^{s_j}$ |
| Cross-entropy | $L_i=-\log p_{y_i}$, $\ \nabla_sL_i=p-e_{y_i}$ |
| Loss regolarizzata | $L=\frac1N\sum_iL_i+\lambda R(W)$ |
| SGD | $W\leftarrow W-\eta\,\widehat\nabla_WL$ |
| Catena | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial z}\dfrac{\partial z}{\partial x}$ |
| Backprop matriciale | $\partial_xL=(\partial_yL)w^\top$, $\ \partial_wL=x^\top(\partial_yL)$ |
| Convoluzione | $f[m,n]=\sum_{k,l}I[m-k,n-l]g[k,l]$ |
| Dimensione output | $(W+2P-F)/S+1$ |
| Parametri di CONV | $C_{out}(C_{in}F^2+1)$ |
| Receptive field | $1+L(K-1)$ |
