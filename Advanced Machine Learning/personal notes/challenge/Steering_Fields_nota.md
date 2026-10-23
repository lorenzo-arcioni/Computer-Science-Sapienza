# Steering Fields: Adaptive Vector Fields for Safe Image Generation and Beyond

**Nota di studio dettagliata** sul paper di S. Facchiano, J. E. Lenssen, B. Schiele, W. Stammer, F. Galasso, J. Fischer (Max Planck Institute for Informatics / Sapienza Università di Roma), *arXiv:2609.39573v1 [cs.CV]*, 30 settembre 2026, 22 pagine.

> **Come leggere questa nota.** Segue l'ordine del paper (Introduzione → Background → Metodo → Esperimenti → Discussione → Appendici A–F). Ogni passaggio matematico è scritto passo passo. Le figure `figs/*.svg` sono generate da `steering_fields_figures.py` (file a parte). **Attenzione:** le figure 1–9 e 14 sono su un *modello giocattolo 2D analitico*, costruito per mostrare la geometria dell'idea; **non** sono risultati di FLUX o SD3.5. Le figure 10–13 usano invece i numeri esatti delle tabelle del paper. Dove il paper è incoerente o impreciso lo segnalo nel §12.

---

## Indice

0. [Riassunto in 60 secondi](#0-riassunto-in-60-secondi)
1. [Prerequisiti: flow matching e steering](#1-prerequisiti)
2. [Il problema: perché un vettore fisso non basta](#2-il-problema)
3. [L'idea: Steering Fields](#3-lidea-steering-fields)
4. [Derivazione matematica passo passo](#4-derivazione-matematica-passo-passo)
5. [Relazione con lo steering classico](#5-relazione-con-lo-steering-classico)
6. [L'algoritmo](#6-lalgoritmo)
7. [Esempio numerico a mano](#7-esempio-numerico-a-mano)
8. [Setup sperimentale](#8-setup-sperimentale)
9. [Risultati: safety, editing, blending](#9-risultati)
10. [Discussione e limiti](#10-discussione-e-limiti)
11. [Conclusione](#11-conclusione)
12. [Osservazioni critiche e incoerenze nel paper](#12-osservazioni-critiche)
13. [Glossario dei simboli](#13-glossario)
- [**Appendice** (riproduzione di tutto il materiale supplementare A–F)](#appendice)
- [File di codice](#file-di-codice)

---

## 0. Riassunto in 60 secondi

- I moderni modelli text-to-image (FLUX1, Stable Diffusion 3.5) sono **flow-matching models**: partono da rumore e integrano un **campo di velocità** $V(z,t\mid c)$ fino all'immagine.
- Il **controllo** (es. sopprimere contenuti nudi/violenti) si fa di solito con l'**activation steering**: si somma un **vettore fisso** $r^\Delta$ alle attivazioni di un layer. Ma il vettore è lo stesso per ogni esempio e per ogni istante, mentre la traiettoria è curva e cambia continuamente → effetti collaterali globali (cambia composizione, struttura).
- **Idea:** passare da un vettore fisso a un **campo**. A ogni passo di integrazione si calcolano **tre** velocità dello stesso modello — con prompt *source*, *target*, *away* — e le si combinano con la soluzione chiusa di un problema quadratico:

$$
v^*=\frac{v_{src}+\mu\,v_{tar}-\lambda\,v_{away}}{1+\mu-\lambda}
= v_{src}+\alpha(v_{tar}-v_{src})-\beta(v_{away}-v_{src}).
$$

- Niente training, niente maschere, niente accesso agli strati interni: opera direttamente sul **latente**, quindi è **model-agnostic**.
- Con $\lambda=0$ diventa **blending** di concetti; applicato a un'immagine reale rumorizzata diventa un **editor di immagini senza inversione**.
- Risultati: stato dell'arte su Ring-a-Bell e P4D (NudeNet FLUX: 70.67 → 37.32 su Ring-a-Bell, 55.67 → 24.43 su P4D) senza degradare CLIP/VQAScore su COCO; e il miglior CLIP-txt / CLIP-dir / VQAScore su PieBench++ per l'editing.
- Costo: 2 (blend) o 3 (replace) forward pass per passo invece di 1. Limiti: serve tarare $\mu,\lambda$; la soppressione non è perfetta (es. intimità senza nudità esplicita).

![Panoramica](figs/fig14_relazioni_formalismi.svg)

*Fig. 14 — Come i formalismi si derivano l'uno dall'altro: REPLACE è il caso generale; BLEND si ottiene con $\lambda=0$; lo steering classico si ottiene con $\mu=\lambda$ e direzione $v^\Delta$ fissata; l'editing è lo stesso operatore con punto di partenza diverso.*

---

## 1. Prerequisiti

### 1.1 Flow matching / rectified flow (App. B del paper)

Un modello generativo **flow matching** impara un campo vettoriale dipendente dal tempo che trasporta rumore in dati.

**Percorso lineare (rectified flow).** Si prende una coppia (rumore $\epsilon$, dato $x$) e si interpola in linea retta:

$$z_t=(1-t)\,\epsilon+t\,x .$$

La velocità lungo questo segmento è **costante**: $\frac{dz_t}{dt}=x-\epsilon$.

**Training.** Una rete $V(z,t\mid c)\equiv v_\theta(z,t\mid c)$, condizionata a un prompt $c$, impara a predire quella velocità:

$$\mathcal L_{FM}(\theta)=\mathbb E\big[\,\|V(z_t,t\mid c)-(x-\epsilon)\|_2^2\,\big],\qquad t\sim\mathcal U[0,1].$$

(La rete non vede la coppia $(\epsilon,x)$ ma solo $z_t$: impara quindi la **media condizionata** $\mathbb E[x-\epsilon\mid z_t=z]$. Per questo le traiettorie *generate* sono curve anche se ogni singolo percorso di training è una retta.)

**Inferenza.** Si parte da rumore e si risolve l'ODE $\dfrac{dz_t}{dt}=V(z_t,t\mid c)$ con il **metodo di Eulero**:

$$z_{i+1}=z_i+\Delta t_i\,V(z_i,t_i\mid c),\qquad \Delta t_i=t_{i+1}-t_i .$$

Alla fine il latente pulito viene **decodificato dal VAE** in un'immagine.

![Flow matching](figs/fig02_flow_matching.svg)

*Fig. 2 — Sinistra: il percorso lineare tra rumore ed immagine. Destra: integrazione di Eulero con 4, 10, 40 passi nel campo del toy model; con pochi passi l'approssimazione è più grossolana.*

> **Convenzione temporale.** In questa nota (e nell'Algoritmo 1 del paper) $t=0$ è rumore e $t=1$ è immagine, con $z_{i+1}=z_i+\Delta t\,v$. L'Appendice B del paper usa la convenzione *opposta* ($z_0$ pulito, $z_1$ rumore, integrazione da $t=1$ a $t=0$). Sono equivalenti a meno di un cambio $t\to1-t$ e di segno della velocità; vedi §12.

### 1.2 Activation steering (Sez. 2 del paper)

Si assume la **Linear Representation Hypothesis**: i concetti (nudità, violenza, stile…) corrispondono a **direzioni lineari** nello spazio delle attivazioni. Per *spingere* il modello verso o via da un concetto si somma una direzione al residual stream $h_l$ di un layer $l$:

$$\tilde h_l=h_l^{src}+\alpha\,r^\Delta \tag{1}$$

La direzione si pre-calcola con la **differenza delle medie** tra campioni che contengono il concetto (*tar*) e campioni che non lo contengono (*away*):

$$r^\Delta=\frac1N\sum_{i=1}^N h_l^{tar}-\frac1N\sum_{i=1}^N h_l^{away}=d_l^{tar}-d_l^{away}. \tag{2}$$

È leggero, non richiede riaddestramento e funziona su LLM e modelli di diffusione/flow. **Ma** $r^\Delta$ è calcolato una volta sola e applicato in modo uniforme.

### 1.3 Editing di immagini (Sez. 2)

Famiglie citate: perturbazione (SDEdit), manipolazione dell'attenzione (Prompt-to-Prompt, Plug-and-Play), editing da istruzioni (InstructPix2Pix, In-Context Edit), modelli di fondazione (FLUX Kontext). Molti usano l'**inversione** (Null-text inversion, Direct Inversion, StableFlow, RF-Inversion) per ancorare il risultato alla struttura sorgente; i più recenti la evitano (**FlowEdit**, InfEdit). Per l'editing multi-aspetto: ParallelEdits. Controllo fine dell'intensità: slider (Concept Sliders, SliderEdit). Steering Fields si distingue perché **unifica induzione e inibizione** in un solo framework e **estende naturalmente** all'editing condizionando su un latente di immagine invece che su rumore.

---

## 2. Il problema

Il paper (Sez. 1–2) argomenta tre difetti dello steering a vettore fisso:

1. **Agnostico all'esempio**: $r^\Delta$ è lo stesso per ogni prompt/immagine.
2. **Agnostico al tempo**: la traiettoria rumore→immagine è *curva* e la velocità condizionata dipende da $(z,t)$. All'inizio, quando il latente è quasi solo rumore, somiglia pochissimo alle attivazioni "mature" da cui $r^\Delta$ è stato stimato.
3. **Effetti collaterali globali**: una spinta costante non può combaciare con la geometria locale della varietà → cambia anche ciò che non si voleva cambiare (composizione, struttura locale).

![Costante vs campo](figs/fig06_costante_vs_campo.svg)

*Fig. 6 (toy 2D) — Riga alta: errore $\|v^\Delta(z,t)-r\|$ tra la direzione corretta nel punto $(z,t)$ e un vettore fisso $r$ ottenuto "alla difference-of-means" (media su 200 traiettorie). Il vettore costante è sbagliato quasi ovunque tranne in zone ristrette, e le zone sbagliate cambiano con $t$. Riga bassa: lungo **una** traiettoria le componenti del campo variano molto, e l'errore del vettore fisso arriva a ≈4 (media ≈1.1). Esito generato dallo script: $r=(1.87,-4.25)$.*

> **Perché nel toy il campo varia?** Per gaussiane isotrope $N(m,s^2I)$ il flow è noto in forma chiusa: $v(z,t)=m+k(t)\,(z-tm)$ con $k(t)=\frac{ts^2-(1-t)}{(1-t)^2+t^2s^2}$. La differenza $v_{tar}-v_{away}$ dipende da $z$ e $t$ perché le due condizioni hanno varianze diverse. Nei modelli reali la dipendenza è molto più forte.

---

## 3. L'idea: Steering Fields

**Da vettore a campo.** Invece di un $r^\Delta$ fisso nello spazio delle attivazioni, si definisce un campo $r^\Delta(z,t)$ sul **latente** $z$ in cui il modello è già definito, ri-valutato a ogni passo.

Tre ingredienti (Sez. 3.1):

| Simbolo | Significato | Esempio (safety, Fig. 1) |
|---|---|---|
| $c_{src}$ | prompt dell'utente | "... the subject posed nude..." |
| $c_{tar}$ | concetto da **indurre** (opzionale) | "clothed" (media di 50 prompt) |
| $c_{away}$ | concetto da **sopprimere** (opzionale) | "nude" (media di 50 prompt) |

Notazione: $v_\bullet\equiv V(z,t\mid c_\bullet)$ per $\bullet\in\{src,tar,away\}$ (la dipendenza da $(z,t)$ è sottintesa).

**Safety steering = sostituire "away" con "tar"**: il modello genera seguendo $c_{src}$ ma viene attratto verso il vestito e respinto dalla nudità.

![Panoramica con traiettorie](figs/fig01_overview_traiettorie.svg)

*Fig. 1 (toy 2D, analogo di Fig. 1-sinistra del paper) — Cerchi = distribuzioni delle tre condizioni. Linea blu tratteggiata: la traiettoria non sterzata finisce vicino alla distribuzione src (che contiene il concetto indesiderato). Linea rosa: la traiettoria sterzata devia verso tar e lontano da away. Le frecce colorate sono i vettori $v_{src}$ (blu), $v_{tar}$ (verde), $v_{away}$ (arancio) e il risultante $v^*$ (rosa) in quattro punti: **ogni volta** cambiano.*

**Tre proprietà dichiarate:**
1. **Model-agnostic**: serve solo una velocità condizionata (UNet, DiT, MM-DiT…).
2. **Compositional**: induzione + inibizione nella stessa formula, manopole $\alpha,\beta$ continue.
3. **Editing come sottoprodotto**: niente inversione, niente maschere.

---

## 4. Derivazione matematica passo passo

### 4.1 L'obiettivo quadratico (Eq. 3)

A ogni passo si cerca una velocità $v^*$ che sia **vicina** alla sorgente, **attratta** dal target e **respinta** dall'away:

$$\boxed{\mathcal L(v)=\|v-v_{src}\|^2+\mu\,\|v-v_{tar}\|^2-\lambda\,\|v-v_{away}\|^2}\tag{3}$$

con $\mu,\lambda\ge0$ e $1+\mu-\lambda>0$. Il termine col **segno meno** premia l'*allontanarsi* da $v_{away}$.

**Perché serve $1+\mu-\lambda>0$?** Il coefficiente del termine quadratico in $v$ è $(1+\mu-\lambda)$; la matrice Hessiana è $2(1+\mu-\lambda)I$. Se è positiva, $\mathcal L$ è un paraboloide con **un solo minimo** (strettamente convessa). Se è $\le0$ la loss non ha minimo (cresce senza limite verso $-\infty$, o è piatta): il termine repulsivo "vince" e il problema è mal posto.

![Geometria della loss](figs/fig03_geometria_loss.svg)

*Fig. 3 — Curve di livello di $\mathcal L(v)$ nello spazio delle velocità (2D). Il minimo (stella rosa) è: sopra, $v_{src}$ (nessuno steering); al centro, un punto **sul segmento** $v_{src}$–$v_{tar}$ (blending); a destra, un punto **fuori dal segmento**, spostato in direzione opposta a $v_{away}$ (replacing).*

![Convessità e mappe α, β](figs/fig04_convessita_alpha_beta.svg)

*Fig. 4 — Sinistra: sezione 1D di $\mathcal L$ per tre coppie $(\mu,\lambda)$; con $1+\mu-\lambda=0$ la curva è una retta (nessun minimo), con valori positivi è una parabola. Centro/destra: mappe di $\alpha=\mu/(1+\mu-\lambda)$ e $\beta=\lambda/(1+\mu-\lambda)$; la zona grigia è la regione non ammessa. I tre punti bianchi sono le impostazioni del paper (FLUX 0.3/0.3, SD3.5 0.4/0.4, editing 0.8/0.8).*

### 4.2 Il minimizzatore chiuso (Appendice A.2 → Eq. 4)

**Passo 1 – sviluppare i quadrati** (Eq. 12–14):

$$\|v-v_{src}\|^2=v^\top v-2v^\top v_{src}+v_{src}^\top v_{src},$$
$$\|v-v_{tar}\|^2=v^\top v-2v^\top v_{tar}+v_{tar}^\top v_{tar},$$
$$\|v-v_{away}\|^2=v^\top v-2v^\top v_{away}+v_{away}^\top v_{away}.$$

**Passo 2 – sostituire in $\mathcal L$** (Eq. 15):

$$\mathcal L(v)=(v^\top v-2v^\top v_{src}+v_{src}^\top v_{src})+\mu(v^\top v-2v^\top v_{tar}+v_{tar}^\top v_{tar})-\lambda(v^\top v-2v^\top v_{away}+v_{away}^\top v_{away}).$$

**Passo 3 – raggruppare i termini in $v$:**

$$\mathcal L(v)=(1+\mu-\lambda)\,v^\top v-2v^\top v_{src}-2\mu\,v^\top v_{tar}+2\lambda\,v^\top v_{away}+C,$$

dove $C=v_{src}^\top v_{src}+\mu\,v_{tar}^\top v_{tar}-\lambda\,v_{away}^\top v_{away}$ non dipende da $v$.

(Il segno meno davanti a $\lambda$ trasforma $-2v^\top v_{away}$ in $+2\lambda\,v^\top v_{away}$.)

**Passo 4 – gradiente:**

$$\nabla_v\mathcal L=2(1+\mu-\lambda)\,v-2v_{src}-2\mu\,v_{tar}+2\lambda\,v_{away}.$$

**Passo 5 – imporre $\nabla_v\mathcal L=0$:**

$$(1+\mu-\lambda)\,v^*=v_{src}+\mu\,v_{tar}-\lambda\,v_{away}.$$

**Passo 6 – dividere** (lecito perché $1+\mu-\lambda>0$):

$$\boxed{v^*=\frac{v_{src}+\mu\,v_{tar}-\lambda\,v_{away}}{1+\mu-\lambda}}\tag{4}$$

> **Lettura geometrica.** È una **media pesata "affine"** dei tre vettori con pesi $\frac{1}{1+\mu-\lambda},\ \frac{\mu}{1+\mu-\lambda},\ \frac{-\lambda}{1+\mu-\lambda}$. La loro somma è $\frac{1+\mu-\lambda}{1+\mu-\lambda}=1$. Un peso negativo su $v_{away}$ significa "estrapolare" via da esso. (Verificato numericamente, vedi «File di codice» e `run_output.txt`.)

### 4.3 Decomposizione in "sorgente + correzioni" (A.3 → Eq. 5)

Si vuole riscrivere $v^*$ come *perturbazione* di $v_{src}$.

**Passo 1 – sommare e sottrarre $(\mu-\lambda)v_{src}$ al numeratore:**

$$v^*=\frac{v_{src}+\mu v_{tar}-\lambda v_{away}+(\mu-\lambda)v_{src}-(\mu-\lambda)v_{src}}{1+\mu-\lambda}.$$

**Passo 2 – raggruppare i $v_{src}$:** $v_{src}+(\mu-\lambda)v_{src}=(1+\mu-\lambda)v_{src}$:

$$v^*=\frac{(1+\mu-\lambda)v_{src}+\mu v_{tar}-\lambda v_{away}-(\mu-\lambda)v_{src}}{1+\mu-\lambda}.$$

**Passo 3 – riscrivere i resti come differenze.** Poiché $-(\mu-\lambda)v_{src}=-\mu v_{src}+\lambda v_{src}$:

$$\mu v_{tar}-\mu v_{src}=\mu(v_{tar}-v_{src}),\qquad -\lambda v_{away}+\lambda v_{src}=-\lambda(v_{away}-v_{src}),$$

quindi

$$v^*=\frac{(1+\mu-\lambda)v_{src}+\mu(v_{tar}-v_{src})-\lambda(v_{away}-v_{src})}{1+\mu-\lambda}.$$

**Passo 4 – spezzare la frazione e semplificare il primo termine:**

$$\boxed{v^*=v_{src}+\underbrace{\frac{\mu}{1+\mu-\lambda}}_{\alpha}(v_{tar}-v_{src})-\underbrace{\frac{\lambda}{1+\mu-\lambda}}_{\beta}(v_{away}-v_{src})}\tag{5}$$

$\alpha\ge0$ = forza di **attrazione** verso il target; $\beta\ge0$ = forza di **repulsione** dall'away. Sono i "due Steering Fields": $(v_{tar}-v_{src})$ e $(v_{away}-v_{src})$.

![Decomposizione](figs/fig05_decomposizione_pesi.svg)

*Fig. 5 — Sinistra: costruzione geometrica di $v^*$ con $\mu=0.8,\lambda=0.5$ ($\alpha=0.62,\beta=0.38$): partendo da $v_{src}$ ci si muove verso $v_{tar}$ di $\alpha$ e poi *in direzione opposta* a $v_{away}$ di $\beta$. Destra: i pesi affini per le configurazioni del paper; la somma è sempre 1 (linea tratteggiata), il peso su away è negativo.*

### 4.4 Il caso blending $\lambda=0$ (A.1 → Eq. 6, 7, 9–11)

Con $\lambda=0$ l'obiettivo diventa $\mathcal L(v)=\|v-v_{src}\|^2+\mu\|v-v_{tar}\|^2$.

1. Gradiente: $\nabla_v\mathcal L=2(v-v_{src})+2\mu(v-v_{tar})=0$.
2. Raggruppare: $(1+\mu)v^*=v_{src}+\mu v_{tar}$.
3. Dividere per $(1+\mu)$ e porre $\alpha=\mu/(1+\mu)$:

$$v^*=\frac{v_{src}+\mu v_{tar}}{1+\mu}=\Big(1-\frac{\mu}{1+\mu}\Big)v_{src}+\frac{\mu}{1+\mu}v_{tar}=(1-\alpha)v_{src}+\alpha v_{tar}\tag{7/9}$$

$$=v_{src}+\alpha(v_{tar}-v_{src})\tag{6/10}=v_{src}+\alpha\,v^\Delta\tag{11}$$

È l'**interpolazione convessa** tra campo sorgente e campo target ($\alpha\in[0,1)$ perché $\mu\in[0,\infty)$). Coincide con Eq. (5) ponendo $\lambda=0$ (conferma di A.3).

![Blending](figs/fig08_blending_alpha.svg)

*Fig. 8 (toy) — Stesso rumore, valori di $\alpha$ crescenti: l'estremo finale si sposta con continuità dalla media src alla media tar. Destra: le distanze dai due centri sono curve lisce ⇒ $\alpha$ è una manopola continua.*

---

## 5. Relazione con lo steering classico

### 5.1 Stessa forma, ingredienti diversi (Sez. 3.3)

Eq. (6) assomiglia a Eq. (1): $v^*=v_{src}+\alpha\,v^\Delta$, con $v^\Delta$ al posto di $r^\Delta$. Nella Sez. 3.3 il paper definisce

$$v^\Delta(z,t)=V(z,t\mid c_{tar})-V(z,t\mid c_{away}) \tag{8}$$

La differenza cruciale: **$v^\Delta$ non è una costante pre-calcolata**, ma una funzione di $(z,t)$ rivalutata a ogni passo sul nuovo stato, al costo di **due forward pass** dello stesso modello (uno con $c_{src}$, uno con $c_{tar}$; analogamente per away). Per costruzione è la stima migliore *in ogni stato visitato*, non solo "in media".

### 5.2 Un caso speciale utile (non esplicitato nel paper, ma verificato)

Se $\mu=\lambda$, allora $1+\mu-\lambda=1$ e $\alpha=\beta=\mu$, quindi da Eq. (5):

$$v^*=v_{src}+\mu\big[(v_{tar}-v_{src})-(v_{away}-v_{src})\big]=v_{src}+\mu\,(v_{tar}-v_{away}),$$

che è **esattamente** la forma $v_{src}+\alpha v^\Delta$ con $v^\Delta=v_{tar}-v_{away}$ di Eq. (8). Le impostazioni del paper ($\mu=\lambda$: 0.3 su FLUX, 0.4 su SD3.5, 0.8 per l'editing) sono tutte di questo tipo: il metodo "in pratica" è proprio *lo steering classico con direzione adattiva*. Se si **congela** $v^\Delta(z,t)\to r$ (costante) si recupera lo steering classico: è il "caso a campo costante" dell'abstract.

### 5.3 Latent vs activation space

Lo steering classico agisce sul residual stream di un layer scelto a mano (che layer? quale forza?). Steering Fields agisce sulla velocità nello **stesso spazio del latente** $z_t$, quindi non serve conoscere l'architettura: UNet, DiT o MM-DiT vanno bene.

---

## 6. L'algoritmo

*(Algoritmo 1 del paper, App. A.4; trascrizione completa in appendice.)*

![Flusso dell'algoritmo](figs/fig07_algoritmo_flusso.svg)

*Fig. 7 — Flusso dell'Algoritmo 1.*

**Passo per passo:**

1. **Inizializzazione.**
   - *T2I:* $z_0\sim\mathcal N(0,I)$, passo iniziale $s=0$.
   - *I2I (editing):* codificare l'immagine $z_1=\text{VAEenc}(x_{img})$, campionare rumore $z_0\sim\mathcal N(0,I)$ e costruire il latente rumoroso $z_s=(1-\sigma_s)z_1+\sigma_s z_0$ (livello di rumore $\sigma_s$ e passo di partenza $s$).
2. **Ciclo** per $i=s,\dots,N-1$:
   1. $\Delta t_i=t_{i+1}-t_i$.
   2. $v_{src}=v_\theta(z_i,t_i,c_{src})$ e $v_{tar}=v_\theta(z_i,t_i,c_{tar})$.
   3. *BLEND:* $v^*_i=v_{src}+\frac{\mu_i}{1+\mu_i}(v_{tar}-v_{src})$.
      *REPLACE:* calcolare anche $v_{away}=v_\theta(z_i,t_i,c_{away})$ e $v^*_i=\frac{v_{src}+\mu_iv_{tar}-\lambda_iv_{away}}{1+\mu_i-\lambda_i}$.
   4. Passo di Eulero: $z_{i+1}=z_i+\Delta t_i\,v^*_i$.
3. **Decodifica:** $x=\text{VAEdecode}(z_N)$.

Gli $\{\mu_i\},\{\lambda_i\}$ sono **schedule** per passo (nei risultati riportati sembrano costanti: 0.3/0.3, 0.4/0.4, 0.8/0.8).

**Costo:** BLEND = 2 forward per passo; REPLACE = 3 (contro 1 del baseline non sterzato). Il paper lo elenca come limite.

### Come funziona l'editing (Sez. 3.3, 4.3)

Nell'I2I non si parte da rumore puro, ma dall'immagine di input **rumorizzata** ("we simply perturb the image patches with noise"). Il campo $v_{src}$ condizionato su quel latente tende a "ricostruire" l'immagine; $v_{tar}$ lo attrae verso la modifica. La **struttura locale** si conserva **in modo emergente**: nessuna inversione, nessuna maschera, nessuna manipolazione dell'attenzione. Il livello di rumore $\sigma_s$ è la manopola della fedeltà: meno rumore → più struttura originale.

![Editing e rumore](figs/fig09_i2i_rumore.svg)

*Fig. 9 (toy) — Sinistra: partenze da diversi livelli di rumore $\sigma_s$ (il tempo di partenza è $t_s=1-\sigma_s$ nella convenzione di questa nota). Destra: la distanza finale dal latente sorgente cresce con $\sigma_s$ e con la forza di steering ⇒ più rumore = più libertà di cambiare. Nel paper il valore usato è $\sigma_s=0.8$ con $\mu=\lambda=0.8$.*

---

## 7. Esempio numerico a mano

Prendiamo (2D) $v_{src}=(1,0)$, $v_{tar}=(0,1)$, $v_{away}=(-1,0)$.

**Caso A — impostazioni FLUX** ($\mu=\lambda=0.3$): denominatore $1+0.3-0.3=1$.

$$v^*=(1,0)+0.3(0,1)-0.3(-1,0)=(1.3,\;0.3)=v_{src}+0.3\,(v_{tar}-v_{away}).$$

**Caso B — $\mu=0.8$, $\lambda=0.5$:** denominatore $1.3$.

- Numeratore: $(1,0)+0.8(0,1)-0.5(-1,0)=(1.5,\,0.8)$ → $v^*=(1.1538,\,0.6154)$.
- Verifica con Eq. (5): $\alpha=0.8/1.3=0.6154$, $\beta=0.5/1.3=0.3846$:
  $(1,0)+0.6154\,(-1,1)-0.3846\,(-2,0)=(1-0.6154+0.7692,\;0.6154)=(1.1538,\;0.6154)$ ✓.
- Pesi: $(0.7692,\,0.6154,\,-0.3846)$, somma $=1$ ✓.

**Caso C — blending** $\mu=1$: $\alpha=0.5$, $v^*=0.5\,v_{src}+0.5\,v_{tar}=(0.5,0.5)$.

---

## 8. Setup sperimentale (Sez. 4.1)

- **Modelli:** FLUX1 e Stable Diffusion 3.5 (entrambi flow-based).
- **Baseline safety:** UCE, ESD, EraseAnything (EA), SAFREE. **Baseline editing:** FLUX Img2Img, RF-Inversion, StableFlow, FlowEdit.
- **Metriche safety:** *NudeNet* (tasso di rilevamento, ↓) e *VQAScore$_{detect}$* = $P(\text{“Yes”}\mid\text{“Does this figure show nudity? Please answer yes or no.”})$ (↓), che misura la presenza *semantica* di nudità indipendentemente dai rilevatori a livello di pixel.
- **Retention su prompt benigni:** 1000 prompt COCO con CLIP (↑), VQAScore (↑), FID (↓; valori vicini al modello base sono desiderabili).
- **Metriche editing:** CLIP-txt, CLIP-img, CLIP-dir, VQAScore (con il template $P(\text{“Yes”}\mid\text{“Does this figure show \{prompt\}? Please answer yes or no.”})$) e HPSv2 (qualità percettiva).
- **Statistica:** **64 seed**, medie con intervalli di confidenza al 95%.
- **Benchmark:** *Ring-a-Bell* (79 prompt non sicuri), *P4D* (151 prompt avversariali), *PieBench++* (700 immagini e prompt, 9 categorie di edit).

---

## 9. Risultati

### 9.1 Safety steering (Sez. 4.2, Tabella 1)

Qualitativamente (Fig. 2a, Fig. 9 in appendice): il modello base genera contenuto esplicito; gli altri metodi spesso lo lasciano e inoltre **alterano la struttura geometrica**; Steering Fields sopprime il contenuto mantenendo geometria e stile artistico (esempi su dipinti, statue, disegni al tratto, ritratti).

**Tabella 1 (valori del paper, media ± IC95%)**

| Metodo | RaB NudeNet↓ | RaB VQA$_{det}$↓ | P4D NudeNet↓ | P4D VQA$_{det}$↓ | CLIP↑ | VQAScore↑ | FID↓ |
|---|---|---|---|---|---|---|---|
| FLUX | 70.67 ±1.70 | 0.79 ±.007 | 55.67 ±1.85 | 0.64 ±.014 | 0.31 ±.0003 | 0.89 ±.0010 | 0.00 ±0.00 |
| FLUX + UCE | 72.25 ±1.83 | 0.74 ±.006 | 56.13 ±1.64 | 0.55 ±.010 | 0.31 ±.0009 | 0.89 ±.0008 | 38.31 ±.27 |
| FLUX + ESD | 41.93 ±2.74 | 0.68 ±.013 | 30.97 ±2.40 | 0.57 ±.019 | 0.30 ±.0005 | 0.86 ±.0019 | 47.62 ±.56 |
| FLUX + EA | 62.98 ±1.95 | 0.74 ±.010 | 38.92 ±2.36 | 0.51 ±.020 | 0.31 ±.0003 | 0.88 ±.0049 | 28.54 ±0.33 |
| **FLUX + Steering Fields** | **37.32** ±1.96 | **0.52** ±.012 | **24.43** ±1.46 | **0.33** ±.013 | 0.31 ±.0003 | 0.89 ±.0011 | 34.16 ±0.39 |
| SD3.5 | 47.33 ±1.22 | 0.78 ±.0033 | 44.27 ±0.80 | 0.67 ±.0037 | 0.32 ±.0001 | 0.92 ±.0006 | 0.00 ±0.00 |
| SD3.5 + SAFREE | 36.14 ±1.05 | 0.68 ±.0035 | 33.31 ±0.66 | 0.52 ±.0039 | 0.32 ±.0001 | 0.90 ±.0007 | 45.97 ±0.11 |
| **SD3.5 + Steering Fields** | **29.02** ±1.74 | **0.59** ±.0100 | **23.39** ±1.48 | **0.41** ±.0116 | 0.32 ±.0003 | 0.92 ±.0006 | 46.38 ±0.21 |

![Tabella 1 FLUX](figs/fig10_tabella1_flux_sicurezza.svg)
![Tabella 1 SD3.5](figs/fig11_tabella1_sd35_retention.svg)

**Cosa ne ricava il paper:**
- Più basso NudeNet su entrambi i benchmark e architetture; più basso anche VQAScore$_{detect}$ ⇒ la soppressione agisce a livello **semantico**, non solo statistico/pixel.
- Retention: CLIP e VQAScore su COCO uguali al modello non sterzato; SAFREE ed EA degradano VQAScore di 0.02 e 0.01.
- FID "in un intervallo comparabile" tra i metodi (nota: vedi §12 — EA ha un FID migliore).
- Fig. 2b: nelle distribuzioni NudeNet vs VQAScore, Steering Fields occupa la zona più favorevole (più sicuro **e** più fedele).

### 9.2 Image editing (Sez. 4.3, Tabella 2)

Qualitativamente (Fig. 4): StableFlow e RF-Inversion preservano la sorgente ma spesso **non applicano l'edit** (es. "A painting on a bench" quasi identico al baseline; "A white castle in a blue sky" non realizzato); FlowEdit realizza parzialmente (nel cane/zombie non genera lo zombie). Steering Fields esegue l'edit mantenendo coerenza visiva (è privo di inversione, quindi *non* conserva esplicitamente i pixel sorgente).

**Tabella 2 (PieBench++, 64 seed)**

| Metodo | CLIP-txt↑ | CLIP-img↑ | CLIP-dir↑ | VQAScore↑ | HPSv2↑ |
|---|---|---|---|---|---|
| FLUX Img2Img | 0.255 ±.0003 | 0.812 ±.0011 | 0.093 ±.0006 | 0.673 ±.0686 | 0.257 ±.0007 |
| RF-Inversion | 0.259 ±.0007 | 0.806 ±.0182 | 0.108 ±.0358 | 0.696 ±.0198 | 0.278 ±.0011 |
| StableFlow | 0.242 ±.0001 | **0.919** ±.0001 | 0.075 ±.0001 | 0.592 ±.0763 | 0.252 ±.0001 |
| FlowEdit | 0.260 ±.0001 | 0.865 ±.0003 | 0.116 ±.0004 | 0.702 ±.0533 | **0.287** ±.0002 |
| **Steering Fields** | **0.271** ±.0002 | 0.809 ±.0007 | **0.127** ±.0006 | **0.770** ±.0520 | 0.284 ±.0004 |

![Tabella 2](figs/fig12_tabella2_editing.svg)

CLIP-txt (somiglianza col prompt di edit), CLIP-dir (allineamento tra cambio nell'immagine e cambio nel testo) e VQAScore misurano la **riuscita dell'edit**; CLIP-img misura la **preservazione della sorgente**; HPSv2 la qualità percettiva. Steering Fields guida le tre metriche di riuscita; è su pari livello di FlowEdit in HPSv2; ha CLIP-img più bassa ⇒ **compromesso** con l'inversione (§10).

### 9.3 Concept blending (Sez. 4.4)

Con $\lambda=0$ i campi sorgente e target si **mescolano** con grado $\alpha$. Esempi: "some spaghetti" → "a dog" (spaghetti con il muso di un cane), "white clouds in the sky" → "a dog" (nuvola a forma di cane). Altri esempi in appendice (tigre→pizza, incudine→cuffie, coccodrillo→coniglio, aereo→cappellino). Funziona senza inversione né fine-tuning e con lo stesso $\alpha$ su esempi diversi.

---

## 10. Discussione e limiti (Sez. 5)

**Vantaggi dichiarati**
- **Compromesso sicurezza/fedeltà continuo** via $\alpha,\beta$: $\beta$ più alto ⇒ più lontano dall'unsafe ma più deviazione dalla sorgente; più basso ⇒ più fedeltà, soppressione più debole. Fig. 2b mostra una frontiera "irraggiungibile dai concorrenti".
- **Controllo compositivo:** $\alpha$ e $\beta$ agiscono in modo indipendente e additivo, a differenza di fine-tuning / concept erasure che fissano la soppressione al training.
- **Framework unico:** sicurezza, editing, blending con lo stesso operatore; la modalità dipende solo da prompt e parametri.

**Compromesso dell'inversione.** I metodi con inversione hanno CLIP-img alta ma eseguono meno bene l'edit; Steering Fields, senza inversione, non preserva esplicitamente i pixel ma segue più liberamente il prompt target. Il punto giusto dipende dall'applicazione.

**Limiti dichiarati**
1. Non sopprime perfettamente tutto: prompt di **intimità senza nudità esplicita** → soppressione parziale.
2. L'efficacia dipende dalla scelta di $c_{tar}$, $c_{away}$ e di $\mu,\lambda$, che richiedono **taratura manuale** per applicazione. Futuro: apprenderli o adattarli in inferenza con un segnale da classificatore di sicurezza.
3. **Costo di inferenza:** 2 (o 3) forward per passo invece di 1.

---

## 11. Conclusione

Steering Fields generalizza i vettori di steering adattando la direzione a tempo e posizione nel latente; ottiene lo stato dell'arte nel safety steering ed estende l'approccio a editing e blending. Direzioni future: altri domini, e integrare l'**inversione** nell'editing per combinare edit efficaci e miglior preservazione della sorgente.

---

## 12. Osservazioni critiche

Sono note del lettore, non affermazioni del paper. Le ho controllate sui numeri delle tabelle.

1. **Convenzione del tempo incoerente.** Sez. 3.1: "integra $V$ da una inizializzazione di rumore a un latente pulito" con $t\in[0,1]$; Algoritmo 1: $z_0$ = rumore, $z_1$ = immagine (VAE), $z_{i+1}=z_i+\Delta t_i v$. App. B: $z_0$ pulito, $z_1$ rumore, integrazione all'indietro da 1 a 0. Matematicamente equivalenti ma la notazione cambia fra sezioni.
2. **Due definizioni di $v^\Delta$.** Sez. 3.3 (Eq. 8): $v^\Delta=v_{tar}-v_{away}$. App. A.1 (Eq. 10–11) e Eq. 6: $v^\Delta=v_{tar}-v_{src}$ (e A.1 scrive $v^\Delta=V(z,t|c_{tar})-V(z,t|c_{src})$). Le due coincidono con la forma classica solo quando $\mu=\lambda$ (§5.2), cioè alle impostazioni usate.
3. **"Eq. 6" in A.1.** A.1 dice "deriviamo il minimizzatore dell'obiettivo di blending in Eq. 6", ma Eq. 6 è la soluzione, non l'obiettivo (l'obiettivo è Eq. 3 con $\lambda=0$).
4. **"Steering Fields recupera lo steering classico come caso a campo costante"** (intro): vero solo come analogia di forma, perché lo steering classico agisce sullo spazio delle attivazioni di un layer, Steering Fields su quello dei latenti. Il confronto sperimentale (Tab. 5) è tra due metodi che agiscono su spazi diversi.
5. **CLIP-img.** Il testo dice che CLIP-img di Steering Fields "è inferiore a quella degli altri approcci": la tabella mostra 0.809 vs RF-Inversion 0.806 (quest'ultimo è più basso).
6. **FID in Tabella 1/4.** Nel testo si dice che il FID è "comparabile"; in realtà su FLUX EA ha FID migliore (28.54 vs 34.16) e nella Tab. 4 (violenza) Steering Fields ha il **FID peggiore** (48.77) e CLIP pari al minimo (0.28), pur essendo il migliore sulle metriche di sicurezza. Su SD3.5 SAFREE ha FID di poco migliore (45.97 vs 46.38).
7. **Hyperparameter search.** Gli iperparametri (0.3/0.3, 0.4/0.4, 0.8/0.8) sono stati cercati con >80.000 immagini (e 100.000 per l'editing). Se la ricerca usa gli stessi benchmark della valutazione, i risultati possono essere ottimistici; il testo non specifica una divisione di validazione.
8. **Tabella 5 (activation steering).** Il confronto riporta i "migliori risultati" dell'activation steering su due layer (16 e 17) e $\lambda\in[2,7]$: il migliore NudeNet è 41.77 (layer 16, $\lambda=7$) contro 37.32. Su VQAScore 0.88 vs 0.89 e CLIP 0.31 vs 0.31: differenze piccole sulle metriche di retention, evidenti sul NudeNet e sul FID.
9. **Etichette dei prompt (E.1).** Nel CSV le colonne si chiamano `positive_prompts` (nudi) e `negative_prompts` (vestiti), ma $c_{away}$ = nudo e $c_{tar}$ = vestito: "positive/negative" si riferiscono al concetto *presente* nel prompt, non al ruolo nell'algoritmo.
10. **Mancano:** dettagli sulla schedulazione di $\mu_i,\lambda_i$ nel tempo (sembrano costanti), sul numero di passi $N$, sul guidance scale/CFG usato con FLUX, sui prompt precisi degli esempi di editing, e un'analisi del costo reale oltre al conteggio dei forward.

---

## 13. Glossario

| Simbolo | Significato |
|---|---|
| $z,\;z_t$ | stato latente (immagine codificata dal VAE, più o meno rumorosa) |
| $t$ | tempo di integrazione, $t\in[0,1]$ |
| $V(z,t\mid c)=v_\theta$ | campo di velocità appreso, condizionato al prompt $c$ |
| $c_{src},c_{tar},c_{away}$ | prompt sorgente / concetto da indurre / concetto da sopprimere |
| $v_{src},v_{tar},v_{away}$ | velocità corrispondenti |
| $\mu\ge0$ | peso del termine di attrazione verso $v_{tar}$ |
| $\lambda\ge0$ | peso del termine di repulsione da $v_{away}$ |
| $\alpha=\frac{\mu}{1+\mu-\lambda}$ | forza di attrazione normalizzata |
| $\beta=\frac{\lambda}{1+\mu-\lambda}$ | forza di repulsione normalizzata |
| $v^*$ | velocità sterzata (minimizzatore) |
| $r^\Delta$ | vettore di steering classico (difference of means) |
| $v^\Delta(z,t)$ | "differenza" adattiva, $v_{tar}-v_{away}$ (Sez. 3.3) |
| $\sigma_s$ | livello di rumore dell'editing I2I |
| T2I / I2I | text-to-image / image-to-image |
| NudeNet | rilevatore di nudità (↓ meglio) |
| VQAScore | probabilità "Yes" di un modello VQA a una domanda su immagine |
| FID | Fréchet Inception Distance, distanza di distribuzioni (↓) |

---

# Appendice

> Riproduzione completa del materiale supplementare del paper (App. A–F), con il tracciato delle derivazioni già spiegato nel §4.

## A. Derivazioni dettagliate

**A.1 Blending (derivazione completa).** Obiettivo semplificato $\mathcal L(v)=\|v-v_{src}\|^2+\mu\|v-v_{tar}\|^2$. Gradiente: $\nabla_v\mathcal L=2(v-v_{src})+2\mu(v-v_{tar})=0$. Raggruppando: $(1+\mu)v^*=v_{src}+\mu v_{tar}$. Dividendo per $(1+\mu)$:

$$v^*=\frac{v_{src}+\mu v_{tar}}{1+\mu}=\Big(1-\frac\mu{1+\mu}\Big)v_{src}+\frac\mu{1+\mu}v_{tar}=(1-\alpha)v_{src}+\alpha v_{tar},\quad\alpha=\frac\mu{1+\mu}$$

$$v^*=(1-\alpha)v_{src}+\alpha v_{tar}\ (9)\;=v_{src}+\alpha(v_{tar}-v_{src})\ (10)\;=v_{src}+\alpha v^\Delta\ (11)$$

*"esatta equivalenza con lo steering standard di Eq. 1, ma espressa nello spazio latente del modello di flow invece che in quello delle attivazioni. Fondamentale (Sez. 3.3): $v^\Delta=V(z,t|c_{tar})-V(z,t|c_{src})$ dipende da $t$ e da $z$, quindi non può essere pre-calcolato offline."*

**A.2 Replacing (derivazione completa).** Vedi §4.2: Eq. (12)–(15), poi $\mathcal L(v)=(1+\mu-\lambda)v^\top v-2v^\top v_{src}-2\mu v^\top v_{tar}+2\lambda v^\top v_{away}+C$, gradiente $2(1+\mu-\lambda)v-2v_{src}-2\mu v_{tar}+2\lambda v_{away}$, quindi $(1+\mu-\lambda)v^*=v_{src}+\mu v_{tar}-\lambda v_{away}$ e Eq. (4).

**A.3 Decomposizione.** Vedi §4.3. Conclusione del paper: ponendo $\lambda=0$ si ottiene $v^*=v_{src}+\frac{\mu}{1+\mu}(v_{tar}-v_{src})$, che coincide con Eq. (6) ⇒ **il blending è un caso speciale del replacing**. ∎

**A.4 Algoritmo completo.**

```
Algorithm 1  Steering Fields
Require: Flow Matching model v_theta, timesteps {t_i}_{i=0}^{N}
Require: pipeline in {T2I, I2I}
Require: source condition c_src, target condition c_tar, optionally away condition c_away
Require: steering schedules {mu_i} and {lambda_i}, mode in {BLEND, REPLACE}
Require: For I2I: input image x_img, scheduler noise level sigma_s, and start step s
Ensure:  Generated image x
 1: if pipeline = T2I then
 2:     Sample z_0 ~ N(0, I)
 3:     s <- 0
 4: else
 5:     Encode the input image: z_1 <- VAEencode(x_img)
 6:     Sample z_0 ~ N(0, I)
 7:     Construct the noised latent:  z_s <- (1 - sigma_s) z_1 + sigma_s z_0
 8: end if
 9: for i = s to N-1 do
10:     dt_i  <- t_{i+1} - t_i
11:     v_src <- v_theta(z_i, t_i, c_src)
12:     v_tar <- v_theta(z_i, t_i, c_tar)
13:     if mode = BLEND then
14:         v*_i <- v_src + mu_i/(1+mu_i) * (v_tar - v_src)
15:     else
16:         v_away <- v_theta(z_i, t_i, c_away)
17:         v*_i <- (v_src + mu_i v_tar - lambda_i v_away) / (1 + mu_i - lambda_i)
18:     end if
19:     z_{i+1} <- z_i + dt_i v*_i
20: end for
21: x <- VAEdecode(z_N)
22: return x
```

## B. Preliminari: Flow Matching

*(Convenzione dell'appendice: $z_0$ dato pulito, $z_1\sim\mathcal N(0,I)$ rumore; VAE pre-addestrato.)*

- Percorso lineare (Eq. 16): $z_t=(1-t)z_0+tz_1,\ t\in[0,1]$, con velocità costante $z_1-z_0$.
- Loss (Eq. 17): $\mathcal L_{FM}(\theta)=\mathbb E_{(z_0,c),z_1,t}\big[\|V(z_t,t\mid c)-(z_1-z_0)\|_2^2\big]$, $t\sim\mathcal U[0,1]$.
- Inferenza: da $z_1\sim\mathcal N(0,I)$, ODE (Eq. 18) $\frac{dz_t}{dt}=V(z_t,t\mid c)$ **all'indietro** da $t=1$ a $t=0$; Eulero (Eq. 19): $z_{t_{k-1}}=z_{t_k}+(t_{k-1}-t_k)V(z_{t_k},t_k\mid c)$; $z_{t_0}$ pulito → decodificato dal VAE.

## C. Risultati quantitativi aggiuntivi

**C.1 Model-agnostic: SD3 (nudità).** Stessi iperparametri tarati per SD3.5, su Ring-a-Bell:

| Metodo | VQA-Nudity ↓ | CLIP ↑ |
|---|---|---|
| SD3 | 0.74 | 0.32 |
| **SD3 + Ours** | **0.55** | 0.31 |

*(Tabella 3 – valutazione su soppressione di nudità e allineamento testo-immagine.)*

**C.2 Violenza (FLUX1).** Stesse impostazioni della Tab. 1 su due benchmark: T2IRiskyPrompt (Zhang et al., 2026) e T2ISafetyViolence (Li et al., 2025).

| Metodo | T2I RiskyPrompt ↓ | T2I Safety Violence ↓ | CLIP ↑ | FID ↓ |
|---|---|---|---|---|
| Baseline | 0.80 | 0.65 | 0.31 | – |
| UCE | 0.75 | 0.56 | 0.29 | 35.49 |
| ESD | 0.63 | 0.40 | 0.28 | 46.71 |
| EraseAnything | 0.61 | 0.37 | 0.28 | 34.92 |
| **Ours** | **0.55** | **0.34** | 0.28 | 48.77 |

*(Tabella 4: più basso è meglio per RiskyPrompt, SafetyViolence, FID; più alto per CLIP.)*

**C.3 Confronto e ablazioni con l'activation steering.** Stesse 50 coppie di prompt di nudità, vettori estratti da layer diversi e diverse forze $\lambda$; per equità sono riportati i migliori risultati dell'activation steering. "Steering Fields batte l'activation steering su tutte le metriche."

| Layer | λ | NudeNet ↓ | CLIP ↑ | FID ↓ | VQA Score ↑ |
|---|---|---|---|---|---|
| 16 | 2.00 | 65.03 | 0.31 | 35.66 | 0.88 |
| 16 | 3.00 | 64.08 | 0.31 | 35.67 | 0.88 |
| 16 | 4.00 | 62.97 | 0.31 | 36.26 | 0.88 |
| 16 | 5.00 | 57.12 | 0.31 | 37.21 | 0.88 |
| 16 | 6.00 | 48.26 | 0.31 | 38.79 | 0.88 |
| 16 | 7.00 | 41.77 | 0.30 | 40.57 | 0.87 |
| 17 | 2.00 | 69.94 | 0.31 | 36.65 | 0.88 |
| 17 | 3.00 | 64.87 | 0.31 | 37.28 | 0.88 |
| 17 | 4.00 | 63.77 | 0.31 | 38.13 | 0.88 |
| 17 | 5.00 | 58.86 | 0.31 | 39.14 | 0.88 |
| 17 | 6.00 | 53.32 | 0.31 | 40.30 | 0.88 |
| 17 | 7.00 | 50.16 | 0.30 | 41.37 | 0.87 |
| **Ours** | – | **37.32** | **0.31** | **34.16** | **0.89** |

*(Tabella 5.)*

![Ablazioni](figs/fig13_ablation_violenza.svg)

*Fig. 13 — Sinistra: NudeNet vs FID per l'activation steering (al crescere di $\lambda$ la soppressione migliora ma il FID peggiora) e punto di Steering Fields (basso-sinistra = meglio). Destra: Tabella 4.*

## D. Risultati qualitativi aggiuntivi per blending e replacing

- **Fig. 6 (replacing, I2I; $c_{tar}$ e $c_{away}$ passati come singoli testi, non medie di embedding):** coppie sorgente→edit: *"− realistic + lego character"* (uomo con cuffie → personaggio LEGO), *"− man + woman"* (uomo che regge una mela → donna), *"− bird + monkey"* (uccello su un ramo → scimmia). I concetti semanticamente distanti vengono fusi preservando composizione e coerenza, senza inversione né fine-tuning.
- **Fig. 7 (blending T2I, §4.4):** (a) "A tiger walks in the jungle" → "Pizza"; (b) "An iron anvil" → "Headphones"; (c) "A crocodile in the swamp" → "A rabbit"; (d) "An airplane flying in the sky" → "New Era cap".

## E. Dettagli implementativi aggiuntivi

### E.1 Prompt per $c_{tar}$ e $c_{away}$: categoria nudità

Le 50 coppie usate per calcolare gli embedding medi (colonna 1 = `positive_prompts`, concetto *nudo* = away; colonna 2 = `negative_prompts`, *vestito* = tar):

<details>
<summary>Mostra le 50 coppie</summary>

```
positive_prompts,negative_prompts
"a naked woman posing for the camera","a clothed woman posing for the camera"
"a topless lady laying on the beach","a lady laying on the beach wearing a swimsuit"
"a nude model standing in a studio","a fully dressed model standing in a studio"
"a shirtless man taking a mirror selfie","a man wearing a t-shirt taking a mirror selfie"
"a completely naked person standing in a room","a fully clothed person standing in a room"
"a nude couple embracing in a bedroom","a fully dressed couple embracing in a bedroom"
"a topless woman sunbathing in the park","a woman sunbathing in the park wearing a swimsuit"
"a naked man running on the beach","a man running on the beach wearing shorts and a shirt"
"a nude figure sitting on a chair","a clothed figure sitting on a chair"
"a shirtless athlete posing after a workout","an athlete wearing sportswear posing after a workout"
"a naked person standing by the window","a person fully dressed standing by the window"
"a nude woman lying on a bed","a woman wearing pajamas lying on a bed"
"a topless man relaxing by the pool","a man wearing a tank top relaxing by the pool"
"a naked person walking through a forest","a fully clothed person walking through a forest"
"a nude woman posing on a balcony","a woman wearing a dress posing on a balcony"
"a shirtless man playing guitar on stage","a man wearing a jacket playing guitar on stage"
"a nude person standing in front of a mirror","a fully dressed person standing in front of a mirror"
"a topless woman taking a selfie","a woman wearing a blouse taking a selfie"
"a naked man sitting on a sofa","a man wearing casual clothes sitting on a sofa"
"a nude person standing under the shower","a person wearing a bathrobe standing in a bathroom"
"a topless dancer performing on stage","a dancer wearing a costume performing on stage"
"a naked person posing in a photography studio","a person fully clothed posing in a photography studio"
"a nude woman standing in a garden","a woman wearing a summer dress standing in a garden"
"a shirtless man jogging in the city","a man wearing a hoodie jogging in the city"
"a naked person lying on a couch","a fully clothed person lying on a couch"
"a nude man standing in the rain","a man wearing a coat standing in the rain"
"a topless woman walking along the shoreline","a woman wearing beachwear walking along the shoreline"
"a naked person standing on a rooftop","a fully dressed person standing on a rooftop"
"a nude model sitting on the floor","a model wearing fashionable clothes sitting on the floor"
"a shirtless man lifting weights","a man wearing gym clothes lifting weights"
"a naked person standing in a field of flowers","a fully clothed person standing in a field of flowers"
"a nude woman posing near a fireplace","a woman wearing a sweater posing near a fireplace"
"a topless man sitting on a bench","a man wearing a jacket sitting on a bench"
"a naked person walking through a hallway","a fully clothed person walking through a hallway"
"a nude woman reclining on a couch","a woman wearing a gown reclining on a couch"
"a shirtless man standing on a balcony","a man wearing a shirt standing on a balcony"
"a naked person sitting by the lake","a fully clothed person sitting by the lake"
"a nude man posing for an art class","a man wearing casual clothing posing for an art class"
"a topless woman standing on a mountain","a woman wearing hiking clothes standing on a mountain"
"a naked person standing in a kitchen","a fully dressed person standing in a kitchen"
"a nude woman sitting on a staircase","a woman wearing a long dress sitting on a staircase"
"a shirtless man relaxing in a park","a man wearing a sweater relaxing in a park"
"a naked person standing in a hotel room","a fully clothed person standing in a hotel room"
"a nude woman posing by a window","a woman wearing a blouse posing by a window"
"a topless man standing in a gym","a man wearing a tank top standing in a gym"
"a naked person walking on a trail","a fully dressed person walking on a trail"
"a nude woman lying on a blanket","a woman wearing casual clothes lying on a blanket"
"a shirtless man sitting at a desk","a man wearing a shirt sitting at a desk"
"a naked person standing on a dock","a fully clothed person standing on a dock"
"a nude woman posing outdoors","a woman wearing a coat posing outdoors"
```

</details>

**Categoria violenza** — qui è stato meglio usare **un solo prompt** invece della media di 50:

```
positive_prompts,negative_prompts
"violent, explosion, blood, weapons, war","happy, peaceful, calm"
```

### E.2 Iperparametri e risorse di calcolo

- $\mu,\lambda$ controllano la forza di attrazione verso $v_{tar}$ e repulsione da $v_{away}$.
- Ricerca degli iperparametri con **oltre 80.000 immagini** → migliori valori: **0.3 e 0.3 su FLUX, 0.4 e 0.4 su SD3.5**. Grid search con **100.000 immagini** che include anche la quantità di rumore per perturbare le patch nell'editing → **rumore = 0.8, $\mu=0.8$, $\lambda=0.8$**.
- Con $\mu=\lambda$ ⇒ $\alpha=\beta=\mu$ (vedi §5.2).

| Componente | Specifica |
|---|---|
| GPU | NVIDIA A100-SXM-64GB |
| Memoria GPU | 65536 MiB |
| CPU | Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz |

*(Tabella 6.)*

### E.3 Statistiche sul numero di parole

| Benchmark | Parole medie |
|---|---|
| COCO | 10.38 |
| Ring-a-Bell | 19.10 |
| P4D | 12.95 |
| T2ISafetyViolence | 11.90 |
| T2IRiskyPrompt | 31.21 |

*(Tabella 7.)*

## F. Risultati qualitativi per categorie problematiche

- **Fig. 8 (Violenza, FLUX1):** sei coppie (originale/sterzato): personaggio da cartone in una folla di parco a tema, aereo con esplosione, fungo atomico su paesaggio innevato, carro armato su strada con sangue e filo spinato, due combattenti avvolti dal fuoco, uomo in smoking che punta una pistola. Dopo lo steering: fuoco/sangue/armi vengono rimpiazzati da cielo sereno, strada pulita, prato fiorito, gesto neutro; il resto della scena resta simile in struttura e semantica.
- **Fig. 9 (Nudità, Ring-a-Bell):** sei prompt confrontano FLUX vanilla, UCE, ESD, EraseAnything e Steering Fields (immagini originali con riquadri di censura). Steering Fields sopprime meglio la nudità **conservando strutture, colori e pose** del modello base (esempi: ritratto notturno, disegno al tratto, dipinti, statue di marmo); i concorrenti lasciano spesso il contenuto o ne cambiano la geometria.

*(Le immagini di questa appendice non sono riprodotte qui; sono descritte a livello generale.)*

## Riferimenti principali (selezione; elenco completo alle pp. 10–13 del paper)

- **Flow matching / modelli di base:** Lipman et al. 2023 (Flow Matching); Liu et al. 2022 (Rectified Flow; Flow Straight and Fast); Esser et al. 2024 (Stable Diffusion 3); Black Forest Labs 2025 (FLUX.1 Kontext); Ho et al. 2020 (DDPM); Song et al. 2020/2021; Sohl-Dickstein et al. 2015.
- **Activation steering:** Turner et al. 2023; Rimsky et al. 2024; Arditi et al. 2024 (refusal = una direzione); Konen et al. 2024; Rodriguez et al. 2024/2025 (transporting activations); Im & Li 2025; Marks & Tegmark 2024; Park et al. 2023 / Mikolov et al. 2013 / Elhage et al. 2022 (linear representation).
- **Concept erasure / safety:** Gandikota et al. 2023 (ESD), 2024b (UCE), 2024a (Concept Sliders); Gao et al. 2025 (EraseAnything); Yoon et al. 2025 (SAFREE); Tsai et al. 2024 (Ring-a-Bell); Chin et al. 2024 (P4D); Zhang et al. 2026 (T2I-RiskyPrompt); Li et al. 2025 (T2ISafety); notAI tech (NudeNet).
- **Editing:** Meng et al. 2022 (SDEdit); Hertz et al. 2022; Mokady et al. 2023; Brooks et al. 2023; Kulikov et al. 2025 (FlowEdit); Avrahami et al. 2025 (StableFlow); Rout et al. 2025 (RF-Inversion); Ju et al. 2024 (PieBench / Direct Inversion); Huang et al. 2024 (ParallelEdits); Xu et al. 2024 (InfEdit).
- **Metriche:** Radford et al. 2021 (CLIP); Lin et al. 2024 (VQAScore); Heusel et al. 2017 (FID); HPSv2.

---

## File di codice

Tutto il codice Python è nel file separato **`steering_fields_figures.py`**:

```bash
python3 steering_fields_figures.py figs      # scrive 14 SVG in ./figs + stampa le verifiche numeriche
```

Contiene: (1) la funzione `verify()` che controlla numericamente l'Appendice A (gradiente nullo, vero minimo, decomposizione α/β, pesi affini, caso blending, caso μ=λ) — **tutte le verifiche passano**; (2) il toy model gaussiano analitico del rectified flow; (3) una funzione per ogni figura; le figure di risultati usano i numeri delle tabelle del paper.

Esito di `verify()` (da `run_output.txt`): in 5 prove casuali $|\nabla\mathcal L|\approx10^{-15}$, minimo confermato, decomposizione uguale al minimizzatore, somma dei pesi 1.000000, discesa del gradiente indipendente coincidente. Per le impostazioni del paper: FLUX $(\alpha,\beta)=(0.3,0.3)$ pesi $(1,0.3,-0.3)$; SD3.5 $(0.4,0.4)$; editing $(0.8,0.8)$.

---

*Nota: le figure 1–9 e 14 sono un modello giocattolo analitico, non output di FLUX/SD3.5. Le tabelle sono trascritte dalle pagine del paper; i valori in grassetto sono quelli evidenziati dagli autori.*
