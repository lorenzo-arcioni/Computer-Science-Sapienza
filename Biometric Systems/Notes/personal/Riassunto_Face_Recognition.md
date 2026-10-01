# Face Recognition — Riassunto Completo

---

## 1. Introduzione: perché il volto come tratto biometrico

I due fattori principali che determinano la fattibilità di un sistema biometrico sono **Accuratezza/Affidabilità** e **Accettabilità**.

- **DNA**: il tratto più accurato, ma anche il più intrusivo.
- **Impronte digitali**: accurate e ben accettate, ma richiedono un utente collaborativo e consapevole (possono essere di scarsa qualità, es. lavoratori manuali).
- **Volto**: altissima accettabilità (l'utente può essere inconsapevole della cattura), ma l'accuratezza è ancora migliorabile. È naturale riconoscere una persona dal volto ed è comune essere fotografati (a differenza delle impronte, associate a un senso di sospetto criminale). Ha un tasso di riconoscimento molto alto in condizioni controllate; i dispositivi di acquisizione sono facili da distribuire; si integra bene con login/accesso logico e con applicazioni di controllo remoto.

Il volto è però un **oggetto complesso**: i punti MPEG-4 (Facial Animation Parameters, FAP) usati per sintetizzare e animare un volto mostrano come ogni punto rilevante possa cambiare configurazione relativa durante un cambio di espressione.

### 1.1 Confronto con le proprietà classiche dei tratti biometrici

| Proprietà | Valore per il volto |
|---|---|
| Universalità | Alto |
| Collectability | Alto (basta una foto) |
| Accettabilità | Alto |
| Permanenza | Medio (entro un certo range di età il volto non cambia) |
| Unicità | Basso (parenti stretti possono somigliarsi) |
| Performance (accuratezza raggiungibile) | Basso |
| Resistenza alla circonvenzione (spoofing) | Basso |

> **FTE = Failed To Enroll.** Il face recognition si usa in: forense, accesso logico (senza ingresso fisico), controllo di frontiera, sorveglianza di folle in situazioni speciali.

**Biometric Comparison Chart — confronto completo tra tratti biometrici** (H=High, M=Medium, L=Low):

| Tratto | Universalità | Unicità | Permanenza | Collectability | Performance | Accettabilità | Circonvenzione |
|---|---|---|---|---|---|---|---|
| Volto (Face) | H | L | M | H | L | H | L |
| Impronta digitale | M | H | H | M | H | M | M |
| Geometria della mano | M | M | M | H | M | M | M |
| Keystroke Dynamics | L | L | L | M | L | M | M |
| Hand vein | M | M | M | M | M | M | H |
| Iride | H | H | H | M | H | L | H |
| Retina | H | H | M | L | H | L | H |
| Firma | L | L | L | H | L | H | L |
| Voce | M | L | L | M | L | H | L |
| Termogramma facciale | H | H | L | H | M | H | H |
| DNA | H | H | H | L | H | L | L |

Il volto emerge come il tratto con la **migliore accettabilità** (assieme a firma e voce) ma tra i **peggiori** per unicità, performance e resistenza alla circonvenzione — da qui la necessità delle tecniche approfondite in questo documento.

### 1.2 Problemi principali

- **Intra-personal variations**: posa, illuminazione ed espressione (**PIE**) aumentano la variabilità tra template della stessa persona.
- **Inter-personal variations**: somiglianza tra soggetti anche non imparentati.
- **Aging**: rilevante soprattutto per applicazioni forensi; è difficile reperire **dataset longitudinali** (immagini della stessa persona dall'infanzia all'età adulta).

### 1.3 Dataset storici

AR-Faces, FERET, MIT, ORL, Harvard, MIT/CMU, CMU test set II. I dataset evolvono nel tempo: risolto un problema, ne emergono di nuovi.

### 1.4 Esperimento "Bubbles": quali regioni del volto sono davvero diagnostiche

Esperimento psicofisico che studia come un **osservatore umano** riesca a distinguere genere ed espressione da un volto, per capire quali regioni portino davvero informazione utile (invece di assumerlo a priori).

- Compiti testati: **GENDER** (maschio/femmina) ed **EXENK** (riconoscimento dell'espressione, es. gioia vs neutra).
- Tecnica delle **Bubbles**: si applicano maschere gaussiane casuali che lasciano visibili solo piccole "bolle" random dell'immagine originale ad ogni presentazione; si registra quali bolle portano a una classificazione corretta.
- Ripetendo il procedimento su molte prove si ottiene una mappa di **salienza/varianza** che mostra le regioni con il maggior potere diagnostico per quel compito.
- Le regioni più informative **non coincidono** necessariamente con quelle che un "osservatore ideale" (classificatore statistico ottimo) selezionerebbe, né sono le stesse per compiti diversi (es. per GENDER e per EXENK emergono regioni diverse — tipicamente occhi/sopracciglia e bocca sono più diagnostiche a seconda del task).
- **Implicazione per il face recognition automatico**: motiva l'uso di rappresentazioni **locali/basate su patch** (piuttosto che sull'intero volto in modo uniforme), poiché non tutte le regioni contribuiscono allo stesso modo al riconoscimento di genere, espressione o identità.

### 1.5 Dataset storici (FERET, AR-Faces, CMU-PIE)

- **FERET**: variazioni di posa (rotazione su sedia rispetto all'asse della camera), illuminazione (non per tutti i soggetti), tempo (sessioni diverse). 14.051 immagini totali, con file di posizione di occhi e bocca per ciascuna — necessari per **normalizzare** le immagini rispetto alla **distanza interoculare** (altrimenti dimensioni diverse alterano le geometrie estratte).
- **AR-Faces**: espressioni (Neutra — quella tipicamente usata in gallery, Sorriso, Rabbia, Urlo), diverse sorgenti di luce, occhiali da sole, sciarpa. Limite: sfondo omogeneo (irrealistico rispetto al rumore di sfondo reale).
- **CMU-PIE**: apparato che simula pitch (altezza della camera) e yaw (angolo camera); 68 soggetti, 608 foto a colori ciascuno, 13 pose, 43 condizioni di illuminazione, 4 espressioni.

---

## 2. Struttura di un Face Recognizer

1. **Acquisizione dell'immagine** (Image Capture)
2. **Miglioramento dell'immagine** (Enhancement: nitidezza, deblurring, contrasto)
3. **Detection e Localizzazione**:
   - *Detection* → risposta binaria (presenza/assenza dell'oggetto).
   - *Localizzazione* → fornisce anche la posizione esatta.
4. **Ritaglio della ROI** (Region of Interest) — riduce il tempo di elaborazione successivo; approcci a **patch** sono popolari perché gestiscono meglio occlusioni/illuminazione non uniforme.
5. **Estrazione delle feature e costruzione del template** (chiave biometrica).

La Face Localization deve essere indipendente da: posizione, orientamento (posa), scala, espressione, differenze tra soggetti, illuminazione, sfondo affollato (cluttered background — può generare falsi positivi).

> **Adversarial Fashion**: makeup o elementi sul volto possono ingannare la localizzazione, "nascondendo" il soggetto dal sistema.

---

## 3. Approcci alla Face Localization

- **Proprietà speciali dei pixel**: bordi, colore della pelle.
- **Proprietà geometriche del volto**: Constellation/Landmark (Feature searching).
- **Template matching**: correlazione, Snakes, Active Shape Models.
- **Image-based**: trattano la localizzazione come un problema generico di pattern recognition (classe "volti" vs "non volti") — SVM, reti neurali, Hidden Markov Model.
- **Approcci recenti basati su esemplari**: confronto con distribuzioni di riferimento (Constellation di landmark).
- **Deep Learning**: mappe di "partness" a livello di parti facciali locali, senza face detection preliminare.

### 3.1 Valutazione della localizzazione

- **False Positives**: percentuale di finestre classificate come volto che non contengono volti.
- **Not Localized Faces**: percentuale di volti non identificati.
- **C-Error**: tiene conto non solo della classificazione ma anche della **precisione della localizzazione**. È la distanza euclidea tra il centro reale del volto e quello stimato, **normalizzata** rispetto alla somma degli assi dell'ellisse che contiene il volto.

---

## 4. Algoritmo di Hsu, Mottaleb e Jain (approccio feature-based)

Composto da due macro-fasi: **Face Candidates Detection** e **Face Candidates Verification** (tramite rilevamento di feature facciali attese).

### 4.1 Pre-processing: Illumination Compensation

Il tono della pelle dipende dall'illuminazione complessiva della scena (e dal sensore). Si usa un **"reference white"** per normalizzare l'aspetto del colore:

1. Si calcola la **luma** di ogni pixel: somma pesata delle componenti RGB gamma-compresse (R'G'B').
2. Si ordinano i pixel per luma decrescente e si prende il **5% più luminoso** come riferimento (deve essere un numero sufficiente di pixel rispetto alla dimensione dell'immagine).
3. Se il riferimento è valido (numero sufficiente e colore medio non simile alla tonalità della pelle), si scala linearmente il livello di grigio medio del reference white fino a **255** (bianco puro in RGB), e tutte le altre componenti vengono scalate di conseguenza.

### 4.2 Color Space Transformation

RGB **non è uno spazio percettivamente uniforme**: colori vicini in RGB possono apparire diversi, e colori percepiti come simili possono avere valori RGB molto diversi (perché il computer non "interpreta" i colori, ma i numeri). Questo causa:

- **Over-segmentation**: pixel della stessa regione percettiva vengono divisi in regioni diverse.
- Il fenomeno opposto: colori vicini in RGB ma percettivamente diversi vengono uniti.

Per questo si preferisce effettuare la segmentazione della pelle in **un altro spazio colore** (tipicamente YCbCr).

### 4.3 Localizzazione basata su Skin Model — Segmentazione con soglia (Thresholding)

Metodo più semplice: **thresholding** (fisso o adattivo). Metodi popolari: massima entropia, **metodo di Otsu** (massima varianza), k-means (limite: k deve essere noto a priori).

**Formulazione matematica (metodo variance-based / Otsu):**

Immagine a livelli di grigio $G = [0, 1, ..., L-1]$, dimensione $M \times N$; $n_i$ = numero di pixel con livello di grigio $i$.

Probabilità normalizzata (istogramma):
$$
p_i = \frac{n_i}{M \times N}, \qquad p_i \ge 0, \qquad \sum_{i=0}^{L-1} p_i = 1
$$

Si dividono i pixel in due classi $C_0$ (livelli $0..t$) e $C_1$ (livelli $t+1..L-1$), con probabilità aggregate:
$$
\omega_0 = \sum_{i=0}^{t} p_i, \qquad \omega_1 = \sum_{i=t+1}^{L-1} p_i
$$

Si calcolano media e varianza per ciascuna classe, e la **varianza within-class** come somma pesata delle varianze di classe. La soglia ottima è quella che **minimizza** questa varianza within-class:

$$
t^{*} = \arg\min_{t \in G} \left[\sigma_w^2(t)\right]
$$

**Connected Components**: una regione senza buchi, tale che da ogni pixel si può raggiungere ogni altro pixel senza uscire dalla regione. I componenti connessi vengono raggruppati per vicinanza spaziale e colore simile → sono i **candidati volto**. Gli **operatori morfologici** (erosione, dilatazione) chiudono i buchi in queste regioni omogenee.

### 4.4 Localizzazione degli occhi (Eye Map)

Si costruiscono due mappe:

**Chrominance Map** — la regione attorno agli occhi ha valori alti della componente **blu (Cb)** e bassi della componente **rossa (Cr)** (per la concavità intorno all'occhio):

$$
EyeMapC = \frac{1}{3}\left\{ (C_b^2) + (\tilde{C_r}^2) + \left(\frac{C_b}{C_r}\right) \right\}, \qquad \tilde{C_r} = 255 - C_r
$$

dove $C_b^2$, $(\tilde{C_r})^2$, $C_b/C_r$ sono tutti normalizzati nel range $[0, 255]$.

**Luminance Map** — sfrutta il fatto che gli occhi contengono sia zone chiare che scure, evidenziate da operatori morfologici (dilatazione ed erosione con elementi strutturanti emisferici): il numeratore è la **dilatazione**, il denominatore l'**erosione**.

$$
EyeMapL = \frac{Y(x,y) \oplus g_\sigma(x,y)}{Y(x,y) \ominus g_\sigma(x,y) + 1}
$$

dove $Y$ è il canale di luminanza, $g_\sigma$ è l'elemento strutturante (emisferico), $\oplus$ indica la **dilatazione** e $\ominus$ l'**erosione** (il "+1" al denominatore evita la divisione per zero).

Il Chroma Map viene migliorato con **equalizzazione dell'istogramma** (riduce il contrasto tra regioni per mantenere l'equiprobabilità dei pixel); le due mappe vengono poi combinate con un **AND**; segue dilatazione, mascheramento e normalizzazione per scartare le altre regioni del volto ed evidenziare gli occhi.

### 4.5 Localizzazione della bocca

Nella regione della bocca la componente **Cr è più alta di Cb**; la risposta a Cr/Cb è bassa, mentre la risposta a Cr² è alta.

$$
MouthMap = C_r^2 \cdot \left(C_r^2 - \eta \cdot \frac{C_r}{C_b}\right)^2, \qquad \eta = 0.95\,\frac{\dfrac{1}{n}\sum_{(x,y)\in FG} C_r^2(x,y)}{\dfrac{1}{n}\sum_{(x,y)\in FG} \dfrac{C_r(x,y)}{C_b(x,y)}}
$$

dove $FG$ è la regione del volto (face group) su cui si calcolano le medie e $n$ il numero di pixel in tale regione. Il fattore $\eta$ pesa il rapporto $C_r/C_b$ in base al contenuto medio della regione, enfatizzando la zona della bocca rispetto al resto del volto.

### 4.6 Contorno del volto (Face Contour)

Si analizzano tutti i triangoli formati da due candidati-occhio e un candidato-bocca. Si controllano: variazioni attese di luma e orientamento del gradiente (direzione della variazione di colore tra pixel vicini) all'interno del triangolo, geometria e orientamento del triangolo, e presenza di un contorno del volto attorno al triangolo. Ogni candidato riceve un punteggio; si seleziona il triangolo con lo score più alto.

### 4.7 Operatori morfologici: Dilatazione ed Erosione

Sia $X$ l'insieme delle coordinate del pixel (immagine binaria), $K$ l'elemento strutturante (kernel), $K_x$ la traslazione di $K$ centrata in $x$.

**Dilatazione**: l'insieme di tutti i punti $x$ tali che l'intersezione tra $K_x$ e $X$ **non è vuota**. Se, centrando $K$ su $x$, c'è qualunque sovrapposizione con pixel neri, allora $x$ diventa nero (qualunque fosse il suo colore originale).

**Erosione** (operazione opposta): l'insieme di tutti i punti $x$ tali che $K_x$ è un **sottoinsieme** di $X$. Tutti i pixel neri nel kernel devono essere neri anche nell'immagine originale; basta un solo pixel bianco nel kernel dove nell'immagine c'era nero perché il pixel centrale diventi bianco.

---

## 5. Algoritmo Viola-Jones (approccio image-based, AdaBoost)

Efficiente su immagini di buona qualità (dal punto di vista della posa). Training lento, detezione molto veloce (real-time). Può essere applicato anche a occhi/bocca in modo gerarchico.

> **Buone regole generali per il training** (valide per qualsiasi classificatore addestrato su dati, non solo Viola-Jones): il training set deve contenere il **maggior numero possibile di condizioni diverse** che si incontreranno nel test set o nella vita reale (variazioni di posa, illuminazione, qualità), deve includere le **distorsioni attese**, e deve avere sia **buoni campioni positivi** sia **buoni campioni negativi** — questi ultimi intesi come esempi che assomigliano alla classe target ma non lo sono (es. per un rilevatore di volti, pattern con struttura geometrica confondibile ma che non sono volti), in modo da rendere il classificatore più discriminativo.

### 5.1 AdaBoost — Teoria generale

Il **Boosting** combina $M$ weak learner (classificatori lineari) in un classificatore forte $H_M(x)$ (ensemble non lineare). Un **classificatore debole** ha accuratezza appena superiore al caso (50%, come una moneta).

$$
x \text{ è un pattern da classificare}, \quad h_i(x) \in \{-1, +1\} \text{ sono i classificatori deboli}
$$
$$
\alpha_i \ge 0 \text{ sono i pesi relativi}, \quad \sum_{i=1}^{M} \alpha_i \text{ è un fattore di normalizzazione}
$$

$$
H_M(x) = \frac{\sum_{i=1}^{M} \alpha_i h_i(x)}{\sum_{i=1}^{M} \alpha_i}
$$

**AdaBoost (Adaptive Boost)** apprende la sequenza ottima di classificatori deboli e i relativi pesi. Dato un training set $\{(x_1,y_1),...,(x_N,y_N)\}$ con $y_i = -1$ (non-volto) o $+1$ (volto):

- Si mantiene una **distribuzione di pesi** $[w_1, ..., w_N]$, una per pattern, inizialmente uguale per tutti.
- Dopo l'iterazione $m$, ai pattern **più difficili da classificare** viene assegnato un peso $w_i^{(m)}$ più alto → maggiore attenzione all'iterazione $m+1$.
- Ad ogni round, si sceglie la linea (classificatore debole) che ottimizza la somma dei pesi correttamente classificati rispetto a quelli scorretti — minimizzando il peso dei pattern misclassificati.
- Il classificatore finale forte è la **combinazione (non lineare)** di più rette/classificatori deboli in sequenza.

### 5.2 Haar Features

Feature rettangolari (verticali, orizzontali, o miste — pattern chiaro/scuro):

$$
\text{Value} = \sum(\text{pixel nell'area bianca}) - \sum(\text{pixel nell'area nera})
$$

Per una regione di rilevazione 24×24, il numero di possibili feature rettangolari è ~**180.000**. Questo rende impraticabile la valutazione esaustiva → si usa **AdaBoost per selezionare un piccolo sottoinsieme** di feature discriminative.

### 5.3 Integral Image

Per calcolare rapidamente le somme sotto le aree bianche/nere, si pre-calcola l'**immagine integrale**:

$$
II(x,y) = \sum_{x' \le x,\, y' \le y} I(x', y')
$$

cioè la somma di tutti i pixel sopra e a sinistra di $(x,y)$. Grazie a questa rappresentazione, la somma dei pixel in un rettangolo qualunque si ottiene con **sole 4 operazioni** (es. rettangolo $D$ = punto 4 + punto 1 − punto 2 − punto 3), indipendentemente dalla dimensione del rettangolo.

### 5.4 Costruzione del classificatore forte (livello di dettaglio)

Supponiamo di aver costruito $M-1$ classificatori deboli $\{h_m(x)\}$ e di voler costruire $h_M(x)$. Il nuovo classificatore confronta il valore di una feature $z_{k^*}$ con una soglia fissata $t_{k^*}$:

$$
h_M(x) = +1 \text{ se } z_{k^*} > t_{k^*}, \qquad = -1 \text{ altrimenti}
$$

I parametri $z_{k^*}$ e $t_{k^*}$ si scelgono minimizzando l'**errore di classificazione**: per ogni feature $z_k$ si sceglie la soglia $t_k$ che minimizza l'errore; si sceglie poi la feature $z_{k^*}$ che, con la sua soglia ottima, dà l'errore più basso in assoluto.

> AdaBoost apprende quindi una sequenza di classificatori deboli $h_m$ e li combina in un classificatore robusto $H_M$, minimizzando il limite superiore dell'errore di classificazione.

**Complessità computazionale del training**: $O(MNT)$, dove $M$ = numero di filtri, $N$ = numero di esempi, $T$ = numero di soglie possibili.

### 5.5 Cascade Classifier

Sequenza di classificatori di complessità crescente. Ogni sotto-finestra deve superare **tutti** gli stadi per essere classificata come volto; un esito negativo in qualunque stadio porta al **rigetto immediato** (nessun recupero possibile in seguito). Questo permette di scartare rapidamente la maggior parte delle finestre non-volto, concentrando il calcolo sulle regioni promettenti.

**Addestramento della cascata:**
- Si regola la soglia di ogni classificatore debole per **minimizzare i falsi negativi** (non l'errore totale, disponibile solo a fine catena).
- Ogni classificatore è addestrato sui **falsi positivi** degli stadi precedenti.
- Esempio di progressione (dati storici del training):
  - 1 feature → 100% detection rate, ~50% false positive rate
  - 5 feature → 100% detection rate, 40% false positive rate (20% cumulativo)
  - 20 feature → 100% detection rate, 10% false positive rate (2% cumulativo)

---

## 6. Face Recognition 2D — Rappresentazioni possibili del volto

Un'immagine $I(x,y)$ è una funzione bidimensionale (matrice) che associa un valore (grigio o colore) a ogni posizione. Può essere rappresentata come:

1. **Vettore linearizzato** (concatenando righe) — punto in uno spazio $n$-dimensionale (**Image Space**), dove $n = w \times h$.
2. **Feature Space**: si estraggono feature (Gabor, DCT, LBP, PIFS...) invece di usare i pixel grezzi.

### 6.1 Il problema della Curse of Dimensionality

Quando la dimensionalità aumenta, il volume dello spazio cresce così rapidamente che i dati diventano **sparsi** (Hughes effect). Conseguenze:
- Servono quantità di dati enormi per garantire significatività statistica.
- Con una distanza euclidea su molte coordinate, le distanze tra coppie di campioni diventano poco discriminanti (problema per k-NN).

Le tecniche di **dimensionality reduction** (PCA, LDA...) affrontano questo problema identificando i sottospazi più informativi.

---

## 7. PCA / Eigenfaces

**PCA (Principal Component Analysis)** è una procedura statistica che effettua una trasformazione **ortogonale** (= verso dimensioni non correlate). Feature correlate hanno lo stesso andamento → basso potere discriminativo e ridondanza. In signal processing è nota anche come trasformata discreta di **Karhunen–Loève (KLT)**.

### 7.1 Procedura matematica

Training set: $TS = \{x_i \in \mathbb{R}^n \mid i=1,...,m\}$ (immagini linearizzate).

**Vettore medio:**
$$
\bar{x} = \frac{1}{m}\sum_{i=1}^{m} x_i
$$

**Matrice di covarianza** ($n \times n$):
$$
C = \frac{1}{m}\sum_{i=1}^{m}(x_i - \bar{x})(x_i - \bar{x})^T
$$

Si calcolano gli **autovettori** (eigenfaces) e i corrispondenti **autovalori** di $C$; si ordinano gli autovalori in ordine decrescente e si scelgono i $k$ autovettori con gli autovalori più alti. Questi definiscono il nuovo sottospazio $k$-dimensionale (proiezione), preservando la "energia" (varianza) massima con il $k$ più piccolo possibile.

> L'autovettore con l'autovalore più alto identifica la direzione di **massima variazione** dei dati (massima informazione). Gli autovalori rappresentano le varianze lungo i rispettivi autovettori. Questi autovettori sono automaticamente ortogonali tra loro.

**Proiezione** di un'immagine (dopo sottrazione della media):
$$
Proj(x) = \varphi_k^T (x - \bar{x})
$$

dove $\varphi_k$ è la matrice di proiezione ($k \times n$, composta dai $k$ autovettori come colonne), $x - \bar{x}$ è $n \times 1$, il risultato è un vettore $k \times 1$ di **coefficienti di proiezione**.

Sirovich e Kirby (primi ad applicare PCA al riconoscimento facciale) dimostrarono che un volto si può rappresentare e **ricostruire approssimativamente** con poche "eigenpicture" e i relativi coefficienti. Turk e Pentland estesero l'idea: le proiezioni sugli eigenfaces diventano **feature di classificazione** — approccio **olistico** (Holistic).

### 7.2 Riconoscimento

Ogni immagine di gallery viene proiettata sul sottospazio KLT; la classificazione avviene con il criterio del **vicino più prossimo** (nearest neighbor) tra i coefficienti di proiezione — non è necessario confrontare pixel per pixel. Non tutti i soggetti della gallery devono partecipare al training: serve solo che il training set catturi le variazioni più rilevanti.

### 7.3 Vantaggi e svantaggi delle Eigenfaces

| Vantaggi | Svantaggi |
|---|---|
| Fase di identificazione veloce | Fase di training lenta |
| Se si preservano gli autovettori, è possibile ricostruire l'informazione originale | Aggiungere molti nuovi soggetti richiede ri-addestramento |
| | Molto sensibile a illuminazione, posa, occlusioni |

**Problema principale**: la varianza massimizzata da PCA dipende sia dalla separazione **inter-classe** (utile) sia dalle differenze **intra-classe** (dannosa). In presenza di forti variazioni PIE, la similarità nello spazio delle facce può riflettere l'espressione/illuminazione piuttosto che l'identità → PCA è più incline a **False Rejection**.

---

## 8. LDA / Fisherfaces

Belhumeur propose le Fisherfaces come variante che risolve i limiti di PCA, applicando la **Fisher's Linear Discriminant (FLD)**, nel contesto della **Linear Discriminant Analysis (LDA)**.

> **Differenza chiave con PCA**: LDA è **supervisionata** — il training set è partizionato secondo le etichette di classe reali (identità), mentre PCA è non supervisionata.

### 8.1 Formulazione

Training set partizionato in $S$ classi $PTS = \{P_1,...,P_S\}$, ciascuna di cardinalità $m_i$. Si proietta $x$ su una retta: $y = w^T x$.

**Centroide di classe e centroide dei centroidi:**
$$
\mu_i = \frac{1}{m_i}\sum_{j=1}^{m_i} x_j, \qquad \mu_{TS} = \frac{1}{m}\sum_{i=1}^{S} m_i \mu_i
$$

**Matrice di covarianza per ciascuna classe:**
$$
C_i = \frac{1}{m_i}\sum_{j=1}^{m_i}(x_j - \mu_i)(x_j - \mu_i)^T
$$

**Within-class scatter matrix** (somma pesata delle covarianze di classe):
$$
S_W = \sum_{i=1}^{S} m_i C_i
$$

**Between-class scatter matrix:**
$$
S_B = \sum_{i=1}^{S} m_i (\mu_i - \mu_{TS})(\mu_i - \mu_{TS})^T
$$
(per due classi: $S_B = (\mu_1 - \mu_2)(\mu_1-\mu_2)^T$)

> ⚠️ La sola distanza tra le medie proiettate **non è un buon criterio**: non tiene conto dello scatter (dispersione) within-class. Un asse può avere media più distante ma classi più sovrapposte di un altro asse con media meno distante ma classi più compatte.

### 8.2 Criterio di Fisher

Nella proiezione, lo scatter within-class diventa:
$$
\tilde{s}_i^2 = \sum_{y \in P_i}(y - \tilde{\mu}_i)^2 = \sum_{x \in P_i} w^T(x-\mu_i)(x-\mu_i)^T w = w^T S_i w
$$
$$
\tilde{s}_1^2 + \tilde{s}_2^2 = w^T S_W w
$$

La differenza tra medie proiettate:
$$
(\tilde{\mu}_1 - \tilde{\mu}_2)^2 = w^T (\mu_1-\mu_2)(\mu_1-\mu_2)^T w = w^T S_B w
$$

**Criterio di Fisher (da massimizzare):**
$$
J(w) = \frac{|\tilde{\mu}_1 - \tilde{\mu}_2|^2}{\tilde{s}_1^2 + \tilde{s}_2^2} = \frac{w^T S_B w}{w^T S_W w}
$$

Si cerca la proiezione che **massimizza il numeratore** (separazione tra classi) e **minimizza il denominatore** (dispersione interna alle classi).

### 8.3 Soluzione: problema agli autovalori generalizzato

La matrice di proiezione ottima $W^*$ è composta dagli autovettori corrispondenti ai maggiori autovalori del problema:

$$
W^{*} = [w_1^*|w_2^*|...|w_{C-1}^*] = \arg\max \frac{|W^T S_B W|}{|W^T S_W W|} \;\Rightarrow\; (S_B - \lambda_i S_W)\, w_i^{*} = 0
$$

> LDA è più robusta di PCA rispetto alle variazioni PIE, poiché separa esplicitamente inter-classe da intra-classe.

---

## 9. Feature Space: Wavelet, Gabor, EBGM

### 9.1 Limiti della Trasformata di Fourier

Fourier fornisce solo informazione di **frequenza**, non di **posizione temporale/spaziale** — adatta a segnali stazionari. Due segnali diversi possono apparire simili nello spettro se contengono le stesse frequenze ma in tempi/posizioni diverse.

### 9.2 Wavelet Transform

La wavelet fornisce **simultaneamente** informazione di tempo e frequenza. Una **"mother wavelet"** è una funzione a supporto compatto (si annulla fuori da un intervallo finito) e oscillatoria, traslata e scalata per generare l'intera famiglia di wavelet figlie.

**Continuous Wavelet Transform (1D):**
$$
CWT_x^{\psi}(\tau, s) = \Psi_x^{\psi}(\tau, s) = \int x(t)\, \psi^*_{\tau,s}(t)\, dt, \qquad \psi_{\tau,s} = \frac{1}{\sqrt{s}}\psi\left(\frac{t-\tau}{s}\right)
$$

($f^*(x)$ è il coniugato complesso di $f(x)$.)

**Estensione a due dimensioni:**
$$
\psi_\theta(b_x, b_y, x, y, x_0, y_0) = \frac{1}{\sqrt{b_x b_y}}\, \psi_\theta\left(\frac{x-x_0}{b_x} + \frac{y-y_0}{b_y}\right)
$$

- **Frequenza alta** → wavelet stretta → cattura **dettagli piccoli**.
- **Frequenza bassa** → wavelet larga → cattura **dettagli grandi** (macro-struttura).

### 9.3 Filtri di Gabor

I filtri di Gabor sono compatibili con l'espressione wavelet 2D (a meno dello spostamento spaziale). Il **filtro di Gabor 2D** $\psi_{f,\theta}(x,y)$ è un **segnale sinusoidale complesso modulato da un kernel Gaussiano**: la Gaussiana è caratterizzata da deviazione standard e orientamento; il filtro ha componente reale e immaginaria (direzioni ortogonali).

Funziona come **filtro passa-banda** per la distribuzione di frequenza spaziale locale — ottima risoluzione sia nel dominio spaziale che in quello di frequenza. Sono ispirati alla corteccia visiva dei mammiferi.

Un **filter bank** varia per:
- **Orientamento** (es. 8 direzioni)
- **Scala/frequenza spaziale**

I filtri Gabor enfatizzano i contorni (bordi di occhi, naso, bocca, nei, cicatrici). La convoluzione può essere fatta su tutti i pixel (alta dimensionalità), su una griglia regolare, o solo sui punti salienti (griglia "a mano" — con rischio di catturare punti diversi tra immagini diverse — oppure punti ad alta energia di risposta).

### 9.4 Elastic Bunch Graph Matching (EBGM)

Applicazione dei filtri Gabor a grafi facciali. Un **jet** descrive una piccola patch di grigi attorno a un pixel $p=(x,y)$, ottenuto da una trasformata wavelet (**5 frequenze × 8 orientamenti = 40 coefficienti**).

- Un **grafo immagine** collega una collezione sparsa di jet con informazioni sulla loro posizione relativa.
- Un **Bunch Graph** raccoglie, per ogni punto fiduciale (landmark: pupille, angoli della bocca, punta del naso, orecchie, ecc.), i jet di **N persone diverse** (nel caso descritto: 70 persone).
- Costruzione in due fasi: (1) struttura qualitativa (nodi+archi, jet e distanze) fornita manualmente su un'immagine iniziale; (2) estrazione semi-automatica dalle immagini campione, con correzione manuale decrescente dei punti fiduciali mal identificati.
- Un **Face Bunch Graph (FBG)** è una struttura a pila di più modelli individuali, tutti con la stessa struttura a griglia, per coprire un'ampia varietà di forme/tipi di volto.
- Un insieme di jet riferiti allo stesso punto fiduciale si chiama **bunch**.

---

## 10. Local Binary Pattern (LBP)

Operatore **basato sulla texture**, lavora pixel per pixel senza kernel di convoluzione classico.

### 10.1 Calcolo base (finestra 3×3, raggio 1)

Per ogni pixel centrale, si usa il suo valore come **soglia locale adattiva**: per ciascun vicino, se il valore è **≥** al centro → 1, altrimenti → 0. Si interpreta la sequenza circolare di 8 bit come un numero binario (pesi $2^0, 2^1, ..., 2^7$ a partire da un punto di partenza convenzionale), ottenendo un valore tra 0 e 255 per il pixel centrale nella nuova immagine LBP.

Si può anche estrarre un'**immagine di contrasto**: sottraendo il valore medio dei vicini "più bassi" dal valore medio dei vicini "più alti o uguali" rispetto al centro.

### 10.2 LBPH (istogramma)

L'uso più diffuso non è l'immagine LBP in sé, ma l'**istogramma** dei valori di grigio estratti dall'immagine LBP (non dall'immagine originale).

### 10.3 Varianti

- **Raggio variabile**: si può usare un raggio maggiore (es. raggio 2.5 → finestra 7×7, fino a 12 vicini). Finestre piccole → dettagli fini; finestre grandi → dettagli macro (come per le wavelet).
- **Uniform Patterns**: pattern con **al massimo due transizioni** 0→1 o 1→0 nella sequenza circolare. Servono a risparmiare memoria: con $P$ vicini, invece di $2^P$ bin si usano solo $P \times (P-1) + 2$ bin (il "+2" per gli stati Spot e Spot/Flat), poiché i pattern uniformi identificano **strutture significative** (bordi, angoli).
- **Rotation invariant LBP**: per ogni finestra, si sceglie sempre il **numero decimale più basso** ottenibile ruotando ciclicamente il pattern, così da essere invarianti a rotazioni dell'immagine.

**Feature vector finale**: l'immagine è partizionata in una griglia $k \times k$; si calcola l'istogramma LBP (eventualmente normalizzato) per ogni cella; si concatenano tutti gli istogrammi.

---

## 10bis. Deep Learning per il Face Recognition

Prima di qualsiasi pipeline deep, le immagini vengono tipicamente **normalizzate a livello di pixel** — riscalate da [0,255] a [0,1] o [-1,1] — per tre motivi: stabilità numerica (evita gradienti esplosivi in backpropagation), convergenza più rapida (superficie della loss più simmetrica) e maggiore robustezza a bruschi cambi di illuminazione (la "I" di PIE).

### 10bis.1 DeepFace (Facebook, 2014)

Include un esplicito passaggio di **frontalizzazione 3D**:
1. Si localizzano **6 punti fiduciali** e si produce un ritaglio 2D del volto;
2. Si individuano **67 punti fiduciali**, con **triangolazione di Delaunay**, che guidano un modello generico 2D→3D;
3. Questo modello **frontalizza la posa** del volto prima che l'immagine allineata venga passata alla rete profonda.

### 10bis.2 FaceNet (Google, 2015) e la Triplet Loss

FaceNet apprende direttamente un **embedding** tramite una **Triplet Loss**: proietta i volti in uno spazio compatto in cui la distanza tra i punti corrisponde alla similarità tra identità (volti della stessa persona vicini, volti di persone diverse lontani).

Ogni tripletta di training è composta da:
- **Anchor**: immagine di riferimento di una persona;
- **Positive**: un'altra immagine della stessa persona;
- **Negative**: un'immagine di una persona diversa.

La loss calcola le distanze nello spazio degli embedding e impone di avvicinare Anchor–Positive e allontanare Anchor–Negative di **almeno un margine minimo α**.

Per un training efficace è necessario l'**hard triplet mining**: si selezionano triplette "difficili", cioè casi in cui il Positive è insolitamente lontano dall'Anchor oppure il Negative insolitamente vicino, forzando la rete a imparare caratteristiche più discriminative. Gli embedding vengono tipicamente **L2-normalizzati** (riscalati a norma unitaria) prima del confronto.

---

## 11. Metodi globali vs metodi locali/feature-based

### 11.1 Metodi globali (Olistici)

Es. PCA, LDA, ICA (rappresentano il volto come combinazione lineare di vettori base), reti neurali, **sparse representations** (l'intero training set è la "base", senza riduzione — il vettore più vicino appartiene alla stessa persona).

| Vantaggi | Svantaggi |
|---|---|
| Non distruggono informazione (l'intera immagine è usata) | Ogni pixel è considerato rilevante anche quando non lo è |
| Flessibili, adattabili per compensare variazioni PIE | Computazionalmente costosi |
| | Richiedono alta correlazione tra training e test |
| | Non performano bene con forti variazioni PIE/scala |

### 11.2 Metodi locali/feature-based (es. EBGM, LBP)

| Vantaggi | Svantaggi |
|---|---|
| Robusti a variazioni di posizione | Scelta arbitraria delle feature importanti |
| Possono essere resi invarianti a scala/orientamento/luce | Se le feature non sono discriminative, nessun processing successivo può compensare |
| Rappresentazione compatta, matching veloce | |

### 11.3 Neural Networks

Il volto linearizzato passa a una prima rete per riduzione dimensionale, poi a una rete di classificazione (un neurone per pixel sarebbe eccessivo).

- **Vantaggi**: riducono l'ambiguità tra classi simili; con accorgimenti sono robuste alle occlusioni.
- **Svantaggi**: richiedono più immagini per il training; soggette a **overfitting** (rete della stessa dimensione dell'input), **overtraining** (perdita di generalizzazione), inefficienza al crescere del numero di soggetti (database size).

### 11.4 Sistemi basati su grafo

Filtri e funzioni di localizzazione individuano punti di riferimento connessi da archi pesati (grafo per volto; il matching = matching tra grafi).

- **Vantaggi**: robusti a posa e illuminazione; non richiedono ri-addestramento.
- **Svantaggi**: training lento; testing molto lento (graph matching è **NP-Hard**).

### 11.5 Termogramma

Acquisizione tramite sensore termico (variazioni di temperatura della pelle), immagine segmentata e indicizzata.

- **Vantaggi**: robusto a illuminazione e variazioni temporali; efficiente indoor/outdoor.
- **Svantaggi**: dispositivi costosi; molto sensibili al movimento del soggetto (bassa risoluzione); influenzati dallo stato emotivo; un vetro tra soggetto e sensore rende inefficace la cattura.

---

## 12. Riconoscimento facciale 3D

### 12.1 Cosa risolve il 3D rispetto al 2D

| Variazione | 2D | 3D |
|---|---|---|
| Posa (roll, pitch, yaw) | Influenza | **Non influenza** (il modello si può ruotare) |
| Illuminazione | Influenza | **Non influenza** (si può sintetizzare) |
| Make-up | Influenza | **Non influenza** (solo geometria/volume) |
| Espressione | Influenza | **Influenza ancora** (deforma il modello 3D) |
| Aging | Influenza | **Influenza ancora** (i volumi anatomici cambiano) |
| Chirurgia plastica | Influenza | Dipende dall'estensione dell'intervento |
| Occlusioni | Influenza | **Influenza ancora** |

**Pro del 3D**: molta più informazione, modelli robusti a molte distorsioni, possibilità di sintetizzare pose/espressioni 2D da un modello 3D.
**Contro**: costo dei dispositivi, costo computazionale, rischio per gli occhi di alcuni scanner (es. laser).

### 12.2 Spazi di rappresentazione

- **2D — Intensity Images**: valore del pixel = intensità della luce riflessa (dipende da superficie e illuminazione); anche termiche o NIR.
- **2.5D — Range Image**: griglia 2D dove ogni pixel rappresenta la **distanza** dal sensore (non è un modello 3D completo, ma con più range image da pose diverse si può ricostruire un modello 3D accurato).
- **3D — Shaded Model**: mesh di punti e poligoni; più piccoli i patch poligonali, migliore la ricostruzione.

> In sintesi: le immagini 2D codificano l'interazione luce-riflettanza; le 2.5D/3D codificano l'interazione con la **forma**.

### 12.3 Dispositivi di acquisizione 3D

| Dispositivo | Principio | Costo | Accuratezza | Robustezza a illuminazione | Rischio |
|---|---|---|---|---|---|
| Stereoscopico | Più camere, feature omologhe triangolate | Basso | Media | Bassa | Nessuno |
| Structured light scanner | Pattern di luce proiettato, deformazione misurata | Medio-Alto | Medio-Alta | Medio-Alta | Nessuno |
| Laser scanner | Specchio oscillante + fotocellula, fascio deformato dalla superficie | Medio-Alto | Alta | Alta | **Pericoloso per gli occhi** |

### 12.4 Problemi tipici delle scansioni 3D/2.5D

- **Rumore/spike**: dovuto a polvere nell'aria o altre interferenze — correggibile sfruttando il fatto che una superficie reale cambia **gradualmente** (non a scatti).
- **Buchi**: pixel non catturati — si usano Gaussian smoothing, interpolazione lineare o simmetrica, operatori morfologici.
- **Smoothing**: inaccuratezze di cattura da smussare.
- **Allineamento**: i landmark principali devono essere perfettamente allineati tra scansioni diverse, altrimenti si introducono distorsioni nel modello.

### 12.5 Costruzione del modello 3D (da 2.5D a mesh)

1. Per ciascuna immagine 2.5D si genera una **nuvola di punti 3D** (x, y equispaziate; z = profondità dal valore 2.5D).
2. La nuvola viene **triangolarizzata** (mesh di triangoli adiacenti).

Un **poligono** è una sequenza di punti coplanari connessi da segmenti; la **coplanarità** è l'approssimazione intrinseca della rappresentazione a mesh (poligoni più piccoli → approssimazione migliore).

**Normali:**
- Normale a un poligono: vettore perpendicolare al piano del poligono, calcolato come **prodotto vettoriale** di due vettori sul piano.
- Normale a un vertice: somma normalizzata delle normali (unitarie) dei poligoni adiacenti.

Dopo la mesh geometrica, si assegnano i **colori** (struttura percettiva) a vertici/poligoni; il **texture mapping** calcola posizione e orientamento di una texture sulla superficie.

### 12.6 Shape from Shading e Morphable Model

**Shape from shading**: sfrutta la relazione tra intensità e forma, assumendo una **superficie Lambertiana**. Non è possibile recuperare la forma da una singola immagine (né con dispositivi 3D puri) → servono più immagini o tecniche statistiche (PCA su rappresentazioni di forma della stessa classe).

Kemelmacher-Shlizerman e Basri (2011) evitano di rappresentare i volti come combinazione di centinaia di modelli 3D memorizzati: usano una **singola immagine 2D** per "modellare" (morphare) un **singolo modello di riferimento** fino a ricostruire la forma 3D cercata.

**Formalizzazione del modello morphable:**

La geometria di un volto è rappresentata da un vettore di forma con le coordinate X,Y,Z degli $n$ vertici 3D:
$$
S = (X_1, Y_1, Z_1, ..., X_n, Y_n, Z_n)^T \in \mathbb{R}^{3n}
$$

Un vettore analogo $T$ rappresenta la texture (valori RGB). Il modello morphable è costruito su un dataset di $m$ volti esemplari, ciascuno con i suoi $S_i, T_i$. Nuove forme e texture si ottengono come **combinazione lineare** degli esemplari:
$$
S_{\text{new}} = \bar{S} + \sum_{i=1}^{m} \alpha_i (S_i - \bar{S}), \qquad T_{\text{new}} = \bar{T} + \sum_{i=1}^{m} \beta_i (T_i - \bar{T})
$$
(coefficienti $\alpha_i, \beta_i$ ottimizzati per adattare il modello all'immagine 2D di input tramite un **Face Analyzer**).

I morphable model permettono anche di **sintetizzare espressioni facciali** (per aggiungere campioni in gallery, riprodurre l'espressione del probe, o simulare l'invecchiamento). **FaceGen Modeller** è un tool commerciale che genera modelli 3D da poche foto 2D, adattando un modello generico morphable alla forma e al colore del soggetto.

### 12.7 Feature 3D

Le informazioni più importanti sono **normali** e **curvature locali/globali**:
- **Crest Lines**: aree con la maggiore curvatura (variazioni più brusche di orientamento).
- **Local Curvature**: rappresentata con colori (più chiaro = maggiore curvatura).
- **Local Features**: segmentazione in regioni di interesse (occhi, naso...) con landmark 3D annotati.

### 12.8 Allineamento: ICP (Iterative Closest Point)

Algoritmo per allineare geometricamente due modelli 3D quando è nota una stima iniziale della posa relativa (fondamentale: **migliore la stima iniziale, minore il costo computazionale**, essendo ICP molto oneroso).

**Procedura:**
1. Trovare un allineamento iniziale approssimativo (curve, punti rilevanti, punti di massima variazione).
2. Calcolare la distanza sommando le distanze tra punti omologhi nelle due superfici allineate approssimativamente.
3. Calcolare la trasformazione che **minimizza** tale distanza.
4. Applicare la trasformazione e **ripetere** finché la distanza non è sotto una soglia.

### 12.9 Normal Maps

Si proietta la geometria 3D in 2D secondo regole di proiezione standard, generando una **Normal Map**: un'immagine RGB regolare dove i canali R, G, B corrispondono rispettivamente alle coordinate X, Y, Z della normale di superficie (l'intensità del colore rappresenta la lunghezza della componente).

- Permette un mapping inverso dal modello 3D a un'immagine 2D confrontabile.
- Leggere/processare un'immagine 2D è molto più veloce che processare un modello 3D.
- La **Difference Map** è l'immagine differenza tra due normal map: ogni pixel rappresenta la **distanza angolare** tra le due normali in quel punto.

### 12.10 Riconoscimento 3D tramite Iso-Geodesic Stripes

Le **distanze geodetiche** (percorso più breve tra due punti su una superficie curva) sono **poco influenzate** dai cambi di espressione — le regioni convesse del volto sono meno affette dall'espressione rispetto a quelle concave.

**Weighted Walkthroughs (WW) — definizione 2D:**

Dati due punti $a=(x_a,y_a)$ e $b=(x_b,y_b)$, la proiezione di $b$ rispetto ad $a$ su ciascun asse può essere: prima, coincidente o dopo → codificata con una coppia di indici $\langle i,j \rangle$, $i,j \in \{-1,0,+1\}$:

$$
i = \begin{cases} -1 & x_b < x_a \\ 0 & x_b = x_a \\ +1 & x_b > x_a \end{cases}
\qquad
j = \begin{cases} -1 & y_b < y_a \\ 0 & y_b = y_a \\ +1 & y_b > y_a \end{cases}
$$

Dati due regioni continue $A$ e $B$, si conta il numero di coppie $(a,b)$ connesse dallo stesso displacement $\langle i,j \rangle$: $w_{i,j}(A,B)$. La matrice $3\times 3$ dei pesi è il **2D Weighted Walkthrough (2DWW)**, che modella lo spostamento relativo tra due insiemi di punti. L'estensione a 3D dà il **3DWW**; gli **indici direzionali** sono misure aggregate in $[0,1]$ derivabili direttamente dai pesi della matrice 3DWW.

**Iso-geodesic stripes:**

Ogni volto è partizionato in un numero fisso di **strisce iso-geodetiche** di uguale larghezza, pseudo-concentriche e centrate sulla **punta del naso**:

1. Si calcola la **distanza geodetica normalizzata** $\gamma$ tra ogni punto del volto e la punta del naso (algoritmo di **Dijkstra** sui punti della superficie).
2. Si **quantizza** $\gamma$ in $N$ intervalli $c_1,...,c_N$: la striscia $i$-esima raccoglie i punti con $\gamma \in c_i$.
3. Il **fattore di normalizzazione** è la distanza euclidea occhi-naso (somma delle distanze tra punta del naso e i due endocantion) → garantisce invarianza a scala e cambi di espressione.

Ogni volto diventa un **grafo** con nodi = strisce, e archi annotati con i **3DWW** tra tutte le coppie di punti delle due strisce corrispondenti → riconoscimento ridotto a un problema di **graph matching**, molto efficiente da calcolare e confrontare.

---

## 13. Face Recognition — Valutazione e protocolli storici

### 13.1 Limiti delle metriche pure

FAR, FRR, CMS... da sole non bastano: sono troppo legate al contesto sperimentale specifico. Bisogna considerare anche: numero di dataset usati (generalizzabilità), dimensione delle immagini (risoluzione vs rumore del sensore), dimensione di probe/gallery, numero/livello delle variazioni tollerate (dipende dall'applicazione).

### 13.2 FERET Protocol

Prima del database FERET, molti articoli riportavano risultati >95% su database piccoli (<50 individui), senza un protocollo standard comune → impossibile confrontare gli algoritmi in modo affidabile.

- **Aug94**: prima valutazione — riconoscimento su gallery di 316 individui, test di falso allarme (rigetto di volti non in gallery), baseline degli effetti della posa.
- **Mar95**: gallery più ampia (817 individui); introduzione delle "duplicate images" (stesso soggetto, foto di gallery scattata in data diversa).
- **Sep96** (finale): matching tra 3.323 e 3.816 immagini (~12,6 milioni di confronti); risultati riportati per: stesso giorno/stessa luce; giorni diversi; oltre un anno di distanza; stesso giorno ma luce diversa. Due versioni: algoritmi **parzialmente automatici** (con coordinate occhi fornite) e **completamente automatici** (solo immagini).

### 13.3 Face Recognition Vendor Test (FRVT)

- **FRVT 2006**: dati "sequestered" (mai visti prima dai ricercatori); infrastruttura standard **Biometric Experimentation Environment (BEE)**.
- **FRVT 2002**: dimostrò il drastico calo di prestazioni con immagini dello stesso soggetto ma illuminazioni molto diverse (anche indoor/outdoor nello stesso giorno).
- **FRVT 2000/2002**: dimostrarono la difficoltà nel riconoscere volti non frontali (le prestazioni calano al variare della posa orizzontale e/o verticale).

### 13.4 Face Recognition Grand Challenge (FRGC)

Obiettivo: aumentare le prestazioni degli algoritmi 2D e 3D. Dataset con oltre 50.000 record (Training Set + Validation Set). A FAR fissato a 0.1%, i sistemi esistenti raggiungevano l'80% di verification rate; obiettivo FRGC = **98%**.

Include anche modelli 3D e test di confronto 2D vs 3D:
- **Controlled indoor still vs indoor still**: PCA come baseline (livello minimo di accuratezza); matrice di similarità test-vs-gallery.
- **Indoor multi-still vs indoor multi-still**: texture e forma di due modelli 3D confrontate separatamente, poi fuse in un unico valore.
- **3D vs 3D**: confronto tra N>1 immagini di test e M>1 immagini di gallery, con fusione dei valori ottenuti.

---

## 14. Problemi aperti nel Face Recognition e strategie di mitigazione

### 14.1 Illuminazione

Le condizioni di luce cambiano molto (ora del giorno, indoor/outdoor); luci dirette producono ombre e zone iperilluminate. Il feature vector di uno stesso soggetto in condizioni diverse può risultare **più vicino** a un soggetto diverso con illuminazione simile che a se stesso con illuminazione diversa. Nell'identificazione open set, il ranking può risentirne indipendentemente dalla soglia scelta.

**Tre famiglie di algoritmi:**
1. **Shape from Shading**: estrae la forma 3D dai livelli di grigio, la proietta in 2D per un confronto migliore.
2. **Representation Based Methods**: classificatori intrinsecamente robusti all'illuminazione (LBP **non** rientra qui: nelle zone d'ombra i suoi valori diventano inaffidabili).
3. **Generative Methods**: da un modello 3D si generano molte immagini con diverse illuminazioni, usate per l'enrollment.

### 14.2 Posa

- **Sistema multiview**: enrollment sotto pose diverse, ciascuna come istanza separata in gallery (caso particolare di multiple instance).
- **Pose correction systems**: basati su modelli 3D che vengono ruotati/morphati per ottenere un template comparabile con il probe, o per ottenere una **forma canonica**.

### 14.3 Occlusioni

Capelli, occhiali da sole, sciarpe, make-up. Con occlusioni ampie non c'è modo affidabile di "riempire" l'informazione mancante senza introdurre errore. Le strategie più diffuse: elaborazione **localizzata** (patch-wise) — si **rileva** l'occlusione e si tratta come regione "don't care", confrontando solo la parte residua del volto.

### 14.4 Tempo ed Età

Anche una settimana tra due acquisizioni può ridurre le prestazioni. Il **termogramma** è più robusto a queste variazioni. Le variazioni di età sono ancora più critiche (utile per identificazione di persone scomparse/ricercate): si cerca di "mimare" il processo di invecchiamento sulla base di studi etnografici sullo sviluppo delle regioni somatiche.

### 14.5 Altre variazioni

Make-up, chirurgia plastica, cambio di genere, cambio di peso.

---

## 15. Face Recognition — Spoofing e Anti-Spoofing

### 15.1 Definizioni

Uno **spoofing attack** biometrico consiste nell'ingannare l'applicazione presentando una copia o un'imitazione del tratto biometrico usato per l'autenticazione, per farsi passare per un utente legittimo. Richiede di conoscere il/i tratti biometrici utilizzati dal sistema.

> **Spoofing vs Camouflage/Disguise**: nello spoofing l'attaccante presenta un tratto biometrico artefatto per fingere di **essere** qualcun altro (essere riconosciuto come utente genuino). Nel Camouflage/Disguise, l'obiettivo opposto: presentare un tratto artefatto per **non essere riconosciuti** (fingere di non essere se stessi) — es. introdurre elementi estranei sul volto per far fallire la detection.

### 15.2 Punti di attacco del sistema

Un sistema biometrico può essere attaccato: nel canale di trasmissione, nel feature extractor, nel comparatore (iniettando dati per forzare l'esito desiderato), nel database dei template (iniettando template non appartenenti a utenti enrollati). In caso di **aggiornamento automatico della gallery**, è possibile "avvelenare" la sotto-gallery di un soggetto iniettando gradualmente immagini sempre più diverse, sfruttando la soglia di accettazione, fino a farsi riconoscere anche con un'identità diversa.

- **Attacchi Indiretti**: sui canali, sul database ecc. — competenza della **cybersecurity**.
- **Attacchi Diretti (Presentation Attacks)**: l'obiettivo è rilevare l'attacco stesso, non riconoscere la persona.

### 15.3 Classificazione degli attacchi di spoofing

- **2D Spoofs**: superficie 2D — foto (hard copy, hard copy con foro per gli occhi, schermo) o **video** (Replay Attack, video preregistrato del soggetto attaccato).
- **3D Spoofs**: superficie tridimensionale — **maschere** (Mask Attacks).

### 15.4 Contromisure: livelli di intervento

- **Sensor-level (Hardware based)**: sfruttano proprietà intrinseche di un corpo vivo (riflettanza, micromovimenti oculari involontari assenti in foto/maschere). Esempio: **Eye Blink**.
- **Feature Extractor level (Software based)**: statiche (microtexture dell'immagine acquisita) o dinamiche (feature di movimento naturale).
- **Score level fusion**: fusione di anti-spoofing e riconoscimento in un unico modulo, oppure anti-spoofing preliminare seguito da riconoscimento solo se il probe è genuino.

### 15.5 Print Attack — tecniche di Liveness Detection

**Struttura 3D vs 2D (structure from motion / depth):** un volto vivo è un oggetto 3D, una foto è planare → si può stimare la profondità dal movimento. Svantaggi: difficile con testa ferma; molto sensibile a rumore e illuminazione.

**Optical Flow:** stima i vettori di movimento confrontando la posizione di ciascun pixel tra frame consecutivi. Vulnerabile a "photo motion in depth" e "photo bending" (piegare la foto per simulare la profondità).

**Approccio multimodale (audio-video):** correla il movimento delle labbra con l'audio durante il parlato.

**Eye Blink (analisi del battito di ciglia):**
- Frequenza naturale: 15-30 battiti/minuto (un battito ogni 2-4 secondi); durata media ~250 ms.
- Con una camera a ≥15 fps (intervallo tra frame ≤70 ms) si catturano almeno 2 frame per battito.
- Stati dell'occhio: $Q = \{\alpha: \text{aperto}, \gamma: \text{chiuso}, \beta: \text{ambiguo}\}$; pattern tipico: $\alpha \to \beta \to \gamma \to \beta \to \alpha$.
- Modellazione con **Conditional Random Field (CRF)**: dato un grafo $G=(V,E)$ e una sequenza di osservazioni $S$, $(Y,S)$ è un CRF se, condizionato su $S$, le variabili $Y$ obbediscono alla proprietà di Markov rispetto al grafo — si predice la probabilità dello stato successivo dato quanto osservato finora. Un classificatore AdaBoost viene addestrato per identificare precisamente lo stato "occhio chiuso" e avviare il rilevamento del pattern di battito.

**Micro-texture Analysis:**
- Volto reale e stampa riflettono la luce diversamente (oggetto 3D non rigido vs oggetto planare rigido); pigmenti naturali vs pigmenti di inchiostro (spesso con componenti metalliche) riflettono in modo diverso.
- Si usano **Multi-scale Local Binary Patterns** (finestre di raggio crescente: 3×3 raggio 1, poi 5×5 raggio 2 con 8 o 16 vicini).
- Pipeline tipica (LBP uniforme, notazione $LBP^{u2}_{P,R}$): volto rilevato, ritagliato e normalizzato a 64×64 px → $LBP^{u2}_{8,1}$ diviso in regioni 3×3 sovrapposte (overlap 14px) → istogrammi locali a 59 bin concatenati in un istogramma da **531 bin** → si aggiungono due istogrammi globali con $LBP^{u2}_{8,2}$ (59 bin) e $LBP^{u2}_{16,2}$ (243 bin) → istogramma finale di **833 bin** (531+59+243) → training di un **SVM** (kernel RBF) su campioni positivi (volti reali) e negativi (volti falsi).

**Captured-Recaptured Approach (Kose & Dugelay):** usa **LBP Variance (LBPV)** rotation-invariant + pre-processing con **Difference of Gaussian (DoG)** (isola la banda di frequenza più discriminante). Un'immagine ricatturata (foto di una foto) ha meno nitidezza/alta frequenza di una cattura diretta — visibile nello spettro di Fourier 2D. LBPV aggiunge informazione di **contrasto** (varianza locale) come peso ai bin dell'istogramma LBP. Il confronto avviene tramite **distanza chi-quadrato** tra istogrammi del probe e i modelli genuino/falso.

**Gaze Stability (Ali et al.):** la coordinazione spazio-temporale di occhi, testa e (eventualmente) mano nel seguire uno stimolo visivo è diversa in un tentativo genuino rispetto a certi tipi di spoofing. In un attacco fotografico serve un movimento manuale della mano per orientare la foto verso lo stimolo → questa componente manuale altera la coordinazione occhio-testa. Lo stimolo appare in punti/sequenze casuali (per prevenire attacchi video predittivi). Si estraggono i landmark con **STASM**; le **Collinearity features** = errore quadratico medio (MSE) tra posizione attesa (dalla traiettoria dello stimolo) e posizione rilevata dei landmark, concatenato in un vettore $F_{colin}$; le **Colocation features** ($F_{coloc}$) considerano la differenza di posizione quando lo stimolo riappare nello stesso punto. Due soglie: una per rilevare movimenti minimizzati (per aggirare il sistema), una per rilevare movimenti "troppo ripetibili".

**Optical Flow Correlation:** confronta il movimento della testa con quello dello sfondo — se si muovono **insieme** è un segnale di spoofing (volto e sfondo dovrebbero muoversi in modo indipendente). Non funziona con maschere o con foto piegate.

**Image Distortion Analysis (IDA):** combina riflessi speculari (carta stampata o schermo LCD), sfocatura da mancato fuoco, distorsione di cromaticità e contrasto (gamma di colori più ristretta di un'immagine naturale), distorsione di diversità del colore (risoluzione più bassa). Le feature dei singoli fattori vengono concatenate e usate per addestrare un **classificatore ensemble** (fusione finale delle risposte separate). Modello di **doppia riflessione** (doppia cattura → distorsione amplificata). Serve un dataset con stampe di più stampanti e catture da più smartphone (fattori come inchiostro e dispositivo di visualizzazione cambiano le feature).

### 15.6 Replay Attack (Video)

Si genera un particolare **moiré pattern aliasing**, dovuto alla sovrapposizione tra la griglia di pixel del video originale e quella dello schermo su cui viene mostrato. Il pattern occupa l'intero frame se il video riempie tutta l'inquadratura, altrimenti solo la regione ricatturata.

Rilevamento con **Multi-scale LBP + DSIFT**: ogni frame → detection del volto → estrazione MLBP e DSIFT (**8 bin di orientamento, 16 segmenti**), usati singolarmente o combinati per il rilevamento dello spoof.

### 15.7 3D Mask Attack

Metodi basati sull'assunzione di superficie planare falliscono contro le maschere 3D. Contromisure:

- **Analisi di riflettanza**: materiali diversi (plastica, silicone, pasta di carta) riflettono diversamente a diverse lunghezze d'onda; si sceglie la lunghezza d'onda dopo aver ispezionato le curve di **albedo** (caratteristica della luce riflessa) di pelle e materiali, anche in funzione della distanza. Limite: serve conoscere l'albedo di ogni possibile materiale; un materiale nuovo può eludere il sistema.
- **LBP su texture + depth map**: maschere e pelle reale hanno texture e levigatezza diverse — LBP applicato sia all'immagine di texture sia alla mappa di profondità.
- **Fotopletismogramma remoto (rPPG)**: misura volumetrica di un organo — con video ad alta risoluzione si rilevano i micro-cambiamenti volumetrici dovuti al flusso sanguigno sotto la pelle (assenti sotto una maschera). Si può apprendere una **mappa di confidenza** basata sulla forza del segnale del battito per pesare la correlazione locale rPPG (sistema **FaceReader**, usato anche per diagnosi remota e stima delle emozioni).

### 15.8 Dataset per la ricerca in anti-spoofing

Variazioni tipiche negli attacchi con foto: spostamento orizzontale/verticale/avanti-indietro, rotazione in profondità (asse verticale e orizzontale), piegatura verso l'interno/esterno (asse verticale e orizzontale).

### 15.9 Valutazione dell'Anti-Spoofing

- **Spoof False Acceptance Rate (SFAR)**: numero di attacchi di spoofing accettati per errore.
- Distinzione fondamentale: nello **Scenario Licit**, il False Acceptance deriva solo da **zero-effort attack** (l'impostore dichiara un'identità senza sforzarsi di somigliare al genuino) — è un problema del classificatore di riconoscimento. Nello **Scenario Spoof**, si aggiungono anche i tentativi **volontari** di riprodurre l'aspetto della persona attaccata → si introduce il **SFAR cumulativo**, da confrontare con la FRR.
- La **False Rejection**, in ambito spoof, può riferirsi non solo al mancato riconoscimento ma anche alla **misclassificazione di un campione genuino come spoof**.
- **False Living Rate (FLR)** e **False Fake Rate (FFR)**: sostituiscono concettualmente FRR e SFAR nella valutazione della contromisura. FLR = percentuale di attacchi di spoofing misclassificati come reali; FFR = percentuale di accessi reali misclassificati come falsi.
- **Half Total Error Rate (HTER)**: media di FAR e FRR (nello scenario Licit), calcolata per ogni soglia; si cerca il punto operativo **EER**.
- **Fusione recognition + anti-spoofing**: possono essere combinati in un'unica risposta Accetta/Rifiuta, unendo un classificatore biometrico con un classificatore binario (anti-spoofing).

---

## 16. FACE ANTISPOOFING @ BIPlab — Invarianti Geometrici e Cross-Ratio

Uno dei metodi più semplici, efficienti e accettabili per la Liveness Detection è la **richiesta di interazione** (challenge) all'utente in un momento preciso, per osservare la reazione dell'aspetto del probe. I sistemi più robusti si basano su due attività: **verifica della tridimensionalità del volto** e **interazione con l'utente** (parametrizzata da tempo e tipo di movimento).

> Richiedere un movimento a **tempo casuale** è sufficiente per evitare un attacco con video preregistrato (replay). Un challenge-response con movimento fisso e prevedibile (es. "gira la testa da sinistra a destra") può però essere spoofato con un video che riproduce esattamente quel movimento — per questo serve un **tipo di movimento specifico a tempo casuale**, che richiede un modello 3D per tracciarlo e distinguerlo da una foto opportunamente presentata. La difesa anti-spoofing più forte comporta un aumento significativo della complessità del sistema.

### 16.1 Invarianti Geometrici (Geometric Invariants)

Descrittori di forma **non influenzati** da posa, scala, proiezione prospettica e parametri intrinseci della camera. Espressi come rapporti di distanze/misure o combinazioni di coordinate 3D/2D.

- **Invarianti su punti complanari**: quando i punti coplanari cambiano orientamento rispetto al dispositivo di cattura, il rapporto rimane **costante** — è una "firma" dell'oggetto, utile per inferirne la forma.
- **Invarianti su punti collineari**: punti sulla stessa retta; il rapporto resta costante indipendentemente dall'orientamento della retta rispetto al dispositivo.

Tutti calcolabili da una **singola vista** (2D/3D invariants), usati come descrizione grezza preliminare della forma dell'oggetto (volto).

### 16.2 Ragionamento in "reverse" (Cross-Ratio)

Dato un insieme di punti noti per **non essere coplanari** su un vero volto 3D, si calcola su più immagini consecutive un invariante geometrico che **richiederebbe** la coplanarità. Se la posa del soggetto cambia ma il **cross-ratio calcolato rimane costante**, allora i punti da cui è calcolato **devono essere coplanari** — cosa impossibile per un vero volto 3D. Quindi l'oggetto **non è 3D** (è una foto).

Candidati ideali per il volto: **centro degli occhi, punta del naso, mento** — punti che in un vero modello 3D violano fortemente collinearità/coplanarità, ma le soddisfano rigidamente in una rappresentazione 2D (foto) dello stesso modello.

**Parametri del sistema:**
- La variazione $v$ del cross-ratio $c$ è calcolata sulle ultime $K$ frame (**observation window**) e confrontata con una soglia predeterminata $th$ (diversa per ciascun cross-ratio).
- Il numero di frame classificate come genuine ($v > th$) deve superare un'ulteriore soglia $th_v$, impostata secondo il livello di sicurezza richiesto.
- Frame con errori di localizzazione (volto non trovato, punti determinati in modo scorretto) vengono **scartate** e non entrano nella observation window.
- $K$ (numero di frame considerate) è un parametro cruciale per le prestazioni del sistema.

Per aggiungere l'interazione orientata all'anti-spoofing, il sistema richiede di muovere il volto solo in **intervalli temporali ben definiti**, durante i quali viene verificato il cross-ratio.

### 16.3 FATCHA: Face CAPTCHA

**CAPTCHA** ("Completely Automated Public Turing test to tell Computers and Humans Apart") impedisce ai bot di abusare di certi servizi, richiedendo un compito banale per un umano ma difficile per un sistema automatico.

- **CAPTCHA testuali**: problemi di usabilità (a volte difficili anche per l'utente umano).
- **CAPTCHA basati su immagini**: richiedono sistemi automatici più sofisticati, ma pongono problemi di **accessibilità** (persone non vedenti o ipovedenti non possono risolverli).

**FATCHA** propone un approccio diverso: richiede un **gesto specifico ma molto semplice** all'utente (facile anche da rilevare per il sistema — meglio una combinazione casuale di gesti semplici). Non c'è alcun compito percettivo o cognitivo: si chiede all'utente di **produrre** un'azione, non di **analizzare** qualcosa. Il volto dell'utente stesso è il CAPTCHA.

---

## 17. Riepilogo — mappa concettuale delle formule chiave

$$
\boxed{II(x,y) = \sum_{x'\le x,\,y'\le y} I(x',y')} \qquad \text{(Integral Image)}
$$

$$
\boxed{H_M(x) = \frac{\sum_{i=1}^M \alpha_i h_i(x)}{\sum_{i=1}^M \alpha_i}} \qquad \text{(AdaBoost strong classifier)}
$$

$$
\boxed{C = \frac{1}{m}\sum_{i=1}^m (x_i-\bar{x})(x_i-\bar{x})^T} \qquad \text{(PCA covariance)}
$$

$$
\boxed{J(w) = \frac{w^T S_B w}{w^T S_W w}} \qquad \text{(Fisher criterion, LDA)}
$$

$$
\boxed{(S_B - \lambda_i S_W)\,w_i^* = 0} \qquad \text{(LDA generalized eigenvalue problem)}
$$

$$
\boxed{t^* = \arg\min_{t\in G}[\sigma_w^2(t)]} \qquad \text{(Otsu thresholding)}
$$

**Tabella di sintesi — famiglie di metodi 2D:**

| Famiglia | Esempi | Robustezza PIE | Velocità matching | Retraining necessario |
|---|---|---|---|---|
| Globale/Olistico | PCA, LDA, ICA, NN | Bassa (PCA) / Media (LDA) | Alta | Sì, se cambia molto la popolazione |
| Locale/Feature-based | LBP, EBGM | Media-Alta | Alta (LBP) / Bassa (EBGM, NP-hard) | No (EBGM) |
| Grafo | EBGM, sistemi a grafo | Alta (posa/luce) | Bassa | No |
| Termico | Termogramma | Alta (luce/tempo) | — | — |

**Tabella di sintesi — anti-spoofing per tipo di attacco:**

| Attacco | Difficoltà attaccante | Contromisure principali |
|---|---|---|
| Print (foto) | Bassa | Eye blink, micro-texture (LBP+SVM), captured-recaptured (DoG+LBPV), gaze stability, optical flow, IDA |
| Replay (video) | Media | Multi-scale LBP + DSIFT (rilevamento moiré pattern) |
| 3D Mask | Alta | Analisi albedo/riflettanza, LBP su texture+depth, rPPG |

