# Multibiometric Systems — Riassunto Completo

---

## 1. Motivazione: come migliorare la robustezza di un sistema biometrico

Nel processing biometrico esistono numerosi limiti intrinseci (similarità inter-personale, differenze intra-personali, ecc.) e la possibilità concreta di **spoofing** su alcuni tratti — il **volto**, ad esempio, è un tratto biometrico estremamente facile da falsificare.

Sono necessarie contromisure, ma l'approccio generalmente più efficace per migliorare l'accuratezza di un sistema biometrico è sfruttare un **Sistema Multibiometrico**.

> **Multibiometric Systems**, detti anche **Ensemble of Classifiers**, sono una delle soluzioni proposte per migliorare le prestazioni di un sistema biometrico, indipendentemente da quale sia la limitazione da superare.

**Perché i sistemi single-biometry sono deboli?**
- La maggior parte dei sistemi attuali si basa su un'**unica biometria** → risultano vulnerabili a possibili attacchi e poco robusti a diversi problemi.
- Usando **più tratti biometrici** insieme, diventa molto più difficile per un attaccante effettuare uno spoofing efficace, poiché dovrebbe falsificare un campione per **ciascuna** delle biometrie coinvolte nell'approccio multibiometrico.

**Vantaggi di un sistema multimodale:**
- Le carenze dei sistemi a singolo tratto vengono **controbilanciate** grazie alla disponibilità di più biometrie.
- Possibilità di **limitare vincoli** relativi a un singolo tratto e ottenere prestazioni migliori nelle situazioni in cui un tratto isolato fornirebbe bassa accuratezza.

### 1.1 Il concetto di ortogonalità tra tratti

Si parla di tratti che devono avere più o meno lo stesso **potere di riconoscimento**, ma soprattutto che devono essere tra loro **ortogonali**: se, ad esempio, si considera l'intero volto e solo la regione perioculare, tra le due immagini esiste una **forte correlazione** — non sono quindi tratti indipendenti/ortogonali, e la loro combinazione porta benefici limitati.

> ⚠️ **Attenzione — combinare un tratto forte con uno debole:** quando il sistema include un tratto biometrico molto forte (iride, impronte digitali, vasi sanguigni, retina), può sembrare inutile aggiungere ad esempio il volto, poiché i primi sono già molto robusti da soli. È infatti possibile riscontrare che l'unione di un tratto molto forte con uno più debole porti a un **decremento delle prestazioni** complessive. Questo problema può essere risolto con approcci intelligenti, come l'assegnazione di un **peso maggiore** alla biometria più forte.

### 1.2 Architettura generale

Nei sistemi multibiometrici sono presenti **più dispositivi di acquisizione**, tipicamente uno per ciascun tratto biometrico coinvolto. Ogni tratto può essere processato individualmente e, a un certo punto della pipeline, i risultati ottenuti vengono **fusi** (*fusione*, vedi §3).

---

## 2. Tassonomia dei sistemi multibiometrici (sorgenti di informazione)

Esistono diversi modi per combinare le risposte biometriche, a seconda della **sorgente** delle informazioni combinate:

### 2.1 Multimodal Approach (approccio puro)

Il modo più ovvio: si combinano **tratti biometrici multipli e diversi** (es. volto + impronta + iride). Questo è il **puro approccio Multimodale**.

### 2.2 Multiple Instances (istanze multiple)

Si sfruttano **istanze diverse dello stesso tratto** (es. due impronte digitali provenienti da **due dita diverse** della stessa persona — indice e medio). Non è sempre possibile applicarlo (dipende dal tratto).

### 2.3 Repeated Instances (istanze ripetute)

Si acquisiscono **due campioni della stessa identica istanza** del tratto (es. due impronte del **medesimo dito**, catturate due volte).

> **Differenza chiave Multiple vs Repeated Instances:**
> - **Multiple Instances** → due impronte di **due dita diverse** (es. indice e medio).
> - **Repeated Instances** → due catture del **medesimo dito** (es. indice, 2 volte).
>
> L'idea alla base delle Repeated Instances è che **pressione diversa** o problemi occasionali di acquisizione possano generare differenze tra i campioni ottenuti dallo stesso tratto/istanza.

### 2.4 Multiple Algorithms

Approccio puro di **Ensemble of Classifiers**: più algoritmi diversi partecipano alla decisione finale. Esempio: si applicano **LBP** e **Wavelet** per processare la stessa immagine di volto; ciascun algoritmo di matching è adatto al proprio metodo di estrazione feature; si fondono infine i risultati dei due algoritmi.

### 2.5 Multiple Sensors

Lo stesso tratto (o, in generale, tratti diversi ma tipicamente lo stesso) viene catturato da **sensori diversi**. Esempio: si usano due sensori differenti per acquisire la stessa impronta digitale (entrambi ottici, entrambi capacitivi, oppure uno ottico e uno capacitivo), e si fondono poi i risultati.

**Tabella riassuntiva delle 5 sorgenti multibiometriche:**

| Approccio | Tratto | Istanza | Sensore | Algoritmo |
|---|---|---|---|---|
| **Multimodal** | Diversi | — | — | — |
| **Multiple Instances** | Stesso | Diverse (es. dita diverse) | — | — |
| **Repeated Instances** | Stesso | Stessa (ripetuta) | Stesso | — |
| **Multiple Algorithms** | Stesso | Stessa | Stesso | Diversi |
| **Multiple Sensors** | Stesso (in genere) | Stessa | Diversi | — |

---

## 3. Livelli di fusione (Fusion Levels)

Una volta disponibili più campioni (per lo stesso tratto, per tratti diversi, o per diverse istanze dello stesso tratto), sorge la domanda: **come fondere i risultati per ottenere la risposta finale?**

La combinazione delle diverse biometrie può essere effettuata in ciascuno dei **quattro moduli** del sistema: modulo di **acquisizione** (capture), modulo di **estrazione feature**, modulo di **riconoscimento** (matching), modulo di **decisione**.

Si distinguono quindi (in ordine di "precocità" della fusione lungo la pipeline):

$$
\text{Sensor Level} \;\rightarrow\; \text{Feature Level} \;\rightarrow\; \text{Score Level} \;\rightarrow\; \text{Decision Level}
$$

### 3.1 Sensor Level Fusion

È una fusione **molto precoce**: avviene **prima** dell'estrazione delle feature. Esempio tipico: fondere le informazioni provenienti da **tre immagini 2D** in un unico **modello 3D** — si può considerare un caso di Sensor Level Fusion. Le feature usate per matching e decisione vengono quindi estratte **direttamente dal modello fuso**.

> **Difficoltà:** la Sensor Level Fusion è **difficile da realizzare**, perché richiede innanzitutto lo **stesso tratto** (in alcuni casi è possibile fondere anche tratti diversi) e soprattutto richiede che il **tipo di segnale** estratto sia **omogeneo/dello stesso tipo**.

### 3.2 Feature Level Fusion

$$
\text{Sensore 1} \rightarrow \text{Feature Extraction} \rightarrow \text{Feature Vector 1} \quad \searrow
$$
$$
\text{Sensore 2} \rightarrow \text{Feature Extraction} \rightarrow \text{Feature Vector 2} \quad \nearrow \quad \text{Fusion} \rightarrow \text{Combined Feature Vector} \rightarrow \text{Matching} \rightarrow \text{Score} \rightarrow \text{Decision}
$$

Si acquisiscono campioni diversi (uno per ciascun tratto/sorgente) e si fondono i **vettori di feature** già estratti. Esempio: si estraggono separatamente i due **iris code** dall'occhio destro e sinistro, e si fondono in un **unico vettore**, che viene poi confrontato tramite matching, producendo la distanza su cui si basa la decisione finale.

Analogamente, è possibile estrarre istogrammi **LBP** dal volto e fonderli con gli istogrammi LBP delle due iridi, applicando poi una strategia di **histogram matching** che tratta l'istogramma finale come se fosse un unico istogramma.

**Vantaggio atteso:** risultati migliori, poiché molta più informazione è ancora presente rispetto a un livello di fusione successivo (es. score level), dato che si fondono i vettori di feature completi, non solo i punteggi finali.

**Problemi principali:**

| Problema | Descrizione |
|---|---|
| **Compatibilità** | Se una strategia di estrazione restituisce un istogramma e un'altra un insieme di coefficienti wavelet, la Feature Level Fusion **non è fattibile** |
| **Curse of dimensionality** | Molte feature fuse insieme possono creare uno spazio troppo **sparso** → servono strategie di *feature selection* o di *dimensionality reduction*; può inoltre essere richiesto un **matcher più complesso** |
| **Dati rumorosi/ridondanti** | I vettori combinati possono includere dati noisy e/o ridondanti |
| **Inflessibilità** | Il matcher viene addestrato usando la nozione (struttura) del **vettore fuso**: aggiungere un nuovo sotto-probe, o sostituirne uno, richiede di **cambiare il classificatore finale** o almeno parte dei suoi modelli |

**Strategie di Feature Level Fusion:**
- **Linking (concatenazione) semplice** dei vettori di feature.
- **Fusione parallela** (*Parallel*): limitata a **due** vettori di feature biometrici. Il vettore combinato viene trattato come un **vettore complesso**: la feature di un tratto costituisce la **parte reale**, l'altra la **parte immaginaria**.
$$
\mathbf{v}_{\text{combined}} = \mathbf{v}_1 + i\,\mathbf{v}_2
$$
- **Canonical Correlation Analysis (CCA)**: individua una coppia di **trasformazioni lineari** tali da **massimizzare il coefficiente di correlazione** tra le caratteristiche. Prima dell'applicazione, i vettori vengono ridotti in dimensione, poiché la CCA soffre del problema della **"small sample size"**.

### 3.3 Score Level Fusion (approccio più popolare e flessibile)

$$
\text{Sensore 1} \rightarrow \text{FE} \rightarrow \text{Feature Vector 1} \rightarrow \text{Matching (vs Template)} \rightarrow \text{Score 1} \quad \searrow
$$
$$
\text{Sensore 2} \rightarrow \text{FE} \rightarrow \text{Feature Vector 2} \rightarrow \text{Matching (vs Template)} \rightarrow \text{Score 2} \quad \nearrow \quad \text{Fusion} \rightarrow \text{Total Score} \rightarrow \text{Decision}
$$

Ogni tratto viene processato **da un sottosistema separato**: estrazione feature separata, matching separato, e la **fusione avviene dopo il matching**, sui punteggi (*score*) restituiti dai diversi algoritmi.

> Con questo approccio, il problema si sposta interamente sul trovare una **buona strategia di fusione**.

**Due approcci principali:**

1. **Transformation-based**: gli score dei diversi matcher vengono prima **normalizzati** (trasformati) in un dominio comune, e poi combinati tramite regole di fusione.
   - **Sensibile agli outlier** (valori insolitamente alti o bassi dovuti a errori di misura o condizioni particolari).
   - Può essere influenzato da una **scarsa conoscenza** dell'effettivo minimo e massimo ottenibili da una certa misura.

2. **Classifier-based**: gli score dei diversi classificatori vengono considerati come **feature** e inclusi in un nuovo **vettore di feature**. Un **classificatore binario** (es. **Reti Neurali**, **SVM**) viene addestrato per discriminare tra vettori di score genuini e impostori — invece di fondere direttamente gli score, li si raccoglie in un nuovo vettore su cui si addestra il classificatore.
   - Processo **complesso**; si perde un certo grado di **flessibilità**: aggiungere, eliminare o modificare un classificatore richiede di **cambiare il classificatore del vettore di score**.

#### 3.3.1 Fusion Rules per la Score Level Fusion

**a) ABSTRACT (livello astratto)**

Il classificatore restituisce semplicemente un'**etichetta di classe** al pattern in input: non è né un rank né uno score, ma una singola **class label** che rappresenta la risposta del classificatore a una determinata operazione di matching.

- **Verifica**: il sistema restituisce o un'etichetta "*unknown*" (persona non riconosciuta) oppure l'etichetta con l'**identità dichiarata**.
- **Identificazione**: ogni classificatore può riconoscere il probe come un'**identità diversa**.

**Majority Voting**: ciascun classificatore vota per una classe; il pattern viene assegnato alla **classe più votata**. L'affidabilità del multi-classificatore si calcola mediando le singole confidenze.

**b) RANK (livello di ranking)**

Il sistema non restituisce una singola etichetta, ma una **classifica (ranking) di candidati**: ogni classificatore produce un proprio ranking delle classi in base alla probabilità che il pattern appartenga a ciascuna di esse (più alta la probabilità, più alta la posizione nella lista).

I ranking vengono convertiti in **punteggi** che vengono poi **sommati**; la classe con il punteggio finale più alto è quella scelta dal multi-classificatore. Questa regola è nota come **Borda count**: ogni classificatore produce una classifica delle classi secondo la probabilità che il pattern vi appartenga, i ranking vengono convertiti in punteggi e sommati, e la classe con il punteggio finale più alto è quella scelta.

**Esempio numerico (3 classificatori C1, C2, C3; 4 classi a, b, c, d):**

$$
\begin{array}{c|ccc}
\text{Rank} & C1 & C2 & C3 \\\hline
1 & c & b & b \\
2 & b & d & a \\
3 & d & c & c \\
4 & a & a & d
\end{array}
$$

I punteggi di rank (posizione invertita: 1° posto = 4 punti, 4° posto = 1 punto) si sommano per ciascuna classe:

$$
r_a = r_a^{(1)} + r_a^{(2)} + r_a^{(3)} = 1 + 4 + 3 = 8
$$
$$
\boxed{r_b = r_b^{(1)} + r_b^{(2)} + r_b^{(3)} = 3 + 3 + 4 = 10}
$$
$$
r_c = r_c^{(1)} + r_c^{(2)} + r_c^{(3)} = 4 + 1 + 2 = 7
$$
$$
r_d = r_d^{(1)} + r_d^{(2)} + r_d^{(3)} = 2 + 2 + 1 = 5
$$

La classe scelta è quella con **punteggio totale massimo**: in questo esempio, $b$ (con $r_b = 10$).

**c) MEASUREMENT (livello di misura)**

Ogni classificatore restituisce il proprio **punteggio di classificazione** per il pattern, confrontato con ciascuna classe.

L'unico problema da risolvere è portare i valori di misura nello **stesso intervallo** (range), applicando quindi una qualche forma di **normalizzazione** (vedi §3.3.2).

Un metodo banale consiste semplicemente nel **sommare** gli score e passare il risultato alla regola di fusione. L'unico punto che limita la piena flessibilità di questo approccio è il possibile cambiamento nel tipo di **soglia** usata per ottenere il risultato finale.

**Formalizzazione (schema generale):** dati $N$ classificatori, ciascuno produce un vettore di punteggi $\left(p_1^{(n)}, \dots, p_k^{(n)}\right)$ per le $k$ classi; la **regola di fusione** aggrega i punteggi corrispondenti in un unico vettore $(p_1, \dots, p_k)$, e la decisione finale è:

$$
\text{classe scelta} = \arg\max_i\ p_i
$$

Metodi di fusione possibili: **sum** (somma), **weighted sum** (somma pesata), **mean** (media), **product** (prodotto), **weighted product** (prodotto pesato), **max**, **min**, ecc.

#### 3.3.2 Normalizzazione degli score

Gli score prodotti da matcher diversi sono tipicamente **non omogenei**:
- Possono essere **similarità** o **distanze** (semantica opposta);
- Possono avere **range diversi** (es. $[0,1]$ oppure $[0,100]$);
- Possono avere **distribuzioni diverse**.

Per supportare una fusione a livello di score coerente, si applicano **trasformazioni di normalizzazione**, prestando particolare attenzione ai valori che ricadono nella **regione di sovrapposizione** tra distribuzione genuina e impostore.

**Criteri di scelta di un metodo di normalizzazione:**
- **Robustezza (Robustness)**: non deve essere influenzato dagli outlier.
- **Efficacia (Effectiveness)**: la distribuzione degli score normalizzati deve presentare la **stessa forma** e lo **stesso andamento** della distribuzione degli score originali.

**Le funzioni di normalizzazione standard**, in ordine crescente di quanto "comprimono" aggressivamente i valori (dato lo score originale $s_k$):

| Funzione | Formula | Note |
|---|---|---|
| **Min-Max** | $s_k' = \dfrac{s_k - min}{max - min}$ | Mappa (shift + compressione/dilatazione) l'intervallo tra minimo e massimo in $[0,1]$; assume che minimo e massimo generati dal modulo di matching siano noti. |
| **Z-score** | $s_k' = \dfrac{s_k - \mu}{\sigma}$ | Usa media aritmetica $\mu$ e deviazione standard $\sigma$ degli score del singolo sottosistema; è la più diffusa, ma **non garantisce** un intervallo comune per gli score normalizzati provenienti da sottosistemi diversi. |
| **Median/MAD** | $s_k' = \dfrac{s_k - \text{median}}{MAD}$ | Usa la mediana e la *Median Absolute Deviation*; poco efficace, specialmente con distribuzioni **non Gaussiane** — in tal caso non preserva né la forma della distribuzione originale né trasforma i valori in un intervallo comune. |
| **Sigmoid** | $s_k' = \dfrac{1}{1+c\,e^{-k s_k}}$ | Codominio nell'intervallo aperto $(0,1)$; introduce una distorsione eccessiva quando $x$ tende agli estremi dell'intervallo; la forma dipende fortemente dai due parametri $c$ e $k$, che a loro volta dipendono dal dominio di $x$. |
| **Tanh** | $s_k' = \dfrac{1}{2}\left[\tanh\left(0.01\dfrac{s_k - E[s_k]}{\sigma(s_k)}\right)+1\right]$ | Garantisce la proiezione nell'intervallo aperto $(0,1)$; concentra eccessivamente i valori attorno al centro dell'intervallo (0.5). |

> Esistono diversi metodi derivati per questo scopo: ad esempio, la **reliability** (affidabilità) dei diversi sistemi può essere usata come **fattore di peso** nella fusione. Una possibile soluzione per stimare l'affidabilità è rappresentata dai **confidence margins** (margini di confidenza) — tra i più diffusi (Poh e Bengio, 2004): $M(\Delta) = |FAR(\Delta) - FRR(\Delta)|$.

### 3.4 Decision Level Fusion

$$
\text{Sensore 1} \rightarrow \text{FE} \rightarrow \text{Matching} \rightarrow \text{Score 1} \rightarrow \text{Decision 1 (Yes/No)} \quad \searrow
$$
$$
\text{Sensore 2} \rightarrow \text{FE} \rightarrow \text{Matching} \rightarrow \text{Score 2} \rightarrow \text{Decision 2 (Yes/No)} \quad \nearrow \quad \text{Fusion} \rightarrow \text{Decisione finale}
$$

A questo livello, **ogni sistema ha già fornito una decisione**. Ogni classificatore restituisce il proprio esito (accetta/rifiuta per la verifica, oppure un'identità per l'identificazione). La decisione finale si ottiene combinando le singole decisioni tramite una **regola di fusione**.

**Strategie di combinazione più semplici (combinazione logica):**

| Regola | Descrizione |
|---|---|
| **Serial combination (AND)** | L'autenticazione globale richiede **tutte** le decisioni positive |
| **Parallel combination (OR)** | L'utente può essere autenticato anche da **una sola** modalità biometrica |

Un'ulteriore importante regola di fusione a livello decisionale è il **Majority Voting**.

**Co-Update Method**: in base ai risultati dei sistemi multibiometrici, i template "**highly genuine**" (altamente affidabili) classificati come tali da un sistema possono essere **aggiunti alla gallery** insieme al campione corrispondente all'altro tratto.

> **Beneficio del Co-Update**: se gli identificatori sono **complementari**, si aiutano a vicenda nell'identificare pattern "difficili", catturando le variazioni intra-classe nei dati di input **senza abbassare** la soglia di accettazione.

---

## 4. Tabella riassuntiva dei livelli di fusione

| Livello | Momento della fusione | Cosa si fonde | Vantaggio | Svantaggio principale |
|---|---|---|---|---|
| **Sensor Level** | Prima della feature extraction | Segnali grezzi (es. 2D → 3D) | Massima informazione disponibile | Difficile: richiede stesso tratto e segnale omogeneo |
| **Feature Level** | Dopo la feature extraction | Vettori di feature | Molta informazione ancora presente | Compatibilità, curse of dimensionality, inflessibilità |
| **Score Level** | Dopo il matching | Punteggi di similarità/distanza | Il più popolare e flessibile | Necessità di normalizzazione robusta |
| **Decision Level** | Dopo la decisione finale | Etichette accetta/rifiuta o identità | Il più semplice da progettare/implementare | Perdita di tutta l'informazione utile a valutare il comportamento del sistema in caso di errore |

---

## 5. Classificazione per posizione rispetto al matching

Gli approcci possono essere classificati anche in base a **quando** avviene la combinazione rispetto all'operazione di matching:

$$
\textbf{Prima del matching (Pre-matching):}\quad \text{Sensor Level} \ \lor\ \text{Feature Level}
$$
$$
\textbf{Dopo il matching (Post-matching):}\quad \text{Dynamic Selection of Features} \ \lor\ \text{Pure Classifier Fusion}
$$

- **Dynamic Selection of Features**: invece di utilizzare **tutte** le risposte di tutti i matcher, si seleziona solo un **sottoinsieme** dei risultati, in base a un criterio di valutazione della qualità stabilito in precedenza.
- **Pure Classifier Fusion**: può essere effettuata a **Score Level** o a **Decision Level** (vedi §3.3 e §3.4). A sua volta, la Score Level Fusion può essere: **Abstract**, basata su **Rank**, oppure basata su **Measurement** (vedi §3.3.1).

---

## 6. Aspetti critici dei sistemi multibiometrici

I principali **aspetti critici** (*Critical Aspects*) da affrontare nella progettazione di un sistema multibiometrico sono:

1. La **normalizzazione** degli score (necessaria per rendere confrontabili misure eterogenee).
2. La decisione su **quali sistemi siano più affidabili** di altri, in modo che le loro risposte vengano considerate con un **peso maggiore** nella fusione finale.

---

## 7. Riepilogo — mappa concettuale

$$
\boxed{\text{5 sorgenti: Multimodal, Multiple Instances, Repeated Instances, Multiple Algorithms, Multiple Sensors}}
$$

$$
\boxed{\text{4 livelli di fusione: Sensor} \to \text{Feature} \to \text{Score} \to \text{Decision}}
$$

$$
\boxed{\text{Score Level Fusion Rules: Abstract (Majority Voting)} \;\vert\; \text{Rank (somma dei rank)} \;\vert\; \text{Measurement (sum, weighted sum, mean, product, max, min...)}}
$$

$$
\boxed{\text{Decision Level Rules: AND (serial)} \;\vert\; \text{OR (parallel)} \;\vert\; \text{Majority Voting} \;\vert\; \text{Co-Update}}
$$

**Schema riassuntivo — dove si fondono le informazioni:**

| Modulo del sistema | Tipo di fusione associata |
|---|---|
| Capture (acquisizione) | Sensor Level |
| Feature Extraction | Feature Level |
| Recognition (matching) | Score Level |
| Decision | Decision Level |
