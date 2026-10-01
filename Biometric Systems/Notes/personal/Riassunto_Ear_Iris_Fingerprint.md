# Ear, Iris e Fingerprint Recognition — Riassunto Completo

---

## 1. Ear Recognition (Riconoscimento dell'orecchio)

### 1.1 Caratteristiche generali

A differenza del volto, l'orecchio ha pochi aggettivi descrittivi disponibili:
- **Forma:** ovale, rotonda, ...
- **Aspetto:** delicato, paffuto, ...
- (Il volto invece dispone di molte più feature descrittive: colore degli occhi, forma degli zigomi, forma di naso e mento, spessore delle labbra, ...)

**Problemi principali:**
- **Occlusione**, dovuta alla presenza dei capelli.
- Le orecchie sono **strutture intrinsecamente 3D**: un cambio di orientazione richiede una procedura di normalizzazione per correggere la deformazione prospettica.

### 1.2 Anatomia (struttura non casuale, ben definita)

- **Helix**: bordo esterno
- **Anti-helix**: protrusione che corre parallela e interna all'helix
- **Lobo**
- **Intertragic notch** (tacca intratragica): incavo a forma di U tra l'ingresso del canale uditivo (meato) e il lobo
- Altri landmark: *crus of helix*, *tragus*, *concha*, *antitragus*

### 1.3 Classificazione biometrica

L'orecchio è una biometria **passiva** e **statica**. Essendo associato a uno dei sensi (l'udito), è tipicamente lasciato scoperto per non ostacolare la percezione uditiva; se poco visibile, è comunque possibile richiedere un'interazione esplicita all'utente.

**Vantaggi rispetto al volto:**
- Meno dettagli → richiede una risoluzione inferiore
- Distribuzione di colore più uniforme
- Minore (se non nulla) sensibilità alle variazioni di espressione

**Svantaggi:**
- La struttura 3D la rende **sensibile a illuminazione e variazioni di posa**
- Le piccole dimensioni sono una "medaglia a doppia faccia" (facilitano l'acquisizione ma limitano il contenuto informativo)

> L'uso di questa biometria è tuttora relativamente limitato all'ambito forense (scena del crimine, es. analisi di impronte auricolari latenti — *latent ear print analysis*).

### 1.4 Classificazione di Iannarelli

**Iannarelli** dimostrò che, prendendo un certo numero di landmark sull'orecchio e misurando le distanze tra essi, si ottiene un **vettore di feature** estremamente discriminativo, specialmente in condizioni di acquisizione ottimali.

Iannarelli classificò l'orecchio in 4 categorie morfologiche:
$$
\text{Round},\ \text{Rectangle},\ \text{Triangle},\ \text{Oval}
$$

Questa è una prima classificazione utile a **ridurre lo spazio di ricerca** (e quindi il tempo computazionale) nel confronto tra template, ma non è mai sufficiente da sola per un riconoscimento definitivo.

### 1.5 Problemi di localizzazione ed estrazione

- **Localizzazione difficile**: è complicato separare (segmentare) l'orecchio dalla pelle circostante, poiché ha lo stesso colore della cute adiacente, pur avendo dimensioni e complessità inferiori rispetto al volto.

**Approcci principali all'estrazione dell'orecchio (ear detection):**

1. **Localizzazione di punti di interesse** tramite **Reti Neurali**
   - Si acquisisce una grande quantità di immagini di training.
   - Le immagini vengono ridotte di risoluzione, poiché il tempo di training è proporzionale alla dimensione dell'immagine in input e in questa fase i dettagli fini non sono prioritari.
   - Il contrasto di ogni immagine viene normalizzato per migliorare i risultati.
   - I punti selezionati definiscono un **sistema di riferimento** sull'immagine: tutti gli altri punti sono considerati rispetto ad essi (come su una mappa), permettendo di estrarre il rettangolo che racchiude l'orecchio e di normalizzarne la dimensione ai fini del matching.

2. **Object Detection** (classificatori addestrati a rilevare orecchie)
   - Si intende la localizzazione (posizione) di un oggetto appartenente a una determinata classe all'interno dell'immagine.
   - Algoritmo tipico: **AdaBoost**.
     - Training basato su: **campioni positivi** (istanze multiple della classe di interesse) e **campioni negativi** (immagini che non contengono l'oggetto).
     - L'unione dei due insiemi costituisce il training set; durante l'addestramento vengono estratte e selezionate le feature più discriminative.
     - È possibile aggiungere nuovi campioni positivi o negativi e ripetere il training se, rispettivamente, il tasso di oggetti mancati (*missed rate*) o il tasso di falsi allarmi (*false alarm rate*) crescono eccessivamente.

3. **Metodi geometrici 3D**
   - Grazie alla sua tridimensionalità intrinseca, l'orecchio è ben catturato da sensori di profondità (*range sensor*), consentendo approcci **3D o 2.5D**.
   - Il training viene svolto **offline**: partendo da un modello 3D del profilo del volto si identificano i punti di massima curvatura, si crea un'immagine binaria (0 nero, 1 bianco) e la regione corrispondente all'orecchio viene estratta manualmente.
   - **Chen e Bhanu (2004)**: costruiscono un template come istogramma medio dello *shape index* (basato sulle curvature principali $k_1, k_2$ di ogni punto $p$) calcolato sulle immagini di training; in fase di test (online) si calcola l'immagine binaria del nuovo modello, si individuano i punti di curvatura massima/minima e si cercano le regioni corrispondenti al template.

### 1.6 Studi storici sull'orecchio come biometria

- **Alfred Iannarelli (1989)** conduce due studi fondativi:
  - Il primo, su **10.000 campioni casuali** raccolti in California, dimostra che l'orecchio ha una variabilità sufficiente a distinguere due soggetti diversi e rispetta le proprietà fondamentali di un tratto biometrico (universalità, unicità, permanenza, collectability).
  - Il secondo, su **fratelli e gemelli identici**, verifica che anche questi soggetti presentano feature auricolari diverse.
  - Sulla **permanenza**: i cambiamenti più significativi riguardano il lobo, che si allunga (per gravità) tra i 4 mesi e gli 8 anni, resta stabile dagli 8 ai 70 anni, per poi allungarsi nuovamente per il rilassamento dei tessuti.
- **Imhofer (1906)** osserva che, su un campione di 500 orecchie, **4 feature sono sufficienti** a distinguerle.
- La società **La Bromba GmbH** ha confrontato diverse biometrie rispetto alla loro permanenza nel tempo.

### 1.7 Approcci al riconoscimento dell'orecchio (Ear Recognition)

Gli approcci al riconoscimento si possono classificare come: **2D geometrici/globali** (curve/landmark), **modelli 3D** (profondità/curvatura) e **termogrammi**.

**Approcci globali 2D:**

| Metodo | Idea | Note |
|---|---|---|
| **Sistema di Iannarelli** | La ROI viene normalizzata per dimensione; si identifica il **crus of helix** come origine del sistema di misura; a partire da questo punto si eseguono **12 misurazioni geometriche**; il vettore di feature include genere, etnia e le 12 misure. | Estremamente sensibile alla corretta identificazione del punto centrale: se sbagliato, tutte le misure risultano errate. |
| **Diagrammi di Voronoi** (Burge & Burger) | Gli edge vengono rilevati con l'operatore di **Canny** e uniti (edge relaxation) in segmenti di curva più ampi; si costruisce un **grafo di vicinato di Voronoi** delle curve (un diagramma di Voronoi partiziona il piano in regioni/celle in base alla vicinanza a punti seme); il matching cerca **isomorfismi di sottografi**, invarianti a trasformazioni affini e a piccoli cambi di forma dovuti all'illuminazione. | Robusto rispetto alla debolezza del singolo punto di Iannarelli, ma la segmentazione resta sensibile a posa/illuminazione; il metodo non è mai stato validato sperimentalmente in modo estensivo. |
| **Force Fields** (Hurley, Nixon & Carter, 2002) | Ogni pixel è trattato come una **particella carica** (0 = neutro, 255 = carica massima) e come **attrattore Gaussiano**, sorgente di un campo di forza sferico che agisce su tutti gli altri pixel in modo direttamente proporzionale all'intensità e inversamente proporzionale al **quadrato della distanza**: $F_i(r_j) = P(r_i)\frac{f_i - r_j}{\|f_i - r_j\|^3}$. Si fissa una serie di punti lungo un'ellisse attorno all'orecchio e si seguono le linee di campo: quando due percorsi si uniscono non possono più dividersi, e le linee convergono in punti detti **sink** (mappa di convergenza). | Robusto rispetto a punti di partenza, risoluzione dell'immagine e rumore. |
| **Gabor Jets** (Watabe et al.) | Un banco di filtri di Gabor orientati viene convoluto con l'immagine, producendo vettori di feature (**Gabor Jets**) usati come feature visiva in ogni punto dell'immagine. Si costruisce un **"ear graph"** i cui nodi sono etichettati con i Jets calcolati sul corpo dell'antihelix e sulle crus superiore/inferiore dell'antihelix; questi Jets sono salvati come grafi in gallery, e la **PCA** produce un "eigenear graph". Il riconoscimento sfrutta la similarità tra i Jets campionati e quelli ricostruiti dalla probe. | Riusa direttamente la stessa macchina dei Gabor Jets/EBGM del volto. |
| **Angle Vectors** (Shailaja & Gupta, 2006) | Dopo l'edge detection si individua la **max-line** (la linea più lunga con entrambi gli estremi sul bordo esterno dell'orecchio); linee normali, perpendicolari alla max-line, la dividono in $(n+1)$ parti uguali. Gli angoli formati dalle intersezioni con il bordo esterno costituiscono il **vettore primario**, quelli con gli altri bordi il **vettore secondario**. Il confronto tra due campioni avviene in modo gerarchico: prima si confronta il vettore primario (esterno), poi quello secondario (interno) come criterio di spareggio. | — |
| **Active Shape Model** (Cootes & Taylor, 1995) | Modelli statistici di forma che si deformano iterativamente per adattarsi a un'istanza dell'oggetto in una nuova immagine. | Usato anche per l'allineamento. |
| **SIFT** (Lowe, 1999) | Trasforma l'immagine in un'ampia collezione di vettori di feature, ciascuno invariante a traslazione, scala e rotazione, parzialmente invariante a cambi di illuminazione e robusto a distorsioni geometriche locali. I punti chiave sono i massimi/minimi della differenza di Gaussiane (DoG) applicata nello spazio delle scale. | Usato anche per l'allineamento. |

**Approcci basati su modelli 3D**: valutano profondità e curvatura di regioni rilevanti dell'orecchio; in alcuni casi il matching avviene tra regioni corrispondenti di due modelli 3D dette **patch**.

**Termogrammi**: l'immagine dell'orecchio è catturata con una camera termica.
- *Vantaggi*: l'orecchio è facilmente localizzabile; robustezza all'occlusione da capelli; il diverso colore facilita la segmentazione.
- *Svantaggi*: sensibilità al movimento, bassa risoluzione, costi elevati.

### 1.8 Sistemi multimodali e confronto orecchio-volto

Diversi sistemi combinano informazioni 2D e 3D per migliorare le prestazioni; la sfida principale è scegliere la migliore **strategia di fusione** (si veda il capitolo sui sistemi multibiometrici).

**Victor, Bowyer e Sarkar (2002)** confrontano il riconoscimento facciale e auricolare usando la **PCA**, con una pipeline in tre fasi:
1. **Pre-processing**: ridimensionamento di ogni immagine a 400×500 pixel;
2. **Normalizzazione**: due punti di riferimento (per volto e per orecchio) più normalizzazione fotometrica;
3. **Identificazione**.

I test furono condotti su **294 soggetti** e **808 immagini** totali, con almeno un'immagine per volto e una per orecchio a testa, in tre condizioni sperimentali: stesso giorno; giorni diversi con stessa espressione; giorni diversi con espressione diversa. **Il volto ha sempre fornito prestazioni migliori dell'orecchio.**

**Chang, Bowyer e Sarkar (2003)** ripetono uno studio simile considerando variazioni di giorno di acquisizione, illuminazione e posa: i risultati sono stati **più contrastanti** rispetto al primo studio.

> La combinazione (fusione) di volto e orecchio, testata su larga scala, ha mostrato che la **combinazione parallela** è generalmente l'opzione migliore — un'anticipazione della logica generale di fusione multibiometrica.

---

## 2. Iris Recognition (Riconoscimento dell'iride)

### 2.1 Anatomia e proprietà

L'**iride** è una membrana muscolare dell'occhio, di colore variabile, con forma e funzione di **diaframma**. È pigmentata, situata posteriormente alla cornea e anteriormente al cristallino, ed è perforata dalla pupilla.

È composta da:
- uno strato piatto di **fibre muscolari** che circondano circolarmente la pupilla;
- un sottile strato di **fibre muscolari lisce** che dilatano la pupilla (regolando la quantità di luce che entra nell'occhio);
- posteriormente, due strati di **cellule epiteliali pigmentate**.

Landmark visibili: *pupillary frill*, *pupillary zone*, *ciliary zone*, *collarette*, *freckles* (efelidi), *crypts* (cripte), *contractile furrow*.

Colore, texture "regolare" (dovuta principalmente ai solchi/*furrows*) e pattern "irregolari" (efelidi e cripte) forniscono un **livello di discriminazione molto alto**, paragonabile a quello delle impronte digitali.

### 2.2 Vantaggi

| Vantaggio | Descrizione |
|---|---|
| **Piccola e protetta** | Superficie ridotta e difficile da danneggiare (salvo incidenti estremi); poco soggetta a occlusione |
| **Visibile ma protetta** | All'aumentare della risoluzione del dispositivo di cattura, aumenta anche la distanza massima di acquisizione utile |
| **Time invariant** | Stabile dopo circa 2 anni di età |
| **Estremamente distintiva** | Occhio destro ≠ occhio sinistro; anche i gemelli hanno iridi diverse |
| **Acquisizione senza contatto** | A differenza della retina, che richiede il contatto con il dispositivo |
| **Doppia modalità** | Acquisibile sia in **infrarosso vicino (NIR)** che in **luce visibile** |
| **Componente randotipica elevata** | Non è influenzata dall'ereditarietà genetica (nessuna familiarità: non si eredita da genitori/parenti) |

### 2.3 Svantaggi e problemi

- **Superficie molto limitata**: solo circa $3.64\ \text{cm}^2$ → serve un dispositivo di cattura di alta qualità, con distanza inferiore a un metro per garantire risoluzione sufficiente.
- **Riflessioni** (dipendenti dalla sorgente luminosa; lenti a contatto e occhiali possono influenzarle).
- Con **iridi molto scure**, la texture può diventare praticamente invisibile alla distanza richiesta.
- **Alta risoluzione richiesta** dall'apparecchiatura.
- **Profondità di campo limitata** → problemi di focus.
- Necessità di allineamento con l'asse ottico, mitigabile con l'**off-axis problem**: se la persona guarda altrove, l'iride non è centrata nella sclera ma spostata verso un lato.
- **Riflessioni speculari**, presenza di occhiali/lenti a contatto.

**Possibili contromisure tecniche:** CCD ad alta risoluzione; ottiche progettate per migliorare la profondità di campo (DOF); autofocus adattivo; sensori/stime di distanza; feedback audio/visivo all'utente; camera dual-eye; dispositivi pan/tilt per diverse altezze e pose; tracking del volto per guidare l'acquisizione; acquisizione infrarossa o near-infrared.

### 2.4 Modalità di cattura: visibile vs infrarosso

| Modalità | Effetto sulla melanina | Colore | Texture |
|---|---|---|---|
| **Luce visibile** | La melanina **assorbe** la luce visibile | Colori ben visibili (strati dell'iride distinguibili) | Immagine con informazione di texture rumorosa |
| **Luce infrarossa (NIR)** | La melanina **riflette** la maggior parte della luce IR | Nessuna informazione di colore | Texture più visibile, ma richiede equipaggiamento speciale |

Nella banda visibile l'iride rivela una texture ricca, casuale e intrecciata, detta **trabecular meshwork** (dovuta essenzialmente alla muscolatura). In illuminazione infrarossa, anche occhi marroni scuri mostrano una texture ricca, difficilmente visibile in luce visibile.

> La presenza di elementi di disturbo (riflessi, ciglia, palpebre) richiede un buon **pre-processing/segmentazione**.

### 2.5 Fasi di elaborazione (pipeline generale)

$$
\text{Acquisizione} \ \rightarrow\ \text{Segmentazione} \ \rightarrow\ \text{Normalizzazione} \ \rightarrow\ \text{Coding} \ \rightarrow\ \text{Matching}
$$

1. **Segmentazione**: si scartano le riflessioni e si mantiene solo la parte compresa entro i bordi circolari rilevati (pupilla e iride); si ottiene una **maschera** dei pixel appartenenti effettivamente all'iride.
2. **Normalizzazione**: non è una semplice proiezione polare, ma tiene conto del fatto che la pupilla spesso **non è concentrica** all'iride → si usa una mappatura pseudo-polare (**Rubber Sheet Model**, vedi §2.6).
3. **Coding**: si estraggono feature evitando gli elementi di "background" (*don't care elements*), limitandosi alle regioni classificate come iride secondo la maschera normalizzata (es. tramite **LBP** o filtri di **Gabor**, vedi §2.7).
4. **Matching**: confronto tra due codici iride tramite una misura di distanza (es. **Hamming distance**), la cui interpretazione dipende dal task (verifica o identificazione).

### 2.6 Rubber Sheet Model (Daugman)

Il modello mappa ogni punto dell'iride in coordinate polari $(r,\theta)$, con:
$$
r \in [0,1], \qquad \theta \in [0, 2\pi]
$$

Il centro delle coordinate polari è il centro della **pupilla**, ma la trasformazione **non è una semplice trasformazione polare**: le nuove coordinate di ogni punto sono una **combinazione lineare** dei punti del contorno pupillare e di quelli del contorno esterno dell'iride.

Prendendo un numero fisso di punti su ciascun raggio compreso tra il bordo della pupilla e il bordo dell'iride, la distanza deformata viene normalizzata: nelle regioni più ampie il campionamento sarà meno denso, in quelle più strette più denso.

**Proprietà del modello:**
- **Compensa** la dilatazione della pupilla e le variazioni di dimensione, producendo una rappresentazione invariante.
- **Non compensa** le rotazioni: in fase di matching, la compensazione avviene traslando il template in coordinate polari fino all'allineamento ottimale.

> Perché le coordinate polari? Ciò che è circolare nell'immagine sorgente diventa **rettangolare** nell'immagine proiettata (bande circolari → strisce orizzontali), e le linee sono matematicamente molto più semplici da processare rispetto ai cerchi.

### 2.7 Codifica: filtri di Gabor e Iris Code (Daugman)

Le feature vengono estratte applicando **filtri di Gabor** all'immagine $I(r,\theta)$ in coordinate polari.

Per ciascun elemento di coordinate $(r,\theta)$ nell'immagine $I(\rho,\phi)$, il metodo calcola una coppia di bit come segue:

$$
h_{Re} = 1 \quad \text{se} \quad \text{Re}\left[\iint_{\rho\,\phi} I(\rho,\phi)\, e^{-i\omega(\theta_0-\phi)}\, e^{-(r_0-\rho)^2/\alpha^2}\, e^{-(\theta_0-\phi)^2/\beta^2}\; \rho\, d\rho\, d\phi \right] \ge 0
$$
$$
h_{Re} = 0 \quad \text{se} \quad \text{Re}[\cdots] < 0
$$
$$
h_{Im} = 1 \quad \text{se} \quad \text{Im}\left[\iint_{\rho\,\phi} I(\rho,\phi)\, e^{-i\omega(\theta_0-\phi)}\, e^{-(r_0-\rho)^2/\alpha^2}\, e^{-(\theta_0-\phi)^2/\beta^2}\; \rho\, d\rho\, d\phi \right] \ge 0
$$
$$
h_{Im} = 0 \quad \text{se} \quad \text{Im}[\cdots] < 0
$$

In pratica, questi integrali diventano **convoluzioni con kernel** applicate all'immagine digitale: in base al segno del risultato (maggiore o minore di 0), la parte reale e la parte immaginaria assumono rispettivamente valore 1 o 0. L'**iris code** finale è quindi composto da una coppia di bit per ogni posizione dell'immagine normalizzata.

### 2.8 Matching: Hamming Distance

Il confronto tra due iris code avviene tramite la **Hamming distance**:

$$
HD = \frac{1}{N}\sum_{j=1}^{N} A_j \otimes B_j
$$

dove:
- $N$ è il numero totale di valori (bit) nel rettangolo del codice;
- $\otimes$ è l'operatore **XOR**: per ogni coppia di punti corrispondenti, se i valori sono uguali si somma 0, se sono diversi si somma 1;
- il risultato viene poi diviso per il numero totale di valori.

Poiché non tutti i pixel sono validi (alcuni sono occlusi da ciglia/palpebre e mascherati), si usa la **Hamming distance con maschera**:

$$
HD = \frac{\lVert (codeA \otimes codeB)\ \cap\ maskA \cap maskB \rVert}{\lVert maskA \cap maskB \rVert}
$$

> Un pixel classificato come "iride" in una sola delle due immagini (a causa di maschere diverse) non può essere considerato un pixel valido di confronto: viene escluso da entrambe.

### 2.9 Algoritmo di Daugman — pipeline completa

$$
\text{Acquisizione (IR, condizioni controllate)} \rightarrow \text{Iris location \& unwrapping} \rightarrow \text{Feature extraction \& coding} \rightarrow \text{Matching (vs template DB)} \rightarrow \text{Result}
$$

**Localizzazione (circular edge detector — operatore integro-differenziale):**

L'operatore sfrutta la **convoluzione** dell'immagine con una funzione di smoothing **Gaussiana** con centro e deviazione standard dati, cercando il **percorso circolare** lungo il quale la variazione dei pixel è massimizzata, al variare del centro $(x_0, y_0)$ e del raggio $r$ di un contorno circolare candidato.

Quando il cerchio candidato ha lo stesso raggio e centro dell'iride (o della pupilla), l'operatore produce un **picco** di risposta. Si ripete la convoluzione per ogni centro e ogni raggio possibile, cercando il massimo.

> Anche in presenza di **circonferenze parziali** (occluse), queste producono comunque una risposta più alta rispetto ad altri cerchi centrati altrove, poiché corrispondono a un reale cambiamento di intensità lungo quella porzione circolare.

**Eyelids location:** Daugman effettua anche la localizzazione delle palpebre, poiché una parte dell'area interna al cerchio dell'iride può in realtà essere occlusa dalle palpebre (elemento "don't care"). La procedura è analoga a quella circolare, ma cerca **archi**, approssimati tramite **spline**.

### 2.10 Le campagne di valutazione NICE (Noisy Iris Challenge Evaluation)

NICE è un'iniziativa di valutazione biometrica dell'iride con partecipazione mondiale, articolata in due fasi:
- **NICE.I**: valuta tecniche di **segmentazione** e rilevamento del rumore.
- **NICE.II**: valuta strategie di **codifica e matching** delle firme biometriche.

Entrambe le fasi usano i database **UBIRIS**, sviluppati dal *Soft Computing and Image Analysis Group Lab* dell'Università di Beira Interior (Portogallo):
- Immagini iridee a **lunghezza d'onda visibile**, catturate in condizioni di illuminazione eterogenee (che portano ad immagini fortemente degradate);
- Volontari: **~90% caucasici latini, ~8% neri, ~2% asiatici**;
- Due sessioni di acquisizione distinte (ciascuna di due settimane, separate da un intervallo di una settimana), tra le quali sono state cambiate posizione/orientamento del dispositivo e delle fonti di luce artificiale;
- Circa il 60% dei volontari ha partecipato a entrambe le sessioni.

#### 2.10.1 NICE.I — Segmentazione

Il protocollo richiede un eseguibile standalone (qualsiasi linguaggio), eseguito **senza accesso a Internet**.

**Valutazione**: dato un algoritmo $Alg$ che segmenta la regione di iride priva di rumore, un dataset di immagini in ingresso $I$, le corrispondenti immagini di output $O = Alg(I)$ e la **ground truth** $C$ (segmentazione manuale "perfetta"), si usano due misure:

- **Classification error rate ($E_1$)**: proporzione di pixel discordanti (via **XOR**) tra output e ground truth, normalizzata su tutta l'immagine; mediata su tutte le immagini. Range $[0,1]$, dove 0 = ottimale e 1 = pessimo. È la misura principale di classificazione dei partecipanti.
$$E_i^1 = \frac{1}{c\cdot r}\sum_{c'}\sum_{r'} O(c',r') \otimes C(c',r'), \qquad E^1 = \frac{1}{n}\sum_i E_i^1$$
- **Type-I/Type-II error rate ($E_2$)**: compensa la sproporzione tra probabilità a priori di pixel "iride" e "non-iride", mediando i tassi di falsi positivi e falsi negativi:
$$E_i^2 = 0.5 \cdot FPR + 0.5 \cdot FNR, \qquad E^2 = \frac{1}{n}\sum_i E_i^2$$

I migliori 8 partecipanti (con i tassi di errore più bassi) sono stati invitati a pubblicare il proprio approccio.

**L'algoritmo vincitore, ISIS**, è stato presentato da **CASIA** (National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences). Tutti i parametri operativi (es. dimensione delle finestre) e decisionali (es. soglie) sono stati sintonizzati sperimentalmente su un training set separato — **senza bisogno di ri-tarare i parametri per ogni singolo database**, a differenza di altri metodi concorrenti.

ISIS prevede **quattro fasi principali**:

1. **Pre-processing**: dettagli come vasi della sclera, pori e ciglia possono interferire con l'edge detection. Si applica un **filtro di posterizzazione** $F_E$ (Enhance): una finestra quadrata scorre sull'immagine pixel per pixel, si calcola un istogramma della regione e il valore più frequente sostituisce il pixel centrale. Successivamente si applica il filtro di **Canny** con **dieci soglie diverse** ($th = 0.05, 0.10, ..., 0.55$).
2. **Localizzazione della pupilla**: invece della costosa trasformata di **Hough**, ISIS usa il metodo di rilevamento circolare rapido e preciso di **Taubin**, selezionando il miglior cerchio candidato tramite due criteri:
   - **Omogeneità**: assegna punteggi di omogeneità alle regioni dell'immagine (non sempre la regione più scura è la pupilla).
   - **Separabilità**: sia il limbus che il contorno della pupilla rappresentano un confine con un salto marcato da zona scura a chiara; si confrontano i livelli di grigio su due cerchi concentrici, uno interno ($\rho_1 = 0.9\rho$) e uno esterno ($\rho_1 = 1.1\rho$) al cerchio candidato, per ogni angolo $\theta$.
   - Il punteggio finale è $S = S_H + S_D$; il cerchio con punteggio massimo $S_{max}$ è la pupilla stimata.
3. **Linearizzazione**: lungo la direzione verticale si identifica con precisione la regione di confine del limbus che separa iride e sclera.
4. **Localizzazione del limbus**: per ogni colonna si calcola una **differenza pesata** $\Delta(\rho_j,\theta_i)$; il confine del limbus è dato dai punti che massimizzano $\Delta$ per ogni colonna. Poiché questi punti dovrebbero avere $\rho$ pressoché costante, un **criterio di levigatezza** (basato sulla deviazione dalla mediana $\rho_{med}$) scarta i punti anomali (outlier) causati dal rumore.

**Risultati sperimentali**: confrontato con un sistema basato sull'implementazione di Masek (segmentazione alla Wildes, riconoscimento alla Daugman), ISIS non ha richiesto alcuna ri-taratura delle soglie per adattarsi al singolo database, a differenza degli altri metodi.

#### 2.10.2 NICE.II — Codifica e matching

Il protocollo è lo stesso di NICE.I. La feature principale dell'iride è la sua **texture**:

- **LBP** (si veda anche il capitolo Face Recognition) analizza le regolarità tessiturali: calcola 256 valori da un vicinato di 8 elementi; l'istogramma dei valori rappresenta la texture. L'iride normalizzata è divisa in bande **orizzontali** (anatomicamente più significative) o verticali — Codice = insieme di istogrammi + maschera. La divisione orizzontale è risultata migliore di quella verticale; è preferibile evitare bande troppo strette.
- Un'altra feature è data dai **blob** dell'iride (solchi e cripte), rilevati tramite il **Laplaciano del Gaussiano (LoG)** a diverse scale; dopo normalizzazione, fusione (punto con LoG più alto) e binarizzazione, il matching usa la **Hamming distance pesata dalla maschera di segmentazione**, considerando anche shift di 10 pixel per l'allineamento ottimale.
- **LBP-BLOB**: fonde le due codifiche concatenandole; il punteggio di matching è la media dei due punteggi (LBP e BLOB) — poiché per alcune immagini un metodo funziona meglio dell'altro, la fusione è generalmente più efficace di ciascun metodo singolo.

I partecipanti al contest NICE.II sono stati classificati in base ai valori di **decidability** (più alta = migliore). Il gruppo italiano **BIPlab** si è classificato **6°** utilizzando un descrittore di texture basato su LBP combinato con rilevamento BLOB delle singolarità tessiturali.

---

## 3. Fingerprints (Impronte digitali)

### 3.1 Caratteristica comune con iride e vasi sanguigni

$$
\text{Impronte digitali} \equiv \text{Iride} \equiv \text{Pattern vascolare} \;\;\Rightarrow\;\; \text{prevalenza di elementi \textbf{randotipici}}
$$

Tale componente randotipica si origina durante la **gravidanza** (formazione fetale), rendendo il pattern non ereditario in senso stretto, pur non essendo completamente casuale (vedi §3.5).

### 3.2 Definizione e acquisizione

Un'impronta digitale appare come una serie di **linee scure** che rappresentano la porzione alta e sporgente della pelle a creste (*friction ridge skin*), mentre le valli tra le creste appaiono come spazio bianco (porzione bassa e superficiale).

Le impronte sono tracce di un'impressione lasciata dalle creste di frizione di qualsiasi parte di una mano o piede (umano o di altro primate); vengono depositate facilmente su superfici adatte (vetro, metallo, pietra levigata) grazie alle secrezioni naturali di sudore delle **ghiandole eccrine** presenti nelle creste epidermiche.

**Due tipologie di impronte:**
- **Da sensore/inchiostro**: prodotte direttamente dalla pelle della regione.
- **Latenti** (*latent fingerprints*): lasciate involontariamente su una superficie al tocco; raccolte, ad esempio, con polvere speciale durante le indagini forensi. In questo caso si confronta un'impronta completa (quella arruolata in gallery) con un'impronta latente che di solito è **un frammento** → nasce il **problema dell'allineamento**.

### 3.3 Storia e classificazione: Galton

**Sir Francis Galton** (1822–1911), polimatico e antropologo inglese, pubblicò un modello statistico dettagliato per l'analisi e l'identificazione delle impronte, promuovendone l'uso in ambito forense nel libro *"Finger Prints"*.

Galton definì i **tre tipi base** di impronta in base al numero di **delta**:

$$
\begin{cases}
\text{Whorl (vortice)} & \rightarrow 2 \text{ delta} \\
\text{Loop (cappio)} & \rightarrow 1 \text{ delta} \\
\text{Arch (arco)} & \rightarrow 0 \text{ delta}
\end{cases}
$$

Queste sono dette **Macro-Singolarità** (o *first level features*): comportano una considerazione **globale** del pattern delle creste, e furono tra le prime feature usate per confrontare impronte.

> **Classificazione completa (5 classi)**: in pratica, le impronte si dividono in cinque gruppi in base alla forma generale e alla direzione di apertura delle creste: **Left-Loop (LL)** e **Right-Loop (RL)** (1 delta ciascuno, a seconda della direzione di apertura del cappio), **Whorl (W)** (2 delta), **Plain Arch (PA)** e **Tented Arch (TA)** (0 delta). PA e TA possono essere unificate nella categoria generale **Arch (A)**, riducendo le classi da cinque a quattro. Questa classificazione costituisce una partizione macroscopica (*coarse level*) del database, utile ad assegnare una classe all'impronta query e a restringere lo spazio di ricerca (confrontando la query solo con i template della stessa classe), rendendo il matching più veloce su grandi database.

> Galton era in realtà primariamente interessato a usare le impronte come supporto allo studio dell'ereditarietà razziale.

Galton introdusse anche il concetto di **minuzia** (*minutia*, o **Galton feature**): micro-singolarità determinate dai punti terminali o dalle biforcazioni delle linee di cresta. Queste costituiscono le **second level features** (o **Micro-Singolarità**): il numero di minuzie corrispondenti tra due impronte da confrontare viene usato come misura di distanza.

Un'ulteriore minuzia è il **core** (nucleo): brusco cambio di direzione, corrispondente alla cresta più interna in una sequenza di archi.

**Tabella dei tipi di minuzia (Galton features):**

| Tipo | Descrizione |
|---|---|
| Terminazione (*Termination*) | Fine improvvisa di una cresta |
| Biforcazione (*Bifurcation*) | Una cresta si divide in due |
| Lago (*Lake*) | Cresta che si divide e si ricongiunge |
| Cresta indipendente (*Independent ridge*) | Piccola cresta isolata |
| Punto/isola (*Point/Island*) | Piccolissimo segmento isolato |
| Sperone (*Spur*) | Ramo laterale corto |
| Incrocio (*Crossover*) | Due creste che si intersecano |

### 3.4 Third level features e Dermatoglyphics

Con un sensore ad **altissima risoluzione** (dell'ordine di **1000 dpi**) è possibile studiare i **pori** lungo le creste: queste sono le **third level features**.

**Dr. Harold Cummins** (1894–1976) è riconosciuto come il "padre della *Dermatoglyphics*" (studio scientifico dei pattern delle creste cutanee su palmi delle mani e altre parti del corpo). La sua metodologia (*Cummins Methodology*) è tuttora utilizzata come strumento nella tracciatura di relazioni genetiche ed evolutive, e ha trovato applicazione nella diagnosi di alcune patologie (ritardo mentale, schizofrenia, palatoschisi, malattie cardiache). Nel libro *"Finger Prints, Palms and Soles"* (1943) presentò un albero genealogico di 39 tipi di impronte.

### 3.4bis Il sistema di Henry e la storia degli AFIS

Nel **1899, Edward Henry** propose un sistema di classificazione (**Henry System**) che raggruppa le quattro classi di whorl in un'unica classe $W$ e le due classi di arch in un'unica classe $A$. Le frequenze naturali risultanti nella popolazione sono: $W: 28\%$, $A: 6.6\%$, $L$ (loop sinistro): $33.8\%$, $R$ (loop destro): $31.7\%$.

**Storia degli AFIS (Automated Fingerprint Identification System):**
- Nel **1924** l'FBI possedeva già un database di circa **810.000 impronte complete** (10 dita), trasferite su carta con inchiostro.
- Il ritmo di crescita accelerò rapidamente, portando il totale a superare abbondantemente i **200 milioni di profili registrati**, rendendo di fatto impossibile l'identificazione manuale.
- A partire dagli **anni '60**, i principali corpi di polizia mondiali investirono risorse nello sviluppo di sistemi automatizzati, dando origine agli **AFIS**: i progettisti dei primi AFIS si basarono sull'osservazione del lavoro degli esperti umani, dovendo affrontare la cattura digitale dell'impronta, l'estrazione delle caratteristiche locali di creste e solchi, e un confronto robusto tra nuove acquisizioni e profili registrati.
- Oggi gli AFIS sono usati dalle forze di polizia di tutto il mondo, con vantaggi in termini di efficienza nell'identificazione dei criminali e risparmio di personale altamente specializzato.

> Nella pratica il sistema di classificazione combinato è chiamato **Galton-Henry**, poiché i due sistemi (Galton e Henry) furono fusi in un unico modello.

### 3.5 Formazione e unicità

- La formazione delle impronte è **già completa al settimo mese** di sviluppo fetale, e la configurazione delle creste su ciascun dito rimane **costante per tutto il ciclo di vita**.
- Le creste si formano secondo una modalità di crescita comune a quella dei **vasi capillari/sanguigni** durante l'**angiogenesi**.
- Fattori rilevanti che influenzano il microambiente fetale: il flusso del **liquido amniotico** e la posizione durante il processo di differenziazione.
- La **microdiversità** delle condizioni ambientali attorno a ciascun polpastrello caratterizza la formazione dei dettagli più minuti della superficie: ogni piccola differenza microambientale viene amplificata dal processo di differenziazione cellulare, rendendo ogni impronta praticamente **unica** (proprietà empirica, non dimostrata analiticamente).
- Poiché il punto di partenza del processo di differenziazione è determinato dagli stessi geni, i pattern risultanti **non sono totalmente casuali**.
- Nei **gemelli identici**, i dettagli minuti delle impronte differiscono, ma la maggior parte degli studi mostra **somiglianze significative** negli attributi generali (numero, larghezza, separazione e profondità delle creste).

**Ranking di diversità delle impronte** (in ordine di diversità **decrescente**):

$$
\text{Persone stesso gruppo etnico senza parentela} \;>\; \text{Padre e figlio (metà geni condivisi)} \;>\; \text{Fratelli/sorelle} \;>\; \text{Gemelli}
$$

> La massima differenza tra impronte si trova tra individui di gruppi etnici diversi; tuttavia è **impossibile determinare l'etnia** a partire dalla sola impronta.

### 3.6 Modalità di acquisizione

**Off-line (due fasi):**
1. Il dito viene passato su un tampone d'inchiostro e l'impronta lasciata su carta bianca.
2. Digitalizzazione successiva tramite scansione ottica o fotocamera ad alta risoluzione.

> Il dispositivo di cattura può introdurre rumore sul polpastrello; anche la carta usata può lasciare un pattern caratteristico indesiderato.

**Live-scan (online):** l'immagine digitale viene acquisita direttamente tramite contatto del dito con un sensore dedicato (non necessariamente basato su visione: es. su pressione). Anche qui possono presentarsi rumori dovuti a sporco sul sensore o altri fattori.

**Impronte latenti:** acquisite tramite polveri speciali e tecniche di trasferimento su carta apposita — processo concettualmente simile al caso off-line.

### 3.7 Parametri caratterizzanti l'immagine digitale

| Parametro | Descrizione |
|---|---|
| **Risoluzione** | Numero di punti per pollice (dpi) |
| **Area di acquisizione** | Un'area $\ge 1 \times 1$ pollice consente l'acquisizione di un'impronta intera e chiara |
| **Profondità (depth)** | Numero di bit usati per codificare l'intensità di ogni pixel |
| **Contrasto** | Un'immagine nitida contiene dettagli migliori |
| **Distorsione geometrica** | Distorsione massima introdotta dal dispositivo, o causata da pressione eccessiva del dito |

> La soluzione migliore prevede un **operatore umano** che verifichi la qualità dell'impronta acquisita: con pressione **troppo bassa** non tutte le creste vengono evidenziate; con pressione **troppo alta** si ha uno "schiacciamento" (*smashing*) dell'impronta.

### 3.8 Tipi di sensore

**Scanner Ottici**

| Vantaggi | Svantaggi |
|---|---|
| Tollerano, in una certa misura, le fluttuazioni di temperatura | Dimensioni: la piastra sensibile deve essere sufficientemente grande per una buona immagine |
| Costo relativamente basso | Impronte residue di utenti precedenti possono degradare l'immagine (sovrapposizione di due set di impronte) |
| Risoluzioni fino a 500 dpi | Il coating e gli array CCD si usurano nel tempo, riducendo l'accuratezza |
| — | Molti produttori si stanno spostando verso tecnologia a base di silicio |

**Scanner Capacitivi**

| Vantaggi | Svantaggi |
|---|---|
| Chip in silicio ~200×200 linee su un wafer di ~1×1.5 cm → buona risoluzione | Nonostante le dichiarazioni dei produttori, la durabilità del silicio non è ancora pienamente dimostrata |
| Migliore qualità immagine, con area minore rispetto agli ottici | Con la riduzione delle dimensioni del sensore, diventa ancora più importante curare enrollment e verifica |
| Costo inferiore grazie alle dimensioni ridotte | — |
| Miniaturizzazione → integrazione in numerosi dispositivi | — |

**Scanner Termici**

| Vantaggi | Svantaggi |
|---|---|
| Forte immunità alle scariche elettrostatiche | L'immagine svanisce rapidamente |
| Funzionano bene sia a temperatura ambiente sia in condizioni estreme | Dopo il contatto iniziale, dito e array di pixel raggiungono l'equilibrio termico e il segnale scompare (mitigabile facendo scorrere il dito su un sensore stretto e alto) |
| Impossibili da ingannare con un dito artificiale semplice | — |

**Sensori piezoelettrici**: sfruttano l'**effetto piezoelettrico** per misurare variazioni di pressione, accelerazione, deformazione o forza, convertendole in carica elettrica.

### 3.9 Basi del matching gerarchico

Il matching segue lo stesso workflow degli esperti umani, organizzato in un **albero decisionale gerarchico**:

1. **Accordo qualitativo globale**: si verifica innanzitutto che le impronte da confrontare condividano la stessa **tipologia di pattern globale** (whorl/loop/arch). Ciò implica, nel caso delle impronte latenti, che il frammento sia sufficientemente ampio da poter contenere il pattern globale (che è generalmente abbastanza esteso da essere visibile anche in frammenti latenti).
2. **Accordo qualitativo locale**: le minuzie corrispondenti devono essere identiche (in una certa regione si trova, ad esempio, una biforcazione insieme a un punto terminale, e così via).
3. **Fattore quantitativo**: deve esserci un numero minimo di minuzie corrispondenti tra le due impronte.
4. **Corrispondenza interrelazionale profonda**: le minuzie devono essere identicamente interrelazionate — non basta trovare due biforcazioni nella stessa regione, ma queste devono essere separate dallo stesso numero di creste in entrambe le impronte.

### 3.10 Approcci metodologici al matching

**a) Matching basato su correlazione**
- Si sovrappongono le due immagini e si calcola la **correlazione** tra i pixel corrispondenti.
- Problema principale: l'**allineamento** — il calcolo va iterato per diversi allineamenti fino a trovare quello ottimale.
- Sensibile a **trasformazioni non lineari** (es. dovute allo scorrimento del dito durante la cattura).
- **Alta complessità computazionale**.
- Usato principalmente per caratteristiche globali (matching completo, macro-caratteristiche).

**b) Matching basato su ridge features**
- L'estrazione delle minuzie in immagini di bassa qualità è problematica → si usano feature alternative: orientazione delle creste e frequenza locale, forma delle creste, texture.
- Feature più **affidabili e facili da estrarre**, ma **meno distintive**.
- Basso potere discriminante, ma utile come **primo step** per ridurre lo spazio di ricerca o arrivare più rapidamente a una decisione.

**c) Matching basato su minuzie**
- Le minuzie vengono estratte da entrambe le impronte e memorizzate come **due insiemi di punti** in uno spazio bidimensionale.
- Problema principale: l'**orientazione** — si cerca il **miglior allineamento** tra i due insiemi (quello che massimizza il numero di coppie corrispondenti), tenendo conto che, a causa di problemi temporanei del sensore, alcune minuzie possono mancare in una delle due immagini.
- Dopo l'allineamento si può approfondire l'analisi misurando l'**interrelazione** (pattern rispetto alle creste su cui compaiono, orientazione, distanza tra le minuzie, ecc.).

### 3.11 Problemi tipici del matching

- **Sovrapposizione scarsa** (*scarce overlap*): il dito può non essere ben centrato sul sensore → si considera solo la parte comune, allineando le porzioni condivise.
- **Condizioni della pelle diverse** (es. pelle secca ostacola una cattura corretta).
- **Distorsione non lineare** dovuta a pressione differente.
- **Movimento e/o distorsione eccessivi**.
- **Distorsione non lineare della pelle**: essendo una struttura 3D deformabile elasticamente in base alla pressione, la stessa impronta acquisita in due sessioni diverse con pressioni diverse può apparire leggermente differente.
- **Pressione variabile e condizioni della pelle**.
- **Errori nell'estrazione delle feature**: gli algoritmi di estrazione sono imperfetti e introducono spesso errori di misura, in particolare su impronte di bassa qualità.

### 3.12 Pipeline di estrazione delle feature

$$
\text{Acquisizione} \rightarrow \text{Directional Map \& Density Map} \rightarrow \text{Singolarità (Poincaré)} \rightarrow \text{Ridge pattern} \rightarrow \text{Minuzie}
$$

**Segmentazione**: separazione tra il **foreground** (pattern striato e orientato dell'impronta — *anisotropo*) e il **background** (*isotropo*).

> **Anisotropia** = dipendenza dalla direzione. Il background ha proprietà identiche in tutte le direzioni; nell'impronta, contrasto e intensità dipendono dall'orientazione. Questa proprietà è usata per distinguere l'impronta dal background circostante.

**Misure di anisotropia (criteri per la segmentazione), con i relativi autori:**

| Metodo | Autori | Idea |
|---|---|---|
| **Picco nell'istogramma delle orientazioni locali** | Mehtre et al. (1987) | Si stima l'orientazione della cresta in ogni pixel e si calcola un istogramma per ogni blocco $16\times16$; un picco marcato indica un pattern orientato, un istogramma piatto è tipico del segnale isotropo (background). |
| **Varianza dei livelli di grigio in direzione perpendicolare al gradiente** | Ratha, Chen & Jain (1995) | Nelle regioni rumorose il pattern non dipende dalla direzione; l'area dell'impronta è invece caratterizzata da varianza molto alta in direzione ortogonale all'orientazione della cresta e molto bassa lungo la cresta stessa. |
| **Magnitudine del gradiente** | Maio & Maltoni (1997) | L'area dell'impronta è ricca di edge per l'alternanza cresta/valle, quindi il gradiente è alto nel foreground e più basso altrove. |
| **Combinazione di più caratteristiche** | Bazen & Gerez (2001) | Per ogni pixel si calcolano coerenza del gradiente, media e varianza dell'intensità; l'assegnazione a foreground/background è affidata a un **classificatore**. |

#### 3.12.1 Directional Map (mappa direzionale)

Il flusso delle linee di cresta può essere descritto efficacemente da una struttura chiamata **directional map** (o *directional image*): una matrice discreta i cui elementi denotano l'orientazione della tangente alle linee di cresta.

L'**orientazione locale** della linea di cresta nella posizione $[i,j]$ è definita come l'angolo $\theta(i,j)$ che la linea di cresta (o la tangente ad essa), attraversando un intorno del punto $[i,j]$, forma con l'**asse orizzontale**.

Ogni elemento $[i,j]$ della griglia sovrapposta all'immagine indica l'**orientazione media** della tangente alle creste in un intorno del punto $(x_i, y_j)$. Per limitare il costo computazionale, molti metodi misurano l'orientazione solo su una griglia di punti fissi, anziché in ogni punto.

**Approccio basato sul gradiente**: estrae l'orientazione a partire dal gradiente dell'immagine, ma la stima di una singola orientazione è un'analisi di basso livello troppo sensibile al rumore. Non è possibile fare una semplice media di più gradienti a causa della **circolarità** degli angoli (il concetto di orientazione media non è ben definito, specialmente con una griglia poco densa). Soluzione tipica: **raddoppiare gli angoli** e considerare separatamente le medie lungo i due assi.

#### 3.12.2 Frequency Map (mappa di frequenza)

Si basa sul numero di creste per unità di superficie che attraversano un segmento ipotetico, centrato in un dato punto e ortogonale all'orientazione locale della cresta.

Si calcola un'immagine di frequenza $F$ (analoga alla directional image $D$), stimando la frequenza in posizioni discrete disposte su griglia — tipicamente contando il numero medio di pixel tra picchi consecutivi di livello di grigio, lungo la direzione ortogonale all'orientazione locale della cresta.

#### 3.12.3 Indice di Poincaré (ricerca delle singolarità)

Una **directional map** è un **campo vettoriale** $G$ (regione riempita di vettori con orientazioni diverse). L'impronta è vista come una **curva** $C$ immersa in questo campo vettoriale.

**Definizione:** dato un campo vettoriale $G$ e una curva $C$ immersa in $G$, l'**indice di Poincaré** $P_{G,C}$ è definito come la **rotazione totale** dei vettori di $G$ lungo $C$:

$$
P_{G,C}(i,j) = \sum_{k=0}^{7} \text{angle}\big(\mathbf{d}_k, \mathbf{d}_{(k+1)\bmod 8}\big)
$$

dove $G$ è il campo associato all'immagine delle orientazioni $D$ dell'impronta, e $[i,j]$ è la posizione dell'elemento $\theta_{ij}$ nell'immagine.

**Procedura di calcolo:**
- La curva $C$ è un percorso **chiuso**, definito come sequenza ordinata di elementi di $D$, tale che $[i,j]$ siano punti interni.
- L'indice $P_{G,C}(i,j)$ si calcola sommando **algebricamente** le differenze di orientazione tra elementi adiacenti di $C$.
- La somma delle differenze di orientazione richiede di associare una **direzione** a ciascuna orientazione: si può selezionare arbitrariamente la direzione del primo elemento, e assegnare a ogni elemento successivo la direzione più vicina a quella del precedente.
- Per curve chiuse, l'indice di Poincaré assume **solo uno** dei valori discreti $0°, \pm 180°, \pm 360°$.

**Classificazione delle singolarità tramite indice di Poincaré:**

$$
P_{G,C} =
\begin{cases}
0° & \text{nessuna singolarità} \\
360° & \text{whorl (vortice)} \\
180° & \text{loop (cappio)} \\
-180° & \text{delta}
\end{cases}
$$

### 3.13 Estrazione delle minuzie

**Fasi principali:**
1. **Binarizzazione**: conversione da immagine a livelli di grigio a immagine binaria (l'analisi dell'istogramma è usata, ma è meno banale di quanto sembri).
2. **Thinning** (assottigliamento): trasforma ogni cresta più spessa di 1 pixel in una linea di spessore **1 pixel**.
3. **Scansione** delle creste assottigliate per localizzare i pixel corrispondenti alle minuzie.

#### 3.13.1 Crossing Number (numero di attraversamento)

La localizzazione delle minuzie si basa sull'analisi del **crossing number**:

$$
cn(\mathbf{p}) = \frac{1}{2}\sum_{i=1}^{8} \left| val(\mathbf{p}_{i \bmod 8}) - val(\mathbf{p}_{i-1}) \right|
$$

dove $p_0, p_1, \dots, p_7$ sono i pixel nell'intorno (8-connesso) di $p$, e $val(p) \in \{0,1\}$ è il valore binario del pixel $p$. Il **mod 8** serve perché, arrivati alla fine della "stella" di vicini, si deve ripartire dal primo.

Il *crossing number* rappresenta il numero di cambi di colore che avvengono nell'intorno del pixel considerato.

**Classificazione di un pixel $p$ con $val(p) = 1$:**

$$
cn(p) =
\begin{cases}
2 & \text{punto interno} \text{ di una linea di cresta (non rilevante, non è una minuzia)} \\
1 & \text{terminazione} \text{ (un solo cambio di direzione)} \\
3 & \text{biforcazione} \\
>3 & \text{minuzia più complessa}
\end{cases}
$$

#### 3.13.2 Ridge Count

Il **ridge count** è una misura astratta della distanza tra due punti $a$ e $b$ di un'impronta: è il numero di linee di cresta intersecate dal segmento $ab$ (il numero di creste che separano due minuzie).

I punti $a$ e $b$ sono tipicamente scelti tra i punti "rilevanti" (es. **core** e **delta**). Questo calcolo non può essere effettuato per ogni coppia di minuzie, poiché i punti terminali non sono sufficientemente affidabili (possono essere causati da interruzioni di una cresta dovute a un cattivo thresholding o altro).

### 3.14 Allineamento (Image Alignment)

Prima di qualsiasi confronto affidabile, le impronte devono essere **allineate**. La fase di allineamento procede come segue:

1. Estrazione delle minuzie sia dall'input sia dal template.
2. Un algoritmo di **point matching** seleziona preliminarmente una **coppia di minuzie di riferimento** (una per immagine).
3. Si determina il numero di coppie di minuzie corrispondenti usando l'insieme dei punti rimanenti.
4. La **coppia di riferimento** che produce il massimo numero di coppie corrispondenti determina il **miglior allineamento**.

Dall'allineamento ottimale si ricavano anche i parametri di trasformazione:
- **Parametro di rotazione**: media dei valori di rotazione stimati per tutte le singole coppie di minuzie corrispondenti.
- **Parametri di traslazione**: calcolabili dalle coordinate spaziali della coppia di minuzie di riferimento che ha prodotto il miglior allineamento.

> Grazie alla disponibilità di informazioni relative ai solchi per ciascuna minuzia locale, non è necessario effettuare una valutazione esaustiva di tutte le corrispondenze tra i punti.

### 3.15 Approccio ibrido (minuzie + texture, Gabor filtering)

$$
\text{Fingerprint Sensing} \rightarrow \text{Fingerprint Image Alignment} \rightarrow \text{Minutiae Extraction} \rightarrow \text{Gabor Filtering} \rightarrow \text{Fingerprint Matching}
$$

**Procedura dettagliata:**
1. Le immagini di input e i template vengono **normalizzati**, costruendo una griglia che li divide in finestre non sovrapposte della stessa dimensione, e normalizzando l'intensità luminosa dei pixel di ciascuna finestra rispetto a una **media e varianza costanti**.
2. La distanza inter-solco media, per una risoluzione di acquisizione di 300×300 dpi, è di circa **30 pixel** → dimensione ottimale della cella: **30×30 pixel**.
3. Per l'estrazione delle feature da ciascuna cella si usa un gruppo di **8 filtri di Gabor**, tutti con la stessa frequenza ma con orientazione variabile → si producono 8 immagini filtrate per cella.
4. La **deviazione media assoluta** dell'intensità in ciascuna cella filtrata rappresenta il suo valore caratteristico → si ottengono 8 valori caratteristici per cella.
5. I valori caratteristici di tutte le celle vengono concatenati in un **vettore caratteristico** (feature vector).
6. I valori relativi alle regioni mascherate nell'immagine di input **non vengono usati** nel confronto e sono marcati come valori mancanti.

**Matching:**
- Confronto tra i vettori caratteristici, calcolando la **somma delle differenze al quadrato** tra vettori corrispondenti, dopo aver scartato i valori mancanti:
$$
d_{\text{tessellation}} = \sum_{k} \left( f_A^{(k)} - f_B^{(k)} \right)^2
$$
- Il punteggio di similarità così ottenuto viene **combinato** con quello ottenuto dal confronto delle minuzie, usando la **regola della somma** delle combinazioni.
- Se il punteggio di similarità è **al di sotto di una soglia**, si dichiara che l'immagine in input ha un template corrispondente in memoria e il riconoscimento ha successo.

### 3.16 Fake fingerprints e Liveness Detection

**Impronte false**: non è particolarmente difficile riprodurre impronte digitali false, soprattutto a partire da un utente cooperativo. Possono essere create con diversi materiali (**gelatina, silicone, lattice**), ciascuno con qualità e caratteristiche diverse, sia dal punto di vista ottico sia elettrico. La risposta a un possibile attacco dipende quindi dal tipo di sensore impiegato (ottico, capacitivo, piezoelettrico).

**Liveness Detection**: misura di sicurezza volta a contrastare i tentativi di frode nei sistemi di impronte digitali, verificando che la sorgente del segnale (il dito) sia un tratto biometrico **vivo e genuino**, piuttosto che un simulacro (es. un guanto che riproduce l'impronta di un utente autorizzato).

> Premessa logica del test di vitalità: se il dito è vivo, la sua impronta è effettivamente quella della persona a cui appartiene.

**Approcci comuni al test di vitalità:**
- Segni vitali comuni (polso, temperatura).
- Scanner ottici live-scan con tecnologia **FTIR** (*Frustrated Total Internal Reflection*): meccanismo di acquisizione differenziale per creste e solchi, intrinsecamente più resistente agli attacchi con impressioni bidimensionali del dito.
- Scansione ad alta risoluzione: rivela dettagli caratteristici della struttura dei pori, molto difficili da imitare artificialmente.
- Cambiamento del **colore della pelle** dovuto alla pressione esercitata sulla superficie di scansione.
- Rilevamento del **flusso sanguigno e della sua pulsazione**, tramite misurazione accurata della luce riflessa o trasmessa attraverso il dito.
- **Differenza di potenziale** tra due punti specifici della muscolatura del dito (assente in un dito "morto").
- Misurazione dell'**impedenza complessa** del dito.
- **Sudorazione** del dito nel tempo.

> È un'idea errata pensare che il problema del riconoscimento digitale delle impronte sia già completamente risolto, essendo stato tra le prime aree applicative della ricerca sul pattern recognition. Numerose sfide scientifiche e tecnologiche restano aperte, specialmente per impronte di bassa qualità: i sistemi automatici, per quanto validi, non riescono ancora a competere con le capacità di un esperto in tecniche manuali. Tuttavia, i sistemi automatizzati offrono, in media, una soluzione affidabile, veloce, consistente ed economica per un numero crescente di applicazioni reali.

---

## 4. Tabella riassuntiva comparativa

| Biometria | Tipo | Vantaggio chiave | Svantaggio chiave | Algoritmo/Metrica principale |
|---|---|---|---|---|
| **Orecchio** | Passiva, statica | Bassa risoluzione richiesta, colore uniforme | Occlusione da capelli, sensibile a posa/illuminazione | Landmark di Iannarelli, AdaBoost |
| **Iride** | Passiva, protetta | Estremamente discriminativa, time-invariant | Superficie ridotta ($3.64\ \text{cm}^2$), alta risoluzione richiesta | Rubber Sheet Model + Gabor + Hamming Distance |
| **Impronta digitale** | Attiva/passiva (contatto) | Alta discriminabilità, forte base scientifica | Occlusioni, condizioni della pelle, spoofing | Crossing Number, indice di Poincaré, minutiae matching |

---

## 5. Riepilogo — mappa concettuale delle formule chiave

$$
\boxed{HD = \frac{1}{N}\sum_{j=1}^{N} A_j \otimes B_j} \qquad
\boxed{HD_{mask} = \frac{\lVert (codeA \otimes codeB) \cap maskA \cap maskB\rVert}{\lVert maskA \cap maskB \rVert}}
$$

$$
\boxed{cn(\mathbf p) = \frac12 \sum_{i=1}^{8}\left|val(\mathbf p_{i \bmod 8}) - val(\mathbf p_{i-1})\right|}
\qquad
\boxed{P_{G,C}(i,j) = \sum_{k=0}^{7}\text{angle}(\mathbf d_k,\mathbf d_{(k+1)\bmod 8})}
$$

$$
\boxed{r \in [0,1],\ \theta \in [0,2\pi]} \quad \text{(Rubber Sheet Model, Daugman)}
$$

**Schema riassuntivo delle singolarità (impronte):**

| Indice di Poincaré | Singolarità |
|---|---|
| $0°$ | Nessuna |
| $\pm 180°$ | Loop / Delta |
| $360°$ | Whorl |

**Schema riassuntivo del crossing number (minuzie):**

| $cn(p)$ | Significato |
|---|---|
| $1$ | Terminazione |
| $2$ | Punto interno (non minuzia) |
| $3$ | Biforcazione |
| $>3$ | Minuzia complessa |
