# Introduzione ai Sistemi Biometrici — Riassunto Completo

---

## 1. Cos'è la biometria

Il termine **biometria** deriva dal greco *bios* (vita) e *metron* (misura). In generale indica lo studio e l'uso di metodi per rilevare e misurare le caratteristiche di organismi viventi, traendone classificazioni comparative. In Informatica indica il **riconoscimento o la verifica automatica dell'identità di una persona** sulla base di caratteristiche fisiche o comportamentali. Il Biometric Consortium la definisce come "il riconoscimento automatico di una persona sulla base di caratteristiche discriminative".

Un possibile approccio ai sistemi biometrici è il **pattern recognition**: due pattern sono simili se la distanza tra i loro vettori di feature è piccola. In biometria, le **classi sono gli individui**, e il pattern permette di distinguerli. Le domande centrali del corso, che ritornano in ogni capitolo, sono:
- Qual è una buona misura di distanza?
- Quali sono le feature migliori?
- Qual è il margine di differenza da accettare (soglia)?

Anche assumendo che ogni persona sia unica, restano tre problemi ingegneristici aperti:
1. Determinare le feature uniche in grado di identificare una persona;
2. Trovare tecniche affidabili per misurare tali feature;
3. Ideare algoritmi affidabili per riconoscere/classificare una persona sulla base delle misure ottenute.

L'accesso può essere di due tipi: **fisico** (stanze, edifici, aree) o **logico** (risorse elettroniche, dati critici, login). Il riconoscimento per l'autenticazione può basarsi su:
- **Qualcosa che si possiede** (una carta, un documento) — può essere rubato o copiato: il sistema autentica in realtà l'oggetto, non il proprietario;
- **Qualcosa che si conosce** (una password) — può essere indovinata o dimenticata;
- **Ciò che si è** (le caratteristiche biometriche).

---

## 2. Un po' di storia

- **1882 — Alphonse Bertillon** (capo del servizio identificazione della polizia di Parigi) introduce un sistema di misure corporee pensato per identificare univocamente i criminali, la **Bertillonage**. Raggruppava le persone in 1701 categorie (misure del corpo + colore di occhi e capelli), in modo che ogni nuova scheda dovesse essere confrontata solo con le altre della stessa categoria.
- **1887 — McClaughry** introduce l'antropometria negli Stati Uniti, traducendo il libro di Bertillon.
- **1903** — un caso di scambio di persona basato sulle schede Bertillon ne mette in luce i limiti.
- **Fine '800 — Francis Galton** critica il sistema di Bertillon da un punto di vista statistico. Nel **1892** introduce la nozione di **minuzia** e un primo semplice sistema di classificazione delle impronte digitali.
- **1893** — lo Home Office britannico riconosce che non esistono due individui con le stesse impronte digitali.
- **1900 — classificazione Galton-Henry**: è ancora oggi alla base dei sistemi di riconoscimento delle impronte digitali usati da molti dipartimenti di polizia nel mondo.

> I metodi biometrici non funzionano sempre, ma combinati con i metodi tradizionali possono migliorare il livello di sicurezza.

---

## 3. Architettura di un sistema biometrico

Un sistema biometrico ha due macro-fasi che condividono la stessa pipeline di moduli:

- **Enrollment (iscrizione)**: acquisizione ed elaborazione dei dati biometrici dell'utente per l'uso da parte del sistema nelle successive operazioni di autenticazione. Il risultato è un **template** salvato nella **gallery** (l'archivio dei template iscritti).
- **Recognition (riconoscimento)**: acquisizione ed elaborazione dei dati biometrici dell'utente al fine di rendere una decisione di autenticazione, basata sull'esito di un processo di matching tra template salvato e template corrente (verifica 1:1, identificazione 1:N).

Il **probe** è ogni template sottoposto per il riconoscimento; la **gallery** è l'insieme dei template appartenenti ai soggetti iscritti.

Un sistema biometrico è generalmente composto da **quattro moduli**:

| Modulo | Funzione |
|---|---|
| **Sensore** | Cattura i dati biometrici grezzi |
| **Feature Extraction** | Estrae un insieme di caratteristiche dai dati acquisiti; in fase di enrollment produce i template da salvare |
| **Matching** | Confronta le feature estratte con i template salvati, restituendo uno o più matching score |
| **Decisione** | Prende una decisione in base ai risultati del matching (soglia + eventuale policy) |

---

## 4. Classificazione di utenti, contesti e operazioni

### 4.1 Tipi di utente

| Dimensione | Categorie |
|---|---|
| Atteggiamento | **Cooperativo** (interessato al riconoscimento — un impostore qui cerca di essere riconosciuto come utente legale) vs **Non cooperativo** (indifferente o avverso — un impostore qui cerca di evitare il riconoscimento) |
| Popolazione | **Pubblico/Privato** (clienti vs dipendenti dell'ente); **Usato/Non usato** (in base alla frequenza d'uso); **Consapevole/Non consapevole** del processo di riconoscimento |

### 4.2 Tipi di contesto di acquisizione

- **Controllato**: le condizioni di cattura possono essere controllate, le distorsioni evitate, i template difettosi rifiutati e le acquisizioni ripetute.
- **Non controllato/parzialmente controllato**: le condizioni non possono essere controllate, il template può presentare vari livelli di distorsione e, sebbene i template difettosi possano essere rifiutati, l'acquisizione **non può essere ripetuta**.

### 4.3 Tipi di operazione di riconoscimento

- **Verifica (1:1)**:
  - l'utente dichiara un'identità (es. mostrando una carta d'identità o altro),
  - il sistema esegue un matching 1:1 per verificare l'identità dichiarata,
  - risultati possibili: **accetta** o **rifiuta**.
- **Identificazione (1:N)**:
  - nessuna dichiarazione da parte dell'utente,
  - il sistema deve determinare la corrispondenza con uno dei soggetti nella gallery tramite un matching 1:N,
  - risultati possibili: identità riconosciuta o identità non riconosciuta.
  - **Closed set**: tutte le probe appartengono a soggetti iscritti; l'unico errore possibile è restituire l'identità sbagliata.
  - **Open set**: il sistema determina se la probe appartiene a un soggetto nella gallery; alcune probe potrebbero non appartenere a nessun soggetto, quindi il sistema ha un'opzione di rifiuto; errore possibile: rifiutare una probe appartenente a un soggetto iscritto (oltre ad accettare erroneamente uno sconosciuto o restituire l'identità sbagliata).
  - **Watch list**: il sistema ha una lista di soggetti e verifica se la probe appartiene alla lista.
    - **White list**: i soggetti nella lista ottengono l'accesso.
    - **Black list**: i soggetti nella lista vengono rifiutati/segnalati.

> I capitoli su Performance e Reliability costruiscono l'intero vocabolario delle metriche (FAR/FRR, EER, CMC/DIR, …) esattamente su questa tricotomia verifica / closed set / open set — è la base da cui parte tutto il resto del corso.

---

## 5. I requisiti di un buon tratto biometrico

I requisiti fondamentali per un tratto biometrico sono:

1. **Universalità**: il tratto deve essere posseduto da qualunque persona.
2. **Unicità**: ogni coppia di persone dovrebbe risultare diversa secondo il tratto biometrico.
3. **Permanenza**: il tratto biometrico non dovrebbe cambiare nel tempo.
4. **Collectability**: il tratto biometrico deve essere misurabile tramite qualche sensore.
5. **Accettabilità**: le persone coinvolte non devono avere obiezioni a permettere la raccolta/misurazione del tratto.

Un tratto che soddisfa **tutti e cinque** i requisiti è detto **strong/hard biometric trait** (es. iride, impronte digitali, retina, vene di mano/dita, volto). Un tratto a cui manca anche solo uno di questi requisiti — più spesso l'unicità o la permanenza — è un **soft biometric trait** (es. colore dei capelli, altezza, forma del viso, andatura): da solo non basta a identificare univocamente una persona, ma è molto utile come **filtro preliminare economico** per restringere lo spazio di ricerca prima di applicare un tratto più lento ma più forte (cascading — vedi anche il capitolo sui sistemi multibiometrici).

### 5.1 Lo standard ANSI X9.84 (2003)

Lo standard **X9.84-2003** (requisiti minimi di sicurezza per un uso efficace della biometria) riconosce le seguenti tecniche, raggruppate come:

| Categoria | Tratti |
|---|---|
| **Fisiologici** | Impronte digitali, occhio (iride e retina), volto (foto, infrarosso), orecchio, geometria della mano (dita) |
| **Comportamentali** | Firma (statica e dinamica), dinamica di battitura (keystroke) |
| **Misti** | Voce |
| **Tracce biologiche** | DNA |

### 5.2 Tratti genotipici vs randotipici *(concetto ricorrente nella question bank)*

- **Tratti genotipici**: derivano direttamente dal DNA (quasi identici tra parenti/gemelli — es. la forma di base del viso, la geometria della mano).
- **Tratti randotipici**: nascono da processi di sviluppo caotici e non genetici (flusso del liquido amniotico, micro-pressioni) e differiscono anche tra gemelli identici o tra l'occhio destro e sinistro della stessa persona — es. la posizione delle minuzie delle impronte digitali, la texture trabecolare dell'iride, la ramificazione dei vasi retinici/della mano. **I tratti randotipici sono i più discriminativi**, perché sono unici anche al di là della componente genetica.

### 5.3 Biometria comportamentale via wearable (esempio: smartwatch)

Un tratto comportamentale come l'andatura o il gesto del polso può essere catturato da un dispositivo indossabile. Oltre al **giroscopio** (rilevamento della rotazione), sono utili:
- **Accelerometro**: misura l'accelerazione lineare del polso/braccio, catturando pattern personali di movimento (andatura, gesti tipici).
- **Sensore di temperatura**: rileva variazioni termiche tipiche di ciascun individuo.
- **Sensore di battito cardiaco (PPG — fotopletismografia)**: sfrutta la variabilità della frequenza cardiaca (HRV) come ulteriore tratto distintivo.

La combinazione di più sensori rende il riconoscimento comportamentale più stabile e affidabile (si veda anche il concetto di fusione multi-sensore nel capitolo sui sistemi multibiometrici).

---

## 6. Benefici e svantaggi dei tratti biometrici

I tratti biometrici costituiscono una metodologia di autenticazione "naturale", con pro e contro.

**Benefici:**
- Non possono essere persi, prestati, rubati o dimenticati;
- L'utente deve solo presentarsi di persona.

**Svantaggi:**
- Non garantiscono il 100% di accuratezza;
- Alcuni utenti non possono essere riconosciuti da alcune tecnologie (es. lavoratori manuali con impronte digitali danneggiate);
- Alcuni tratti possono cambiare nel tempo (es. il volto, il peso);
- Se un tratto viene "copiato" (spoofing), l'utente non può cambiarlo come farebbe con una password;
- I dispositivi biometrici possono essere inaffidabili in certe condizioni;
- Una foto del volto è più facile da rubare di una password → lo spoofing è un tema di ricerca centrale.

---

## RIEPILOGO DEL CAPITOLO

- La biometria è pattern recognition dove le classi sono le persone; le tre domande eterne sono: quali feature, quale distanza, quale soglia.
- Storia: Bertillon (Bertillonage, 1882) → Galton (minuzie, 1892) → classificazione Galton-Henry (1900), ancora alla base degli AFIS di polizia.
- Pipeline: Sensore → Feature Extraction → Matching → Decisione; l'enrollment costruisce la gallery, il riconoscimento confronta una probe con essa.
- Verifica = 1:1 contro un'identità dichiarata; identificazione = 1:N; closed set assume il soggetto iscritto, open set no (le watch list sono il caso open-set, non-cooperativo).
- Un buon tratto "hard" richiede universalità, unicità, permanenza, collectability, accettabilità; l'assenza anche di un solo requisito lo rende un tratto "soft", comunque utile come filtro veloce.
- Lo standard ANSI X9.84 classifica i tratti in fisiologici, comportamentali, misti e tracce biologiche.
- I tratti randotipici (non genetici) sono i più discriminativi, perché unici anche oltre la componente genetica condivisa dai parenti.
