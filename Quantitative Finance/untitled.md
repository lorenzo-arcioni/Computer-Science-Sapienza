# Ottimizzazione del Portafoglio

## **Trading Automatico e Allocazione del Capitale**

Quando facciamo trading con più strategie o deteniamo posizioni in diversi strumenti finanziari all'interno di un portafoglio, la modalità con cui distribuiamo il capitale tra di essi è cruciale tanto quanto la performance delle singole strategie.  

L'ottimizzazione del portafoglio è un tema centrale nella finanza quantitativa e ha ricevuto ampio studio nella ricerca accademica. Tuttavia, vi sono alcune sottigliezze che meritano di essere evidenziate.

### **Rendimento Atteso e Variazione del Portafoglio**

Prima di addentrarci nei dettagli dell'ottimizzazione, è importante distinguere tra:
1. **Massimizzazione del rendimento in un singolo periodo** (giorno, mese, anno, ecc.).
2. **Massimizzazione del rendimento composto su un orizzonte temporale infinito (CAGR).**

La prima è trattata dal noto metodo di **ottimizzazione del portafoglio di Markowitz**, il cui obiettivo è:
- **Massimizzare il rendimento atteso** soggetto a una varianza costante dei rendimenti.
- Oppure, in modo equivalente e più comune, **minimizzare la varianza dei rendimenti** soggetta a un rendimento atteso costante.

Per calcolare il rendimento atteso e la varianza su un periodo, supponiamo che il nostro portafoglio contenga **N strumenti finanziari** (azioni, obbligazioni, futures, ecc.) o **strategie di trading** (mean-reverting, momentum, event-driven, opzioni, ecc.). Da ora in poi, useremo il termine generico **"strumenti"** per riferirci a questi componenti del portafoglio.

Ogni strumento $i$ ha un rendimento atteso $m_i$ e ipotizziamo che questi rendimenti abbiano una covarianza $C_{i,j}$.

### **Rendimento Netto vs. Rendimento Logaritmico**

Quando parliamo di "rendimento", possiamo riferirci a due tipi distinti:

- **Rendimento netto**:
  $$
  \text{Rendimento netto}(t) = \frac{\text{Prezzo}(t)}{\text{Prezzo}(t-1)} - 1
  $$

- **Rendimento logaritmico**:
  $$
  \text{Rendimento log}(t) = \log(\text{Prezzo}(t)) - \log(\text{Prezzo}(t-1))
  $$

A livello teorico, i rendimenti netti **non seguono mai una distribuzione gaussiana**, mentre i rendimenti logaritmici possono farlo. Questa distinzione è fondamentale in finanza quantitativa.

Esiste una relazione interessante tra il rendimento logaritmico medio $\mu$ e il rendimento netto medio $m$ di una serie di prezzi, se i rendimenti logaritmici seguono una distribuzione normale:

$$
\mu \approx m - \frac{s^2}{2}
$$

(Equazione 1.1)

dove $s$ è la deviazione standard del rendimento netto. Questa equazione diventa esatta quando i periodi vengono suddivisi in intervalli sempre più piccoli, avvicinandosi al tempo continuo. Questo risultato deriva dal **Lemma di Itô**, una formula chiave nella finanza matematica utilizzata nel modello di **Black-Scholes** per la valutazione delle opzioni.

---

## **Dimostrazione Numerica: Equazione 1.1**

In questo esempio, dimostreremo numericamente che l'**equazione 1.1** è approssimativamente corretta quando i rendimenti logaritmici hanno una distribuzione gaussiana e che, suddividendo un periodo in sempre più sottoperiodi, l'approssimazione diventa esatta.

Supponiamo che il rendimento logaritmico su un certo periodo abbia:
- **Media** $\mu = 100\%$
- **Deviazione standard** $s = 200\%$

Useremo la funzione `numpy.random.randn` per generare $N$ sottoperiodi di rendimenti logaritmici $r_i$, con:

$$
\mu = \frac{100\%}{N}, \quad s = \frac{200\%}{\sqrt{N}}
$$

Proviamo con $N = 100, 10.000, 1.000.000$.

### **Codice Python**

```python
import numpy as np

N_values = [100, 10000, 1000000]

for N in N_values:
    r = (100/N) + (200/np.sqrt(N)) * np.random.randn(N)
    R = np.exp(r) - 1
    mu = np.mean(r)
    sigma = np.std(r)
    m = np.mean(R)
    s = np.std(R)
    print(f'N={N}: m={m:.6f}, mu={mu:.6f}, s={s:.6f}, m - s^2/2={m - s**2/2:.6f}')
```

### **Risultati della Simulazione**

Osserviamo che **$m - s^2/2 \to \mu$ al tendere di $N$ all'infinito**, come previsto dall'equazione 1.1.

---

## **Frontiera Efficiente e Ottimizzazione del Portafoglio**

Se denotiamo con $F_i$ il capitale (o leva) allocato a uno strumento $i$, allora:

- Il **rendimento atteso** del portafoglio su un periodo è dato da:

  $$
  F^T M
  $$

- La **varianza attesa** del portafoglio è:

  $$
  F^T C F
  $$

dove $C$ è la matrice di covarianza $C_{i,j}$.

Variando $F$, possiamo minimizzare la varianza attesa per un livello fissato di rendimento atteso, utilizzando **programmazione quadratica numerica**. Il risultato di questa ottimizzazione può essere rappresentato nel **grafico della frontiera efficiente**, che mostra il rendimento atteso del portafoglio rispetto alla sua varianza minima attesa.

### **Il Portafoglio di Tangente e il Rapporto di Sharpe**

Secondo Markowitz, la soluzione ottimale si trova scegliendo il **portafoglio di tangente**, che massimizza il **Rapporto di Sharpe**:

$$
S = \frac{\text{Rendimento Atteso}}{\text{Deviazione Standard}}
$$

Questo portafoglio si trova nel punto di tangenza tra la **frontiera efficiente** e la linea che parte dall'asset privo di rischio.

---

## **Conclusioni**

L'ottimizzazione del portafoglio è uno strumento essenziale nella finanza quantitativa. La comprensione delle differenze tra rendimento netto e logaritmico, nonché il concetto di frontiera efficiente, sono fondamentali per sviluppare strategie di investimento robuste.