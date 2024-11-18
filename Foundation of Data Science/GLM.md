# Generalized Linear Models (GLMs)

I **Generalized Linear Models (GLMs)** sono una classe flessibile di modelli statistici che generalizzano la regressione lineare per target che seguono distribuzioni appartenenti alla famiglia esponenziale. Forniscono un framework unificato per trattare problemi con target continui, binari, o discreti.

---

## Componenti principali dei GLMs

Un GLM è definito da tre componenti principali:

1. **Distribuzione dei target**:
   - I target $y$ appartengono alla **famiglia esponenziale delle distribuzioni**:
     $$
     p(y|\theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y, \phi)\right)
     $$
     Dove:
     - $\theta$: parametro naturale.
     - $b(\theta)$: funzione cumulativa logaritmica.
     - $\phi$: parametro di dispersione.
     - $c(y, \phi)$: funzione di normalizzazione.

     **Esempi comuni**:
     - **Normale**: $\theta = \mu$, $\phi = \sigma^2$.
     - **Bernoulliana**: $\theta = \log\frac{\pi}{1-\pi}$.
     - **Poissoniana**: $\theta = \log(\lambda)$.

2. **Funzione di link**:
   - Collega la media condizionale $\mu = \mathbb{E}[y|x]$ al predictor lineare $\eta$:
     $$
     g(\mu) = \eta = \mathbf{w}^\top \mathbf{x} + b
     $$
     Dove:
     - $g(\cdot)$: funzione di link.
     - $\eta$: combinazione lineare degli input.

     **Esempi comuni di funzioni di link**:
     - **Identità**: $g(\mu) = \mu$ (per la regressione lineare).
     - **Logit**: $g(\mu) = \log\frac{\mu}{1-\mu}$ (per la regressione logistica).
     - **Logaritmica**: $g(\mu) = \log(\mu)$ (per la regressione di Poisson).

3. **Funzione di varianza**:
   - Nei GLMs, la varianza di $y$ è una funzione della media:
     $$
     \text{Var}(y) = \phi v(\mu)
     $$
     Dove $v(\mu)$ è la **funzione di varianza** specifica della distribuzione.

---

## Formulazione matematica

1. **Probabilità condizionale**:
   $$
   p(y | \mathbf{x}, \mathbf{w}) = \exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y, \phi)\right)
   $$
   Dove $\theta = g^{-1}(\eta)$ e $\eta = \mathbf{w}^\top \mathbf{x}$.

2. **Stima dei parametri**:
   - La stima dei parametri $\mathbf{w}$ avviene massimizzando la log-verosimiglianza:
     $$
     \ell(\mathbf{w}) = \sum_{i=1}^N \left( \frac{y_i \theta_i - b(\theta_i)}{\phi} + c(y_i, \phi) \right)
     $$
     Dove $\theta_i = g^{-1}(\mathbf{w}^\top \mathbf{x}_i)$.
   - Si utilizza un metodo iterativo come il **reweighted least squares** o il **metodo di Newton-Raphson**.

3. **Predizione**:
   - Per un nuovo input $\mathbf{x}^*$, calcoliamo:
     $$
     \eta^* = \mathbf{w}^\top \mathbf{x}^* \quad \text{e} \quad \mu^* = g^{-1}(\eta^*)
     $$

---

## Esempi comuni di GLMs

1. **Regressione lineare**:
   - Distribuzione: Normale.
   - Funzione di link: Identità $g(\mu) = \mu$.
   - Modello: $y = \mathbf{w}^\top \mathbf{x} + \epsilon$, con $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

2. **Regressione logistica**:
   - Distribuzione: Bernoulliana.
   - Funzione di link: Logit $g(\mu) = \log\frac{\mu}{1-\mu}$.
   - Modello: $\pi = \sigma(\mathbf{w}^\top \mathbf{x})$, dove $\sigma(\cdot)$ è la sigmoide.

3. **Regressione di Poisson**:
   - Distribuzione: Poissoniana.
   - Funzione di link: Logaritmica $g(\mu) = \log(\mu)$.
   - Modello: $\lambda = \exp(\mathbf{w}^\top \mathbf{x})$.

---

## Proprietà dei GLMs

1. **Flessibilità**:
   - Possono modellare diversi tipi di dati scegliendo la distribuzione e una funzione di link appropriata.

2. **Proprietà analitiche**:
   - La linearità nei parametri consente una stima efficiente tramite metodi iterativi standard.

3. **Limitazioni**:
   - Assumono che i target siano indipendenti e distribuiti in modo identico (i.i.d.).
   - Scelte errate di distribuzione o funzione di link possono compromettere le prestazioni.

---

## Espansioni e variazioni

1. **GLMMs (Generalized Linear Mixed Models)**:
   - Estensione dei GLMs che include effetti randomici per dati gerarchici o correlati.

2. **Regularizzazione**:
   - Tecniche come Lasso, Ridge o Elastic Net possono essere utilizzate per evitare overfitting.

3. **Modelli non lineari**:
   - I GLMs possono essere combinati con funzioni di base per modellare relazioni non lineari.
