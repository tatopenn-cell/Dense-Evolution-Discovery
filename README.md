# 🔬 Transizione di Fase Quantistica, Gradienti e Mitigazione dell'Errore (24 Qubit)

Questo repository raccoglie lo studio empirico, i dati reali e i protocolli di mitigazione eseguiti sul simulatore quantistico *Statevector* ad alte prestazioni **Dense Evolution (v8.0.4)**. Sfruttando la precisione a 64 bit (`complex128`) e la compilazione statica accelerata via JAX XLA, è stata mappata la fisica del modello di Ising in campo trasverso simulando uno spazio di Hilbert di **16.777.216 ampiezze complesse**.

---

## 📊 Struttura dell'Ecosistema

*   **`run_simulation.py`**: Script di benchmark e test dei canali stocastici di Kraus (*Depolarizing* e *Amplitude Damping*).
*   **`scan_ising.py` & `plot_ising.py`**: Pipeline di scansione fine e rendering grafico della transizione ferromagnetica ideale.
*   **`scan_noisy_ising.py`**: Simulazione della degradazione indotta dal rilassamento termico (*Amplitude Damping* NISQ).
*   **`zne_mitigation.py`**: Protocollo di estrapolazione a rumore zero (ZNE Richardson di 2° ordine) per la rimozione dell'errore.
*   **`vqe_gradient.py`**: Tracciamento del gradiente variazionale e isolamento sperimentale dei *Barren Plateaus*.
*   **`transizione_fase_ising.csv`**: Dataset grezzo con le osservabili estratte direttamente dai vettori di probabilità JAX.
*   **`report_quantistico_24qubit_REALE.log`**: Telemetria certificata delle metriche hardware e delle osservabili quantistiche microscopiche.

---

## 🔬 Evidenze Scientifiche Estratte

### 1. Transizione di Fase Quantistica
Validazione rigorosa dell'osservabile d'ordine $\langle H_{zz} \rangle$. Il sistema decade da $-1.0000$ (fase ferromagnetica con ordine perfetto a campo nullo $g = 0.0$) fino a $-0.6975$ a $g = 2.5$, mappando l'inizio della transizione verso il regime paramagnetico.

### 2. Impatto della Decoerenza Termica ($T_1$)
L'iniezione del canale di *Amplitude Damping* ($p = 0.04$) altera la traiettoria ideale del sistema. L'interazione dissipativa con l'ambiente accelera artificialmente la perdita di allineamento parallelo degli spin lungo l'asse quantistico $Z$.

### 3. Mitigazione dell'Errore via ZNE Richardson
L'applicazione del protocollo di estrapolazione lineare a rumore zero:
$$E(0) = 2E(\lambda_1) - E(\lambda_2)$$
ha permesso di ricostruire con successo il profilo energetico ideale partendo esclusivamente da punti campionati in regime rumoroso, neutralizzando efficacemente lo *shot noise* statistico.

### 4. Localizzazione Sperimentale di Barren Plateaus
Rilevamento di una massiccia "zona morta" a magnitudo del gradiente nulla ($\nabla_\theta \langle H_{zz} \rangle = 0.000000$) nell'ottimizzazione variazionale tra $\theta = 1.40$ e $\theta = 4.89$. Il fenomeno evidenzia empiricamente la diluizione dello spazio di Hilbert indotta dall'alto livello di entanglement su strutture NISQ profonde.

---

## ⚙️ Requisiti di Sistema e Riproducibilità

*   **Stack Software**: Python 3.9+ | JAX (XLA Hardware Engine) | NumPy | Pandas | Matplotlib | psutil.
*   **RAM Footprint**: $\approx 256.0$ MB allocati per singolo Statevector a 24 qubit in doppia precisione IEEE 754.
