\# 🔬 Transizione di Fase Quantistica, Gradienti e Mitigazione dell'Errore (24 Qubit)



Questo repository raccoglie lo studio empirico, i dati reali e i protocolli di mitigazione eseguiti sul simulatore quantistico \*Statevector\* ad alte prestazioni \*\*Dense Evolution (v8.0.4)\*\*. Sfruttando la precisione a 64 bit (`complex128`) e la compilazione statica accelerata via JAX XLA, è stata mappata la fisica del modello di Ising in campo trasverso simulando uno spazio di Hilbert di \*\*16.777.216 ampiezze complesse\*\*.



\## 📊 Contenuto del Repository



\*   `run\_simulation.py`: Script di inizializzazione e test dei canali stocastici di Kraus (Depolarizing e Amplitude Damping).

\*   `scan\_ising.py` \& `plot\_ising.py`: Scansione fine e plotting della transizione ferromagnetica ideale.

\*   `scan\_noisy\_ising.py`: Simulazione della degradazione indotta dal rumore termico (\*Amplitude Damping\* NISQ).

\*   `zne\_mitigation.py`: Protocollo di mitigazione dell'errore tramite estrapolazione a rumore zero (ZNE Richardson di 2° ordine).

\*   `vqe\_gradient.py`: Tracciamento del gradiente variazionale e isolamento sperimentale dei \*Barren Plateaus\*.

\*   `transizione\_fase\_ising.csv`: Dati grezzi estratti direttamente dai vettori di probabilità JAX.

\*   `report\_quantistico\_24qubit\_REALE.log`: Log certificato delle metriche hardware e osservabili quantistiche microscopiche.



\## 🔬 Risultati Scientifici Evidenziati



1\.  \*\*Transizione di Fase Quantistica\*\*: Validazione dell'osservabile d'ordine $\\langle H\_{zz} \\rangle$ che decade da $-1.0000$ (ordine ferromagnetico perfetto a campo nullo $g=0$) fino a $-0.6975$ a $g=2.5$.

2\.  \*\*Impatto della Decoerenza Termica ($T\_1$)\*\*: Il canale di \*Amplitude Damping\* ($p=0.04$) distorce la curva ideale, anticipando artificialmente la perdita dell'allineamento degli spin a causa dell'interazione dissipativa con l'ambiente.

3\.  \*\*Mitigazione ZNE Riuscita\*\*: L'algoritmo di Richardson $E(0) = 2E(\\lambda\_1) - E(\\lambda\_2)$ ha ricostruito con successo il profilo ideale a rumore zero partendo esclusivamente da campionamenti rumorosi, nonostante le fluttuazioni statistiche (\*shot noise\*).

4\.  \*\*Isolamento di Barren Plateau\*\*: Rilevamento sperimentale di una massiccia "zona morta" a gradiente nullo ($0.000000$) nell'ottimizzazione variazionale tra $\\theta = 1.40$ e $\\theta = 4.89$, causata dalla diluizione dello spazio di Hilbert all'aumentare dell'entanglement.



\## ⚙️ Requisiti di Sistema e Riproducibilità



\*   \*\*Framework richiesti\*\*: Python 3.9+, JAX (XLA Hardware Engine), NumPy, Pandas, Matplotlib, psutil.

\*   \*\*RAM Footprint\*\*: \~256.0 MB per singola allocazione dello Statevector a 24 qubit.



