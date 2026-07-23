# -*- coding: utf-8 -*-
"""
Robustezza delle quantum many-body scar (modello PXP, validato in
verify_pxp_positive_control.py) sotto rumore realistico di dispositivo,
usando il vero modulo NoiseModel di dense_evolution (canale di Kraus
depolarizzante, simulazione a traiettorie quantistiche / quantum jump).

Idea: l'evoluzione coerente sotto H_PXP e' esatta (via autobase, nessuna
Trotterizzazione). Tra un intervallo e l'altro inietto rumore stocastico
reale con NoiseModel.apply_to_sv su OGNI qubit, poi medio molte traiettorie
indipendenti per ottenere l'osservabile fisico (fedelta' media = elemento
di matrice <Neel|rho(t)|Neel> dell'insieme).

Domanda: quanto rumore serve per distruggere i revival della scar? E'
un confronto quantitativo che (per quanto ne so) non e' lo scopo tipico
dei paper PXP standard (che studiano dinamica unitaria pulita) -- ed e'
esattamente cio' per cui NoiseModel e' stato costruito.

Richiede: pip install dense-evolution numpy scipy matplotlib
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from dense_evolution.registry import NoiseModel

N = 12
DIM = 2 ** N

# --- Costruzione H_PXP (identica a verify_pxp_positive_control.py) ---
I2s = sp.identity(2, format="csr", dtype=np.complex128)
Xs = sp.csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
Zs = sp.csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)


def operatore_su_qubit(op, q, n=N):
    mats = [I2s] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


print("Costruzione H_PXP e diagonalizzazione esatta...")
Z_cache = {q: operatore_su_qubit(Zs, q) for q in range(N)}
X_cache = {q: operatore_su_qubit(Xs, q) for q in range(N)}
Id = sp.identity(DIM, format="csr", dtype=np.complex128)
P_cache = {q: 0.5 * (Id + Z_cache[q]) for q in range(N)}

H_pxp = sp.csr_matrix((DIM, DIM), dtype=np.complex128)
for i in range(N):
    term = X_cache[i]
    if i > 0:
        term = P_cache[i - 1] @ term
    if i < N - 1:
        term = term @ P_cache[i + 1]
    H_pxp = H_pxp + term
H_pxp = H_pxp.toarray()

autovalori, autovettori = eigh(H_pxp)
autovettori_H = autovettori.conj().T
print(f"Fatto. Range energie: [{autovalori.min():.3f}, {autovalori.max():.3f}]\n")


def propaga_esatto(sv, dt):
    c = autovettori_H @ sv
    c *= np.exp(-1j * autovalori * dt)
    return autovettori @ c


def entropia_entanglement(sv, n=N):
    dim_a = 2 ** (n // 2)
    mat = sv.reshape(dim_a, dim_a)
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(s ** 2, 1e-16, None)
    p = p / p.sum()
    return -np.sum(p * np.log(p))


bits_neel = [0 if i % 2 == 0 else 1 for i in range(N)]
idx_neel = int("".join(map(str, bits_neel)), 2)
neel = np.zeros(DIM, dtype=np.complex128)
neel[idx_neel] = 1.0

DT_CHUNK = 0.2
N_CHUNK = 100  # tempo fisico totale = 20, come nel controllo positivo
N_TRAIETTORIE = 30
VALORI_P = [0.0, 0.005, 0.01, 0.02, 0.05]

risultati_fedelta = {}
risultati_entropia = {}

t0_tot = time.time()
for p in VALORI_P:
    t0 = time.time()
    fedelta_acc = np.zeros(N_CHUNK)
    entropia_acc = np.zeros(N_CHUNK)
    rng = np.random.default_rng(hash(("pxp_noise", p)) % (2 ** 32))

    for traiettoria in range(N_TRAIETTORIE):
        sv = neel.copy()
        for step in range(N_CHUNK):
            sv = propaga_esatto(sv, DT_CHUNK)
            if p > 0.0:
                sv = NoiseModel.apply_to_sv(sv, n=N, model='depolarizing', p=p, rng=rng)
            fedelta_acc[step] += np.abs(np.vdot(neel, sv)) ** 2
            entropia_acc[step] += entropia_entanglement(sv)

    fedelta_media = fedelta_acc / N_TRAIETTORIE
    entropia_media = entropia_acc / N_TRAIETTORIE
    risultati_fedelta[p] = fedelta_media
    risultati_entropia[p] = entropia_media

    dt = time.time() - t0
    print(f"p={p:<6} | {N_TRAIETTORIE} traiettorie | "
          f"fedelta' media (dopo transiente): {fedelta_media[25:].mean():.4f} | "
          f"picco max 2° revival: {fedelta_media[35:55].max():.4f} | {dt:.1f}s")

print(f"\nCompletato in {time.time() - t0_tot:.1f}s totali.\n")

tempi = np.arange(1, N_CHUNK + 1) * DT_CHUNK

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.cm.viridis
for i, p in enumerate(VALORI_P):
    colore = cmap(i / max(len(VALORI_P) - 1, 1))
    axs[0].plot(tempi, risultati_fedelta[p], label=f"p={p}", color=colore)
    axs[1].plot(tempi, risultati_entropia[p], label=f"p={p}", color=colore)
axs[0].set_title(f"Fedelta' media di revival ({N_TRAIETTORIE} traiettorie) vs rumore depolarizzante")
axs[0].set_xlabel("Tempo t")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[1].set_title("Entropia di entanglement media vs rumore")
axs[1].axhline(np.log(2 ** (N // 2)), color="gray", ls="--", label="Entropia termica massima")
axs[1].set_xlabel("Tempo t")
axs[1].legend()
axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("verifica_PXP_robustezza_rumore.png", dpi=150)
print("Grafico salvato: verifica_PXP_robustezza_rumore.png")
