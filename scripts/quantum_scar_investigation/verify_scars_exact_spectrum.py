# -*- coding: utf-8 -*-
"""
Diagnostica "da manuale" delle quantum many-body scar: spettro completo +
entropia di entanglement per autostato, su un reticolo ridotto (3x4 = 12
qubit, dimensione 4096) dove la diagonalizzazione esatta e' fattibile in
RAM/tempo ragionevoli su hardware consumer (a 16 qubit, 65536x65536, servirebbero
decine di GB e ore di calcolo con un solver denso: non fattibile qui).

Stessa regola di costruzione dei segni frustrati dello script originale
(alternanza per indice nella lista dei legami), sulla stessa geometria a
griglia con soli legami orizzontali/verticali.

Se il pattern produce vere scar: nello scatter Entropia vs Energia
compaiono alcuni autostati anomali a bassa entropia in mezzo alla "banda"
termica, e lo stato di Neel ha overlap concentrato proprio su quegli
autostati (colore acceso nello scatter).

Nessuna dipendenza da dense_evolution: costruzione diretta con numpy/scipy
sparse per restare leggeri in RAM, poi diagonalizzazione densa (dim 4096
e' piccola per scipy.linalg.eigh).

Richiede: pip install numpy scipy matplotlib
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Reticolo ridotto 3x4 (12 qubit, dim 4096) ---
ROWS, COLS = 3, 4
N = ROWS * COLS
DIM = 2 ** N


def idx(r, c):
    return r * COLS + c


legami = []
for r in range(ROWS):
    for c in range(COLS - 1):
        legami.append((idx(r, c), idx(r, c + 1)))
for r in range(ROWS - 1):
    for c in range(COLS):
        legami.append((idx(r, c), idx(r + 1, c)))

# Stessa regola "alternata per indice nella lista" dell'esperimento originale
segni_frustrati = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(legami))]
segni_omogenei = [-1.0] * len(legami)

# --- Operatori di Pauli sparsi, costruiti via kron, cache per qubit ---
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


print("Precalcolo operatori Z e X per ciascun qubit (sparsi)...")
Z_cache = {q: operatore_su_qubit(Zs, q) for q in range(N)}
X_cache = {q: operatore_su_qubit(Xs, q) for q in range(N)}


def costruisci_hamiltoniana(segni, h=1.0):
    H = sp.csr_matrix((DIM, DIM), dtype=np.complex128)
    for (u, v), j in zip(legami, segni):
        H = H + j * (Z_cache[u] @ Z_cache[v])
    for q in range(N):
        H = H + h * X_cache[q]
    return H.toarray()


def entropia_entanglement(stato, n=N):
    dim_a = 2 ** (n // 2)
    dim_b = 2 ** (n - n // 2)
    mat = stato.reshape(dim_a, dim_b)
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(s ** 2, 1e-16, None)
    p = p / p.sum()
    return -np.sum(p * np.log(p))


# Stato di Neel per 12 qubit: 010101010101 (alternato, MSB-first)
neel_bits = [0 if i % 2 == 0 else 1 for i in range(N)]
neel_idx = int("".join(map(str, neel_bits)), 2)
print(f"Indice stato di Neel ({N} qubit): {neel_idx}")

for nome, segni in [("Omogenea", segni_omogenei), ("Frustrata Alternata", segni_frustrati)]:
    print(f"\n--- Diagonalizzazione esatta: {nome} ({DIM}x{DIM}) ---")
    H = costruisci_hamiltoniana(segni)
    autovalori, autovettori = eigh(H)  # denso, fattibile a dim 4096

    entropie = np.array([entropia_entanglement(autovettori[:, i]) for i in range(DIM)])
    overlap = np.abs(autovettori[neel_idx, :]) ** 2  # |<autostato|Neel>|^2

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        autovalori, entropie, c=overlap, cmap="inferno", s=8,
        norm=mcolors.LogNorm(vmin=max(overlap.min(), 1e-6), vmax=overlap.max()),
    )
    plt.colorbar(sc, label="|<autostato | Neel>|^2 (scala log)")
    ax.set_xlabel("Energia autostato")
    ax.set_ylabel("Entropia di entanglement")
    ax.set_title(f"Spettro completo - {nome} ({N} qubit)")
    ax.axhline(np.log(2 ** (N // 2)), color="gray", ls="--", alpha=0.5,
               label="Entropia termica massima")
    ax.legend()
    plt.tight_layout()
    fname = f"verifica_D_spettro_{nome.split()[0].lower()}.png"
    plt.savefig(fname, dpi=150)
    print(f"Grafico salvato: {fname}")

    overlap_ordinato = np.sort(overlap)[::-1]
    top5 = overlap_ordinato[:5].sum()
    print(f"Overlap dello stato di Neel sui 5 autostati piu' rilevanti: {top5:.4f} "
          f"(vicino a 1 = overlap concentrato, tipico delle scar; "
          f"vicino a 0 = overlap diffuso, tipico della termalizzazione)")

print("\n=== COSA GUARDARE NEI GRAFICI ===")
print("- 'Frustrata Alternata': se vedi punti isolati a bassa entropia in mezzo")
print("  a una nuvola ad alta entropia (vicino a log(dim_A)), E quei punti hanno")
print("  colore acceso (overlap alto con lo stato di Neel) -> segnale classico")
print("  di quantum many-body scar.")
print("- Se l'overlap e' spalmato su tanti autostati o la nuvola e' omogenea")
print("  senza outlier -> l'oscillazione vista a 16 qubit e' piu' probabilmente")
print("  un effetto di dimensione finita o coincidenza, non una vera scar.")
print("- Confronta anche con 'Omogenea': se ANCHE quella mostra outlier simili,")
print("  l'effetto non e' specifico della frustrazione e va rivalutato.")
