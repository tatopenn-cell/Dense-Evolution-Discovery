# -*- coding: utf-8 -*-
"""
Controllo positivo: modello PXP (catena 1D con vincolo di blocco di Rydberg),
il sistema in cui le quantum many-body scar sono state scoperte per la prima
volta (Bernien et al. 2017, spiegazione teorica Turner et al. 2018).

Se le nostre metriche (fedelta' di revival, entropia di entanglement, spettro
con overlap) rilevano correttamente le scar QUI, dove sappiamo che esistono,
allora la pipeline di verifica usata su frustrazione_quantistica.py e'
affidabile e possiamo fidarci del suo verdetto negativo su quel caso.

H_PXP = Somma_i P_{i-1} X_i P_{i+1}   (catena aperta, P_i = (I+Z_i)/2
        proietta sul sito i "non eccitato", vieta eccitazioni adiacenti)

Stato di Neel |0101...> = nessuna coppia di eccitazioni adiacenti per
costruzione -> soddisfa il vincolo automaticamente, e' il candidato scar.
Stato "generico" di controllo: un'altra configurazione che soddisfa il
vincolo ma non e' periodica -> ci aspettiamo che termalizzi normalmente.

N=12 (dim 4096): stessa dimensione usata nel controllo sull'Ising, cosi'
il confronto tra i due spettri e' diretto.

Richiede: pip install numpy scipy matplotlib
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

N = 12
DIM = 2 ** N

# --- Operatori di Pauli sparsi per singolo qubit, cache ---
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


print(f"Costruzione operatori sparsi per catena PXP a N={N} siti (dim={DIM})...")
Z_cache = {q: operatore_su_qubit(Zs, q) for q in range(N)}
X_cache = {q: operatore_su_qubit(Xs, q) for q in range(N)}
Id = sp.identity(DIM, format="csr", dtype=np.complex128)
P_cache = {q: 0.5 * (Id + Z_cache[q]) for q in range(N)}  # proietta su |0> (non eccitato)

# --- Hamiltoniana PXP, catena aperta ---
H_pxp = sp.csr_matrix((DIM, DIM), dtype=np.complex128)
for i in range(N):
    term = X_cache[i]
    if i > 0:
        term = P_cache[i - 1] @ term
    if i < N - 1:
        term = term @ P_cache[i + 1]
    H_pxp = H_pxp + term
H_pxp = H_pxp.toarray()
print("Hamiltoniana PXP costruita. Diagonalizzazione esatta in corso...")

autovalori, autovettori = eigh(H_pxp)
print(f"Diagonalizzazione completata. Range energie: [{autovalori.min():.3f}, {autovalori.max():.3f}]")


def entropia_entanglement(stato, n=N):
    dim_a = 2 ** (n // 2)
    dim_b = 2 ** (n - n // 2)
    mat = stato.reshape(dim_a, dim_b)
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(s ** 2, 1e-16, None)
    p = p / p.sum()
    return -np.sum(p * np.log(p))


def stato_da_bitstring(bits):
    idx = int("".join(map(str, bits)), 2)
    v = np.zeros(DIM, dtype=np.complex128)
    v[idx] = 1.0
    return v, idx


# Stato di Neel (Z2): 0101...  -- candidato scar
bits_neel = [0 if i % 2 == 0 else 1 for i in range(N)]
stato_neel, idx_neel = stato_da_bitstring(bits_neel)

# Stato generico di controllo: soddisfa il vincolo (no due 1 adiacenti) ma non e' periodico
bits_generico = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
assert all(not (bits_generico[i] == 1 and bits_generico[i + 1] == 1) for i in range(N - 1)), \
    "lo stato generico deve rispettare il vincolo di blocco"
stato_generico, idx_generico = stato_da_bitstring(bits_generico)

print(f"Stato Neel:     indice {idx_neel}, bits {bits_neel}")
print(f"Stato generico: indice {idx_generico}, bits {bits_generico}")

# ============================================================
# EVOLUZIONE ESATTA (via autobase, nessuna Trotterizzazione: e' esatta per costruzione)
# ============================================================
def evolvi_esatto(stato0, autovalori, autovettori, tempi):
    c = autovettori.conj().T @ stato0            # coefficienti nella base di energia
    fasi = np.exp(-1j * np.outer(autovalori, tempi))  # (DIM, n_tempi)
    coeff_t = c[:, None] * fasi                  # (DIM, n_tempi)
    stati_t = autovettori @ coeff_t               # (DIM, n_tempi)
    fedelta = np.abs(np.sum(np.conj(c)[:, None] * coeff_t, axis=0)) ** 2
    entropie = np.array([entropia_entanglement(stati_t[:, k]) for k in range(len(tempi))])
    return fedelta, entropie


tempi = np.linspace(0, 20, 400)
print("\nEvoluzione esatta in corso (Neel + generico)...")
fedelta_neel, entropia_neel = evolvi_esatto(stato_neel, autovalori, autovettori, tempi)
fedelta_gen, entropia_gen = evolvi_esatto(stato_generico, autovalori, autovettori, tempi)

# ============================================================
# GRAFICO 1: Fedelta' di revival + entropia nel tempo
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(tempi, fedelta_neel, label="Neel |0101...> (candidato scar)", color="gold")
axs[0].plot(tempi, fedelta_gen, label="Stato generico (controllo)", color="steelblue")
axs[0].set_title("Fedelta' di revival |<psi(0)|psi(t)>|^2")
axs[0].set_xlabel("Tempo t")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(tempi, entropia_neel, label="Neel |0101...> (candidato scar)", color="gold")
axs[1].plot(tempi, entropia_gen, label="Stato generico (controllo)", color="steelblue")
axs[1].axhline(np.log(2 ** (N // 2)), color="gray", ls="--", label="Entropia termica massima")
axs[1].set_title("Entropia di entanglement S(t)")
axs[1].set_xlabel("Tempo t")
axs[1].legend()
axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("verifica_PXP_dinamica.png", dpi=150)
print("Grafico salvato: verifica_PXP_dinamica.png")

picchi = fedelta_neel[10:]  # scarta il transiente iniziale
n_revival = np.sum((picchi[1:-1] > picchi[:-2]) & (picchi[1:-1] > picchi[2:]) & (picchi[1:-1] > 0.3))
print(f"\nNumero di picchi di revival (fedelta' > 0.3) per lo stato di Neel: {n_revival}")
print(f"Fedelta' media stato Neel (dopo il transiente): {fedelta_neel[50:].mean():.4f}")
print(f"Fedelta' media stato generico (dopo il transiente): {fedelta_gen[50:].mean():.4f}")

# ============================================================
# GRAFICO 2: Spettro completo con overlap sullo stato di Neel
# ============================================================
overlap_neel = np.abs(autovettori[idx_neel, :]) ** 2
entropie_autostati = np.array([entropia_entanglement(autovettori[:, i]) for i in range(DIM)])

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(
    autovalori, entropie_autostati, c=overlap_neel, cmap="inferno", s=10,
    norm=mcolors.LogNorm(vmin=max(overlap_neel.min(), 1e-6), vmax=overlap_neel.max()),
)
plt.colorbar(sc, label="|<autostato | Neel>|^2 (scala log)")
ax.set_xlabel("Energia autostato")
ax.set_ylabel("Entropia di entanglement")
ax.set_title(f"Spettro completo PXP ({N} siti) - overlap con stato di Neel")
ax.axhline(np.log(2 ** (N // 2)), color="gray", ls="--", alpha=0.5, label="Entropia termica massima")
ax.legend()
plt.tight_layout()
plt.savefig("verifica_PXP_spettro.png", dpi=150)
print("Grafico salvato: verifica_PXP_spettro.png")

overlap_ordinato = np.sort(overlap_neel)[::-1]
top10 = overlap_ordinato[:10].sum()
print(f"\nOverlap dello stato di Neel sui 10 autostati piu' rilevanti: {top10:.4f}")

print("\n=== VERDETTO ATTESO (per confronto col caso Ising) ===")
print("Se la pipeline funziona, qui ci aspettiamo: fedelta' con revival")
print("marcati e periodici per Neel (non per il controllo generico), entropia")
print("soppressa/oscillante per Neel contro crescita monotona per il generico,")
print("e nello scatter una 'torre' di autostati a bassa entropia equispaziati")
print("in energia con overlap alto (colore acceso) sullo stato di Neel --")
print("il classico segnale di scar, assente nel caso Ising frustrato.")
