# -*- coding: utf-8 -*-
"""
Ricerca sistematica di quantum many-body scar nella griglia frustrata,
usando la diagnostica dell'insieme diagonale (diagonal ensemble entropy):

Per un dato Hamiltoniano H = diagonalizzato in autostati |n> con energie E_n,
e per OGNI possibile stato iniziale |b> della base computazionale, l'entropia
di entanglement media di lungo periodo (insieme diagonale) e':

    S_DE(b) = Somma_n |<n|b>|^2 * S_entanglement(|n>)

Se S_DE(b) e' un outlier molto piu' basso della maggioranza degli altri stati
di base per lo stesso H, |b> e' un candidato scar genuino: la sua dinamica
resta a bassa entropia nel lungo periodo invece di termalizzare.

Questo e' molto piu' sistematico che provare a mano Neel/omogenea/checkerboard:
scandaglia TUTTI i 4096 stati di base automaticamente, per piu' pattern di
segni e piu' rapporti campo trasverso/accoppiamento (h/J).

Reticolo 3x4 (12 qubit, dim 4096) -- stessa dimensione gia' validata contro
il modello PXP in verify_pxp_positive_control.py.

Richiede: pip install numpy scipy
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

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

# --- Spin di ogni stato di base, vettorizzato (MSB-first, bit 0 -> +1) ---
indici = np.arange(DIM)
bit_shifts = np.arange(N - 1, -1, -1)
bits = (indici[:, None] >> bit_shifts) & 1          # shape (DIM, N)
spins = np.where(bits == 0, 1.0, -1.0)               # shape (DIM, N)


def diag_zz(segni):
    e = np.zeros(DIM, dtype=np.float64)
    for (u, v), j in zip(legami, segni):
        e += j * spins[:, u] * spins[:, v]
    return e


# --- Operatore X_totale (somma di X su ogni qubit), sparso, costruito una volta sola ---
I2s = sp.identity(2, format="csr", dtype=np.complex128)
Xs = sp.csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)


def operatore_su_qubit(op, q, n=N):
    mats = [I2s] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


print("Costruzione operatore X_totale (una volta sola, riusato per ogni combinazione)...")
X_sum = sp.csr_matrix((DIM, DIM), dtype=np.complex128)
for q in range(N):
    X_sum = X_sum + operatore_su_qubit(Xs, q)
X_sum = X_sum.toarray()
print("Fatto.\n")


def entropia_entanglement_batch(autovettori, n=N):
    """Entropia di Von Neumann (taglio a meta) per OGNI autovettore in colonna."""
    dim_a = 2 ** (n // 2)
    dim_b = 2 ** (n - n // 2)
    dim_tot = autovettori.shape[1]
    entropie = np.empty(dim_tot)
    for i in range(dim_tot):
        mat = autovettori[:, i].reshape(dim_a, dim_b)
        s = np.linalg.svd(mat, compute_uv=False)
        p = np.clip(s ** 2, 1e-16, None)
        p = p / p.sum()
        entropie[i] = -np.sum(p * np.log(p))
    return entropie


# --- Pattern di segni da testare ---
segni_checkerboard = []
for (u, v) in legami:
    r_u, c_u = u // COLS, u % COLS
    segni_checkerboard.append(-1.0 if (r_u + c_u) % 2 == 0 else 1.0)

rng = np.random.default_rng(42)
pattern_dict = {
    "Omogenea": [-1.0] * len(legami),
    "Indice alternato": [-1.0 if i % 2 == 0 else 1.0 for i in range(len(legami))],
    "Checkerboard geometrico": segni_checkerboard,
    "Random seed A": [float(x) for x in np.sign(rng.standard_normal(len(legami)))],
    "Random seed B": [float(x) for x in np.sign(rng.standard_normal(len(legami)))],
}

CAMPI_H = [0.3, 0.7, 1.0, 1.5, 2.5]

risultati = []
n_combo = len(pattern_dict) * len(CAMPI_H)
i_combo = 0
t0_tot = time.time()

print(f"Scansione di {n_combo} combinazioni (pattern x campo h)...\n")
for nome_pattern, segni in pattern_dict.items():
    diag_base = diag_zz(segni)
    for h in CAMPI_H:
        i_combo += 1
        t0 = time.time()
        H = np.diag(diag_base).astype(np.complex128) + h * X_sum
        autovalori, autovettori = eigh(H)
        entropie_auto = entropia_entanglement_batch(autovettori)

        probs = np.abs(autovettori) ** 2          # (DIM, DIM): [stato_base, autostato]
        S_DE = probs @ entropie_auto               # (DIM,): un valore per stato di base

        mediana = np.median(S_DE)
        std = np.std(S_DE)
        i_min = np.argmin(S_DE)
        outlier_score = (mediana - S_DE[i_min]) / (std + 1e-12)

        bits_min = "".join(str(int(b)) for b in bits[i_min])
        risultati.append({
            "pattern": nome_pattern, "h": h, "S_DE_min": S_DE[i_min],
            "mediana": mediana, "std": std, "outlier_score": outlier_score,
            "bitstring": bits_min, "indice": int(i_min),
        })
        dt = time.time() - t0
        print(f"[{i_combo:2d}/{n_combo}] {nome_pattern:<24} h={h:<4} | "
              f"S_DE min={S_DE[i_min]:.4f} mediana={mediana:.4f} | "
              f"outlier_score={outlier_score:.2f} | {dt:.1f}s")

print(f"\nScansione completata in {time.time() - t0_tot:.1f}s totali.\n")

risultati_ordinati = sorted(risultati, key=lambda r: -r["outlier_score"])
print("=== TOP 5 CANDIDATI SCAR (outlier_score piu' alto = piu' sospetto) ===")
for r in risultati_ordinati[:5]:
    print(f"pattern={r['pattern']:<24} h={r['h']:<4} | outlier_score={r['outlier_score']:.2f} | "
          f"S_DE_min={r['S_DE_min']:.4f} vs mediana={r['mediana']:.4f} | "
          f"stato={r['bitstring']} (indice {r['indice']})")

print("\n=== PER CONFRONTO: outlier_score del PXP validato prima ===")
print("(nello scan precedente, la torre di scar del PXP corrispondeva a")
print(" un overlap del 49% su 10 autostati -- un outlier_score tipicamente")
print(" ben sopra 3-4 in scansioni analoghe. Punteggi qui sotto ~1-2 indicano")
print(" fluttuazioni statistiche normali, non vere scar.)")
