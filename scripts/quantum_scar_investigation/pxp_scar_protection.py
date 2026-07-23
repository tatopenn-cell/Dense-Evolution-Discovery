# -*- coding: utf-8 -*-
"""
Protezione delle scar PXP dal rumore: proiezione sulla "torre" di autostati
scar dopo ogni iniezione di rumore.

Dallo scan precedente (verify_pxp_positive_control.py) sappiamo che esiste
una torre isolata di ~N+1 autostati (per una catena aperta a N siti) a bassa
entropia, quasi equispaziati in energia, con overlap alto sullo stato di
Neel -- e' la colonna verticale visibile nello scatter dello spettro.

Idea: dopo ogni "colpo" di rumore (NoiseModel.apply_to_sv), proietto lo
stato sul sottospazio spannato da quella torre e rinormalizzo. E' un
limite teorico ideale (nessun protocollo fisico reale realizza una
proiezione su autostati a molti corpi senza tomografia completa) --
serve a rispondere alla domanda "vale la pena provare a proteggerle?"
PRIMA di preoccuparsi di come implementarlo su hardware vero.

Confronto a tre vie, stesso p di rumore:
  A) nessun rumore (limite superiore assoluto)
  B) rumore, nessuna protezione (quanto visto nello script precedente)
  C) rumore + proiezione sulla torre dopo ogni colpo

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

# --- Costruzione H_PXP ---
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

bits_neel = [0 if i % 2 == 0 else 1 for i in range(N)]
idx_neel = int("".join(map(str, bits_neel)), 2)
neel = np.zeros(DIM, dtype=np.complex128)
neel[idx_neel] = 1.0

# --- Identificazione della torre: top K autostati per overlap con Neel ---
overlap_neel = np.abs(autovettori[idx_neel, :]) ** 2
K = N + 1  # N+1 stati di torre attesi per una catena aperta (letteratura PXP)
indici_torre = np.argsort(overlap_neel)[::-1][:K]
peso_torre = overlap_neel[indici_torre].sum()
print(f"Torre: {K} autostati, peso totale sullo stato di Neel: {peso_torre:.4f}")
print(f"Energie della torre: {np.sort(autovalori[indici_torre])}\n")

V_torre = autovettori[:, indici_torre]           # (DIM, K)


def proietta_su_torre(sv):
    coeff = V_torre.conj().T @ sv                # (K,)
    sv_proj = V_torre @ coeff
    norm = np.linalg.norm(sv_proj)
    if norm < 1e-12:
        return sv_proj  # stato quasi interamente fuori dalla torre
    return sv_proj / norm


def propaga_esatto(sv, dt):
    c = autovettori_H @ sv
    c *= np.exp(-1j * autovalori * dt)
    return autovettori @ c


DT_CHUNK = 0.2
N_CHUNK = 100
N_TRAIETTORIE = 30
P_RUMORE = 0.01  # il valore che aveva gia' distrutto quasi del tutto i revival

configurazioni = {
    "A) Nessun rumore": {"p": 0.0, "protezione": False},
    "B) Rumore, senza protezione": {"p": P_RUMORE, "protezione": False},
    "C) Rumore + proiezione sulla torre": {"p": P_RUMORE, "protezione": True},
}

risultati = {}
t0_tot = time.time()
for nome, cfg in configurazioni.items():
    t0 = time.time()
    fedelta_acc = np.zeros(N_CHUNK)
    rng = np.random.default_rng(hash(("pxp_protect", nome)) % (2 ** 32))

    for traiettoria in range(N_TRAIETTORIE):
        sv = neel.copy()
        for step in range(N_CHUNK):
            sv = propaga_esatto(sv, DT_CHUNK)
            if cfg["p"] > 0.0:
                sv = NoiseModel.apply_to_sv(sv, n=N, model='depolarizing', p=cfg["p"], rng=rng)
                if cfg["protezione"]:
                    sv = proietta_su_torre(sv)
            fedelta_acc[step] += np.abs(np.vdot(neel, sv)) ** 2

    fedelta_media = fedelta_acc / N_TRAIETTORIE
    risultati[nome] = fedelta_media
    dt = time.time() - t0
    print(f"{nome:<35} | picco max 2 revival (finestra t=4-6): "
          f"{fedelta_media[20:30].max():.4f} | {dt:.1f}s")

print(f"\nCompletato in {time.time() - t0_tot:.1f}s totali.\n")

tempi = np.arange(1, N_CHUNK + 1) * DT_CHUNK
plt.figure(figsize=(9, 5.5))
colori = {"A) Nessun rumore": "gold", "B) Rumore, senza protezione": "steelblue",
          "C) Rumore + proiezione sulla torre": "limegreen"}
for nome, fedelta in risultati.items():
    plt.plot(tempi, fedelta, label=nome, color=colori[nome])
plt.title(f"Protezione delle scar PXP: proiezione sulla torre (p={P_RUMORE}, {N_TRAIETTORIE} traiettorie)")
plt.xlabel("Tempo t")
plt.ylabel("Fedelta' media |<Neel|psi(t)>|^2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("verifica_PXP_protezione_torre.png", dpi=150)
print("Grafico salvato: verifica_PXP_protezione_torre.png")
