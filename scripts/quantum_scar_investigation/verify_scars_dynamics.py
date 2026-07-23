# -*- coding: utf-8 -*-
"""
Verifica dell'ipotesi "quantum many-body scar" osservata in
frustrazione_quantistica.py sulla griglia 4x4 (16 qubit).

Tre controlli indipendenti, tutti sul sistema reale a 16 qubit
(nessuna diagonalizzazione esatta pesante):

  A) Entropia di entanglement nel tempo, non solo energia. Le scar
     mostrano crescita soppressa/oscillante; gli stati generici
     saturano verso il valore termico (quasi il massimo teorico).
  B) Convergenza Trotter: dt dimezzato/quadruplicato deve dare la
     stessa dinamica fisica, altrimenti l'effetto e' un artefatto
     dell'integrazione numerica, non fisica reale.
  C) Robustezza: pattern di segni alternativi + stato iniziale
     leggermente perturbato. Una vera scar sopravvive a piccole
     perturbazioni; un allineamento accidentale con lo stato di
     Neel collassa rapidamente.

Richiede: pip install dense-evolution[jax]
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dense_evolution.compiler import _apply_gate_fast_step

jax.config.update("jax_enable_x64", True)

N_QUBITS = 16
DIM = 2 ** N_QUBITS

# --- Geometria 4x4 (identica allo script originale) ---
legami_4x4 = []
for r in range(4):
    for c in range(3):
        legami_4x4.append((r * 4 + c, r * 4 + c + 1))
for r in range(3):
    for c in range(4):
        legami_4x4.append((r * 4 + c, (r + 1) * 4 + c))


def genera_diag_vettore(config_segni):
    dim = DIM
    diagonale = np.zeros(dim, dtype=np.float64)
    for stato_int in range(dim):
        spins = [1.0 if ((stato_int >> (N_QUBITS - 1 - i)) & 1) == 0 else -1.0
                 for i in range(N_QUBITS)]
        e = 0.0
        for idx, (u, v) in enumerate(legami_4x4):
            e += config_segni[idx] * spins[u] * spins[v]
        diagonale[stato_int] = e
    return jnp.array(diagonale, dtype=jnp.float64)


def genera_ops_passo(config_segni, dt, h=1.0):
    ops = []
    for idx, (u, v) in enumerate(legami_4x4):
        ops.append([22.0, float(u), float(v), 2.0 * config_segni[idx] * dt])
    for i in range(N_QUBITS):
        ops.append([9.0, float(i), 0.0, 2.0 * h * dt])
    return jnp.array(ops, dtype=jnp.float64)


def half_chain_entropy(sv, n_qubits=N_QUBITS):
    """Entropia di Von Neumann del taglio a meta reticolo (8 vs 8 qubit)."""
    dim_a = 2 ** (n_qubits // 2)
    mat = jnp.reshape(sv, (dim_a, dim_a))
    s = jnp.linalg.svd(mat, compute_uv=False)
    p = jnp.clip(s ** 2, 1e-16, None)
    p = p / jnp.sum(p)
    return float(-jnp.sum(p * jnp.log(p)))


def evolvi(config_segni, dt, n_steps, stato_iniziale):
    diag = genera_diag_vettore(config_segni)
    ops_passo = genera_ops_passo(config_segni, dt)

    stato = stato_iniziale
    energie, entropie = [], []
    for _ in range(n_steps):
        stato, _ = jax.lax.scan(_apply_gate_fast_step, stato, ops_passo)
        prob = jnp.abs(stato) ** 2
        energie.append(float(jnp.sum(prob * diag)))
        entropie.append(half_chain_entropy(stato))
    return np.array(energie), np.array(entropie)


NEEL_INDEX = 21845  # 0101...01, lo stesso stato iniziale dell'esperimento originale
stato_neel = jnp.zeros(DIM, dtype=jnp.complex128).at[NEEL_INDEX].set(1.0 + 0j)

configurazioni = {
    "Omogenea (Tutti -1)": [-1.0] * 24,
    "Frustrata Alternata": [-1.0 if i % 2 == 0 else 1.0 for i in range(24)],
}

# ============================================================
# A) ENTROPIA DI ENTANGLEMENT NEL TEMPO
# ============================================================
print("=== A) Entropia di entanglement (taglio 8|8 qubit) ===")
risultati_A = {}
for nome, segni in configurazioni.items():
    energie, entropie = evolvi(segni, dt=0.05, n_steps=120, stato_iniziale=stato_neel)
    risultati_A[nome] = (energie, entropie)
    print(f"{nome:<25} | S finale: {entropie[-1]:.4f} | S max: {entropie.max():.4f} "
          f"(max teorico ln(256)={np.log(256):.4f})")

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for nome, (energie, entropie) in risultati_A.items():
    axs[0].plot(energie, label=nome)
    axs[1].plot(entropie, label=nome)
axs[0].set_title("Energia <H>(t)")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[1].set_title("Entropia di entanglement S(t) [taglio 8|8]")
axs[1].axhline(np.log(256), color="gray", ls="--", label="Valore termico max ln(256)")
axs[1].legend()
axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("verifica_A_entropia.png", dpi=150)
print("Grafico salvato: verifica_A_entropia.png\n")

# ============================================================
# B) CONVERGENZA TROTTER
# ============================================================
print("=== B) Convergenza Trotter (Frustrata Alternata) ===")
segni_frustrati = configurazioni["Frustrata Alternata"]
T_TOTALE = 0.05 * 120  # stesso tempo fisico totale = 6.0

risultati_B = {}
for dt in [0.05, 0.025, 0.0125]:
    n_steps = int(round(T_TOTALE / dt))
    energie, entropie = evolvi(segni_frustrati, dt=dt, n_steps=n_steps, stato_iniziale=stato_neel)
    risultati_B[dt] = (energie, entropie)
    print(f"dt={dt:<7} | n_steps={n_steps:<5} | Energia finale: {energie[-1]:.6f} "
          f"| Entropia finale: {entropie[-1]:.6f}")

plt.figure(figsize=(8, 5))
for dt, (energie, _) in risultati_B.items():
    t_asse = np.linspace(0, T_TOTALE, len(energie))
    plt.plot(t_asse, energie, label=f"dt={dt}")
plt.title("Convergenza Trotter - Energia(t), Frustrata Alternata")
plt.xlabel("Tempo fisico t")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("verifica_B_trotter.png", dpi=150)
print("Grafico salvato: verifica_B_trotter.png\n")

# ============================================================
# C) ROBUSTEZZA: pattern multipli + stato iniziale perturbato
# ============================================================
print("=== C) Robustezza a pattern e perturbazioni ===")

rng = np.random.default_rng(0)

segni_checkerboard = []
for idx, (u, v) in enumerate(legami_4x4):
    r_u, c_u = u // 4, u % 4
    segni_checkerboard.append(-1.0 if (r_u + c_u) % 2 == 0 else 1.0)

pattern_extra = {
    "Alternata (shift 1)": [1.0 if i % 2 == 0 else -1.0 for i in range(24)],
    "Checkerboard geometrico (r+c)": segni_checkerboard,
}

for nome, segni in pattern_extra.items():
    energie, entropie = evolvi(segni, dt=0.05, n_steps=120, stato_iniziale=stato_neel)
    print(f"[pattern] {nome:<30} | S finale: {entropie[-1]:.4f}")

# Stato iniziale perturbato: piccola rotazione random su ogni qubit prima di evolvere
ops_perturbazione = jnp.array(
    [[9.0, float(i), 0.0, float(rng.normal(0, 0.1))] for i in range(N_QUBITS)],
    dtype=jnp.float64,
)
stato_perturbato, _ = jax.lax.scan(_apply_gate_fast_step, stato_neel, ops_perturbazione)
energie_p, entropie_p = evolvi(segni_frustrati, dt=0.05, n_steps=120,
                                stato_iniziale=stato_perturbato)
print(f"[stato perturbato] Frustrata Alternata | S finale: {entropie_p[-1]:.4f} "
      f"(confronta con {risultati_A['Frustrata Alternata'][1][-1]:.4f} non perturbato)")

print("\n=== COME LEGGERE I RISULTATI ===")
print("Indizio di vera scar: in A) la curva 'Frustrata Alternata' resta")
print("sotto la soglia termica mentre 'Omogenea' la satura; in B) l'effetto")
print("e' stabile al variare di dt; in C) sopravvive (attenuato ma presente)")
print("a pattern e perturbazioni simili.")
print("Indizio di artefatto/coincidenza: crolla con piccole perturbazioni,")
print("cambia sostanzialmente con dt, o e' specifico solo all'esatto pattern")
print("originale e a nessun altro simile.")
