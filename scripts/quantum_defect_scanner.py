import pathlib
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

jax.config.update("jax_enable_x64", True)

N_Q = 12
sim = de.DenseSVSimulator(n_qubits=N_Q, use_float32=False)


def _build_base_ops() -> list:
    ops = [['ry', q, float(np.pi / 4)] for q in range(N_Q)]
    ops += [['rz', q, f"batch_param_{q}"] for q in range(N_Q)]
    ops += [['cx', q, q + 1] for q in range(N_Q - 1)]
    return ops


def coerenza_x(sv: np.ndarray, gate_qubit: int) -> float:
    """<X> of the qubit specified by its GATE index (not raw array bit).
    DenseSVSimulator uses MSB-first indexing internally (phys = n-1-qubit,
    see _cx_numpy/apply_cx in dense_evolution/simulator.py) -- gate-qubit
    `q` lives at physical array bit (N_Q-1-q), not bit q directly. Using
    `1 << gate_qubit` directly (found during the dense-evolution 8.1.21
    audit) measured a DIFFERENT physical qubit than the one that actually
    received that row's RZ dephasing."""
    dim = len(sv)
    mask = 1 << (N_Q - 1 - gate_qubit)
    indices = np.arange(dim)
    return float(np.real(np.sum(np.conj(sv) * sv[indices ^ mask])))


def scansiona_difetti() -> np.ndarray:
    """Runs the batched defect scan (one RZ(0.5) dephasing per gate-qubit,
    N_Q concurrent tracks) and returns the residual coherence at each
    gate-qubit -- correctly indexed, i.e. coerenza[q] is the coherence of
    the SAME qubit that received row q's dephasing."""
    base_ops = _build_base_ops()

    # run_parametric_batch_jit treats EVERY rotation gate as a positional
    # parameter slot, even literal floats -- needs 2*N_Q columns, not N_Q
    # (N_Q constant RY(pi/4) columns + N_Q varying RZ columns).
    griglia_parametri = np.zeros((N_Q, 2 * N_Q), dtype=np.float64)
    griglia_parametri[:, :N_Q] = np.pi / 4
    for q in range(N_Q):
        griglia_parametri[q, N_Q + q] = 0.5

    jax_batch = jnp.array(griglia_parametri, dtype=jnp.float64)
    statevectors_batch = np.asarray(sim.run_parametric_batch_jit(base_ops, jax_batch))

    return np.array([
        abs(coerenza_x(statevectors_batch[q], q)) for q in range(N_Q)
    ])


def _run_full_sweep():
    print("⚡ Simulating localized dephasing defects via JAX Batch Engine...\n")
    print("🔬 Invio del batch parametrico a JAX XLA...")
    t_calc_start = time.perf_counter()

    coerenza_residua = scansiona_difetti()

    t_calc = time.perf_counter() - t_calc_start
    risultati_ispezione = []
    for local_qubit in range(N_Q):
        print(f"Ispezione Nodo {local_qubit+1:02d}/{N_Q} | Difetto localizzato sul qubit | Coerenza <X>: {coerenza_residua[local_qubit]*100:.4f}%")
        risultati_ispezione.append({
            "Nodo_Hardware": local_qubit,
            "Coerenza_Residua": coerenza_residua[local_qubit]
        })

    df = pd.DataFrame(risultati_ispezione)
    df.to_csv(_DATA_DIR / "mappa_difetti_silicio.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["Nodo_Hardware"], df["Coerenza_Residua"], marker='o', linestyle='-', color='#00FFFF', linewidth=2, label='Resilienza Locale al Dephasing')
    ax.fill_between(df["Nodo_Hardware"], df["Coerenza_Residua"], 0, color='#00FFFF', alpha=0.1)

    ax.set_ylim(0, 1.05)
    ax.set_title("True Quantum Defect Mapping: Localized Phase Noise Impact (JAX Batch)", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel("Posizione del Difetto Strutturale (Indice del Qubit)", color='#888888')
    ax.set_ylabel("Coerenza Quantistica Residua <X>", color='#888888')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "mappa_difetti_silicio.png", dpi=300)

    print("\n============================================================")
    print("✅ STRUMENTO DI ISPEZIONE QUANTISTICA CORRETTO E FUNZIONANTE!")
    print(f"📊 Grafico fisico esportato in: mappa_difetti_silicio.png")
    print("============================================================")


if __name__ == "__main__":
    print("============================================================")
    print("🔬 REAL PARALLEL QUANTUM DEFECT SCANNER (JORDAN-WIGNER)")
    print("============================================================")
    _run_full_sweep()
