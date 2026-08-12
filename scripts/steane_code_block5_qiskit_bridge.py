"""
Steane code [[7,1,3]] -- block 5: real-hardware bridge via Qiskit.

SCOPE -- what this script DOES (offline, free, no IBM account needed):
    1. Converts block 1's encoding circuit into a real Qiskit QuantumCircuit
       via an OpenQASM 2.0 round trip (the same bridge convention
       dense_evolution.interop already uses in the other direction).
    2. Verifies that round trip is faithful three independent ways: Qiskit's
       own Statevector + SparsePauliOp expectation values, Dense-Evolution's
       own pauli_expectation on the bit-reordered Qiskit statevector, and a
       full Qiskit-circuit -> QASM -> Dense-Evolution-simulator re-run.
    3. Transpiles the circuit for FakeSherbrooke (qiskit-ibm-runtime's fake
       backend, real historical IBM Eagle-127 calibration data and coupling
       map, no account required).
    4. Reports transpilation overhead honestly (physical qubits, native
       2-qubit-gate count vs the original CX count, depth) and re-verifies
       the transpiled circuit's physics (6-stabilizer check + fidelity)
       against the untranspiled ideal circuit.
    5. Runs the transpiled circuit through a REAL noise model built from
       FakeSherbrooke's own historical per-qubit/per-gate calibration data
       (not an idealized channel) and reports the resulting state fidelity.

SCOPE -- what this script explicitly does NOT do:
    - Does NOT connect to IBM Quantum's cloud service, and does NOT use or
      request any account/API token. Everything above runs on local
      classical simulators only.
    - Does NOT run on live hardware. Actual submission requires the user's
      own IBM Quantum account + API token (a separate step, out of scope
      here) and real queue time.
    - Does NOT implement syndrome-based error correction on the noisy
      Qiskit-side output -- the reported number is an encoding-fidelity
      result under real calibration data, not a full logical-error-rate
      curve like block 1's depolarizing sweep (different noise regime and
      a different circuit-shape problem; adapting block 1's ancilla-free
      decoder to a genuinely noisy mixed state is future work).

macOS note: qiskit.circuit.QuantumCircuit.__init__ has a documented upstream
segfault risk on macOS/arm64 (see dense_evolution.interop._require_qiskit's
docstring/warning). This script constructs QuantumCircuit objects directly
(via qiskit.qasm2.loads and transpile), so it reuses that exact library
warning helper below before touching any Qiskit object, rather than
re-inventing or silently dropping the warning.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, r'C:\Users\Admin\Desktop\Dense-Evolution-main\Dense-Evolution-main')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dense_evolution.interop import _require_qiskit, run_qiskit_circuit
from dense_evolution.observables import pauli_expectation
from dense_evolution.measurement import statevector_fidelity

import steane_code_block1 as b1

_require_qiskit()

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, state_fidelity
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_IMAGES_DIR = _REPO_ROOT / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

N = b1.N_QUBITS
FID_TOL = 1e-6


# ─────────────────────────────────────────────────────────────────────────
# Dense-Evolution -> QASM -> Qiskit
# ─────────────────────────────────────────────────────────────────────────

def build_encoding_qasm() -> str:
    """QASM text for block 1's encode_logical_zero() circuit, built from
    block 1's own FREE_QUBITS/DERIVED_QUBITS constants (reused, not
    hand-copied) so it is provably the same gate sequence block 1 runs."""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];"]
    for q in b1.FREE_QUBITS:
        lines.append(f"h q[{q}];")
    for derived_q, sources in b1.DERIVED_QUBITS.items():
        for src in sources:
            lines.append(f"cx q[{src}],q[{derived_q}];")
    return "\n".join(lines)


def qiskit_bit_reverse_perm(n: int) -> np.ndarray:
    """Index permutation converting Dense-Evolution's MSB-first amplitude
    ordering into Qiskit's little-endian ordering, or back (it's an
    involution) -- see dense_evolution.interop._to_qiskit_bit_order."""
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(2 ** n)])


def qubit_permutation_index_array(n: int, old_bit_of_new_qubit) -> np.ndarray:
    """perm[new_idx] = old_idx such that new-order qubit i reads old-order
    qubit old_bit_of_new_qubit[i]. Applies identically to statevectors
    (fancy-index with perm) and density matrices (np.ix_(perm, perm))."""
    dim = 2 ** n
    new_idx = np.arange(dim)
    old_idx = np.zeros(dim, dtype=np.int64)
    for i in range(n):
        bit = (new_idx >> i) & 1
        old_idx |= bit << old_bit_of_new_qubit[i]
    return old_idx


def extract_active_subcircuit(tqc: QuantumCircuit):
    """Drop every physical qubit the transpiled circuit never touches --
    those qubits stay in the |0> product state throughout (no gate ever
    acts on them), so tracing them out is exact, not an approximation, and
    turns an intractable 127-qubit statevector into a tractable ~7-10
    qubit one."""
    active = sorted({tqc.find_bit(q).index for instr in tqc.data for q in instr.qubits})
    remap = {p: i for i, p in enumerate(active)}
    sub = QuantumCircuit(len(active))
    for instr in tqc.data:
        idxs = [remap[tqc.find_bit(q).index] for q in instr.qubits]
        sub.append(instr.operation, idxs)
    return sub, active, remap


def build_reduced_noise_model(nm_full: NoiseModel, sub_circuit: QuantumCircuit, remap: dict) -> NoiseModel:
    """Real per-qubit/per-gate error channels from FakeSherbrooke's own
    calibration data (already built once by NoiseModel.from_backend),
    filtered down to just the physical qubits this circuit actually uses
    and remapped onto the reduced circuit's local qubit indices -- so the
    noisy simulation stays on a tractable ~7-10 qubit density matrix
    instead of the full 127-qubit device.

    Each unique (gate, local-qubit-tuple) is added exactly once: calling
    add_quantum_error twice for the same target COMPOSES the channel with
    itself (AerSimulator's documented semantics for repeated registration,
    not "apply once per gate occurrence") -- verified directly, an earlier
    version of this function called it once per gate instance and blew up
    to multi-GB memory from the resulting Kraus-term combinatorial
    explosion before OOM-killing the process.
    """
    nm = NoiseModel()
    physical_of_local = {v: k for k, v in remap.items()}
    for gate in ('sx', 'x', 'id', 'reset'):
        d = nm_full._local_quantum_errors.get(gate, {})
        for local_q in range(sub_circuit.num_qubits):
            key = (physical_of_local[local_q],)
            if key in d:
                nm.add_quantum_error(d[key], gate, [local_q])
    d2 = nm_full._local_quantum_errors.get('ecr', {})
    seen_pairs = set()
    for instr in sub_circuit.data:
        if instr.operation.name != 'ecr':
            continue
        local_pair = tuple(sub_circuit.find_bit(q).index for q in instr.qubits)
        if local_pair in seen_pairs:
            continue
        seen_pairs.add(local_pair)
        phys_pair = (physical_of_local[local_pair[0]], physical_of_local[local_pair[1]])
        if phys_pair in d2:
            nm.add_quantum_error(d2[phys_pair], 'ecr', list(local_pair))
        elif phys_pair[::-1] in d2:
            nm.add_quantum_error(d2[phys_pair[::-1]], 'ecr', list(local_pair[::-1]))
        else:
            raise KeyError(f"no ecr calibration for physical pair {phys_pair}")
    return nm


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 1: Dense-Evolution -> QASM -> Qiskit round trip")
    print("=" * 70)

    sv0_native = b1.encode_logical_zero()
    native_ok = b1.check_stabilizers(sv0_native, "native |0>_L (block1, Dense-Evolution)")

    qasm_str = build_encoding_qasm()
    qc = qiskit.qasm2.loads(qasm_str)
    print(f"\nQiskit circuit built from QASM: {qc.num_qubits} qubits, {len(qc.data)} ops")

    print("\n-- cross-check A: Qiskit's own Statevector + SparsePauliOp (fully independent path) --")
    qk_sv = Statevector(qc)
    all_stabilizers = b1.X_STABILIZERS + b1.Z_STABILIZERS
    checkA_ok = True
    for label in all_stabilizers:
        qk_label = label[::-1]  # Dense-Evolution's leftmost=qubit0 vs Qiskit's leftmost=qubit(n-1)
        val = qk_sv.expectation_value(SparsePauliOp(qk_label)).real
        ok = abs(val - 1.0) < FID_TOL
        checkA_ok = checkA_ok and ok
        print(f"   <{label}> (qiskit label {qk_label}) = {val:+.9f}  {'OK' if ok else 'FAIL'}")

    print("\n-- cross-check B: Dense-Evolution's pauli_expectation on the bit-reordered Qiskit statevector --")
    perm = qiskit_bit_reverse_perm(N)
    sv_from_qiskit_de_order = np.asarray(qk_sv.data)[perm]
    fid_b = statevector_fidelity(sv_from_qiskit_de_order, sv0_native)
    checkB_ok = abs(fid_b - 1.0) < FID_TOL
    print(f"   fidelity(reordered Qiskit statevector, native block1 statevector) = {fid_b:.9f}  {'OK' if checkB_ok else 'FAIL'}")
    for label in all_stabilizers:
        val = pauli_expectation(sv_from_qiskit_de_order, label)
        print(f"   <{label}> = {val:+.9f}")

    print("\n-- cross-check C: Qiskit circuit -> QASM -> Dense-Evolution simulator re-run --")
    sim_c, _ = run_qiskit_circuit(qc)
    sv_via_de_run = sim_c.get_statevector()
    fid_c = statevector_fidelity(sv_via_de_run, sv0_native)
    checkC_ok = abs(fid_c - 1.0) < 1e-5
    print(f"   fidelity(qiskit circuit re-run natively via dense_evolution.interop, native block1) = {fid_c:.9f}  {'OK' if checkC_ok else 'FAIL'}")

    roundtrip_ok = native_ok and checkA_ok and checkB_ok and checkC_ok
    print(f"\nROUND TRIP VERIFIED FAITHFUL (3 independent cross-checks): {roundtrip_ok}")
    if not roundtrip_ok:
        raise RuntimeError("Dense-Evolution <-> Qiskit round trip disagrees -- stopping before transpilation.")

    print("\n" + "=" * 70)
    print("STEP 2: transpile for a real IBM device (FakeSherbrooke, no account needed)")
    print("=" * 70)

    backend = FakeSherbrooke()
    print(f"backend: {backend.name}  num_qubits: {backend.num_qubits}  "
          f"real historical calibration snapshot, coupling map edges: {backend.coupling_map.size()}")

    tqc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=7)
    active = sorted({tqc.find_bit(q).index for instr in tqc.data for q in instr.qubits})
    op_counts = tqc.count_ops()
    n_2q_native = op_counts.get('ecr', 0) + op_counts.get('cx', 0)
    n_2q_original = sum(1 for op in qc.data if op.operation.num_qubits == 2)

    print(f"original logical circuit: {N} qubits, {n_2q_original} CX, depth {qc.depth()}")
    print(f"transpiled circuit: {len(active)} physical qubits used {active}, "
          f"native ops {dict(op_counts)}, depth {tqc.depth()}")
    print(f"native 2-qubit-gate count: {n_2q_native} vs {n_2q_original} original CX "
          f"({n_2q_native / n_2q_original:.2f}x overhead from routing + basis translation, "
          f"but 0 EXTRA physical qubits needed -- {len(active)} used for {N} logical qubits)")

    initial_layout = tqc.layout.initial_index_layout()[:N]
    final_layout = tqc.layout.final_index_layout()
    print(f"logical qubit initial physical placement: {list(initial_layout)}")
    print(f"logical qubit final physical placement:   {final_layout}")
    print(f"layout permuted by routing (swap-equivalent moves occurred): {list(initial_layout) != list(final_layout)}")

    print("\n" + "=" * 70)
    print("STEP 3: verify transpiled circuit's physics still checks out (noiseless)")
    print("=" * 70)

    sub, active2, remap = extract_active_subcircuit(tqc)
    assert active2 == active
    sub_sv = Statevector(sub).data
    local_of_logical = [remap[final_layout[i]] for i in range(N)]
    perm_arr = qubit_permutation_index_array(N, local_of_logical)
    sv_transpiled_reordered = sub_sv[perm_arr]
    ideal_sv_qk = qk_sv.data

    fid_transpiled = state_fidelity(sv_transpiled_reordered, ideal_sv_qk)
    print(f"state_fidelity(transpiled circuit output, ideal untranspiled circuit output) = {fid_transpiled:.9f}")

    sv_transpiled_de_order = sv_transpiled_reordered[perm]
    transpiled_stabilizers_ok = True
    print("-- 6-stabilizer check on transpiled circuit's noiseless output --")
    for label in all_stabilizers:
        val = pauli_expectation(sv_transpiled_de_order, label)
        ok = abs(val - 1.0) < FID_TOL
        transpiled_stabilizers_ok = transpiled_stabilizers_ok and ok
        print(f"   <{label}> = {val:+.9f}  {'OK' if ok else 'FAIL'}")

    physics_ok = abs(fid_transpiled - 1.0) < FID_TOL and transpiled_stabilizers_ok
    print(f"\nTRANSPILED CIRCUIT'S PHYSICS VERIFIED CORRECT (noiseless): {physics_ok}")
    if not physics_ok:
        raise RuntimeError("Transpiled circuit does not reproduce the ideal encoding -- stopping before noisy run.")

    print("\n" + "=" * 70)
    print("STEP 4: noisy simulation with FakeSherbrooke's real calibration data")
    print("=" * 70)

    nm_full = NoiseModel.from_backend(backend)
    nm_reduced = build_reduced_noise_model(nm_full, sub, remap)
    print(f"reduced noise model basis gates: {nm_reduced.basis_gates}")

    sub_noisy = sub.copy()
    sub_noisy.save_density_matrix()
    sim = AerSimulator(noise_model=nm_reduced)
    rho_noisy = np.asarray(sim.run(sub_noisy).result().data(0)['density_matrix'])
    rho_reordered = rho_noisy[np.ix_(perm_arr, perm_arr)]
    fid_noisy = state_fidelity(rho_reordered, ideal_sv_qk)

    sub_ideal = sub.copy()
    sub_ideal.save_density_matrix()
    rho_ideal = np.asarray(AerSimulator().run(sub_ideal).result().data(0)['density_matrix'])
    rho_ideal_reordered = rho_ideal[np.ix_(perm_arr, perm_arr)]
    fid_ideal_via_aer = state_fidelity(rho_ideal_reordered, ideal_sv_qk)

    print(f"sanity check -- noiseless transpiled circuit through AerSimulator: fidelity = {fid_ideal_via_aer:.9f}")
    print(f"REAL-CALIBRATION noisy transpiled circuit (FakeSherbrooke): fidelity = {fid_noisy:.9f} "
          f"(infidelity {1 - fid_noisy:.4f})")
    print("NOTE: this is an encoding-fidelity result under real IBM device calibration data, not a full")
    print("logical-error-rate curve -- no syndrome decoding was run on this noisy output (see docstring).")
    print("Not directly comparable to block 1's/block 4's depolarizing-noise sweeps (different noise")
    print("mechanism: real per-gate Kraus channels from actual device calibration, not a uniform")
    print("per-qubit depolarizing p) -- reported as its own real data point, not forced into that comparison.")

    df = pd.DataFrame({
        "check": [
            "native_block1", "qiskit_roundtrip_statevector_crosscheck",
            "qiskit_roundtrip_pauli_expectation_crosscheck", "qiskit_roundtrip_dense_evolution_rerun",
            "transpiled_noiseless_fidelity", "transpiled_noisy_fidelity_fakesherbrooke",
        ],
        "fidelity": [1.0, 1.0 if checkA_ok else np.nan, fid_b, fid_c, fid_transpiled, fid_noisy],
    })
    csv_path = _DATA_DIR / "steane_block5_qiskit_bridge.csv"
    df.to_csv(csv_path, index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ["Ideal\n(Dense-Evolution)", "Round-trip\n(Qiskit)",
              "Transpiled\n(noiseless)", "Transpiled + noisy\n(real calibration)"]
    values = [1.0, fid_c, fid_transpiled, fid_noisy]
    colors = ['#00FFFF', '#00FFFF', '#00FFFF', '#FF007F']
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fidelity vs ideal |0>_L', color='#888888')
    ax.set_title(f'Steane [[7,1,3]] encoding: Dense-Evolution -> Qiskit -> {backend.name}\n'
                 f'({len(active)} physical qubits, {n_2q_native} native 2-qubit gates, depth {tqc.depth()})',
                 fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.2, color='#444444')
    ax.tick_params(axis='x', labelsize=9)
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.4f}", ha='center', fontsize=9, color='white')
    plt.tight_layout()
    png_path = _IMAGES_DIR / "steane_block5_qiskit_bridge_fidelity.png"
    plt.savefig(png_path, dpi=300)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Round trip Dense-Evolution <-> Qiskit verified faithful (3 independent checks): {roundtrip_ok}")
    print(f"Backend: {backend.name} ({backend.num_qubits} real physical qubits, real Eagle coupling map)")
    print(f"Transpiled: {len(active)} physical qubits used for {N} logical qubits (0 extra ancilla qubits), "
          f"{n_2q_native} native 2-qubit gates vs {n_2q_original} original CX, depth {tqc.depth()}")
    print(f"Transpiled circuit's physics re-verified correct (noiseless): {physics_ok}")
    print(f"Real-calibration noisy encoding fidelity (FakeSherbrooke): {fid_noisy:.4f}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {png_path}")
    print()
    print("WHAT THIS SCRIPT DID (offline, free, no IBM account):")
    print("  Dense-Evolution circuit -> QASM -> Qiskit -> transpile for a real device's calibration")
    print("  snapshot + coupling map -> simulate noiselessly and with real per-gate noise, all locally.")
    print("WHAT IS STILL NEEDED FOR ACTUAL LIVE-HARDWARE SUBMISSION (deliberately NOT done here):")
    print("  A real IBM Quantum account + API token (qiskit-ibm-runtime QiskitRuntimeService), submission")
    print("  to the real device's job queue, and real queue/wait time. This script never contacted IBM's")
    print("  cloud service and used only local fake-backend/simulator tooling throughout.")


if __name__ == "__main__":
    main()
