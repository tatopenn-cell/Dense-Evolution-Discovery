import time, psutil, pytest, numpy as np, jax, jax.numpy as jnp, dense_evolution as de
import matplotlib.pyplot as plt

NUM_QUBITS, TUNNELING_HOPPING_PARAM, CHUNK_SIZE, MEMORY_THRESHOLD_PERCENT, POINTS_FOR_PYTEST, POINTS_FOR_MAIN_BENCHMARK = 12, 2.11, 4000, 0.15, 15000, 1000
jax.config.update("jax_enable_x64", True)
_sim = de.DenseSVSimulator(n_qubits=NUM_QUBITS, use_gpu=False, use_float32=False)
from dense_evolution.chunk import SafeMemoryGuard, MemoryChunker
_guard = SafeMemoryGuard(threshold_pct=MEMORY_THRESHOLD_PERCENT)
_chunker = MemoryChunker(n_qubits=NUM_QUBITS)

def _get_ram() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)

def _get_e(state_vector: np.ndarray) -> float:
    dim = len(state_vector)
    indices = np.arange(dim, dtype=np.int32)
    energy = 0.0
    for q in range(NUM_QUBITS):
        qn = (q + 1) % NUM_QUBITS
        mask = (1 << q) | (1 << qn)
        flipped_state_vector = state_vector[indices ^ mask]
        energy += np.real(np.sum(np.conj(state_vector) * flipped_state_vector))
        energy += np.real(np.sum(np.conj(state_vector) * flipped_state_vector * np.where(((indices & (1 << q)) >> q) == ((indices & (1 << qn)) >> qn), -1.0, 1.0)))
    return -(TUNNELING_HOPPING_PARAM / 2.0) * energy

# run_parametric_batch_jit treats EVERY rotation gate as a positional
# parameter slot -- each of the NUM_QUBITS-1 bonds contributes 2 ry gates
# (param_vqe, param_vqe_inv), so parameter_batch needs 2*(NUM_QUBITS-1)
# columns, not 2 fixed. With only 2 columns, only the first bond got the
# intended (t, -t*0.5); the other 10 bonds ran out of bounds and JAX
# silently clipped to the last valid column instead of raising.
N_BONDS = NUM_QUBITS - 1
N_PARAMS = 2 * N_BONDS

def _build_operations() -> list:
    operations = [['x', 0]]
    for q in range(NUM_QUBITS - 1):
        operations.extend([['cx', q + 1, q], ['ry', q + 1, 'param_vqe'], ['cx', q, q + 1], ['ry', q + 1, 'param_vqe_inv'], ['cx', q + 1, q]])
    return operations

def _row(t: float):
    return [t, -t * 0.5] * N_BONDS

def _build_grid(points: np.ndarray) -> np.ndarray:
    grid = np.zeros((len(points) * 3, N_PARAMS), dtype=np.float64)
    for idx, t in enumerate(points):
        grid[idx * 3]     = _row(t)
        grid[idx * 3 + 1] = _row(t + np.pi / 2)
        grid[idx * 3 + 2] = _row(t - np.pi / 2)
    return grid

def _energy_single(theta: float) -> float:
    """Independent reference: single-circuit run_circuit_jit_beast_mode,
    not the batch/vmap code path -- every bond gets the same (theta, -theta*0.5)."""
    ops = [['x', 0]]
    for q in range(NUM_QUBITS - 1):
        ops.extend([['cx', q + 1, q], ['ry', q + 1, float(theta)], ['cx', q, q + 1], ['ry', q + 1, float(-theta * 0.5)], ['cx', q + 1, q]])
    _sim.set_initial_state()
    _sim.run_circuit_jit_beast_mode(ops)
    return _get_e(np.asarray(_sim.get_statevector()))

def _run_chunked(operations: list, parameter_grid: np.ndarray, chunk_size: int) -> np.ndarray:
    results = []
    _sim.set_initial_state()
    for i in range(0, len(parameter_grid), chunk_size):
        _guard.check(f"Chunk Split {i}")
        results.append(_sim.run_parametric_batch_jit(operations, jnp.array(parameter_grid[i : i + chunk_size], dtype=jnp.float64)))
    return np.concatenate(results, axis=0)

def test_vqe_jax_parameter_shift_rule():
    _guard.check("Pytest Pre-flight")
    points = np.linspace(0.1, 2 * np.pi - 0.1, POINTS_FOR_PYTEST)
    operations = _build_operations()
    grid = _build_grid(points)
    state_vector_batch = _run_chunked(operations, grid, CHUNK_SIZE)
    assert len(state_vector_batch) == (POINTS_FOR_PYTEST * 3)
    energy_plus = _get_e(state_vector_batch[1])
    energy_minus = _get_e(state_vector_batch[2])
    assert not np.isnan(0.5 * (energy_plus - energy_minus))

def test_vqe_jax_batch_grid_has_one_column_per_rotation_slot():
    """Regression guard for the audit finding: run_parametric_batch_jit
    consumes one parameter_batch column per rotation gate IN ORDER, even
    when base_circuit passes a string placeholder -- there is no name
    matching. With NUM_QUBITS-1 bonds x 2 ry gates/bond, the grid used to
    have only 2 fixed columns, so only the first bond got the intended
    (t, -t*0.5); the other 10 silently ran on a clipped out-of-bounds
    read instead of raising. Cross-checked against an independent
    single-circuit run_circuit_jit_beast_mode reference."""
    theta0 = 1.7
    grid = _build_grid(np.array([theta0]))
    assert grid.shape[1] == N_PARAMS == 2 * (NUM_QUBITS - 1)

    batch = _sim.run_parametric_batch_jit(_build_operations(), jnp.array(grid, dtype=jnp.float64))
    e_batch = _get_e(np.asarray(batch[0]))
    e_ref = _energy_single(theta0)
    assert abs(e_batch - e_ref) < 1e-9, (
        f"batch energy {e_batch:.10f} != independent single-circuit reference {e_ref:.10f}"
    )

def run_main_benchmark():
    print("==================================================================================")
    print(f" CHUNKED JAX BATCH ({POINTS_FOR_MAIN_BENCHMARK * 3} TRACKS) @ {NUM_QUBITS} QUBITS")
    print("==================================================================================")
    _guard.check("Benchmark Startup")
    ram_start = _get_ram()
    t0 = time.perf_counter()
    points = np.linspace(0.1, 2 * np.pi - 0.1, POINTS_FOR_MAIN_BENCHMARK)
    operations = _build_operations()
    grid = _build_grid(points)
    ram_grid = _get_ram()
    state_vector_batch = _run_chunked(operations, grid, CHUNK_SIZE)
    ram_jit = _get_ram()
    t_vmap = time.perf_counter() - t0
    t_values, energy_values, gradient_values = [], [], []
    for idx, t in enumerate(points):
        energy_center = _get_e(state_vector_batch[idx * 3])
        energy_plus_shift = _get_e(state_vector_batch[idx * 3 + 1])
        energy_minus_shift = _get_e(state_vector_batch[idx * 3 + 2])
        gradient = 0.5 * (energy_plus_shift - energy_minus_shift)
        t_values.append(t)
        energy_values.append(energy_center)
        gradient_values.append(gradient)
        if (idx + 1) % 250 == 0 or idx == 0 or idx == len(points) - 1:
            print(f"Step {idx+1:04d}/{POINTS_FOR_MAIN_BENCHMARK} | t: {t:.3f} rad | E(t): {energy_center:+.4f} eV | PSR Grad: {gradient:+.6f}")
    t_total = time.perf_counter() - t0
    print("----------------------------------------------------------------------------------")
    print(f"Base RAM Overhead : {ram_start:.2f} MB\nGrid Build RAM    : {ram_grid:.2f} MB")
    print(f"JAX VMAP Peak RAM : {ram_jit:.2f} MB | Net Delta: {ram_jit - ram_start:+.2f} MB")
    print("----------------------------------------------------------------------------------")
    print(f"Guard Status      : {repr(_guard)}")
    print(f"VMAP Compute Time : {t_vmap*1000:.2f} ms\nTotal Execution   : {t_total:.2f} s")
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    ax1.plot(t_values, energy_values, color='#00FFFF', linewidth=2.5, label='Energy E(t)')
    ax1.set_title('Energy Profile as a Function of Parameter t', fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel('Parameter t (rad)', color='#888888', fontsize=9)
    ax1.set_ylabel('Energy (eV)', color='#888888', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.3, color='#444444')
    ax1.legend(loc='upper right', framealpha=0.1)
    ax2.plot(t_values, gradient_values, color='#FF007F', linewidth=2.5, label='PSR Gradient')
    ax2.axhline(0.0, color='#888888', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_title('Parameter Shift Rule Gradient Profile', fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel('Parameter t (rad)', color='#888888', fontsize=9)
    ax2.set_ylabel('Gradient Magnitude', color='#888888', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.3, color='#444444')
    ax2.legend(loc='upper right', framealpha=0.1)
    plt.tight_layout()
    plt.savefig("vqe_parameter_shift_plot.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_main_benchmark()
