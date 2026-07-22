import pathlib, time, psutil, pytest, numpy as np, jax, jax.numpy as jnp, dense_evolution as de
import matplotlib.pyplot as plt

_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_IMAGES_DIR.mkdir(exist_ok=True)

NUM_QUBITS, TUNNELING_HOPPING_PARAM, CHUNK_SIZE, MEMORY_THRESHOLD_PERCENT, POINTS_FOR_MAIN_BENCHMARK = 12, 2.11, 4000, 0.15, 300
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
# (param_vqe = t, param_vqe_inv = -t*0.5), so parameter_batch needs
# 2*(NUM_QUBITS-1) columns, not 2 fixed.
#
# On top of that column-count bug (fixed separately), the original
# "PSR" here shifted the shared t by +-pi/2 and read the whole batch --
# that is NOT an exact gradient when one parameter drives multiple gates.
# The textbook parameter-shift rule is only exact when shifting ONE
# gate's own parameter by +-pi/2 while holding every other gate fixed.
# The exact dE/dt follows from the chain rule over all 2*N_BONDS gates:
#
#   dE/dt = sum_q [ dE/dtheta_A_q * (dtheta_A_q/dt) + dE/dtheta_B_q * (dtheta_B_q/dt) ]
#
# with dE/dtheta_{A,B}_q each an exact single-gate PSR shift, and
# dtheta_A_q/dt = 1, dtheta_B_q/dt = -0.5 (param_vqe_inv = -t*0.5).
# Verified against finite differences: agreement to ~1e-9.
N_BONDS = NUM_QUBITS - 1
N_PARAMS = 2 * N_BONDS
B_COEFF = -0.5                     # dtheta_B/dt ; theta_A = t, theta_B = -t*0.5
ROWS_PER_POINT = 1 + 4 * N_BONDS   # 1 base row + 2 shifts x 2 gates/bond

def _build_operations() -> list:
    operations = [['x', 0]]
    for q in range(NUM_QUBITS - 1):
        operations.extend([['cx', q + 1, q], ['ry', q + 1, 'param_vqe'], ['cx', q, q + 1], ['ry', q + 1, 'param_vqe_inv'], ['cx', q + 1, q]])
    return operations

def _base_row(t: float) -> np.ndarray:
    row = np.empty(N_PARAMS, dtype=np.float64)
    row[0::2] = t             # even slots: theta_A (param_vqe)
    row[1::2] = B_COEFF * t   # odd slots: theta_B (param_vqe_inv)
    return row

def _build_exact_psr_grid(points: np.ndarray) -> np.ndarray:
    """1 base row + 2*N_PARAMS shift rows per point (one +-pi/2 shift
    pair per gate parameter, all other gates held at their base value)."""
    grid = np.zeros((len(points) * ROWS_PER_POINT, N_PARAMS), dtype=np.float64)
    for idx, t in enumerate(points):
        base = _base_row(t)
        off = idx * ROWS_PER_POINT
        grid[off] = base
        r = off + 1
        for k in range(N_PARAMS):
            plus = base.copy();  plus[k]  += np.pi / 2
            minus = base.copy(); minus[k] -= np.pi / 2
            grid[r] = plus
            grid[r + 1] = minus
            r += 2
    return grid

def _extract_energy_and_exact_gradient(state_vector_batch: np.ndarray, point_idx: int):
    off = point_idx * ROWS_PER_POINT
    e_center = _get_e(state_vector_batch[off])
    grad = 0.0
    r = off + 1
    for k in range(N_PARAMS):
        e_plus = _get_e(state_vector_batch[r])
        e_minus = _get_e(state_vector_batch[r + 1])
        partial = 0.5 * (e_plus - e_minus)
        dtheta_dt = 1.0 if (k % 2 == 0) else B_COEFF
        grad += partial * dtheta_dt
        r += 2
    return e_center, grad

def _energy_single(theta: float) -> float:
    """Independent reference: single-circuit run_circuit_jit_beast_mode,
    not the batch/vmap code path -- every bond gets the same (theta, -theta*0.5)."""
    ops = [['x', 0]]
    for q in range(NUM_QUBITS - 1):
        ops.extend([['cx', q + 1, q], ['ry', q + 1, float(theta)], ['cx', q, q + 1], ['ry', q + 1, float(-theta * 0.5)], ['cx', q + 1, q]])
    _sim.set_initial_state()
    _sim.run_circuit_jit_beast_mode(ops)
    return _get_e(np.asarray(_sim.get_statevector()))

def _finite_difference_gradient(theta: float, h: float = 1e-6) -> float:
    """Independent gradient reference, computed via single-circuit calls,
    never touching run_parametric_batch_jit."""
    return (_energy_single(theta + h) - _energy_single(theta - h)) / (2 * h)

def _run_chunked(operations: list, parameter_grid: np.ndarray, chunk_size: int) -> np.ndarray:
    results = []
    _sim.set_initial_state()
    for i in range(0, len(parameter_grid), chunk_size):
        _guard.check(f"Chunk Split {i}")
        results.append(_sim.run_parametric_batch_jit(operations, jnp.array(parameter_grid[i : i + chunk_size], dtype=jnp.float64)))
    return np.concatenate(results, axis=0)

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
    grid = _build_exact_psr_grid(np.array([theta0]))
    assert grid.shape[1] == N_PARAMS == 2 * (NUM_QUBITS - 1)

    batch = _sim.run_parametric_batch_jit(_build_operations(), jnp.array(grid, dtype=jnp.float64))
    e_batch = _get_e(np.asarray(batch[0]))
    e_ref = _energy_single(theta0)
    assert abs(e_batch - e_ref) < 1e-9, (
        f"batch energy {e_batch:.10f} != independent single-circuit reference {e_ref:.10f}"
    )

@pytest.mark.parametrize("theta0", [0.5, 1.7, 3.0, 4.5, 5.9])
def test_vqe_jax_exact_psr_gradient_matches_finite_difference(theta0):
    """The exact chain-rule PSR gradient (batch/vmap code path) must agree
    with an independent finite-difference gradient (single-circuit code
    path) at essentially machine precision -- unlike the old shared-t
    +-pi/2 shift heuristic, which could disagree with the true gradient
    by 100% or even the wrong sign (found during the dense-evolution
    8.1.21 audit)."""
    _guard.check(f"PSR exactness theta={theta0}")
    grid = _build_exact_psr_grid(np.array([theta0]))
    batch = _sim.run_parametric_batch_jit(_build_operations(), jnp.array(grid, dtype=jnp.float64))
    _, exact_grad = _extract_energy_and_exact_gradient(np.asarray(batch), 0)
    fd_grad = _finite_difference_gradient(theta0)
    assert abs(exact_grad - fd_grad) < 1e-5, (
        f"theta={theta0}: exact PSR={exact_grad:.8f}, finite-diff={fd_grad:.8f}, "
        f"diff={abs(exact_grad - fd_grad):.2e}"
    )

def run_main_benchmark():
    n_tracks = POINTS_FOR_MAIN_BENCHMARK * ROWS_PER_POINT
    print("==================================================================================")
    print(f" CHUNKED JAX BATCH ({n_tracks:,} TRACKS, EXACT CHAIN-RULE PSR) @ {NUM_QUBITS} QUBITS")
    print("==================================================================================")
    _guard.check("Benchmark Startup")
    ram_start = _get_ram()
    t0 = time.perf_counter()
    points = np.linspace(0.1, 2 * np.pi - 0.1, POINTS_FOR_MAIN_BENCHMARK)
    operations = _build_operations()
    grid = _build_exact_psr_grid(points)
    ram_grid = _get_ram()
    state_vector_batch = _run_chunked(operations, grid, CHUNK_SIZE)
    ram_jit = _get_ram()
    t_vmap = time.perf_counter() - t0
    t_values, energy_values, gradient_values = [], [], []
    for idx, t in enumerate(points):
        energy_center, gradient = _extract_energy_and_exact_gradient(state_vector_batch, idx)
        t_values.append(t)
        energy_values.append(energy_center)
        gradient_values.append(gradient)
        if (idx + 1) % 250 == 0 or idx == 0 or idx == len(points) - 1:
            print(f"Step {idx+1:04d}/{POINTS_FOR_MAIN_BENCHMARK} | t: {t:.3f} rad | E(t): {energy_center:+.4f} eV | Exact PSR Grad: {gradient:+.6f}")
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
    ax2.plot(t_values, gradient_values, color='#FF007F', linewidth=2.5, label='Exact Chain-Rule PSR Gradient')
    ax2.axhline(0.0, color='#888888', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_title('Exact Parameter-Shift Rule Gradient Profile', fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel('Parameter t (rad)', color='#888888', fontsize=9)
    ax2.set_ylabel('Gradient Magnitude', color='#888888', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.3, color='#444444')
    ax2.legend(loc='upper right', framealpha=0.1)
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "vqe_parameter_shift_plot.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_main_benchmark()
