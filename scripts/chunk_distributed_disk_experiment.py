"""
Real, end-to-end check that Chunk's three execution strategies -- in-RAM
multi-piece, real multi-device (distributed), and disk-backed overflow --
all give the identical physical result for the same circuit, plus real
wall-clock numbers for each path.

The main library's own docs/api/chunk.md demonstrates each strategy in
isolation (and the distributed path only as far as its RuntimeError when
too few JAX devices are present -- it never actually forces extra CPU
devices to run it for real). This script goes one step further: it sets
XLA_FLAGS to fake 8 CPU devices *before* dense_evolution/jax are ever
imported (JAX's device count is fixed at first initialization -- this
must happen first, not as an afterthought), so run_chunk_distributed
actually exercises jax.lax.ppermute for real, not just the guard clause
in front of it.

Follows two real papers already cited in docs/api/chunk.md:
- LaRose 2018, arXiv:1801.01037 -- the chunk-select/local qubit split and
  pairwise cross-chunk communication this experiment's distributed path
  reproduces (originally on a real MPI/OpenMP supercomputer).
- Pednault et al. 2019, arXiv:1910.09534 -- the disk-backed secondary
  storage this experiment's disk-overflow path reproduces (originally
  used to break the 49-qubit classical-simulation barrier for Sycamore).
"""
import os
import pathlib
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax

import dense_evolution as de
from dense_evolution.backends.chunk import Chunk
from dense_evolution import chunk as chunk_mod
from dense_evolution.circuits.diagram import plot_circuit

ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "assets" / "chunk_distributed_disk_experiment"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 4
QASM = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; '
    "h q[0]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];"
)
circuit = de.QASMParser().parse(QASM)
ops = circuit.to_tuples()

circuit_fig = plot_circuit(ops, N_QUBITS, title="4-qubit GHZ chain")
circuit_fig.savefig(ASSETS_DIR / "circuit.png", dpi=150, bbox_inches="tight")
plt.close(circuit_fig)

# ── 0. Ground truth: plain, unchunked simulator ─────────────────────────
_t0 = time.perf_counter()
baseline_sim = de.DenseSVSimulator(N_QUBITS)
baseline_sim.run_circuit_jit(ops)
baseline_probs = np.asarray(baseline_sim.get_probabilities())
baseline_time = time.perf_counter() - _t0

# Force the same tiny geometry docs/api/chunk.md's own Step 2 uses --
# 4 chunks of 2 qubits each -- so the multi-piece/distributed/disk paths
# below are all exercised for real, deterministically, on any machine.
chunk_mod.get_dynamic_chunk = lambda dtype_target: 2

# ── 1. In-RAM multi-chunk path ──────────────────────────────────────────
_t0 = time.perf_counter()
ram_chunk = Chunk(N_QUBITS)
ram_chunk.run_chunk(ops)
ram_probs = np.asarray(ram_chunk.get_probabilities())
ram_time = time.perf_counter() - _t0
ram_num_chunks = ram_chunk.num_chunks
ram_m = ram_chunk._m

stride_pairs = []
for q in range(ram_m):
    stride = 1 << (ram_m - 1 - q)
    for i in range(ram_num_chunks):
        j = i ^ stride
        if i < j:
            stride_pairs.append((i, j, q))


def _draw_chunk_layout(pairs, num_chunks, title, disk=False):
    TEXT = "#24292f"
    with matplotlib.rc_context({"figure.facecolor": "white", "axes.facecolor": "white",
                                 "text.color": TEXT, "axes.edgecolor": TEXT}):
        fig, ax = plt.subplots(figsize=(6, 2.2), facecolor="white")
        ax.set_facecolor("white")
        xs = np.linspace(0, 1, num_chunks)
        for i, x in enumerate(xs):
            face = "#f2f4f7" if not disk else "#eef6ff"
            edge = "#57606a" if not disk else "#0086a8"
            ax.add_patch(plt.Rectangle((x - 0.06, 0.3), 0.12, 0.4, facecolor=face, edgecolor=edge, lw=1.5))
            ax.text(x, 0.5, f"c{i}", ha="center", va="center", family="monospace", fontsize=11, color=TEXT)
            if disk:
                ax.text(x, 0.15, "disk", ha="center", va="center", family="monospace", fontsize=8, color="#0086a8")
        for i, j, q in pairs:
            ax.annotate(
                "", xy=(xs[j], 0.75), xytext=(xs[i], 0.75),
                arrowprops=dict(arrowstyle="<->", color="#00875f", lw=1.3,
                                 connectionstyle=f"arc3,rad={0.25 + 0.1*q}"),
            )
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=11, family="monospace", color=TEXT)
        fig.tight_layout()
    return fig

layout_fig = _draw_chunk_layout(stride_pairs, ram_num_chunks, "chunk-select stride pairing (m=2 qubits)")
layout_fig.savefig(ASSETS_DIR / "chunk_layout.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(layout_fig)

disk_layout_fig = _draw_chunk_layout(stride_pairs, ram_num_chunks, "disk overflow: only the active pair leaves disk", disk=True)
disk_layout_fig.savefig(ASSETS_DIR / "disk_layout.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(disk_layout_fig)

# ── 2. Real multi-device distributed path (8 fake CPU devices) ─────────
distributed_available = jax.device_count() >= ram_num_chunks
if distributed_available:
    _t0 = time.perf_counter()
    dist_chunk = Chunk(N_QUBITS)
    dist_chunk.run_chunk_distributed(ops)
    distributed_probs = np.asarray(dist_chunk.get_probabilities())
    distributed_time = time.perf_counter() - _t0
else:  # pragma: no cover -- only if XLA_FLAGS above didn't take effect
    distributed_probs = None
    distributed_time = None

# ── 3. Disk-backed overflow path ────────────────────────────────────────
_t0 = time.perf_counter()
disk_chunk = Chunk(N_QUBITS, memory_threshold=0.999999, allow_disk_overflow=True)
disk_chunk.run_chunk(ops)
disk_probs = np.asarray(disk_chunk.get_probabilities())
disk_time = time.perf_counter() - _t0
disk_storage_dir = disk_chunk._disk_dir
disk_chunk.close()

# ── Cross-checks ─────────────────────────────────────────────────────────
ram_matches_baseline = bool(np.allclose(ram_probs, baseline_probs))
disk_matches_baseline = bool(np.allclose(disk_probs, baseline_probs))
distributed_matches_baseline = (
    bool(np.allclose(distributed_probs, baseline_probs))
    if distributed_probs is not None
    else None
)

if __name__ == "__main__":
    print(f"JAX devices available: {jax.device_count()}")
    print(f"num_chunks (forced 2-qubit geometry): {ram_num_chunks}")
    print(f"baseline   : {baseline_time*1000:6.1f} ms  matches=n/a (reference)")
    print(f"ram chunk  : {ram_time*1000:6.1f} ms  matches_baseline={ram_matches_baseline}")
    if distributed_probs is not None:
        print(f"distributed: {distributed_time*1000:6.1f} ms  matches_baseline={distributed_matches_baseline}")
    print(f"disk chunk : {disk_time*1000:6.1f} ms  matches_baseline={disk_matches_baseline}")
