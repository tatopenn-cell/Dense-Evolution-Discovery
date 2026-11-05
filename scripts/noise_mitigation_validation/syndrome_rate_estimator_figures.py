import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dense_evolution as de
from syndrome_rate_estimator import estimate_error_rates_from_syndromes

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "assets" / "syndrome_rate_estimator"
OUT.mkdir(parents=True, exist_ok=True)

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; x q[1]; cx q[0],q[3]; cx q[1],q[3]; cx q[1],q[4]; cx q[2],q[4];'
circ = de.QASMParser().parse(qasm)
fig = de.plot_circuit(circ.to_tuples(), n_qubits=5, title="Two checks (q3, q4) watching three qubits, one X error on q1")
fig.savefig(OUT / "step1_circuit.png", dpi=130, bbox_inches="tight")
plt.close(fig)

n = 9
h = np.zeros((n - 1, n), dtype=int)
for i in range(n - 1):
    h[i, i] = h[i, i + 1] = 1
p = np.full(n, 0.01)
p[2:7] = 0.25
rng = np.random.default_rng(3)
e = (rng.random((2000, n)) < p).astype(int)
r = estimate_error_rates_from_syndromes(h, (e @ h.T) % 2)

fig, ax = plt.subplots(figsize=(7, 3.4))
x = np.arange(n)
ax.bar(x - 0.2, p, 0.4, label="true error probability")
ax.bar(x + 0.2, r, 0.4, label="estimated from the check alarms")
ax.set_xlabel("qubit")
ax.set_ylabel("error probability")
ax.set_xticks(x)
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "step4_estimate.png", dpi=130)
plt.close(fig)
import pymatching

w = -np.log(r + 1e-3)
ee = (np.random.default_rng(11).random((20000, n)) < p).astype(int)
ss = ((ee @ h.T) % 2).astype(np.uint8)
for name, wt in (("blind", None), ("weighted", w)):
    m = pymatching.Matching.from_check_matrix(h, weights=wt)
    print(name, "logical failure rate over 20000 cycles:", float(((ee ^ m.decode_batch(ss)).sum(1) == n).mean()))
print("figures written to", OUT)
