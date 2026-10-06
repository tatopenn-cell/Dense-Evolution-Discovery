"""
Estimate each qubit's error probability from the syndromes of a code.

A stabilizer code reports once per cycle which of its checks fired. Where
errors are more frequent, the nearby checks fire more often. This script turns
the firing frequency of every check over a window of cycles into one error
probability per qubit.

For independent errors of one type, a check fires with probability
(1 - prod(1 - 2 p_q)) / 2 over the qubits it touches. With
a_q = -ln(1 - 2 p_q) and a measured firing frequency f, -ln(1 - 2 f) is the sum
of a_q over the check's qubits: a linear system, solved by non-negative least
squares. A code has fewer checks than qubits, so a small pull toward the
baseline rate fixes the remaining freedom.
"""
import numpy as np
from scipy.optimize import nnls


def estimate_error_rates_from_syndromes(h, hist, p0=0.01, lam=0.1):
    h = np.asarray(h, dtype=float)
    s = np.asarray(hist, dtype=float)
    if h.ndim != 2:
        raise ValueError("h must be 2-D (n_checks, n_qubits)")
    if s.ndim != 2 or s.shape[1] != h.shape[0]:
        raise ValueError(f"hist must have shape (n_cycles, {h.shape[0]}), got {s.shape}")
    if s.shape[0] == 0:
        raise ValueError("hist needs at least one cycle")
    if not np.isin(s, (0.0, 1.0)).all():
        raise ValueError("hist must contain only 0 and 1")
    if not 0.0 < p0 < 0.5:
        raise ValueError(f"p0 must be in (0, 0.5), got {p0}")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    nq = h.shape[1]
    f = np.clip(s.mean(axis=0), 0.0, 0.49)
    y = -np.log(1.0 - 2.0 * f)
    a0 = -np.log(1.0 - 2.0 * p0)
    r = np.sqrt(lam)
    a, _ = nnls(np.vstack([h, r * np.eye(nq)]), np.concatenate([y, r * np.full(nq, a0)]))
    return (1.0 - np.exp(-a)) / 2.0
