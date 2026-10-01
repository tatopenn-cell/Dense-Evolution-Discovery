"""
Erasure decoders, state of the art, reproduced and checked against each other.

  erasure_ml_decode   maximum-likelihood decoder for any stabilizer code, by
                      Gaussian elimination over GF(2) (Delfosse and Zemor,
                      arXiv:1703.01517; Kuo and Ouyang, arXiv:2411.13509).
  peeling_decode      linear-time maximum-likelihood decoder for graph-like
                      CSS codes (Delfosse and Zemor, Algorithm 1).
"""
from collections import deque
from typing import Optional, Sequence

import numpy as np

from dense_evolution.physics.qec import (
    _gf2_rref, _in_gf2_span, _pauli_to_symplectic, compute_syndrome,
)


def erasure_ml_decode(
    observed_syndrome: tuple,
    heralded_qubits: Sequence[int],
    n_qubits: int,
    stabilizers: Sequence[str],
) -> Optional[str]:
    """Maximum-likelihood decoder for erasures at known locations, for any
    stabilizer code.

    With the erased qubits known, the error is supported on them, and the
    syndrome becomes a linear system over GF(2) in the X and Z bits of those
    qubits (Delfosse & Zemor, arXiv:1703.01517; Kuo & Ouyang,
    arXiv:2411.13509). Solving it by Gaussian elimination costs O(n^3), where
    `erasure_aware_decode` tries 4**m assignments for m erased qubits. Any
    solution is a valid correction when every zero-syndrome operator on the
    erased qubits is a stabilizer element, so degenerate errors (several
    errors that differ by a stabilizer) are decoded too, not rejected.

    Parameters
    ----------
    observed_syndrome : sequence of int
        One bit per stabilizer, same convention as `compute_syndrome`.
    heralded_qubits : sequence of int
        Indices of the erased qubits.
    n_qubits : int
        Number of physical qubits.
    stabilizers : sequence of str
        Stabilizer generators as Pauli strings of length `n_qubits`.

    Returns
    -------
    str or None
        A Pauli string supported on the erased qubits that reproduces the
        syndrome and is equivalent, up to a stabilizer, to every other
        solution. `None` when no error on the erased qubits explains the
        syndrome, or when the erased qubits contain a logical operator so the
        correction is ambiguous. With no erased qubits it returns the identity
        for a zero syndrome and `None` otherwise.

    Raises
    ------
    ValueError
        If the syndrome length does not match the stabilizers, a stabilizer
        has the wrong length, or an erased index is out of range.

    Examples
    --------
    >>> stabs = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX', 'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']
    >>> syndrome = compute_syndrome('XIIIIIX', stabs)
    >>> erasure_ml_decode(syndrome, [0, 6], 7, stabs)
    'XIIIIIX'
    """
    stabs = list(stabilizers)
    if len(observed_syndrome) != len(stabs):
        raise ValueError(
            f"observed_syndrome has {len(observed_syndrome)} entries but there are "
            f"{len(stabs)} stabilizers"
        )
    for i, s in enumerate(stabs):
        if len(s) != n_qubits:
            raise ValueError(f"stabilizers[{i}] has length {len(s)}, expected n_qubits={n_qubits}")
    erased = sorted({int(q) for q in heralded_qubits})
    if any(q < 0 or q >= n_qubits for q in erased):
        raise ValueError(f"heralded_qubits must be in range(0, {n_qubits})")

    syn = np.array(observed_syndrome, dtype=np.uint8) % 2
    k = len(erased)
    if k == 0:
        return 'I' * n_qubits if not syn.any() else None

    sym = np.array([_pauli_to_symplectic(s) for s in stabs], dtype=np.uint8)
    sx, sz = sym[:, :n_qubits], sym[:, n_qubits:]
    a = np.concatenate([sz[:, erased], sx[:, erased]], axis=1)
    rref, pivots = _gf2_rref(np.concatenate([a, syn[:, None]], axis=1))
    if 2 * k in pivots:
        return None

    sol = np.zeros(2 * k, dtype=np.uint8)
    for row, col in enumerate(pivots):
        sol[col] = rref[row, 2 * k]
    free = [c for c in range(2 * k) if c not in pivots]
    stab_rref, stab_pivots = _gf2_rref(sym)

    def embed(v):
        full = np.zeros(2 * n_qubits, dtype=np.uint8)
        for j, q in enumerate(erased):
            full[q] = v[j]
            full[n_qubits + q] = v[k + j]
        return full

    for f in free:
        vec = np.zeros(2 * k, dtype=np.uint8)
        vec[f] = 1
        for row, col in enumerate(pivots):
            if rref[row, f]:
                vec[col] = 1
        if not _in_gf2_span(embed(vec), stab_rref, stab_pivots):
            return None

    full = embed(sol)
    letters = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}
    return ''.join(letters[(int(full[q]), int(full[n_qubits + q]))] for q in range(n_qubits))


def _peel(h, erased, syn):
    m = h.shape[0]
    adj = {}
    for q in erased:
        rows = np.flatnonzero(h[:, q])
        u, v = (int(rows[0]), int(rows[1])) if len(rows) == 2 else (int(rows[0]), m)
        adj.setdefault(u, []).append((v, q))
        adj.setdefault(v, []).append((u, q))
    syn = [int(b) for b in syn] + [0]
    seen, order, parent = set(), [], {}
    starts = ([m] if m in adj else []) + [v for v in adj if v != m]
    for s in starts:
        if s in seen:
            continue
        seen.add(s)
        queue = deque([s])
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, q in adj[u]:
                if v not in seen:
                    seen.add(v)
                    parent[v] = (u, q)
                    queue.append(v)
    corr = set()
    for u in reversed(order):
        if u in parent:
            p, q = parent[u]
            if syn[u]:
                corr.add(q)
                syn[p] ^= 1
                syn[u] = 0
        elif u != m and syn[u]:
            return None
    if any(syn[v] for v in range(m) if v not in seen):
        return None
    return corr


def peeling_decode(stabilizers, observed_syndrome, heralded_qubits, n_qubits) -> Optional[str]:
    """Peeling decoder for a CSS code whose X-type and Z-type checks each join
    every qubit to at most two checks (repetition and surface codes)."""
    stabs = list(stabilizers)
    syn = np.array(observed_syndrome, dtype=np.uint8)
    erased = sorted({int(q) for q in heralded_qubits})
    zi = [i for i, s in enumerate(stabs) if 'Z' in s and 'X' not in s]
    xi = [i for i, s in enumerate(stabs) if 'X' in s and 'Z' not in s]
    hz = np.array([[c != 'I' for c in stabs[i]] for i in zi], dtype=np.uint8)
    hx = np.array([[c != 'I' for c in stabs[i]] for i in xi], dtype=np.uint8)
    x_part = _peel(hz, erased, syn[zi])
    z_part = _peel(hx, erased, syn[xi])
    if x_part is None or z_part is None:
        return None
    return ''.join('IXZY'[(q in x_part) + 2 * (q in z_part)] for q in range(n_qubits))


def _uf_grow(h, erased, syn):
    m, n = h.shape
    ends = []
    for q in range(n):
        rows = np.flatnonzero(h[:, q])
        ends.append((int(rows[0]), int(rows[1])) if len(rows) == 2 else (int(rows[0]), m))
    parent = list(range(m + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    support = [0] * n
    for q in erased:
        support[q] = 2
        parent[find(ends[q][0])] = find(ends[q][1])
    s = [int(b) for b in syn] + [0]

    def odd_roots():
        par = {}
        for v in range(m + 1):
            r = find(v)
            par[r] = par.get(r, 0) ^ s[v]
        return {r for r, p in par.items() if p and r != find(m)}

    odd = odd_roots()
    while odd:
        grown = False
        for q in range(n):
            if support[q] < 2:
                ru, rv = find(ends[q][0]), find(ends[q][1])
                inc = (ru in odd) + (rv in odd and rv != ru)
                if inc:
                    support[q] = min(2, support[q] + inc)
                    grown = True
        if not grown:
            return None
        for q in range(n):
            if support[q] == 2:
                parent[find(ends[q][0])] = find(ends[q][1])
        odd = odd_roots()
    return [q for q in range(n) if support[q] == 2]


def _css_split(stabilizers, observed_syndrome):
    stabs = list(stabilizers)
    syn = np.array(observed_syndrome, dtype=np.uint8)
    zi = [i for i, s in enumerate(stabs) if 'Z' in s and 'X' not in s]
    xi = [i for i, s in enumerate(stabs) if 'X' in s and 'Z' not in s]
    hz = np.array([[c != 'I' for c in stabs[i]] for i in zi], dtype=np.uint8)
    hx = np.array([[c != 'I' for c in stabs[i]] for i in xi], dtype=np.uint8)
    return hz, syn[zi], hx, syn[xi]


def union_find_decode(stabilizers, observed_syndrome, heralded_qubits, n_qubits) -> Optional[str]:
    """Union-Find decoder with erasures and Pauli errors (Delfosse and
    Nickerson, arXiv:1709.06218): clusters start from the erased qubits and
    grow by half-edges until every cluster has even syndrome parity or touches
    the boundary, then each grown cluster is peeled."""
    hz, sz, hx, sx = _css_split(stabilizers, observed_syndrome)
    erased = sorted({int(q) for q in heralded_qubits})
    parts = []
    for h, s in ((hz, sz), (hx, sx)):
        grown = _uf_grow(h, erased, s)
        if grown is None:
            return None
        part = _peel(h, grown, s)
        if part is None:
            return None
        parts.append(part)
    return ''.join('IXZY'[(q in parts[0]) + 2 * (q in parts[1])] for q in range(n_qubits))


def matching_erasure_decode(stabilizers, observed_syndrome, heralded_qubits, n_qubits) -> str:
    """Minimum-weight perfect matching with weight 0 on the erased qubits
    (Stace, Barrett and Doherty, arXiv:0904.3556), via pymatching."""
    import pymatching

    hz, sz, hx, sx = _css_split(stabilizers, observed_syndrome)
    w = np.ones(n_qubits)
    w[list(heralded_qubits)] = 0.0
    xp = pymatching.Matching.from_check_matrix(hz, weights=w).decode(sz)
    zp = pymatching.Matching.from_check_matrix(hx, weights=w).decode(sx)
    return ''.join('IXZY'[int(xp[q]) + 2 * int(zp[q])] for q in range(n_qubits))


def estimate_edge_probabilities_from_detection_events(check_matrix, events) -> np.ndarray:
    """Error probability of every qubit, from detection events (Spitz et al.).

    Implements the exact inversion of S. T. Spitz, B. Tarasinski,
    C. W. J. Beenakker and T. E. O'Brien, "Adaptive weight estimator for
    quantum error correction in a time-dependent environment",
    arXiv:1712.02360, Eqs. (13) and (16). For a code in which every qubit is
    checked by at most two checks (repetition and surface codes), each qubit is
    an edge between two checks, or between one check and the boundary. A qubit
    shared by checks ``i`` and ``j`` has probability

    ``p = 1/2 - sqrt(1/4 - (<v_i v_j> - <v_i><v_j>) / (1 - 2 <v_i xor v_j>))``

    where ``v`` are the detection events and ``<.>`` the average over cycles. A
    qubit on the boundary of check ``i`` has

    ``p = 1/2 + (<v_i> - 1/2) / prod(1 - 2 p_ij)`` over the other qubits of ``i``.

    The result can be passed as ``weights`` (``-log(p / (1 - p))``) to
    `pymatching_decode`.

    Parameters
    ----------
    check_matrix : array_like of shape (n_checks, n_qubits)
        0/1 matrix, entry ``[c, q] = 1`` when check ``c`` detects an error on
        qubit ``q``. Every column has one or two ones; no two qubits may join
        the same pair of checks.
    events : array_like of shape (n_cycles, n_checks)
        0/1 detection events: ``1`` when a check changed value since the
        previous cycle (the syndrome of one cycle's new errors).

    Returns
    -------
    numpy.ndarray of shape (n_qubits,)
        Estimated error probability of each qubit, clipped to [0, 0.5].

    Raises
    ------
    ValueError
        If the shapes do not match, ``events`` is not 0/1, a column of
        ``check_matrix`` does not have one or two ones, or two qubits join the
        same pair of checks.

    Notes
    -----
    Valid for independent errors and one error type at a time. Needs about
    ``1 / p`` cycles per qubit for a stable estimate (the paper's Eq. 18). A
    pair of checks whose correlation is below the statistical noise gives a
    probability near zero.

    Examples
    --------
    >>> import numpy as np
    >>> from dense_evolution.physics.qec import estimate_edge_probabilities_from_detection_events
    >>> checks = np.array([[1, 0], [1, 1]])
    >>> events = np.array([[1, 1]] * 20 + [[0, 0]] * 80)
    >>> p = estimate_edge_probabilities_from_detection_events(checks, events)
    >>> bool(p[0] < 0.5 and p[1] < 0.5)
    True
    """
    h = np.asarray(check_matrix, dtype=int)
    v = np.asarray(events, dtype=float)
    if h.ndim != 2:
        raise ValueError("check_matrix must be 2-D (n_checks, n_qubits)")
    if v.ndim != 2 or v.shape[1] != h.shape[0]:
        raise ValueError(
            f"events must have shape (n_cycles, {h.shape[0]}), got {v.shape}"
        )
    if v.shape[0] == 0:
        raise ValueError("events needs at least one cycle")
    if not np.isin(v, (0.0, 1.0)).all():
        raise ValueError("events must contain only 0 and 1")

    n_q = h.shape[1]
    ends = []
    for q in range(n_q):
        rows = np.flatnonzero(h[:, q])
        if len(rows) not in (1, 2):
            raise ValueError(
                f"qubit {q} is checked by {len(rows)} checks; need one or two"
            )
        ends.append(tuple(int(r) for r in rows))
    pairs = [e for e in ends if len(e) == 2]
    if len(set(pairs)) != len(pairs):
        raise ValueError("two qubits join the same pair of checks")

    mean = v.mean(axis=0)
    p = np.zeros(n_q)
    for q, e in enumerate(ends):
        if len(e) == 2:
            i, j = e
            cov = (v[:, i] * v[:, j]).mean() - mean[i] * mean[j]
            xor = np.abs(v[:, i] - v[:, j]).mean()
            denom = 1.0 - 2.0 * xor
            inside = 0.25 - cov / denom if denom > 0 else 0.25
            p[q] = 0.5 - np.sqrt(max(inside, 0.0))
    for q, e in enumerate(ends):
        if len(e) == 1:
            i = e[0]
            prod = 1.0
            for r, other in enumerate(ends):
                if len(other) == 2 and i in other:
                    prod *= 1.0 - 2.0 * p[r]
            if prod <= 0:
                p[q] = 0.5
            else:
                p[q] = 0.5 + (mean[i] - 0.5) / prod
    return np.clip(p, 0.0, 0.5)
