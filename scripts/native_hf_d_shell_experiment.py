"""
Real, end-to-end check of native_hf's d-shell (angular momentum L=2)
support: builds the overlap/core-Hamiltonian/electron-repulsion matrices
and runs RHF/6-31G* on a neon atom -- 6-31G* adds one real Cartesian d
shell (dxx,dyy,dzz,dxy,dxz,dyz) on top of Ne's s/p valence shells, so
this exercises the actual new capability, not just the individual
integral functions in isolation.

The main library's own fix (dense_evolution/native_hf/{cartesian,basis,
assembly}.py) generalized the pre-existing Obara-Saika recursions --
already correct for any angular momentum, verified by direct inspection
-- to shells beyond s/p by (1) generalizing the hardcoded Cartesian-
component count and (2) applying the per-Cartesian-component
normalization ratio (dxx and dxy have different norms; verified sqrt(3)
via overlap_3d's own self-overlap, matching the standard closed-form
sqrt((2L-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!)) formula) that a single
shared reference normalization missed. The main repo's own test suite
covers the fast, unit-level checks (raw d-orbital integrals against
independent scipy quadrature, core Hamiltonian symmetry) -- this script
covers the one thing that's too slow to run on every push: assembling
the FULL electron-repulsion tensor for a real mixed s/p/d basis and
running SCF to convergence.

Real, measured cost worth recording here: building the ERI tensor took
~10 minutes on the machine this was first run on -- not a hang, but a
genuine scaling cost of the existing per-shell-quartet JIT-caching
architecture (dense_evolution/native_hf/assembly.py's _quartet_block)
once 3 distinct angular momenta (s,p,d) are mixed instead of 2: Ne/6-31G*
needs 35 distinct (degree_a,degree_b,degree_c,degree_d) JIT
specializations instead of the handful a pure s/p basis needs, and each
is individually fast (2-6s) but they add up. Optimizing that is a
separate, not-yet-started piece of work (its own before/after benchmark,
not a preamble to this one) -- see the main repo's prog.txt.
"""
import time

import numpy as np

import dense_evolution as de
from dense_evolution.native_hf.basis import build_molecule_shells, n_cartesian_functions
from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor
from dense_evolution.native_hf.scf import run_scf

# The well-established numerical (complete-basis-set) non-relativistic
# Hartree-Fock limit for the neon atom (Clementi & Roetti 1974,
# Atomic Data and Nuclear Data Tables 14, 177 -- the standard reference
# atomic HF energy table). Any finite-basis RHF energy, including the
# 6-31G* result below, must be >= this value by the variational
# principle -- a necessary but not sufficient correctness check (it
# only rules out gross errors, not a subtly-wrong d-shell integral that
# happens to still land above this bound). A basis-specific frozen
# reference value (e.g. from PySCF) would be the stronger anchor this
# script doesn't yet have -- see the note at the bottom of the docs page.
NE_NUMERICAL_HF_LIMIT = -128.5470


def main():
    geometry_bohr = np.array([[0.0, 0.0, 0.0]])

    shells = build_molecule_shells([10], geometry_bohr, "6-31g*")
    degrees = sorted({s.degree for s in shells})
    print(f"shells: {len(shells)}, degrees present: {degrees}, AO functions: {n_cartesian_functions(shells)}")
    assert 2 in degrees, "expected a real d shell (degree 2) from 6-31G* on neon"

    t0 = time.time()
    S = build_overlap_matrix(shells)
    t_overlap = time.time() - t0
    print(f"overlap matrix: {t_overlap:.1f}s, diag min/max = {np.diag(S).min():.10f}/{np.diag(S).max():.10f}")
    assert np.allclose(S, S.T, atol=1e-9)
    assert np.allclose(np.diag(S), 1.0, atol=1e-8)

    t0 = time.time()
    H_core = build_core_hamiltonian(shells, [10.0], geometry_bohr)
    t_core = time.time() - t0
    print(f"core Hamiltonian: {t_core:.1f}s")
    assert np.allclose(H_core, H_core.T, atol=1e-9)

    t0 = time.time()
    V = build_repulsion_tensor(shells)
    t_eri = time.time() - t0
    print(f"electron repulsion tensor: {t_eri:.1f}s")
    assert np.allclose(V, V.transpose(1, 0, 2, 3), atol=1e-9)
    assert np.allclose(V, V.transpose(0, 1, 3, 2), atol=1e-9)
    assert np.allclose(V, V.transpose(2, 3, 0, 1), atol=1e-9)
    assert np.allclose(V, V.transpose(3, 2, 1, 0), atol=1e-9)

    t0 = time.time()
    result = run_scf(S, H_core, V, 10, [10.0], geometry_bohr)
    t_scf = time.time() - t0
    print(f"SCF: {t_scf:.1f}s, converged={result.converged}, iterations={result.n_iterations}")
    print(f"total energy: {result.total_energy}")

    assert result.converged
    assert result.total_energy > NE_NUMERICAL_HF_LIMIT, (
        f"{result.total_energy} is below the numerical HF limit {NE_NUMERICAL_HF_LIMIT} "
        f"-- violates the variational principle, a real bug"
    )

    print(f"\nvariational check: {result.total_energy} > {NE_NUMERICAL_HF_LIMIT} (numerical HF limit) -- OK")
    print(f"total wall time: {t_overlap + t_core + t_eri + t_scf:.1f}s")


if __name__ == "__main__":
    main()
