# native_hf: d-Shell (Angular Momentum L=2) Support

!!! note
    The implementation lives in the main library:
    `dense_evolution.native_hf` (not yet released to PyPI at the time this
    page was written -- see the note at the bottom). This page is the
    experimental log for the one thing too slow to run on every push to
    the main repo: assembling a full electron-repulsion tensor for a real
    mixed s/p/d basis and running SCF to convergence.

**In plain terms**: `native_hf` computes molecular integrals from Gaussian basis functions. Until now it only handled s and p orbitals (spherical and dumbbell-shaped) -- d orbitals (the four-lobed ones used by every polarized basis set, and by any first-row transition metal) raised `NotImplementedError`. The underlying math (Obara-Saika recursion) already worked for any angular momentum; the two real gaps were a hardcoded orbital-count table and a missing per-orbital normalization correction that only matters once you go past p. This page runs the fixed code end to end on a real d-containing basis and checks the result against physics, not just against itself.

## Why d orbitals are harder than s and p

A Cartesian d shell has six components: dxx, dyy, dzz, dxy, dxz, dyz. For s
and p shells, every component of a given shell has the same normalization
constant by symmetry (px, py, pz are interchangeable). That stops being
true at d: dxx and dxy are *not* interchangeable (dxx points along one
axis, dxy is diagonal), so they don't have the same self-overlap, and
naively reusing the s/p-era "one normalization for the whole shell" logic
silently mis-scales every off-diagonal Cartesian component.

## Step 1. The normalization ratio, checked against the library's own overlap integral

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dense_evolution.native_hf.gaussians import GaussianShell3D
from dense_evolution.native_hf.overlap import overlap_3d

g = GaussianShell3D(degree=2, exponent=jnp.asarray(0.9), center=jnp.zeros(3))
S = overlap_3d(g, g)
ratio = (S[2,0,0,2,0,0] / S[1,1,0,1,1,0]) ** 0.5
print(float(ratio))
```

```
1.7320508075688774
```

Exactly √3, to machine precision, independent of the exponent (checked at
0.3, 0.9, 5.7) -- matching the standard closed-form result
`sqrt((2L-1)!! / ((2lx-1)!!(2ly-1)!!(2lz-1)!!))` for the normalization
ratio between two Cartesian components of the same shell. The main repo's
fix applies this ratio at assembly time (not baked into the shared
per-primitive contraction coefficients, which are a genuine physical
fact about the basis set, not something to overload with a
component-specific correction).

## Step 2. An independent cross-check: raw numerical quadrature

Not just self-consistency against the library's own machinery -- `dxx`
and `dxy` self-overlap checked against `scipy.integrate.quad` on the raw
Gaussian polynomials directly, no `dense_evolution` code involved at all:

```python
from scipy import integrate
import numpy as np

a = 0.9
Ix4, _ = integrate.quad(lambda x: x**4 * np.exp(-2*a*x*x), -np.inf, np.inf)
Ix2, _ = integrate.quad(lambda x: x**2 * np.exp(-2*a*x*x), -np.inf, np.inf)
I0, _  = integrate.quad(lambda x: np.exp(-2*a*x*x), -np.inf, np.inf)

dxx_quad = Ix4 * I0 * I0
dxy_quad = Ix2 * Ix2 * I0
print(dxx_quad, dxy_quad)
```

```
0.5337431379428554 0.1779143793142851
```

Matches `overlap_3d`'s own `(2,0,0,2,0,0)` and `(1,1,0,1,1,0)` entries
exactly (`np.isclose(..., rtol=1e-10)` is `True` for both).

## Step 3. Real molecule: neon, 6-31G*, full SCF

```python
import dense_evolution as de
from dense_evolution.native_hf.basis import build_molecule_shells, n_cartesian_functions
from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor
from dense_evolution.native_hf.scf import run_scf
import numpy as np

geometry_bohr = np.array([[0.0, 0.0, 0.0]])
shells = build_molecule_shells([10], geometry_bohr, "6-31g*")
S = build_overlap_matrix(shells)
H_core = build_core_hamiltonian(shells, [10.0], geometry_bohr)
V = build_repulsion_tensor(shells)
result = run_scf(S, H_core, V, 10, [10.0], geometry_bohr)
```

6-31G* gives neon 6 shells (s, s, p, s, p, **d**) -- 15 Cartesian AO
functions total, including the 6 real d components. Real results from
this run:

| Check | Result |
|---|---|
| Overlap matrix diagonal | exactly 1.0 on all 15 AOs, including all 6 d components |
| Overlap matrix symmetric | yes |
| Core Hamiltonian symmetric | yes |
| ERI 8-fold permutational symmetry | holds on all 4 independent transposes |
| SCF converged | yes, 10 iterations |
| Total energy | **-128.4744065199038 Hartree** |

## Step 4. The real anchor: PySCF, and the 5d/6d gotcha it caught

Run on Google Colab (`!pip install pyscf --quiet`, a real C++ toolchain
this machine doesn't have):

```python
from pyscf import gto, scf

mol = gto.M(atom="Ne 0 0 0", basis="6-31g*", charge=0, spin=0, unit="Bohr")
mf = scf.RHF(mol)
print(mf.kernel())
```

```
converged SCF energy = -128.473876870668
```

That's **-128.47387687066836** against this implementation's
**-128.4744065199038** -- a ~5e-4 Hartree discrepancy. Exactly the
subtle, not-caught-by-the-variational-bound kind of wrongness worth
worrying about (still safely above the -128.5470 numerical HF limit,
still looking fine by Step 3's own checks). Before concluding this
implementation has a real bug, one alternative explanation: PySCF
defaults to **spherical** d functions (5 components -- the physical
d-orbital basis, dropping the linear combination that transforms like an
s function) unless told otherwise, while this implementation only
supports **Cartesian** d functions (6 components: dxx,dyy,dzz,dxy,dxz,dyz)
-- the classic 5d-vs-6d basis-set convention mismatch, not a bug in
either implementation, just two different (both valid) definitions of
"the d part of 6-31G*".

Forcing PySCF to match this implementation's convention:

```python
mol = gto.M(atom="Ne 0 0 0", basis="6-31g*", charge=0, spin=0,
            unit="Bohr", cart=True)
mf = scf.RHF(mol)
print(mf.kernel(), mol.nao)
```

```
converged SCF energy = -128.474406519905
nao = 15
```

`nao=15` confirms the AO count now matches this script's own count
exactly (6 Cartesian d components, not 5 spherical). And the energy:
**-128.47440651990485** against this implementation's
**-128.4744065199038** -- agreement to **~1e-12 Hartree**, machine
precision. The hypothesis was right: the first run's ~5e-4 discrepancy
was purely the spherical/Cartesian convention mismatch, not a bug. This
is now frozen in the script as `NE_631GSTAR_PYSCF_RHF_ENERGY` and
checked with `abs=1e-6` (a real, basis-specific external anchor -- the
same kind the main repo's H₂/H₂O golden tests use, not available when
this page was first written; see "PySCF, resolved" below).

## Step 5. The variational bound, kept as a secondary check

The neon atom's numerical (complete-basis-set) non-relativistic
Hartree-Fock limit is a well-established reference value: **-128.5470
Hartree** (Clementi & Roetti 1974, *Atomic Data and Nuclear Data Tables*
14, 177 -- the standard atomic HF reference table). Any finite-basis RHF
energy must sit *above* (less negative than) this limit by the
variational principle:

```
-128.4744065199038 > -128.5470   # True
```

On its own this would only rule out gross errors (wrong sign, wrong
magnitude, a missing factor) -- Step 4's PySCF anchor is what actually
confirms correctness at the millihartree scale that would matter for a
real quantum-chemistry application. Both checks run in the script.

## Real timing (one run on this machine)

| Step | Time |
|---|---|
| Overlap matrix | 11 s |
| Core Hamiltonian | 48 s |
| Electron repulsion tensor | 623 s |
| SCF (10 iterations) | 0.04 s |

The ERI step dominates. Not a hang -- a genuine scaling cost of the main
library's existing per-shell-quartet JIT-caching architecture
(`assembly.py`'s `_quartet_block`, `static_argnames=("degrees", ...)`):
a pure s/p basis needs at most `2^4=16` distinct
`(degree_a,degree_b,degree_c,degree_d)` JIT specializations, but mixing
in d pushes that to `3^4=81` possible combinations (35 actually needed
for this specific 6-shell system). Each individual compile is fast
(2-6 s, checked in isolation), but they add up across the shell-quartet
loop. Optimizing this is real, separate work with its own before/after
benchmark -- not started here, and not a prerequisite for the
correctness result above.

---

## Details

### PySCF, resolved

This machine itself has no C++ toolchain (`pyscf`'s and `pyquante2`'s
wheel builds both failed on `Microsoft Visual C++ 14.0` requirements), no
usable WSL Linux distribution (only Docker Desktop's internal, Python-
free distro was present), no Docker CLI, and no conda -- a targeted web
search for a published HF/6-31G* neon value also came up empty (it is a
less commonly tabulated number than the H₂/H₂O values the main repo's
own golden tests already anchor against). Google Colab has a real C++
toolchain and installs `pyscf` in seconds, which is where Step 4's two
runs above were actually produced.

### Why this lives here and not in the main repo's CI

The main repo's own test suite (`tests/unit/test_native_hf_engine.py`)
covers the fast checks from Steps 1-2 (microseconds to seconds each) on
every push. The full Step 3 run (~11 minutes) would add roughly an hour
of CI time per push across the 6-job platform matrix, for a check that's
still only a necessary-not-sufficient correctness signal (see Step 4) --
not a cost worth paying on every one-line PR. This experiment log is
where that full run lives instead, checked here rather than gating the
library's own CI.

### Not yet importable from PyPI at the time this was written

This script imports `dense_evolution.native_hf` functions that, as of
this page's writing, exist only on a feature branch of the main repo
(not yet merged to `main`, and not yet part of any PyPI release --
`dense-evolution` is at 8.1.73 on PyPI, this needs at least 8.1.74).
`requirements-lock.txt` pins `dense-evolution==8.1.67`, older still.
Running this script today requires an editable/local install of the
main repo's feature branch; it will run against a real PyPI release
once one ships. No dedicated `tests/test_native_hf_d_shell_experiment.py`
is added yet for the same reason -- this repo's own CI installs
`dense-evolution` from PyPI (see `requirements-ci.txt`) and would have
nothing real to import against otherwise.

## See Also

- [Chunk: Multi-Device and Disk-Backed Simulation](chunk_distributed_disk_experiment.md) -- another experiment log built on top of a main-library feature, same page structure.
