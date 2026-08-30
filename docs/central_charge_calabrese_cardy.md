# Central Charge from Entanglement Entropy: Calabrese-Cardy Confirmed, With a Real Lesson About "Critical"

Tests whether the critical transverse-field Ising model's ground-state entanglement entropy really follows the open-chain CFT prediction (Calabrese & Cardy, J. Stat. Mech. 2004, P06002; Tong SS4.4.3 "c is for Cardy"):

    S(L) = (c/6) * ln[ (2N/pi) * sin(pi*L/N) ] + const

and whether fitting it recovers the known Ising CFT central charge c=1/2.

**Formula verified directly against the paper's own text** (now indexed in `quantumrag`'s `quantum_info` collection, `calabrese_cardy_2004_entanglement_entropy_qft.pdf`): eq. (4) confirms the single-boundary-point asymptotic form $S_A \sim (c/6)\log(\ell/a)$, and eq. (19) gives the finite periodic-system form $(c/3)\log[(L/\pi a)\sin(\pi\ell/L)]$ this experiment's open-chain formula is derived from via the standard doubling trick (an open chain of length $N$ unfolds to a periodic chain of length $2N$ with a reflection symmetry; a boundary-touching subsystem of length $L$ maps to half of a periodic interval of length $2L$, giving $(c/6)\log[(2N/\pi)\sin(\pi L/N)]$).

## Two independent validations

**1. Computation check**: `dense_evolution.partial_trace`/`von_neumann_entropy` on the many-body Lanczos ground state is cross-checked against a completely independent method -- free-fermion (Jordan-Wigner + Bogoliubov-de Gennes) exact diagonalization via Peschel's formula (J. Phys. A 36, L205, 2003) for the entanglement entropy of a Gaussian fermionic state through its Majorana covariance matrix. The Majorana-correlator algebra was self-tested against brute-force many-body ED at N=6 (diff ~1e-15) before being trusted at N=12. At N=12, the two independent methods agree to ~1e-10 across every L tested.

**2. Critical-point check -- a real methodological pitfall found and fixed**: `ising_exact_verification.py`'s `g*=0.8600` is the finite-size **susceptibility-peak** location -- the right point for that script's `<ZZ>` ansatz-tracking purpose, but **not** the same as the textbook **self-dual CFT critical point g=1.0** (`H = -sum ZZ - g*sum X` is Kramers-Wannier self-dual at g=1 in the thermodynamic limit). At finite N these two different notions of "critical" do not coincide.

## Result

| g | meaning | extracted c | R² | vs theory (0.5) |
|---|---|---|---|---|
| 1.0 | self-dual CFT point | **0.565** | 0.999924 | Δ=0.065 |
| 0.86 | susceptibility peak (finite-size) | 0.983 | 0.999997 | Δ=0.483 |
| 1.8 | off-critical (negative control) | 0.014 | 0.912692 | -- (should not fit) |

Fitting at `g*=0.86` gives an almost perfectly clean fit (R²=0.999997) to a **wrong** answer -- extracted c≈0.98, roughly double the true 0.5. It looks even more convincing than the correct point's fit, which is exactly why this is worth documenting: a high R² alone does not mean the extracted physics is right if the wrong reference point was used. At the true self-dual point g=1.0, extracted c=0.565 -- much closer to 0.5, with the residual ~13% gap a plausible finite-size correction at this modest N=12 (the CFT formula is an asymptotic large-N/L result). The off-critical negative control behaves correctly: near-zero extracted c and a visibly worse fit (R²=0.91), consistent with area-law saturation instead of CFT log-scaling.

## Status

Confirmed, not a confound (unlike Experiment 35) -- the CFT prediction genuinely holds at the correct critical point, and the methodological lesson (susceptibility-peak != CFT point at finite size) is itself a real, reusable finding for any future work in this repo that needs "the critical point" of this specific N=12 open TFIM chain.

**Not yet done**: validating `dense_evolution.MPSSimulator`'s JSD-budget bond-truncation against this benchmark (does truncation bias the extracted central charge?) -- the original motivation for building this. Left as a follow-up: doing it properly requires isolating truncation error from state-preparation error (MPS is circuit-based, so reaching the ground state needs a real preparation circuit, e.g. VQE or Trotterized adiabatic evolution, not just loading the exact statevector), which is a separate, real piece of work.

## Reproduce

```bash
python scripts/central_charge_calabrese_cardy.py
```