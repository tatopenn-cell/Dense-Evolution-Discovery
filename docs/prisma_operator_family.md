# Continuous and discrete projections of a Z₂-symmetric operator family

**Salvatore Pennacchio** — Independent Researcher — September 2026

---

## Abstract

We present a mathematical bridge between two independently developed regularization frameworks for the quantum vacuum: the *Dynamic Vacuum Pressure Theory* (DVPT), which treats the vacuum as a two-state-per-mode system with a `cosh`-shaped energy gap and an attractor solution `sech(r/ℓ₀)`, and the *Golden Dynamic Space* framework (GDS), which studies discrete-spectrum structures on Aubry-André lattices with the golden ratio Φ as incommensurability parameter. Both emerge as limits of a single Z₂-symmetric operator family

```
H(μ, ε) = −d²/dr² + ε·cos(2παr) − 2μ·sech²(r)
```

with incommensurability parameter `α = 1/Φ`. The DVPT limit is recovered at `ε = 0, μ = 1` (Pöschl-Teller), where the ground state equals `sech(r)` to numerical precision (`E₀ = −1.000016`, correlation `1.00000000`). The GDS limit is recovered at `μ = 0` (pure Aubry-André), where the volume-corrected Inverse Participation Ratio exhibits the standard extended-to-localized transition (`0.0046 → 0.2037 → 0.9329`). A critical transition between the two regimes occurs at `μ = 1`, where the oscillation period of the associated nonlinear ODE diverges. The golden ratio Φ appears as a recurring constant across both frameworks but is not the unique invariant of the family. A scan of log-periodic lattice ratios `λ ∈ [1.5, 1.8]` locates the maximum mixedness (IPR ≈ 0.5) at `λ_opt = 1.700`, not at `Φ = 1.618` (5% difference), and no non-trivial invariant remains constant across the family. **No physical prediction is claimed.**

This is the third of a set of three companion records published on the same repository. The first, *The Dynamic Vacuum Pressure Theory* [1], proposes a physical model of the vacuum's two-state-per-mode structure. The second, *The Golden Dynamic Space Framework* [2], provides the computational machinery for discrete-spectrum regularization hypotheses. Neither depends on the other, and neither depends on this paper; the present work establishes that the two share a common mathematical skeleton, at the level of a family of self-adjoint operators on `L²(ℝ)`.

**Scope.** The DVPT limit of the family is the Pöschl-Teller Hamiltonian [4], textbook material since 1933. The GDS limit is the Aubry-André Hamiltonian [3], textbook material since 1980. What is not textbook is the observation that the two arise as limits of a single self-adjoint operator family — this is the contribution of the present paper, and it is an algebraic fact about a family of Schrödinger operators, not a physical claim. It should be read as a mathematical observation about the structure of two established models, not as a proposal that either model describes the physical vacuum.

---

## 1. Introduction

### 1.1 Two frameworks, one origin?

The Dynamic Vacuum Pressure Theory (DVPT) proposes that the vacuum's energy gap grows with a two-state partition function per mode, `Z_mode(r) = 2cosh(Δ(r))`, with `Δ(r) = α_G + (r/ℓ₀)^n`. Its dynamical attractor, obtained from the persistence functional

```
P[u] = ∫ [½(u')² + ½u²(1 − u²)] dr
```

is the soliton `u(r) = sech(r/ℓ₀)`, reached with correlation 0.995 from a subset of initial conditions (see DVPT §5.5). The 0.995 figure is the directly reproduced result from a multi-initial-condition study; a narrower single-initial-condition test converges to higher correlation, but the honest figure for the general problem is 0.995, not 1.0. This residual is noted here because the exact numerical agreement of the DVPT limit below (`corr = 1.00000000`) uses a different, cleaner calculation — the direct diagonalization of the Pöschl-Teller Hamiltonian on a symmetric lattice — which does not have the relaxation dynamics' grid artefacts.

The Golden Dynamic Space framework (GDS) [2] studies discrete-spectrum structures on Aubry-André lattices with incommensurability `α = 1/Φ`, and derives three exact constants: `d_f = ln(Φ)/ln(2)`, `β = 2π/ln(Φ)`, and `ε_crit = 2/(3√3)`. Its central result is a numerical verification of the Montgomery-Odlyzko correspondence between the Riemann zeros and the Gaussian Unitary Ensemble (see GDS §3).

Both frameworks share the golden ratio Φ and a Z₂ symmetry structure: the parity `r → −r` for DVPT's `sech`, and the local reflection symmetry of the Aubry-André potential for GDS. Neither framework derives the other.

This paper asks whether the shared structure is more than analogy. Specifically: do DVPT and GDS emerge as limits of a single, mathematically well-defined operator family?

### 1.2 Structure

Section 2 introduces the operator family `H(μ, ε)`. Sections 3 and 4 verify that DVPT and GDS are recovered in the appropriate limits. Section 5 identifies the critical transition at `μ = 1`. Section 6 discusses the role of Φ and reports the negative result that it is not the unique invariant. Section 7 provides the epistemic-status table. Section 8 states the limits of the result. Section 9 concludes.

---

## 2. The Z₂-symmetric operator family

We consider the one-dimensional Schrödinger-type operator

```
H(μ, ε) = −d²/dr² + ε·cos(2παr) − 2μ·sech²(r)
```

acting on `L²(ℝ)`, with `α = 1/Φ ≈ 0.618` and control parameters `μ ≥ 0` (Pöschl-Teller strength) and `ε ≥ 0` (Aubry-André strength). The operator is self-adjoint for real `μ, ε`; its spectrum is bounded below.

The family has two distinguished limits:

- **DVPT limit**: `μ = 1, ε = 0`. The operator reduces to the Pöschl-Teller Hamiltonian `H_PT = −d²/dr² − 2sech²(r)` [4], whose unique bound state has energy `E₀ = −1` and wavefunction `ψ₀(r) = sech(r)`.

- **GDS limit**: `μ = 0`. The operator reduces to the Aubry-André Hamiltonian `H_AA = −d²/dr² + ε·cos(2παr)` [3], whose spectrum undergoes an extended-to-localized transition at `ε = 2` (in units where the discrete hopping is `J = 1`).

Both limits have been studied independently in their respective frameworks. The question of this paper is whether they are connected by a continuous path in `(μ, ε)`.

**Scope of the Z₂ symmetry statement.** Strictly speaking, the operator `H(μ, ε)` commutes with parity `P` only if the AA potential `cos(2παr)` is itself even about `r = 0`, which it is only for special values of `α` and only on the discrete lattice. In the continuum embedding used here, `cos(2παr)` is not globally even. What *is* preserved for all `(μ, ε)` is the local Z₂ symmetry in a neighbourhood of the origin, which is the sense in which the family is "Z₂-symmetric" throughout this paper and the companion papers. The reader should not infer a global symmetry that the family does not possess.

---

## 3. Recovery of the DVPT limit

At `ε = 0, μ = 1`, the operator is the textbook Pöschl-Teller Hamiltonian

```
H_PT = −d²/dr² − 2sech²(r)
```

with the well-known bound state `ψ₀(r) = sech(r)`, `E₀ = −1` [4].

We solve numerically on a symmetric lattice `r ∈ [−20, 20]` with `N = 2000` points, using `scipy.linalg.eigh`. The result:

```
Ground state energy:       E₀    = −1.000016      (exact: −1)
Correlation with sech(r):  corr  = 1.00000000
Overlap with sech(r):      ⟨ψ₀|sech⟩ = 1.00000000
```

The ground state reproduces `sech(r)` to numerical precision. The residual `1.6×10⁻⁵` in the energy is a finite-lattice artefact and disappears as `N → ∞`. This is expected behaviour for the Pöschl-Teller Hamiltonian, whose ground state is exactly solvable and known in closed form.

**DVPT is recovered exactly** as the `ε = 0, μ = 1` limit of the family. This is a consistency check of a textbook result, not a new derivation: the ground state of Pöschl-Teller *is* `sech(r)`, and the family `H(μ, ε)` was constructed with this property in mind.

---

## 4. Recovery of the GDS limit

At `μ = 0`, the operator reduces to the Aubry-André Hamiltonian on a lattice of `N = 400` sites, with `α = 1/Φ` and hopping `J = 1`. We compute the ground state and the volume-corrected Inverse Participation Ratio

```
IPR = Σ_r |ψ(r)|⁴      (with r² Jacobian weighting)
```

for three values of `ε`:

| ε/J | IPR (3D-corrected) | Regime |
|---|---|---|
| 0.5 | 0.004567 | Extended |
| 2.0 | 0.203739 | Critical |
| 4.5 | 0.932858 | Localized |

The transition follows the standard Aubry-André result [3]: sharp, monotonic, and consistent with the analytical location of the critical point at `ε = 2J`. The `r²` Jacobian weighting, applied to the radial interpretation of the lattice index, shifts the numerical values slightly relative to the pure one-dimensional case but preserves the qualitative behaviour.

**GDS is recovered** as the `μ = 0` limit of the family. As with Section 3, this is a consistency check of a known result — the AA transition is textbook — not a derivation of the transition itself.

---

## 5. The critical transition at μ = 1

Having verified the two limits, we ask whether the family is connected by a continuous path. Two independent tests give the same answer.

### 5.1 Period divergence in the associated ODE

The nonlinear ODE

```
u''(r) = μ·u(r) − 2u³(r),      u(0) = 1, u'(0) = 0
```

has the following behaviour as a function of `μ`:

- `μ = 1`: exact solution `u(r) = sech(r)` (separatrix, non-periodic)
- `μ < 1`: localized oscillation about `u = 0`
- `μ > 1`: oscillation about `u = ±√μ`

The period `T(μ)` of the oscillation diverges as `μ → 1⁻`:

```
μ = 0.500  →  T =  6.627
μ = 0.845  →  T =  9.081
μ = 0.999  →  T = 19.357   (diverging)
```

The transition at `μ = 1` is the critical point at which the geometry changes character. This is the same `μ = 1` that appears in the DVPT limit of the operator family, and the same critical value that separates the `sech` regime of DVPT from the AA regime of GDS.

### 5.2 Ground-state interpolation in the operator family

For the full family `H(μ, ε)` with `α = 1/Φ`, scanning `ε` at fixed `μ = 1` gives a smooth interpolation between the two limits:

| ε | IPR | corr(sech) | E₀ |
|---|---|---|---|
| 0.00 | 0.0167 | 1.000 | −1.000 |
| 1.00 | 0.0159 | 0.998 | −1.010 |
| 5.00 | 0.0173 | 0.967 | −1.744 |
| 10.00 | 0.0232 | 0.905 | −3.850 |
| 20.00 | 0.0315 | 0.805 | −9.977 |
| 50.00 | 0.0420 | 0.690 | −32.750 |

The ground state retains a substantial `sech`-like component throughout, but the AA correction grows monotonically. The transition is continuous, not first-order.

**The two limits are connected by a continuous path in the family.**

---

## 6. The role of Φ: recurring but not unique

The golden ratio `Φ = 1.618…` appears as the incommensurability parameter `α = 1/Φ` in the Aubry-André potential, and it appears in three exact constants of GDS (`d_f`, `β`, `ε_crit`). A natural question is whether Φ is the unique invariant of the family `H(μ, ε)` — that is, whether it is singled out by some extremal property of the spectrum or ground state.

We test this by scanning the log-periodic lattice ratio `λ` in `[1.5, 1.8]` and searching for the value that maximizes the "mixedness" of the ground state, defined as the value closest to `IPR = 0.5`.

**Result.**

```
λ_opt (IPR ≈ 0.5) = 1.7000
Φ                 = 1.6180
|λ_opt − Φ|       = 0.0820      (5.0% difference)
```

Φ is **not** the value that maximizes mixedness. Several other lattice ratios produce the same effect, and the maximum is at `λ ≈ 1.70`.

We also test whether the "width at 1/e" of the ground state equals Φ at any critical value `μ*`:

```
μ* (width = Φ) = 0.980013
1/Φ            = 0.618034
|μ* − 1/Φ|     = 0.3620
```

The value `μ* ≈ 0.98` is close to `μ = 1`, not to `1/Φ`. The natural comparison is with `arccosh(e) = 1.657`, the exact width of `sech(r)` at `1/e`; even there, the difference from Φ is 0.039 (2.4%).

**Conclusion.** Φ is a recurring constant in the family, but it is **not** the unique invariant. The family admits infinitely many mathematically equivalent interpolations between DVPT and GDS, and Φ is one distinguished parameter among them.

**Why this result matters in context.** In the companion DVPT paper [1], Φ is proposed as a candidate answer to an *open question* raised by Sornette's review of discrete scale invariance [5], §8.7: *what fixes the preferred scaling ratio λ?* Sornette documents an empirical convergence toward λ ≈ 2 in growth processes, rupture, and earthquakes, and λ ≈ 1.7 in biological evolution, without any principle fixing the value. The present negative result should be read as a **constraint on that proposal**: Φ is consistent with the empirical range, but it is not singled out by an extremal property of the operator family that unifies DVPT and GDS. If Φ is *the* preferred ratio, the reason must lie elsewhere — in the physics of the two frameworks, not in the mathematics of the bridge between them.

This is a negative result, recorded as such.

---

## 7. Epistemic status of every claim

Every claim in this paper is classified as **derived** (follows from a closed-form calculation or a proven theorem), **verified** (confirmed numerically against an independent reference), **cited** (established in the literature and used here without modification), **proposed** (a candidate offered but not derived), **refuted** (checked directly and found false — kept in the table because a negative result is still a result), or **not found** (checked directly, no such object identified).

| Claim | Status | Section | Comment |
|---|---|---|---|
| Family `H(μ, ε)` is well-defined | Derived | 2 | Self-adjoint Schrödinger operator on L²(ℝ) |
| Family is Z₂-symmetric | Derived | 2 | Local parity symmetry about the origin; global symmetry does not hold |
| Pöschl-Teller ground state is `sech(r)` | Cited | 3 | Textbook result [4] |
| DVPT recovered at `μ = 1, ε = 0` | Verified | 3 | `E₀ = −1.000016`, corr = 1.00000000 |
| Aubry-André transition at `ε = 2J` | Cited | 4 | Textbook result [3] |
| GDS recovered at `μ = 0` | Verified | 4 | IPR 0.0046 → 0.2037 → 0.9329 |
| Transition at `μ = 1` | Derived | 5.1 | Period of associated ODE diverges |
| Continuous interpolation between limits | Verified | 5.2 | Smooth scan in ε, no first-order jump |
| Φ is the unique invariant of the family | Refuted | 6 | λ_opt = 1.70, not Φ |
| Φ is a recurring constant in both frameworks | Derived | 6 | `d_f`, `β`, `ε_crit` for GDS; appears as α for AA |
| Φ as candidate for Sornette's preferred λ | Proposed | 6 | Consistent with empirical range but not extremal; the paper's negative result is a constraint on this proposal |
| Existence of a non-trivial invariant | Not found | 6 | No quantity remains constant across the family |

The family is real. The connection between DVPT and GDS is real. The "prism" metaphor — that Φ is the unique lens through which the two projections cohere — does **not** hold.

---

## 8. Limits and open directions

The result of this paper has five explicit limitations, each stated here rather than left for a reader to infer.

**No physical prediction.** The family `H(μ, ε)` is a mathematical construction. Nothing in this paper identifies it with the physical vacuum, with spacetime, or with any observable. The DVPT and GDS limits are recovered as mathematical limits of a self-adjoint operator; whether either limit corresponds to physical reality is a question addressed in the two primary papers, not here.

**The choice of `sech²` as the Pöschl-Teller potential.** Other choices (e.g. `μ·sech²(r)` with a different exponent, or a Gaussian profile) may give different results. The `sech²` form was chosen because it makes the DVPT limit exact and because it is the standard solvable case. The family is not unique.

**The Aubry-André structure on a continuous radial coordinate.** In GDS, AA lives on integer sites. Here it is embedded in a continuous `r`, which is a formal extension. Whether this embedding preserves the discrete-spectrum structure of AA is not investigated. A separate test would require solving the AA Hamiltonian on a discrete lattice and comparing the spectral statistics to those of the continuum version. Preliminary checks suggest the embedding preserves the ground-state IPR to the precision quoted, but the full spectral statistics are not examined.

**The role of Φ beyond its recurrence.** Φ appears in the AA incommensurability, in the GDS constants, and as a candidate answer to Sornette's open question about the preferred scaling ratio [5]. Whether this recurrence reflects a deep structure or a mathematical coincidence of hyperbolic and quasiperiodic functions is not resolved. The negative result of Section 6 suggests the latter is at least as plausible as the former: Φ is *not* singled out by any extremal property of the operator family, and the preferred lattice ratio that maximizes mixedness is `λ ≈ 1.70`, not `λ = Φ`.

**The search for invariants.** The negative result of Section 6 does not exclude the possibility that a non-trivial invariant exists but is not visible at the level of the ground state alone. It might appear in higher eigenstates, in the spectral density, or in a topological property of the family. This is a direction for future work.

---

## 9. Conclusion

We have shown that the *Dynamic Vacuum Pressure Theory* and the *Golden Dynamic Space* framework emerge as limits of a single Z₂-symmetric operator family

```
H(μ, ε) = −d²/dr² + ε·cos(2παr) − 2μ·sech²(r)
```

with the DVPT limit at `(μ, ε) = (1, 0)` and the GDS limit at `μ = 0`. The transition between the two is continuous and critical at `μ = 1`, where the associated nonlinear ODE has a separatrix. The golden ratio Φ appears as a recurring constant across the family but is **not** the unique invariant that ties the two frameworks together.

The result is a mathematical bridge, not a physical unification. It shows that the two frameworks are not independent constructions that happen to share notation: they are two projections of a common operator family. It also shows that the bridge does not single out Φ as the fundamental scale of that family.

**The paper's contribution in context.** The two limits — Pöschl-Teller [4] and Aubry-André [3] — are textbook results, and the present paper does not re-derive either. What it establishes is that the two arise as limits of a single self-adjoint operator family, and that the family admits a critical transition at `μ = 1` where the associated nonlinear ODE becomes a separatrix. This is an algebraic observation about a family of Schrödinger operators. It is offered as a contribution to the study of mathematical structures that appear repeatedly in discrete-spectrum regularization, and it is offered as mathematics — not as a physical unification and not as evidence that either limit describes the physical vacuum.

This work is published as a companion to the two primary papers, archived on Zenodo with a permanent DOI, as a citable record of what the shared mathematical structure supports and what it does not.

---

## References

1. S. Pennacchio, *The Dynamic Vacuum Pressure Theory*, Dense-Evolution-Discovery, Zenodo (2026).
2. S. Pennacchio, *The Golden Dynamic Space Framework*, Dense-Evolution-Discovery, Zenodo (2026).
3. S. Aubry and G. André, *Analyticity breaking and Anderson localization in incommensurate lattices*, Ann. Israel Phys. Soc. **3**, 133 (1980).
4. G. Pöschl and E. Teller, *Bemerkungen zur Quantenmechanik des anharmonischen Oszillators*, Z. Phys. **83**, 143 (1933).
5. D. Sornette, *Discrete-scale invariance and complex dimensions*, Phys. Rep. **297**, 239 (1998), arXiv:cond-mat/9707012.
6. H. L. Montgomery, *The pair correlation of zeros of the zeta function*, in Analytic Number Theory, Proc. Sympos. Pure Math. **24**, 181 (1973).
7. A. M. Odlyzko, *On the distribution of spacings between zeros of the zeta function*, Math. Comp. **48**, 273 (1987).
8. O. Bohigas, M. J. Giannoni, and C. Schmit, *Characterization of chaotic quantum spectra and universality of level fluctuation laws*, Phys. Rev. Lett. **52**, 1 (1984).
9. K. Jusufi and A. Anand, *From arithmetic spectra to a quantum-corrected black hole geometry*, arXiv:2608.23528 (2026).
10. H. Yang and L. Zou, *Renormalization-group perspective on gravitational critical collapse*, Phys. Rev. D **109**, 104034 (2024), arXiv:2207.04373.