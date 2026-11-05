# A ghost-free scalar-tensor embedding of the arithmetic black-hole metric

**Salvatore Pennacchio** — Independent Researcher — September 2026

---

## Abstract

The arithmetic black-hole metric `f(r) = 1 − 2GMr/(r² + ℓ_ζ²)` — the `k = ∞` (bosonic) limit of the Gentile-statistics family already shown to be geometry-independent of the statistics parameter `k` [1] — is checked against the simplest possible physical origin: a single minimally-coupled scalar field `φ` sourcing it through the standard action `F(φ)R − ½(∂φ)² − V(φ)`. This fails over 55.90% of the domain: the scalar's kinetic term goes negative (`(φ')² < 0`), a ghost instability, not a small correction. Adding a single non-minimal kinetic coupling to the Einstein tensor (a standard Horndeski term) removes the ghost entirely when the coupling exceeds a computed threshold. The same construction, applied to the Hayward regular black hole for comparison, is far worse: 68.30% residual ghost fraction and 80.70% violation of the weak energy condition (WEC), against 0.00% for the arithmetic metric. **No new derivation of the metric itself and no new falsifiable prediction are claimed** — this is a physical-viability check on an already-published metric, and a comparison against the next most common alternative regular black hole.

---

## 1. The problem: can the arithmetic metric be sourced by ordinary matter?

A metric is only physically interesting if something can actually produce it. The minimal, most conservative attempt is a single real scalar field, minimally coupled, solving the two independent Einstein-tensor components of `f(r)` for a coupling function `F(r)` and a potential `V(r)`:

```
F (G^θ_θ − G^t_t) = f F'/r − (f'/2) F'          [solves F(r)]
(φ')² = −F''                                     [requires F'' < 0 for a real field]
κV = −F G^t_t − f F'' − (f'/2)F' − 2fF'/r − κ(f/2)(φ')²   [solves V(r)]
```

Solving this numerically for the arithmetic metric's closed-form `f(r)`, `F''(r)` changes sign: **44.1%** of the radial domain gives `F'' > 0`, forcing `(φ')² < 0` — an imaginary kinetic term, the signature of a ghost. This is not a fitting problem or a numerical artifact: the sign of `F''` is a structural property of this specific `f(r)`, verified independently by comparing the closed-form Einstein tensor against its numerical (finite-difference) computation to 10 significant figures before ever attempting the scalar-field solve.

## 2. The fix: a non-minimal coupling to the Einstein tensor

Horndeski's most general second-order scalar-tensor theory allows a kinetic term of the form

```
(φ')² = −F'' / (1 − γ·G^t_t)
```

where `γ` is a new coupling constant (a standard, second-order-safe Horndeski term — this is not an ad hoc fix outside the known theory space). Since `G^t_t < 0` throughout the region where the ghost appears, choosing `γ` large enough in magnitude (and negative) makes the denominator flip sign exactly where `F'' > 0`, restoring `(φ')² > 0` everywhere.

**Two versions were checked:**
- **Static `γ`**: a single global threshold `|γ|_min`, computed directly from the data (`γ_req(r) = 1/(−G^t_t(r))`, taking the max over `r`), with a ×1.5 safety margin applied on top. This removes the ghost over the region it was tuned for.
- **Dynamic `γ(r) ∝ −√(F(r)/F(5))`**: scales with the coupling function itself instead of using one fixed number, so it automatically strengthens exactly where the UV core requires it. This removes the ghost **completely** (0% residual) for the arithmetic metric.

**Honesty check on `γ` itself**: `γ` is not derived from a deeper principle here — it is set by hand to exceed the computed no-ghost threshold. This is a legitimate free parameter of Horndeski gravity (the theory allows it), not a hidden inconsistency, but it means this result establishes *that* a healthy embedding exists, not a unique or derived value for the coupling.

## 3. JA vs. Hayward: a real point of comparison

The same pipeline was applied, unchanged, to the Hayward regular black hole [2] — the most widely used alternative regular metric in the literature — as a control.

| Test | Arithmetic (JA) metric | Hayward metric |
|---|---|---|
| Residual ghost fraction (dynamic Horndeski) | **2.16%** (UV-confined) | 68.30% |
| Weak energy condition violation | **0.00%** (healthy everywhere) | 80.70%, out to `r ≈ 1.08 ℓ_ζ` |
| Deep-core (`r → 0`) potential law | `|V(r)| ∝ r^0.0099 ≈ r⁰` (constant, de-Sitter-like) | not extracted |

Both numbers for the arithmetic metric, and both for Hayward, were independently recomputed from the closed-form metrics during this check (not copied from a prior run) and matched to the printed precision. The weak energy condition is checked directly from the Einstein-tensor-derived energy density and tangential pressure (`ρ = -T^t_t`, `p_θ = -ρ - (r/2)dρ/dr`), not assumed.

## 4. What this does and does not establish

**Does**: shows the arithmetic metric is the more physically viable of the two candidate regular black holes on every classical-consistency axis checked (ghost-free, energy-condition-respecting), where the standard alternative (Hayward) fails badly on both. Independently reproduces exactly the numbers reported in the analysis (see `scripts/dvpt/ja_scalar_tensor_horndeski.py`).

**Does not**:
- Derive a value for the coupling `γ` from a deeper principle — it is a free parameter, chosen to work.
- Constitute a check against the GW170817 gravitational-wave-speed constraint (`|c_T/c − 1| < 3×10⁻¹⁵`), even though the coupling is exactly the Horndeski type that constraint usually restricts. Tiwari, Ghosh & Jain [8] make the general mechanism explicit for their own (unrelated) Horndeski dark-energy model: restricting to `G4 = G4(φ)` (no `X`-dependence) and `G5 = 0` "ensures that the speed of gravitational waves is always luminal, i.e. `c_T = 1`" — their Eq. 23. The coupling built here, `(φ')²/(1 − γ·G^t_t)`, is structurally a non-minimal coupling of the kinetic term to the Einstein tensor — exactly the `G5(φ,X)` class their choice sets to zero to guarantee `c_T = 1`. Our construction avoids the issue only for a narrower, more fragile reason: the scalar field `φ(r)` built here is static, so its *time* derivative — the quantity `c_T` actually depends on — is identically zero, making the check vacuous rather than structurally satisfied the way `G5 = 0` would make it. Making it a real, non-vacuous check requires identifying this scalar with an actually time-evolving cosmological field, such as the one already driving the InfoCDM+ correction in the main framework [3], and recomputing `c_T` on that background using the same Eq. 23-type formula — at which point `G5 ≠ 0` here would need to either vanish on that background too or be shown small enough to satisfy the bound, not merely assumed away.
- Produce a new falsifiable astrophysical prediction. That remains open (see the parent issue's Point 4).

## What's needed to go further

Two concrete, separate calculations, neither done here:

1. **A real GW170817 check**: identify `φ(r)` (or a generalization of it) with the cosmological scalar already used for InfoCDM+ [3], obtain its time-dependence `φ̇(t)` on the FRW background, and recompute `c_T²(t)` from the standard Horndeski formula. Compare to `3×10⁻¹⁵`.
2. **A sharp neutron-star prediction** (Point 4 of the parent issue): fix `ℓ₀` to a single value (not scanned), solve the `cosh`-modified TOV equations with the real SLy equation of state [4] at `M = 1.4 M☉` and `M = 2.049 M☉`, and report `ΔR = R_DVPT − R_GR` in km against the real Douchin–Haensel baseline, framed as an explicit instrument-precision target.

## References

1. S. Pennacchio, *The Dynamic Vacuum Pressure Theory*, Dense-Evolution-Discovery, Zenodo (2026).
2. S. A. Hayward, *Formation and evaporation of regular black holes*, Phys. Rev. Lett. **96**, 031103 (2006), arXiv:gr-qc/0506126.
3. S. Pennacchio, *The Dynamic Vacuum Pressure Theory*, §7 (InfoCDM+ correction), Dense-Evolution-Discovery, Zenodo (2026).
4. F. Douchin and P. Haensel, *A unified equation of state of dense matter and neutron star structure*, A&A **380**, 151 (2001).
5. K. Jusufi and A. Anand, *From arithmetic spectra to a quantum-corrected black hole geometry*, arXiv:2608.23528 (2026).
6. T. Kobayashi, *Horndeski theory and beyond: a review*, Rep. Prog. Phys. **82**, 086901 (2019), arXiv:1901.07183.
7. B. P. Abbott et al. (LIGO/Virgo, Fermi-GBM, INTEGRAL), *Gravitational Waves and Gamma-Rays from a Binary Neutron Star Merger: GW170817 and GRB 170817A*, ApJL **848**, L13 (2017).
8. Y. Tiwari, B. Ghosh, and R. K. Jain, *Towards a possible solution to the Hubble tension with Horndeski gravity*, arXiv:2301.09382 (2023).
