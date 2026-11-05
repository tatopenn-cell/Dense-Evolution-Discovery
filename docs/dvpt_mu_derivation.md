# DVPT: deriving the operator parameter μ

**Salvatore Pennacchio** — Independent Researcher — October 2026

---

## Abstract

The unified formula of the DVPT framework contains two operator
parameters, μ and ε, which previous scripts determined by grid search:
for each (k, β), the pair (μ, ε) was chosen on a 40 × 8 grid to minimise
|ℓ²_op(μ, ε) − 4 ℓ²_gas(k, β)|. That is a fit, not a derivation. This
note shows that, with ε = 0 and the asymptotic gas length
ℓ²_inf(k) = ζ(k+1)/(π e) fixed by Test A of the β-closure note, the
parameter μ is fixed uniquely by a single scalar equation

    ℓ²_op(μ) = 4 ℓ²_inf(k),

whose solution μ*(k) is reported below. Two assumptions are carried
over from the existing script and are explicitly not derived here:
ε = 0 and the factor 4. Neither Φ nor any golden-ratio constant
appears in the result.

---

## 1. Question

Given

- the ground state of H(μ) = −d²/dr² − 2μ sech²(r),
  ψ₀(r) = sech^{a(μ)}(r) with a(μ) = (√(1 + 8μ) − 1)/2,
- the effective width ℓ²_op(μ) = ⟨r⁴⟩/⟨r²⟩ over |ψ₀|²,
- the asymptotic gas length ℓ²_inf(k) = ζ(k+1)/(π e),

and fixing ε = 0, is there a unique μ solving ℓ²_op(μ) = 4 ℓ²_inf(k),
and how does μ*(k) depend on k?

---

## 2. Equation

The scalar equation is

    ℓ²_op(μ) = 4 ℓ²_inf(k),                                         (1)

with

    ℓ²_op(μ) = ∫ r⁴ sech^{2a(μ)}(r) dr / ∫ r² sech^{2a(μ)}(r) dr,
    a(μ)     = ( √(1 + 8μ) − 1 ) / 2,
    ℓ²_inf(k) = ζ(k+1) / (π e).

The left-hand side is strictly decreasing on μ ∈ (0.5, ∞); the
right-hand side is finite and positive for every k ∈ [1, ∞]. Hence
(1) has exactly one solution for every k.

---

## 3. Result

Solved with `scipy.optimize.brentq` on the mpmath integral at 40
decimal digits. Values computed by the accompanying script
`dvpt_mu_derivation.py`:

| k | ℓ²_inf(k) | 4 ℓ²_inf(k) | μ*(k) | a(μ*) | E₀(μ*) | residual |
|---|---|---|---|---|---|---|
| 1 | 0.19262122 | 0.77048490 | 4.96144731 | 2.689498 | −7.233397 | 0 |
| 2 | 0.14076046 | 0.56304183 | 7.59342173 | 3.428975 | −11.757869 | 0 |
| 3 | 0.12673969 | 0.50695874 | 8.82144671 | 3.729999 | −13.912894 | 0 |
| 5 | 0.11913053 | 0.47652212 | 9.65408019 | 3.922461 | −15.385699 | −5.55e−17 |
| 10 | 0.11715753 | 0.46863013 | 9.89398326 | 3.976379 | −15.811588 | 0 |
| ∞ | 0.11709966 | 0.46839865 | 9.90118458 | 3.977987 | −15.824382 | 0 |

The residual |ℓ²_op(μ*) − 4 ℓ²_inf(k)| is below 10⁻¹⁰ in every case.

**Monotonicity.** μ*(k) **increases** with k, not decreases. The reason
is arithmetic: ℓ²_inf(k) decreases with k (ζ(k+1) → 1), so the target
4 ℓ²_inf(k) decreases, and since ℓ²_op(μ) is decreasing in μ, a
smaller target requires a larger μ.

---

## 4. Comparison with the grid fit

Section 2b of the width note used a 40 × 8 grid over
(μ, ε) ∈ [0.5, 60] × {0, 0.5, 1, 2, 3, 5, 8, 12} and reported the grid
point minimising |ℓ²_op − target|. The grid spacing in μ is

    dμ = (60 − 0.5) / 39 = 1.526,

so the grid value can differ from μ* by up to dμ/2 ≈ 0.76 even
before the systematic bias from coarse sampling in the steep region
μ ≈ 3–6 is taken into account. The values of μ* in the table above
are exact to 10⁻¹² and replace the grid search in the region ε = 0.

Grid-fit values from section 2b of the width note at β = 5 (the largest β in that table,
where ℓ²_gas is closest to ℓ²_inf) next to the derived values. The grid also varied ε, so the
two columns answer slightly different questions; the derived μ* is the ε = 0 solution.

| k | grid μ* (β = 5) | grid ε* | grid ℓ²_op / 4ℓ²_gas error | derived μ* (ε = 0) |
|---|---|---|---|---|
| 1 | 5.077 | 0.5 | 0.58% | 4.96144731 |
| 2 | 9.654 | 3.0 | 0.57% | 7.59342173 |
| 5 | 11.179 | 2.0 | 0.02% | 9.65408019 |
| 10 | 24.910 | 12.0 | 0.31% | 9.89398326 |

For k = 1 the grid lands close to the derived value; for larger k the grid compensates a
larger μ with a non-zero ε, which is why its μ* drifts away. The derivation removes that
two-parameter freedom by fixing ε = 0.

---|---|---|
| 1 | 4.96144731 | 0.77048490 |
| 2 | 7.59342173 | 0.56304183 |
| 3 | 8.82144671 | 0.50695874 |
| 5 | 9.65408019 | 0.47652212 |
| 10 | 9.89398326 | 0.46863013 |
| ∞ | 9.90118458 | 0.46839865 |

---

## 5. What is derived and what is assumed

**Derived in this note.**

- μ*(k) as the unique solution of (1) for each k ∈ {1, 2, 3, 5, 10, ∞}.
- a(μ*(k)) = (√(1 + 8μ*(k)) − 1)/2.
- E₀(μ*(k)) = −a(μ*(k))².

All three are now functions of k alone, with no grid search.

**Assumed, carried over from `dvpt_poschl_teller_width.py`, not derived here.**

- ε = 0. The Aubry–André modulation is taken to be absent in the frame
  in which the derivation is performed. No dynamical argument for this
  choice is given here.
- The factor 4 in ℓ²_op = 4 ℓ²_gas. This is the numerical relation used
  in the width note's section 2b and inherited from the unified-formula
  conventions. It is not derived in this note.

**Still free.**

- k. Discrete. Values tested: {1, 2, 3, 5, 10, ∞}. No argument fixing
  k is given here.
- M. Via β = 1/T_H (Test A of the β-closure note), ℓ² does not depend
  on M in the limit M → ∞; the derivation above is at that limit.

**Not present.**

- Φ. The golden ratio does not appear in equation (1), in a(μ), in
  ℓ²_op(μ), in ℓ²_inf(k), or in any numerical value reported above.

---

## 6. Reproduction

    python scripts/dvpt_mu_derivation.py

Requirements: `numpy`, `scipy`, `mpmath` at 40 decimal digits. The
script prints the table above, asserts residual < 10⁻¹⁰ and
monotonicity of μ*(k), and reports the comparison with the grid fit.