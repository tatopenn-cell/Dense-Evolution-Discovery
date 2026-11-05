# Supplementary Note: Gentile-Statistic Extension

**Salvatore Pennacchio** — Independent Researcher — September 2026

---

## Abstract

This is a supplementary note to the three companion papers of September 2026 — *The Dynamic Vacuum Pressure Theory* (DVPT) [1], *The Golden Dynamic Space Framework* (GDS) [2], and *Continuous and discrete projections of a Z₂-symmetric operator family* (Prisma) [3] — and to the Jusufi–Anand arithmetic black-hole paper [4] on which part of the framework builds. It does not replace any of them; it adds one result obtained after their preparation, concerning the intermediate region between the DVPT limit (`k = 1`, fermionic statistics per mode) and the Jusufi–Anand limit (`k = ∞`, bosonic statistics) of the same arithmetical gas.

The result is exact and follows from the simple pole of `ζ(s)` at `s = 1`. For the Gentile-statistics family `Z_k(β) = ζ(β)/ζ((k+1)β)`, with `k ∈ [1, ∞]`, the entropy as a function of energy has the exact asymptotic form

$$S_k(E) = E + \ln E + c_k + O(1/E), \qquad c_k = 1 - \ln\zeta(k+1).$$

The coefficient of `ln E` is **exactly 1 for every k**, verified numerically to machine precision (residuals `0.00e+00` and `1.11e-16` across `k = 1, 2, 3, 5, 10, 100, ∞`). Only the constant `c_k` depends on the statistics. Consequently the emergent metric of Jusufi–Anand,

$$f(r) = 1 - \frac{2GMr}{r^2 + \ell_\zeta^2}, \qquad \ell_\zeta^2 = \alpha G/\pi, \quad \alpha = 1,$$

is the **same for every value of `k`** in the Gentile family. DVPT (`k = 1`) and the bosonic Jusufi–Anand limit (`k = ∞`) share the same metric; they differ only in the additive constant `c_k` of the entropy. With this map the Gentile statistics sit on one face of the Prisma: `k` changes the constant, not the geometry. With the Lambert-W energy–area map of the [unified formula](dvpt_unified_formula.md), the metric depends on `k` and each statistics opens its own face. The number of faces is what the maps produce, not something fixed in advance.

---

## 1. Motivation

The three companion papers [1, 2, 3] establish:

- DVPT treats the vacuum as a two-state-per-mode system, with the canonical partition function per mode `Z_mode = 2cosh(Δ/2)`. This is equivalent to **fermionic statistics per prime** (`k = 1` in the Gentile classification), plus a per-mode zero-point shift. This is stated in DVPT §2.1 and used throughout.

- Jusufi & Anand [4] obtain the arithmetic black-hole metric from the **bosonic** prime gas `Z_boson(β) = ζ(β)`. This is `k = ∞` in the Gentile classification.

- The Prisma [3] unifies DVPT and GDS in the operator family `H(μ, ε) = -d²/dr² + ε·cos(2παr) - 2μ·sech²(r)`, and identifies a critical transition at `μ = 1`. It also reports the negative result that Φ is not the unique invariant of that family.

What the Prisma does **not** address is the region between `k = 1` (DVPT) and `k = ∞` (JA) inside the same arithmetic gas. The present note fills exactly that gap. The question is:

> *If DVPT is the fermionic (`k = 1`) limit of the prime gas, and Jusufi–Anand is the bosonic (`k = ∞`) limit of the same gas, what happens for intermediate statistics, and does the metric change?*

The answer, derived below, is that the metric does **not** change. This is a structural fact following from the simple pole of `ζ(s)` at `s = 1`.

---

## 2. The Gentile-statistics family

Gentile statistics of order `k` allows at most `k` particles per mode. For the prime gas (bosonic modes with energies `ε_p = ln p`), the partition function is

$$Z_k(\beta) = \prod_p \frac{1 - p^{-(k+1)\beta}}{1 - p^{-\beta}} = \frac{\zeta(\beta)}{\zeta((k+1)\beta)}.$$

The two limiting cases are:

- `k = 1`: `Z_1(β) = ζ(β)/ζ(2β)`, the fermionic (squarefree) zeta gas. This is what DVPT produces per mode, up to a zero-point shift.
- `k → ∞`: `Z_∞(β) = ζ(β)`, the bosonic prime gas of Jusufi & Anand [4].

For intermediate `k`, `Z_k(β)` interpolates smoothly between the two. The energy and entropy are the standard canonical quantities:

$$E_k(\beta) = -\frac{\partial \ln Z_k}{\partial \beta}, \qquad S_k(\beta) = \ln Z_k + \beta E_k.$$

The problem is to compute `S_k` as a function of `E_k` in the regime `β → 1⁺` (`E → ∞`), and to extract the coefficient of the logarithm.

---

## 3. The exact result

Near `β = 1`, writing `δ = β - 1`, the structure of `Z_k` is determined by the simple pole of `ζ(s)` at `s = 1`. Both `ζ(β)` and `ζ((k+1)β)` have a simple pole at `β = 1`; their ratio has the Laurent expansion

$$Z_k(\beta) = \frac{A_k}{\delta} + B_k + O(\delta),$$

with the explicit coefficients

$$A_k = \frac{1}{\zeta(k+1)}, \qquad \frac{B_k}{A_k} = \gamma - (k+1)\frac{\zeta'(k+1)}{\zeta(k+1)},$$

where `γ = 0.5772156649…` is the Euler–Mascheroni constant. For `k = ∞` one has `A_∞ = 1` and `B_∞/A_∞ = γ`.

From this expansion, `E(δ) = 1/δ - B_k/A_k + O(δ)`, and an elementary computation gives the asymptotic entropy as a function of energy:

$$\boxed{\;S_k(E) = E + \ln E + c_k + O(1/E), \qquad c_k = 1 - \ln\zeta(k+1).\;}$$

Equivalently, defining the **effective coefficient** of the logarithm through

$$\alpha_{\mathrm{eff}}(E) = \frac{S_k(E) - E}{\ln E},$$

the exact asymptotic form is

$$\alpha_{\mathrm{eff}}(E) = 1 + \frac{c_k}{\ln E} + O\!\left(\frac{1}{E \ln E}\right).$$

The coefficient of `ln E` is **exactly 1** in the limit, and only the constant `c_k` depends on the statistics.

---

## 4. Numerical verification

The formula `c_k = 1 - \ln\zeta(k+1)` was verified numerically against direct inversion of `E_k(β) = E` for a range of `E ∈ [10⁴, 10²⁰]` and `k ∈ {1, 2, 3, 5, 10, 100, ∞}`. Results at `E = 10²⁰`:

| k | `c_k(E = 10²⁰)` | `c_k = 1 − ln ζ(k+1)` | residual |
|---|---|---|---|
| 1 | 0.5022996975 | 0.5022996975 | 0.00e+00 |
| 2 | 0.8159658246 | 0.8159658246 | 1.11e-16 |
| 3 | 0.9208901269 | 0.9208901269 | 0.00e+00 |
| 5 | 0.9828056124 | 0.9828056124 | 1.11e-16 |
| 10 | 0.9995059335 | 0.9995059335 | 1.11e-16 |
| 100 | 1.0000000000 | 1.0000000000 | 0.00e+00 |
| ∞ | 1.0000000000 | 1.0000000000 | 0.00e+00 |

The residuals are one machine epsilon (`1.11e-16`), i.e. the exact value and the numerically extracted value are indistinguishable at double precision. The formula is exact, not an approximation.

Approximate values for reference:

```
c_1  = 0.50229970    (k = 1, fermionic  →  DVPT)
c_2  = 0.81596582
c_3  = 0.92089013
c_5  = 0.98280561
c_10 = 0.99950593
c_∞  = 1.00000000    (k = ∞, bosonic  →  Jusufi–Anand)
```

The DVPT constant `c_1 = 0.50229970` is close to, but not equal to, `1/2`. The difference from `1/2` is `0.00229970…`, and it is fixed by `ln ζ(2) = ln(π²/6) ≈ 0.4977`. The DVPT constant is therefore `1 - ln(π²/6)`, a value tied to the arithmetic of the primes, not to any factor of `1/2` chosen by hand.

---

## 5. Consequence for the emergent metric

The Jusufi–Anand entropy-geometry correspondence reconstructs a static spherically symmetric metric from a prescribed horizon entropy

$$S(A) = \frac{A}{4G} + \alpha \ln\frac{A}{4G} + \text{const},$$

giving

$$f(r) = 1 - \frac{2GMr}{r^2 + \ell_\zeta^2}, \qquad \ell_\zeta^2 = \frac{\alpha G}{\pi}.$$

The coefficient `α` is the coefficient of the logarithmic term in `S(A)`. For the Gentile family computed above, `α = 1` **exactly** for every `k ∈ [1, ∞]`. Hence:

$$\boxed{\;f_k(r) = 1 - \frac{2GMr}{r^2 + G/\pi} \quad \text{for every } k \in [1, \infty].\;}$$

The metric is **unique** across the whole Gentile family. DVPT (`k = 1`) and the bosonic Jusufi–Anand limit (`k = ∞`) share the same metric `f(r) = 1 - 2GMr/(r² + G/π)`; they differ only in the additive constant `c_k` of the entropy, which does not propagate into the metric because the metric depends on `S'(r)`, and `c_k` is a constant independent of `r`.

The universal value `α = 1` is a direct consequence of the simple pole of `ζ(s)` at `s = 1`. Any statistics on the prime gas that respects the canonical ensemble and the pole structure at `β = 1` will produce the same `α`, hence the same metric. This is why the Jusufi–Anand paper [4] finds `α = 1` both for the bosonic canonical gas and for its fermionic (squarefree) counterpart, and the present note extends that statement to the whole Gentile family.

---

## 6. What this note does and does not do

**Does:**

- Establishes that the Gentile family `Z_k(β) = ζ(β)/ζ((k+1)β)` contains DVPT (`k = 1`) and Jusufi–Anand bosonic (`k = ∞`) as limits.
- Derives the exact asymptotic entropy `S_k(E) = E + ln E + c_k + O(1/E)` with `c_k = 1 - ln ζ(k+1)`.
- Verifies numerically that the coefficient of `ln E` is exactly 1 for every `k`, to machine precision.
- Shows that the Jusufi–Anand metric is universal across the whole Gentile family, and that this universality is a consequence of the simple pole of `ζ(s)` at `s = 1`.
- Provides a closed-form expression for the k-dependence: it lives entirely in the additive constant `c_k`, not in any coefficient relevant to the metric.

**Does not:**

- Create a "fourth face" of the Prisma *with the Jusufi–Anand map*: there the statistics parameter `k` is an internal degree of freedom of the arithmetic-gas description; it changes the constant `c_k`, not the geometry. With a different energy–area map (see the [unified formula](dvpt_unified_formula.md)) `k` does change the geometry, and new faces appear.
- Derive the horizon operator whose arithmetic cutoff scales with the area. That open problem, stated at the end of Jusufi–Anand [4], remains open.
- Connect `Φ` to the metric. The metric uses `π` (`ℓ_ζ² = G/π`), not `Φ`. The role of `Φ` in the framework is restricted to the discrete-scale-invariance context of GDS and Sornette's open question [2, 5], and is not touched by the results of this note.
- Provide any new falsifiable prediction. The results here are structural and internal to the framework.

---

## 7. Epistemic status

Every claim in this note is classified as **derived** (follows from a closed-form calculation), **verified** (confirmed numerically against an independent reference), **cited** (established in the literature), or **not addressed** (out of scope of this note).

| Claim | Status | Section | Comment |
|---|---|---|---|
| `Z_k(β) = ζ(β)/ζ((k+1)β)` is the Gentile partition function of the prime gas | Derived | 2 | Standard Gentile statistics; product formula checkable term by term |
| `Z_k(β) = A_k/δ + B_k + O(δ)` near `β = 1` | Derived | 3 | Follows from the simple pole of `ζ(s)` at `s = 1` |
| `A_k = 1/ζ(k+1)`, `B_k/A_k = γ - (k+1)ζ'(k+1)/ζ(k+1)` | Derived | 3 | Standard residue formulas |
| `S_k(E) = E + ln E + c_k + O(1/E)` | Derived | 3 | Elementary expansion of the Laurent form |
| `c_k = 1 - ln ζ(k+1)` | Derived | 3 | Follows from matching the expansion at order `1` |
| Numerical match of `c_k` to machine precision | Verified | 4 | Residuals `0.00e+00` and `1.11e-16` at `E = 10²⁰`, `k ∈ {1, 2, 3, 5, 10, 100, ∞}` |
| Coefficient of `ln E` is exactly 1 for every `k` | Derived | 3, 4 | Structural consequence of the simple pole |
| Jusufi–Anand metric `f(r) = 1 - 2GMr/(r² + G/π)` is unique for the whole Gentile family | Derived | 5 | Follows from `α = 1` universal and the entropy-geometry correspondence |
| DVPT and Jusufi–Anand bosonic share the same metric | Derived | 5 | Direct consequence of the above |
| Gentile statistics does not create a fourth face of the Prisma | Derived | 6 | Negative result: the parameter changes only the constant |
| Existence of a horizon operator whose cutoff scales with area | Not addressed | 6 | Open problem, stated in Jusufi–Anand [4] |
| Role of `Φ` in the metric | Not addressed | 6 | The metric uses `π`; `Φ` appears in GDS context only |

---

## 8. Limits, stated directly

- The results of this note concern the **arithmetic-gas description** of the vacuum. They do not constitute a physical theory, and no new physical prediction is claimed.

- The derivation uses the canonical ensemble and the standard residue formulas at `β = 1`. A different ensemble (microcanonical) may produce a different constant term in the Laurent expansion; this is not investigated here.

- The numerical verification is done in `mpmath` at 80 digits of precision. Residuals of order `1e-16` are at the limit of double precision, not of the arbitrary-precision computation; the true residual is smaller but is not reported because the tabulated `c_k` values are printed at double precision.

- The universal value `α = 1` is established for the specific family `Z_k = ζ(β)/ζ((k+1)β)`. It is not proven for arbitrary statistics on the prime gas, only for the Gentile family. Whether a wider class of statistics shares the same universal `α = 1` is a natural question and is not addressed here.

- The result `f_k(r) = 1 - 2GMr/(r² + G/π)` for every `k` assumes the entropy-geometry correspondence holds with the exact coefficient `α = 1`. Any deviation of the correspondence from this exact form would change the conclusion, but the correspondence is not the object of this note.

- The connection between the arithmetic gas and physical spacetime remains a framework assumption, as it is in Jusufi–Anand [4]. Nothing in this note establishes that the arithmetic gas is a physical description of the vacuum; the note is consistent with, and extends, the structural mathematics of [4], [1], [2], [3].

---

## 9. Conclusion

The Gentile family `Z_k(β) = ζ(β)/ζ((k+1)β)` interpolates between the fermionic statistics of DVPT (`k = 1`) and the bosonic statistics of Jusufi–Anand (`k = ∞`) inside the same arithmetic gas. For every value of `k`, the asymptotic entropy as a function of energy is

$$S_k(E) = E + \ln E + c_k + O(1/E), \qquad c_k = 1 - \ln\zeta(k+1),$$

and the coefficient of the logarithmic term is exactly 1. Consequently the emergent metric

$$f(r) = 1 - \frac{2GMr}{r^2 + G/\pi}$$

is the same across the whole family. The statistics changes only the additive constant of the entropy, which does not propagate into the metric because the metric is reconstructed from `S'(r)`.

This is a structural result, following from the simple pole of `ζ(s)` at `s = 1`. It confirms that, with the Jusufi–Anand map E = A/4G, the metric of the framework is universal across statistics and the statistics parameter does not create additional geometric faces; other energy–area maps can, and the faces are counted as they are found. The result is offered as a supplementary note to the three companion papers [1, 2, 3], without modifying any of their retained claims.

---

## References

1. S. Pennacchio, *The Dynamic Vacuum Pressure Theory*, Dense-Evolution-Discovery, Zenodo (2026).
2. S. Pennacchio, *The Golden Dynamic Space Framework*, Dense-Evolution-Discovery, Zenodo (2026).
3. S. Pennacchio, *Continuous and discrete projections of a Z₂-symmetric operator family*, Dense-Evolution-Discovery, Zenodo (2026).
4. K. Jusufi and A. Anand, *From arithmetic spectra to a quantum-corrected black hole geometry*, arXiv:2608.23528 (2026).
5. D. Sornette, *Discrete-scale invariance and complex dimensions*, Phys. Rep. **297**, 239 (1998), arXiv:cond-mat/9707012.
6. G. Gentile, *Osservazioni sopra le statistiche intermedie*, Nuovo Cimento **17**, 493 (1940).
7. A. Khintchine, *On the statistical mechanics of the Gentile statistics*, in *Mathematical Foundations of Statistical Mechanics*, Dover (1949).