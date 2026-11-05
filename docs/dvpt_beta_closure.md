# Closure of the inverse-temperature parameter β in the DVPT unified formula

**Salvatore Pennacchio** — Independent Researcher — October 2026

---

## Abstract

The unified DVPT formula contains one statistical parameter β (the inverse
temperature of the Gentile prime gas) in addition to the physical mass M and
the constants G, ħ, c, k_B. The question addressed here is whether β can be
fixed by the physics of the black hole itself, so that the formula becomes a
one-parameter theory like Schwarzschild. Two candidate identifications are
tested numerically:

- **Test A**: β = 1 / T_H(r_+), with T_H the Hawking temperature of the
  modified metric f_k(r) = 1 − 2Mr/(r² + ℓ²).

- **Test B**: β = 1 / E_c, with E_c = T_H · A/(4G) the cutoff energy of Jusufi & Anand,
  arXiv:2608.23528 (the identification β = 1/E_c is ours, not theirs).

Both are self-consistent equations of the form F(β; M, k) = 0, where k is the
Gentile order of the gas. All computations use G = ħ = c = k_B = 1.

Result: neither identification yields a non-trivial M-dependence of the
effective length ℓ² in the observable region. Test A has a solution for every
M and gives ℓ² constant in M (ℓ² = 0.1926 G for k = 1, ℓ² = 0.1171 G for
k → ∞). Test B has a solution only for M ≲ 2.11 (Planck units), and gives
ℓ² weakly dependent on k in that narrow window. Outside that window the
gas does not exist below the Hagedorn pole, because T_H · A/(4G) < 1.

---

## 1. Setup

Units: G = ħ = c = k_B = 1.

**Gentile gas.** For k ∈ [1, ∞],

    Z_k(β) = ζ(β) / ζ((k+1)β)
    E_k(β) = −ζ'(β)/ζ(β) + (k+1) ζ'((k+1)β)/ζ((k+1)β)
    c_k    = 1 − ln ζ(k+1)

The commutation factor α(k, β) is fixed by the requirement
S_J(α E) = S_k(E), where S_J(x) = x + ln x and S_k(E) = E + ln E + c_k:

    α(k, β) = W(E_k(β) · exp(E_k(β) + c_k)) / E_k(β)

with W the Lambert W function. The asymptotic value is
α(k, ∞) = e^{c_k} = e / ζ(k+1).

**Effective length.** ℓ²(k, β) = 1 / (π α(k, β)).

**Metric.** f_k(r) = 1 − 2 M r / (r² + ℓ²).

Outer horizon: r_+(M, β) = M + sqrt(M² − ℓ²), requiring M ≥ ℓ.

Hawking temperature:

    T_H(r_+) = (r_+² − ℓ²) / (4 π r_+ (r_+² + ℓ²))

which reduces to 1/(8πM) for r_+ ≫ ℓ.

---

## 2. Test A — β = 1 / T_H

The self-consistent equation is

    F_A(β; M, k) = β − 1 / T_H(r_+(M, β); ℓ²(β, k)) = 0.

Solved with `scipy.optimize.brentq` and 40-digit mpmath for ζ and ζ′.

Results:

| M | k | β* | r_+ | T_H | ℓ² | ℓ²_∞ |
|---|---|---|---|---|---|---|
| 1 | 1 | 26.5517 | 1.89854 | 3.7662e-02 | 0.19262123 | 0.19262122 |
| 1 | 2 | 26.1230 | 1.92695 | 3.8280e-02 | 0.14076046 | 0.14076046 |
| 1 | 10 | 25.9406 | 1.93960 | 3.8550e-02 | 0.11715753 | 0.11715753 |
| 1 | ∞ | 25.9401 | 1.93963 | 3.8550e-02 | 0.11709967 | 0.11709966 |
| 2 | 1 | 50.8934 | 3.95125 | 1.9649e-02 | 0.19262122 | 0.19262122 |
| 2 | ∞ | 50.6416 | 3.97051 | 1.9747e-02 | 0.11709966 | 0.11709966 |
| 5 | 1 | 125.9072 | 9.98070 | 7.9424e-03 | 0.19262122 | 0.19262122 |
| 10 | 1 | 251.4486 | 19.99036 | 3.9770e-03 | 0.19262122 | 0.19262122 |
| 100 | 1 | 2513.2862 | 199.99904 | 3.9789e-04 | 0.19262122 | 0.19262122 |

Observations:

- β*(M, k) approaches 8πM from above for large M. The relative difference
  scales as 2π ℓ² / M.

- ℓ² is **constant in M** to 8 significant digits, because β* ≥ 26 for every M ≥ 1, where α(k, β) has already reached its β → ∞ value e/ζ(k+1). It equals its asymptotic
  value ℓ²_∞ = 1/(π e^{c_k}) = ζ(k+1)/(π e) G.

- k-dependence is entirely in the constant ℓ²_∞:

| k | ℓ²_∞ |
|---|---|
| 1 | 0.19262122 G |
| 2 | 0.14076046 G |
| 10 | 0.11715753 G |
| ∞ | 0.11709966 G |

Assertions passed:

- F_A(β*) = 0 to 1e-10 for every (M, k) with a solution.

- T_H(β*) > 0 for every non-extremal solution.

- |ℓ²(100) − ℓ²(1000)| / ℓ²(100) < 1%, in fact 0.

---

## 3. Test B — β = 1 / E_c (Jusufi–Anand cutoff)

In Jusufi & Anand the arithmetic energy cutoff is E_c = T_H · A/(4G), where
A = 4π r_+². Identifying, as an extra assumption of this note, β = 1 / E_c gives

    F_B(β; M, k) = β − 4 / (T_H(r_+; ℓ²) · A(r_+)) = 0.

Results:

| M | k | β_c | r_+ | T_H | A/(4G) | E_c | ℓ² |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.39940 | 1.88459 | 3.7352e-02 | 11.1579 | 0.41677 | 0.21750617 |
| 1 | 2 | 2.32222 | 1.90445 | 3.7793e-02 | 11.3944 | 0.43062 | 0.18196502 |
| 1 | 10 | 2.28887 | 1.91336 | 3.7987e-02 | 11.5012 | 0.43690 | 0.16577485 |
| 1 | ∞ | 2.28879 | 1.91338 | 3.7988e-02 | 11.5015 | 0.43691 | 0.16573128 |
| 2 | 1 | 1.06181 | 3.92135 | 1.9495e-02 | 48.3082 | 0.94179 | 0.30841050 |
| 2 | ∞ | 1.06015 | 3.92337 | 1.9506e-02 | 48.3580 | 0.94326 | 0.30065533 |
| 5 | — | no solution |
| 10 | — | no solution |
| 100 | — | no solution |

Observations:

- A solution exists only for M ≲ 2.11 in Planck units. For M ≥ 5, the
  equation F_B = 0 has no root in (1, 1000): the function F_B(β) stays
  positive on the whole interval. The physical reason is that
  E_c = T_H · A/(4G) ≈ (1/8πM)(4πM²) = M/2 → β_c ≈ 2/M, which falls below 1
  (the Hagedorn pole of the gas) for M ≳ 2; with ℓ² included the exact threshold is M ≈ 2.11.

- In the range where a solution exists, ℓ² depends on k but only mildly
  (0.218 G for k = 1, 0.166 G for k = ∞ at M = 1).

- ℓ² at M = 2 has moved from 0.218 to 0.308 for k = 1. The identification
  does not give ℓ² a non-trivial M-dependence in the large-M region,
  because no large-M region exists.

Assertions passed:

- F_B(β_c) = 0 to 1e-10 for every converged entry.

- No solution for M ≥ 5.

---

## 4. What is closed and what remains free

**Fixed.**

| Quantity | Value | Source |
|---|---|---|
| k (Test A) | any value, ℓ² depends on it | statistics of the gas |
| k (Test B) | any value, ℓ² depends on it | statistics of the gas |

**Derived.** α(M), E(M), ℓ²(M), β*(M) — all functions of the identification
used.

**Free.** M, plus the discrete choices of k and of the identification (A or B).

**Not present.** Φ does not appear in any of the results above.

In both Tests A and B, the effective length ℓ² is constant in M to
8 significant digits (Test A) or has no region of non-trivial M-dependence
accessible (Test B). The correction to Schwarzschild at the horizon is
ℓ²/r_+² ≈ ℓ_P²/(4M²), i.e. \~10⁻⁷⁷ for a solar-mass black hole and
\~10⁻⁴¹ for a 10¹⁵ g primordial one.

---

## 5. Conclusion

The free parameter β can be formally closed by either of two
self-consistent identifications. Neither yields an M-dependent ℓ² in the
observable region. The theory reduces to Schwarzschild with a Planck-scale
correction. The choice between Test A and Test B is not fixed by the
present analysis; the two identifications are not equivalent, and only
Test A has a solution for every M.

The residual freedom is M alone (both tests), plus the discrete choice of
k (both tests) and the discrete choice of identification (A or B). No
golden-ratio constant enters these results.

All numerical values above were computed by the accompanying script and
reproduced here without rounding beyond the digits shown.
