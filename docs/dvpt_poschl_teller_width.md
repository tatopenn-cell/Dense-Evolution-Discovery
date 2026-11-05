# On the Asymptotic Width of the Pöschl–Teller Ground State

**Salvatore Pennacchio** — Independent Researcher — October 2026

---

## Abstract

We consider the one-dimensional Schrödinger operator

$$
H(\mu) = -\frac{d^2}{dr^2} - 2\mu\,\mathrm{sech}^2(r), \qquad \mu > 0,
$$

and compute the effective width of its ground state

$$
\ell^2_{\rm op}(\mu) = \frac{\langle r^4\rangle}{\langle r^2\rangle},
\qquad
\langle r^n\rangle = \frac{\int r^n\,|\psi_0(r)|^2\,dr}{\int |\psi_0(r)|^2\,dr}.
$$

We show that the ground state is

$$
\psi_0(r) = \mathrm{sech}^{a(\mu)}(r),
\qquad
a(\mu) = \frac{\sqrt{1+8\mu}-1}{2},
$$

and that in the limit \(\mu\to\infty\) the exact asymptotic relation holds:

$$
\boxed{\;\ell^2_{\rm op}(\mu)\;\sim\;\frac{3}{2\,a(\mu)}\;\sim\;\frac{3}{\sqrt{8\mu}}\;}
$$

The coefficient \(3/2\) is derived from Gaussian moments and has no relation to the golden ratio. The exact form of \(\psi_0\) is checked against direct diagonalization to \(<0.3\%\) on \(\mu\in[1,20]\), and \(a\,\ell^2_{\rm op}\) approaches \(3/2\) from above as \(\mu\) grows.

---

## 1. Setup

### 1.1 The operator

Let

$$
H(\mu) = -\frac{d^2}{dr^2} + V_\mu(r),
\qquad
V_\mu(r) = -2\mu\,\mathrm{sech}^2(r),
\qquad \mu>0.
$$

This is a particular case of the Pöschl–Teller potential, for which the eigenvalue equation

$$
H(\mu)\,\psi = E\,\psi
$$

is exactly integrable. The potential is a symmetric well centered at \(r=0\), of depth \(2\mu\), with exponential decay as \(|r|\to\infty\).

### 1.2 Ground state — exact form

We propose the form \(\psi(r) = \mathrm{sech}^a(r)\) for an exponent \(a>0\) to be determined. We compute

$$
\psi'(r) = -a\,\mathrm{sech}^a(r)\tanh(r),
$$

$$
\psi''(r) = a\,\mathrm{sech}^a(r)\bigl[a\tanh^2(r) - \mathrm{sech}^2(r)\bigr].
$$

Using \(\tanh^2 = 1-\mathrm{sech}^2\),

$$
\psi'' = a\,\mathrm{sech}^a(r)\bigl[a - (a+1)\,\mathrm{sech}^2(r)\bigr].
$$

Substituting into the Schrödinger equation:

$$
-\psi'' - 2\mu\,\mathrm{sech}^2(r)\,\psi
= -a^2\,\psi + \bigl[a(a+1) - 2\mu\bigr]\,\mathrm{sech}^2(r)\,\psi.
$$

For \(\psi\) to be an eigenfunction, the \(\mathrm{sech}^2\) term must vanish:

$$
a(a+1) = 2\mu.
$$

The only positive root is

$$
a(\mu) = \frac{-1 + \sqrt{1+8\mu}}{2}.
$$

The corresponding eigenvalue is

$$
E_0(\mu) = -a(\mu)^2.
$$

Therefore

$$
\boxed{\;\psi_0(r) = \frac{\mathrm{sech}^{a(\mu)}(r)}{\|\cdot\|},
\qquad
E_0(\mu) = -a(\mu)^2\;}
$$

**Remark.** \(\psi_0 = \mathrm{sech}^\mu(r)\) holds **only** for \(\mu=1\) (where \(a(1)=1\)). For \(\mu>1\), \(a(\mu)<\mu\): for example \(a(2)\approx 1.5616\), \(a(5)\approx 2.7016\), \(a(10)=4\).

---

## 2. Effective width

We define the effective width of the ground state as

$$
\ell^2_{\rm op}(\mu) = \frac{\langle r^4\rangle}{\langle r^2\rangle},
\qquad
\langle r^n\rangle = \frac{\int_{-\infty}^{\infty} r^n\,\mathrm{sech}^{2a(\mu)}(r)\,dr}{\int_{-\infty}^{\infty}\mathrm{sech}^{2a(\mu)}(r)\,dr}.
$$

The normalization is known:

$$
\int_{-\infty}^{\infty}\mathrm{sech}^{2a}(r)\,dr
= \frac{\sqrt{\pi}\,\Gamma(a)}{\Gamma(a+\tfrac12)}.
$$

The moments \(\langle r^2\rangle\) and \(\langle r^4\rangle\) are computed numerically with arbitrary precision (mpmath, 40 digits).

---

## 3. Asymptotic behavior

### 3.1 Theorem

**Theorem.** For \(\mu\to\infty\), with \(a(\mu) = (\sqrt{1+8\mu}-1)/2\),

$$
\ell^2_{\rm op}(\mu) = \frac{3}{2\,a(\mu)}\Bigl(1 + O(a^{-1})\Bigr).
$$

In particular:

$$
a(\mu)\,\ell^2_{\rm op}(\mu) \xrightarrow[\mu\to\infty]{} \frac{3}{2}.
$$

### 3.2 Proof

For \(a\to\infty\), the function \(\mathrm{sech}^{2a}(r)\) is concentrated in a neighborhood of \(r=0\). Using the Taylor expansion

$$
\ln\mathrm{sech}(r) = -\frac{r^2}{2} + O(r^4),
$$

we obtain

$$
\mathrm{sech}^{2a}(r) = e^{-a r^2}\bigl(1 + O(a r^4)\bigr)
\quad\text{for } r \ll 1.
$$

The density \(|\psi_0(r)|^2\) is therefore asymptotically Gaussian with variance

$$
\sigma^2 = \frac{1}{2a}.
$$

For a Gaussian centered at \(0\), the moments are exact:

$$
\langle r^2\rangle = \sigma^2, \qquad
\langle r^4\rangle = 3\sigma^4.
$$

Hence

$$
\ell^2_{\rm op} = \frac{\langle r^4\rangle}{\langle r^2\rangle}
= \frac{3\sigma^4}{\sigma^2}
= 3\sigma^2
= \frac{3}{2a}.
$$

The corrections \(O(a^{-1})\) arise from the non-Gaussian terms of the expansion of \(\ln\mathrm{sech}(r)\), that is, from the coefficient \(-r^4/12\). \(\square\)

### 3.3 Limit in \(\mu\)

Since \(a(\mu)\sim\sqrt{2\mu}\) as \(\mu\to\infty\),

$$
\ell^2_{\rm op}(\mu) \sim \frac{3}{2\sqrt{2\mu}} = \frac{3}{\sqrt{8\mu}}.
$$

The factor \(8 = 2^3\) is \(\sigma^{-2}\) in units of \(\mu\):
\(\sigma^2 = 1/(2a) \sim 1/(2\sqrt{2\mu}) = 1/\sqrt{8\mu}\).

---

## 4. Numerical verification

### 4.1 Numerical vs analytical

Comparison between direct numerical computation (matrix \(H\) with \(N=600\) points on \([-25,25]\)) and analytical computation with \(\psi_0 = \mathrm{sech}^{a(\mu)}(r)\) and mpmath integration at 40 digits:

| \(\mu\) | \(a(\mu)\) | \(\ell^2_{\rm num}\) | \(\ell^2_{\rm anal}\) | diff |
|---|---|---|---|---|
| 1.0 | 1.0000 | 3.45361043 | 3.45436154 | 0.022% |
| 1.5 | 1.3028 | 2.22222712 | 2.22293719 | 0.032% |
| 2.0 | 1.5616 | 1.67140280 | 1.67211855 | 0.043% |
| 3.0 | 2.0000 | 1.15812782 | 1.15887005 | 0.064% |
| 5.0 | 2.7016 | 0.76514280 | 0.76592314 | 0.102% |
| 10.0 | 4.0000 | 0.46443338 | 0.46525320 | 0.177% |
| 20.0 | 5.8443 | 0.29627936 | 0.29712267 | 0.285% |

Numerical and analytical agree to within \(0.3\%\). The difference grows with \(\mu\) and is attributable to discretization (the width \(\sigma\sim 1/\sqrt{2a}\) becomes comparable to \(dr\)).

### 4.2 Convergence to \(3/2\)

| \(\mu\) | \(a(\mu)\) | \(\ell^2_{\rm anal}\) | \(a\cdot\ell^2\) | \(3/2 - a\cdot\ell^2\) |
|---|---|---|---|---|
| \(\mu\) | \(a(\mu)\) | \(a\cdot\ell^2\) | \(3/2 - a\cdot\ell^2\) |
|---|---|---|---|
| 10 | 4.00 | 1.86101279 | \(-0.36101\) |
| 20 | 5.84 | 1.73647067 | \(-0.23647\) |
| 50 | 9.51 | 1.63985974 | \(-0.13986\) |
| 100 | 13.65 | 1.59565221 | \(-0.09565\) |
| 200 | 19.51 | 1.56607405 | \(-0.06607\) |
| 500 | 31.13 | 1.54093780 | \(-0.04094\) |
| 1000 | 44.22 | 1.52865045 | \(-0.02865\) |

The product \(a\cdot\ell^2\) decreases monotonically toward \(3/2 = 1.5\) from above. The residual \(a\ell^2 - 3/2\) shrinks roughly as \(1/a\) (\(0.0957\cdot 13.65 \approx 1.31\), \(0.0287\cdot 44.22 \approx 1.27\)), consistent with the \(O(1/a)\) correction; at \(\mu=1000\) it is still \(1.9\%\).

---

## 5. Discussion

### 5.1 What this is not

This note **does not** claim that:

- the number \(3/2\) has any relation to the golden ratio \(\Phi = (1+\sqrt5)/2\);
- there are connections with \(\psi = (1-\sqrt5)/2\), \(5/\Phi\), \(2/\Phi\), or other algebraic constants;
- the factor \(3/2\) encodes any hidden physical structure.

The factor \(3/2\) is the ratio between the fourth and the second moment of a Gaussian:

$$
\frac{\langle r^4\rangle_{\rm Gauss}}{\langle r^2\rangle_{\rm Gauss}} = \frac{3\sigma^4}{\sigma^2} = 3\sigma^2.
$$

The \(3\) is the coefficient of the Gaussian fourth moment, the \(2\) comes from \(\sigma^2 = 1/(2a)\). Nothing else.

### 5.2 What this is

An exact asymptotic result, derivable, with the following content:

1. The ground state of \(H(\mu) = -d^2/dr^2 - 2\mu\,\mathrm{sech}^2(r)\) is \(\psi_0 = \mathrm{sech}^{a(\mu)}(r)\) with \(a(\mu) = (\sqrt{1+8\mu}-1)/2\).
2. The effective width \(\ell^2_{\rm op}(\mu) = \langle r^4\rangle/\langle r^2\rangle\) is asymptotically \(3/(2a(\mu))\).
3. The first-order correction is \(O(a^{-1})\), consistent with the non-Gaussian expansion of \(\ln\mathrm{sech}\).

### 5.3 Application

This result has been used as an **ingredient** in the construction of a map between the Gentile gas \(Z_k(\beta) = \zeta(\beta)/\zeta((k+1)\beta)\) and the Jusufi–Anand metric \(f(r) = 1 - 2GMr/(r^2+\ell^2)\). In that context, \(\ell^2_{\rm op}\) is the effective operator length which, substituted into the metric, reproduces the horizon area. The asymptotic relation \(a\cdot\ell^2 \to 3/2\) is the one that fixes the scale of \(\ell^2\) for large \(\mu\). Section 2b of the script tests the matching \(\ell^2_{\rm op}(\mu,\varepsilon) = 4\,\ell^2_{\rm gas}(k,\beta)\) by choosing \((\mu,\varepsilon)\) on a \(40\times 8\) grid for each \((k,\beta)\): the match is within \(1.4\%\) for \(\beta\ge 1.5\) and between \(4.5\%\) and \(6.9\%\) at \(\beta = 1.05\), where the grid's smallest \(\ell^2_{\rm op}\) is reached.

---

## 6. References

1. G. Pöschl, E. Teller, *Bemerkungen zur Quantenmechanik des anharmonischen Oszillators*, Z. Phys. **83**, 143 (1933).
2. L. D. Landau, E. M. Lifshitz, *Quantum Mechanics: Non-Relativistic Theory*, 3rd ed., Pergamon (1977), §23.
3. F. Cooper, A. Khare, U. Sukhatme, *Supersymmetry and Quantum Mechanics*, Phys. Rep. **251**, 267 (1995).
4. R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey, D. E. Knuth, *On the Lambert W function*, Adv. Comput. Math. **5**, 329 (1996).
5. S. Pennacchio, *A commutative map between the Gentile statistical family and the Jusufi–Anand metric via the Lambert W function*, technical note (2026).

---

## 7. Verification

The verification code is `scripts/dvpt/dvpt_poschl_teller_width.py`: section 1 (setup), section 2a (numerical–analytical verification), section 2b (gas matching), section 3 (convergence to \(3/2\)).

All computations are reproducible with:

```
python scripts/dvpt/dvpt_poschl_teller_width.py
```

Requirements: `numpy`, `scipy`, `mpmath`. Default precision: mpmath at 40 decimal digits.

