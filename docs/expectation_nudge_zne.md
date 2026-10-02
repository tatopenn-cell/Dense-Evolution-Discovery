# Expectation-Value Nudge for ZNE: No Signal

Script: `scripts/expectation_nudge_zne.py`. Negative result.

Question: can the Richardson nudge of the library's predictive ZNE be driven by the three measured expectation values alone (no tomography, any number of qubits)? Signal: `(|y2 - y3| - |y1 - y2|) / (|y2 - y3| + |y1 - y2|)`, rectified.

Benchmark after Majumdar et al. (arXiv:2307.05203, Sect. IV, Fig. 6): 4-qubit brickwork circuits (random single-qubit gates, CZ), local noise after every gate with probability `p` scaled by the factor, factors 1, 2, 3, 8000 shots per factor, observable `Z` on qubit 0, exact density-matrix simulation (checked against full Kronecker products to 3e-17). 100 random circuits per row; RMSE against the exact zero-noise value.

| Channel | Depth | p | Richardson | Linear | Exponential (a = 0) | Nudge | Nudge shuffled | Constant nudge | Curvature test |
|---|---|---|---|---|---|---|---|---|---|
| depolarizing | 4 | 0.005 | 0.0416 | 0.0151 | 0.0321 | 0.0404 | 0.0398 | 0.0369 | 0.0233 |
| depolarizing | 4 | 0.020 | 0.0466 | 0.0241 | 0.0308 | 0.0451 | 0.0457 | 0.0438 | 0.0326 |
| depolarizing | 12 | 0.005 | 0.0437 | 0.0265 | 1.9824 | 0.0414 | 0.0417 | 0.0382 | 0.0347 |
| depolarizing | 12 | 0.020 | 0.0875 | 0.1243 | 0.4225 | 0.0866 | 0.0879 | 0.0894 | 0.1034 |
| amplitude damping | 4 | 0.005 | 0.0466 | 0.0189 | 0.0212 | 0.0443 | 0.0446 | 0.0416 | 0.0299 |
| amplitude damping | 4 | 0.020 | 0.0456 | 0.0186 | 3.0381 | 0.0438 | 0.0444 | 0.0417 | 0.0273 |
| amplitude damping | 12 | 0.005 | 0.0390 | 0.0208 | 0.0502 | 0.0375 | 0.0379 | 0.0357 | 0.0250 |
| amplitude damping | 12 | 0.020 | 0.0513 | 0.0883 | 6.8648 | 0.0509 | 0.0515 | 0.0524 | 0.0693 |

## Reading it

- The nudge equals its shuffled control in every row: the signal from three expectation values carries no information about when to extrapolate less.
- As Majumdar et al. report, the best extrapolator depends on the regime: linear wins at weak noise (up to 3× lower error than Richardson), Richardson wins at depth 12 with `p = 0.02`. The exponential fit is unstable in several rows, also as they describe (Appendix A).
- A curvature test (linear if `|y1 - 2 y2 + y3|` is within two shot-noise standard deviations, Richardson otherwise), following their advice to check whether the data vary enough before choosing, lands between the two and is never the best: at 8000 shots the test has too little power to separate the regimes reliably.
- Nothing here is ready for promotion.

## Bounded exponential extrapolation in this benchmark

Same circuits; the library's `bounded_exponential_extrapolate` (Miranskyy, Sorrenti, Thind, Gravel, arXiv:2604.24475: zero-noise value constrained to [-1, 1], asymptote fitted) and the paper's variant with the asymptote fixed to 0 (their Table I), fitted in the script.

| Channel | Depth | p | Linear | Richardson | Bounded exp (library, 3 parameters) | Bounded exp, asymptote 0 |
|---|---|---|---|---|---|---|
| depolarizing | 4 | 0.005 | 0.0151 | 0.0416 | 0.5720 | 0.1425 |
| depolarizing | 4 | 0.020 | 0.0241 | 0.0466 | 0.3814 | 0.0897 |
| depolarizing | 12 | 0.005 | 0.0265 | 0.0437 | 0.3458 | 0.0306 |
| depolarizing | 12 | 0.020 | 0.1243 | 0.0875 | 0.4366 | 0.2312 |
| amplitude damping | 4 | 0.005 | 0.0189 | 0.0466 | 0.5136 | 0.0206 |
| amplitude damping | 4 | 0.020 | 0.0186 | 0.0456 | 0.2995 | 0.1122 |
| amplitude damping | 12 | 0.005 | 0.0208 | 0.0390 | 0.3896 | 0.1020 |
| amplitude damping | 12 | 0.020 | 0.0883 | 0.0513 | 0.2893 | 0.2785 |

- On the exact expectation values (no shot noise) the library function recovers the zero-noise value to RMSE 0.0001 (40 circuits, depolarizing, depth 4, `p = 0.005`). With 8000 shots it lands on the bound ±1 in 11 of those 40 circuits: three noise factors and three free parameters let the fit follow the shot noise, the instability Majumdar et al. describe for nearly λ-independent data (Appendix A). The bound keeps the result in range but does not make it accurate.
- Fixing the asymptote to 0 removes one parameter and helps, but it is still worse than linear or Richardson in most rows, and 0 is the wrong asymptote for amplitude damping, which drives `<Z>` towards +1.
- Miranskyy et al. also report that the benefit of bounding is model-dependent. For Dense-Evolution this is a limitation worth stating in `bounded_exponential_extrapolate`'s documentation: with three noise factors and realistic shot counts it is not reliable.
