# Variance in float32 with a first-element shift, a local centring and a compensated sum of squares

Salvatore Pennacchio, October 2026.

Code: `dense_armor/utility/stats/stable.py` (Dense-Armor, branch `feature/float32-stats-kernel`, commit `d347071`).
Every number of this note is produced by `scripts/dense_armor_float32_variance/reproduce_float32_variance.py` and stored in `scripts/dense_armor_float32_variance/results.json`.

## Abstract

The variance of float32 data with a large offset and a small spread loses its digits in the subtraction `x - mean(x)`. This note describes a three-step kernel: a shift by the first element, a local centring of the shifted values, and a Kahan compensated sum of the squared residuals. On four synthetic datasets the kernel has a maximum relative error at most equal to the two-term formula of reference on 3 of the four, in eager mode and under `jax.jit` (on the other one, D4, it is worse). It returns exactly `0.0` on identical values up to `1e38` in both modes; at `1e38` the formula of reference returns `nan`. It costs time: the median ratio of its time to the time of the formula of reference on a `[4096, 4096]` array is 2.47, with a range of 1.50 to 2.56 over 5 measurement sessions on one CPU.

## 1. The problem

For `x` in float32 along an axis of `n` samples, the variance is `(1/n) sum (x_i - mean)^2`. When the values sit around a large offset and their spread is small, `mean(x)` in float32 carries a rounding error of the order of `eps * |mean|`, with `eps = 2^-23`, and the subtraction `x - mean` passes that error to every residual.

The formula of reference used here is the one of `tf.nn.moments` in TensorFlow master (`nn_impl.py`, lines 1480-1487, read on 2026-10-29):

    m   = mean(x)
    d   = x - m
    var = max(mean(d^2) - mean(d)^2, 0)

A copy of it is in the script; nothing is sent to TensorFlow. The naive formula `mean((x - mean(x))^2)` is shown for comparison.

## 2. The kernel

Three steps, all in the input dtype (no float64).

1. **Shift.** `y = x - x[0]` along the reduction axis. When the values are within a factor of two of the reference, the subtraction is exact (Sterbenz lemma; Shewchuk 1997, Lemma 5, PDF page 10). On D4 the script compares the float32 shift with the exact difference computed in float64: 100% of the 64000 entries are equal.
2. **Local centring.** `m = sum(y) / n`, `d = y - m`.
3. **Compensated sum.** `s2 = sum(d^2)` is accumulated with the compensated summation of Kahan (a `jax.lax.scan`), and `var = s2 / (n - ddof)`. The mean is `x[0] + m`. A complex input is split into real and imaginary part and the two variances are added, which is `E|z - mean(z)|^2`.

The kernel in the repository and the version written again in the script are equal bit for bit on the datasets D1, D2, D3, D4.

## 3. Results

Datasets: synthetic, `numpy.random.default_rng(0)`, float32. Reference value: `numpy.var` in float64 of the same float32 values. Maximum relative error:

| name | shape | axis | distribution | naive eager | reference eager | kernel eager | reference jit | kernel jit |
|---|---|---|---|---|---|---|---|---|
| D1 | [1000, 64] | 0 | N(1000, 0.01) | 1.86e-04 | 1.44e-07 | 1.17e-07 | 1.18e-07 | 1.17e-07 |
| D2 | [64, 1000] | 1 | N(10000, 1) | 3.99e-06 | 1.59e-07 | 1.15e-07 | 1.59e-07 | 1.15e-07 |
| D3 | [1000, 64] | 0 | N(0, 1) | 1.94e-07 | 1.94e-07 | 1.36e-07 | 1.94e-07 | 1.36e-07 |
| D4 | [1000, 64] | 0 | N(1e+06, 1) | 1.92e-02 | 1.18e-07 | 1.22e-07 | 9.97e-08 | 1.22e-07 |

Kernel at most equal to the formula of reference: eager 3/4, jit 3/4. The dataset where it is not: D4 (jit).

Variance of three identical values (eager / jit; the exact value is 0):

| input | naive | reference | kernel |
|---|---|---|---|
| `[1e10]*3` float32 | 1.04858e+06 / 1.04858e+06 | 0 / 0 | 0 / 0 |
| `[7.5e10]*3` float32 | 6.71089e+07 / 6.71089e+07 | 0 / 0 | 0 / 0 |
| `[1e15]*3` float32 | 4.5036e+15 / 4.5036e+15 | 0 / 0 | 0 / 0 |
| `[1e38]*3` float32 | inf / inf | nan / nan | 0 / 0 |
| `complex64 [7.5e10+7.5e10j]*3` | not run | not run | 0 / 0 |

Complex check: `[1+1j, 2+3j, -1+0.5j, 4-2j]` gives 6.421875 with the kernel and 6.421875 with numpy.

Speed on a synthetic `[4096, 4096]` float32 array, `axis=1`, under `jax.jit`, 7 runs after warm-up per session, 5 sessions. The ratio kernel / reference, per session: 2.13, 1.50, 2.53, 2.47, 2.56. The median time of the reference varied between 35.6 and 69.3 ms from one session to another, so the ratio is stable only to the extent shown.

## 4. Which step does what

Same datasets and the same structure (a `vmap` over one-dimensional slices), one step changed at a time. "At most equal" counts the datasets where the maximum error is not larger than that of the reference; speed is the median ratio to the reference (minimum-maximum over the sessions).

| variant | eager | jit | `[1e38]*3` eager / jit | speed |
|---|---|---|---|---|
| formula of reference (two-term, no shift, tree sum) | 4/4 | 4/4 | nan / nan | 1.00 (1.00-1.00) |
| shift + centring + tree sum (no compensation, no two-term) | 2/4 | 0/4 | 0 / 0 | 0.96 (0.65-1.23) |
| shift + centring + two-term + tree sum | 2/4 | 0/4 | 0 / 0 | 1.81 (1.05-2.42) |
| no shift + two-term + Kahan on `d^2` | 4/4 | 4/4 | 0 / nan | 3.24 (1.89-3.58) |
| shift + centring + Kahan on `d^2` (the kernel) | 3/4 | 3/4 | 0 / 0 | 2.47 (1.50-2.56) |

Reading, from the table: the shift is the step that gives the exact zero on `[1e38]*3` under `jit` (the variant without it returns `nan` there). The two-term correction does not change the result: the two variants that differ only by it have the same errors on all eight cells. Kahan's sum is the step that keeps the error under jit at the level of the reference on D1, D2 and D3; without it the shifted variant is above the reference on all four datasets in jit. Kahan's sum is also the step that costs time.

## 5. Limits

- **D4.** The kernel is above the formula of reference on D4 (N(1e6, 1)), by a small margin (1.22e-07 against 1.18e-07 in eager and 9.97e-08 in jit). The shift is exact on D4 (Section 2), so the loss does not come from the subtraction. The reason is not explained in this note.
- **Time.** Kahan's sum is sequential. The kernel is above a factor of 2 of the reference in 4 of the 5 sessions.
- **Not measured here.** float64 (`[1e307]*3`), GPU and TPU, other array sizes, a parallel compensated sum.
- **Compensated summation** is expected to reduce the accumulation error (Kahan 1965, not read for this note: bibliographic only). This note measures its effect on the datasets above and makes no claim on its error bound.

## 6. Reproduction

    cd scripts/dense_armor_float32_variance
    PYTHONPATH=<Dense-Armor checkout at commit d347071> python reproduce_float32_variance.py --out results.json

Environment of the run: jax 0.11.1, backend cpu, Python 3.12.10, `Intel64 Family 6 Model 142 Stepping 10, GenuineIntel`, 64-bit mode off.

## 7. References

- Kahan, W. (1965). Further remarks on reducing truncation errors. Communications of the ACM 8(1), 40. Bibliographic only.
- Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates. Discrete and Computational Geometry 18(3), 305-363. Lemma 5 (Sterbenz), PDF page 10 of the file read.
- TensorFlow, `tensorflow/python/ops/nn_impl.py`, function `moments`, lines 1480-1487 (master, read on 2026-10-29): the formula of reference.
