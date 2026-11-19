# Variance in float32 with a first-element shift, a local centring and a compensated tree sum

Salvatore Pennacchio, November 2026.

## Abstract

The variance of float32 data with a large offset and a small wobble loses its digits in the subtraction of two almost equal numbers: on 3000 values around 10⁶ with a spread of 1, the one-pass formula `mean(x²) - mean(x)²` returns -196608 where the variance is 0.989742. We describe `stable_var`, a variance for float32 in four steps: a shift by the first element, a local centring, a sum of the squares by a binary tree of error-free `TwoSum` additions, and a two-term correction. It returns exactly `0.0` on identical values, up to `1e38` and on `complex64`. On four datasets, on a CPU and on a GPU, its maximum relative error is between 1.15·10⁻⁷ and 1.64·10⁻⁷, and it is at most equal to the error of the formula of `tf.nn.moments`, which has no shift, on three of the four datasets in each of the four combinations of backend and execution mode (eager, `jit`); it is worse on one dataset on the CPU and on one dataset on the GPU under `jit`, and we do not explain why. It costs 3.59 times the formula of `tf.nn.moments` on a CPU and 2.24 times on a GPU. We state what was not measured.

## 1. The problem

The variance of a sample is the average of the squared distances from the mean, `Var(x) = (1/n) Σ (xᵢ - mean(x))²`. A robot joint reports its angle as a float32 every millisecond; the angle sits around a large offset, the wobble around it is small, and the controller needs the variance of the wobble. The one-pass formula `mean(x²) - mean(x)²` subtracts two numbers of the order of `mean(x)²` to get a number of the order of the wobble squared, and the float32 mantissa is not wide enough.

We take 3000 float32 values from a normal distribution of mean 10⁶ and standard deviation 1 (seed 0). The float64 variance of the same float32 values is 0.989742; the one-pass formula in float32 gives -196608.000000. The two terms are of the order of `10¹²`, their rounding errors of the order of `10⁵`, a million times the wobble.

## 2. Related work

Chan, Golub and LeVeque [1] survey the algorithms for the sample variance with their round-off error bounds, including an analysis of the computation on shifted data: even a crude estimate of the mean used as a shift gives a large gain in accuracy without a first pass. Healy [3], for the mean and the variance in Monte-Carlo simulations, takes the first observation as the shift `K`, sums with Kahan's compensation and applies the two-term formula `T/n - (M - K)²`; his Algorithm 2 is the shifted naive algorithm with Kahan summation. The shift is exact when the values are within a factor of two of it (Sterbenz; Lemma 5 of [4]).

The compensated sum uses `TwoSum` of Knuth, which returns the rounded sum of two floating-point numbers and its exact rounding error [2, Algorithm 3.1]; see also Theorem 7 of [4].

This kernel adds to the algorithm of [3] the local centring `d = y - m` before the squares, a compensated binary tree of `TwoSum` in place of the sequential Kahan sum, and the evaluation one slice at a time with `vmap`. The tree has `log₂(n)` vectorised levels, so it runs on a GPU.

The baseline in the tests is the formula of `tf.nn.moments` in TensorFlow [5], `mean(d²) - mean(d)²` with `d = x - mean(x)`, which has no shift. An issue of TensorFlow reports its inaccuracy in float32 [6].

## 3. The kernel

`stable_var` returns the variance in the input dtype, with the `axis`, `ddof` and `keepdims` arguments of the other reducers of the library. Four steps, all in the input dtype.

1. **Shift.** Subtract the first element along the reduction axis, `y = x - x[0]`. When the values are within a factor of two of it, the subtraction is exact, and the large common exponent is gone. For `[1000.0, 1000.01, 1000.02]` in float32 the shifted values are `[0, 0.01000977, 0.02001953]`.
2. **Local centring.** The mean `m` of the shifted values is at the level of the wobble, and so is its rounding error, not at the level of the offset.
3. **Compensated tree.** The residuals `d = y - m` are formed and `s2 = Σ d²` is accumulated by a binary tree. At each level the pairs of partial sums are added with `TwoSum`, and the rounding errors are carried up the tree.
4. **Two-term correction and division.** With `s1 = Σ d`, the biased variance is `max(s2/n - (s1/n)², 0)`, multiplied by `n / (n - ddof)`. The mean is `x[0] + m`.

The idea of a carry that keeps the low-order bits can be shown with the sequential summation of Kahan [7], which is not the one the kernel uses. For the array `[1.0] + [1e-7] * 100000` in float32 the exact sum is 1.01; the sequential sum gives 1.01192093 (error 1.9·10⁻³) and the Kahan sum 1.00999999 (error 9.5·10⁻⁹).

The dtype policy: float32 in gives float32 out; float64 in gives float64 out when `jax_enable_x64` is on (off, JAX promotes to float32); float16 and bfloat16 are computed in float32 and returned in the input dtype; integers give float32; complex input gives the real `E|z - mean(z)|²`. `ddof = n` returns `nan` when the variance is positive and `0.0` when the values are identical.

The same path is on the estimators, through the `dtype` keyword of `RunningMoments`: on a float32 block of 500 values around 1000 with a spread of 0.01, `RunningMoments(dtype=np.float32)` gives a variance of 1.029553·10⁻⁴ against 1.029553·10⁻⁴ in float64, a relative difference of 4.91·10⁻⁹. The default `dtype=None` keeps the float64 result, bit for bit.

## 4. Experiments

The tests are those of the script `benchmarks/stable_variance_float32.py` of the library, which calls `dense_armor.core.signals.stats.stable.stable_var` itself. Three claims were declared in the first commit of the script and not changed after the run: (1) the kernel returns exactly `0.0` on every identical-values case, under `jit`; (2) on each of the four datasets its maximum relative error is at most the one of the baseline formula, in eager mode and under `jit`; (3) its time is at most 4.0 times the time of the baseline on a `[4096, 4096]` array, `axis=1`. The original target for the time was 2.0 times; it was not reached, and the threshold of 4.0 was chosen after the measurement.

The numbers of the CPU are in `benchmarks/stable_variance_float32.json` (JAX 0.11.1), the ones of the GPU in `benchmarks/stable_variance_float32_gpu.json` (JAX 0.12.0.1).

### 4.1 Identical values

The variance of identical values is exactly `0.0` for `1e10`, `7.5e10`, `1e15` and `1e38` in float32 and for `7.5e10` in `complex64`, on the CPU and on the GPU, under `jit`. The baseline formula gives `nan` at `1e38`. In the example `[1e10, 1e10, 1e10]` the result is the bit pattern `0x00000000`.

### 4.2 The four datasets

Maximum relative error against the float64 variance of the same float32 values, for the baseline formula of `tf.nn.moments` (base) and for `stable_var` (kernel), eager and under `jit`.

CPU:

| dataset | base eager | kernel eager | base jit | kernel jit |
|---|---|---|---|---|
| N(10³, 0.01), `[1000, 64]`, axis 0 | 1.44·10⁻⁷ | 1.17·10⁻⁷ | 1.18·10⁻⁷ | 1.17·10⁻⁷ |
| N(10⁴, 1), `[64, 1000]`, axis 1 | 1.59·10⁻⁷ | 1.15·10⁻⁷ | 1.59·10⁻⁷ | 1.15·10⁻⁷ |
| N(0, 1), `[1000, 64]`, axis 0 | 1.94·10⁻⁷ | 1.36·10⁻⁷ | 1.94·10⁻⁷ | 1.36·10⁻⁷ |
| N(10⁶, 1), `[1000, 64]`, axis 0 | 1.18·10⁻⁷ | **1.22·10⁻⁷** | 0.997·10⁻⁷ | **1.22·10⁻⁷** |

GPU:

| dataset | base eager | kernel eager | base jit | kernel jit |
|---|---|---|---|---|
| N(10³, 0.01), `[1000, 64]`, axis 0 | 1.35·10⁻⁷ | 1.17·10⁻⁷ | 1.35·10⁻⁷ | 1.17·10⁻⁷ |
| N(10⁴, 1), `[64, 1000]`, axis 1 | 1.70·10⁻⁷ | 1.15·10⁻⁷ | 1.70·10⁻⁷ | 1.15·10⁻⁷ |
| N(0, 1), `[1000, 64]`, axis 0 | 1.44·10⁻⁷ | 1.36·10⁻⁷ | 1.44·10⁻⁷ | **1.64·10⁻⁷** |
| N(10⁶, 1), `[1000, 64]`, axis 0 | 1.25·10⁻⁷ | 1.22·10⁻⁷ | 1.25·10⁻⁷ | 1.22·10⁻⁷ |

Claim 2 holds on the GPU in eager mode and fails there under `jit` (the dataset N(0, 1)), and fails on the CPU in both modes (the dataset N(10⁶, 1)). In bold, the cases in which the kernel is worse than the baseline: by 3 % (N(10⁶, 1), eager, CPU), 22 % (the same under `jit`) and 14 % (N(0, 1), `jit`, GPU). Where the kernel is better, the error is lower by up to 32 %.

### 4.3 The speed

A synthetic `[4096, 4096]` float32 array, `axis=1`, under `jit`, 15 runs after warm-up with `block_until_ready`, the median of each of three sessions on the CPU. The ratio is the median of the medians of the kernel over the median of the medians of the baseline.

| | naive two-pass | base | kernel | kernel / base | range over the sessions |
|---|---|---|---|---|---|
| CPU | 51.4 ms | 57.7 ms | 207.4 ms | 3.59 | 3.55 to 4.60 |
| GPU | 0.706 ms | 0.696 ms | 1.558 ms | 2.24 | 2.20 to 2.27 |

The kernel costs 3.59 times the baseline formula on the CPU and 2.24 times on the GPU. The target of 2.0 times was not reached. On the CPU the upper end of the range comes from the first session of the kernel (median 264.9 ms), the other two being 205.2 and 207.4 ms.

## 5. Declared limits

- The accuracy is at most equal to the baseline formula on three of the four datasets in each combination of backend and mode. It is worse on N(10⁶, 1) on the CPU, in eager mode and under `jit`, and on N(0, 1) under `jit` on the GPU. The cause is not explained.
- The speed is 3.59 times the baseline on the CPU and 2.24 times on the GPU; the threshold of 4.0 was chosen after the measurement.
- Not measured: float64 (`[1e307]*3`), TPU, other array sizes.
- The baseline of the tests has no shift: there is no comparison with a shifted baseline.
- `stable_var` is a function, not an estimator: it has no `learn_one`, no `budget_s`, no `memory_class`.
- The 1965 paper of Kahan [7] is bibliographic only, not read.

## 6. Reproduction

`python benchmarks/stable_variance_float32.py` in the library writes `benchmarks/stable_variance_float32.json` with the backend, the JAX version and the sessions of the speed measurement. The GPU numbers were obtained with the same script. As a check, the script was run again on a Kaggle GPU session with JAX 0.11.1, with `stable.py` and the script taken unchanged from the library: the accuracy numbers are equal to the ones of `stable_variance_float32_gpu.json` to all the digits, the claims are the same, and the speed ratio is 2.22 (range 2.18 to 2.25). The code blocks of the documentation page of `stable_var` were run with the function and their printed outputs are the ones shown there.

## References

1. T. F. Chan, G. H. Golub, R. J. LeVeque. *Algorithms for Computing the Sample Variance: Analysis and Recommendations.* The American Statistician 37(3), 242-247, 1983.
2. T. Ogita, S. M. Rump, S. Oishi. *Accurate Sum and Dot Product.* SIAM Journal on Scientific Computing 26(6), 1955-1988, 2005. Algorithm 3.1 (`TwoSum`).
3. J. Healy. *Accurate and consistent calculation of the mean and variance in Monte-Carlo simulations.* arXiv:2206.10662v7, 2022. Algorithm 2.
4. J. R. Shewchuk. *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates.* Discrete and Computational Geometry 18(3), 305-363, 1997. Theorem 7 (`Two-Sum`); Lemma 5 (Sterbenz).
5. TensorFlow, `tensorflow/python/ops/nn_impl.py`, function `moments`, lines 1480-1487 (master, read on 2026-10-29).
6. TensorFlow, issue #117286: the inaccuracy of `tf.nn.moments` in float32.
7. W. Kahan. *Further remarks on reducing truncation errors.* Communications of the ACM 8(1), 40, 1965.
