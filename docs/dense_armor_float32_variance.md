# Variance in float32 with a first-element shift, a local centring and a compensated tree sum

Salvatore Pennacchio, October 2026.

Code: `dense_armor/utility/stats/stable.py` (Dense-Armor, branch `feature/float32-stats-kernel`, commit `1098759`).
Every number of this note is produced by `scripts/dense_armor_float32_variance/reproduce_float32_variance.py` and stored, in the same folder, in `results_cpu.json` (a Kaggle CPU), `results_gpu.json` (a Kaggle GPU) and `results_laptop_cpu.json` (a laptop CPU, used for the speed spread).

## Abstract

The variance of float32 data with a large offset and a small spread loses its digits in the subtraction `x - mean(x)`. This note describes a kernel in three steps, built on the shift by the first element, which was invented by the author: the shift, a local centring of the shifted values, and a sum of the squared residuals by a tree of error-free additions (TwoSum), which runs in `log2(n)` vectorised levels. On four synthetic datasets the kernel has a maximum relative error at most equal to the two-term formula of reference on 3 (eager) and 3 (jit) of the four on the CPU, and on 4 (eager) and 3 (jit) of the four on the GPU. Where it is worse is stated in Section 3. It returns exactly `0.0` on identical values up to `1e38` in eager and in jit, on both backends; at `1e38` the formula of reference returns `nan`. It costs time: the median ratio of its time to the time of the formula of reference on a `[4096, 4096]` array is 2.27 on the CPU (range 2.24 to 2.35) and 2.19 on the GPU (range 2.16 to 2.30).

## 1. The problem

For `x` in float32 along an axis of `n` samples, the variance is `(1/n) sum (x_i - mean)^2`. When the values sit around a large offset and their spread is small, `mean(x)` in float32 carries a rounding error of the order of `eps * |mean|`, with `eps = 2^-23`, and the subtraction `x - mean` passes that error to every residual.

The formula of reference used here is the one of `tf.nn.moments` in TensorFlow master (`nn_impl.py`, lines 1480-1487, read on 2026-10-29):

    m   = mean(x)
    d   = x - m
    var = max(mean(d^2) - mean(d)^2, 0)

A copy of it is in the script; nothing is sent to TensorFlow. The naive formula `mean((x - mean(x))^2)` is shown for comparison.

## 2. The kernel

Three steps, all in the input dtype (no float64).

1. **Shift (invented by the author, and the base of the kernel).** `y = x - x[0]` along the reduction axis. When the values are within a factor of two of the reference, the subtraction is exact (Sterbenz lemma; Shewchuk 1997, Lemma 5, PDF page 10). On D4 the script compares the float32 shift with the exact difference computed in float64: 100% of the 64000 entries are equal.
2. **Local centring.** `m = sum(y) / n`, `d = y - m`.
3. **Compensated tree sum.** The squares `d^2` are summed by a binary tree. At each level the pairs of partial sums are added with `TwoSum` (Ogita, Rump and Oishi 2005, Algorithm 3.1, PDF page 7; Shewchuk 1997, Theorem 7, PDF page 12), which returns the rounded sum and the exact rounding error; the errors are carried up the tree in a second array, and the result is the final sum plus the final carry. The tree has `log2(n)` levels of vectorised operations, so it runs on a GPU. The variance is `max(s2/n - (s1/n)^2, 0)` with `s1 = sum(d)`, rescaled by `n/(n - ddof)`. The mean is `x[0] + m`.

The compensated tree was added to the author's solution. The structure of the tree is my choice: the papers give `TwoSum`, not this tree. A complex input is split into real and imaginary part and the two variances are added, which is `E|z - mean(z)|^2`.

The kernel in the repository and the version written again in the script are equal bit for bit on the datasets D1, D2, D3, D4 on both backends.

## 3. Results

Datasets: synthetic, `numpy.random.default_rng(0)`, float32. Reference value: `numpy.var` in float64 of the same float32 values. Maximum relative error.

CPU:

| name | shape | axis | distribution | naive eager | reference eager | kernel eager | reference jit | kernel jit |
|---|---|---|---|---|---|---|---|---|
| D1 | [1000, 64] | 0 | N(1000, 0.01) | 1.86e-04 | 1.44e-07 | 1.17e-07 | 1.18e-07 | 1.17e-07 |
| D2 | [64, 1000] | 1 | N(10000, 1) | 3.99e-06 | 1.59e-07 | 1.15e-07 | 1.59e-07 | 1.15e-07 |
| D3 | [1000, 64] | 0 | N(0, 1) | 1.94e-07 | 1.94e-07 | 1.36e-07 | 1.94e-07 | 1.36e-07 |
| D4 | [1000, 64] | 0 | N(1e+06, 1) | 1.92e-02 | 1.18e-07 | 1.22e-07 | 9.97e-08 | 1.22e-07 |

GPU:

| name | shape | axis | distribution | naive eager | reference eager | kernel eager | reference jit | kernel jit |
|---|---|---|---|---|---|---|---|---|
| D1 | [1000, 64] | 0 | N(1000, 0.01) | 1.44e-04 | 1.35e-07 | 1.17e-07 | 1.35e-07 | 1.17e-07 |
| D2 | [64, 1000] | 1 | N(10000, 1) | 2.64e-06 | 1.70e-07 | 1.15e-07 | 1.70e-07 | 1.15e-07 |
| D3 | [1000, 64] | 0 | N(0, 1) | 1.44e-07 | 1.44e-07 | 1.36e-07 | 1.44e-07 | 1.64e-07 |
| D4 | [1000, 64] | 0 | N(1e+06, 1) | 1.95e-02 | 1.25e-07 | 1.22e-07 | 1.25e-07 | 1.22e-07 |

Kernel at most equal to the formula of reference: CPU eager 3/4, jit 3/4; GPU eager 4/4, jit 3/4. The cells where it is worse (CPU: D4 eager, D4 jit; GPU: D3 jit).

Variance of three identical values (eager / jit; the exact value is 0). CPU:

| input | naive | reference | kernel |
|---|---|---|---|
| `[1e10]*3` float32 | 1.04858e+06 / 1.04858e+06 | 0 / 0 | 0 / 0 |
| `[7.5e10]*3` float32 | 6.71089e+07 / 6.71089e+07 | 0 / 0 | 0 / 0 |
| `[1e15]*3` float32 | 4.5036e+15 / 4.5036e+15 | 0 / 0 | 0 / 0 |
| `[1e38]*3` float32 | inf / inf | nan / nan | 0 / 0 |
| `complex64 [7.5e10+7.5e10j]*3` | not run | not run | 0 / 0 |

GPU:

| input | naive | reference | kernel |
|---|---|---|---|
| `[1e10]*3` float32 | 1.04858e+06 / 1.04858e+06 | 0 / 0 | 0 / 0 |
| `[7.5e10]*3` float32 | 6.71089e+07 / 6.71089e+07 | 0 / 0 | 0 / 0 |
| `[1e15]*3` float32 | 4.5036e+15 / 4.5036e+15 | 0 / 0 | 0 / 0 |
| `[1e38]*3` float32 | inf / inf | nan / nan | 0 / 0 |
| `complex64 [7.5e10+7.5e10j]*3` | not run | not run | 0 / 0 |

Complex check: `[1+1j, 2+3j, -1+0.5j, 4-2j]` gives 6.421875 with the kernel and 6.421875 with numpy.

Speed on a synthetic `[4096, 4096]` float32 array, `axis=1`, under `jax.jit`, 7 runs after warm-up per session. Ratio kernel / reference, per session. CPU (5 sessions): 2.24, 2.32, 2.27, 2.26, 2.35. GPU (5 sessions): 2.19, 2.30, 2.20, 2.19, 2.16. On the CPU the median time of the reference varied between 64.1 and 67.4 ms from one session to another, so the ratio is stable only to the extent shown; on the GPU the reference takes 0.68 to 0.72 ms. On a laptop CPU (5 sessions) the ratio was: 1.70, 2.24, 4.07, 3.27, 3.12 (median 3.12, range 1.70 to 4.07). The errors of the reference and of the kernel on the laptop CPU are the same as on the Kaggle CPU in all sixteen cells.

## 4. Which step does what

Same datasets and the same structure (a `vmap` over one-dimensional slices), one step changed at a time. "At most equal" counts the datasets where the maximum error is not larger than that of the reference; speed is the median ratio to the reference (minimum-maximum over the sessions).

| variant | CPU eager | CPU jit | GPU eager | GPU jit | `[1e38]*3` CPU eager / jit | `[1e38]*3` GPU eager / jit | speed CPU | speed GPU |
|---|---|---|---|---|---|---|---|---|
| formula of reference (two-term, no shift, tree sum) | 4/4 | 4/4 | 4/4 | 4/4 | nan / nan | nan / nan | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) |
| shift + centring, tree sum, no compensation, no two-term | 2/4 | 0/4 | 3/4 | 2/4 | 0 / 0 | 0 / 0 | 1.01 (0.94-1.02) | 1.01 (1.00-1.05) |
| shift + centring + two-term, tree sum, no compensation | 2/4 | 0/4 | 3/4 | 2/4 | 0 / 0 | 0 / 0 | 1.82 (1.78-1.89) | 1.00 (0.98-1.07) |
| no shift, two-term, compensated tree | 4/4 | 4/4 | 4/4 | 4/4 | 0 / nan | nan / nan | 2.22 (2.11-2.28) | 2.17 (2.16-2.30) |
| shift + centring, compensated tree, no two-term | 3/4 | 3/4 | 4/4 | 3/4 | 0 / 0 | 0 / 0 | 1.75 (1.65-1.80) | 1.87 (1.82-1.93) |
| shift + centring + two-term, compensated tree (the kernel) | 3/4 | 3/4 | 4/4 | 3/4 | 0 / 0 | 0 / 0 | 2.27 (2.24-2.35) | 2.19 (2.16-2.30) |
| shift + centring, Kahan sequential scan (previous version) | 3/4 | 3/4 | 4/4 | 3/4 | 0 / 0 | 0 / 0 | 1.62 (1.55-1.64) | 42.96 (41.71-45.79) |

Reading, from the table: The shift is the step that gives the exact zero on `[1e38]*3` under `jit`: the variant without it (compensated tree, no shift) returns nan there on the CPU, and the kernel returns 0. The two-term correction does not change the result: the kernel and the variant without it have the same errors on all sixteen cells (four datasets, eager and jit, CPU and GPU). It is kept because the formula of reference has it. The compensated tree is the step that keeps the error under jit at the level of the reference: the shifted variants without compensation are at most equal on 0/4 (CPU) and 2/4 (GPU) datasets in jit, the kernel on 3/4 and 3/4. The Kahan sequential scan of the previous version is in the last row for comparison.

## 5. Limits

- **Accuracy.** The kernel is worse than the formula of reference in these cells: CPU: D4 eager, D4 jit; GPU: D3 jit. The shift is exact on D4 (Section 2), so on D4 the loss does not come from the subtraction. The reason is not explained in this note.
- **Time.** The ratio of the kernel to the reference is above 2 in 5 of 5 CPU sessions and 5 of 5 GPU sessions, and above 4 in 0 and 0. The repository states a threshold of 4.0 on the median ratio.
- **Not measured here.** float64 (`[1e307]*3`), TPU, other array sizes. The GPU run records the number of CUDA devices (0.11.1 gpu [CudaDevice(id=0), CudaDevice(id=1)]) but not the model of the card.
- **Compensated summation.** This note measures the effect of the tree on the datasets above and makes no claim on an error bound.

## 6. Reproduction

    cd scripts/dense_armor_float32_variance
    PYTHONPATH=<Dense-Armor checkout at commit 1098759> python reproduce_float32_variance.py --out results_cpu.json

On a GPU machine run the same command with `--out results_gpu.json`. Environment of the CPU run: jax 0.11.1, backend cpu, Python 3.13.15, `x86_64`. Environment of the GPU run: jax 0.11.1, backend gpu, Python 3.13.15, 64-bit mode off. Environment of the laptop run: jax 0.11.1, backend cpu, Python 3.12.10, `Intel64 Family 6 Model 142 Stepping 10, GenuineIntel`.

## 7. References

- Ogita, T., Rump, S. M., Oishi, S. (2005). Accurate Sum and Dot Product. SIAM Journal on Scientific Computing 26(6), 1955-1988. Algorithm 3.1 (`TwoSum`), PDF page 7 of the file read.
- Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates. Discrete and Computational Geometry 18(3), 305-363. Theorem 7 (`Two-Sum`), PDF page 12; Lemma 5 (Sterbenz), PDF page 10 of the file read.
- TensorFlow, `tensorflow/python/ops/nn_impl.py`, function `moments`, lines 1480-1487 (master, read on 2026-10-29): the formula of reference.
