# JAX spectral-retrieval architecture for CASMI26 molecule ID

First step toward a genuine (not copied) JAX contribution for the Kaggle
`enveda-CASMI26-molecule-id-mass-spectra` competition, inspired by
MSAlign (arXiv:2605.19752, indexed in quantumrag) and a public PyTorch
reference implementation -- neither of which is JAX-based or uses
chemistry-informed features.

## Architecture

Fourier-feature m/z encoding, single-head self-attention with
residual+layer-norm, attention pooling, InfoNCE loss against candidate
fingerprints. Verified correct on synthetic toy data: forward-pass
shape/NaN check, gradient-flow check (`jax.value_and_grad`, no
zero-norm/dead parameters).

## Optimizer choice, isolated-variable test

Same architecture, same toy batch, only the optimizer changed:

| optimizer | loss (30-100 steps) |
|---|---|
| plain SGD (lr=0.05) | 2.2739 -> 2.0594 (fails a 50% reduction target) |
| Adam (lr=1e-2) | 2.2094 -> 0.5328 (passes) |

SGD's plateau was diagnosed as an optimizer-choice issue, not a broken
gradient -- the gradient-flow check already confirmed gradients were
nonzero everywhere; switching to Adam (matching every real reference
implementation) fixed it immediately.

## Status

Not yet run on real competition data. Real-data validation was
deliberately deferred to a Kaggle kernel (where `train.parquet` is
already mounted server-side) rather than downloading the competition's
3GB `train.parquet` locally -- an earlier session mistake that
contributed to a real PC freeze.

Script: `scripts/spectral_retrieval_jax.py`.
