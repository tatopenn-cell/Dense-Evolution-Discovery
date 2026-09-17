# JAX spectral-retrieval architecture for CASMI26 molecule ID

Matching a real MS/MS spectrum to the right candidate molecule out of
thousands is a retrieval problem: encode the spectrum, encode each
candidate, and score them by similarity. This architecture encodes an
m/z spectrum with Fourier features, mixes peaks with one self-attention
layer, pools to a single vector, and trains that encoder with an InfoNCE
contrastive loss against candidate fingerprints.

## Step 1: encode a spectrum

```python
params = init_params(key, n_fourier=32, d_model=64)
z = encode_spectrum(params, mz, intensity)
```

`mz`/`intensity` are the peaks of one spectrum. `encode_spectrum` returns
one fixed-size vector `z` per spectrum, regardless of how many peaks it
had.

## Step 2: train with InfoNCE

```python
loss, grads = jax.value_and_grad(info_nce_loss)(params, batch)
```

`info_nce_loss` pulls a spectrum's encoding toward its true candidate's
fingerprint and away from the other candidates in the same batch.

## Optimizer choice, isolated-variable test

Same architecture, same toy batch, only the optimizer changed:

| optimizer | loss (30-100 steps) |
|---|---|
| plain SGD (lr=0.05) | 2.2739 -> 2.0594 |
| Adam (lr=1e-2) | 2.2094 -> 0.5328 |

![SGD plateaus, Adam converges on the same toy batch](assets/jax_spectral_retrieval_casmi26/spectral_retrieval_optimizer_test.png)

## Details

Inspired by MSAlign (arXiv:2605.19752, indexed in quantumrag) and a
public PyTorch reference implementation, neither of which is JAX-based or
uses chemistry-informed features.

Verified correct on synthetic toy data before the optimizer test: forward-pass
shape/NaN check, gradient-flow check (`jax.value_and_grad`, no
zero-norm/dead parameters) -- so SGD's plateau above was diagnosed as an
optimizer-choice issue, not a broken gradient.

Not yet run on real competition data. Real-data validation was
deliberately deferred to a Kaggle kernel (where `train.parquet` is
already mounted server-side) rather than downloading the competition's
3GB `train.parquet` locally -- an earlier session mistake that
contributed to a real PC freeze.

**Script**: `scripts/spectral_retrieval_jax.py`.
