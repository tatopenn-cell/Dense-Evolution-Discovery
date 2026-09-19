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

## Step 3: real data, and a real bug the toy check couldn't see

Run on the Kaggle CASMI26 kernel (2,539,608 real labeled spectra, 275,810
unique molecules, `train.parquet` mounted server-side, never downloaded
locally). First attempt: a fresh random batch of 16 real spectra every
training step, same as the toy script's pattern but with real molecules
sampled each time.

```
[Step 3] train 300 steps, batch=16, on real spectra
  step    0  loss=2.9718
Killed
```

**Real spectra have wildly variable peak counts** (this dataset: from 3 to
over a thousand peaks per spectrum) -- the architecture is never padded to
a fixed length, so a fresh random batch each step means a fresh
combination of array shapes each step. On this machine, that drove JAX's
per-shape compilation caching to grow without bound: the process was
SIGKILL'd by the OOM killer around step 30-50, no Python traceback, just
`Killed` in the log at ~1800s in. The toy check never caught this because
it reuses the exact same fixed-shape batch (`n_peaks=20` for all 8 toy
samples) for all 100 steps -- zero shape variation, so it never compiles
more than once.

**Fix**: one fixed real batch, sampled once, reused for every step -- same
discipline as the toy overfit check above, real spectra substituted in.

```
[Step 3] train 300 steps on ONE fixed real batch (n=16), n_peaks range in this batch: 3-1268
  step    0  loss=2.9172
  step   50  loss=0.0336
  step  299  loss=0.0056
[Step 4] in-batch retrieval accuracy on the SAME fixed real batch
  top1 in-batch accuracy: 16/16 = 1.000 (chance level = 1/16 = 0.062)
```

Real, clean result: loss drops from 2.9172 to 0.0056 over 300 steps on
real MS/MS spectra and real RDKit Morgan fingerprints (0/500 SMILES
failed to parse in this subset), no NaN anywhere, perfect in-batch
retrieval. This validates the full real-data path end to end (parquet
schema, fingerprinting, forward pass, gradients, optimizer) -- it is
**not** a held-out generalization result: 16/16 on the same batch the
model was trained on is expected once loss is this low, closer to a
memorization check than a retrieval benchmark.

## Details

Inspired by MSAlign (arXiv:2605.19752, indexed in quantumrag) and a
public PyTorch reference implementation, neither of which is JAX-based or
uses chemistry-informed features.

Verified correct on synthetic toy data before the optimizer test: forward-pass
shape/NaN check, gradient-flow check (`jax.value_and_grad`, no
zero-norm/dead parameters) -- so SGD's plateau above was diagnosed as an
optimizer-choice issue, not a broken gradient.

**Real schema note**: `train.parquet`'s true top-level columns are
`inchikey14`/`normalized_smiles` (not `molecule_id`/`smiles`, an
unverified guess made -- and never actually exercised, since that kernel
failed on an unrelated path bug first -- in the sibling
`armatura-spectral-retrieval-test` Kaggle kernel). Also:
`pq.ParquetFile(path).schema.names` (the flat physical Parquet schema)
collapses this file's 3 `list<element: double>` columns down to their
shared child name, printing `"element"` three times and hiding the real
names (`ms2_mzs`, `ms2_normalized_intensities`, `collision_energy_ev`)
entirely -- `pq.ParquetFile(path).schema_arrow.names` or plain
`pd.read_parquet(...).columns` give the real top-level names.

**Still open**: a real held-out retrieval comparison against
`armatura-spectral-retrieval-test`'s binned-cosine baseline needs the
architecture padded to a fixed peak length (with an attention mask so
padding doesn't corrupt the pooled representation) -- only then can a
multi-batch training run and a real library-wide retrieval eval run
without hitting the same shape-variation/OOM issue Step 3 found. Not yet
done.

**Scripts**: `scripts/spectral_retrieval_jax.py` (toy/synthetic
verification), `scripts/spectral_retrieval_jax_real_data.py` (the real-data
run above).
