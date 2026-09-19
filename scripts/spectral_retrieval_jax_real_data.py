"""Real-data run of the JAX spectral-retrieval encoder (see
scripts/spectral_retrieval_jax.py for the architecture, verified there on
tiny synthetic data first). This script is the deferred "real-data
validation on Kaggle" step that script's own docstring anticipated.

Runs on the Kaggle CASMI26 kernel, data mounted server-side -- never
downloaded locally (a 3GB local download of this competition's
train.parquet caused a real PC freeze in an earlier session). Path is
/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/, WITH
the "competitions/" segment: a sibling kernel this session
(armatura-spectral-retrieval-test) used the path without it and failed
with FileNotFoundError -- confirmed here against a working sibling kernel
(casmi26-wiring-spot-check) that printed the real mounted paths.

Self-contained (architecture functions copied from spectral_retrieval_jax.py
rather than imported): every kernel script in this project's Kaggle history
is a single uploaded file, no cross-file imports.

PREREGISTERED EXPECTATIONS, stated before running this:
1. The architecture already passed the toy overfit check (loss decreases,
   no dead gradients) -- real spectra could still break it via shapes/
   ranges the toy data never exercised (variable peak counts already
   handled in principle, but real value ranges/precision are untested).
2. A short real-data training run on ONE fixed real batch (not a fresh
   random batch per step -- see Step 3's own comment for the real
   OOM this caused on the first attempt) is a SPOT CHECK for "does this
   learn anything at all and not crash," not a competition-grade result.
3. The retrieval check here is IN-BATCH training accuracy on that same
   fixed batch, not held-out generalization -- a real comparison against
   the sibling armatura-spectral-retrieval-test kernel's held-out
   precursor-tolerance retrieval needs the architecture padded to a fixed
   peak length first (see Details), out of scope for this spot check.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import jax
import jax.numpy as jnp
import optax
from rdkit import Chem
from rdkit.Chem import AllChem

TRAIN_PATH = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
D_MODEL, N_FREQS, FP_SIZE = 64, 16, 256
N_SUBSET = 500        # spot-check scale, not the full competition training set
BATCH_SIZE = 16
N_STEPS = 300
PRECURSOR_TOL_PPM = 20.0


def layer_norm(x, gain, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gain + bias


def init_params(key, d_model, n_freqs, fp_size):
    keys = jax.random.split(key, 8)
    scale = 1.0 / jnp.sqrt(d_model)
    return {
        "fourier_b": jax.random.normal(keys[0], (n_freqs,)) * 10.0,
        "int_w": jax.random.normal(keys[1], (1, d_model - 2 * n_freqs)) * scale,
        "int_b": jnp.zeros((d_model - 2 * n_freqs,)),
        "precursor_w": jax.random.normal(keys[2], (1, d_model)) * scale,
        "precursor_b": jnp.zeros((d_model,)),
        "wq": jax.random.normal(keys[3], (d_model, d_model)) * scale,
        "wk": jax.random.normal(keys[4], (d_model, d_model)) * scale,
        "wv": jax.random.normal(keys[5], (d_model, d_model)) * scale,
        "attn_pool_w": jax.random.normal(keys[6], (d_model, 1)) * scale,
        "head_w": jax.random.normal(keys[7], (d_model, fp_size)) * scale,
        "head_b": jnp.zeros((fp_size,)),
        "ln1_gain": jnp.ones((d_model,)), "ln1_bias": jnp.zeros((d_model,)),
        "ln2_gain": jnp.ones((d_model,)), "ln2_bias": jnp.zeros((d_model,)),
    }


def encode_spectrum(params, mzs, intensities, precursor_mz, n_freqs):
    mz_col = mzs[:, None] / 1000.0
    projected = 2 * jnp.pi * mz_col * params["fourier_b"]
    mz_emb = jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)

    log_intensity = jnp.log1p(intensities)[:, None]
    int_emb = log_intensity @ params["int_w"] + params["int_b"]

    peak_embs = jnp.concatenate([mz_emb, int_emb], axis=-1)

    prec_feat = (jnp.array([[precursor_mz / 1000.0]]) @ params["precursor_w"] + params["precursor_b"])
    x = peak_embs + prec_feat
    x = layer_norm(x, params["ln1_gain"], params["ln1_bias"])

    q = x @ params["wq"]
    k = x @ params["wk"]
    v = x @ params["wv"]
    d_model = x.shape[-1]
    attn_logits = (q @ k.T) / jnp.sqrt(d_model)
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    x_out = x + (attn_weights @ v)
    x_out = layer_norm(x_out, params["ln2_gain"], params["ln2_bias"])

    pool_logits = x_out @ params["attn_pool_w"]
    pool_weights = jax.nn.softmax(pool_logits, axis=0)
    global_repr = jnp.sum(x_out * pool_weights, axis=0)

    return global_repr @ params["head_w"] + params["head_b"]


def info_nce_loss(params, batch, n_freqs, temp=0.1):
    preds = jnp.stack([
        encode_spectrum(params, batch["mzs"][i], batch["intensities"][i],
                         batch["precursor_mz"][i], n_freqs)
        for i in range(len(batch["mzs"]))
    ])
    preds = preds / jnp.linalg.norm(preds, axis=-1, keepdims=True)
    targets = batch["candidate_fps"] / jnp.maximum(jnp.linalg.norm(batch["candidate_fps"], axis=-1, keepdims=True), 1e-8)
    logits = (preds @ targets.T) / temp
    labels = jnp.arange(len(batch["mzs"]))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(len(labels)), labels])


def morgan_fp(smiles, n_bits):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(bv, dtype=np.float32)


def main():
    # NOTE: pq.ParquetFile(...).schema.names (the flat PHYSICAL Parquet schema)
    # flattens list<element: double> columns down to their child's own name
    # ("element"), colliding across the 3 real list columns here and hiding
    # the real top-level names entirely -- confirmed via schema_arrow /
    # pandas .columns on a real run (see scripts/agent_injection, same
    # verify-before-assuming discipline). Use pandas columns, not
    # ParquetFile.schema.names, to see real top-level names for this file.
    real_cols = pq.ParquetFile(TRAIN_PATH).schema_arrow.names
    print(f"[INFO] train.parquet columns: {real_cols}")
    needed = ["inchikey14", "normalized_smiles", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]
    missing = [c for c in needed if c not in real_cols]
    if missing:
        raise SystemExit(f"[FATAL] expected columns missing: {missing} -- real schema: {real_cols}")

    df = pd.read_parquet(TRAIN_PATH, columns=needed)
    print(f"[INFO] loaded {len(df)} labeled spectra, {df['inchikey14'].nunique()} unique molecules")

    rng = np.random.default_rng(2026)
    subset_idx = rng.permutation(len(df))[:N_SUBSET]
    df = df.iloc[subset_idx].reset_index(drop=True)

    print(f"\n[Step 1] compute real Morgan fingerprints (radius=2, {FP_SIZE} bits) from real SMILES")
    fp_cache = {}
    fps = np.zeros((len(df), FP_SIZE), dtype=np.float32)
    n_fp_failed = 0
    for i, smi in enumerate(df["normalized_smiles"]):
        if smi not in fp_cache:
            fp = morgan_fp(smi, FP_SIZE)
            fp_cache[smi] = fp
        fp = fp_cache[smi]
        if fp is None:
            n_fp_failed += 1
        else:
            fps[i] = fp
    print(f"  {n_fp_failed}/{len(df)} SMILES failed to parse (real RDKit failures, not hidden)")
    valid_mask = np.array([fp_cache[s] is not None for s in df["normalized_smiles"]])
    df = df[valid_mask].reset_index(drop=True)
    fps = fps[valid_mask]
    print(f"  {len(df)} spectra remain with a valid fingerprint")

    split = int(len(df) * 0.8)
    train_df, train_fps = df.iloc[:split], fps[:split]
    eval_df, eval_fps = df.iloc[split:].reset_index(drop=True), fps[split:]
    print(f"  train={len(train_df)}  eval={len(eval_df)}")

    key = jax.random.PRNGKey(2026)
    params = init_params(key, D_MODEL, N_FREQS, FP_SIZE)

    print(f"\n[Step 2] verify forward pass on a REAL spectrum (not synthetic)")
    row0 = train_df.iloc[0]
    fp0 = encode_spectrum(
        params, jnp.array(row0["ms2_mzs"], dtype=jnp.float32),
        jnp.array(row0["ms2_normalized_intensities"], dtype=jnp.float32),
        jnp.float32(row0["precursor_mz"]), N_FREQS,
    )
    print(f"  n_peaks={len(row0['ms2_mzs'])}  output shape={fp0.shape}  any NaN={bool(jnp.any(jnp.isnan(fp0)))}")

    # ONE fixed real batch, sampled once and reused for every step -- same
    # "toy overfit check" discipline as spectral_retrieval_jax.py's own
    # Step 2, now with real spectra substituted for synthetic ones. NOT a
    # random-batch-per-step loop: real MS/MS spectra have widely varying
    # peak counts, and a fresh batch each step means a fresh combination of
    # array shapes each step. Confirmed on this exact machine/data (kernel
    # version 2 of this script): that variation drives JAX's per-shape
    # compilation cache to grow without bound across steps, and the process
    # gets SIGKILL'd by the OOM killer around step 30-50 (no Python
    # traceback -- "Killed" in the log, ~1800s in). A fixed batch means a
    # fixed set of shapes for the whole run: compiled once, reused, no leak.
    fixed_idx = rng.choice(len(train_df), size=BATCH_SIZE, replace=False)
    fixed_batch = {
        "mzs": [jnp.array(train_df.iloc[i]["ms2_mzs"], dtype=jnp.float32) for i in fixed_idx],
        "intensities": [jnp.array(train_df.iloc[i]["ms2_normalized_intensities"], dtype=jnp.float32) for i in fixed_idx],
        "precursor_mz": jnp.array(train_df.iloc[fixed_idx]["precursor_mz"].to_numpy(), dtype=jnp.float32),
        "candidate_fps": jnp.array(train_fps[fixed_idx]),
    }
    n_peaks_in_batch = [len(m) for m in fixed_batch["mzs"]]
    print(f"\n[Step 3] train {N_STEPS} steps on ONE fixed real batch (n={BATCH_SIZE}), "
          f"n_peaks range in this batch: {min(n_peaks_in_batch)}-{max(n_peaks_in_batch)}")
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(params)
    p = params
    losses = []
    loss_fn = lambda pp: info_nce_loss(pp, fixed_batch, N_FREQS)
    for step in range(N_STEPS):
        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        if not np.isfinite(float(loss_val)):
            print(f"  [FAIL] non-finite loss at step {step}: {loss_val} -- real-data bug, stopping")
            break
        updates, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        losses.append(float(loss_val))
        if step % 50 == 0 or step == N_STEPS - 1:
            print(f"  step {step:4d}  loss={losses[-1]:.4f}")

    if losses:
        print(f"\n  loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")

    print(f"\n[Step 4] in-batch retrieval accuracy on the SAME fixed real batch "
          f"(training accuracy, not held-out generalization -- see Details in the doc "
          f"for why a real held-out retrieval eval needs the padding/masking fix first)")
    preds = jnp.stack([
        encode_spectrum(p, fixed_batch["mzs"][i], fixed_batch["intensities"][i],
                         fixed_batch["precursor_mz"][i], N_FREQS)
        for i in range(BATCH_SIZE)
    ])
    preds = np.array(preds / jnp.linalg.norm(preds, axis=-1, keepdims=True))
    targets = np.array(fixed_batch["candidate_fps"]) / np.maximum(
        np.linalg.norm(np.array(fixed_batch["candidate_fps"]), axis=-1, keepdims=True), 1e-8)
    sims = preds @ targets.T  # (batch, batch): row i's similarity to every candidate fingerprint
    top1_hits = int(np.sum(np.argmax(sims, axis=1) == np.arange(BATCH_SIZE)))
    print(f"  top1 in-batch accuracy: {top1_hits}/{BATCH_SIZE} = {top1_hits/BATCH_SIZE:.3f} "
          f"(chance level = 1/{BATCH_SIZE} = {1.0/BATCH_SIZE:.3f})")


if __name__ == "__main__":
    main()
