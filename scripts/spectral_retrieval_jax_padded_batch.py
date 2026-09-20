"""Padded/masked, jax.vmap-batched version of the CASMI26 spectral-retrieval
encoder (see scripts/spectral_retrieval_jax_real_data.py for the first
real-data run, which found a real OOM bug from unpadded variable-length
spectra and worked around it with a single fixed batch -- a training/
memorization check, not held-out generalization). This script closes that
gap: pad every spectrum to one fixed peak count with an attention mask,
so a fresh random real batch every step has IDENTICAL shape regardless of
which spectra get sampled -- one JAX compilation, real multi-batch
training, and a real held-out retrieval evaluation become possible.

Correctness of the padding/masking itself was verified locally BEFORE
touching real data (not asserted here, run separately): the padded+masked
encoder is numerically identical (max abs diff ~1e-7, float32 precision)
to the original unpadded encoder on a no-padding-needed case, and
invariant to how much extra padding is added on top.

PREREGISTERED EXPECTATIONS, stated before running this:
1. MAX_PEAKS is chosen from a real percentile of this dataset's num_peaks
   (not guessed) -- spectra above it are truncated to their top-MAX_PEAKS
   most intense peaks (standard MS/MS preprocessing for bounding attention
   cost), not dropped.
2. A real multi-batch run (fresh random batch every step) should no longer
   OOM now that every batch has one fixed shape -- if it still does, that
   is a second, different real bug, not expected.
3. Held-out retrieval accuracy is the real generalization question Step
   3/4 of the real-data run could not answer. v1 of this script's own
   protocol had a real bug (candidate pools from a too-small library,
   never checked for containing the true molecule) -- v2 (this version)
   fixes it: a much larger real candidate library, discarding queries
   whose true molecule isn't verifiably in the pool, reporting real pool
   sizes, and a random baseline conditioned on evaluability.
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
N_SUBSET = 60000      # up from 6000 -- see Step 5's real-candidate-pool bug fix: a 6000-row
                       # subsample gave near-empty precursor-tolerance pools (mean ~1.6 candidates),
                       # not a real retrieval task. Training config (N_TRAIN/BATCH_SIZE/N_STEPS)
                       # deliberately UNCHANGED from the previous run -- isolating the eval-protocol
                       # fix as the one variable, not also changing how much the model is trained.
N_TRAIN = 4800
N_QUERY_HOLDOUT = 300  # held out from the candidate library entirely, so a query never trivially matches its own spectrum
BATCH_SIZE = 32
N_STEPS = 500
PRECURSOR_TOL_PPM = 20.0
MAX_PEAKS_PERCENTILE = 90
MAX_PEAKS_CEILING = 256  # bounds attention cost (O(max_peaks^2)) regardless of the percentile
N_CAPACITY_MOLECULES = 100  # Step 6: small-pool memorization capacity check
N_CAPACITY_STEPS = 5000


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


def encode_spectrum_padded(params, mzs, intensities, precursor_mz, mask, n_freqs):
    """mzs/intensities/mask: (max_peaks,), fixed shape. mask: 1.0 real peak,
    0.0 padding. Masked BOTH in the self-attention (key axis) and in the
    final pooling softmax, so padding contributes exactly zero regardless
    of the (numerically harmless) garbage values self-attention computes
    at padded query rows -- those rows are never read."""
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
    attn_logits = jnp.where(mask[None, :] > 0, attn_logits, -1e9)
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    x_out = x + (attn_weights @ v)
    x_out = layer_norm(x_out, params["ln2_gain"], params["ln2_bias"])

    pool_logits = (x_out @ params["attn_pool_w"])[:, 0]
    pool_logits = jnp.where(mask > 0, pool_logits, -1e9)
    pool_weights = jax.nn.softmax(pool_logits)[:, None]
    global_repr = jnp.sum(x_out * pool_weights, axis=0)

    return global_repr @ params["head_w"] + params["head_b"]


encode_batched = jax.vmap(encode_spectrum_padded, in_axes=(None, 0, 0, 0, 0, None))


def info_nce_loss_batched(params, mzs, intensities, precursor_mz, mask, candidate_fps, n_freqs, temp=0.1):
    preds = encode_batched(params, mzs, intensities, precursor_mz, mask, n_freqs)
    preds = preds / jnp.maximum(jnp.linalg.norm(preds, axis=-1, keepdims=True), 1e-8)
    targets = candidate_fps / jnp.maximum(jnp.linalg.norm(candidate_fps, axis=-1, keepdims=True), 1e-8)
    logits = (preds @ targets.T) / temp
    labels = jnp.arange(preds.shape[0])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def morgan_fp(smiles, n_bits):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(bv, dtype=np.float32)


def pad_one(mzs, intensities, max_peaks):
    n = len(mzs)
    if n >= max_peaks:
        top_idx = np.argsort(-intensities)[:max_peaks]
        top_idx = top_idx[np.argsort(mzs[top_idx])]  # keep m/z ascending, cosmetic only
        return mzs[top_idx].astype(np.float32), intensities[top_idx].astype(np.float32), np.ones(max_peaks, dtype=np.float32)
    pad = max_peaks - n
    mzs_p = np.concatenate([mzs, np.zeros(pad)]).astype(np.float32)
    inten_p = np.concatenate([intensities, np.zeros(pad)]).astype(np.float32)
    mask = np.concatenate([np.ones(n), np.zeros(pad)]).astype(np.float32)
    return mzs_p, inten_p, mask


def main():
    real_cols = pq.ParquetFile(TRAIN_PATH).schema_arrow.names
    needed = ["inchikey14", "normalized_smiles", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "num_peaks"]
    missing = [c for c in needed if c not in real_cols]
    if missing:
        raise SystemExit(f"[FATAL] expected columns missing: {missing} -- real schema: {real_cols}")

    df = pd.read_parquet(TRAIN_PATH, columns=needed)
    print(f"[INFO] loaded {len(df)} labeled spectra, {df['inchikey14'].nunique()} unique molecules")

    rng = np.random.default_rng(2026)
    subset_idx = rng.permutation(len(df))[:N_SUBSET]
    df = df.iloc[subset_idx].reset_index(drop=True)

    pcts = np.percentile(df["num_peaks"].to_numpy(), [50, 90, 95, 99, 100])
    max_peaks = int(min(np.percentile(df["num_peaks"].to_numpy(), MAX_PEAKS_PERCENTILE), MAX_PEAKS_CEILING))
    print(f"\n[Step 1] num_peaks percentiles (this subset, n={N_SUBSET}): "
          f"p50={pcts[0]:.0f} p90={pcts[1]:.0f} p95={pcts[2]:.0f} p99={pcts[3]:.0f} max={pcts[4]:.0f}")
    print(f"  MAX_PEAKS = {max_peaks} (p{MAX_PEAKS_PERCENTILE}, capped at {MAX_PEAKS_CEILING})")

    print(f"\n[Step 2] compute real Morgan fingerprints (radius=2, {FP_SIZE} bits) and pad/truncate spectra")
    fp_cache = {}
    fps = np.zeros((len(df), FP_SIZE), dtype=np.float32)
    n_fp_failed = 0
    all_mzs = np.zeros((len(df), max_peaks), dtype=np.float32)
    all_inten = np.zeros((len(df), max_peaks), dtype=np.float32)
    all_mask = np.zeros((len(df), max_peaks), dtype=np.float32)
    n_truncated = 0
    for i, row in df.iterrows():
        smi = row["normalized_smiles"]
        if smi not in fp_cache:
            fp_cache[smi] = morgan_fp(smi, FP_SIZE)
        fp = fp_cache[smi]
        if fp is None:
            n_fp_failed += 1
        else:
            fps[i] = fp
        mzs = np.asarray(row["ms2_mzs"], dtype=np.float64)
        inten = np.asarray(row["ms2_normalized_intensities"], dtype=np.float64)
        if len(mzs) > max_peaks:
            n_truncated += 1
        all_mzs[i], all_inten[i], all_mask[i] = pad_one(mzs, inten, max_peaks)
    print(f"  {n_fp_failed}/{len(df)} SMILES failed to parse")
    print(f"  {n_truncated}/{len(df)} spectra truncated to top-{max_peaks} most intense peaks")

    valid_mask = np.array([fp_cache[s] is not None for s in df["normalized_smiles"]])
    df = df[valid_mask].reset_index(drop=True)
    fps, all_mzs, all_inten, all_mask = fps[valid_mask], all_mzs[valid_mask], all_inten[valid_mask], all_mask[valid_mask]
    precursor_all = df["precursor_mz"].to_numpy(dtype=np.float32)
    mol_ids_all = df["inchikey14"].to_numpy()
    print(f"  {len(df)} spectra remain")

    all_idx = np.arange(len(df))
    query_idx = rng.choice(all_idx, size=min(N_QUERY_HOLDOUT, len(all_idx)), replace=False)
    query_set = set(query_idx.tolist())
    remaining_idx = np.array([i for i in all_idx if i not in query_set])
    train_idx_pool = rng.choice(remaining_idx, size=min(N_TRAIN, len(remaining_idx)), replace=False)
    library_idx = remaining_idx  # candidate pool for retrieval -- everything NOT held out as a query
    print(f"  train_pool={len(train_idx_pool)}  library={len(library_idx)}  query_holdout={len(query_idx)}")

    key = jax.random.PRNGKey(2026)
    params = init_params(key, D_MODEL, N_FREQS, FP_SIZE)
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def _step(p, opt_state, mzs_b, inten_b, prec_b, mask_b, fps_b):
        loss_fn = lambda pp: info_nce_loss_batched(pp, mzs_b, inten_b, prec_b, mask_b, fps_b, N_FREQS)
        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_state, loss_val

    print(f"\n[Step 3] verify one fresh random batch shape is jit-stable (2 different random batches)")
    for trial in range(2):
        idx = rng.choice(train_idx_pool, size=BATCH_SIZE, replace=False)
        p2, os2, l2 = _step(params, opt_state, all_mzs[idx], all_inten[idx], precursor_all[idx], all_mask[idx], fps[idx])
        print(f"  trial {trial}: loss={float(l2):.4f} (no shape error, no recompilation needed after trial 0)")

    print(f"\n[Step 4] real multi-batch training, {N_STEPS} steps, batch={BATCH_SIZE}, "
          f"FRESH random real batch every step (this is what OOM'd before padding)")
    p = params
    losses = []
    for step in range(N_STEPS):
        idx = rng.choice(train_idx_pool, size=BATCH_SIZE, replace=False)
        p, opt_state, loss_val = _step(p, opt_state, all_mzs[idx], all_inten[idx], precursor_all[idx], all_mask[idx], fps[idx])
        loss_val = float(loss_val)
        if not np.isfinite(loss_val):
            print(f"  [FAIL] non-finite loss at step {step} -- stopping")
            break
        losses.append(loss_val)
        if step % 50 == 0 or step == N_STEPS - 1:
            print(f"  step {step:4d}  loss={loss_val:.4f}")
    if losses:
        chance_loss = float(np.log(BATCH_SIZE))
        print(f"\n  loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}  "
              f"(log({BATCH_SIZE})={chance_loss:.4f} = in-batch InfoNCE loss for a uniform-random guesser)")

    print(f"\n[Step 5] REAL held-out retrieval accuracy -- FIXED protocol (v2), against a "
          f"{len(library_idx)}-item real candidate library (not the {N_QUERY_HOLDOUT}-item "
          f"query set itself, which never overlaps it). Fixes a real bug found in the v1 run: "
          f"a query only counts as evaluable if (a) its precursor-tolerance pool is non-empty "
          f"AND (b) the true molecule is actually IN that pool -- v1 checked neither, and its "
          f"random-baseline formula (mean of 1/pool_size) came out to 0.6248, revealing pools "
          f"averaging ~1.6 real candidates from a too-small 6000-row library, most of which "
          f"could not possibly contain the true molecule. Candidates are compared against their "
          f"REAL ground-truth Morgan fingerprints directly (not another spectrum's predicted "
          f"fingerprint) -- also removes the need to JAX-encode the whole library, only the "
          f"{N_QUERY_HOLDOUT} queries need encoding.")
    library_fps = fps[library_idx]
    library_fps_norm = library_fps / np.maximum(np.linalg.norm(library_fps, axis=-1, keepdims=True), 1e-8)
    library_mol_ids = mol_ids_all[library_idx]
    library_precursor = precursor_all[library_idx]

    query_preds = np.array(encode_batched(p, jnp.array(all_mzs[query_idx]), jnp.array(all_inten[query_idx]),
                                           jnp.array(precursor_all[query_idx]), jnp.array(all_mask[query_idx]), N_FREQS))
    query_preds = query_preds / np.maximum(np.linalg.norm(query_preds, axis=-1, keepdims=True), 1e-8)

    RANKS = (1, 5, 10, 25)
    hits = {k: 0 for k in RANKS}
    n_scored = 0
    n_empty_pool = 0
    n_true_absent = 0
    pool_sizes = []
    random_baseline_sum = 0.0
    freq_baseline_hits = 0
    for qi_local, qi_global in enumerate(query_idx):
        true_id = mol_ids_all[qi_global]
        tol = precursor_all[qi_global] * PRECURSOR_TOL_PPM / 1e6
        cand_mask = np.abs(library_precursor - precursor_all[qi_global]) <= tol
        cand_indices = np.where(cand_mask)[0]
        if len(cand_indices) == 0:
            n_empty_pool += 1
            continue
        if true_id not in library_mol_ids[cand_indices]:
            n_true_absent += 1
            continue
        n_scored += 1
        pool_sizes.append(len(cand_indices))

        sims = library_fps_norm[cand_indices] @ query_preds[qi_local]
        order = np.argsort(-sims)
        ranked_mol_ids = library_mol_ids[cand_indices][order]
        for k in RANKS:
            if true_id in ranked_mol_ids[:k]:
                hits[k] += 1

        random_baseline_sum += 1.0 / len(cand_indices)
        uniq, counts = np.unique(library_mol_ids[cand_indices], return_counts=True)
        if uniq[np.argmax(counts)] == true_id:
            freq_baseline_hits += 1

    print(f"  query_holdout={len(query_idx)}  empty_pool={n_empty_pool}  "
          f"true_molecule_absent_from_pool={n_true_absent}  evaluable(n_scored)={n_scored}")
    if pool_sizes:
        pool_sizes = np.array(pool_sizes)
        print(f"  pool size among evaluable queries: mean={pool_sizes.mean():.1f}  "
              f"median={np.median(pool_sizes):.0f}  min={pool_sizes.min()}  max={pool_sizes.max()}")
    if n_scored:
        rank_str = "  ".join(f"top{k}={hits[k]/n_scored:.4f}" for k in RANKS)
        print(f"  encoder:          {rank_str}")
        print(f"  random baseline:  top1={random_baseline_sum/n_scored:.4f}  "
              f"(mean of 1/pool_size, conditioned on the true molecule being IN the pool)")
        print(f"  most-frequent-in-pool baseline: top1={freq_baseline_hits/n_scored:.4f}")
    else:
        print("  [FATAL for interpretation] zero evaluable queries -- library still too small/sparse")

    print(f"\n[Step 5b] representation-collapse check: distribution of pairwise cosine "
          f"similarities across the query set's predicted fingerprints -- if these cluster "
          f"near 1.0 regardless of molecule identity, the encoder has collapsed to mapping "
          f"every spectrum near the same point instead of separating them")
    sim_matrix = query_preds @ query_preds.T
    off_diag = sim_matrix[~np.eye(len(query_preds), dtype=bool)]
    print(f"  pairwise cosine similarity (n={len(query_preds)} spectra, {len(off_diag)} pairs): "
          f"mean={off_diag.mean():.4f}  std={off_diag.std():.4f}  "
          f"min={off_diag.min():.4f}  max={off_diag.max():.4f}")

    print(f"\n[Step 6] capacity diagnostic: can the SAME architecture memorize a SMALL real pool "
          f"({N_CAPACITY_MOLECULES} molecules) given many more steps? Distinguishes "
          f"'needs more training' from 'architecture/features can't separate real spectra'")
    cap_pool_size = min(N_CAPACITY_MOLECULES, len(train_idx_pool))
    cap_idx_pool = rng.choice(train_idx_pool, size=cap_pool_size, replace=False)
    key_cap = jax.random.PRNGKey(7)
    params_cap = init_params(key_cap, D_MODEL, N_FREQS, FP_SIZE)
    opt_state_cap = optimizer.init(params_cap)
    p_cap = params_cap
    cap_batch_size = min(BATCH_SIZE, cap_pool_size)
    cap_losses = []
    for step in range(N_CAPACITY_STEPS):
        idx = rng.choice(cap_idx_pool, size=cap_batch_size, replace=False)
        p_cap, opt_state_cap, loss_val = _step(p_cap, opt_state_cap, all_mzs[idx], all_inten[idx],
                                                precursor_all[idx], all_mask[idx], fps[idx])
        cap_losses.append(float(loss_val))
        if step % 1000 == 0 or step == N_CAPACITY_STEPS - 1:
            print(f"  step {step:5d}  loss={cap_losses[-1]:.4f}")
    cap_preds = np.array(encode_batched(p_cap, jnp.array(all_mzs[cap_idx_pool]), jnp.array(all_inten[cap_idx_pool]),
                                         jnp.array(precursor_all[cap_idx_pool]), jnp.array(all_mask[cap_idx_pool]), N_FREQS))
    cap_preds = cap_preds / np.maximum(np.linalg.norm(cap_preds, axis=-1, keepdims=True), 1e-8)
    cap_fps = fps[cap_idx_pool] / np.maximum(np.linalg.norm(fps[cap_idx_pool], axis=-1, keepdims=True), 1e-8)
    cap_sims = cap_preds @ cap_fps.T
    cap_top1 = int(np.sum(np.argmax(cap_sims, axis=1) == np.arange(cap_pool_size)))
    print(f"  in-pool top1 after {N_CAPACITY_STEPS} steps on {cap_pool_size} molecules: "
          f"{cap_top1}/{cap_pool_size} = {cap_top1/cap_pool_size:.4f} "
          f"(chance = 1/{cap_pool_size} = {1.0/cap_pool_size:.4f})")


if __name__ == "__main__":
    main()
