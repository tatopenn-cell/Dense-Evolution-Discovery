"""Stage 2: a cross-encoder-style re-ranker on top of the already-validated
stage-1 encoder (spectral_retrieval_jax_padded_batch.py) -- the real
pattern behind modern large-scale search (fast approximate retrieval via
independent embeddings + cosine, THEN a joint scorer that looks at query
and candidate TOGETHER to re-rank the shortlist), not a bigger/different
stage-1 model. Same two-stage principle as Goldman, Bradshaw, Xin & Coley,
"Prefix-Tree Decoding for Predicting Mass Spectra from Molecules"
(NeurIPS 2023, arXiv:2303.06470) and its ICEBERG successor (Goldman et
al. 2024): generate/retrieve a candidate set first, then score/rank it
with a second, more expensive model that looks at query and candidate
jointly.

Stage 1 (unchanged architecture): padded/masked Fourier+attention encoder,
jax.vmap batched, trained with InfoNCE against real Morgan fingerprints.

Stage 2 (new): for each query, take its stage-1 top-K candidates (by
cosine similarity to real fingerprints) and score EACH (query, candidate)
pair jointly with a small MLP over [pred_fp, cand_fp, |diff|, product] --
trained with a listwise softmax over the real shortlist (the TRUE
candidate should get the highest score), not random in-batch negatives:
the shortlist already contains the hardest, most informative negatives
(near neighbors in fingerprint space), which random negatives never do.

Three-way split, no leakage: train_pool (stage 1), rerank_train_pool
(builds stage-2 training shortlists), query_holdout (final evaluation
only, never touched by either training stage).

PREREGISTERED EXPECTATION: stage1+stage2 top1 on query_holdout should
beat stage-1-alone top1 on the SAME query_holdout -- if the joint scorer
adds nothing, that is a real, reportable negative result, not hidden.
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
N_SUBSET = 300000
N_TRAIN = 4800
N_RERANK_TRAIN = 300
N_QUERY_HOLDOUT = 300
BATCH_SIZE = 32
N_STEPS = 500
PRECURSOR_TOL_PPM = 20.0
MAX_PEAKS_PERCENTILE = 90
MAX_PEAKS_CEILING = 256
TOP_K_SHORTLIST = 25
RERANK_HIDDEN = 128
RERANK_STEPS = 400


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
    mz_col = mzs[:, None] / 1000.0
    projected = 2 * jnp.pi * mz_col * params["fourier_b"]
    mz_emb = jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)
    log_intensity = jnp.log1p(intensities)[:, None]
    int_emb = log_intensity @ params["int_w"] + params["int_b"]
    peak_embs = jnp.concatenate([mz_emb, int_emb], axis=-1)
    prec_feat = (jnp.array([[precursor_mz / 1000.0]]) @ params["precursor_w"] + params["precursor_b"])
    x = peak_embs + prec_feat
    x = layer_norm(x, params["ln1_gain"], params["ln1_bias"])
    q = x @ params["wq"]; k = x @ params["wk"]; v = x @ params["wv"]
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


def init_reranker(key, fp_size, hidden):
    k1, k2 = jax.random.split(key)
    in_dim = fp_size * 4
    scale1 = 1.0 / jnp.sqrt(in_dim)
    scale2 = 1.0 / jnp.sqrt(hidden)
    return {
        "w1": jax.random.normal(k1, (in_dim, hidden)) * scale1,
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(k2, (hidden, 1)) * scale2,
        "b2": jnp.zeros((1,)),
    }


def rerank_score(params, pred_fp, cand_fp):
    feat = jnp.concatenate([pred_fp, cand_fp, jnp.abs(pred_fp - cand_fp), pred_fp * cand_fp])
    h = jax.nn.relu(feat @ params["w1"] + params["b1"])
    return (h @ params["w2"] + params["b2"])[0]


rerank_score_batched = jax.vmap(rerank_score, in_axes=(None, None, 0))


def pad_one(mzs, intensities, max_peaks):
    n = len(mzs)
    if n >= max_peaks:
        top_idx = np.argsort(-intensities)[:max_peaks]
        top_idx = top_idx[np.argsort(mzs[top_idx])]
        return mzs[top_idx].astype(np.float32), intensities[top_idx].astype(np.float32), np.ones(max_peaks, dtype=np.float32)
    pad = max_peaks - n
    mzs_p = np.concatenate([mzs, np.zeros(pad)]).astype(np.float32)
    inten_p = np.concatenate([intensities, np.zeros(pad)]).astype(np.float32)
    mask = np.concatenate([np.ones(n), np.zeros(pad)]).astype(np.float32)
    return mzs_p, inten_p, mask


def morgan_fp(smiles, n_bits):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(bv, dtype=np.float32)


def get_shortlist(query_pred, library_fps_norm, library_precursor, library_mol_ids, query_precursor, k):
    tol = query_precursor * PRECURSOR_TOL_PPM / 1e6
    cand_mask = np.abs(library_precursor - query_precursor) <= tol
    cand_indices = np.where(cand_mask)[0]
    if len(cand_indices) == 0:
        return None
    sims = library_fps_norm[cand_indices] @ query_pred
    order = np.argsort(-sims)[:k]
    return cand_indices[order]


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

    max_peaks = int(min(np.percentile(df["num_peaks"].to_numpy(), MAX_PEAKS_PERCENTILE), MAX_PEAKS_CEILING))
    print(f"[Step 1] MAX_PEAKS={max_peaks}")

    fp_cache = {}
    fps = np.zeros((len(df), FP_SIZE), dtype=np.float32)
    all_mzs = np.zeros((len(df), max_peaks), dtype=np.float32)
    all_inten = np.zeros((len(df), max_peaks), dtype=np.float32)
    all_mask = np.zeros((len(df), max_peaks), dtype=np.float32)
    for i, row in df.iterrows():
        smi = row["normalized_smiles"]
        if smi not in fp_cache:
            fp_cache[smi] = morgan_fp(smi, FP_SIZE)
        fp = fp_cache[smi]
        if fp is not None:
            fps[i] = fp
        mzs = np.asarray(row["ms2_mzs"], dtype=np.float64)
        inten = np.asarray(row["ms2_normalized_intensities"], dtype=np.float64)
        all_mzs[i], all_inten[i], all_mask[i] = pad_one(mzs, inten, max_peaks)

    valid_mask = np.array([fp_cache[s] is not None for s in df["normalized_smiles"]])
    df = df[valid_mask].reset_index(drop=True)
    fps, all_mzs, all_inten, all_mask = fps[valid_mask], all_mzs[valid_mask], all_inten[valid_mask], all_mask[valid_mask]
    precursor_all = df["precursor_mz"].to_numpy(dtype=np.float32)
    mol_ids_all = df["inchikey14"].to_numpy()
    print(f"[Step 2] {len(df)} spectra with valid fingerprints")

    all_idx = np.arange(len(df))
    rerank_train_idx = rng.choice(all_idx, size=min(N_RERANK_TRAIN, len(all_idx)), replace=False)
    remaining1 = np.array([i for i in all_idx if i not in set(rerank_train_idx.tolist())])
    query_idx = rng.choice(remaining1, size=min(N_QUERY_HOLDOUT, len(remaining1)), replace=False)
    remaining2 = np.array([i for i in remaining1 if i not in set(query_idx.tolist())])
    train_idx_pool = rng.choice(remaining2, size=min(N_TRAIN, len(remaining2)), replace=False)
    library_idx = remaining2
    print(f"[Step 3] train_pool={len(train_idx_pool)}  library={len(library_idx)}  "
          f"rerank_train={len(rerank_train_idx)}  query_holdout={len(query_idx)}")

    key = jax.random.PRNGKey(2026)
    params = init_params(key, D_MODEL, N_FREQS, FP_SIZE)
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def _step(p, opt_state, mzs_b, inten_b, prec_b, mask_b, fps_b):
        loss_fn = lambda pp: info_nce_loss_batched(pp, mzs_b, inten_b, prec_b, mask_b, fps_b, N_FREQS)
        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(p, updates), new_opt_state, loss_val

    print(f"\n[Step 4] STAGE 1: train {N_STEPS} steps, batch={BATCH_SIZE}")
    p = params
    for step in range(N_STEPS):
        idx = rng.choice(train_idx_pool, size=BATCH_SIZE, replace=False)
        p, opt_state, loss_val = _step(p, opt_state, all_mzs[idx], all_inten[idx], precursor_all[idx], all_mask[idx], fps[idx])
        if step % 100 == 0 or step == N_STEPS - 1:
            print(f"  step {step:4d}  loss={float(loss_val):.4f}")

    library_fps = fps[library_idx]
    library_fps_norm = library_fps / np.maximum(np.linalg.norm(library_fps, axis=-1, keepdims=True), 1e-8)
    library_precursor = precursor_all[library_idx]
    library_mol_ids = mol_ids_all[library_idx]

    def encode_one(gi):
        pred = np.array(encode_spectrum_padded(p, jnp.array(all_mzs[gi]), jnp.array(all_inten[gi]),
                                                jnp.float32(precursor_all[gi]), jnp.array(all_mask[gi]), N_FREQS))
        return pred / max(np.linalg.norm(pred), 1e-8)

    print(f"\n[Step 5] STAGE-1-ONLY baseline on query_holdout ({len(query_idx)} queries)")
    stage1_hits, stage1_scored = 0, 0
    query_preds_cache = {}
    for gi in query_idx:
        pred = encode_one(gi)
        query_preds_cache[gi] = pred
        shortlist = get_shortlist(pred, library_fps_norm, library_precursor, library_mol_ids, precursor_all[gi], TOP_K_SHORTLIST)
        if shortlist is None or mol_ids_all[gi] not in library_mol_ids[shortlist]:
            continue
        stage1_scored += 1
        if library_mol_ids[shortlist[0]] == mol_ids_all[gi]:
            stage1_hits += 1
    print(f"  stage1 top1: {stage1_hits}/{stage1_scored} evaluable ({stage1_hits/max(stage1_scored,1):.4f})")

    print(f"\n[Step 6] Build STAGE-2 training shortlists from rerank_train_pool ({len(rerank_train_idx)} queries)")
    rerank_examples = []
    for gi in rerank_train_idx:
        pred = encode_one(gi)
        shortlist = get_shortlist(pred, library_fps_norm, library_precursor, library_mol_ids, precursor_all[gi], TOP_K_SHORTLIST)
        if shortlist is None or mol_ids_all[gi] not in library_mol_ids[shortlist]:
            continue
        true_pos = int(np.where(library_mol_ids[shortlist] == mol_ids_all[gi])[0][0])
        rerank_examples.append((pred, library_fps_norm[shortlist], true_pos))
    print(f"  {len(rerank_examples)} usable (query,shortlist) training examples for the re-ranker")

    print(f"\n[Step 7] STAGE 2: train re-ranker MLP, {RERANK_STEPS} steps")
    rr_params = init_reranker(jax.random.PRNGKey(7), FP_SIZE, RERANK_HIDDEN)
    rr_opt = optax.adam(1e-3)
    rr_opt_state = rr_opt.init(rr_params)

    def rerank_loss(rp, pred_fp, cand_fps, true_pos):
        scores = rerank_score_batched(rp, pred_fp, cand_fps)
        log_probs = jax.nn.log_softmax(scores)
        return -log_probs[true_pos]

    @jax.jit
    def rr_step(rp, opt_state, pred_fp, cand_fps, true_pos):
        loss_fn = lambda p_: rerank_loss(p_, pred_fp, cand_fps, true_pos)
        loss_val, grads = jax.value_and_grad(loss_fn)(rp)
        updates, new_opt_state = rr_opt.update(grads, opt_state)
        return optax.apply_updates(rp, updates), new_opt_state, loss_val

    if rerank_examples:
        for step in range(RERANK_STEPS):
            ex = rerank_examples[rng.integers(0, len(rerank_examples))]
            pred_fp, cand_fps, true_pos = ex
            rr_params, rr_opt_state, loss_val = rr_step(rr_params, rr_opt_state, jnp.array(pred_fp), jnp.array(cand_fps), true_pos)
            if step % 100 == 0 or step == RERANK_STEPS - 1:
                print(f"  step {step:4d}  loss={float(loss_val):.4f}")

    print(f"\n[Step 8] STAGE1+STAGE2 (re-ranked) on the SAME query_holdout")
    stage2_hits, stage2_scored = 0, 0
    for gi in query_idx:
        pred = query_preds_cache[gi]
        shortlist = get_shortlist(pred, library_fps_norm, library_precursor, library_mol_ids, precursor_all[gi], TOP_K_SHORTLIST)
        if shortlist is None or mol_ids_all[gi] not in library_mol_ids[shortlist]:
            continue
        stage2_scored += 1
        scores = np.array(rerank_score_batched(rr_params, jnp.array(pred), jnp.array(library_fps_norm[shortlist])))
        best = shortlist[np.argmax(scores)]
        if library_mol_ids[best] == mol_ids_all[gi]:
            stage2_hits += 1
    print(f"  stage1+2 top1: {stage2_hits}/{stage2_scored} evaluable ({stage2_hits/max(stage2_scored,1):.4f})")

    print(f"\n[SUMMARY] stage1-only top1={stage1_hits/max(stage1_scored,1):.4f}  "
          f"stage1+2 top1={stage2_hits/max(stage2_scored,1):.4f}  (same {stage1_scored} evaluable queries)")


if __name__ == "__main__":
    main()
