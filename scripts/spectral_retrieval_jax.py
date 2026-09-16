"""JAX spectral-retrieval prototype for the Kaggle CASMI26 molecule-ID
competition (enveda-CASMI26-molecule-id-mass-spectra).

Not a port of any existing PyTorch implementation -- a fresh JAX
reimplementation of the same core idea (Fourier-feature m/z encoding +
attention pooling -> predicted molecular fingerprint, trained via InfoNCE
against real candidate fingerprints), inspired by the architecture pattern
in Jia et al. 2026's MSAlign (arXiv:2605.19752, indexed in quantumrag) and
a public reference PyTorch implementation found on GitHub
(abrilrisso/MassSpecGym-Molecular-Identification-from-MS-MS-Spectra) --
neither of which is JAX-based or uses any chemistry-informed feature beyond
raw peaks.

This step verifies the architecture and loss are CORRECT on tiny synthetic
data (no real spectra yet) before touching the real competition dataset --
same "verify on a toy case first" discipline as every other Discovery
script. The real-data validation runs on Kaggle (data already mounted
there), never downloaded locally (a 3GB local download of this
competition's train.parquet caused a real PC freeze earlier this session).
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax


def fourier_features(x, n_freqs, sigma, key):
    """Random Fourier features for a scalar continuous input (e.g. m/z).
    x: (..., 1). Returns (..., 2*n_freqs)."""
    b = jax.random.normal(key, (n_freqs,)) * sigma
    projected = 2 * jnp.pi * x * b
    return jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)


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
    """mzs/intensities: (n_peaks,). precursor_mz: scalar. Returns (fp_size,)
    predicted molecular fingerprint (unnormalized logits)."""
    mz_col = mzs[:, None] / 1000.0
    projected = 2 * jnp.pi * mz_col * params["fourier_b"]
    mz_emb = jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)

    log_intensity = jnp.log1p(intensities)[:, None]
    int_emb = log_intensity @ params["int_w"] + params["int_b"]

    peak_embs = jnp.concatenate([mz_emb, int_emb], axis=-1)  # (n_peaks, d_model)

    prec_feat = (jnp.array([[precursor_mz / 1000.0]]) @ params["precursor_w"] + params["precursor_b"])
    x = peak_embs + prec_feat  # broadcast over peaks
    x = layer_norm(x, params["ln1_gain"], params["ln1_bias"])

    q = x @ params["wq"]
    k = x @ params["wk"]
    v = x @ params["wv"]
    d_model = x.shape[-1]
    attn_logits = (q @ k.T) / jnp.sqrt(d_model)
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    x_out = x + (attn_weights @ v)  # residual connection
    x_out = layer_norm(x_out, params["ln2_gain"], params["ln2_bias"])

    pool_logits = x_out @ params["attn_pool_w"]  # (n_peaks, 1)
    pool_weights = jax.nn.softmax(pool_logits, axis=0)
    global_repr = jnp.sum(x_out * pool_weights, axis=0)  # (d_model,)

    return global_repr @ params["head_w"] + params["head_b"]


def info_nce_loss(params, batch, n_freqs, temp=0.1):
    """batch: dict with 'mzs', 'intensities', 'precursor_mz' (each a list of
    per-sample arrays, variable n_peaks -- vmap can't ragged-batch this
    directly, so this loop-over-batch version is the correctness reference;
    a fixed-n_peaks-padded version is what a real training loop would vmap.)
    'candidate_fps': (batch, fp_size) the TRUE fingerprint for each query
    (one positive per query here, for the toy check)."""
    preds = jnp.stack([
        encode_spectrum(params, batch["mzs"][i], batch["intensities"][i],
                         batch["precursor_mz"][i], n_freqs)
        for i in range(len(batch["mzs"]))
    ])
    preds = preds / jnp.linalg.norm(preds, axis=-1, keepdims=True)
    targets = batch["candidate_fps"] / jnp.linalg.norm(batch["candidate_fps"], axis=-1, keepdims=True)

    logits = (preds @ targets.T) / temp  # (batch, batch) -- in-batch negatives
    labels = jnp.arange(len(batch["mzs"]))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(len(labels)), labels])


def _toy_batch(key, n_samples, n_peaks, fp_size):
    keys = jax.random.split(key, 4)
    mzs = [jax.random.uniform(keys[0], (n_peaks,), minval=50.0, maxval=500.0) for _ in range(n_samples)]
    intensities = [jax.random.uniform(keys[1], (n_peaks,), minval=0.0, maxval=1.0) for _ in range(n_samples)]
    precursor_mz = jax.random.uniform(keys[2], (n_samples,), minval=100.0, maxval=600.0)
    candidate_fps = (jax.random.uniform(keys[3], (n_samples, fp_size)) > 0.9).astype(jnp.float32)
    return {"mzs": mzs, "intensities": intensities, "precursor_mz": precursor_mz, "candidate_fps": candidate_fps}


if __name__ == "__main__":
    key = jax.random.PRNGKey(2026)
    d_model, n_freqs, fp_size = 64, 16, 256
    params = init_params(key, d_model, n_freqs, fp_size)

    batch = _toy_batch(jax.random.PRNGKey(1), n_samples=8, n_peaks=20, fp_size=fp_size)

    print("Step 0: verify forward pass shape and no NaN")
    fp0 = encode_spectrum(params, batch["mzs"][0], batch["intensities"][0], batch["precursor_mz"][0], n_freqs)
    print(f"  predicted fingerprint shape: {fp0.shape} (expect ({fp_size},))")
    print(f"  any NaN: {bool(jnp.any(jnp.isnan(fp0)))}")

    print("\nStep 1: verify InfoNCE loss is finite and gradients flow")
    loss_fn = lambda p: info_nce_loss(p, batch, n_freqs)
    loss0, grads = jax.value_and_grad(loss_fn)(params)
    grad_norms = {k: float(jnp.linalg.norm(v)) for k, v in grads.items()}
    print(f"  initial loss: {float(loss0):.4f} (expect ~log({len(batch['mzs'])})={jnp.log(len(batch['mzs'])):.4f} at init)")
    print(f"  any zero-norm gradient (dead param): {[k for k, v in grad_norms.items() if v == 0.0]}")

    print("\nStep 2: verify Adam optimization actually reduces the loss (toy overfit check)")
    # Plain SGD (even tuned across several lr values) plateaued around
    # ~90% of the initial loss on this toy batch -- not a broken gradient
    # (Step 1 already confirmed no dead params), just a poor optimizer
    # choice for this architecture. Every real reference (including the
    # PyTorch baseline this is inspired by) trains transformers with
    # Adam/AdamW, never plain SGD -- switched to match, not tuned around
    # SGD's own weakness.
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(params)
    p = params
    losses = []
    for step in range(100):
        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        losses.append(float(loss_val))
    print(f"  loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f} (expect a clear decrease)")
    assert losses[-1] < losses[0] * 0.5, "loss did not decrease meaningfully -- architecture/loss bug"
    print("  PASSED: loss decreased on the toy overfit check")
