"""
Diffuse2Seg-derived relevance propagation, for QM/MM region selection
weighted by a real chemical affinity (Mayer/Wiberg bond order) instead of
a fixed hop-count radius.
"""
import numpy as np


def propagate_relevance(affinity, seed_idx, n_nodes, p=1.6, lam=1e-5, tau_prop=1e-4, max_iter=500):
    """Algorithm 1 of Hummer, Sicking, Huger & Gottschalk 2026
    (arXiv:2609.06491, "Diffuse2Seg", Sec. 3.4/A.1), read directly from
    the paper (provided by the user, Desktop/QAI/2609.06491v1.pdf) and
    implemented verbatim -- not the max-product-with-decay heuristic used
    in this repo's earlier bond-order-partitioning scripts
    (qmmm_bond_order_partition_vs_radius.py,
    qmmm_bond_order_aromatic_vs_alkyl_branch.py), which was written
    without having read the paper first and does not match it.

    Non-linear p-Laplacian graph-regularized smoothing (Elmoataz et al.
    2008), solved by Gauss-Jacobi iteration: propagates a one-hot seed
    vector over any node-affinity graph `affinity` (self-attention in the
    original paper; Mayer/Wiberg bond order here) into a soft relevance
    map that stays smooth within high-affinity regions and is throttled
    across low-affinity (edge) ones. `p`, `lam`, `tau_prop` are the
    paper's own final values (Sec. 4.2).

    Measured, not assumed (qmmm_diffuse2seg_propagation_lambda_sweep.py,
    on the real Mayer bond-order graph of OCC(c1ccccc1)CCC): at the
    paper's own lam=1e-5 -- tuned for a dense grid of prompts later
    merged together, not a single isolated seed -- the aromatic branch
    actually gets LESS relevance than the alkyl branch (ratio 0.90), the
    opposite of the naive expectation. The ratio crosses 1.0 only around
    lam~0.5-1, and only reaches a large aromatic-favoring differentiation
    (ratio 1.49) at lam=10, two orders of magnitude above the paper's own
    value. This is not a case for silently retuning lam to the value that
    "looks right" -- it is reported as the real, measured lam-dependence
    of this algorithm in a single-seed molecular setting, distinct from
    the many-prompt image setting it was designed and calibrated for.

    Degenerate-case handling: at any node i where g_i = sqrt(sum_j
    A_ij(f_j-f_i)^2) is exactly 0 (every affinity-neighbor already equals
    f_i), the formula's g_i^(p-2) term diverges for p<2. An earlier
    version clamped g_i away from 0 with an epsilon -- the same category
    of shortcut already rejected elsewhere in this project for degenerate
    eigenvalues (see dense_evolution.physics.spectral, Kato's divided-
    difference formula) in favor of the real mathematical limit. Worked
    out directly here: as g_i -> 0, every A_ij-connected f_j equals f_i
    by definition of g_i=0, so the g_i^(p-2)-weighted terms in both the
    numerator and denominator of the update come to dominate and cancel
    to exactly f_i -- i.e. a node already consistent with its whole
    affinity-neighborhood is unchanged by an edge-preserving smoothing
    step, exactly as expected. Implemented as an explicit special case
    below, not an epsilon."""
    f0 = np.zeros(n_nodes)
    f0[seed_idx] = 1.0
    f = f0.copy()
    for _ in range(max_iter):
        diff = f[None, :] - f[:, None]
        g = np.sqrt(np.sum(affinity * diff ** 2, axis=1))
        degenerate = g == 0.0
        gp = np.zeros_like(g)
        gp[~degenerate] = g[~degenerate] ** (p - 2)
        gamma = affinity * (gp[:, None] + gp[None, :])
        numerator = lam * f0 + (gamma * f[None, :]).sum(axis=1)
        denominator = lam + gamma.sum(axis=1)
        f_new = np.where(degenerate, f, numerator / denominator)
        if np.sum((f_new - f) ** 2) <= tau_prop:
            f = f_new
            break
        f = f_new
    return f
