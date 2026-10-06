"""
Approach B: QR-based MPS truncation (arXiv:2212.09782, Unfried/Hauschild/
Pollmann, "Fast Time-Evolution of Matrix-Product States using the QR
decomposition"). Their own motivation is the one this project already
confirmed independently in mps_gpu_optimization.md: SVD truncation runs
disproportionately slower on GPU than CPU. Their fix replaces the single
d^3*chi^3-scaling SVD of the full two-site block with two QR/LQ
decompositions (d^2*chi^3 scaling) plus one small, d-independent
chi x chi eigendecomposition for the actual truncation step -- only
that last, small step needs singular values at all.

IMPORTANT CAVEAT, taken directly from the paper and not glossed over:
a single QR/LQ projection pass is only an approximation of the true SVD
truncation in general -- exact only in the limit of infinitesimal/near-
identity gates (U = 1 + O(dt)), which is exactly the Trotter-gate regime
this project's own TFIM benchmark circuit uses, but NOT true for an
arbitrary (e.g. Haar-random, non-infinitesimal) unitary. The test below
checks BOTH regimes honestly: a real small-dt TFIM two-qubit gate (where
one iteration should already be accurate) and a Haar-random gate (where
one iteration is expected to show a real, larger residual error) -- and
verifies iterating the QR/LQ projection converges toward the exact SVD
answer in the random case, confirming the mechanism is implemented
correctly rather than accidentally close by luck.

One QR/LQ iteration, given: A[m] (chi_l,d,chi_m), Lambda (chi_m,),
B[n] (chi_m,d,chi_r) [right-isometric], a 2-qubit gate U, and an enlarged
trial bond eta >= chi_m (this project's own existing provable bound,
min(chi_l*d, chi_r*d), already used for bucket selection):

  0. Theta[a,i,j,b] = Lambda[a] * A[m][a,i,m] * B[n][m,j,b] * U[i,j,i',j']
     -- the standard two-site evolved block (a,b = outer bond legs).
  (i) Bootstrap a valid eta-dim right isometry B0[n] living in the (j,b)
      leg space (any full-rank slice/pad works as a starting guess).
  (ii) X[a,i,mu] = sum_{j,b} Theta[a,i,j,b] * conj(B0)[mu,j,b] -- project
       Theta onto B0. QR-decompose X reshaped to (a*i, mu) -> Q[m], R.
  (iii) Y[nu,j,b] = sum_{a,i} conj(Q[m])[a,i,nu] * Theta[a,i,j,b] --
       project Theta onto the new left isometry Q[m]. LQ-decompose Y
       reshaped to (nu, j*b) -> L (small, nu x nu2), Q[n] (nu2,j,b).
  (iv) Update B0 <- Q[n] and repeat (ii)-(iii) for more iterations (each
       one refines the isometry pair; converges to the exact SVD answer).
  (v) Diagonalize the small Hermitian L^dagger L = V S^2 V^dagger
      (d-independent) -- S are the real Schmidt values. Truncate to the
      top chi' <= nu2, giving Lambda_new=S[:chi'],
      B[n]_new=(V^dagger Q[n])[:chi'],
      A[m]_new = einsum(Q[m], R, V)[..., :chi'].
"""
import numpy as np

rng = np.random.default_rng(0)
d = 2


def random_isometry(rows, cols, rng):
    m = rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(m)
    return q[:, :cols]


def random_mps_pair(chi_l, chi_r, chi_m, rng):
    a_left = random_isometry(chi_l * d, chi_m, rng).reshape(chi_l, d, chi_m)
    lam = np.sort(rng.uniform(0.3, 1.0, size=chi_m))[::-1]
    lam = lam / np.linalg.norm(lam)
    q, _ = np.linalg.qr(rng.normal(size=(chi_m * d, chi_r)) + 1j * rng.normal(size=(chi_m * d, chi_r)))
    b_right = q[:, :chi_r].reshape(chi_m, d, chi_r)
    return a_left, lam, b_right


def random_2q_unitary(rng):
    m = rng.normal(size=(d * d, d * d)) + 1j * rng.normal(size=(d * d, d * d))
    q, _ = np.linalg.qr(m)
    return q.reshape(d, d, d, d)


def small_dt_tfim_gate(dt, J, g):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    i2 = np.eye(2, dtype=complex)
    zz = np.kron(z, z)
    xi = np.kron(x, i2) + np.kron(i2, x)
    h = J * zz + g * xi
    evals, evecs = np.linalg.eigh(h)
    u = evecs @ np.diag(np.exp(-1j * dt * evals)) @ evecs.conj().T
    return u.reshape(d, d, d, d)


def build_theta(a_left, lam, b_right, gate):
    theta0 = np.einsum("aim,m,mjb->aijb", a_left, lam, b_right, optimize=True)
    theta = np.einsum("aijb,ijkl->aklb", theta0, gate, optimize=True)
    return theta


def svd_reference(theta, chi_max):
    chi_l, d1, d2, chi_r = theta.shape
    mat = theta.reshape(chi_l * d1, d2 * chi_r)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    chi_new = min(chi_max, len(s))
    u, s, vh = u[:, :chi_new], s[:chi_new], vh[:chi_new, :]
    a_new = u.reshape(chi_l, d1, chi_new)
    b_new = vh.reshape(chi_new, d2, chi_r)
    return a_new, s, b_new


def _lq(mat, k):
    q, r = np.linalg.qr(mat.conj().T)
    return r[:k, :].conj().T, q[:, :k].conj().T


def qr_truncate(theta, eta, chi_max, n_iter, rng, b_warm_start=None):
    chi_l, d1, d2, chi_r = theta.shape

    if b_warm_start is not None:
        chi_old = b_warm_start.shape[0]
        b_mat = b_warm_start.reshape(chi_old, d2 * chi_r).conj().T
        extra = eta - chi_old
        if extra > 0:
            pad = rng.normal(size=(d2 * chi_r, extra)) + 1j * rng.normal(size=(d2 * chi_r, extra))
            proj = np.eye(d2 * chi_r) - b_mat @ b_mat.conj().T
            pad = proj @ pad
            y0 = np.concatenate([b_mat, pad], axis=1)
        else:
            y0 = b_mat[:, :eta]
    else:
        y0 = rng.normal(size=(d2 * chi_r, eta)) + 1j * rng.normal(size=(d2 * chi_r, eta))
    q0, _ = np.linalg.qr(y0)
    eta_eff = q0.shape[1]
    b_cur = q0.conj().T.reshape(eta_eff, d2, chi_r)

    for _ in range(n_iter):
        x = np.einsum("aijb,mjb->aim", theta, b_cur.conj(), optimize=True)
        x_mat = x.reshape(chi_l * d1, eta_eff)
        nu = min(x_mat.shape)
        q_m, r_mat = np.linalg.qr(x_mat)
        q_left = q_m[:, :nu].reshape(chi_l, d1, nu)
        r_mat = r_mat[:nu, :]

        y = np.einsum("ain,aijb->njb", q_left.conj(), theta, optimize=True)
        y_mat = y.reshape(nu, d2 * chi_r)
        nu2 = min(y_mat.shape)
        l_mat, q_n = _lq(y_mat, nu2)
        b_cur = q_n.reshape(nu2, d2, chi_r)
        eta_eff = nu2

    # Theta = Q_left @ L @ Q_n exactly (Q_left is left-isometric, this
    # project's own square/full-rank case even makes it exactly unitary,
    # so Q_left @ Q_left^dagger @ Theta = Theta). L's own (small,
    # d-independent) SVD gives the truncation directly: L = P @ S @ Vh.
    p_mat, s_full, vh_mat = np.linalg.svd(l_mat, full_matrices=False)

    chi_new = min(chi_max, nu2, int(np.sum(s_full > 1e-12)))
    chi_new = max(chi_new, 1)
    p_trunc, s_trunc, vh_trunc = p_mat[:, :chi_new], s_full[:chi_new], vh_mat[:chi_new, :]

    b_new = np.einsum("kn,njb->kjb", vh_trunc, b_cur, optimize=True)
    a_new = np.einsum("ain,nk->aik", q_left, p_trunc, optimize=True)
    return a_new, s_trunc, b_new


def reconstruct_theta(a_new, s_new, b_new):
    return np.einsum("aim,m,mjb->aijb", a_new, s_new, b_new, optimize=True)


def run_case(name, gate, n_iters):
    chi_l, chi_m, chi_r = 3, 4, 3
    a_left, lam, b_right = random_mps_pair(chi_l, chi_r, chi_m, rng)
    theta = build_theta(a_left, lam, b_right, gate)

    a_ref, s_ref, b_ref = svd_reference(theta, chi_max=100)
    theta_ref = reconstruct_theta(a_ref, s_ref, b_ref)
    ref_norm = np.linalg.norm(theta_ref)

    eta = min(chi_l * d, chi_r * d)
    print(f"\n--- {name} (reference rank {len(s_ref)}, eta={eta}) ---")
    for n_iter in n_iters:
        a_qr, s_qr, b_qr = qr_truncate(theta, eta=eta, chi_max=100, n_iter=n_iter, rng=np.random.default_rng(1))
        theta_qr = reconstruct_theta(a_qr, s_qr, b_qr)
        err = np.linalg.norm(theta_ref - theta_qr) / ref_norm
        print(f"  iterations={n_iter}: rank={len(s_qr)}, relative error={err:.3e}")


def run_truncating_case(name, gate, chi_max, n_iters, warm_start=False, mps_seed=42):
    chi_l, chi_m, chi_r = 6, 8, 6
    a_left, lam, b_right = random_mps_pair(chi_l, chi_r, chi_m, np.random.default_rng(mps_seed))
    theta = build_theta(a_left, lam, b_right, gate)

    a_ref, s_ref, b_ref = svd_reference(theta, chi_max=chi_max)
    theta_ref = reconstruct_theta(a_ref, s_ref, b_ref)
    ref_norm = np.linalg.norm(theta_ref)
    full_rank = min(theta.reshape(chi_l * d, d * chi_r).shape)

    tag = "warm-start (old B[n])" if warm_start else "random bootstrap"
    print(f"\n--- {name}, TRUNCATING chi_max={chi_max} (full rank {full_rank}), {tag} ---")
    for n_iter in n_iters:
        b_init = b_right if warm_start else None
        a_qr, s_qr, b_qr = qr_truncate(theta, eta=chi_max, chi_max=chi_max, n_iter=n_iter,
                                        rng=np.random.default_rng(1), b_warm_start=b_init)
        theta_qr = reconstruct_theta(a_qr, s_qr, b_qr)
        err = np.linalg.norm(theta_ref - theta_qr) / ref_norm
        print(f"  iterations={n_iter}: rank={len(s_qr)}, relative error vs SVD-truncated reference={err:.3e}")


if __name__ == "__main__":
    run_case("small-dt TFIM gate (dt=0.05)", small_dt_tfim_gate(0.05, 1.0, 1.0), n_iters=[1, 2, 5])
    run_case("small-dt TFIM gate (dt=0.1)", small_dt_tfim_gate(0.1, 1.0, 1.0), n_iters=[1, 2, 5])
    run_case("Haar-random 2q unitary (adversarial)", random_2q_unitary(rng), n_iters=[1, 2, 5, 15])

    run_truncating_case("small-dt TFIM gate (dt=0.1)", small_dt_tfim_gate(0.1, 1.0, 1.0), chi_max=6, n_iters=[1, 2, 5])
    run_truncating_case("Haar-random 2q unitary (adversarial)", random_2q_unitary(rng), chi_max=6, n_iters=[1, 2, 5, 15])
    run_truncating_case("small-dt TFIM gate (dt=0.1)", small_dt_tfim_gate(0.1, 1.0, 1.0), chi_max=6, n_iters=[1, 2, 5], warm_start=True)
    run_truncating_case("small-dt TFIM gate (dt=0.01)", small_dt_tfim_gate(0.01, 1.0, 1.0), chi_max=6, n_iters=[1, 2, 5], warm_start=True)
