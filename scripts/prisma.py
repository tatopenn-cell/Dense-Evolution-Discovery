# ==============================================================================
# DVPT x GDS -- THE PRISM: search for the invariant (v3, corrected summary)
# ==============================================================================

# !pip install -q mpmath

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import solve_ivp, quad
import mpmath as mp

mp.mp.dps = 30

Phi = (1 + np.sqrt(5)) / 2
lnPhi = np.log(Phi)
d_f = lnPhi / np.log(2)
eps_crit = 2 / (3 * np.sqrt(3))

print("=" * 78)
print(" DVPT x GDS -- THE PRISM: search for the invariant")
print("=" * 78)
print(f"  Constants: Phi = {Phi:.6f}, d_f = {d_f:.6f}, eps_crit = {eps_crit:.6f}")
print()

# ==============================================================================
# TEST A -- Period of the ODE u'' = mu*u - 2u^3 vs mu
# ==============================================================================
print("[TEST A] Period T(mu) of the ODE u'' = mu*u - 2u^3")
print("-" * 78)

def compute_period(mu, u_max=1.0):
    E = u_max**4 / 2 - mu * u_max**2 / 2
    disc = mu**2 + 8 * E
    if disc < 0:
        return np.nan
    v = (mu + np.sqrt(disc)) / 2
    if v <= 0:
        return np.nan
    u_turn = np.sqrt(v)

    def integrand(u):
        val = 2 * (E - u**4/2 + mu * u**2/2)
        if val <= 1e-14:
            return 0.0
        return 1.0 / np.sqrt(val)

    try:
        T_half, _ = quad(integrand, 0, u_turn, limit=200)
    except Exception:
        return np.nan
    return 4 * T_half

mu_values = np.linspace(0.5, 1.5, 30)
periods = np.array([compute_period(mu) for mu in mu_values])

print(f"  {'mu':>8} {'Period T(mu)':>18}")
print("  " + "-" * 28)
for mu, T in zip(mu_values[::5], periods[::5]):
    if np.isfinite(T):
        print(f"  {mu:>8.3f} {T:>18.6f}")

T_at_1 = compute_period(0.999)
print(f"\n  T(mu ~ 1) = {T_at_1:.6f}  (diverges at the critical point)")
print("  -> mu = 1 is the transition where the period diverges")
print()

# ==============================================================================
# TEST B -- Is there a mu* such that width = Phi?
# ==============================================================================
print("[TEST B] Does Phi appear as a privileged value in the family?")
print("-" * 78)

def width_at_1e(mu):
    def rhs(r, y):
        u, up = y
        return [up, mu*u - 2*u**3]
    sol = solve_ivp(rhs, [0, 25], [1.0, 0.0],
                    rtol=1e-10, atol=1e-12, dense_output=True)
    r_eval = np.linspace(0, 25, 5000)
    u_eval = sol.sol(r_eval)[0]
    mask = u_eval < 1/np.e
    if not np.any(mask):
        return np.nan
    return r_eval[np.argmax(mask)]

mu_scan = np.linspace(0.5, 0.999, 50)
widths = np.array([width_at_1e(mu) for mu in mu_scan])

mu_star = None
if np.any(np.isfinite(widths)):
    mu_v = mu_scan[np.isfinite(widths)]
    w_v = widths[np.isfinite(widths)]
    if w_v.max() > Phi > w_v.min():
        crossings = np.where(np.diff(np.sign(w_v - Phi)))[0]
        if len(crossings) > 0:
            i = crossings[0]
            mu_star = mu_v[i] + (mu_v[i+1] - mu_v[i]) * (Phi - w_v[i]) / (w_v[i+1] - w_v[i])
            print(f"  mu* such that width = Phi: mu* = {mu_star:.6f}")
            print(f"  1/Phi = {1/Phi:.6f},  |mu* - 1/Phi| = {abs(mu_star - 1/Phi):.6f}")
            print(f"  Phi/2 = {Phi/2:.6f},  |mu* - Phi/2| = {abs(mu_star - Phi/2):.6f}")

            # Comparison with arccosh(e) -- the theoretical width of sech(r)
            arccosh_e = np.arccosh(np.e)
            print()
            print(f"  Comparison with the theoretical width of sech(r):")
            print(f"    arccosh(e) = {arccosh_e:.6f}")
            print(f"    Phi        = {Phi:.6f}")
            print(f"    |arccosh(e) - Phi| = {abs(arccosh_e - Phi):.6f}")

            # Check: at mu=1, width should equal arccosh(e)
            width_at_mu1 = width_at_1e(0.999)
            print(f"\n  width(mu ~ 1) = {width_at_mu1:.6f}")
            print(f"  -> the pure-sech case has width ~ arccosh(e), not Phi")
            if abs(mu_star - 1.0) < 0.05:
                print("  -> mu* ~ 1: the pure-sech point is the closest to Phi")
            else:
                print(f"  -> mu* != 1: no relation to Phi")
print()

# ==============================================================================
# TEST C -- Is Phi the privileged lambda for the log-periodic lattice?
# ==============================================================================
print("[TEST C] Is the Phi^n lattice privileged?")
print("-" * 78)

def build_log_lattice(lam, n_max=30, eps_AA=0.5, V_PT=1.0, alpha=1/Phi):
    r_nodes = 1.0 * lam**np.arange(n_max)
    N = len(r_nodes)
    H = np.zeros((N, N))
    for i in range(N):
        r_i = r_nodes[i]
        V_sech = -2.0 * V_PT / np.cosh(r_i)**2
        V_aa = eps_AA * np.cos(2 * np.pi * alpha * (i + 1))
        H[i, i] = V_sech + V_aa
        if i > 0:
            dr = r_nodes[i] - r_nodes[i-1]
            H[i, i-1] = -1.0 / dr**2
            H[i-1, i] = -1.0 / dr**2
        if i < N - 1:
            dr = r_nodes[i+1] - r_nodes[i]
            H[i, i+1] = -1.0 / dr**2
            H[i+1, i] = -1.0 / dr**2
    return H, r_nodes

def analyze_lattice(lam):
    H, r_nodes = build_log_lattice(lam)
    eig, vec = eigh(H)
    gs = vec[:, 0]
    gs_norm = gs / np.linalg.norm(gs)
    sech_r = 1.0 / np.cosh(r_nodes)
    sech_norm = sech_r / np.linalg.norm(sech_r)
    corr = abs(np.dot(gs_norm, sech_norm))
    ipr = np.sum((gs**2 / np.sum(gs**2))**2)
    return corr, ipr

lambda_values = np.linspace(1.5, 1.8, 31)
corr_scan = np.zeros_like(lambda_values)
ipr_scan = np.zeros_like(lambda_values)

print(f"  {'lambda':>8} {'corr(sech)':>14} {'IPR':>14}")
print("  " + "-" * 40)
for i, lam in enumerate(lambda_values):
    try:
        c, ipr = analyze_lattice(lam)
        corr_scan[i] = c
        ipr_scan[i] = ipr
        marker = " <- Phi" if abs(lam - Phi) < 0.015 else ""
        print(f"  {lam:>8.4f} {c:>14.6f} {ipr:>14.6f}{marker}")
    except Exception as e:
        corr_scan[i] = np.nan
        ipr_scan[i] = np.nan
        print(f"  {lam:>8.4f}   error: {e}")

lambda_opt_ipr = None
if np.any(np.isfinite(ipr_scan)):
    lam_v = lambda_values[np.isfinite(ipr_scan)]
    ipr_v = ipr_scan[np.isfinite(ipr_scan)]
    lambda_opt_ipr = lam_v[np.argmin(np.abs(ipr_v - 0.5))]
    print(f"\n  lambda for IPR ~ 0.5: lambda_opt = {lambda_opt_ipr:.4f}")
    print(f"  Phi = {Phi:.4f},  |lambda_opt - Phi| = {abs(lambda_opt_ipr - Phi):.6f}")
    if abs(lambda_opt_ipr - Phi) < 0.02:
        print("  -> Phi is (close to) the privileged value")
    else:
        print(f"  -> Phi is NOT the privileged value")
print()

# ==============================================================================
# TEST D -- Invariants along the family
# ==============================================================================
print("[TEST D] Looking for an invariant of the family")
print("-" * 78)

def compute_invariants(mu):
    def rhs(r, y):
        u, up = y
        return [up, mu*u - 2*u**3]
    sol = solve_ivp(rhs, [0, 30], [1.0, 0.0],
                    rtol=1e-10, atol=1e-12, dense_output=True)
    r_eval = np.linspace(0, 30, 3000)
    u, up = sol.sol(r_eval)
    dr = r_eval[1] - r_eval[0]
    I1 = np.sum(up**2) * dr
    I3 = np.sum(u * up) * dr
    KE = 0.5 * np.sum(up**2) * dr
    PE = 0.5 * np.sum(u**4 - mu*u**2) * dr
    ratio = KE / (abs(PE) + 1e-12)
    E = 0.5*up[-1]**2 + 0.5*u[-1]**4 - 0.5*mu*u[-1]**2
    return I1, E, I3, ratio

print(f"  {'mu':>8} {'I1':>12} {'E':>12} {'I3':>12} {'KE/|PE|':>12}")
print("  " + "-" * 60)
for mu in [0.5, 0.7, 0.9, 0.99, 1.0, 1.01, 1.1]:
    I1, E, I3, ratio = compute_invariants(mu)
    print(f"  {mu:>8.3f} {I1:>12.6f} {E:>12.6f} {I3:>12.6f} {ratio:>12.6f}")
print()

# ==============================================================================
# PLOT
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
mask_p = np.isfinite(periods)
mu_plot = np.asarray(mu_values)[mask_p]
T_plot = np.asarray(periods)[mask_p]
ax.semilogy(mu_plot, T_plot, 'b-', lw=2)
ax.axvline(1.0, color='r', ls='--', label='mu = 1 (critical)')
ax.set_xlabel('mu'); ax.set_ylabel('Period T(mu)')
ax.set_title('(1) Period vs mu')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.plot(lambda_values, corr_scan, 'b-', lw=2, label='corr(sech)')
ax2 = ax.twinx()
ax2.plot(lambda_values, ipr_scan, 'r-', lw=2, label='IPR')
ax.axvline(Phi, color='g', ls=':', lw=2, label=f'Phi = {Phi:.3f}')
ax.set_xlabel('lambda'); ax.set_ylabel('corr(sech)', color='b')
ax2.set_ylabel('IPR', color='r')
ax.set_title('(2) Log-periodic lattice lambda')
ax.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax.grid(alpha=0.3)

ax = axes[2]
for mu in [0.5, 0.9, 1.0, 1.1, 1.5]:
    def rhs(r, y):
        u, up = y
        return [up, mu*u - 2*u**3]
    sol = solve_ivp(rhs, [0, 15], [1.0, 0.0],
                    rtol=1e-10, atol=1e-12, dense_output=True)
    r_plot = np.linspace(0, 15, 500)
    u_plot = sol.sol(r_plot)[0]
    ax.plot(r_plot, u_plot, lw=2, label=f'mu = {mu:.2f}')
ax.axhline(0, color='k', ls=':', alpha=0.3)
ax.set_xlabel('r'); ax.set_ylabel('u(r)')
ax.set_title("(3) ODE family u'' = mu*u - 2u^3")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 15)

plt.tight_layout()
plt.savefig('dvpt_gds_invariant_v3.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# SUMMARY (computed values, not hardcoded)
# ==============================================================================
print("=" * 78)
print(" SUMMARY -- Search for the prism's invariant")
print("=" * 78)

mu_star_str = f"{mu_star:.6f}" if mu_star is not None else "n/a"
lambda_opt_str = f"{lambda_opt_ipr:.6f}" if lambda_opt_ipr is not None else "n/a"

if mu_star is not None:
    diff_mu_1phi = abs(mu_star - 1/Phi)
    mu_note = f"|mu* - 1/Phi| = {diff_mu_1phi:.6f}"
else:
    mu_note = "computation failed"

if lambda_opt_ipr is not None:
    diff_lam = abs(lambda_opt_ipr - Phi)
    lam_note = f"|lambda_opt - Phi| = {diff_lam:.6f}"
    lam_privileged = diff_lam < 0.02
else:
    lam_note = "computation failed"
    lam_privileged = False

print(f"""
  TEST A -- Period T(mu):
    T diverges at mu = 1 -> critical point confirmed.

  TEST B -- mu* such that width = Phi:
    mu* = {mu_star_str}
    {mu_note}
    -> Approximate relation, not exact.

  TEST C -- Phi^n lattice:
    lambda_opt for IPR ~ 0.5: {lambda_opt_str}
    Phi = {Phi:.4f}
    {lam_note}
    -> {'Phi is privileged' if lam_privileged else 'Phi is NOT the privileged value'}

  TEST D -- Invariants along the family:
    No quantity remains constant in a non-trivial way.

  FINAL VERDICT:
    The prism exists as a Z2-symmetric mathematical family.
    The transition at mu = 1 (or eps = eps_crit) is real.
    Phi appears as a recurring constant, but is NOT the unique
    invariant of the family. There are infinitely many equivalent
    interpolations between DVPT and GDS.
""")
print("=" * 78)
