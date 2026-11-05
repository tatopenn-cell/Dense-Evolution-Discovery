# ============================================================
# FULL TEST — Unified formula RG + QM
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import lambertw
from scipy.optimize import brentq
from mpmath import mp, mpf, zeta, diff as mpdiff, log, pi as mpi

mp.dps = 40
line = "=" * 74

# ---------- SCHRÖDINGER OPERATOR ----------
N = 600
r_max = 20.0
r = np.linspace(-r_max, r_max, N)
dr = r[1] - r[0]
PHI = (1 + np.sqrt(5)) / 2
alpha_AA = 1.0 / PHI

def H_matrix(mu, eps):
    main = 2.0/dr**2 * np.ones(N)
    off  = -1.0/dr**2 * np.ones(N-1)
    H = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    H += np.diag(eps*np.cos(2*np.pi*alpha_AA*r) - 2*mu/np.cosh(r)**2)
    return H

# ---------- GENTILE GAS ----------
def E_gas(beta, k):
    beta = mpf(beta)
    if k == np.inf:
        return float(-mpdiff(zeta, beta)/zeta(beta))
    return float(-mpdiff(zeta, beta)/zeta(beta)
                 + (k+1)*mpdiff(zeta, (k+1)*beta)/zeta((k+1)*beta))

def c_k(k):
    return 1.0 if k == np.inf else 1.0 - float(log(zeta(k+1)))

def S_k(beta, k):
    E = E_gas(beta, k)
    if E <= 0: return np.nan
    return E + np.log(E) + c_k(k)

# ---------- MAP Φ ----------
def alpha_LW(E, k):
    if E <= 0: return np.nan
    return lambertw(E * np.exp(E + c_k(k))).real / E

def alpha_brentq(E, k):
    ck = c_k(k)
    f = lambda a: a*E + np.log(a*E) - (E + np.log(E) + ck)
    try: return brentq(f, 1e-6, 200)
    except: return np.nan

# ---------- METRIC ----------
def f_k(r_val, beta, k, G=1.0, M=1.0):
    a = alpha_LW(E_gas(beta, k), k)
    ell2 = G / (np.pi * a)
    return 1 - 2*G*M*r_val / (r_val**2 + ell2)

def S_J(A_over_4G):
    return A_over_4G + np.log(A_over_4G)

# ============================================================
# TEST 1 — Schrödinger
# ============================================================
print(line); print(" TEST 1 — H(μ,ε): sech as ground state"); print(line)
print(f"\n  {'μ':>5} {'ε':>5} {'E₀':>14} {'IPR':>12} {'corr(sech)':>12}")
print("  " + "-" * 52)
for mu, eps in [(1.0, 0.0), (1.0, 0.5), (1.0, 2.0), (0.0, 2.0), (2.0, 2.0)]:
    E_arr, psi = eigh(H_matrix(mu, eps))
    psi0 = psi[:, 0] / np.sqrt(np.sum(psi[:, 0]**2)*dr)
    IPR = np.sum(psi0**4) * dr
    corr = np.abs(np.corrcoef(psi0, 1/np.cosh(r))[0, 1])
    print(f"  {mu:>5.1f} {eps:>5.1f} {E_arr[0]:>14.8f} {IPR:>12.6f} {corr:>12.6f}")
    if (mu, eps) == (1.0, 0.0):
        assert corr > 0.9999 and abs(E_arr[0] + 1.0) < 1e-3

# ============================================================
# TEST 2 — Gentile
# ============================================================
print(); print(line); print(" TEST 2 — Gentile gas"); print(line)
print(f"\n  {'k':>5} {'E(β=1.05)':>14} {'c_k':>14} {'S_k':>14}")
print("  " + "-" * 52)
for k in [1, 2, 5, 10, 100, np.inf]:
    E = E_gas(1.05, k); ck = c_k(k); Sk = S_k(1.05, k)
    assert E > 0 and np.isfinite(Sk)
    k_l = k if k != np.inf else "∞"
    print(f"  {k_l:>5} {E:>14.6f} {ck:>14.8f} {Sk:>14.6f}")

# ============================================================
# TEST 3 — Lambert W = brentq
# ============================================================
print(); print(line); print(" TEST 3 — α(k,β): Lambert W vs brentq"); print(line)
print(f"\n  {'k':>5} {'β':>6} {'α_LW':>16} {'α_brentq':>16} {'diff':>10}")
print("  " + "-" * 60)
for k in [1, 2, 5, 100]:
    for beta in [1.05, 1.5, 3.0]:
        E = E_gas(beta, k)
        aL = alpha_LW(E, k); aB = alpha_brentq(E, k)
        assert abs(aL - aB) < 1e-9
        print(f"  {k:>5} {beta:>6.2f} {aL:>16.10f} {aB:>16.10f} {abs(aL-aB):>10.2e}")

# ============================================================
# TEST 4 — Commutation
# ============================================================
print(); print(line); print(" TEST 4 — S_J(αE) = S_k(E)"); print(line)
print(f"\n  {'k':>5} {'β':>6} {'S_k':>18} {'S_J':>18} {'diff':>12}")
print("  " + "-" * 64)
for k in [1, 2, 5, 100, np.inf]:
    for beta in [1.05, 1.5, 2.5]:
        E = E_gas(beta, k); a = alpha_LW(E, k)
        assert abs(S_J(a*E) - S_k(beta, k)) < 1e-12
        k_l = k if k != np.inf else "∞"
        print(f"  {k_l:>5} {beta:>6.2f} {S_k(beta,k):>18.8f} {S_J(a*E):>18.8f} {S_J(a*E)-S_k(beta,k):>12.2e}")

# ============================================================
# TEST 5 — Metric f_k(r)
# ============================================================
print(); print(line); print(" TEST 5 — f_k(r) with α(k,β)"); print(line)
print(f"\n  {'r':>6} {'k=1':>14} {'k=2':>14} {'k=10':>14} {'k=∞':>14}")
print("  " + "-" * 66)
for r_val in [0.3, 0.5, 1.0, 2.0, 5.0]:
    row = f"  {r_val:>6.2f}"
    for k in [1, 2, 10, np.inf]:
        row += f" {f_k(r_val, 50.0, k):>14.8f}"
    print(row)

# ============================================================
# TEST 6 — α_∞(k) = e/ζ(k+1)
# ============================================================
print(); print(line); print(" TEST 6 — α_∞(k) = e/ζ(k+1)"); print(line)
print(f"\n  {'k':>5} {'α(β=50)':>20} {'e/ζ(k+1)':>20} {'diff':>12}")
print("  " + "-" * 62)
for k in [1, 2, 3, 5, 10, 20, 50, 100]:
    a50 = alpha_LW(E_gas(50, k), k)
    a_inf = np.exp(1) / float(zeta(k+1))
    assert abs(a50 - a_inf) < 1e-12
    print(f"  {k:>5} {a50:>20.14f} {a_inf:>20.14f} {abs(a50-a_inf):>12.2e}")

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
for mu, eps, col in [(1.0, 0.0, 'blue'), (1.0, 2.0, 'orange'), (0.0, 2.0, 'red')]:
    E_arr, psi = eigh(H_matrix(mu, eps))
    psi0 = psi[:, 0] / np.sqrt(np.sum(psi[:, 0]**2)*dr)
    ax.plot(r, psi0, lw=2, color=col, label=f'μ={mu}, ε={eps}')
ax.plot(r, 1/np.cosh(r), 'k--', lw=2, label='sech(r)')
ax.set_xlim(-6, 6); ax.set_xlabel('r'); ax.set_ylabel('ψ₀')
ax.set_title('(1) Schrödinger ground state'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
betas = np.linspace(1.05, 5, 60)
for k, col in zip([1, 2, 5, 100], ['blue', 'green', 'orange', 'red']):
    ax.plot(betas, [E_gas(b, k) for b in betas], lw=2, color=col, label=f'k={k}')
ax.set_xlabel('β'); ax.set_ylabel('E(β)')
ax.set_title('(2) Gentile gas energy'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
for k, col in zip([1, 2, 5, 100], ['blue', 'green', 'orange', 'red']):
    b_plot = np.linspace(1.05, 50, 60)
    ax.plot(b_plot, [alpha_LW(E_gas(b, k), k) for b in b_plot], lw=2, color=col, label=f'k={k}')
    ax.axhline(np.exp(c_k(k)), color=col, ls=':', alpha=0.5)
ax.set_xlabel('β'); ax.set_ylabel('α(k,β)')
ax.set_title('(3) Map Φ: α → e^{c_k}'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
r_plot = np.linspace(0.1, 6, 300)
for k, col in zip([1, 2, 10, 100], ['blue', 'green', 'orange', 'red']):
    ax.plot(r_plot, [f_k(rr, 50, k) for rr in r_plot], lw=2, color=col, label=f'k={k}')
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.set_xlabel('r'); ax.set_ylabel('f_k(r)')
ax.set_title('(4) k-dependent metric'); ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-2, 1.5)

plt.tight_layout()
plt.savefig('images/dvpt_unified_formula.png', dpi=130, bbox_inches='tight')
plt.show()

# ============================================================
print(); print(line); print(" ALL TESTS PASSED"); print(line)