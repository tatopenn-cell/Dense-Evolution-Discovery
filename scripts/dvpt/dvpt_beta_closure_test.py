# ============================================================
# Two closure tests for the free parameter beta
# in the DVPT unified formula.
#
# Units: G = hbar = c = k_B = 1
#
# Test A: beta = 1 / T_H(r_+), T_H from the modified metric
# Test B: beta = 1 / E_c,    E_c = T_H * A/(4G)   (Jusufi-Anand condition)
#
# Metric:   f_k(r) = 1 - 2 M r / (r^2 + ell^2)
#           ell^2(beta, k) = 1 / (pi * alpha(k, beta))
#           r_+(M, beta)   = M + sqrt(M^2 - ell^2)
#           T_H            = (r_+^2 - ell^2) / (4 pi r_+ (r_+^2 + ell^2))
#
# No claims. Only what the solver returns.
# ============================================================
import numpy as np
from mpmath import mp, mpf, zeta, diff as mpdiff, log
from scipy.special import lambertw
from scipy.optimize import brentq
import matplotlib.pyplot as plt

mp.dps = 40
LINE = "=" * 76

# ---------- Gentile gas (unchanged) ----------
def E_gas(beta, k):
    b = mpf(beta)
    if k == np.inf:
        return float(-mpdiff(zeta, b) / zeta(b))
    return float(-mpdiff(zeta, b) / zeta(b)
                 + (k+1) * mpdiff(zeta, (k+1)*b) / zeta((k+1)*b))

def c_k(k):
    return 1.0 if k == np.inf else 1.0 - float(log(zeta(k+1)))

def alpha_LW(E, k):
    if E <= 1e-60:
        return float(np.exp(c_k(k)))
    return float(lambertw(E * np.exp(E + c_k(k))).real / E)

def ell2_of_beta(beta, k, G=1.0):
    return G / (np.pi * alpha_LW(E_gas(beta, k), k))

# ---------- Metric ----------
def r_plus(M, ell2):
    disc = M*M - ell2
    return M + np.sqrt(disc) if disc >= 0 else np.nan

def T_H(M, ell2):
    rp = r_plus(M, ell2)
    if np.isnan(rp) or rp <= 0:
        return np.nan
    return (rp*rp - ell2) / (4 * np.pi * rp * (rp*rp + ell2))

# ---------- Test A: beta = 1 / T_H ----------
def F_A(beta, M, k):
    ell2 = ell2_of_beta(beta, k)
    if ell2 >= M*M:
        return np.nan
    TH = T_H(M, ell2)
    if np.isnan(TH) or TH <= 0:
        return np.nan
    return beta - 1.0 / TH

def solve_A(M, k, lo=1.05, hi=None):
    if hi is None:
        hi = 50.0 * M + 100.0
    try:
        Flo = F_A(lo, M, k)
    except Exception:
        return None, 'F(lo) error'
    if np.isnan(Flo):
        for b in np.linspace(lo, lo*100, 500):
            v = F_A(b, M, k)
            if not np.isnan(v):
                lo, Flo = b, v
                break
        else:
            return None, 'no lower bracket'
    Fhi = F_A(hi, M, k)
    if np.isnan(Fhi) or Flo * Fhi > 0:
        return None, f'bracket fail: F({lo:.3f})={Flo:+.3e}, F({hi:.3f})={Fhi:+.3e}'
    try:
        return brentq(F_A, lo, hi, args=(M, k),
                      xtol=1e-13, rtol=1e-14, maxiter=300), 'ok'
    except Exception as e:
        return None, f'brentq: {e}'

# ---------- Test B: beta = 1 / E_c with E_c = T_H * A/(4G) ----------
def F_B(beta, M, k):
    ell2 = ell2_of_beta(beta, k)
    if ell2 >= M*M:
        return np.nan
    rp = r_plus(M, ell2)
    TH = T_H(M, ell2)
    if np.isnan(TH) or TH <= 0:
        return np.nan
    A_over_4G = np.pi * rp * rp          # A/(4G) in units G=1
    E_c = TH * A_over_4G
    return beta - 1.0 / E_c

def solve_B(M, k, lo=1.01, hi=1000.0):
    Flo = F_B(lo, M, k)
    Fhi = F_B(hi, M, k)
    if np.isnan(Flo) or np.isnan(Fhi) or Flo * Fhi > 0:
        return None, f'bracket: F({lo})={Flo}, F({hi})={Fhi}'
    return brentq(F_B, lo, hi, args=(M, k), xtol=1e-13), 'ok'

# ============================================================
# TEST A
# ============================================================
print(LINE)
print(" TEST A - beta = 1 / T_H(r_+)")
print(" Units: G = hbar = c = k_B = 1")
print(LINE)
print()
hdr = (f"  {'M':>7} {'k':>5} {'beta*':>16} {'r_+':>12} {'T_H':>14} "
       f"{'ell^2':>14} {'ell^2_inf':>14} {'status':>8}")
print(hdr)
print("  " + "-" * (len(hdr)-2))

Ms_A = [1.0, 2.0, 5.0, 10.0, 100.0]
ks_A = [1, 2, 10, np.inf]
results_A = []
for M in Ms_A:
    for k in ks_A:
        b, status = solve_A(M, k)
        if b is None:
            print(f"  {M:>7.2f} {str(k):>5}  {status}")
            continue
        e = ell2_of_beta(b, k)
        rp = r_plus(M, e)
        TH = T_H(M, e)
        e_inf = 1.0 / (np.pi * np.exp(c_k(k)))
        results_A.append((M, k, b, rp, TH, e, e_inf))
        print(f"  {M:>7.2f} {str(k):>5} {b:>16.8f} {rp:>12.8f} "
              f"{TH:>14.8e} {e:>14.8f} {e_inf:>14.8f} {status:>8}")
    print()

# ============================================================
# TEST B
# ============================================================
print(LINE)
print(" TEST B - beta = 1 / E_c,  E_c = T_H * A/(4G)  (Jusufi-Anand condition)")
print(LINE)
print()
hdr = (f"  {'M':>7} {'k':>5} {'beta_c':>16} {'r_+':>12} {'T_H':>14} "
       f"{'A/(4G)':>12} {'E_c':>14} {'ell^2':>14} {'status':>8}")
print(hdr)
print("  " + "-" * (len(hdr)-2))

Ms_B = [1.0, 2.0, 5.0, 10.0, 100.0]
ks_B = [1, 2, 10, np.inf]
results_B = []
for M in Ms_B:
    for k in ks_B:
        b, status = solve_B(M, k)
        if b is None:
            print(f"  {M:>7.2f} {str(k):>5}  {status}")
            continue
        e = ell2_of_beta(b, k)
        rp = r_plus(M, e)
        TH = T_H(M, e)
        A4 = np.pi * rp * rp
        Ec = TH * A4
        results_B.append((M, k, b, rp, TH, A4, Ec, e))
        print(f"  {M:>7.2f} {str(k):>5} {b:>16.8f} {rp:>12.8f} "
              f"{TH:>14.8e} {A4:>12.6f} {Ec:>14.8e} {e:>14.8f} {status:>8}")
    print()

# ============================================================
# Assert-based checks
# ============================================================
print(LINE)
print(" ASSERT-BASED CHECKS")
print(LINE)
print()

print("  Test A — F(beta*) = 0 (1e-10), T_H > 0")
for M in [1.0, 5.0, 100.0]:
    for k in [1, np.inf]:
        b, _ = solve_A(M, k)
        assert b is not None
        fv = F_A(b, M, k)
        e = ell2_of_beta(b, k)
        TH = T_H(M, e)
        assert abs(fv) < 1e-10, f'A: M={M}, k={k}, F={fv}'
        assert TH > 0, f'A: M={M}, k={k}, T_H={TH}'
print("    OK")

print()
print("  Test A — Schwarzschild limit: |ell^2(100) - ell^2(1000)|/ell^2(100) < 1%")
for k in [1, np.inf]:
    b1, _ = solve_A(100.0, k)
    b2, _ = solve_A(1000.0, k)
    e1 = ell2_of_beta(b1, k)
    e2 = ell2_of_beta(b2, k)
    rel = abs(e1 - e2)/e1
    assert rel < 0.01, f'A limit: k={k}, rel={rel}'
    print(f"    k={str(k):>4}: ell^2(100)={e1:.8f}, ell^2(1000)={e2:.8f}, rel={rel:.2e}  OK")

print()
print("  Test B — F(beta_c) = 0 (1e-10) for every M with a solution")
for M, k, b, *_ in results_B:
    fv = F_B(b, M, k)
    assert abs(fv) < 1e-10, f'B: M={M}, k={k}, F={fv}'
print("    OK (all converged entries)")

print()
print("  Test B — no solution for M >= 5")
for M in [5.0, 10.0, 100.0]:
    for k in [1, np.inf]:
        b, _ = solve_B(M, k)
        assert b is None, f'B: unexpected solution at M={M}, k={k}'
print("    OK")

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (1) Test A: beta*(M) for each k
ax = axes[0]
Ms_plot = np.logspace(0, 3, 30)
for k, col in zip([1, 2, 10, np.inf], ['blue', 'green', 'orange', 'red']):
    bs = []
    for M in Ms_plot:
        b, _ = solve_A(M, k)
        bs.append(b if b is not None else np.nan)
    ax.loglog(Ms_plot, bs, lw=2, color=col, label=f'k={k}')
ax.set_xlabel('M'); ax.set_ylabel(r'$\beta^*(M,k)$')
ax.set_title(r'(1) Test A: $\beta^* = 1/T_H$')
ax.legend(); ax.grid(alpha=0.3, which='both')

# (2) Test A: ell^2(M) for each k
ax = axes[1]
for k, col in zip([1, 2, 10, np.inf], ['blue', 'green', 'orange', 'red']):
    es = []
    for M in Ms_plot:
        b, _ = solve_A(M, k)
        es.append(ell2_of_beta(b, k) if b is not None else np.nan)
    ax.semilogx(Ms_plot, es, lw=2, color=col, label=f'k={k}')
    ax.axhline(1.0/(np.pi*np.exp(c_k(k))), color=col, ls=':', alpha=0.5)
ax.set_xlabel('M'); ax.set_ylabel(r'$\ell^2$')
ax.set_title(r'(2) Test A: $\ell^2(M)$')
ax.legend(); ax.grid(alpha=0.3, which='both')

# (3) Test B: F_beta(beta) for M=1, k=1 and M=5, k=1
ax = axes[2]
betas = np.linspace(1.05, 5, 300)
for M, col in zip([1.0, 2.0, 5.0], ['blue', 'green', 'red']):
    Fv = [F_B(b, M, 1) for b in betas]
    ax.plot(betas, Fv, lw=2, color=col, label=f'M={M}, k=1')
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$F_B(\beta)$')
ax.set_title(r'(3) Test B: $F_B(\beta)$ (no root for M=5)')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('beta_closure_tests.png', dpi=130, bbox_inches='tight')
plt.show()

print()
print(LINE)
print(" ALL ASSERTIONS PASSED")
print(LINE)