# ============================================================
# Derivation of the operator parameter mu
#
# Instead of fitting (mu, eps) on a grid, we solve the matching
# equation for mu analytically:
#
#     ell^2_op(mu) = 4 * ell^2_inf(k)
#
# where
#     ell^2_op(mu) = <r^4> / <r^2> over |psi_0|^2 = sech^{2a(mu)}(r)
#     a(mu)         = ( sqrt(1 + 8 mu) - 1 ) / 2
#     ell^2_inf(k)  = zeta(k+1) / (pi e)
#
# Units: G = hbar = c = k_B = 1
#
# Inputs (already verified, not re-derived here):
#   - ground state of H(mu) = -d^2/dr^2 - 2 mu sech^2(r) is sech^{a(mu)}(r)
#   - with beta = 1/T_H (Test A of dvpt_beta_closure), ell^2 -> ell^2_inf(k)
#     in the limit M -> infinity
#
# Assumed (carried over from dvpt_poschl_teller_width.py, not derived here):
#   - eps = 0
#   - the factor 4 in ell^2_op = 4 ell^2_gas
#
# Nothing else is assumed. The script solves and reports.
# ============================================================
import numpy as np
from mpmath import mp, mpf, zeta, quad, cosh, sqrt, pi as mpi, exp
from scipy.optimize import brentq

mp.dps = 40
LINE = "=" * 76
FACTOR = 4.0          # assumed, not derived here

# ---------- Operator: effective width ell^2_op(mu) ----------
def a_of_mu(mu):
    return (-1.0 + np.sqrt(1.0 + 8.0 * mu)) / 2.0

def I2_anal(a):
    a_mp = mpf(a)
    f = lambda x: x**2 / cosh(x)**(2 * a_mp)
    peak = 1 / sqrt(2 * a_mp)
    return 2 * quad(f, [0, peak, mpf(10), mpf("inf")])

def I4_anal(a):
    a_mp = mpf(a)
    f = lambda x: x**4 / cosh(x)**(2 * a_mp)
    peak = 1 / sqrt(2 * a_mp)
    return 2 * quad(f, [0, peak, mpf(10), mpf("inf")])

def ell2_op(mu):
    if mu <= 0.5:
        return np.nan
    a = a_of_mu(mu)
    return float(I4_anal(a) / I2_anal(a))

# ---------- Gas: asymptotic length ell^2_inf(k) ----------
def ell2_inf(k):
    zk = 1.0 if k == np.inf else float(zeta(k+1))
    return zk / (float(mpi) * float(exp(1)))

# ---------- Matching equation: solve ell^2_op(mu) = 4 ell^2_inf(k) ----------
def solve_mu(k, mu_lo=0.6, mu_hi=100.0):
    target = FACTOR * ell2_inf(k)
    def f(mu):
        return ell2_op(mu) - target
    try:
        mu_star = brentq(f, mu_lo, mu_hi, xtol=1e-12, rtol=1e-14, maxiter=300)
        return mu_star, 'ok'
    except Exception as e:
        return None, f'brentq: {e}'

# ============================================================
# Sanity check: ell^2_op values against the note's table
# ============================================================
print(LINE)
print(" SANITY CHECK: ell^2_op(mu) against the note's table")
print(LINE)
print()
note_values = {1.0: 3.4544, 1.5: 2.2229, 2.0: 1.6721,
               3.0: 1.1589, 5.0: 0.7659, 10.0: 0.4653, 20.0: 0.2971}
print(f"  {'mu':>6} {'ell^2_op':>14} {'note':>14} {'diff %':>10}")
print("  " + "-" * 48)
for mu, val_note in note_values.items():
    val = ell2_op(mu)
    diff = abs(val - val_note) / val_note * 100.0
    print(f"  {mu:>6.2f} {val:>14.8f} {val_note:>14.8f} {diff:>10.4f}")
print()

# ============================================================
# Main loop
# ============================================================
print(LINE)
print(" DERIVATION OF mu FROM ell^2_op(mu) = 4 ell^2_inf(k)")
print(" Units: G = hbar = c = k_B = 1")
print(LINE)
print()
print(f"  {'k':>5} {'ell^2_inf':>14} {'4 ell^2_inf':>14} {'mu*':>14} "
      f"{'a(mu*)':>12} {'E_0':>14} {'ell^2_op(mu*)':>16} {'residual':>12}")
print("  " + "-" * 110)

ks = [1, 2, 3, 5, 10, np.inf]
results = []
for k in ks:
    e_inf = ell2_inf(k)
    target = FACTOR * e_inf
    mu_star, status = solve_mu(k)
    if mu_star is None:
        print(f"  {str(k):>5}  {status}")
        continue
    a_star = a_of_mu(mu_star)
    E0 = -a_star**2
    e_op = ell2_op(mu_star)
    res = e_op - target
    results.append((k, e_inf, target, mu_star, a_star, E0, e_op, res))
    print(f"  {str(k):>5} {e_inf:>14.8f} {target:>14.8f} {mu_star:>14.8f} "
          f"{a_star:>12.6f} {E0:>14.6f} {e_op:>16.10f} {res:>12.2e}")
print()

# ============================================================
# Assertions
# ============================================================
print(LINE)
print(" ASSERT-BASED CHECKS")
print(LINE)
print()

print("  Residual |ell^2_op(mu*) - 4 ell^2_inf(k)| < 1e-10")
for k, e_inf, target, mu_star, a_star, E0, e_op, res in results:
    ok = abs(res) < 1e-10
    print(f"    k={str(k):>5}: residual = {res:+.3e}   {'OK' if ok else 'FAIL'}")
    assert ok, f'residual fails at k={k}'

print()
print("  Monotonicity of mu*(k)")
mus_finite = [(k, mu_star) for k, e_inf, target, mu_star, *_ in results if k != np.inf]
mus_finite.sort(key=lambda t: t[0])
for i in range(1, len(mus_finite)):
    assert mus_finite[i][1] > mus_finite[i-1][1], \
        f'mu* not increasing at k={mus_finite[i][0]}'
print(f"    mu*(k) increases with k (finite values):")
for k, m in mus_finite:
    print(f"      k={k:>3}: mu* = {m:.8f}")
m_inf = [m for k, e, t, m, *_ in results if k == np.inf][0]
print(f"      k=inf: mu* = {m_inf:.8f}")
print()
print("    NOTE: the task statement said 'mu*(k) decreases with k'. The")
print("    computation shows it INCREASES. Reason: ell^2_inf(k) decreases")
print("    with k (zeta(k+1) -> 1), so 4*ell^2_inf(k) decreases; since")
print("    ell^2_op(mu) is decreasing in mu, the solution mu* increases.")

print()
print("  Consistency of mu*(inf) with the pure bosonic target")
target_inf = FACTOR * 1.0 / (float(mpi) * float(exp(1)))
e_inf_check = ell2_op(m_inf)
print(f"    target 4/(pi e)            = {target_inf:.10f}")
print(f"    ell^2_op(mu*(inf))         = {e_inf_check:.10f}")
print(f"    |ell^2_op - target|        = {abs(e_inf_check - target_inf):.2e}")
assert abs(e_inf_check - target_inf) < 1e-10

# ============================================================
# Comparison with the grid fit of section 2b
# ============================================================
print()
print(LINE)
print(" COMPARISON WITH THE GRID FIT (section 2b of the width note)")
print(LINE)
print()
print("  The 40 x 8 grid in (mu, eps) in [0.5, 60] x {0, 0.5, 1, 2, 3, 5, 8, 12}")
print("  has spacing dmu = (60 - 0.5)/39 = 1.526. For small targets the")
print("  minimiser lands on a grid point near the boundary. The values below")
print("  are exact to 1e-12; the grid values differ by up to dmu/2 ~ 0.76")
print("  plus the systematic bias from the coarse spacing.")
print()
print(f"  {'k':>5} {'mu* (derived)':>16} {'ell^2_inf(k)':>16} {'4 ell^2_inf':>16}")
print("  " + "-" * 60)
for k, e_inf, target, mu_star, *_ in results:
    print(f"  {str(k):>5} {mu_star:>16.8f} {e_inf:>16.8f} {target:>16.8f}")

# ============================================================
# Summary
# ============================================================
print()
print(LINE)
print(" WHAT IS DERIVED AND WHAT IS ASSUMED")
print(LINE)
print(f"""
  DERIVED (this script):
    mu*(k)      = solution of ell^2_op(mu) = {FACTOR} * ell^2_inf(k)
    a(mu*)      = ( sqrt(1 + 8 mu*) - 1 ) / 2
    E_0(mu*)    = -a(mu*)^2
  All three are functions of k alone, no grid search.

  ASSUMED (carried over from dvpt_poschl_teller_width.py, not derived here):
    eps = 0
    the factor {FACTOR} in ell^2_op = {FACTOR} * ell^2_gas

  STILL FREE:
    k (discrete; values tested: {ks})
    M (via beta = 1/T_H, Test A of the closure note; ell^2 does not
       depend on M once beta = 1/T_H is imposed)

  NOT PRESENT:
    Phi does not appear in any of these results.
""")
print(LINE)