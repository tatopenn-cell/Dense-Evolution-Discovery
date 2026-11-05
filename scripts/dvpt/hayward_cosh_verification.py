import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

print("=" * 74)
print(" VERIFICA RAMO FALSIFICABILE — Hayward + cosh")
print("=" * 74)

# ============================================================
# Setup: metrica Hayward-cosh
# ============================================================
# f(r) = 1 - r_s r² / (r³ + r_s ℓ(r)²)
# ℓ(r) = ℓ0 / cosh((r/ℓ0)^n)
# Unità: r_s = 1, G = c = 1

r_s = 1.0
ell0 = 0.1 * r_s   # ramo conservativo
n = 2

def ell(r):
    return ell0 / np.cosh((r / ell0) ** n)

def f_HC(r):
    if r < 1e-12:
        return 1.0
    return 1 - r_s * r**2 / (r**3 + r_s * ell(r)**2)

def f_Schw(r):
    return 1 - r_s / r

# ============================================================
# [1] Verifica: f_HC ≈ f_Schw per r ≳ r_s
# ============================================================
print("\n[1] La metrica è esattamente Schwarzschild fuori dal core?")
print(f"\n    {'r/r_s':>8} {'f_HC':>16} {'f_Schw':>16} {'|Δf|':>14}")
print("    " + "-" * 58)
for r in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    fhc = f_HC(r)
    fschw = f_Schw(r)
    print(f"    {r:>8.2f} {fhc:>16.10f} {fschw:>16.10f} {abs(fhc-fschw):>14.2e}")

# Valore della regolarizzazione a r = 1.5 r_s (paper: 3.84e-99)
r_test = 1.5
rho_reg = ell(r_test)
print(f"\n    ℓ(1.5 r_s) = {rho_reg:.2e}")
print(f"    paper: 3.84e-99")

# ============================================================
# [2] Photon sphere: r f'(r) - 2 f(r) = 0
# ============================================================
print("\n[2] Photon sphere e shadow radius")

def dV_dr(r):
    dr = 1e-7
    fp = (f_HC(r + dr) - f_HC(r - dr)) / (2 * dr)
    return r * fp - 2 * f_HC(r)

r_ph = brentq(dV_dr, 1.4, 1.6, xtol=1e-12)
b_HC = r_ph / np.sqrt(f_HC(r_ph))

r_ph_schw = 1.5 * r_s
b_schw = r_ph_schw / np.sqrt(f_Schw(r_ph_schw))

print(f"    r_ph (Hayward-cosh)  = {r_ph:.10f} r_s")
print(f"    r_ph (Schwarzschild) = {r_ph_schw:.10f} r_s")
print(f"    b (Hayward-cosh)     = {b_HC:.10f} r_s")
print(f"    b (Schwarzschild)    = {b_schw:.10f} r_s")
print(f"    Δb/b                 = {(b_HC/b_schw - 1):.3e}")

# ============================================================
# [3] Shadow di M87* (in μas)
# ============================================================
print("\n[3] Shadow di M87* — confronto con EHT")

G_SI = 6.674e-11
c_SI = 3e8
M_sun = 1.989e30
pc = 3.086e16
muas_per_rad = 206265 * 1e6

M_M87 = 6.5e9 * M_sun           # kg
d_M87 = 16.8e6 * pc             # m
r_s_M87 = 2 * G_SI * M_M87 / c_SI**2  # m

# Raggio shadow in μas (non diametro)
theta_HC = b_HC * r_s_M87 / d_M87 * muas_per_rad
theta_Schw = b_schw * r_s_M87 / d_M87 * muas_per_rad

print(f"    Predizione Schwarzschild: {theta_Schw:.3f} μas")
print(f"    Predizione Hayward-cosh:  {theta_HC:.3f} μas")
print(f"    Misura EHT (raggio):      21.0 ± 1.5 μas")
print(f"    Tensione Hayward-cosh:    {abs(theta_HC - 21.0)/1.5:.4f}σ")
print(f"    Paper dichiara:           19.85 μas, 0.77σ")

# ============================================================
# [4] Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

r_plot = np.linspace(0.1, 5, 500)
ax = axes[0]
ax.plot(r_plot, [f_HC(r) for r in r_plot], 'b-', lw=2, label='Hayward + cosh')
ax.plot(r_plot, [f_Schw(r) for r in r_plot], 'k--', lw=1.5, label='Schwarzschild')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.axvline(r_ph, color='r', ls=':', alpha=0.7, label=f'r_ph = {r_ph:.3f} r_s')
ax.set_xlabel('r / r_s'); ax.set_ylabel('f(r)')
ax.set_title('Metrica Hayward-cosh vs Schwarzschild')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylim(-0.5, 1)

ax = axes[1]
ax.semilogy(r_plot, [ell(r) for r in r_plot], 'r-', lw=2)
ax.set_xlabel('r / r_s'); ax.set_ylabel('ℓ(r) / r_s')
ax.set_title('Regolatore cosh — decadimento')
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()

print("\n" + "=" * 74)
print(" VERDETTO")
print("=" * 74)
print("""
  [1] f_HC ≈ f_Schw fuori dal core          → confermato
  [2] Photon sphere ≈ 1.5 r_s                → confermato
  [3] Shadow M87* compatibile con EHT 1σ     → confermato
  [4] Ringdown shift +2.27%                  → NON verificato (serve
                                                un solver Regge-Wheeler)
""")