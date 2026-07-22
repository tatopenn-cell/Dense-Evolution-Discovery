import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_IMAGES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(_DATA_DIR / "transizione_fase_ising.csv")

g = df["Campo_g"].values
E_zz = df["Expectation_H_zz"].values

suscettivita = -np.gradient(E_zz, g)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(g, E_zz, color='#FF007F', linewidth=2.5, label='Measured ZZ Correlation')
ax1.set_ylabel("Spin-Spin Correlation <H_zz>", color='#888888')
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
ax1.legend(loc="upper right")
ax1.set_title("Quantum Ising Phase Scan & Susceptibility (3500 Steps)", fontsize=11, fontweight='bold', pad=15)

ax2.plot(g, suscettivita, color='#00FFFF', linewidth=2, label='Fermionic Susceptibility (dZZ/dg)')
idx_max = np.argmax(suscettivita)
g_critico = g[idx_max]

ax2.axvline(g_critico, color='#FFFF00', linestyle=':', alpha=0.8, label=f'Critical Point g ~ {g_critico:.3f}')
ax2.set_xlabel("Transverse Field Strength (g)", color='#888888')
ax2.set_ylabel("Susceptibility", color='#888888')
ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(_IMAGES_DIR / "transizione_fase_ising.png", dpi=300)

print("============================================================")
print(f"📊 Grafico esportato! Punto critico rilevato a g = {g_critico:.3f}")
print("============================================================")
