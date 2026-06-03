import matplotlib.pyplot as plt
import pandas as pd

# Carica i dati reali estratti dal simulatore
df = pd.read_csv("transizione_fase_ising.csv")

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

# Disegna la curva d'ordine quantistica
ax.plot(df["Campo_g"], df["Expectation_H_zz"], marker='o', linestyle='-', color='#00FFFF', linewidth=2, label=r'$\langle H_{zz} \rangle$ Osservabile Reale')
ax.fill_between(df["Campo_g"], df["Expectation_H_zz"], color='#00FFFF', alpha=0.1)

# Personalizzazione estetica del grafico scientifico
ax.set_title("Transizione di Fase Quantistica: Modello di Ising in Campo Trasverso (24 Qubit)", fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel("Intensità del Campo Trasversale (g)", fontsize=10, color='#888888')
ax.set_ylabel("Correlazione Ferromagnetica Z-Z", fontsize=10, color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="upper right")

# Evidenzia la direzione della transizione
ax.annotate('Fase Ferromagnetica Ordinata', xy=(0.1, -0.98), xytext=(0.4, -0.85),
            arrowprops=dict(facecolor='#FF007F', shrink=0.05, width=1, headwidth=6))

plt.tight_layout()
plt.savefig("curva_transizione_ising.png", dpi=300)
print("✅ Grafico ad alta risoluzione salvato in: curva_transizione_ising.png")
