import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from singularity_avoidance_validation import run

mu_nocbf, _ = run(eps=0.0, tag="no-CBF", dt_ctrl=0.01, n_substeps=5)
mu_100, _ = run(eps=0.03, tag="CBF 100Hz", dt_ctrl=0.01, n_substeps=5)
mu_500, _ = run(eps=0.03, tag="CBF 500Hz", dt_ctrl=0.002, n_substeps=1)

t_nocbf = np.arange(len(mu_nocbf)) * 0.01
t_100 = np.arange(len(mu_100)) * 0.01
t_500 = np.arange(len(mu_500)) * 0.002

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_nocbf, mu_nocbf, label="no CBF (unconstrained PD)", color="#d62728")
ax.plot(t_100, mu_100, label="CBF, 100Hz control", color="#1f77b4")
ax.plot(t_500, mu_500, label="CBF, 500Hz control", color="#2ca02c")
ax.axhline(0.03, color="black", linestyle="--", linewidth=1, label="eps = 0.03")
ax.set_xlabel("time (s)")
ax.set_ylabel("manipulability index mu(q)")
ax.set_title("Gen3 end-effector reaching for a singularity, with and without the CBF")
ax.legend()
fig.tight_layout()
fig.savefig("manipulability_cbf_comparison.png", dpi=150)
print("saved")
