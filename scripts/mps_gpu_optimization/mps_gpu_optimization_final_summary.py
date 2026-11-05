"""
Final real-GPU number for the whole MPS optimization line of work
(bucketed-SVD dispatch + gate blocking), after all three production
bugs found along the way were fixed (Dense-Evolution PR #226, #228,
#229). Measured on Kaggle T4 via the literal public API,
run_circuit_jit(ops, fuse_gates=True/False), same N=50 5-step TFIM
circuit used throughout this investigation, fresh |0...0> instance +
shared already-compiled kernel methodology (the only correct one, see
docs/mps_gpu_optimization.md).
"""
import csv
import os

import matplotlib.pyplot as plt

warm_default = 2.895
warm_fused = 1.342
speedup = warm_default / warm_fused

os.makedirs("docs/assets/mps_gpu_optimization", exist_ok=True)

with open("docs/assets/mps_gpu_optimization/final_gpu_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "warm_seconds", "speedup_vs_default"])
    w.writerow(["default (fuse_gates=False)", warm_default, 1.0])
    w.writerow(["fuse_gates=True", warm_fused, round(speedup, 2)])

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["default\n(fuse_gates=False)", "fuse_gates=True"],
               [warm_default, warm_fused],
               color=["#8a8f98", "#00b37a"])
for bar, val in zip(bars, [warm_default, warm_fused]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.3f}s",
            ha="center", fontsize=11)
ax.set_ylabel("warm time, seconds (Kaggle T4, N=50 TFIM)")
ax.set_title(f"run_circuit_jit, real API, both fixes applied: {speedup:.2f}x")
fig.tight_layout()
fig.savefig("docs/assets/mps_gpu_optimization/final_gpu_summary.png", dpi=150)
print(f"speedup: {speedup:.2f}x")
