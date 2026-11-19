"""Figures of the paper, from results_cpu.json and results_gpu.json (written by experiments.py).

    python make_figures.py

Writes docs/papers/assets/float32/fig_kappa.png, fig_longn.png and fig_layernorm.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "docs" / "papers" / "assets" / "float32"
OUT.mkdir(parents=True, exist_ok=True)
U = 2.0**-24
COLORS = {"naive": "#9aa5b1", "reference": "#d1495b", "kernel": "#00798c"}
LABELS = {
    "naive": "naive two-pass",
    "reference": "corrected two-pass (reference)",
    "kernel": "kernel (shift + compensated tree)",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(name):
    return json.loads((HERE / f"results_{name}.json").read_text(encoding="utf-8"))


def fig_kappa(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, r) in zip(axes, res.items()):
        for m in ("naive", "reference", "kernel"):
            pts = sorted((k["kappa"], k[m]["max_rel"]) for k in r["kappa"])
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "o",
                ms=4,
                color=COLORS[m],
                alpha=0.75,
                label=LABELS[m],
            )
        ax.axhline(U, color="k", lw=0.8, ls="--")
        ax.text(1.2e4, U * 1.15, "u = 2^-24", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"condition number $\kappa = |\mathrm{mean}| / \mathrm{std}$")
        env = r["environment"]
        ax.set_title(f"{name.upper()} ({env['backend']}, jax {env['jax']})")
    axes[0].set_ylabel("maximum relative error of the variance")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_kappa.png", dpi=160)
    plt.close(fig)


def fig_longn(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, r) in zip(axes, res.items()):
        for m in ("reference", "kernel"):
            worst = {}
            for x in r["longn"]:
                worst[x["n"]] = max(worst.get(x["n"], 0.0), x[m])
            ns = sorted(worst)
            ax.plot(ns, [worst[n] for n in ns], "o-", color=COLORS[m], label=LABELS[m])
        ax.axhline(U, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of values n in the reduction")
        ax.set_title(f"{name.upper()} ({r['environment']['backend']})")
    axes[0].set_ylabel("worst maximum relative error over the three offsets")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_longn.png", dpi=160)
    plt.close(fig)


def fig_layernorm():
    res = {"cpu": load("cpu_ln"), "gpu": load("gpu_ln")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {
        "standard_reference": ("#d1495b", "o", "mean of the data, reference variance"),
        "standard_kernel": ("#edae49", "s", "mean of the data, kernel variance"),
        "centred_kernel": ("#00798c", "^", "centred on the shift, kernel variance"),
    }
    for ax, (name, r) in zip(axes, res.items()):
        rows = r["layernorm_centred"]
        for key, (col, mark, lab) in styles.items():
            ax.plot(
                [x["mu"] for x in rows],
                [x[key]["forward_max_abs_err"] for x in rows],
                mark + "-",
                color=col,
                label=lab,
                ms=5,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("offset of the rows (mean)")
        ax.set_title(f"{name.upper()} ({r['environment']['backend']})")
    axes[0].set_ylabel("maximum error of the normalised output")
    axes[0].legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_layernorm.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    res = {"cpu": load("cpu"), "gpu": load("gpu")}
    fig_kappa(res)
    fig_longn(res)
    fig_layernorm()
    print("written", sorted(p.name for p in OUT.iterdir()))
