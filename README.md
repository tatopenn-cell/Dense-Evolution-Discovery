<p align="center">
  <img src="docs/assets/banner.svg" alt="Dense Evolution Discovery — Research lab for robot safety and quantum error mitigation" width="900">
</p>

# 🔬 Dense Evolution Discovery — Quantum Simulation Experiments and Robustness Studies

[![Cross-Validation CI](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square)](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/)
[![Dense Evolution](https://img.shields.io/pypi/v/dense-evolution?style=flat-square&color=00e5ff&label=dense-evolution)](https://pypi.org/project/dense-evolution/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/Backend-JAX_XLA-f9ab00?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![Latest Release](https://img.shields.io/github/v/release/tatopenn-cell/Dense-Evolution-Discovery?style=flat-square&color=blueviolet)](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/releases)
[![Last Commit](https://img.shields.io/github/last-commit/tatopenn-cell/Dense-Evolution-Discovery?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/commits/main)
[![Issues](https://img.shields.io/github/issues/tatopenn-cell/Dense-Evolution-Discovery?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/issues)
[![Stars](https://img.shields.io/github/stars/tatopenn-cell/Dense-Evolution-Discovery?style=flat-square&color=yellow)](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/stargazers)
[![DOI](https://zenodo.org/badge/1258407155.svg)](https://doi.org/10.5281/zenodo.21855619)

This is the research lab for [Dense Evolution](https://pypi.org/project/dense-evolution/) — every claim below is a real script, run and checked before it was written up, with negative results kept in alongside the positive ones.

📖 **[Full documentation, every experiment, every image →](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/)**

## Sections

- **[MPS GPU Optimization](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/mps_gpu_optimization/)** — the matrix-product-state simulator backend, made faster on GPU (2.16x, real API, real hardware), with the three real bugs found along the way.
- **[Dense-Armor: Robot Safety Monitoring](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/live_safety_loop/)** — drift/fault detection and safety filtering validated on real IMU, Lidar, and robot-arm (LeRobot) data.
- **[Dense-Evolution: Quantum Simulation & Physics](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#scientific-discoveries-empirical-evidence)** — TFIM phase transitions, VQE, ZNE, tight-binding materials, and a critical replication of the traversable-wormhole teleportation protocol.
- **[Diagnostics & Healing](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/sandwiched_renyi_density_matrix/)** — density-matrix noise diagnostics (Rényi divergence, magic entropy) and vector-healing bug fixes shipped back into Dense-Evolution.

## Quick Start

```bash
git clone https://github.com/tatopenn-cell/Dense-Evolution-Discovery.git
cd Dense-Evolution-Discovery
pip install -r requirements-ci.txt
pytest tests/ -v
```

Every script under `scripts/` is runnable the same way (`python scripts/<name>.py`) and writes its own `data/*.csv`/`images/*.png` — both gitignored, so what you see after running is always fresh. Full script-by-script index: **[Repository Architecture](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/repository_architecture/)**. Pre-made results without running anything: [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/releases).

## 📜 License

MIT License — © 2026 Salvatore Pennacchio (tatopenn-cell)
This repository depends on Dense Evolution, licensed under Business Source License 1.1. See https://github.com/tatopenn-cell/Dense-Evolution for license terms.

## 📖 Cite This

Archived on Zenodo — cite via [CITATION.cff](CITATION.cff) (recognized by GitHub's own "Cite this repository" button), or directly:

- **Concept DOI** (always resolves to the latest version): [10.5281/zenodo.21855620](https://doi.org/10.5281/zenodo.21855620)
- **This release (v2.20.0)**: [10.5281/zenodo.21855619](https://doi.org/10.5281/zenodo.21855619)
