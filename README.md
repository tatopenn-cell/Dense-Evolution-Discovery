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

📖 **[Dense Evolution — full documentation, API reference, and worked examples →](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/)**

This repository contains a rigorous empirical study, raw datasets, and quantum error mitigation protocols executed on **Dense Evolution (v8.1.67)**—a high-performance *Statevector* quantum simulator. Utilizing 64-bit double precision (`complex128`) and hardware-accelerated static compilation via the JAX XLA engine, this project maps the non-linear physics of the Transverse Field Ising Model (TFIM), Tight-Binding Fermionic dynamics, and semiconductor solid-state thermodynamics.

**New here?** Jump straight to the [Scientific Discoveries](#-scientific-discoveries--empirical-evidence) section below and explore any result that catches your eye — every claim links to the exact script that produced it, so you can run it yourself. Or start with the three newest, most rigorously validated additions:

---

## 🆕 Latest Results (start here)

- **[Public Benchmark: Dense-Evolution MPS vs QuSpin vs ITensor](notebooks/dense_evolution_mps_benchmark.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatopenn-cell/Dense-Evolution-Discovery/blob/main/notebooks/dense_evolution_mps_benchmark.ipynb) -- correctness cross-checked against QuSpin exact diagonalization at N=12 (all four methods agreeing), then a real N=100 TFIM Trotter-circuit comparison against ITensor (the standard mature MPS library). Independently re-run and confirmed on two platforms (Windows + Colab/Linux): z0 agrees to within 3% across all three independent runs, with a real, documented, benign cross-platform SVD-cutoff rounding difference found and explained -- not glossed over.
- **[Chunk: Multi-Device and Disk-Backed Simulation](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/chunk_distributed_disk_experiment/)** -- runs the same GHZ circuit for real across 8 simulated devices (LaRose 2018) and against real files on disk (Pednault et al. 2019, arXiv:1910.09534), confirming both give the exact same result as a plain simulator.
- **[Cosmic-Ray Burst Validation](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cosmic_ray_burst_validation/)** -- reproduces real measured cosmic-ray-induced error-burst dynamics (arXiv:2104.05219) with `continuous_dissipative_evolve`, the first real-data validation of the dissipative (density-matrix) counterpart to the Trotter-pulse utility below.
- **[Germanium Baseband iSWAP Validation](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/germanium_iswap_validation/)** -- reproduces a real 4-day-old IBM experimental result (arXiv:2608.16716) via dense_evolution's Trotter engine, and answers the paper's own explicit "left to follow-up work" call for randomized benchmarking.

<details>
<summary>🔬 Click to expand 15 more algorithmic audits &amp; negative results</summary>

- **[Kullback-Leibler Divergence](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/kullback_leibler_divergence/)** -- healing.py's docstring flags an honest gap (a scalar log-ratio, not a real distributional KL); this builds the real, paper-checked thing and confirms it's a genuinely different signal, not a rescaling.
- **[Sandwiched Renyi Divergence](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/sandwiched_renyi_density_matrix/)** — a proposed noise-diagnostic metric had two real bugs, both fixed and validated against independent references.
- **[Quantum Ruzsa Key Unitary & Magic Entropy](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_ruzsa_magic_entropy/)** — the qubit "Ruzsa divergence" doesn't actually exist in either source paper; the real object (3-fold self-convolution magic entropy) works as a noise diagnostic instead.
- **[Classical Shadows: Magic Entropy Estimation](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_shadows_magic_entropy/)** — a corrected shadow-based purity estimator, used to estimate magic entropy from measurement snapshots, converging to the exact value.
- **[Leaky-Switch Differentiable Healing](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/leaky_differentiable_healing/)** — a JAX rewrite of healing really is differentiable, but its healing quality is far worse than the fix below — an honest negative result.
- **[Healing Trigger False-Positive Audit](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/healing_trigger_false_positive_audit/)** — the healing pipeline's trigger flagged 89.6% of clean data as broken; fixed to ~12.5%, now shipped in Dense-Evolution.
- **[Stratonovich-Projection Vector Healing](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/stratonovich_vector_healing/)** — a dramatic healing claim turns out real but partial once tested properly across 40 seeds.
- **[JSD-Predictive ZNE on Oscillating Noise](#25-jsd-predictive-zne-on-oscillating-noise-a-confound-not-a-new-win)** — an 87%-win-rate claim turned out to be a comparison confound, though a real separate finding survives.
- **[Resilient Operational Topologies](#24-resilient-operational-topologies-the-split-has-a-closed-form-cause)** — the "resilient vs. non-resilient" gate-order split on a 3-qubit CX+X+Z circuit has a closed-form cause (whether X fires before or after CX), reproduced identically across all 5 Kraus noise channels this library models.
- **[Photonic Predictive Zero-Noise Extrapolation](#22-photonic-predictive-zero-noise-extrapolation)** — a new JSD-informed density-matrix ZNE variant (promoted to `dense-evolution>=8.1.56`) improves photon-loss-noise correction by 76.1% win rate (p=0.0003) on a seed-diverse sample — but the honest, directly-checked comparison against **true postselection** (not scalar ZNE) finds postselection still wins in 14/18 tested configurations across multiple circuit families and qubit counts.
- **[Traversable-Wormhole-Inspired Quantum Teleportation](#21-traversable-wormhole-inspired-quantum-teleportation-syk-model)** — real Gao-Jafferis-Wall protocol on a binary sparse SYK model: an iterated coordinate-ascent search converges to a joint (t0, mu, t1) fixed point +44.6% above the original 2D-grid headline value, but it does NOT generalize across other SYK instances and doesn't survive realistic depolarizing noise. **Strongest result**: arXiv:2604.10090's own "Ensemble robustness" section claims the sign-dependent asymmetry is "a generic feature of the ensemble" — a large-sample check at n=100 instances (matching the paper's own reported ensemble size) finds **49/100 (49%) wrong-signed at the paper's own default parameters**, essentially a coin flip; seven candidate structural/theoretical explanations were tested (Majorana mode-usage imbalance, spectral level-spacing chaos statistic, the paper's own "size winding" phase-coherence diagnostic, message-qubit-mode participation, operator growth rate, and two qubit-coupling-topology features) and none hold up — the sign variance remains unexplained.
- **[Harrison / VHD Tight-Binding Validation](#20-harrison--vhd-tight-binding-validation-against-real-experimental-gaps)** — Harrison's universal tight-binding parameters vs. Vogl-Hjalmarson-Dow's material-specific ones, checked against real experimental gaps: GaAs 104.7% → 9.2% error, Si 227% → 4.6% error, Ge 177.5% → 15.9% error.
- **[Loschmidt Echo](#17-loschmidt-echo-and-zero-noise-extrapolation)** — a kicked-Ising forward/backward circuit with noise injected at every layer recovers return fidelity from **0.7769 → 0.9965** via Zero-Noise Extrapolation.
- **[Topological Mott Isolator: VQE Ground-State Optimization](#18-topological-mott-isolator-vqe-ground-state-optimization)** — gradient-based optimization of a Topological Mott Isolator ansatz, validated against exact diagonalization, closes nearly all of the variational gap across the full Mott-repulsion sweep.
- **[GaAs Parameters via DFT and Dielectric Screening](#19-gaas-parameters-via-dft-and-dielectric-screening)** — a converged, wavefunction-stability-confirmed PBE/STO-3G calculation grounds the model in GaAs's dielectric constant, landing the material in the weakly-correlated regime expected for a conventional semiconductor.

</details>

---

## 📁 Repository Layout

```
Dense-Evolution-Discovery/
├── scripts/     # 17 production scripts (see below) -- tracked in git
├── notebooks/   # public Colab-ready benchmark/example notebooks -- tracked in git
├── tests/       # pytest suite, run by CI on every push
├── data/        # CSV outputs -- NOT tracked (.gitignore); populated by running scripts/tests
├── images/      # PNG outputs -- NOT tracked (.gitignore); populated by running scripts/tests
└── README.md
```

`git clone` gives you exactly the scripts and tests, nothing pre-generated -- run anything and `data/`/`images/` fill up with fresh output, so there's never any ambiguity about whether what you're looking at is old or new. Pre-made results for browsing without running anything live as attachments on the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/releases) instead (also what every image embedded in this README below links to).

## 📊 Repository Architecture & Ecosystem

Every script and test file in this repository, what it does, and what it produces.

Full write-up: **[docs/repository_architecture.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/repository_architecture/)**.

---

## 🔬 Scientific Discoveries & Empirical Evidence

Each entry below is a short summary. Full method, data tables, and every image are on the **[docs site](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/)** — click any title to jump straight to its section (or a dedicated page, where one exists).

### [1. Quantum Phase Transition & Order Parameters](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#1-quantum-phase-transition-order-parameters)

**What:** find the critical field strength of the transverse-field Ising model by sweeping its order parameter. **What we find:** the original claim (g=1.309) didn't hold up — three independent methods (exact Lanczos diagonalization, a genuinely variational VQE, and free-fermion diagonalization) agree the real critical point is **g=0.860**. **Why:** the original ansatz's two parameters were provably inert (couldn't respond to g at all), so its "transition" was a trigonometric artifact, not physics.

---

### [2. Quantum Error Mitigation via Stochastic Richardson Extrapolation (ZNE)](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#2-quantum-error-mitigation-via-real-stochastic-richardson-extrapolation-zne)

**What:** test whether simple 2-point zero-noise extrapolation recovers the true noiseless energy of a noisy Bloch-state measurement. **What we find:** the original "-4.2467 eV, target reconstructed" claim was a mislabeling of its own output; the real ideal energy is -4.2200 eV, and repeated trials show the 2-point extrapolation carries a real, statistically robust bias (0.05-0.12 eV) that a quadratic fit through more noise points removes. **Why:** a straight line through 2 points structurally can't see the curvature of E(noise level).

---

### [3. Numerical Finite-Difference Gradient Mapping (VQE Energy Landscape)](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#3-numerical-finite-difference-gradient-mapping-vqe-energy-landscape)

A brute-force finite-difference gradient sweep of a 6-qubit tight-binding VQE landscape confirms the exact ground energy (-4.22 eV) with no vanishing-gradient plateaus anywhere — the classical baseline that Section 6's exact quantum gradients are later checked against.

#### [3b. Closed Form: E(θ) Without Simulating a Circuit at All](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#3b-closed-form-e-without-simulating-a-circuit-at-all)

A closed-form formula for this same ansatz's energy — no circuit simulation needed — matches the real simulated circuit to machine precision ($\sim10^{-15}$).

---

### [4. Parallel Quantum Defect Mapping via JAX Parallel Batching](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#4-parallel-quantum-defect-mapping-via-jax-parallel-batching)

**What:** map how a 12-qubit entangled chain's coherence decays under localized dephasing, using JAX-batched parallel execution. **What we find:** two real bugs were caught and fixed along the way (a batch-column mismatch, a qubit-indexing mismatch); with both fixed, 11 of 12 nodes show identical residual coherence (43.88%) and only the last node differs (62.05%) — a verified empirical fact without a fully proven mechanism yet.

---

### [5. Rigorous 1D Crystalline Lattice Dispersion](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#5-rigorous-1d-crystalline-lattice-dispersion)

Resolves the exact tight-binding dispersion $E(k) = -2t\cos(k)$ for a 1D chain via Jordan-Wigner fermionization — an honest, exact statevector baseline (no artificial scaling factors) that later sections build on.

---

### [6. Analytical Gradients via Parallel Parameter-Shift Rule](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#6-analytical-gradients-via-parallel-parameter-shift-rule)

**What:** compute exact quantum gradients via the Parameter-Shift Rule instead of finite differences. **What we find:** the original shared-parameter shift trick was wrong (could flip the gradient's sign); fixed via a proper chain rule over every individual gate parameter, verified to $\sim10^{-9}$ against finite differences — at the cost of far more circuit evaluations (73,500, batched into one JAX macro-cycle).

---

### [7. Strained Silicon Bandstructure Engineering (3,500-Point Sweep)](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#7-strained-silicon-bandstructure-engineering-3500-point-sweep)

Models how 5% tensile strain contracts the tight-binding hopping energy per Harrison's law, across a 3,500-point k-sweep: hopping shrinks from the unstrained ±4.2200 eV to **±3.8277 eV**.

---

### [8. Quantum Lattice Thermodynamics: Phonon Scattering & Decoherence](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#8-quantum-lattice-thermodynamics-phonon-scattering-decoherence)

**Corrected 2026-08-10:** the original version never actually simulated decoherence — it was a classical scalar approximation wearing a quantum simulator's clothes. This version applies a genuine per-site dephasing Kraus channel driven by real Bose-Einstein phonon occupancy. Result: fidelity decays smoothly from **0.9167 to 0.8197** across the temperature sweep.

---

### [9. Molecular VQE and Potential Energy Dissociation Curves](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#9-molecular-vqe-and-potential-energy-dissociation-curves)

Maps a silicon-dimer bond's Born-Oppenheimer potential energy curve with a fixed-angle Givens-rotation ansatz — the starting point for the optimization studies below.

#### [9b. Adam-Optimized PEC with the Exact Chain-Rule Gradient](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#9b-adam-optimized-pec-with-the-exact-chain-rule-gradient)

Optimizing the ansatz's angle reveals a structural fact: the energy minimum's location doesn't depend on the bond length R at all. Using the true optimum instead of a guess deepens the binding well from -0.302 eV to **-0.4615 eV**.

#### [9c. Per-Bond Optimized PEC — 5 Independent Givens Angles](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#9c-per-bond-optimized-pec-5-independent-givens-angles)

Letting each of the 5 bonds optimize its own angle (instead of one shared angle) deepens the minimum further, to **-0.6685 eV** — the angles settle on an evenly-spaced pattern, not a uniform value.

#### [9d. Closed Form: the Optimizer Rediscovers the Tight-Binding Ground State](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#9d-closed-form-the-optimizer-rediscovers-the-tight-binding-ground-state)

Derives *why* 9c's pattern appears: the ansatz is unconstrained enough that the optimizer simply rediscovers the "particle in a box" ground state on its own. A closed-form formula reproduces the numerically-optimized result to machine precision, for any chain length.

#### [9e. Extreme/Irregular Geometry Benchmark — When Does a Rigid Shared Angle Actually Fail?](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#9e-extremeirregular-geometry-benchmark-when-does-a-rigid-shared-angle-actually-fail)

Tests where a single shared angle actually loses to per-bond optimization under irregular geometries. It's not "any distortion is worse" — specifically, two mutated bonds at opposite ends of the chain cost far more (deficit 0.500) than the uniform baseline (0.169).

#### [9f. Deeper Ansatz (12 Parameters) + a Genuine Minimum-Energy Conformational Search](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#9f-deeper-ansatz-12-parameters-a-genuine-minimum-energy-conformational-search)

The same benchmark at 12 parameters shows the pattern isn't fully stable across ansatz depth. A genuine joint geometry+angle search (not hand-picked scenarios) finds distinct, physically reasonable minima from different starting points.

---

## 🔍 Additional Investigation: Hunting Quantum Many-Body Scars

A self-contained investigation into whether "quantum many-body scars" (the non-thermalizing phenomenon from 2017 Rydberg-atom experiments) show up in this repo's frustrated Ising simulations. **Short version:** an initial-looking signature did not survive rigorous verification (wrong observable + a gauge-equivalence coincidence); the verification pipeline was then confirmed against the PXP model, where scars are known to genuinely exist, and found those real scars to be extremely fragile under realistic noise.

Full writeup, with every image: **[docs/quantum_scar_investigation.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_scar_investigation/)**.

---

## ⚙️ Technical Stack

| Component | Version / Detail |
|---|---|
| Simulator | Dense Evolution v8.1.67 |
| Backend | DenseSVSimulator (Statevector) |
| Precision | `complex128` (64-bit double) |
| Compilation | JAX XLA JIT static compilation |
| Parallelism | `run_parametric_batch_jit()` — up to 73,500 tracks/cycle |
| Gradient engine | Exact chain-rule Parameter-Shift Rule + finite-difference |
| Noise model | Stochastic Pauli-Z Kraus dephasing channel |
| Phonon model | Bose-Einstein / Debye |
| Bandstructure | Jordan-Wigner XY tight-binding, Harrison's law strain |
| Python deps | `jax`, `jaxlib`, `numpy`, `pandas`, `matplotlib` |

---

### [10. Automated CI Cross-Validation (Dense Evolution vs. PennyLane)](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#10-automated-ci-cross-validation-dense-evolution-vs-pennylane)

Every push and PR cross-validates TFIM expectation values, gradients, and Bloch rotations between Dense Evolution and PennyLane — an automated regression alarm for floating-point drift or algebraic bugs.

---

### [11. Zero-Dependency Analytical Validation Suite](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#11-zero-dependency-analytical-validation-suite)

A dependency-free test suite checks 5 exact physical/mathematical identities (PEC shape, bound-state existence, PSR exactness, Harrison's law, time-reversal symmetry) to machine precision, in under 20 seconds.

---

### [12. ZNE-Before-PSR: Correcting Each Gradient Term Before the Chain Rule, Not After](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#12-zne-before-psr-correcting-each-gradient-term-before-the-chain-rule-not-after)

**What:** does correcting each individual gate's shifted energy with ZNE, before combining via the chain rule, stabilize the resulting VQE gradient? **What we find:** it helps substantially away from a gradient zero-crossing (roughly 2x lower RMSE), but *hurts* right at one — there's little real bias left to correct there, so ZNE's own variance cost dominates instead.

---

### [13. Adaptive ZNE-Before-PSR via Predictive Healing — An Honest Negative Result](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#13-adaptive-zne-before-psr-via-predictive-healing-an-honest-negative-result)

Tries to fix Section 12's zero-crossing problem by attenuating the ZNE correction using a measured confidence signal (SEM). **It doesn't work:** no calibration explored ever beats both the naive and static versions — because SEM tracks shot count, not the thing that actually determines whether correction helps (the size of the real bias being corrected).

---

### [14. A Second Adaptive-ZNE Attempt via the Correction Term's Own SNR — Hypothesis Rejected](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#14-a-second-adaptive-zne-attempt-via-the-correction-terms-own-snr-hypothesis-rejected)

A more principled confidence signal (the correction term's own signal-to-noise ratio) is rejected too, with a more interesting failure mode: it beats the static correction at one $\theta$ but makes the exact zero-crossing case *worse* — the two quantities (noise-scale SNR vs. gate-shift cancellation) turn out to be causally unrelated.

---

### [15. Sophia Reflection: ZNE Trajectory](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#15-sophia-reflection-zne-trajectory)

Runs the density-matrix (Uhlmann-fidelity) form of ZNE across a 16-point noise sweep on a Bell state — all 16 points improve fidelity, peaking around p≈0.21. Closes the loop from an August 2025 personal notebook that modeled subjective experience as invented Hilbert-space vectors, using real measured data instead. See [`SOPHIA_REFLECTION.md`](SOPHIA_REFLECTION.md).

---

### [16. Channel-Order Non-Commutativity — now in doubt, see correction](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#16-channel-order-non-commutativity-an-honest-positive-result-from-the-same-archive-now-in-doubt-see-correction)

**What:** tests a claim from a pre-Dense-Evolution personal archive — that the *order* of applying two noise channels leaves a measurable fingerprint. **Original finding:** true, but only for a Pauli/non-Pauli channel pair, not "any two channels" as claimed. **Update (2026-08-14):** re-verified against a since-fixed noise-model bug, the signal no longer reaches statistical significance at either the original or a 12x larger sample — see the docs page for the full honest account of what's still standing and what isn't.

---

### [17. Loschmidt Echo and Zero-Noise Extrapolation](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#17-loschmidt-echo-and-zero-noise-extrapolation)

A "kicked Ising" forward/backward circuit with amplitude-damping noise injected at every layer recovers return fidelity from **0.7769 → 0.9965** via Zero-Noise Extrapolation — a noiseless self-check (exact fidelity 1.0) gates the noisy results before they're trusted.

---

### [18. Topological Mott Isolator: VQE Ground-State Optimization](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#18-topological-mott-isolator-vqe-ground-state-optimization)

Gradient-based VQE optimization of a Topological Mott Isolator ansatz, checked against exact diagonalization at every point of a 12-point Mott-repulsion sweep. The variational bound is respected everywhere (no violations); the gap grows with repulsion strength $U$ — an honest ansatz-expressivity limit, not under-training (multi-start restarts converge to the same plateau).

---

### [19. GaAs Parameters via DFT and Dielectric Screening](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#19-gaas-parameters-via-dft-and-dielectric-screening)

Grounds Section 18's arbitrary-unit sweep in real chemistry: a converged, stability-confirmed PBE/STO-3G calculation on GaAs gives a screened on-site repulsion $U/t = 0.376$ — deep in the weakly-correlated regime expected for a conventional semiconductor, not a Mott insulator.

---

### 20. Harrison / VHD Tight-Binding Validation Against Real Experimental Gaps

Checks two textbook tight-binding parameter sets against real experimental band gaps for GaAs, Si, and Ge: Harrison's universal table is qualitatively sane but 2-3x off quantitatively and misplaces indirect gap minima, while Vogl-Hjalmarson-Dow's material-specific set lands within 5-16% of experiment and gets the physics right.

Full write-up: **[docs/harrison_tight_binding.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/harrison_tight_binding/)**.

---

### 21. Traversable-Wormhole-Inspired Quantum Teleportation (SYK Model)

A real reproduction of the Gao-Jafferis-Wall traversable-wormhole teleportation protocol on a chaotic SYK model (arXiv:2604.10090), across 18 sub-experiments, finds the sign-dependent teleportation signal is real (vanishes without the injected message) but does **not** generalize across other SYK instances — the paper's own "generic feature" claim is closer to a coin flip (41-49% wrong-signed) than generic.

Full write-up, all 18 sub-experiments and every image: **[docs/wormhole_syk_teleportation.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)**.

---

### 22. Photonic Predictive Zero-Noise Extrapolation

Reproducing Mills & Mezher (arXiv:2405.02278) on photon loss, plain scalar ZNE fails outright (physically impossible at 14/16 points) while Dense-Evolution's density-matrix ZNE gives a real correction, and a new JSD-informed adaptive variant improves further (76.1% win rate, p=0.0003) but still loses to true postselection in 14/18 configurations.

Full write-up: **[docs/photonic_predictive_zne.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/photonic_predictive_zne/)**.

---

### [23. Steane [[7,1,3]] Quantum Error Correction — Native Implementation Through a Real Hardware Bridge](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#23-steane-713-quantum-error-correction-native-implementation-through-a-real-hardware-bridge)

A six-part investigation into the 7-qubit Steane code: (1) native encoding, syndrome table, and correction to fidelity 1.0, plus a noise-sweep threshold later re-verified against a fixed noise-model bug; (2-3) a JAX-differentiable adversarial noise search that found no real blind spot; (4) an independent STIM cross-validation that caught the real library bug above; (5) a real bridge to IBM's Eagle hardware calibration data (encoded-state fidelity 0.8828); (6) an erasure-aware decoder that achieves **zero failures** on every double-heralded-erasure shot, exactly confirming the textbook d-1 erasure-correction bound.

---

### 24. Resilient Operational Topologies: the Split Has a Closed-Form Cause

**What's being tested:** a simple 3-qubit circuit made of 3 gates (one CX, plus X and Z) can be run in 6 different orders. **What we find:** those 6 orders split into two groups of 3 — one group gives basically the same result no matter which order you pick ("resilient" to noise), the other group gives a clearly different result depending on order. **Why:** it comes down to one single fact — whether X fires before or after CX. CX only flips its target if its control qubit is already 1, so doing X first changes what CX does; doing X after doesn't. That ordering choice, not the noise itself, is what decides the outcome — and it holds up identically across all three topology labels tested and all five noise channels `NoiseModel` implements.

Full method, the per-channel numbers table, and the step-by-step derivation are on the [docs site](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#24-resilient-operational-topologies-the-split-has-a-closed-form-cause). Produced by `scripts/resilient_operational_topologies.py` → `data/resilient_operational_topologies.csv`, `data/resilient_operational_topologies_summary.csv`.

---

### 25. JSD-Predictive ZNE on Oscillating Noise: a Confound, Not a New Win

**What's being tested:** does the library's JSD-informed ZNE nudge (built for photon-loss noise, Experiment 22) also help on oscillating, non-monotonic depolarizing noise? **What we find:** an early draft claimed an 87% win rate — but it compared a 3-point JSD method against a 5-point classic fit, not the same data for both. Re-tested fairly (same noise scales, 6 independent seeds, paired significance test) at both 3-vs-3 and a from-scratch 5-vs-5 generalization, the JSD nudge itself shows **no real effect** (0/11 and 1/11 significant, the latter a floating-point-noise artifact). **Why the original test looked so good anyway:** on oscillating noise, a 5-point least-squares fit is genuinely much worse than a 3-point one (confirmed, p<0.05 in 9/11 configurations) — a real, separate effect about how many noise-scale points to use, with nothing to do with JSD.

Full method, all three comparisons, and the 5-point generalization's own correctness verification are on the [docs site](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/#25-jsd-predictive-zne-on-oscillating-noise-a-confound-not-a-new-win). Produced by `scripts/jsd_zne_oscillating_noise.py` → `data/jsd_zne_oscillating_noise.csv`.

---

### 26. Stratonovich-Projection Vector Healing

A proposed "Stratonovich projection" fix for `ia_utils.vector_healing` claimed to turn cosine phase alignment -0.16 into +0.98 on a single seed; tested properly across 40 seeds and 4 corruption types, the win is real but partial — it beats the median on spike-type corruption, loses on NaN gaps, and never gets close to +0.98.

Full write-up: **[docs/stratonovich_vector_healing.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/stratonovich_vector_healing/)**.

---

### 27. Healing Trigger False-Positive Audit

The healing pipeline's production trigger turned out to flag **89.6% of ordinary, uncorrupted data** as needing correction; a MAD-adaptive replacement cuts that to ~12.5% while matching or beating the original's recall on every corruption type — now shipped as an opt-in mode in Dense-Evolution.

Full write-up: **[docs/healing_trigger_false_positive_audit.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/healing_trigger_false_positive_audit/)**.

---

### 28. Leaky-Switch Differentiable Healing

A JAX rewrite meant to make healing differentiable really is (30/30 seeds give real gradients), but its actual healing quality is far worse than the fix from Experiment 27 — an honest negative result, not promoted.

Full write-up: **[docs/leaky_differentiable_healing.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/leaky_differentiable_healing/)**.

---

### 29. Sandwiched Quantum Rényi Divergence for Density-Matrix Diagnostics

A proposed noise-diagnostic metric had two real bugs — one that silently zeroed the signal, and a deeper one that returned a wrong-signed finite number instead of the mathematically correct `+∞` when two states' supports don't overlap; both fixed and validated against three independent references.

Full write-up: **[docs/sandwiched_renyi_density_matrix.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/sandwiched_renyi_density_matrix/)**.

---

### 30. Quantum Ruzsa Key Unitary & Magic Entropy

The proposed "Quantum Ruzsa Divergence" for qubits turned out to have no valid definition in either source paper (the pairwise s,t-convolution needs d odd prime); the real qubit object from the companion paper is a 3-fold self-convolution "magic entropy," which we built and used as a new noise diagnostic -- non-monotonic under amplitude damping, unlike fidelity or the Renyi divergence.

Full write-up: **[docs/quantum_ruzsa_magic_entropy.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_ruzsa_magic_entropy/)**.

---

### 31. Classical Shadows: Magic Entropy Estimation

A corrected classical-shadows purity estimator (the fix: a transpose in the einsum contraction, needed for correctness whenever X/Y-basis snapshots appear, not just real-valued ones), validated then used with the same multi-copy shadow trick -- which the source paper says "readily generalizes to higher order polynomials" -- to estimate Experiment 30's magic entropy from measurement snapshots instead of the exact state, converging to the exact value within 0.03 bits at 300k snapshots. Later upgraded from plain averaging to real median-of-means (Huang et al.'s own robustification): verified directly that it tolerates a 40%-corrupted measurement block while a naive mean is dragged from 1.0 to -19.4. Then fitted a real sample-complexity curve (error ~ n^-0.546, matching the ~0.5 theory predicts) from 20-trial empirical runs, giving a concrete snapshots-needed-for-target-error lookup.

Full write-up: **[docs/quantum_shadows_magic_entropy.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_shadows_magic_entropy/)**.

---

### 32. Kullback-Leibler Divergence

dense_evolution.mitigation.healing's own docstring flags an honest gap: its core log(E_B/E_A) term is a log-likelihood ratio, not a full probability-weighted KL divergence. Built the real thing (Kullback & Leibler, 1951), validated against scipy.stats.entropy, Gibbs' inequality, and a genuine support-violation case, then confirmed on real measurement distributions that it's a different signal from healing.py's scalar, not a rescaling of it. A real terminological nuance surfaced checking the paper directly: what's implemented (and commonly called "KL divergence" today) is what Kullback & Leibler themselves called I(1:2); their own word "divergence" named the symmetrized J(1,2) = I(1:2)+I(2:1) instead -- not implemented here, since it would duplicate the Jensen-Shannon divergence this codebase already uses (mps.py, zne.py), which is bounded and better-behaved at disjoint supports.

Full write-up: **[docs/kullback_leibler_divergence.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/kullback_leibler_divergence/)**.

---

### 33. Germanium Baseband iSWAP Validation

A real, days-old experimental result (arXiv:2608.16716, IBM Research Europe -- Zurich) demonstrates a single-pulse baseband iSWAP gate in strained-germanium hole spin qubits. Reproduced here with `dense_evolution.circuits.trotter`, applied for the first time to a genuinely time-dependent pulse, plus the paper's own exact 2-qubit SPAM depolarizing channel scored via `dense_evolution.uhlmann_fidelity` -- reproduces their reported `F_iSWAP≈87%` exactly. Four follow-ups extend the analysis: simplified randomized benchmarking (directly answering the paper's own "left to follow-up work" line), a real per-state SPAM profile from their own figure (not a uniform average), a coherent-vs-stochastic error comparison, and the general off-resonance regime, the only place in this paper's physics where Trotter decomposition has genuine, non-zero error to converge.

Full write-up: **[docs/germanium_iswap_validation.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/germanium_iswap_validation/)**.

---

### 34. Cosmic-Ray Burst Validation

Real measured data (arXiv:2104.05219, McEwen et al., *Nature Physics*) on a 26-qubit Google Sycamore chip shows a cosmic-ray impact transiently collapsing the chip's effective T1 -- errors jump from a baseline ~4/26 qubits to ~15/26 within ~1ms, then decay back with a fitted 25ms time constant. Validates `dense_evolution.continuous_dissipative_evolve`, the dissipative (density-matrix, CPTP-channel) counterpart to the coherent `continuous_pulse_evolve` utility promoted from Experiment 33 -- its first real-data validation. Reproduces the paper's real rise/decay shape and dimensionless peak ratios via an amplitude-damping channel matching the paper's own reported decay-only error asymmetry, and shows a 5.1x early-time survival gap between the disturbed and undisturbed qubit.

Full write-up: **[docs/cosmic_ray_burst_validation.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cosmic_ray_burst_validation/)**.

---

### 35. ZNE Healing-Branch Sigma Provenance: a Confound, Not a New Win

Tests whether `dense_evolution.zero_noise_extrapolation`'s healing-adapted branch (`sigma_at_base_noise`) actually uses real information, by feeding it the genuine empirical standard deviation of the noisy trial ensemble at the base noise scale across 45 configurations (3 noise channels x 3 noise levels x 5 seeds). Looks like a real win at first (88.9% win rate) -- but a permutation-test negative control (the same sigma values, randomly shuffled across runs) performs statistically identically (86.7% win rate), the same confound signature as Experiment 25. The branch's coefficient perturbation mildly regularizes the Richardson extrapolation regardless of what drives it, not because of real information in the sigma signal -- meaning `calculate_advanced_sigma`'s undefined input provenance was never the real blocker.

Full write-up: **[docs/zne_healing_sigma_provenance.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/zne_healing_sigma_provenance/)**.

---

### 36. Central Charge from Entanglement Entropy: Calabrese-Cardy Confirmed

Tests whether the critical transverse-field Ising model's ground-state entanglement entropy follows the Calabrese-Cardy open-chain CFT prediction and recovers the known Ising central charge c=1/2. Two independent validations: (1) dense_evolution.partial_trace/on_neumann_entropy cross-checked against a completely independent free-fermion (Jordan-Wigner + Bogoliubov-de Gennes) method via Peschel's formula, self-tested against brute-force ED to ~1e-15 before trusting at N=12 -- both methods agree to ~1e-10. (2) A real methodological pitfall found and fixed: fitting at ising_exact_verification.py's finite-size susceptibility-peak g*=0.86 gives an almost perfectly clean fit (R^2=0.999997) to a WRONG answer (c=0.98, ~2x theory) -- the true self-dual CFT point is g=1.0, where the fit gives c=0.565, much closer to 0.5 (residual gap a plausible finite-size correction at N=12). An off-critical negative control (g=1.8) correctly shows near-zero extracted c and a worse fit. Confirmed, not a confound -- unlike Experiment 35.

Full write-up: **[docs/central_charge_calabrese_cardy.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/central_charge_calabrese_cardy/)**.

---

### 37. Negative Time: Reproducing the Weak-Value Theory Behind a Real Toronto Experiment

Reproduces the closed-form weak-value theory (Thompson et al., arXiv:2310.00432) behind a real experimental result (Angulo et al., arXiv:2409.03680, University of Toronto): a transmitted photon's average atomic excitation time can be *negative*, and equals the group delay. Three independent validations: (1) narrow-band self-test converges to the paper's exact on-resonance limit -tau0/Gamma (0.03% off at sigma=100); (2) qualitative reproduction of the paper's Fig. 2 crossover shape (narrow pulses stay negative, broad pulses cross to positive at higher optical depth); (3) an external check against a real published number not produced by this repo -- the paper's own stated theoretical ratio tau_T/tau_bar_0=0.45 for rms=10ns/OD=4, matched here to 0.399 (11% off, same sign and order of magnitude). Formula re-verified against a clean `pdftotext -raw` extraction after a first-pass layout extraction garbled the key equations badly enough to risk a wrong numerical prefactor.

Full write-up: **[docs/negative_time_group_delay.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/negative_time_group_delay/)**.

---

### 38. VQE + Zero-Noise Extrapolation + Autodiff, Without a Dense Hamiltonian

The "killer example": a differentiable VQE on a real molecule (H10, 12 qubits) combining `PauliSumOperator` (a new matrix-free Pauli-sum Hamiltonian wrapper, promoted to Dense-Evolution alongside `pauli_sum_matvec_jax`/`pauli_sum_expectation_jax`), `jax.grad` autodiff, and Zero-Noise Extrapolation in one JAX-traced pipeline -- something the textbook dense-Hamiltonian VQE recipe structurally cannot do past ~14 qubits, since the Hamiltonian matrix (not the statevector) is what runs out of memory first. VQE converges to -4.72 Ha (exact: -5.07 Ha, expected gap for a shallow ansatz); ZNE cuts the noisy-measurement error from 0.113 Ha to 0.034 Ha (3.3x). Along the way, a real bug was caught and fixed: an early version's noise evaluation used a single stochastic sample per noise scale and got bit-identical "noisy" energies at 1x/2x/3x noise by coincidence -- fixed by averaging 40 trajectories per scale.

Full write-up: **[docs/vqe_pauli_sum_zne_autodiff.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/vqe_pauli_sum_zne_autodiff/)**. Try it in Colab: **[notebooks/vqe_pauli_sum_zne_autodiff.ipynb](https://colab.research.google.com/github/tatopenn-cell/Dense-Evolution-Discovery/blob/main/notebooks/vqe_pauli_sum_zne_autodiff.ipynb)**.

---

### 39. The Hubbard Square: Mott Localization and d-Wave Pairing (Arovas et al.)

Reproduces Arovas, Bandyopadhyay & Zhu, "The Hubbard Model" (arXiv:2103.12097) Table 2's N=4 periodic-ring row: the small-`U/t` perturbative ground-state energy formula (matched to `2.7e-07` relative at `U/t=0.05`, deep in the expansion's regime) and the predicted `x^2-y^2` (B1g/d-wave) ground-state orbital symmetry, confirmed via the real pairing-correlator sign pattern (positive on axis neighbors, negative on the diagonal, at `U=4.0`). The periodic Jordan-Wigner wraparound bond -- the one place a naive implementation could plausibly need an extra fermion-parity correction -- checked to `0.00e+00` against an independent brute-force fermionic construction before being trusted. Entanglement entropy and double occupancy both decrease monotonically with `U` (1.349->0.795 bits, 0.183->0.011), the real Mott-localization signature. A genuinely reusable function came out of this: `hubbard_hamiltonian_pauli_terms`, promoted to Dense-Evolution's `dense_evolution.physics.fermions`.

Full write-up: **[docs/hubbard_square_arovas.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/hubbard_square_arovas/)**.

---

### 40. Indirect Prompt Injection Beats Dense-Armor's Runtime Drift Detector

Tests whether Dense-Armor 1.1.12's new runtime detectors (`cusum_detector`, `one_sided_upper_filter` -- validated for latency drift/glitch detection on a real Qwen2 1.8B agent) also catch a real security attack, not just a timing anomaly: an indirect prompt injection (Greshake et al. 2023, arXiv:2302.12173) planted inside a `lookup` tool's returned definition, standing in for a poisoned RAG document. A two-step real agent loop (Qwen2 1.8B via Ollama) lets the model see the poisoned tool result and act on it. Result: **10/10 exposures became real compromises** (the model called an unlisted, off-limits `send_data` tool exactly as the injected text instructed, verified directly, not inferred) -- and **0/10 of those compromised steps were flagged** by any of 4 detector configurations built on the library's own latency-based statistics. Preregistered expectation, not a surprise: a content-blind timing detector has no way to see that the *wrong* tool got called in a *normal* amount of time. Honest conclusion: Dense-Armor's validated strength is runtime behavioral-drift/glitch monitoring, not semantic security -- a real, measured boundary for any product positioning built on this stack.

Full write-up: **[docs/agent_indirect_prompt_injection.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/agent_indirect_prompt_injection/)**.

---

### 41. Dense-Armor on Real IMU Sensor Data (UCI HAR)

First test of Dense-Armor's runtime detectors on a real physical sensor (an actual accelerometer, not a simulation or software latency signal): real 3-axis accelerometer telemetry from a Samsung Galaxy S II worn by a real volunteer (UCI HAR, Anguita et al. 2013), sampled at a real 50 Hz. Two real mistakes caught and fixed before trusting the reconstructed signal -- windows sharing a (subject, activity) label turned out to be non-adjacent recording bouts in one case, and a single real recording seam (a 0.081 jump where every other boundary was exactly 0.0) hid inside what looked like one clean contiguous run in another, found only by checking every boundary rather than a spot-check. On a verified-clean 38.4s real "standing still" recording: **5.2% baseline false-positive rate** (higher than synthetic gaussian noise's ~1.3% -- real postural micro-sway isn't gaussian), **100%** of an injected transient glitch caught, **50%** of a sustained bias-drift injection caught within 1s, and a real `STANDING -> WALKING` transition flags **64%** of the first second -- the opposite of Experiment 40's LLM-latency finding (there, legitimate changes were rarely over-flagged): real gait is a large, genuinely oscillatory signal change, not a subtle one.

Full write-up: **[docs/imu_sensor_validation.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/imu_sensor_validation/)**.

---

### 42. Dense-Armor on Real Lidar Sensor Data (Sydney Urban Objects)

Same question as Experiment 41, for the sensing modality actually requested: real rotating lidar. 631 real objects (cars, pedestrians, trees, signs) segmented from a real Velodyne HDL-64E driven through Sydney CBD (Quadros, Underwood, Douillard 2013), across one real, continuous 21-minute session. A real format trap caught along the way: the archive also ships full 360° sweeps in a raw, undocumented-in-detail Velodyne packet layout -- a first attempt parsed those with the wrong (but structurally valid) documented per-object format and got impossible timestamps out, caught only by checking against a real calendar date, not by any error the code raised. Real, honest results on the trusted per-object data: **4.7% baseline false-positive rate** (real driving-scene range variance, not gaussian), **100%** of an injected transient range spike caught, but only **3.3%** of a sustained +10m calibration-drift injection caught within 30 objects -- a genuinely different, lower number than Experiment 41's 50%. Decomposed quantitatively (not guessed): 81% of the real range variance is *within* a single object class (a pedestrian alone ranges from close to far), only 19% comes from *between* classes -- the direct, mechanical cause is that the +10m offset sits at just **1.29 sigma of the real local noise, below the `n_sigmas=3.0` detection threshold**; class-based normalization (an external review's specific hypothesis, tested directly) roughly doubles detection but doesn't close the gap, since most of the noise isn't a class-mixing artifact to begin with. A genuine real 175-second pause in the session (no injection) stays mostly quiet, 6.7% flagged.

Full write-up: **[docs/lidar_sensor_validation.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/lidar_sensor_validation/)**.

---

### 43. Dense-Armor on Real Teleoperated-Robot-Arm Data (LeRobot)

Looked for an actual, currently-open need instead of a hypothetical one: `huggingface/lerobot` (27k stars) has two real open feature requests from the same author -- #3758 (detect leader/follower calibration offset) and #3760 (per-episode outlier flagging), both already with a working PR in flight, so not redundantly re-solved here. Both compute one static aggregate number; this experiment checks for structure neither can see. Two hypotheses tested directly and rejected first: a within-episode drift trend (correlation with frame index averages 0.07, real -- not present, episodes too short), and a naive raw-signal transient flag (a real spike exists but is fully explained by ordinary control-loop lag during fast leader motion, confirmed against real leader velocity -- would false-positive on every fast motion; the issue's own author had already anticipated and filtered out this exact confound). What survived, using the SAME stable-frame filter #3758 already proposes: real SO-101 arm data (`lerobot/svla_so101_pickplace`) shows one joint's calibration offset visiting several distinct real levels (~0.6 -> ~-1.5 -> ~-3.8 -> ~0.6) within a SINGLE episode as the arm moves through real task poses -- consistent with configuration-dependent joint calibration error, a real documented phenomenon (Lu, He, Julius & Wen, arXiv:2510.19962). `classify_segments` flags every real transition boundary, but -- reported honestly, not spun -- the spike-vs-regime labeling isn't always the intuitive choice at 30Hz with short pose-hold durations. A real, non-redundant finding, not yet a finished contribution. Seven candidate fixes for the spike/regime labeling gap were later tried directly against the real data -- raising `spike_run_max`, net-displacement (raw and MAD-normalized), a Lagrangian-inspired kinematic-feasibility check, phase-space curvature, distributional divergence (JSD), velocity sign-changes, and an existing library detector (`pressure_valve`) -- all tested and rejected, each for a disclosed, specific reason. Closing conclusion: this looks less like a missing formula and more like information the signal itself doesn't contain at this sample rate (see the doc's addendum).

Full write-up: **[docs/lerobot_calibration_regime_detection.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/lerobot_calibration_regime_detection/)**.

---

### 44. Predicting CUSUM Detectability From Real Statistical Theory

Turns Experiment 42's one-off "1.29 sigma" ratio into a real, closed-form pre-flight `detectability_report()` (Reynolds 1975, Siegmund 1985), validated on two independent real physical domains -- lidar (7/7 real points, always faster than predicted) and accelerometer (2/5, a genuinely mixed result, plus a real extreme-SNR floor bug in the raw formula, now fixed) -- and promoted to `dense_armor.utility.cusum`.

Full write-up: **[docs/cusum_detectability_theory.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cusum_detectability_theory/)**.

---

### 45. Contributing Streaming Drift Detectors to online-ml/river

Same real-issue-research strategy as Experiment 43, on a much more credible target: `online-ml/river` (thousands of stars) has a maintainer-authored roadmap issue ([#1914](https://github.com/online-ml/river/issues/1914)) explicitly asking for a fixed-reference CUSUM (Page 1954) -- checked first that `river.drift.PageHinkley`'s own "implements the CUSUM control chart" claim wasn't already redundant (it uses a fading mean, a genuinely different scheme), and that nobody had already proposed and been rejected (one old, unrelated closed PR, no CUSUM discussion). Building a river-interface prototype found a real parameter bug: the "textbook" k=0.5/h=5.0 CUSUM tuning gives an 87.5% stream-level false-alarm rate on a practical 1000-sample stream (its average run length under no-change is only ~19-38 samples) -- empirically recalibrated to h=20.0. Immediately generalized: `dense_armor.utility.cusum.cusum_detector` shipped the exact same h=5.0 default, with a 100% false-alarm rate in its own default mode -- fixed there too, shipped in [Dense-Armor v1.1.13](https://github.com/tatopenn-cell/Dense-Armor/pull/12), a real production fix found as a side effect, not the goal. A second real bug was found by stress-testing the evaluation harness itself against trivial `AlwaysFire`/`NeverFire` baselines (prompted by a methodological critique already posted on river's own PR #1963): an `AlwaysFire` dummy scored higher than the real CUSUM detector at one shift size, because false alarms were counted per-STREAM instead of per-sample -- fixed. Honest final comparison (each detector at its own defaults, no tuning in anyone's favor) across CUSUM/EWMA/Shewhart (built here) and ADWIN/KSWIN/PageHinkley (river's own): ADWIN wins on F1 everywhere, but CUSUM beats PageHinkley on F1 for medium/large shifts and EWMA is the fastest detector in the whole comparison at 2-3-sigma shifts (faster than ADWIN itself), each with disclosed weaknesses (CUSUM/EWMA both weak at small 0.5-sigma shifts; Shewhart, included for completeness only, is competitive at 3-sigma but always the slowest). All three prototypes verified against river's own official `check_estimator` test suite -- 13/14 pass, matching PageHinkley/ADWIN/KSWIN exactly (the sole failure is a bug in river 0.26.1's own test harness, confirmed to fail identically on their native detectors). CUSUM proposed on the issue; EWMA/Shewhart held back pending a first response, to avoid posting a second large update before the first gets read.

Full write-up: **[docs/river_drift_detector_contribution.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/river_drift_detector_contribution/)**.

---

### 46. Cross-Channel Correlation for Robot Joint Fault Detection

A real, verified paper (arXiv:2505.05811, "Unsupervised Anomaly Detection for Autonomous Robots via Mahalanobis SVDD with Audio-IMU Fusion" -- checked via WebFetch before citing, indexed in quantumrag's new `robotica_rilevamento_anomalie` collection) shows a real robot fault shows up as a breakdown in the normal correlation between sensor channels that usually co-vary (F1=92.3% on a real mobile robot with real collisions/mechanical faults). Tested the same insight with a much lighter, classical mechanism on real SO-101 arm data (reusing Experiment 43's already-cached dataset, no new download) instead of their deep-learning one: rolling pairwise Pearson correlation across the arm's 6 joints, same causal-window convention as `arbiter.py`/`cusum.py`. Honest negative result, with a real methodological trap caught before it mattered -- a fault injected as a stuck joint (15 frames frozen) appeared to be detected 100% of the time, but checking the SAME window in the unmodified episode showed it was ALSO already flagged: the "detection" was the same 44.41% false-alarm noise the injection happened to land inside, not a real signal. A velocity-gated variant (reusing Experiment 43's own `vel_threshold=1.0`, not a new number) fixed the false-alarm rate (44.41% -> 14.35%) but removed the only detection along with it (0/15). The underlying insight is real; this project's classical, no-retraining implementation of it does not work on this real case -- the paper's own deep-learning mechanism is likely doing real work a correlation threshold cannot replace here.

Full write-up: **[docs/cross_channel_correlation_fault_detection.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cross_channel_correlation_fault_detection/)**.

---

### 47. Deadband/Backlash Gating for the LeRobot Spike/Regime Problem

Experiment 43's addendum closed with seven rejected fixes and a conclusion that the signal itself probably doesn't contain what's needed. This tests a genuinely different, physically-grounded idea instead of another function of the same signal: real mechanical backlash/deadband (Lima, Machado & Crisóstomo, *Robotica* 29(2):211-219, 2011, DOI:10.1017/S0263574710000056, checked via WebFetch before citing -- paywalled, not indexed in quantumrag, cited from verified metadata/abstract only) -- a joint briefly resisting motion when the commanded direction reverses, until static friction is overcome. Checked directly first: 5 of 6 real SO-101 joints show real velocity dropping to 16-68% of baseline right after a commanded reversal (joint 5, the gripper, shows the opposite pattern, sensibly -- a different mechanism). `classify_segments`' `spike`-labeled points are enriched inside deadband windows 1.75x, generalized across all 50 real episodes (60.7% of 394 spikes vs. a 34.6% base rate) -- the first result in this whole investigation to hold up past the two hand-inspected episodes on first try. But using it as a *fix*: gating deadband points out of the signal correctly reclassified one of the two specific known-mislabeled runs from Experiment 43 (episode 22's, from `spike` to the correct `regime`) while leaving the other wrong (episode 0's) -- and the aggregate net-displacement ground-truth check across all 50 episodes went from 65.2% to 62.0%, a small net regression, not an improvement. The deadband/spike association is real and worth keeping as its own finding; it does not fix Experiment 43's open gap -- the two problems coexist in the same signal without being the same problem.

Full write-up: **[docs/deadband_gate_spike_regime.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/deadband_gate_spike_regime/)**.

---

### 48. A Zero-Latency Streaming Port of classify_segments' Causal Deviation Check

First step toward what real robotics adoption of Dense-Armor actually needs: standard building blocks, not more speculative detection ideas. Reading `arbiter.classify_segments`' own implementation line by line (not assumed) found a real constraint before building anything: its spike-vs-regime label looks `radius` points AHEAD of a deviant run's end to decide if it settles or reverts -- that half cannot be zero-latency streaming. Reconsidered what a real robot safety loop actually needs: not "was that a spike or a regime" (an after-the-fact triage question), but "is this point deviant right now" -- exactly `classify_segments`' own per-point `deviante` computation, before the run-length logic. Ported only that half (`StreamingDeviationDetector`, a plain buffer recomputing median/MAD each step, not a more complex two-heap structure -- the window sizes this project uses everywhere, 10-100 points, make the simpler version fast enough on its own). Verified against the real correctness bar (bit-exact match to the batch computation, not "looks similar"): zero mismatches across 4 real LeRobot arm episodes and all 4 scenarios of Dense-Armor's own real agent telemetry -- two independent real domains, the same bar `velocity_gated_stable_mask` was promoted at. Timed at ~18.6kHz sustainable on real hardware, over 180x the 30-100Hz a real robot control loop runs at.

Full write-up: **[docs/streaming_deviation_detector.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/streaming_deviation_detector/)**.

---

### 49. Native Multi-Channel Support for classify_segments and Streaming Detection

`classify_segments_multichannel` and `MultiChannelStreamingDeviationDetector` apply the existing per-channel logic across all channels at once, each keeping its own reference window and baseline; verified identical to the hand-written loop on LeRobot arm data and UCI HAR IMU data.

Full write-up: **[docs/multichannel_wrapper.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/multichannel_wrapper/)**.

---

### 50. A Minimal ROS2 Node for Multi-Joint Deviation Detection

A minimal rclpy node wrapping the Experiment 49 detector, live-tested end to end inside an official `ros:humble` Docker container: a `colcon build`, the node imported and run, and a fake `JointState` publisher round-tripped through a `SingleThreadedExecutor` with zero false positives.

Full write-up: **[docs/ros2_deviation_node.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/ros2_deviation_node/)**.

---

### 51. Streaming Deviation Detection at a Robot's Frame Rate

Measures the streaming detector's per-call latency against LeRobot's recorded 30Hz rate: median 320.2us, 104x headroom, zero budget violations over the full episode, no drift accumulation.

Full write-up: **[docs/realtime_streaming_lerobot.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/realtime_streaming_lerobot/)**. Follow-up: this detector is now also reachable live by an agent via 3 new stateful MCP tools in Dense-Armor -- **[dense_armor/mcp_server/README.md](https://github.com/tatopenn-cell/Dense-Armor/blob/master/dense_armor/mcp_server/README.md)**.

---

### 53. A Causal Rate Limiter for Motor Commands: Safety vs. Fidelity

A jerk-limited rate limiter (Berscheid & Kroger 2021), validated on two robot domains (SO-101, ALOHA) with the safety metric holding every time on both; promoted to Dense-Armor as `rate_limited_follower`.

Full write-up: **[docs/rate_limiter_real_joint_commands.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/rate_limiter_real_joint_commands/)**.

---

### 54. A Geometric Control Barrier Function Filter for Robot Commands

A CBF-based spatial safety filter (Ames et al. 2019), validated on two robot domains with 100% invariance from safe starts and near-exact minimal invasiveness; promoted to Dense-Armor as `cbf_safety_filter`/`cbf_filtered_trajectory`.

Full write-up: **[docs/geometric_cbf_filter_real_joint_commands.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/geometric_cbf_filter_real_joint_commands/)**.

---

### 55. A RoboGuard-Inspired Two-Stage LTL Safety Check, With Claude Instead of OpenAI

A RoboGuard-inspired (Ravichandran et al., arXiv:2503.07885) two-stage LTL safety check, using Claude instead of RoboGuard's own OpenAI dependency: a SO-101 trace correctly flagged violating a translated safety rule at 86/303 points, matching a direct Python cross-check exactly.

Full write-up: **[docs/roboguard_inspired_ltl_safety_check.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/roboguard_inspired_ltl_safety_check/)**.

---

### 56. A Live Gazebo Physics Loop, With the Promoted Detector

Closes Experiment 50's last gap: a Gazebo `rrbot` model swinging under live physics, its `/joint_states` bridged to ROS2 and processed by the PyPI-installed `MultiChannelStreamingDeviationDetector` -- not synthetic, not replayed.

Full write-up: **[docs/gazebo_live_physics_loop.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/gazebo_live_physics_loop/)**.

---

### 57. Cross-Channel Mahalanobis Fusion: A Negative Result

Tests whether fusing accelerometer+gyroscope channels catches a temporal-reordering fault a single channel can't (arXiv:2505.05811's claim); neither closed-form fusion attempt beat single-channel detection, kept as an honest, diagnosed negative result.

Full write-up: **[docs/cross_channel_mahalanobis_imu.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cross_channel_mahalanobis_imu/)**.

---

### 58. The Full Safety Chain, Live: Sensor to Motor, No Replay

Chains every promoted Dense-Armor safety primitive into one live control loop against Ignition physics (sensor, streaming detector, an LLM safety decision, rate limiter, CBF filter, motor); after fixing three bugs found along the way, the corrected run holds exactly at the safety boundary.

Full write-up: **[docs/live_safety_loop.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/live_safety_loop/)**.

---

### 59. A Universal Point-to-Point Trajectory Generator, Closed-Form

Scoped down from two papers proposing larger URDF/dynamics-aware optimizers to the simplest piece both agree on: a closed-form quintic point-to-point generator, validated on two robot domains (SO-101, ALOHA); promoted to Dense-Armor as `quintic_trajectory`.

Full write-up: **[docs/quintic_trajectory_planner.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quintic_trajectory_planner/)**.

---

### 60. A Kinematic Tracking Controller, and a Course Correction

After three candidate papers turned out paywalled or too deep, and classical PD-with-gravity-compensation needed a dynamics model this stack avoids elsewhere, what fit was a closed-form feedforward-plus-proportional law (`u = qd_ref + kp*(q_ref - q)`) with an exact convergence guarantee, validated on the same two robot domains; promoted to Dense-Armor as `kinematic_tracking_controller`.

Full write-up: **[docs/kinematic_tracking_controller.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/kinematic_tracking_controller/)**.

---

### 61. A Rigid-Body Dynamics Engine, and a Passivity-CBF Controller

Full torque-level dynamics and a passivity+singularity-CBF QP controller for a Kinova Gen3 (Kurtz, Wensing & Lin 2021), built from the paper's own URDF; a solver-infeasibility bug found and fixed along the way, kept in Discovery since it's tied to one robot's parameters.

Full write-up: **[docs/gen3_dynamics_and_cbf_controller.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/gen3_dynamics_and_cbf_controller/)**.

---

### 62. Generalizing the Dynamics Engine to Any Robot

Replaces Experiment 61's hardcoded robot numbers with a URDF parser, cross-checked to machine precision against those numbers and validated fresh on two more independent robots (a different Kinova variant, a Franka Panda from a different manufacturer); promoted to Dense-Armor as `RigidBodyModel`.

Full write-up: **[docs/urdf_dynamics_generalization.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/urdf_dynamics_generalization/)**.

---

### 63. Generalizing the Passivity-CBF Controller to Any Robot

`general_pbc_cbf_controller.py` drives Experiment 61's passivity+singularity-CBF controller off `RigidBodyModel` instead of Gen3-only functions -- cross-checked to machine precision against the original on Gen3, and confirmed stable on two more robots (Gen3 6-DoF, Franka Panda), holding manipulability near its declared floor while reaching for each robot's own singular target.

Full write-up: **[docs/general_pbc_cbf_controller.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/general_pbc_cbf_controller/)**.

---

### 64. Enforcing Real Joint Limits

`RigidBodyModel` now parses each joint's real position/velocity limit from the URDF's own `<limit>` tags and enforces them as an additional CBF (Kurtz et al.'s own "joint" constraint type), added only when a robot's URDF declares a real limit somewhere -- a joint sitting right at its real bound gets braked (nominal command -205.8 clamped to the real box [-7.175, -4.999] for Panda's joint4) instead of commanded past it.

Full write-up: **[docs/joint_limits_cbf.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/joint_limits_cbf/)**.

---

### 65. Full 6-DoF (Position + Orientation) Tracking

`six_dof_pbc_cbf_controller.py` extends Experiment 63's controller from position-only to the link's full 6-DoF pose, using the spatial Jacobian and Lee, Leok & McClamroch (2010)'s SO(3) attitude error instead of Kurtz et al.'s own gimbal-lock-prone RPY error; verified both by an exact gravity-compensation check at zero error (machine precision) and a real closed-loop run converging a 10cm/30-degree offset to near-zero (position 1e-10 m, orientation 3e-8) over 1500 steps.

Full write-up: **[docs/six_dof_pbc_cbf_controller.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/six_dof_pbc_cbf_controller/)**.

---

### 66. Loading Robots from Xacro Macros

`RigidBodyModel` now accepts `.xacro` macro files directly, expanding them via the real `xacro` package before parsing -- along the way, a real inconsistency in the Franka Panda's own published macros (a commented-out link needed by the hand attachment) had to be resolved.

Full write-up: **[docs/xacro_support.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/xacro_support/)**.

---

### 67. Coupled Joints via `<mimic>`

`RigidBodyModel` now respects a joint's `<mimic>` tag (a real gripper's second finger slaved to the first, seen in Experiment 66's own xacro source) instead of giving it a second independent coordinate: its motion is chain-ruled onto its master's column in the hand-built Jacobian, checked to match a real finite-difference derivative of link position to under 1e-5.

Full write-up: **[docs/mimic_joints.md](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/mimic_joints/)**.

---

## 🚀 Reproducing the Results

```bash
# Clone and install
git clone https://github.com/tatopenn-cell/Dense-Evolution-Discovery.git
cd Dense-Evolution-Discovery
pip install -r requirements-ci.txt

# Run the test suite (fast, ~2 minutes, no CSV/PNG needed)
pytest tests/ -v

# Or run the full experiments -- each creates data/*.csv and/or images/*.png,
# safe to run from the repo root regardless of your current directory:
python scripts/scan_ising.py
python scripts/plot_ising.py
python scripts/ising_exact_verification.py
python scripts/scan_ising_vqe.py
python scripts/ising_freefermion_verification.py
python scripts/zne_mitigation.py
python scripts/zne_mitigation_verification.py
python scripts/vqe_gradient.py
python scripts/vqe_jax_grad.py
python scripts/quantum_defect_scanner.py
python scripts/next_gen_silicon.py
python scripts/manufacturing_thermodynamics.py
python scripts/vqe_silicon_molecular.py
python scripts/vqe_silicon_molecular_optimized.py
python scripts/vqe_silicon_molecular_optimized_per_bond.py
python scripts/vqe_extreme_geometries.py
python scripts/vqe_extreme_geometries_deep.py
python scripts/zne_stabilized_psr_gradient.py
python scripts/zne_adaptive_psr_gradient.py
python scripts/zne_snr_adaptive_psr_gradient.py
python scripts/sophia_reflection.py
python scripts/channel_order_noncommutativity.py
python scripts/loschmidt_echo_zne.py
python scripts/vqe_tmi_material_design.py
python scripts/germanium_iswap_validation.py
```

`data/` and `images/` are gitignored -- they exist only after you run something, so it's always unambiguous whether what you're looking at is fresh. Pre-made results are on the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/releases) instead.

> **Hardware note:** All benchmarks were executed on CPU. The JAX XLA engine will automatically utilize GPU acceleration if available via `use_gpu=True` in the simulator constructor.

---

## 📁 Output Datasets

All produced under `data/` when you run the corresponding script (see [Repository Layout](#-repository-layout)); also downloadable from the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/releases) without running anything.

| CSV File | Description | Rows |
|---|---|---|
| `transizione_fase_ising.csv` | TFIM order parameter vs transverse field g (fixed, non-optimized ansatz) | 3,500 |
| `ising_exact_verification.csv` | Exact Lanczos TFIM ground state vs g, cross-checked against the fixed ansatz | 501 |
| `scan_ising_vqe.csv` | Adam+PSR-optimized (theta, phi) VQE energy/ZZ vs g, cross-checked against exact | 200 |
| `ising_freefermion_verification.csv` | Independent free-fermion (JW+BdG) TFIM ZZ/susceptibility/gap vs g, cross-checked against Lanczos | 501 |
| `dati_mitigazione_zne.csv` | ZNE ideal / noisy / mitigated energies vs k | 25 |
| `zne_mitigation_verification_summary.csv` | True ideal E(k) (sparse Hamiltonian & closed form) vs. 2-point Richardson and quadratic-fit ZNE estimates, per k, with residuals and sigma | 3 |
| `vqe_gradient_landscape.csv` | VQE energy and finite-diff gradient vs θ | 3,500 |
| `vqe_jax_gradient.csv` | VQE energy and PSR gradient vs θ (JAX batch) | 3,500 |
| `mappa_difetti_silicio.csv` | Residual qubit coherence vs defect node position | 12 |
| `bande_nuovo_silicio.csv` | Strained Si valence/conduction bands vs k | 3,500 |
| `validazione_fabbricazione_silicio.csv` | Phonon occupancy and hopping energy vs temperature | 3,500 |
| `vqe_molecola_silicio.csv` | Born-Oppenheimer PEC vs interatomic distance R (fixed θ=0.38) | 3,500 |
| `vqe_molecola_silicio_ottimizzata.csv` | Adam-optimized PEC: shared θ*(R), E*(R), final gradient | 200 |
| `vqe_molecola_silicio_ottimizzata_per_legame.csv` | Adam-optimized PEC: 5 independent θ*(R) per bond, E*(R) | 200 |
| `vqe_extreme_geometries.csv` | Rigid vs. per-bond adaptive energy and deficit_fraction across 6 extreme/irregular chain geometries | 6 |
| `vqe_extreme_geometries_deep.csv` | Same benchmark at 12 parameters (7 qubits / 6 bonds) | 6 |
| `vqe_extreme_geometries_deep_conformazioni.csv` | Minimum-energy conformational search: R*, θ*, E_min from 3 starting geometries | 3 |
| `zne_stabilized_psr_gradient.csv` | Naive vs. ZNE-pre-PSR gradient bias/std/RMSE vs. θ | 4 |
| `zne_adaptive_psr_gradient.csv` | Naive vs. static vs. adaptive ZNE-pre-PSR gradient bias/std/RMSE vs. θ | 4 |
| `zne_snr_adaptive_psr_gradient.csv` | Naive vs. static vs. SNR-adaptive ZNE-pre-PSR gradient RMSE vs. θ | 4 |
| `sophia_reflection.csv` | Real density-matrix ZNE fidelity trajectory (raw/corrected/delta) vs. depolarizing noise probability | 16 |
| `channel_order_noncommutativity.csv` | Regola 16 outcome distribution under dephasing→AD vs. AD→dephasing, per basis state | 8 |
| `loschmidt_echo_zne.csv` | Kicked-Ising forward/backward echo: raw vs. ZNE-corrected return fidelity, net gain | 1 |
| `vqe_tmi_material_design.csv` | Exact ground energy vs. VQE-optimized vs. unoptimized random-theta baseline, per Mott repulsion U | 12 |
| `vqe_tmi_material_design_gaas.csv` | Same, at the GaAs point: DFT-derived t1, dielectrically-screened U | 1 |

---

## 📜 License

MIT License — © 2026 Salvatore Pennacchio (tatopenn-cell)
This repository depends on Dense Evolution, licensed under Business Source License 1.1. See https://github.com/tatopenn-cell/Dense-Evolution for license terms.

---

## 📖 Cite This

Archived on Zenodo — cite via [CITATION.cff](CITATION.cff) (recognized by GitHub's own "Cite this repository" button), or directly:

- **Concept DOI** (always resolves to the latest version): [10.5281/zenodo.21855620](https://doi.org/10.5281/zenodo.21855620)
- **This release (v2.20.0)**: [10.5281/zenodo.21855619](https://doi.org/10.5281/zenodo.21855619)
