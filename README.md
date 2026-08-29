<p align="center">
  <img src="docs/assets/banner.svg" alt="Dense Evolution Discovery — Empirical studies & quantum error mitigation research" width="900">
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
