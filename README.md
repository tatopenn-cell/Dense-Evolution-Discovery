# 🔬 Quantum Phase Transitions, Variational Gradients, and Error Mitigation 
[![Cross-Validation CI](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml)

This repository contains a rigorous empirical study, raw datasets, and quantum error mitigation protocols executed on **Dense Evolution (v8.1.21)**—a high-performance *Statevector* quantum simulator. Utilizing 64-bit double precision (`complex128`) and hardware-accelerated static compilation via the JAX XLA engine, this project maps the non-linear physics of the Transverse Field Ising Model (TFIM), Tight-Binding Fermionic dynamics, and semiconductor solid-state thermodynamics.

---

## 📁 Repository Layout

```
Dense-Evolution-Ising-Tests/
├── scripts/     # 11 production scripts (see below) -- tracked in git
├── tests/       # pytest suite, run by CI on every push
├── data/        # CSV outputs -- NOT tracked (.gitignore); populated by running scripts/tests
├── images/      # PNG outputs -- NOT tracked (.gitignore); populated by running scripts/tests
└── README.md
```

`git clone` gives you exactly the scripts and tests, nothing pre-generated -- run anything and `data/`/`images/` fill up with fresh output, so there's never any ambiguity about whether what you're looking at is old or new. Pre-made results for browsing without running anything live as attachments on the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases) instead (also what every image embedded in this README below links to).

## 📊 Repository Architecture & Ecosystem

- **`scripts/scan_ising.py`**: Automated data pipeline responsible for high-resolution parameter sweeps and graphical rendering of the ideal ferromagnetic phase transition using a true variational ansatz. Produces `data/transizione_fase_ising.csv`.
- **`scripts/plot_ising.py`**: Computes the first-order numerical derivative (quantum susceptibility) from the CSV dataset to locate the exact critical phase boundary. Produces `images/curva_transizione_ising.png`.
- **`scripts/zne_mitigation.py`**: Mathematical implementation of a stochastic Richardson Zero-Noise Extrapolation (ZNE) protocol over discrete Pauli-Z phase dephasing channels with 2,000 hardware shot sampling. Produces `data/dati_mitigazione_zne.csv` and `images/transizione_ising_mitigata.png`.
- **`scripts/vqe_gradient.py`**: Exact numerical finite-difference gradient tracker (`h = 1e-5`) mapping the variational energy landscape and locating stationary points. Produces `data/vqe_gradient_landscape.csv` and `images/vqe_gradient_landscape.png`.
- **`scripts/vqe_jax_grad.py`**: Advanced VQE gradient execution computing the exact Parameter-Shift Rule gate-by-gate via the chain rule over a massively parallel 73,500-track JAX batch array. Produces `data/vqe_jax_gradient.csv` and `images/vqe_jax_gradient.png`.
- **`scripts/quantum_defect_scanner.py`**: Isotropic resilience topology mapper evaluating node-by-node quantum coherence under a localized parametric RZ dephasing rotation via `run_parametric_batch_jit()`. Produces `data/mappa_difetti_silicio.csv` and `images/mappa_difetti_silicio.png`.
- **`scripts/next_gen_silicon.py`**: Solid-state bandstructure designer tracking continuous dispersion shifts induced by 5% mechanical lattice tensile strain via Harrison's hopping law. Produces `data/bande_nuovo_silicio.csv` and `images/confronto_nuovo_silicio.png`.
- **`scripts/manufacturing_thermodynamics.py`**: Quantum lattice thermodynamics simulator modeling electron-phonon scattering and decoherence via Bose-Einstein statistical distributions over a 10–400 K temperature sweep. Produces `data/validazione_fabbricazione_silicio.csv` and `images/validazione_fabbricazione.png`.
- **`scripts/vqe_silicon_molecular.py`**: Variational Quantum Eigensolver tracking self-consistent Potential Energy Curves (PEC) and Born-Oppenheimer molecular dissociation limits for a silicon dimer, at a fixed variational angle $\theta=0.38$ rad. Produces `data/vqe_molecola_silicio.csv` and `images/curva_potenziale_silicio.png`.
- **`scripts/vqe_silicon_molecular_optimized.py`**: Same PEC, but with a single shared $\theta$ found by real Adam optimization across all $R$ using the exact chain-rule Parameter-Shift Rule gradient, batched per epoch. Produces `data/vqe_molecola_silicio_ottimizzata.csv` and `images/curva_potenziale_silicio_ottimizzata.png`. See Section 9b.
- **`scripts/vqe_silicon_molecular_optimized_per_bond.py`**: Same PEC, but with 5 *independent* Givens angles (one per bond) instead of one shared $\theta$ -- a more realistic hardware-efficient VQE ansatz. Produces `data/vqe_molecola_silicio_ottimizzata_per_legame.csv` and `images/curva_potenziale_silicio_ottimizzata_per_legame.png`. See Section 9c.
- **`tests/test_pennylane_comparison.py`**: Automated cross-validation suite integrating PennyLane as a baseline verification engine. It programmatically contrasts the JAX/XLA statevector predictions generated by Dense Evolution against PennyLane's analytical execution to enforce strict regression boundaries in the CI pipeline.
- **`tests/test_analytical.py`**: Built-in mathematical validation suite executing 5 zero-external-dependency tests. It verifies Potential Energy Curve (PEC) physical boundaries, exact Parameter-Shift Rule (PSR) gradients on $RY+\langle Z \rangle$, Harrison's strain-hopping ratios, and time-reversal dispersion symmetries under machine-precision tolerances ($\le 10^{-10}$).
- **`tests/test_integration_smoke.py`**: Imports and executes the REAL functions from `scripts/vqe_gradient.py`, `scripts/zne_mitigation.py`, `scripts/scan_ising.py`, `scripts/next_gen_silicon.py` (not hand-derived copies), cross-validated against PennyLane or closed-form references.



---

## 🔬 Scientific Discoveries & Empirical Evidence

### 1. Quantum Phase Transition & Order Parameters

We present a rigorous physical validation of the longitudinal spin-correlation order parameter $\langle H_{zz} \rangle$ governed by the 1D Transverse Field Ising Model Hamiltonian:

$$H = -\sum_{i} Z_i Z_{i+1} - g\sum_{i} X_i$$

As the transverse field coupling strength $g$ sweeps from $0.0$ to $2.5$ over 3,500 high-resolution steps, the structural expectation value smoothly decays from an absolute ferromagnetic alignment of $+1.0000$ down to $+0.0050$. This continuous trajectory maps the exact critical boundaries where quantum fluctuations dismantle long-range magnetic ordering, steering the system toward a disordered paramagnetic regime. The critical phase transition boundary is resolved via quantum susceptibility metrics at exactly **$g = 1.309$** with a maximum peak susceptibility of **$1.0000$** under zero-drift conditions.

The ansatz deploys alternating CX–RZ–CX entangling blocks across all 11 nearest-neighbor qubit pairs on a 12-qubit chain, followed by parametric RX rotations scaled to the transverse field strength ($\theta = 0.6 \cdot g$). The `<H_zz>` order parameter is computed analytically from the statevector probability distribution via bitwise parity extraction.

[![Quantum Ising Phase Scan and Susceptibility](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_transizione_ising.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_transizione_ising.png)

---

### 2. Quantum Error Mitigation via Real Stochastic Richardson Extrapolation (ZNE)

To circumvent non-unitary noise without physical hardware overhead, a classical-quantum hybrid mitigation protocol was deployed under a realistic stochastic Pauli-Z dephasing Kraus channel. By scaling the noise density via stretching coefficients ($\lambda_1 = 1.0, \lambda_2 = 2.0$) over $2,000$ discrete hardware shots, a linear Richardson extrapolation was computed:

$$E(0) = 2E(\lambda_1) - E(\lambda_2)$$

The protocol operates on Bloch wavevector states $|\psi(k)\rangle = \frac{1}{\sqrt{N}} \sum_q e^{iqk} |1_q\rangle$ injected over 25 k-points spanning the full Brillouin zone $[-\pi, \pi]$. The base dephasing probability per qubit is $p = 0.06 \cdot \lambda$, applied stochastically via per-shot Kraus channel sampling with controlled seeds.

The ZNE protocol successfully reconstructed the unperturbed, zero-noise ideal target trajectory, forcing the corrupted noisy minimum at $k=0$ (degraded up to $-3.3155\text{ eV}$) back to its true analytic target value of **$-4.2467\text{ eV}$** without introducing non-linear artifacts.

[![Stochastic Zero-Noise Extrapolation Results](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_transizione_noisy.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_transizione_noisy.png)

---

### 3. Numerical Finite-Difference Gradient Mapping (VQE Energy Landscape)

A brute-force numerical gradient sweep over the full VQE variational energy landscape was executed using a centered finite-difference scheme with step $h = 10^{-5}$ radians:

$$\frac{\partial E}{\partial \theta} \approx \frac{E(\theta + h) - E(\theta - h)}{2h}$$

The ansatz uses Givens rotation excitation-preserving blocks (CX–RY–CX–RY–CX chains) initialized from a single-excitation Fock state $|100000\rangle$, preserving strict particle-number conservation throughout. 3,500 continuous $\theta$ values spanning $[0, 2\pi]$ are evaluated over a 6-qubit tight-binding Hamiltonian with $t_{hopping} = 2.11$ eV.

The gradient landscape confirms the exact analytic minimum bound at:

$$E_{ground} = -2 \cdot t_{hopping} = -4.22 \text{ eV}$$

with all stationary points and gradient zero-crossings fully resolved, and no vanishing gradient plateaus present under the compact excitation-preserving ansatz.

> **Note:** This script (`vqe_gradient.py`) uses classical finite-difference differentiation. For exact quantum-native analytical gradients via Parameter-Shift Rule, see Section 6 (`vqe_jax_grad.py`).

#### 3b. Closed Form: E(θ) Without Simulating a Circuit at All

This ansatz shares one $\theta$ across every bond instead of the independent per-bond angles of Section 9c/9d — so instead of landing anywhere on the single-excitation manifold, sweeping $\theta$ traces one 1-parameter curve through it. That curve has a closed form, using the same amplitude-cascade recursion behind Section 9d's discovery:

$$c_0(\theta) = \cos^{N-1}(\theta), \qquad c_q(\theta) = \sin(\theta)\cos^{N-1-q}(\theta) \quad (q=1,\ldots,N-1)$$

`calcola_energia_vqe`'s kinetic sum is **periodic** ($q_{\text{next}} = (q+1) \bmod N$, all $N=6$ bonds including the wraparound $5\to0$ — not the $N{-}1$ open-chain bonds used by the molecular PEC scripts), giving:

$$E(\theta) = -2\,t_{hopping}\sum_{q=0}^{N-1} c_q(\theta)\, c_{(q+1)\bmod N}(\theta)$$

Verified exact (machine precision, $\sim 10^{-15}$) against `calcola_energia_vqe` across the full sweep, including at the printed checkpoints — e.g. $\theta=0.4471\text{ rad} \to E=-3.9489\text{ eV}$, gradient $-8.440000$ at $\theta=0$ — with **no circuit simulation** needed to evaluate it, `scripts/vqe_gradient.py`'s `energia_forma_chiusa()`. `tests/test_integration_smoke.py::test_vqe_gradient_closed_form_matches_real_circuit_exactly` checks the identity at 7 points across the range.

---

### 4. Parallel Quantum Defect Mapping via JAX Parallel Batching

Using the native `run_parametric_batch_jit()` engine, we mapped the isotropic resilience of an entangled state against localized dephasing noise. A 12-qubit entangled chain is prepared by uniform RY($\pi/4$) rotations followed by a full CX entangling ladder. A parametric RZ dephasing gate is injected node-by-node on the diagonal of the batch parameter grid, resulting in 12 concurrent independent execution tracks compiled in a single JAX XLA macro-cycle.

The evaluation maps the systematic loss of $\langle X \rangle$ single-qubit coherence:

$$\langle X_q \rangle = \text{Re}\left[\sum_i \psi_i^* \psi_{i \oplus 2^q}\right]$$

> **Correction 1 (audit finding, dense-evolution 8.1.21):** `run_parametric_batch_jit()` assigns one `parameter_batch` column per rotation gate *in the order the gates appear*, even when a gate is given a literal float instead of a string placeholder — the literal is silently discarded. The batch grid used to have only 12 columns while the circuit has 24 rotation slots (12 fixed RY(π/4) + 12 varying RZ), so the RY gates absorbed the intended RZ values and the true RZ columns ran out of bounds (silently clipped by JAX instead of raising). Fixed by supplying all 24 slots explicitly.
>
> **Correction 2 (found the same day, while trying to explain Correction 1's numbers):** `DenseSVSimulator` uses MSB-first indexing internally ($\text{phys} = N_Q{-}1{-}\text{qubit}$, see `_cx_numpy`/`apply_cx` in `dense_evolution/simulator.py`) — gate-qubit $q$ lives at physical array bit $N_Q{-}1{-}q$. The script's coherence measurement used `1 << local_qubit` directly on the *gate*-qubit index, reading a **different physical qubit** than the one that actually received that row's dephasing. The "70.71% / 50% / 43.88%" pattern originally reported here (and the "interplay between the RY layer and the CX ladder" explanation) was this indexing artifact, not real physics.

With both fixed, the true pattern is much simpler: **11 of the 12 nodes give an identical residual coherence, $43.88\%$**, and only the very last node in the chain differs, at $62.05\%$. We don't have a fully derived analytic explanation for *why* exactly one node is different (an attempted derivation via "the CNOT control's coherence is preserved" turned out to rest on a false premise — that invariance holds for Z-basis populations, not X-basis coherence — so it's reported here as a verified empirical fact, not a proven mechanism).

[![True Quantum Defect Mapping Graph](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/mappa_difetti_silicio.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/mappa_difetti_silicio.png)

---

### 5. Rigorous 1D Crystalline Lattice Dispersion

We resolved the exact 1-electron fermionic Bloch state dispersion relation mapped via Jordan-Wigner transformations. By evaluating the pure exchange interactions ($\langle X_i X_{i+1} + Y_i Y_{i+1} \rangle$) and applying strict periodic boundary conditions (PBC), the engine resolves the full, continuous single-band cosine energy spectrum:

$$E(k) = -2t \cos(k)$$

This eliminates artificial scaling factors and rigid offsets, delivering an honest statevector simulation of tight-binding quantum dynamics under strict 1-fermion subspace conservation. The Bloch states are analytically constructed as $|\psi(k)\rangle = \frac{1}{\sqrt{N}} \sum_q e^{iqk} |1_q\rangle$ over 8 qubits, with the kinetic energy expectation evaluated via tensor-product bitwise XY-operator matrix elements.

[![Rigorous Quantum Tight-Binding Dispersion](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/bande_silicio_ibrido.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/bande_silicio_ibrido.png)

---

### 6. Analytical Gradients via Parallel Parameter-Shift Rule

> **Correction (audit finding, dense-evolution 8.1.21):** the original implementation shifted the *shared* variational parameter $t$ by $\pm\pi/2$ and read the resulting energies straight off the batch. The textbook Parameter-Shift Rule is only exact when a **single gate's own parameter** is shifted while every other gate is held fixed — here each bond applies *two* `ry` gates both driven by $t$ (`param_vqe = t`, `param_vqe_inv = -t`), so the shared-shift reading conflated their contributions. Verified against an independent finite-difference reference: the old heuristic could disagree with the true $dE/dt$ by 100%, including the wrong sign.

The corrected, mathematically exact gradient follows from the chain rule over every gate parameter individually:

$$\frac{dE}{dt} = \sum_{q} \left[ \frac{\partial E}{\partial \theta_{A,q}} \cdot \frac{d\theta_{A,q}}{dt} + \frac{\partial E}{\partial \theta_{B,q}} \cdot \frac{d\theta_{B,q}}{dt} \right], \qquad \frac{\partial E}{\partial \theta} = \frac{1}{2}\left[E(\theta+\tfrac{\pi}{2}) - E(\theta-\tfrac{\pi}{2})\right]$$

where each $\partial E/\partial \theta$ is a genuine single-gate Parameter-Shift Rule evaluation (only that one gate shifted, all 21 others held at their base value), and $d\theta_A/dt = 1$, $d\theta_B/dt = -1$ are the known chain-rule coefficients. Verified against finite differences: agreement to $\sim 10^{-9}$, limited by finite-difference truncation rather than by the PSR itself.

By packing every shifted configuration concurrently into `run_parametric_batch_jit()`, JAX XLA processed **73,500 continuous configurations** (3,500 $\theta$ values $\times$ 21 tracks each: 1 center point + 2 shifts $\times$ 10 gate parameters) in a single chunked macro-batch execution completed in **570 seconds** on CPU — substantially more expensive than the old (incorrect) shortcut, since an exact gradient over a shared parameter genuinely requires one PSR evaluation per gate it drives, not one shift of the shared variable.

The exact quantum derivatives successfully map continuous trajectories, verifying the total absence of vanishing gradient dead-zones or artificial plateaus under compact excitation-conserving ansatze.

[![Exact Parameter-Shift Rule Gradients](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/vqe_jax_gradient.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/vqe_jax_gradient.png)

---

### 7. Strained Silicon Bandstructure Engineering (3,500-Point Sweep)

We modeled a continuous dispersion profile mapping a high-mobility Strained Silicon configuration under a $5\%$ tensile strain ($\varepsilon = 0.05$). By perturbing the atomic equilibrium distances, the physical Hamiltonian undergoes an exponential inter-orbital hopping decay dictated by Harrison's law:

$$t(\varepsilon) = \frac{t_0}{(1 + \varepsilon)^2}$$

The high-resolution 3,500-point k-space parameter sweep executed via JAX maps the physical contraction of the modal hopping energy from the standard $\pm 4.2200\text{ eV}$ limits down to the accurate engineered boundary of **$\pm 3.8277\text{ eV}$** across the Brillouin zone. The simulation uses 8-qubit fermionic Bloch states with Jordan-Wigner XY exchange operators evaluated over all 8 bonds under periodic boundary conditions.

[![Strained Silicon Next-Gen Bandstructure](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_nuovo_silicio.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/confronto_nuovo_silicio.png)

---

### 8. Quantum Lattice Thermodynamics: Phonon Scattering & Decoherence

A quantum-statistical simulation of electron-phonon scattering decoherence was executed over a 10–400 K temperature sweep at 3,500 discrete points, modeling the thermal degradation of coherent electronic hopping in a silicon lattice.

The Debye-Bose-Einstein phonon occupancy is computed as:

$$\bar{n}(\omega, T) = \frac{1}{\exp\!\left(\frac{\hbar\omega}{k_B T}\right) - 1}$$

with $\hbar\omega = 32\text{ meV}$ (silicon optical phonon branch). The effective hopping integral degrades with phonon bath population according to:

$$t_{\text{eff}}(T) = t_0 \left(1 - 0.15 \cdot \bar{n}(\omega, T)\right)$$

This captures the physical mechanism by which thermally-activated phonon scattering reduces long-range electronic coherence. A fixed Bloch state $|\psi(k = \pi/4)\rangle$ is used as the probe state on 8 qubits; the coherent kinetic energy $E(k, T)$ is evaluated via XY-operator matrix elements at each temperature step, tracking the monotonic energy suppression from cryogenic to room temperature.

[![Quantum Lattice Thermodynamics: Phonon Decoherence](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/validazione_fabbricazione.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/validazione_fabbricazione.png)

---

### 9. Molecular VQE and Potential Energy Dissociation Curves

We mapped the exact Born-Oppenheimer Potential Energy Curve (PEC) for a silicon dimer system via a classical-quantum hybrid variational loop. The effective Hamiltonian tracks electronic hopping integrals $t(R)$ alongside nuclear Coulomb repulsion fields $V_{rep}(R)$ decaying over the interatomic coordinate:

$$t(R) = t_0 \, e^{-\beta(R - R_0)}, \qquad V_{rep}(R) = V_0 \, e^{-\gamma(R - R_0)}$$

with $t_0 = 2.11$ eV, $\beta = 1.5$ Å$^{-1}$, $R_0 = 2.35$ Å, $V_0 = 5.4$ eV, $\gamma = 3.0$ Å$^{-1}$. The ansatz is a 6-qubit excitation-preserving Givens circuit initialized from the single-fermion Fock state $|100000\rangle$, with a fixed variational angle $\theta = 0.38$ rad.

#### 9b. Adam-Optimized PEC with the Exact Chain-Rule Gradient

`vqe_silicon_molecular_optimized.py` replaces the fixed $\theta = 0.38$ with a real per-$R$ Adam optimization, using the same exact chain-rule Parameter-Shift Rule gradient validated in Section 6 (agreement with finite differences to $\sim 10^{-9}$). All $R$ points are optimized in parallel — every Adam epoch batches every $R$'s current $\theta$ into a single `run_parametric_batch_jit()` call, rather than looping epochs inside a per-$R$ loop.

The result is a genuine physical insight, not just a better number: $E(R,\theta) = -\frac{t(R)}{2}\,K(\theta) + V_{rep}(R)$, where $K(\theta)$ is the ansatz's kinetic term. Since $t(R) > 0$ everywhere and $V_{rep}(R)$ doesn't depend on $\theta$, the location of the energy minimum over $\theta$ — $\arg\max_\theta K(\theta)$ — **does not depend on $R$ at all**. Optimizing independently at 200 $R$ points converges to the same $\theta^\star \approx 0.613$ rad everywhere (confirmed against an independent fine-grained scan of $K(\theta)$, whose true maximum sits at $\theta = 0.6129$), not a curve. This is a property of the model (the $R$-dependence enters only as a positive multiplicative prefactor on a $\theta$-only kinetic term), not a bug in the optimizer.

Because $\theta = 0.38$ was not that optimum, using the correct $\theta^\star$ deepens the binding well substantially: the global minimum moves from $-0.302\text{ eV}$ at $R \approx 3.32\text{ Å}$ (fixed $\theta$) to $-0.4615\text{ eV}$ at $R \approx 3.17\text{ Å}$ (optimized), an improvement that grows to over $3\text{ eV}$ at short range where the kinetic term dominates.

[![Adam-Optimized Silicon Dimer PEC](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata.png)

The 3,500-point variational sweep over $R \in [1.2, 4.5]$ Å cleanly resolves the stable binding landscape, isolating the exact molecular equilibrium bond length and asymptotic dissociation limits without numerical instabilities.

[![Silicon Dimer Potential Energy Curve](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio.png)

#### 9c. Per-Bond Optimized PEC — 5 Independent Givens Angles

`vqe_silicon_molecular_optimized_per_bond.py` asks a sharper question: Section 9b's shared $\theta$ is $R$-independent *by construction*, since $R$ only ever enters as a scalar prefactor on a $\theta$-only kinetic term — true no matter how that single $\theta$ is chosen. A genuinely richer test is whether a **multi-parameter** ansatz (one independent Givens angle $\theta_q$ per bond, $q=0..4$, same particle-number-conserving structure) settles on a uniform value across bonds, or differentiates.

It differentiates, and cleanly: the 5 bonds converge to $\theta_0,...,\theta_4 \approx 0.234, 0.439, 0.632, 0.818, 1.024$ rad — an almost perfectly even $\approx 0.2$ rad spacing, not noise. This value set is itself still $R$-independent (same underlying reason as 9b — the argmax of a multi-variable $K(\vec\theta)$ under a positive scalar prefactor doesn't move with $R$ either), but it is **not** the "all bonds equal" point: the shared-$\theta$ ansatz was leaving real variational power on the table by forcing symmetry across bonds that the true optimum doesn't have. The minimum deepens further, from $-0.4615\text{ eV}$ (shared, optimized) to $-0.6685\text{ eV}$ (per-bond, optimized).

[![Per-Bond Optimized Silicon Dimer PEC](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata_per_legame.png)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases/download/v2.1.0/curva_potenziale_silicio_ottimizzata_per_legame.png)

#### 9d. Closed Form: the Optimizer Rediscovers the Tight-Binding Ground State

Section 9c's near-even $\approx 0.2$ rad spacing between bond angles isn't the real story — it's a side effect of *what* the optimizer actually converges to. The sequential Givens-rotation ansatz (CX–RY–CX–RY–CX per bond, starting from $|100\ldots0\rangle$) can prepare **any** normalized single-excitation state exactly — it's a universal staircase state-preparation circuit for that Hilbert-space sector. So maximizing the total hopping energy $K(\vec\theta)$ over this ansatz is *unconstrained*: it finds the true maximum of $K$ over every possible single-excitation state, which is exactly the top-eigenvalue problem of an open tight-binding chain — the same math as a **particle in a box**. The optimizer, with no physics told to it beyond "maximize this energy," rediscovers the box's ground state on its own.

The amplitude at site $q$ (0-indexed, $N$ sites) settles on the first sine mode:

$$c_q \propto \sin\!\left(\frac{(q+1)\pi}{N+1}\right), \qquad K_{max} = 4\cos\!\left(\frac{\pi}{N+1}\right)$$

and the per-bond Givens angle that *prepares* this profile via the sequential construction has its own closed form ($r_q$ = the tail norm $\sqrt{\sum_{k=q}^{N-1} c_k^2}$):

$$\theta_q = \arcsin\!\left(\frac{c_q}{r_q}\right)$$

No optimizer needed: plugging this formula directly into the circuit reproduces the numerically Adam-optimized result to **machine precision** ($\sim 10^{-15}$), and holds for every chain length tested (4, 5, 6, 7, 8, 10 qubits) — not a coincidence specific to this 6-qubit example. `scripts/vqe_silicon_molecular_optimized_per_bond.py` implements this as `theta_ground_state_closed_form()` / `kinetic_max_closed_form()`, and `tests/test_vqe_molecular_per_bond.py` verifies the identity exactly (`test_closed_form_ground_state_matches_script_kinetic_maximum`, `test_closed_form_generalizes_across_chain_lengths`).

---

## 🔍 Additional Investigation: Hunting Quantum Many-Body Scars

`scripts/quantum_scar_investigation/` contains a self-contained, honestly-reported investigation into whether a "quantum many-body scar" (the non-thermalizing phenomenon first observed in 2017 Rydberg-atom experiments) shows up in Dense Evolution's frustrated Ising simulations. Full writeup: [`report_indagine_scar.md`](scripts/quantum_scar_investigation/report_indagine_scar.md) (Italian).

**Short version**: an initial-looking scar signature on a 4x4 frustrated TFIM grid did **not** survive rigorous verification (entanglement entropy, Trotter convergence, and a systematic 25-combination parameter scan) — it turned out to be the wrong observable (energy instead of entanglement entropy) plus a gauge-equivalence coincidence between sign patterns. The verification pipeline was then validated against the PXP model (Rydberg blockade), where scars are known to genuinely exist — confirmed via fidelity revivals and the characteristic "tower" of low-entanglement eigenstates in the exact spectrum. Using Dense Evolution's own `NoiseModel.apply_to_sv` (real stochastic Kraus channel, averaged over 30 quantum trajectories), the PXP scars turned out to be extremely fragile: **a 0.5-1% per-site depolarizing error rate destroys almost the entire revival signal.** Projecting the noisy state back onto the exact 13-state scar tower recovers ~31x of the lost revival amplitude — an idealized theoretical bound (not a realizable hardware protocol as-is) showing the protection target exists.

Open for anyone who wants to pick it up: translating the PXP dynamics into an actual circuit and testing revival + a physically realizable protection protocol (e.g. constraint-postselection instead of exact-eigenstate projection) on real quantum hardware.

---

## ⚙️ Technical Stack

| Component | Version / Detail |
|---|---|
| Simulator | Dense Evolution v8.1.21 |
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
### 10. Automated CI Cross-Validation (Dense Evolution vs. PennyLane)

To guarantee the mathematical stability and absolute physical accuracy of the simulated quantum dynamics, the repository includes a strict continuous integration (CI) pipeline executed via GitHub Actions (`ci.yml`). 

The test suite (`test_pennylane_comparison.py`) establishes an automated cross-validation layer by mirroring the statevector computations on two completely independent software architectures:
- **Target Simulator:** Dense Evolution (v8.1.21) accelerated via JAX XLA.
- **Baseline Reference:** PennyLane.

The pipeline runs on every code splotch or pull request, evaluating the numerical consistency of the 1D Transverse Field Ising Model (TFIM) expectation values, variational gradients, and Bloch state rotations. By testing the outputs across both engines, the CI automatically flags floating-point drift or algebraic regressions exceeding machine-epsilon tolerances.

---
### 11. Zero-Dependency Analytical Validation Suite

To ensure absolute core-level stability without relying on third-party frameworks, the repository features a dedicated self-contained validation layer (`test_analytical.py`). This suite runs directly against exact mathematical identities and physics boundaries under machine-precision tolerances ($\le 10^{-10}$), keeping execution times strictly below 20 seconds on standard GitHub Actions CPU runners.

The suite enforces verification across five distinct physical and algorithmic benchmarks:

1. **Potential Energy Curve (PEC) Topography (`test_pec_shape`)**: Validates the qualitative Born-Oppenheimer energy landscape of molecular Silicon systems. It guarantees that the simulation resolves the correct three-region behavior: a steep repulsive wall at short range ($R = 1.4\text{ Å}, E > 0$), a stable binding well at intermediate distance ($R = 3.3\text{ Å}, E < 0$), and asymptotic stabilization near the dissociation limit ($R = 7.0\text{ Å}, |E| < 0.01\text{ eV}$).
2. **Bound-State Existence (`test_pec_minimum_is_negative`)**: Scans the well core ($R \in [3.0, 4.5]\text{ Å}$, empirically confirmed bound at every sampled point) and asserts $E < -0.01\text{ eV}$ at each one, proving a genuine stable ground state rather than merely finite output — tightened during the dense-evolution 8.1.21 audit, when this test was found asserting only `np.isfinite` despite its docstring's stronger claim.
3. **Exact Parameter-Shift Rule (`test_psr_exactness_ry_z`)**: Mathematically benchmarks the single-gate PSR primitive underlying the VQE gradient engine (`vqe_jax_grad.py`). By tracking an $RY(\theta)|0\rangle$ state followed by a $\langle Z \rangle$ measurement, it verifies that the computed gradient perfectly mirrors the exact analytical identity $\frac{dE}{d\theta} = -\sin(\theta)$. `tests/test_vqe_jax_gradient.py` extends this same exactness check to the full multi-gate, chain-rule PSR gradient used in production.
4. **Harrison's Hopping Law (`test_harrison_strain_ratio`)**: Verifies the bandstructure deformation engine under mechanical stress (`next_gen_silicon.py`). It enforces that the exact ratio of strained to unstrained tight-binding energies follows Harrison's solid-state scaling law, $t(\varepsilon) = \frac{t_0}{(1+\varepsilon)^2}$, at every non-trivial $k$-point across the Brillouin zone.
5. **Time-Reversal Dispersion Symmetry (`test_dispersion_time_reversal_symmetry`)**: Checks the underlying algebraic symmetry of the tight-binding Bloch states, ensuring that the dispersion relation satisfies the strict time-reversal constraint $E(k) \equiv E(-k)$ to isolate and prevent unphysical symmetry-breaking artifacts.

---

## 🚀 Reproducing the Results

```bash
# Clone and install
git clone https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests.git
cd Dense-Evolution-Ising-Tests
pip install -r requirements-ci.txt

# Run the test suite (fast, ~2 minutes, no CSV/PNG needed)
pytest tests/ -v

# Or run the full experiments -- each creates data/*.csv and/or images/*.png,
# safe to run from the repo root regardless of your current directory:
python scripts/scan_ising.py
python scripts/plot_ising.py
python scripts/zne_mitigation.py
python scripts/vqe_gradient.py
python scripts/vqe_jax_grad.py
python scripts/quantum_defect_scanner.py
python scripts/next_gen_silicon.py
python scripts/manufacturing_thermodynamics.py
python scripts/vqe_silicon_molecular.py
python scripts/vqe_silicon_molecular_optimized.py
python scripts/vqe_silicon_molecular_optimized_per_bond.py
```

`data/` and `images/` are gitignored -- they exist only after you run something, so it's always unambiguous whether what you're looking at is fresh. Pre-made results are on the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases) instead.

> **Hardware note:** All benchmarks were executed on CPU. The JAX XLA engine will automatically utilize GPU acceleration if available via `use_gpu=True` in the simulator constructor.

---

## 📁 Output Datasets

All produced under `data/` when you run the corresponding script (see [Repository Layout](#-repository-layout)); also downloadable from the [Releases page](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/releases) without running anything.

| CSV File | Description | Rows |
|---|---|---|
| `transizione_fase_ising.csv` | TFIM order parameter vs transverse field g | 3,500 |
| `dati_mitigazione_zne.csv` | ZNE ideal / noisy / mitigated energies vs k | 25 |
| `vqe_gradient_landscape.csv` | VQE energy and finite-diff gradient vs θ | 3,500 |
| `vqe_jax_gradient.csv` | VQE energy and PSR gradient vs θ (JAX batch) | 3,500 |
| `mappa_difetti_silicio.csv` | Residual qubit coherence vs defect node position | 12 |
| `bande_nuovo_silicio.csv` | Strained Si valence/conduction bands vs k | 3,500 |
| `validazione_fabbricazione_silicio.csv` | Phonon occupancy and hopping energy vs temperature | 3,500 |
| `vqe_molecola_silicio.csv` | Born-Oppenheimer PEC vs interatomic distance R (fixed θ=0.38) | 3,500 |
| `vqe_molecola_silicio_ottimizzata.csv` | Adam-optimized PEC: shared θ*(R), E*(R), final gradient | 200 |
| `vqe_molecola_silicio_ottimizzata_per_legame.csv` | Adam-optimized PEC: 5 independent θ*(R) per bond, E*(R) | 200 |

---

## 📜 License

MIT License — © 2026 Salvatore Pennacchio (tatopenn-cell)
This repository depends on Dense Evolution, licensed under Business Source License 1.1. See https://github.com/tatopenn-cell/Dense-Evolution for license terms.
