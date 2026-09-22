# The Dynamic Vacuum Pressure Theory: Five Levels, Checked One at a Time

**In plain terms**: this is an independent theoretical-physics paper (Salvatore Pennacchio, 2026, two parts) proposing that the vacuum regularizes itself with a `cosh`-shaped gap, and that this same shape shows up as a black-hole regularization length, a cosmological bounce factor, a neutron-star equation-of-state correction, and a regularized Coulomb potential. This page independently re-derives its central numerical claim and re-runs its own verification code, rather than taking the paper's word for any of it -- and documents honestly what holds up and what doesn't.

The two papers (both included, real, self-authored) are indexed in `quantumrag`'s `pressione_dinamica_vuoto` collection alongside the six external references they build on (k-essence, galileon gravity, super-renormalizable and nonlocal gravity, the SLy neutron-star equation of state, and Endrizal's InfoCDM+ dark-energy model).

## What actually gets checked here

The paper structures itself in "levels" (its own Table 6). Levels 0-1 are a philosophical framing (vacuum &rarr; symmetry &rarr; opposite energies &rarr; pressure &rarr; particle-or-dark-matter &rarr; gravity-as-memory) with no equations -- not evaluated here, since the paper itself doesn't claim it's physics yet. Levels 2.5 through 5 make specific, checkable mathematical claims. This page checks those.

## Level 2 -- an independent correction to a published result

The paper adopts Endrizal (2025)'s InfoCDM+ dark-energy model and its own best-fit parameters (&Omega;<sub>m</sub>=0.3200, &alpha;=-0.5310, &beta;=0.3920), then plugs them into the deceleration parameter `q(z)` the same paper defines. Endrizal's paper reports a transition-to-acceleration redshift z<sub>t</sub> &asymp; 0.70.

**Independently recomputed here** (not just re-quoted): `q(0.70) = -1 + (1+z)/E(z) * dE/dz` with the stated parameters gives `q(0.70) ≈ +0.112` -- positive, meaning the universe is *still decelerating* at z=0.70. This matches the paper's own claim of `q(0.70) = +0.1116` to three digits. The real transition (`q(z)=0`) is closer to z &asymp; 0.56. This is a genuine, verifiable finding, not an invented number: Endrizal's stated z<sub>t</sub> is not reproducible from Endrizal's own formula and parameters.

An independent fit on real data (Pantheon+ 1624 SNe, 32 cosmic chronometers, 4 BAO points, CMB shift parameter) gives &chi;&sup2;/dof = 0.894 and w(0) = -0.9958 -- compatible with plain &Lambda;CDM. The paper's own conclusion is appropriately modest: *current data do not require InfoCDM+*.

## Level 2.5 -- a no-go theorem for one class of quantum-gravity actions

Modesto's Infinite Derivative Gravity (IDG) class of actions uses entire, zero-free form factors. The paper proves that this structural constraint forces the linearized effective mass density's Fourier transform to be strictly positive for every real momentum -- while the `cosh`-based Hayward regularization this paper wants requires 9 sign changes in that same transform (first zero at k&ell;<sub>0</sub> &asymp; 3.40, computed numerically). The two are incompatible: **no IDG action can generate the cosh regularization**, regardless of which specific entire form factor is chosen. Extending to two independent form factors doesn't rescue it either (RMS residual 3.0&times;10<sup>-2</sup>, best fit degenerates to zero weight on one of them). The logic is sound given the stated premises on the form factors (real, positive, zero-free within a disk of radius &Lambda; in the complex plane) -- this is a real constraint on real quantum-gravity model space, not a straw man.

## Level 3 -- deriving `cosh` from a two-state vacuum

**Postulate** (declared as a postulate, not a derivation): at every scale `r`, the vacuum has two pressure states `P+`/`P-` separated by a gap &Delta;(r), with canonical partition function `Z(r) = e^{-Δ} + e^{+Δ}`.

Two independent arguments then fix &Delta;(r) completely, with no free parameters left over: scale symmetry forces &Delta;(r) to be a pure power of `r/&ell;0`; an entanglement-entropy area-law argument (the same counting that gives `S ~ A/4G ~ r^{d-2}`) fixes the exponent to `n = d-2` (= 2 in d=4). Together: `Z(r) = 2cosh((r/ℓ0)^2)`.

![Statistical derivation: gap, partition function, and regularization length agree across 250 orders of magnitude](assets/dynamic_vacuum_pressure_theory/statistical_derivation.png)

Six independent robustness tests (T7a-T7f: normalization-independence, gap-perturbation stability, uniqueness of the power-law form, entropy-maximization consistency, area-law consistency across d=3..7, joint rescaling invariance) all pass. **One rhetorical overreach worth flagging**: the paper calls this "parameter-free," which is true only *conditional on* the two-state postulate -- the postulate itself is not derived from anything more fundamental, and the paper's own Limits section (below) says exactly that. The two claims sit in some tension; the second one is the accurate one.

## Level 4 -- geometric interpretation (no new math)

Argues the vacuum's lack of an "inside" forces convexity, that `cosh` is the unique convex function balancing two symmetric opposing tendencies, and reinterprets gravity as the memory of which configurations stabilized. This level adds no falsifiable content beyond Level 3 -- it's an interpretive layer, correctly labeled as such by the paper's own summary table.

## Level 5 -- is `cosh` a dynamical attractor?

A genuinely well-designed numerical test (`scripts/vacuum_pressure_level5_attractor.py`, literally titled "honest test" -- *"What is equilibrium? I don't set it by hand. I search for it."*). It defines a persistence functional with a symmetric double-well potential, evolves 5 very different initial conditions (Gaussian, exponential, two-step, noise, double-peak) under gradient descent, and checks whether they all converge to the same shape.

![Level 5: five different initial conditions all converge to sech(r), and the physical double-well potential is the one that does it -- an asymmetric variant produces no attractor at all](assets/dynamic_vacuum_pressure_theory/level5_attractor.png)

They do converge, to `sech(r)` -- correlation 0.9996, confirmed independently correct: `sech` is the known real solution of the reaction-diffusion equation this functional's gradient descent produces (`u'' = u - 2u^3`), a standard soliton/kink result, not a fabricated one. The test discriminates properly: an asymmetric potential variant (`P2`) produces no attractor at all (correlation -0.145, amplitude diverges), and grid convergence (N=200&rarr;2000) confirms the result isn't a discretization artifact. Panel (3) of the same figure asks which exponent `n` the functional itself selects -- it comes out at n&asymp;1, not the n=2 that Level 3's area-law argument requires; the paper is upfront that these are two independently-fixed properties (Level 3 fixes the exponent, Level 5 fixes the base functional form) rather than claiming false agreement.

## Four applications, checked against real data

![Cosmology correction, black-hole shadow vs. EHT, ringdown vs. LIGO, Big Bounce, SLy neutron stars vs. NICER, mass-radius sensitivity, and regularized Coulomb potential](assets/dynamic_vacuum_pressure_theory/applications_summary.png)

- **Black hole shadow**: predicted &theta;(M87*) = 19.85 &mu;as vs. EHT's measured 21.0 &plusmn; 1.5 &mu;as -- 0.77&sigma;, consistent (not a discovery, correctly not oversold as one).
- **LIGO ringdown**: an earlier polynomial-form regularization attempt is excluded at >10&sigma; against GW150914's measured ringdown frequency (251 &plusmn; 3 Hz) -- this is *why* the paper switched to the exponential/cosh form in the first place.
- **Neutron stars**: using the real Douchin & Haensel (2001) SLy equation of state via TOV, the unmodified model reproduces the real M<sub>max</sub>=2.049 M<sub>&#9737;</sub>, R=9.86 km -- and the cosh correction only becomes relevant below &ell;0 &asymp; 2 km, comfortably inside NICER's current 1% precision floor.

![Neutron-star deviation from GR stays inside the NICER 1% bound across the tested ℓ0 range](assets/dynamic_vacuum_pressure_theory/neutron_star_robustness.png)

- **Coulomb potential**: the regularized form is finite at r=0 (`V_eff(0) = -q/ℓ0`), giving the electron a finite self-energy -- a real, standard motivation for this kind of regulator, correctly applied.

**None of these currently produce a falsifiable prediction distinguishable from GR at present observational precision** -- the paper says this outright rather than dressing up consistency as confirmation.

## Part II -- a small, precise side-result: Tao, `cosh`, and the Bell state share one symmetry

A second, shorter paper asks a narrower question: is there a real mathematical structure behind noticing that the yin-yang duality, the `cosh` partition function, and quantum entanglement all "feel" related? The answer it gives is careful, not mystical: **all three instantiate the same Z<sub>2</sub> symmetry group**, and this is checked, not asserted.

![Z2 symmetry is rare in random polygons, Fourier series, and 2-qubit states, but exact by construction for cosh and the Bell state; depolarizing noise breaks it exactly as predicted](assets/dynamic_vacuum_pressure_theory/tao_z2_summary.png)

Independently reproduced here via `dense_evolution` (`scripts/vacuum_pressure_tao_z2.py`):

- `de.DenseSVSimulator(2)` prepares the real Bell state `|Φ+⟩` via `h`+`cx`; `‖X⊗X|Φ+⟩ - |Φ+⟩‖ = 0.0` exactly -- the Bell state genuinely is the +1 eigenstate of `X⊗X`, the defining property of this Z<sub>2</sub> symmetry.
- Under `NoiseModel`'s real depolarizing channel (p=0.1), an exact enumeration of all 16 two-qubit Pauli-error pairs shows 8/16 preserve the symmetry, giving a theoretical preservation probability of 0.8756; 1000 independent noisy trials measured 0.8830 (z=+0.71, consistent).
- In three unrelated random ensembles (random polygons, random truncated Fourier series, random 2-qubit states), the same symmetry appears with probability 0.002, 0.0, and 0.0 respectively -- confirming the symmetry is a real, non-generic structural fact about `cosh` and the Bell state, not something that shows up by chance.

**What this explicitly is not**, in the paper's own words: not "the Tao is entanglement" (rejected directly as empty metaphysics), not proof the physical vacuum must have this structure (Postulate 1 stays a postulate), and not a claim that Z<sub>2</sub> is the only possible structure (Z<sub>3</sub>, non-abelian groups remain open). What it is: three independent domains -- Taoist philosophy, statistical mechanics, quantum information -- share one identical, rare, and now-verified algebraic structure, stated at exactly the scope the evidence supports.

## Limits (the paper's own, kept here verbatim in substance)

- The two-state vacuum postulate (Postulate 1) is not derived from anything more fundamental -- it is an assumption, not a theorem, despite Level 3's "parameter-free" framing once that assumption is granted.
- The connection between the regularization scale &ell;0 across different sectors (black holes, cosmology, neutron stars, Coulomb) is not derived from one unifying principle.
- The "independent bubbles" and "probability as a scale criterion" conjectures (Level 4) are stated, not formalized mathematically.
- No application currently produces a prediction falsifiable at present observational precision.

## Where this stands

The Level 3/2.5/5 mathematical core, and the Part II Z<sub>2</sub> result, hold up under independent re-checking -- both the InfoCDM+ correction and the Bell-state symmetry claim were reproduced here from scratch, not taken on faith. The Level 0-1 philosophical framing remains exactly that: an interpretive scaffold around the math, not itself a tested claim. Promotion to `dense_evolution` isn't applicable here (this isn't a simulator primitive) -- this stays a documented Discovery research note.

## References

1. S. Pennacchio, *La Teoria 4,5 della Pressione Dinamica del Vuoto*, 2026 (self-archived).
2. S. Pennacchio, *La Teoria 4,5 della Pressione Dinamica del Vuoto — Parte II: La Struttura Z2*, 2026 (self-archived).
3. J. Endrizal, *Information-Driven Late-Time Cosmology: InfoCDM/InfoCDM+*, Zenodo, 2025, DOI:10.5281/zenodo.17771328.
4. S. A. Hayward, *Formation and evaporation of non-singular black holes*, Phys. Rev. Lett. 96, 031103 (2006).
5. F. Douchin & P. Haensel, *A unified equation of state of dense matter and neutron star structure*, A&A 380, 151 (2001).
6. L. Modesto, *Super-renormalizable Quantum Gravity*, Phys. Rev. D 86, 044005 (2012).
7. Event Horizon Telescope Collaboration, *First M87 EHT Results I*, ApJL 875, L1 (2019); *First Sgr A* EHT Results I*, ApJL 930, L12 (2022).
8. B. P. Abbott et al. (LIGO/Virgo), *Observation of Gravitational Waves from a Binary Black Hole Merger*, PRL 116, 061102 (2016).
9. M. C. Miller et al., NICER mass/radius results for PSR J0030+0451 (ApJL 887, L24, 2019) and PSR J0740+6620 (ApJL 918, L28, 2021).
