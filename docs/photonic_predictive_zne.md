# Photonic Predictive Zero-Noise Extrapolation

!!! note
    The general-purpose `zne_density_matrix` and the promoted
    `jsd_predictive_zne_density_matrix` live in the main library:
    [`dense_evolution.mitigation`](https://tatopenn-cell.github.io/Dense-Evolution/)
    (shipped in `dense-evolution>=8.1.56`). This page is the experimental
    log for how that function was designed, validated, and honestly
    compared against postselection before being promoted there.

**In plain terms**: photonic (light-based) quantum computers lose photons as errors, and a common error-mitigation trick called Zero-Noise Extrapolation (ZNE) tries to guess the noise-free answer by extrapolating from several deliberately-noisier runs. A recent paper found this trick doesn't actually help with photon loss. This page builds a working density-matrix version of ZNE for this case, and honestly compares it against the simpler alternative -- just throwing away runs where a photon was lost.

Does zero-noise extrapolation actually help with photon loss in photonic
quantum computing? Mills & Mezher, "Mitigating photon loss in linear
optical quantum circuits" (**arXiv:2405.02278**), found that plain scalar
ZNE does **not** beat simple postselection for discrete-variable photon
loss -- Richardson/Vandermonde-style extrapolation amplifies statistical
sampling noise faster than its theoretical unbiasedness helps. This page
reproduces that finding directly, builds on Dense-Evolution's existing
density-matrix ZNE extension (which avoids that specific failure mode by
construction), and adds a new adaptive variant -- then honestly checks
whether any of it actually beats postselection.

## Part 1: reproducing the paper's warning, and where density-matrix ZNE helps

Photon loss on a dual-rail-encoded photonic qubit is exactly
Dense-Evolution's `amplitude_damping` channel (a photon "leaking" out of
a mode is the same K0/K1 Kraus pair as a qubit decaying |1⟩→|0⟩). A real
run (Bell state, 16-point transmissivity sweep, K=200 trajectories per
point) confirms the paper's concern concretely: **scalar ZNE goes
physically impossible (fidelity > 1.0) at 14/16 points.**
`zne_density_matrix` (projects the extrapolated result back onto the
nearest physical state) never does -- and gives a real, substantial
correction: mean fidelity delta **+0.086**, 15/16 positive.

[![Photonic predictive ZNE: raw vs. scalar vs. density-matrix vs. JSD-predictive ZNE](assets/photonic_predictive_zne/photonic_predictive_zne.png)](assets/photonic_predictive_zne/photonic_predictive_zne.png)

## Part 2: a self-contained adaptive signal (Jensen-Shannon divergence)

`dense_evolution.mitigation`'s existing "predictive/healing" coefficient
adaptation (`calculate_delta_preemp`) previously existed only on the
scalar ZNE path -- combining it with density-matrix ZNE for the first
time, fed the true photon-loss rate as its signal, gave essentially no
improvement (+0.000005 mean): its fixed nudge constants (0.01/0.02) were
tuned for a differently-scaled use case elsewhere in the library.

A second design uses the Jensen-Shannon divergence between measured
output distributions at consecutive noise scales as the signal instead
-- needs no external calibration or oracle access to the ideal state. An
unrectified first version helped in only 5/16 points despite the signal
itself correlating significantly with success (Pearson r=+0.533,
p=0.0334) -- the fix was **rectifying** the nudge to fire only when the
signal is positive (reducing exactly to plain `zne_density_matrix`
otherwise, verified to ~1e-8, zero risk in that regime), not discarding
the signal.

Validated on a real, seed-diverse sample (72 points: 12 photon-loss
rates × 6 independent seeds, K=200) before promoting it to the library:
among 46 active points, **76.1% improve**, mean fidelity gain **+0.0055**,
one-sample t-test **p=0.0003**, positive in **6/6** independent seeds --
the win rate and effect size were *larger* on the big sample than the
small one that first suggested it, the opposite of the usual
small-sample-regresses-to-null pattern seen elsewhere in this repo's own
experiments.

## Part 3: the honest comparison -- against true postselection, multiple circuits

Everything above compared against *scalar* ZNE, which the paper already
flags as weak. The comparison that actually answers the paper's question
is against **postselection** -- and that comparison was missing until
this pass. `_apply_amplitude_damping_tracked` independently reimplements
`NoiseModel.apply_to_sv`'s Born-rule-correct formula, additionally
tracking whether *any* qubit's decay Kraus branch fired per shot --
verified bit-exact (0.00e+00 diff) against the real library across
`n_qubits` in (2, 3, 4), not assumed from matching formulas alone (a
first version was missing the library's own final global
renormalization step, caught by a direct ~0.18 discrepancy before the
fix). Postselection here means averaging only the shots where no loss
was heralded anywhere -- the real physical meaning of "keep only
heralded-successful shots," not an approximation.

Tested across two circuit families (GHZ states, and a hardware-efficient
VQE-style ansatz -- chosen because VQE is the primary real-world
application of near-term quantum simulation) at `n_qubits` in (2, 3, 4),
18 configurations total:

[![Multi-circuit postselection comparison: GHZ and VQE-style circuits at 2-4 qubits](assets/photonic_predictive_zne/photonic_multi_circuit_postselection.png)](assets/photonic_predictive_zne/photonic_multi_circuit_postselection.png)

**Postselection wins in 14/18 configurations.** Mean gap: JSD ZNE trails
postselection by -0.0345, plain density-matrix ZNE by -0.0362 -- the JSD
nudge narrows the gap slightly, it does not close it. Every loss rate at
the highest tested (η=0.7) favors postselection, several by a wide
margin (up to -0.15 for the 4-qubit VQE circuit).

## Honest conclusion

The photon-loss/density-matrix-ZNE connection is real and now
empirically validated against real literature, and the JSD-informed
adaptive variant is a genuine, seed-confirmed improvement over plain
density-matrix ZNE. **Neither approach generally beats postselection**
when postselection is a viable option (loss events independently
heralded, as they typically are in real linear-optical photonic
hardware). `jsd_predictive_zne_density_matrix` was still promoted to the
main library -- with this exact limitation documented directly in its
own changelog entry, not glossed over -- because it is useful when
postselection genuinely is *not* viable: loss events that aren't
independently heralded, or a lost-shot fraction too high to discard
without wasting an unacceptable amount of measurement budget.

## Reproducing this

```bash
python scripts/photonic_predictive_zne.py
python scripts/photonic_zne_multi_circuit_postselection.py
```

Real data: [`data/photonic_predictive_zne.csv`](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/blob/main/data/photonic_predictive_zne.csv)-equivalent (generated locally, `/data/` is gitignored -- re-run the scripts above to reproduce). Literature grounding this page cites is verified and indexed locally in `quantumrag`'s `fotonica_quantistica` collection (Mills & Mezher arXiv:2405.02278; Borzenkova et al. arXiv:2311.13985; Somhorst et al. arXiv:2601.05947; a broader photonic-QML noise survey, arXiv:2603.09645).
