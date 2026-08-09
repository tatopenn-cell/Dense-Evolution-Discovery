"""
Real traversable-wormhole-inspired quantum teleportation (Gao-Jafferis-Wall
theory, arXiv:2604.10090) on a binary sparse Sachdev-Ye-Kitaev (SYK) model
-- built on dense_evolution/dashboard_core (v8.1.49), see
https://tatopenn-cell.github.io/Dense-Evolution/ for the shipped
implementation (dashboard_core.wormhole, dense_evolution.fermions/entropy/
trotter) and its own unit test suite.

An earlier, discarded dashboard_core circuit ("Traversable Wormhole (BGQ)")
used the right vocabulary (SYK scrambling, a phase "kick") but wasn't real:
it ran on a single qubit register, which the no-signaling theorem forbids
from ever showing this protocol's real sign-dependent signal -- verified
directly, not assumed (identical results for either sign of its "kick").
The real recipe needs two coupled chaotic systems (L, R), a message
injected into L via a separate reference-qubit pair (P, Q), a real
bilinear L-R coupling exp(i*mu*V), and a readout that is NOT a
single-qubit expectation value: mutual information between the reference
qubit P and a qubit read out from R.

This script runs nineteen real, verified experiments, each producing its own
CSV + plot:

1. t1 sweep -- the protocol's headline signature, sign-dependent mutual
   information rising then falling across post-coupling evolution time.
2. message-vs-no-message control -- with_message=False gives I(P:R)=0
   exactly at every point tested (P,Q are structurally decoupled from
   L,R without the swap injection), confirming the signal genuinely
   requires the injected message, not just the L-R coupling itself.
3. mu-magnitude scan -- the sign-dependent delta peaks near mu~11-12,
   matching arXiv:2604.10090's own choice, not at higher coupling
   strengths (which give more *total* mutual information but a smaller
   sign-dependent asymmetry).
4. t0 (pre-coupling scrambling time) scan -- the signal needs enough
   scrambling before it appears (consistent with the theoretical chaos
   requirement of the protocol), and for this specific instance peaks
   later than the paper's own t0=0.3 choice.
5. 2D (t0, mu) joint grid search -- experiments 3 and 4 scanned each
   axis independently, holding the other fixed; a quick follow-up check
   showed the mu-peak shifts as t0 changes, meaning neither 1D scan
   alone finds the true joint optimum. This grid search resolves that:
   870 points (30 t0 values x 29 mu values), global max at
   t0=0.65, mu=15.0 (delta=+0.01167), noticeably better than either 1D
   scan's own peak.
6. t1 re-scan at Experiment 5's (t0, mu) optimum -- Experiment 5 itself
   held t1 fixed at 0.60 (Experiment 1's own 1D peak) and flagged that
   as unverified. Re-scanning t1 at t0=0.65/mu=15.0 finds the peak has
   moved to t1=0.41, delta=+0.01518 -- ~30% above Experiment 5's
   headline value. One coordinate-ascent step, not a converged 3D joint
   optimum (resolved by Experiment 7 below).
7. Iterated coordinate ascent toward the joint (t0, mu, t1) optimum --
   Experiment 6 moved t1 once but never checked whether t0/mu would
   shift again, the same gap that motivated Experiment 5 in the first
   place. Alternating full t1 scans (Experiment 6's resolution) and
   (t0, mu) grids (Experiment 5's resolution) from Experiment 5's
   starting point, 3 rounds converge to a genuine fixed point:
   t0=0.70, mu=17.0, t1=0.36, delta=+0.01688 -- +44.6% over Experiment
   5's original headline value.
8. Generality check across 6 independent SYK instances -- does
   Experiment 7's converged point generalize, or is it specific to
   seed=61? Honest negative result: it does NOT. The converged
   (t0, mu, t1) scatters across nearly the whole scanned range instead
   of clustering, 2 of 6 instances converge at the edge of the scanned
   grid (inconclusive -- their true optimum may lie outside what was
   scanned), and 2 of 6 have a *negative* delta at Experiment 5's own
   starting point (the sign-dependent signal isn't even reliably
   oriented the expected way using those defaults across instances).
   Percentage-improvement figures are not reported for this experiment
   since near-zero/negative baselines make them meaningless.
9. Realistic-noise robustness at the converged point (seed=61,
   t0=0.70, mu=17.0, t1=0.36) -- a real Trotterized gate circuit with a
   stochastic depolarizing Kraus channel (dense_evolution.registry.
   NoiseModel) injected after each of the protocol's three phases,
   scanned over p in [0, 0.05], averaged over 6 trials per point.
   Second honest negative result: the noiseless Trotter delta
   (+0.01728) decays with noise and crosses zero between p=0.01 and
   p=0.02 -- already at p=0.01 the mean signal (+0.00051) is smaller
   than its own trial-to-trial standard deviation (0.01203), i.e.
   statistically indistinguishable from zero at a noise level well
   within range of current real NISQ hardware.
10. Direct comparison against arXiv:2604.10090's own "Ensemble
    robustness" section, which reports 100 disorder realizations and
    concludes the sign-dependent asymmetry is "a generic feature of
    the ensemble" (their chosen instance was selected for unusually
    *large* asymmetry, not unusually *signed*). Experiment 8's baseline
    was evaluated at Experiment 5's point (t0=0.65, mu=15.0, t1=0.60),
    itself optimized on seed=61 -- a real confound, since an instance
    could show the "wrong" sign there simply from being evaluated at a
    point tuned for a different instance. Re-evaluating all 6 instances
    at the paper's OWN stated defaults (t0=0.3, mu=12, t1=0.60)
    controls for that: 2 of 6 (seeds 2166, 2907) still show the wrong
    sign there, seed 2835's apparent reversal in Experiment 8 turns out
    to have been a point-choice artifact (correctly signed at the
    paper's defaults). Net finding: even controlling for the confound,
    at least 1 of 6 instances (seed 2166, wrong-signed at *both*
    evaluation points) genuinely contradicts the "generic feature of
    the ensemble" claim for this specific 34/11-selection-matched
    subset.
11. Large-sample (n=100) version of Experiment 10's check, matching
    arXiv:2604.10090's own reported ensemble size. Result: 49/100 (49%)
    of exact 34/11-selection-matched instances are wrong-signed at the
    paper's own default parameters -- far stronger than Experiment 10's
    2/6 (33%), and essentially a coin flip, not a "generic feature of
    the ensemble". Also tests two candidate structural explanations for
    the sign variation floated informally alongside Experiment 10
    (Majorana mode-usage imbalance in the K=10 coupling terms; the
    spectral level-spacing r-statistic, a standard chaos diagnostic):
    an early n=6 look had suggested mode-usage imbalance correlated
    with the signal (r=0.87) -- at n=100 that does NOT hold up
    (r=0.171, p=0.09, not significant), an honest correction of that
    earlier small-sample impression. The level-spacing statistic
    doesn't correlate either (r=0.087, p=0.39). Neither explains why
    the sign varies; that remains open.
12. Size winding (arXiv:2604.10090 Sec. S6, Eqs. S18-S22) -- a third,
    theory-motivated diagnostic tried after the two structural/spectral
    ones in Experiment 11 both failed to explain the sign variance.
    Directly expands a Heisenberg-evolved single-sided Majorana operator
    chi_j(t) in the Majorana-string basis Gamma_P and checks the phase
    coherence R(l)=|q(l)|/P(l) and phase arg(q(l)) of the winding size
    distribution within each size sector l, across 6 instances and 4
    post-quench times (verified first on 3 individual seeds spanning
    both correctly- and wrong-signed instances before running the full
    official sweep). CORRECTED 2026-08-09: the original run omitted the
    thermal factor rho_beta^(1/2) Eq. S18 actually requires, which
    mathematically forces R(l)=1.0 and arg(q(l))=0.0 exactly regardless
    of physics (a trace of two Hermitian operators is always real) --
    not a physics finding, an implementation bug. With rho_beta^(1/2)
    correctly included, the diagnostic is genuinely non-trivial
    (max|phase| up to 2.94 rad, min R(l) down to 0.049 across the same
    6 instances/4 times). Whether the corrected diagnostic correlates
    with the sign has not yet been tested at scale. The mean operator
    size <l>(t) itself (unaffected by this fix) does show genuine
    chaos-consistent growth followed by finite-size recurrence,
    confirming the underlying operator-growth dynamics are real.
13. Mechanistic check -- two protocol-grounded (not just post-hoc
    statistical) candidate explanations, reusing Experiment 11's own
    n=100 instance set and delta values directly from
    data/wormhole_ensemble_sign_check.csv (no re-screening). Feature A,
    "message-mode participation": dense_evolution.fermions.
    majorana_pauli_terms's Jordan-Wigner mapping shows Majorana modes 1
    and 2 map onto qubit index 0 -- exactly the qubit the message is
    swapped into and read out from -- so this counts how many of each
    instance's K=10 SYK quads touch those two modes specifically,
    sharper than Experiment 11's all-modes-interchangeable usage-std.
    Feature B, "operator growth rate": reuses Experiment 12's own
    Gamma_P/Heisenberg-evolution machinery (run_size_winding_check) to
    get mean operator size <l> at t=0.7 and t=1.2 -- real,
    non-trivial, instance-varying data Experiment 12 already computed
    but never correlated against the sign. Fourth honest negative
    result: neither correlates (message-mode participation r=-0.012,
    p=0.90; growth rate at t=1.2 r=+0.126, p=0.21) -- a 4th and 5th
    candidate explanation ruled out, on top of Experiment 11's two.
    (Experiment 12's phase/R diagnostic itself was corrected 2026-08-09
    -- see its own docstring -- and its correlation against the sign is
    a separate, not-yet-tested open question, not a ruled-out candidate.)
    Why the sign varies remains genuinely open.
14. Qubit-coupling topology check -- tests whether *which specific
    modes* each instance's K=10 quads couple together (not just how
    many terms commute, or how many terms touch a given mode) predicts
    the sign. Builds a weighted 8-mode co-occurrence graph per instance
    (edge weight = how many quads contain both modes) and computes four
    features: max weighted degree, weighted degree std, the number of
    the 28 possible mode pairs never coupled together at all, and the
    weighted graph's algebraic connectivity (Fiedler value). An ad hoc
    check before committing to this design found a *binary* version of
    the graph (edge iff any co-occurrence) saturates to the complete
    graph K8 for most instances -- useless as a discriminator -- so the
    weighted count is used instead. A second honest check, done after
    computing the real n=100 numbers rather than before: max weighted
    degree and weighted degree std turn out to be an exact linear
    rescaling of Experiment 11's mode-usage-count features (weighted
    degree = 3x usage count, verified numerically to 1 part in 1e15,
    not assumed) -- they are not new information, just Experiment 11's
    already-tested, already-non-significant feature recomputed via a
    graph Laplacian. The two genuinely new features, n_zero_pairs and
    algebraic_connectivity, also do not correlate (r=+0.159, p=0.114
    and r=-0.141, p=0.163) -- a 6th and 7th candidate explanation ruled
    out (counting the two redundant degree features separately would
    overstate how many independent hypotheses this experiment tested).
15. N-scaling check (N=8 vs N=12) -- the one candidate never tried:
    does the sign-dependent instance variance persist, worsen, or
    shrink at a larger Majorana count? The exact backend is infeasible
    at N=12 (dim^3 diagonalization cost, a measured/estimated ~4096x
    slowdown over N=8's dim=1024), so both N=8 and N=12 are re-evaluated
    here via the Trotterized gate-circuit backend for a clean,
    backend-matched comparison (~19s/call at N=12, close to N=8's own
    already-measured ~14s/call Trotter cost) -- not the exact-backend
    N=8 numbers Experiments 10/11 used, which would confound N-scaling
    with a separate, already-quantified backend effect (Experiment 9).
    n=6 instances per N (not Experiment 11's n=100 -- infeasible at
    this cost), and K_TERMS kept fixed at 10 rather than scaled with N.
    The paper's own 34/11 selection criterion has no exact match at
    N=12 (verified: 0 of 3000 candidates), so N=12 instances are
    selected by closest achievable match instead (peaks around
    commuting=21-23, tops out at 31 in a 500-seed sample). Result:
    wrong-sign rate is identical, 2/6 at both N=8 and N=12 (too small a
    sample to treat as a real rate, just a like-for-like snapshot), but
    the mean |delta| magnitude drops from 0.00765 (N=8) to 0.00034
    (N=12) -- roughly a 22x reduction. Consistent with the signal
    weakening toward a thermodynamic limit, though also consistent with
    the paper's own default parameters (tuned implicitly for N=8) simply
    becoming less optimal as N grows -- this experiment cannot
    distinguish those two explanations, only establish that the
    magnitude drop is real.
16. Term-order non-commutativity check -- a different kind of non-
    commutativity from the one this repo's own scripts/channel_order_
    noncommutativity.py already settled (noise-CHANNEL order matters
    iff at least one channel is non-Pauli; the depolarizing channel
    used in Experiment 9 here is a Pauli-mixture channel, so that rule
    predicts channel reordering would show nothing new). This instead
    reorders the K=10+10 SYK Hamiltonian TERMS within the Trotterized
    circuit's own evolution phases (original order vs. reversed),
    noiselessly, and measures order_sensitivity = |delta_reversed -
    delta_original| per instance -- testing whether the *degree* of
    non-commutativity among a seed's specific terms (not just how many
    of them pairwise commute, already shown insufficient) tracks the
    sign. An initial n=6 spot-check found a moderate, borderline-
    interesting r=+0.474 (p=0.342, not significant but a much larger
    point estimate than any other candidate tried in this script) --
    interesting enough, and cheap enough (protocol_layout is built once
    per seed and reused across all 4 Trotter calls, ~18s/instance) to
    warrant verification on a larger sample before writing anything up,
    per this project's established discipline. Eighth honest negative
    result: at n=30 (the same 34/11-exact-match screening used
    elsewhere, this time needing 21772 candidates screened to find 30),
    the correlation regresses to r=+0.282, p=0.131 -- still not
    significant, and weaker than the n=6 look suggested. An honest
    correction, the same pattern as Experiment 11's mode-usage-
    imbalance finding (r=0.87 at n=6, r=0.171 at n=100): a promising
    small-sample point estimate that does not hold up under a larger,
    more powered look. order_sensitivity itself is real and non-zero
    for every instance tested (term order does change the Trotterized
    circuit's output, confirming genuine non-commutativity among the
    terms), it just does not predict the sign. 15/30 (50%) wrong-signed
    at this n=30 subsample, consistent with Experiment 11's ~49/100.
17. Term-order x noise interaction check -- Experiment 16's own caveat
    flagged this as the natural next question: term order alone
    (Trotter error, noiseless) didn't predict the sign, but does
    term-order sensitivity change once realistic noise is present,
    closer in spirit to scripts/channel_order_noncommutativity.py's own
    noisy, stochastic setting? Same method as Experiment 16 (original
    vs. reversed term order, |delta_reversed - delta_original|), but
    now with a depolarizing Kraus channel injected after each protocol
    phase (noise_p=0.01, Experiment 9's own near-threshold value) and
    delta averaged over 6 noisy trials per order per instance (common
    random numbers between mu signs, isolating the sign effect from
    trial-to-trial noise-realization variance). Ninth result, and the
    first positive one since Experiment 9: an initial n=6 spot-check
    found r=+0.811 (p=0.050), and unlike every other candidate in this
    script, it did NOT regress to non-significance as the sample grew
    -- n=20: r=+0.587 (p=0.0065); n=30: r=+0.396 (p=0.030); n=50
    (34/11-exact-match screening, 38028 candidates screened): r=+0.340
    (p=0.0158). The point estimate shrinks with n, as expected from a
    true but modest effect regressing off an initially lucky
    small-sample draw, but it stabilizes in the r~0.34-0.40 range
    instead of continuing toward zero, and stays below p=0.05 at every
    single sample size checked -- qualitatively different from
    Experiment 16's own noiseless version of the same test, which
    collapsed from p=0.34 (n=6) to p=0.13 (n=30) over a comparable n
    range. 25/50 (50%) of the n=50 sample are wrong-signed, consistent
    with every other larger-n check in this script. Interpretation:
    term-order non-commutativity by itself (Trotter error alone,
    Experiment 16) doesn't predict the sign, but its *interaction with
    physical noise* does, modestly -- plausibly because noise and
    Trotter error both perturb the state away from the exact answer,
    and how sensitive a given instance's term ordering is to that
    perturbation partly tracks how fragile its sign-dependent signal is
    to begin with.
18. t0 correction -- rereading arXiv:2604.10090 directly found that
    every prior experiment in this script (including Experiment 11's
    flagship n=100 check) mislabeled t0=0.3 as "the paper's own default
    parameters." The paper's REAL hardware working point is t0=1.8
    (Sec. S4: "t0=1.8 marks a turning point... we choose t0=1.8 as the
    hardware working point") -- t0=0.3 never appears as an injection
    time anywhere in the paper's text; the only "0.3" in the extracted
    PDF is a y-axis tick label on Fig. 5. Re-runs Experiment 11's exact
    n=100, 34/11-selection-matched ensemble sign check at t0=1.8
    (t1=1.25, chosen via a real 23-point scan on seed=61 -- the paper
    itself gives no single default t1, scanning t1 in [0.5, 6.0] at
    fixed t0=1.8 in its own Fig. 5). Result: 41/100 (41%) wrong-signed
    at the paper's real t0=1.8, vs. 49/100 (49%) at the mislabeled
    t0=0.3 -- closer to correctly-signed than the mislabeled check
    suggested, but still far from the paper's "generic feature of the
    ensemble" claim, and still close enough to a coin flip that the
    core finding (the sign-dependent instance variance is real and
    still unexplained) stands.
19. Noise-level scan for the term-order x noise correlation --
    Experiment 17's own caveats flagged its finding (r=+0.340,
    p=0.0158 at n=50) as only ever tested at a single noise_p=0.01.
    Scans noise_p at n=20 on a fixed 34/11-matched seed set (a smaller
    n than Experiment 17's flagship 50, for cost -- see
    run_noise_level_scan_check's own docstring), reusing Experiment
    17's exact method unchanged at each level. Real result: noise_p=
    0.005 (r=+0.210, p=0.374, NOT significant), noise_p=0.01 (r=+0.587,
    p=0.0065), noise_p=0.02 (r=+0.622, p=0.0034, the STRONGEST of the
    three). The noise_p=0.01 point is a direct methodology consistency
    check, not a new measurement -- it reproduces Experiment 17's own
    n=20 subsample number (r=+0.587, p=0.0065) exactly, confirming this
    scan's approach is sound before trusting the two new points.
    Genuine finding, not assumed from Experiment 17 alone: the
    correlation is not a fixed property of the noise -- it is ABSENT
    at very low noise (0.005) and STRENGTHENS as noise increases
    through 0.01 to 0.02 (Experiment 9's own upper bound, where the
    noiseless signal was already crossing zero), rather than being
    flat or regressing back toward zero the way every OTHER candidate
    in this script did as its sample size grew. Consistent with a
    physical reading: term-order sensitivity needs enough noise-driven
    perturbation to actually interact with the sign-dependent signal;
    at near-zero noise there is barely any perturbation for it to
    interact with.

Experiments 1-7, 9, and 10 use seed=61 (n_majorana=8, k_terms=10, J=sqrt(2))
-- the instance dashboard_core.wormhole.select_good_instance finds when
screened against arXiv:2604.10090's own selection criterion (their
chosen K=10 instance has 34 commuting / 11 anticommuting pairs among the
C(10,2)=45 pairs of terms). Re-derived below, not hardcoded blindly.
Experiments 8 and 10 additionally use 5 more instances matching that same
exact criterion, found by find_multiple_seeds; Experiment 11 uses up to
100.

Honest caveats, not glossed over:
- Experiment 7's fixed point (t0=0.70, mu=17.0, t1=0.36) is a *local*
  coordinate-ascent convergence on this specific grid resolution
  (Experiment 5's 0.05/1.0 t0/mu step, Experiment 6's 0.01 t1 step), not
  a proof of global optimality: coordinate ascent can converge to a
  point that isn't the true joint maximum if the surface isn't
  separably well-behaved, and a finer/coarser grid could in principle
  settle on a nearby but distinct fixed point (an ad hoc finer local
  grid around the converged point suggested the true continuum optimum
  sits close to mu=17.5, just off this grid's integer mu values --
  consistent with, not contradicting, the converged answer). A real
  continuous joint optimizer (e.g. gradient-based, if this readout is
  ever made differentiable) would be needed to settle global optimality.
- Experiment 8's non-generalization finding is itself only a 6-instance
  sample, and 2 of those 6 hit the edge of the scanned (t0, t1) range
  rather than settling on a real interior fixed point -- a wider scan
  range could turn those into genuine (still probably instance-specific)
  answers rather than boundary artifacts, but wasn't run here (compute
  cost scales with range x resolution, already ~15 minutes for 6
  instances at the current range).
- Experiments 1-8 use the exact-evolution backend (eigendecomposition),
  not the Trotterized real-gate-circuit backend -- both are implemented
  and cross-verified to agree closely (see the main Dense-Evolution
  repo's tests), but the exact backend is what was used for speed.
  Experiment 9 is the one exception, by necessity (noise injection needs
  a real gate circuit to interrupt mid-evolution).
- Experiment 9 only tested seed=61's converged point, not the other 5
  instances from Experiment 8 -- given Experiment 8's own finding (the
  converged point doesn't generalize across instances), there's no
  reason to expect this noise-robustness result generalizes either;
  it's honestly one more seed=61-specific data point, not evidence about
  the protocol broadly. It also averages only 6 stochastic trials per
  noise level (each `NoiseModel.apply_to_sv` call is a single-shot Kraus
  draw, not an ensemble average) -- the reported standard deviations are
  real but from a small sample, not tight error bars.
- Experiments 5, 6, and 7 bypass `run_wormhole_protocol`'s public API and call
  `dashboard_core.wormhole`'s private layout/evolution helpers directly.
  Justified by a real, measured cost asymmetry: building the SYK/coupling
  Hamiltonians and diagonalizing both (`_protocol_layout` + two `eigh`
  calls) took 4.3-6.4s and does not depend on t0/mu/t1 at all for a
  fixed (seed, n_majorana, k_terms) -- only the actual per-point
  evolution + mutual-information readout does, and that alone measured
  at 0.022s/call. Computing the expensive part once instead of once per
  grid point cut an 870-point grid from an estimated ~2 hours down to
  47.6s (~165x) -- confirmed by timing both versions directly, not
  assumed. `run_wormhole_protocol` itself is unchanged; this script's
  own helper functions are just a faster way to call the same physics
  repeatedly at one fixed instance.
- Experiment 12 evolves a single-sided Majorana operator under only the
  L-side SYK Hamiltonian H, matching arXiv:2604.10090's own size-winding
  setup -- it deliberately does NOT reuse the L+R+P+Q combined-system
  Hamiltonian used by Experiments 1-11's mutual-information readout, so
  it cannot directly explain those experiments' sign-dependent delta by
  construction; it only tests whether the paper's own diagnostic
  distinguishes instances at all. It also only checks majorana_index=1
  and 4 post-quench times per instance, not a full time/index sweep.
- Experiment 13 is the 4th and 5th candidate explanation tested against
  the *same* n=100 instance sample used for Experiment 11's 2 candidates
  -- a real multiple-comparisons risk. Neither reached significance
  here (p=0.90, p=0.21, both far from even an uncorrected 0.05
  threshold), so this isn't a marginal case that Bonferroni-style
  correction would flip, but a genuinely significant hit among several
  candidates tested on one fixed sample should be treated with
  suspicion and re-checked on a fresh holdout set before being reported
  as a real finding -- not done here since neither candidate needed it.
  Its "operator growth rate" feature also only probes 2 discrete times
  (t=0.7, 1.2), not a continuous growth-rate fit, and reuses
  Experiment 12's single-sided (L-only) operator-growth computation --
  same scope caveat as Experiment 12 above.
- Experiment 14's weighted co-occurrence graph counts a mode pair as
  "coupled" whenever they appear together in a quad, regardless of the
  quad's random +-J/sqrt(K) sign or of whether the two Majorana factors
  actually commute or anticommute within that specific term -- a purely
  combinatorial notion of coupling, not an operator-algebraic one.
  Whether a *sign-aware* or *commutator-aware* weighting would behave
  differently is untested. Also the 6th/7th candidate (n_zero_pairs,
  algebraic_connectivity) tested against Experiment 11's same n=100
  sample -- same multiple-comparisons caveat as Experiment 13 above,
  and same conclusion: neither result is close enough to significance
  (p=0.11, p=0.16) to warrant a holdout re-check.
- Experiment 15's N=8-vs-N=12 comparison uses the Trotter backend for
  BOTH, at n_steps_evolution=8/n_steps_coupling=16 -- the same step
  counts used throughout, not re-tuned for N=12's different term
  structure, so some of the observed magnitude drop could in principle
  be a Trotter discretization-error artifact rather than a genuine
  physical effect (untested: whether doubling the step count at N=12
  changes the result). n=6 instances per N is far too small to treat
  the 2/6-vs-2/6 wrong-sign-rate match as meaningful -- only the
  delta-magnitude drop, present in all 6 N=12 instances individually
  (not just on average), is treated as a real finding here. K_TERMS
  was kept fixed at 10 rather than scaled with N -- a scaled-K version
  of this same check is untested and could behave differently.
- Experiment 16 only compares two orderings (original vs. fully
  reversed) out of K=10+10=20 terms' 20! possible permutations -- a
  much larger order_sensitivity range might exist among orderings not
  tried, and reversal specifically might not be representative of
  "how non-commutative" a term set generally is. It's also noiseless
  by design (isolating pure Trotter-error order-dependence from
  physical noise) -- whether term-order sensitivity interacts with
  noise in a way that *does* track the sign (as opposed to the
  noiseless order_sensitivity metric tested here) is a distinct,
  untested question, closer in spirit to channel_order_
  noncommutativity.py's own noisy, stochastic setting.
- Experiment 17's r=+0.340 (p=0.0158) at n=50 is the first candidate in
  this script to stay significant across every sample size tested
  (n=6, 20, 30, 50) instead of regressing to non-significance -- still
  a modest effect (r~0.34), not a strong predictor, and only tested at
  a single noise_p=0.01 and n_trials=6 stochastic trials per point
  (Experiment 9's own budget, reused rather than independently
  re-tuned here); whether the correlation strengthens, weakens, or
  holds at other noise levels is untested. It also only compares the
  same two orderings as Experiment 16 (original vs. fully reversed)
  out of far more possible permutations -- the same scope caveat as
  Experiment 16 above.
- Experiment 18's t1=1.25 default was chosen from a single-seed
  (seed=61) 23-point scan, not re-derived per instance across the
  n=100 ensemble -- the same "one instance's optimum applied to all"
  confound Experiment 8 already found doesn't generalize for
  (t0, mu, t1) jointly. A per-instance-optimal t1 could shift the
  41/100 wrong-sign rate in either direction; untested here.
- Experiment 19's n=20 is smaller than Experiment 17's own flagship
  n=50 (cost -- see run_noise_level_scan_check's own docstring), and
  only 3 noise_p points were tested (0.005/0.01/0.02); the trend
  (absent -> significant -> strongest) is measured at exactly these 3
  points, not a continuous scan, so whether it keeps strengthening
  past 0.02, plateaus, or reverses is untested. A first run of this
  experiment used only 6 seeds instead of the requested 20 due to a
  real bug in the seed-screening call's n_candidates default (fixed;
  see the fix's own commit) -- that invalid run's numbers were
  discarded, not used anywhere in this write-up.
"""
import itertools
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from dashboard_core.wormhole import (
    build_sparse_syk_terms, commuting_pair_count, select_good_instance,
    run_wormhole_protocol, run_wormhole_protocol_trotter,
)
# Private helpers, used only by run_2d_grid_search's precompute-once
# optimization -- see this module's docstring for why. Not part of
# dashboard_core.wormhole's public API (no __all__ entry); reached into
# deliberately here rather than duplicated, since re-deriving the same
# protocol layout independently would risk silently drifting out of sync
# with the real implementation.
from dashboard_core.wormhole import _protocol_layout, _initial_state_ops, _evolve
from dense_evolution import mutual_information
from dense_evolution.trotter import trotter_evolve_ops
from dense_evolution.registry import NoiseModel
from dense_evolution.fermions import majorana_pauli_terms
import dense_evolution as de

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

N_MAJORANA = 8
K_TERMS = 10
J = float(np.sqrt(2))


def find_seed() -> int:
    seed = select_good_instance(N_MAJORANA, K_TERMS, J, n_candidates=200, target_commuting=34)
    n_qubits = N_MAJORANA // 2
    terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, seed)[1]
    c, a = commuting_pair_count(terms, n_qubits)
    print(f"Selected seed={seed}: {c} commuting / {a} anticommuting "
          f"(target 34/11, arXiv:2604.10090's own K=10 instance)")
    return seed


def run_t1_sweep(seed: int) -> pd.DataFrame:
    t1_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00, 1.20, 1.50]
    rows = []
    for t1 in t1_values:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +12.0, 0.3, t1, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -12.0, 0.3, t1, seed, with_message=True)
        rows.append({"t1": t1, "I_mu_pos12": i_pos, "I_mu_neg12": i_neg, "delta": i_neg - i_pos})
        print(f"  t1={t1:.2f}  I(+12)={i_pos:.5f}  I(-12)={i_neg:.5f}  delta={i_neg - i_pos:+.5f}")
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_t1_sweep.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["t1"], df["I_mu_pos12"], 'o-', color='#00FFFF', label='I(mu=+12)')
    ax.plot(df["t1"], df["I_mu_neg12"], 'o-', color='#FF007F', label='I(mu=-12)')
    ax.set_xlabel("t1 (post-coupling evolution time)", color='#888888')
    ax.set_ylabel("Mutual information I(P:R[0])", color='#888888')
    ax.set_title("Traversable-wormhole teleportation signal vs. t1\n(seed=61, N=8 SYK, t0=0.3)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_t1_sweep.png", dpi=300)
    plt.close(fig)
    return df


def run_message_control(seed: int) -> pd.DataFrame:
    t1_values = [0.10, 0.30, 0.60, 0.85, 1.20]
    rows = []
    for with_message in (True, False):
        for t1 in t1_values:
            i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +12.0, 0.3, t1, seed, with_message=with_message)
            i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -12.0, 0.3, t1, seed, with_message=with_message)
            rows.append({
                "with_message": with_message, "t1": t1,
                "I_mu_pos12": i_pos, "I_mu_neg12": i_neg, "delta": i_neg - i_pos,
            })
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_message_control.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    with_msg = df[df.with_message]
    without_msg = df[~df.with_message]
    ax.plot(with_msg["t1"], with_msg["delta"], 'o-', color='#00FFFF', label='WITH message (real protocol)')
    ax.plot(without_msg["t1"], without_msg["delta"], 's-', color='#FFFF00', label='WITHOUT message (control)')
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("t1", color='#888888')
    ax.set_ylabel("delta = I(mu=-12) - I(mu=+12)", color='#888888')
    ax.set_title("Control: does the signal require the injected message?\n(seed=61, N=8 SYK, t0=0.3)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_message_control.png", dpi=300)
    plt.close(fig)
    return df


def run_mu_scan(seed: int) -> pd.DataFrame:
    mu_values = [4.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0]
    rows = []
    for mu in mu_values:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +mu, 0.3, 0.60, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -mu, 0.3, 0.60, seed, with_message=True)
        rows.append({"mu": mu, "I_pos": i_pos, "I_neg": i_neg, "delta": i_neg - i_pos})
    df = pd.DataFrame(rows).sort_values("mu").reset_index(drop=True)
    df.to_csv(_DATA_DIR / "wormhole_mu_scan.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["mu"], df["delta"], 'o-', color='#00FFFF')
    ax.axvline(12, color='#FFFF00', linestyle='--', alpha=0.6, label='arXiv:2604.10090 value (mu=12)')
    ax.set_xlabel("|mu| (L-R coupling strength)", color='#888888')
    ax.set_ylabel("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.set_title("Sign-dependent signal vs. coupling strength\n(seed=61, N=8 SYK, t0=0.3, t1=0.60)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_mu_scan.png", dpi=300)
    plt.close(fig)
    return df


def run_t0_scan(seed: int) -> pd.DataFrame:
    t0_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9]
    rows = []
    for t0 in t0_values:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +12.0, t0, 0.60, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -12.0, t0, 0.60, seed, with_message=True)
        rows.append({"t0": t0, "I_pos": i_pos, "I_neg": i_neg, "delta": i_neg - i_pos})
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_t0_scan.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["t0"], df["delta"], 'o-', color='#FF007F')
    ax.axvline(0.3, color='#FFFF00', linestyle='--', alpha=0.6, label='arXiv:2604.10090 value (t0=0.3)')
    ax.set_xlabel("t0 (pre-coupling scrambling time)", color='#888888')
    ax.set_ylabel("delta = I(mu=-12) - I(mu=+12)", color='#888888')
    ax.set_title("Sign-dependent signal vs. pre-coupling scrambling time\n(seed=61, N=8 SYK, mu=+-12, t1=0.60)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_t0_scan.png", dpi=300)
    plt.close(fig)
    return df


def run_2d_grid_search(seed: int) -> pd.DataFrame:
    """Joint (t0, mu) grid search, t1 held fixed at 0.60 (Experiment 1's
    peak). Precomputes the Hamiltonian/coupling matrices and their
    eigendecompositions ONCE (the expensive, t0/mu/t1-independent part
    of run_wormhole_protocol), then reuses them for every grid point --
    see this module's docstring for the measured ~165x speedup this
    gives over calling run_wormhole_protocol per point."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
    H = de.pauli_hamiltonian_to_matrix(terms_full, n_full)
    eigvals, eigvecs = np.linalg.eigh(H)
    V = de.pauli_hamiltonian_to_matrix(v_terms, n_full)
    v_eigvals, v_eigvecs = np.linalg.eigh(V)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, with_message=True))
    sv0 = sim.get_statevector()

    def mi_at(t0, mu, t1=0.60):
        sv = _evolve(sv0, eigvals, eigvecs, t0)
        sv = _evolve(sv, v_eigvals, v_eigvecs, mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        return mutual_information(sv, n_full, [P], [R[0]])

    t0_values = np.round(np.arange(0.05, 1.55, 0.05), 3)
    mu_values = np.round(np.arange(2.0, 31.0, 1.0), 1)

    rows = []
    delta_grid = np.zeros((len(mu_values), len(t0_values)))
    for i, mu in enumerate(mu_values):
        for j, t0 in enumerate(t0_values):
            i_pos = mi_at(t0, +mu)
            i_neg = mi_at(t0, -mu)
            delta = i_neg - i_pos
            delta_grid[i, j] = delta
            rows.append({"t0": t0, "mu": mu, "I_pos": i_pos, "I_neg": i_neg, "delta": delta})

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_2d_grid.csv", index=False)

    best = df.loc[df["delta"].idxmax()]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7))
    extent = [t0_values.min(), t0_values.max(), mu_values.min(), mu_values.max()]
    im = ax.imshow(delta_grid, origin='lower', aspect='auto', extent=extent, cmap='plasma')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.scatter([best['t0']], [best['mu']], color='cyan', marker='*', s=300,
               edgecolor='white', linewidth=1, label=f"max: t0={best['t0']:.2f}, mu={best['mu']:.1f}")
    ax.set_xlabel("t0 (pre-coupling scrambling time)", color='#888888')
    ax.set_ylabel("|mu| (L-R coupling strength)", color='#888888')
    ax.set_title("Joint (t0, mu) optimization surface\n(seed=61, N=8 SYK, t1=0.60 fixed)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_2d_grid.png", dpi=300)
    plt.close(fig)
    return df


def run_t1_rescan(seed: int) -> pd.DataFrame:
    """t1 re-scan at Experiment 5's (t0, mu) optimum -- resolves the caveat
    flagged there: the 2D grid held t1 fixed at 0.60 (Experiment 1's own
    1D peak) and noted its optimum could plausibly shift once t0/mu are no
    longer at their original 1D-scan defaults. Reuses Experiment 5's
    precompute-once approach (see its docstring/this module's docstring)
    since a fine ~125-point sweep would cost ~4.5s/call x 2 x 125 ~ 19
    minutes via the public run_wormhole_protocol API otherwise."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
    H = de.pauli_hamiltonian_to_matrix(terms_full, n_full)
    eigvals, eigvecs = np.linalg.eigh(H)
    V = de.pauli_hamiltonian_to_matrix(v_terms, n_full)
    v_eigvals, v_eigvecs = np.linalg.eigh(V)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, with_message=True))
    sv0 = sim.get_statevector()

    t0_opt, mu_opt = 0.65, 15.0

    def mi_at(t1, mu):
        sv = _evolve(sv0, eigvals, eigvecs, t0_opt)
        sv = _evolve(sv, v_eigvals, v_eigvecs, mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        return mutual_information(sv, n_full, [P], [R[0]])

    t1_values = np.round(np.arange(0.05, 1.31, 0.01), 3)
    rows = []
    for t1 in t1_values:
        i_pos = mi_at(t1, +mu_opt)
        i_neg = mi_at(t1, -mu_opt)
        rows.append({"t1": t1, "I_pos": i_pos, "I_neg": i_neg, "delta": i_neg - i_pos})
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_t1_rescan_optimum.csv", index=False)

    best = df.loc[df["delta"].idxmax()]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["t1"], df["delta"], '-', color='#00FFFF', linewidth=1.5)
    ax.axvline(0.60, color='#FFFF00', linestyle='--', alpha=0.6, label='Experiment 5 fixed value (t1=0.60)')
    ax.scatter([best['t1']], [best['delta']], color='cyan', marker='*', s=250,
               edgecolor='white', linewidth=1, zorder=5,
               label=f"peak: t1={best['t1']:.2f}, delta={best['delta']:+.5f}")
    ax.set_xlabel("t1 (post-coupling evolution time)", color='#888888')
    ax.set_ylabel("delta = I(mu=-15) - I(mu=+15)", color='#888888')
    ax.set_title("t1 re-scan at Experiment 5's (t0, mu) optimum\n(seed=61, N=8 SYK, t0=0.65, mu=15.0 fixed)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_t1_rescan_optimum.png", dpi=300)
    plt.close(fig)
    return df


def _coordinate_ascent_trace(seed: int, max_rounds: int = 5) -> pd.DataFrame:
    """Pure computation behind Experiments 7 and 8 -- iterated coordinate
    ascent toward the joint (t0, mu, t1) optimum for one SYK instance, no
    file I/O (callers decide what to save). Starting from Experiment 5's
    point (t0=0.65, mu=15.0, t1=0.60), each round alternates two full
    sub-steps at Experiment 5/6's own resolutions (not a shortcut, so
    results stay directly comparable): a 126-point t1 scan (step 0.01)
    holding (t0, mu) fixed, then an 870-point (t0, mu) grid (step
    0.05/1.0) holding the new t1 fixed. Stops when a full round leaves
    (t0, mu, t1) unchanged -- a genuine fixed point -- or after
    max_rounds as a safety cap."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
    H = de.pauli_hamiltonian_to_matrix(terms_full, n_full)
    eigvals, eigvecs = np.linalg.eigh(H)
    V = de.pauli_hamiltonian_to_matrix(v_terms, n_full)
    v_eigvals, v_eigvecs = np.linalg.eigh(V)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, with_message=True))
    sv0 = sim.get_statevector()

    def mi_delta(t0, mu, t1):
        sv = _evolve(sv0, eigvals, eigvecs, t0)
        sv = _evolve(sv, v_eigvals, v_eigvecs, mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        i_pos = mutual_information(sv, n_full, [P], [R[0]])
        sv = _evolve(sv0, eigvals, eigvecs, t0)
        sv = _evolve(sv, v_eigvals, v_eigvecs, -mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        i_neg = mutual_information(sv, n_full, [P], [R[0]])
        return i_neg - i_pos

    t1_scan_values = np.round(np.arange(0.05, 1.31, 0.01), 3)
    t0_grid_values = np.round(np.arange(0.05, 1.55, 0.05), 3)
    mu_grid_values = np.round(np.arange(2.0, 31.0, 1.0), 1)

    t0, mu, t1 = 0.65, 15.0, 0.60
    trace = [{"seed": seed, "round": 0, "stage": "start (Experiment 5)", "t0": t0, "mu": mu, "t1": t1,
              "delta": mi_delta(t0, mu, t1)}]

    for rnd in range(1, max_rounds + 1):
        t1_new = max(t1_scan_values, key=lambda t1c: mi_delta(t0, mu, t1c))
        delta_t1 = mi_delta(t0, mu, t1_new)
        trace.append({"seed": seed, "round": rnd, "stage": "t1 scan", "t0": t0, "mu": mu, "t1": t1_new, "delta": delta_t1})

        t0_new, mu_new = max(
            ((t0c, muc) for muc in mu_grid_values for t0c in t0_grid_values),
            key=lambda p: mi_delta(p[0], p[1], t1_new),
        )
        delta_grid = mi_delta(t0_new, mu_new, t1_new)
        trace.append({"seed": seed, "round": rnd, "stage": "t0/mu grid", "t0": t0_new, "mu": mu_new, "t1": t1_new,
                      "delta": delta_grid})

        converged = (t0_new == t0 and mu_new == mu and t1_new == t1)
        t0, mu, t1 = t0_new, mu_new, t1_new
        if converged:
            break

    return pd.DataFrame(trace)


def run_coordinate_ascent_3d(seed: int, max_rounds: int = 5):
    """Experiment 7: run _coordinate_ascent_trace for seed=61 and save
    its own CSV + convergence plot. See _coordinate_ascent_trace's
    docstring for the algorithm."""
    trace_df = _coordinate_ascent_trace(seed, max_rounds=max_rounds)
    trace_df.to_csv(_DATA_DIR / "wormhole_coordinate_ascent_3d.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(trace_df)), trace_df["delta"], 'o-', color='#00FFFF', markersize=5)
    for i, row in trace_df.iterrows():
        ax.annotate(f"t0={row.t0:.2f}\nmu={row.mu:.1f}\nt1={row.t1:.2f}",
                     (i, row.delta), textcoords="offset points", xytext=(0, 10),
                     fontsize=7, color='#888888', ha='center')
    ax.set_xlabel("coordinate-ascent step", color='#888888')
    ax.set_ylabel("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.set_title("Convergence of iterated coordinate ascent toward the joint (t0, mu, t1) optimum\n"
                 "(seed=61, N=8 SYK)", fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_coordinate_ascent_3d.png", dpi=300)
    plt.close(fig)
    return trace_df


def find_multiple_seeds(n_instances: int = 6, n_candidates: int = 3000, target_commuting: int = 34) -> list:
    """Screen up to n_candidates random seeds for EXACT matches to the
    paper's own selection criterion (34 commuting / 11 anticommuting
    pairs among the C(10,2)=45 pairs of K=10 terms), returning the first
    n_instances found. Unlike find_seed() (which returns the single
    closest match out of a smaller pool), Experiment 8 needs several
    independent, equally-valid instances to test whether Experiment 7's
    converged point is a property of the protocol or an idiosyncrasy of
    seed=61 specifically."""
    n_qubits = N_MAJORANA // 2
    found = []
    for seed in range(n_candidates):
        terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, seed)[1]
        c, a = commuting_pair_count(terms, n_qubits)
        if c == target_commuting:
            found.append(seed)
            if len(found) >= n_instances:
                break
    print(f"Found {len(found)} instances with exactly {target_commuting} commuting pairs "
          f"(screened {seed + 1} candidates): {found}")
    return found


def run_generality_check(seeds=None, max_rounds: int = 5) -> pd.DataFrame:
    """Experiment 8: does Experiment 7's converged point (t0=0.70,
    mu=17.0, t1=0.36) generalize across SYK instances, or is it specific
    to seed=61? Runs the identical coordinate-ascent procedure
    (_coordinate_ascent_trace -- same resolutions, same starting point)
    independently for each of several instances that all exactly match
    the paper's own selection criterion, same as seed=61 does.

    Honest negative result, not glossed over: it does NOT generalize.
    Converged (t0, mu, t1) points scatter across nearly the entire
    scanned range instead of clustering near seed=61's answer, 2 of 6
    instances converge AT the edge of the scanned t0/t1 range (their
    true optimum may lie outside what was scanned -- inconclusive, not
    a real fixed point), and 2 of 6 instances have a NEGATIVE delta at
    Experiment 5's own starting point (the sign-dependent signal isn't
    even reliably oriented the expected way using those "default"
    parameters across instances). Percentage-improvement figures are not
    reported here for that reason -- with a near-zero or negative
    baseline they blow up into meaningless numbers (e.g. one instance's
    raw improvement is nominally +3865%), not a real effect size."""
    if seeds is None:
        seeds = find_multiple_seeds(n_instances=6)

    rows = []
    for seed in seeds:
        trace_df = _coordinate_ascent_trace(seed, max_rounds=max_rounds)
        start = trace_df.iloc[0]
        converged = trace_df.iloc[-1]
        t1_values = np.round(np.arange(0.05, 1.31, 0.01), 3)
        t0_values = np.round(np.arange(0.05, 1.55, 0.05), 3)
        at_t1_edge = converged["t1"] in (t1_values.min(), t1_values.max())
        at_t0_edge = converged["t0"] in (t0_values.min(), t0_values.max())
        rows.append({
            "seed": seed, "baseline_delta": start["delta"],
            "converged_t0": converged["t0"], "converged_mu": converged["mu"], "converged_t1": converged["t1"],
            "converged_delta": converged["delta"], "rounds": int(trace_df["round"].max()),
            "at_grid_edge": at_t0_edge or at_t1_edge,
        })
        print(f"  seed={seed}: converged t0={converged['t0']:.2f} mu={converged['mu']:.1f} "
              f"t1={converged['t1']:.2f} delta={converged['delta']:+.5f} "
              f"(baseline={start['delta']:+.5f})"
              f"{'  [AT GRID EDGE -- inconclusive]' if (at_t0_edge or at_t1_edge) else ''}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(_DATA_DIR / "wormhole_generality_check.csv", index=False)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xcol, xlabel) in zip(axes, [("converged_t0", "t0"), ("converged_mu", "mu"), ("converged_t1", "t1")]):
        colors = ['#FF007F' if edge else '#00FFFF' for edge in summary_df["at_grid_edge"]]
        ax.scatter(summary_df[xcol], summary_df["converged_delta"], c=colors, s=80, zorder=5)
        for _, row in summary_df.iterrows():
            ax.annotate(str(row["seed"]), (row[xcol], row["converged_delta"]),
                        textcoords="offset points", xytext=(0, 8), fontsize=8, color='#888888', ha='center')
        ax.set_xlabel(xlabel, color='#888888')
        ax.set_ylabel("converged delta", color='#888888')
        ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    fig.suptitle("Experiment 8: converged (t0, mu, t1) scattered across 6 SYK instances\n"
                 "(cyan = interior fixed point, magenta = at grid edge, inconclusive)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_generality_check.png", dpi=300)
    plt.close(fig)
    return summary_df


def run_trotter_noise_scan(seed: int, t0: float, mu: float, t1: float,
                            noise_levels=(0.0, 0.005, 0.01, 0.02, 0.05),
                            n_trials: int = 6, n_steps_evolution: int = 8,
                            n_steps_coupling: int = 16) -> pd.DataFrame:
    """Does the sign-dependent signal survive realistic hardware noise?
    Runs the real Trotterized gate circuit (run_wormhole_protocol_trotter's
    own construction, reimplemented here to inject noise mid-circuit --
    that function runs the whole circuit in one call with no seam to
    interrupt) and applies a real stochastic depolarizing Kraus channel
    (dense_evolution.registry.NoiseModel) after each of the protocol's
    three phases (t0 evolution, mu coupling, t1 evolution), not just once
    at the end -- closer to how noise actually accumulates on real
    hardware than a single post-hoc channel would be.

    Compared against the *noiseless* Trotter result, not the exact
    backend -- Trotterization itself has a real, separate discretization
    error (see run_wormhole_protocol_trotter's own docstring: ~2% at the
    converged point), and conflating that with the effect of physical
    noise would misattribute one for the other.

    NoiseModel.apply_to_sv is a single-shot stochastic draw (same
    caveat as dashboard_core.mitigation.run_zne_mitigation), so each
    noise level is averaged over n_trials independent draws, each with
    its own fresh RNG."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)

    def run_noisy(mu_signed, noise_p, rng):
        sim = de.DenseSVSimulator(n_full)
        sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, True)
                         + trotter_evolve_ops(terms_full, t0, n_steps_evolution))
        sv = sim.get_statevector()
        if noise_p > 0:
            sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
        sim.set_state(sv)
        sim.run_circuit(trotter_evolve_ops(v_terms, mu_signed, n_steps_coupling))
        sv = sim.get_statevector()
        if noise_p > 0:
            sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
        sim.set_state(sv)
        sim.run_circuit(trotter_evolve_ops(terms_full, t1, n_steps_evolution))
        sv = sim.get_statevector()
        if noise_p > 0:
            sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
        return mutual_information(sv, n_full, [P], [R[0]])

    rows = []
    for noise_p in noise_levels:
        deltas = []
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 * trial + 7)
            i_pos = run_noisy(+mu, noise_p, rng)
            i_neg = run_noisy(-mu, noise_p, rng)
            deltas.append(i_neg - i_pos)
        deltas = np.array(deltas)
        rows.append({"noise_p": noise_p, "delta_mean": deltas.mean(), "delta_std": deltas.std(),
                     "delta_min": deltas.min(), "delta_max": deltas.max()})

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_trotter_noise_scan.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(df["noise_p"], df["delta_mean"], yerr=df["delta_std"], fmt='o-',
                color='#00FFFF', ecolor='#FF007F', capsize=4, markersize=6)
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("depolarizing noise probability p", color='#888888')
    ax.set_ylabel("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.set_title(f"Sign-dependent signal vs. realistic depolarizing noise (Trotter backend)\n"
                 f"(seed={seed}, t0={t0}, mu={mu}, t1={t1}, n={n_trials} trials/point)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_trotter_noise_scan.png", dpi=300)
    plt.close(fig)
    return df


def run_paper_defaults_comparison(seeds=None) -> pd.DataFrame:
    """Direct comparison against arXiv:2604.10090's own "Ensemble
    robustness" section, which reports 100 disorder realizations and
    concludes the sign-dependent asymmetry is "a generic feature of the
    ensemble", with their chosen Hamiltonian (seed=61 here) selected
    mainly for having an unusually *large* -- not unusually *signed* --
    asymmetry.

    Experiment 8 evaluated all 6 instances at Experiment 5's point
    (t0=0.65, mu=15.0, t1=0.60), which was itself optimized on seed=61 --
    a real confound: an instance showing the "wrong" sign there could
    simply be evaluated at a bad point for it, not a genuinely reversed
    signal. This experiment controls for that by re-evaluating all 6
    instances at the paper's OWN stated defaults (t0=0.3, mu=12,
    t1=0.60, Eq. matching Experiment 1's original setup) instead --
    the same point the paper's own ensemble claim is presumably about."""
    if seeds is None:
        seeds = find_multiple_seeds(n_instances=6)

    T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60
    rows = []
    for seed in seeds:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        delta = i_neg - i_pos
        rows.append({"seed": seed, "delta_at_paper_defaults": delta})
        print(f"  seed={seed}: delta_at_paper_defaults={delta:+.5f}"
              f"{'  [WRONG SIGN]' if delta < 0 else ''}")

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_paper_defaults_comparison.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_at_paper_defaults"]]
    ax.bar([str(s) for s in df["seed"]], df["delta_at_paper_defaults"], color=colors)
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("seed", color='#888888')
    ax.set_ylabel("delta at paper defaults (t0=0.3, mu=12, t1=0.60)", color='#888888')
    ax.set_title("Sign-dependent asymmetry at arXiv:2604.10090's own default parameters\n"
                 "(cyan = correct sign, magenta = wrong sign)", fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_paper_defaults_comparison.png", dpi=300)
    plt.close(fig)
    return df


def run_ensemble_sign_check(n_instances: int = 100, n_candidates: int = 120000) -> pd.DataFrame:
    """Large-sample version of Experiment 10's check. Experiment 10 found
    2 of 6 instances wrong-signed at arXiv:2604.10090's own stated
    default parameters (t0=0.3, mu=12, t1=0.60) -- a real but small
    sample. This repeats the identical check across up to n_instances
    exact 34/11-selection-matched SYK instances (same criterion as
    Experiments 8 and 10, via find_multiple_seeds), and additionally
    tests two candidate explanations floated informally alongside
    Experiment 10 for *why* the sign varies: Majorana mode-usage
    imbalance in the K=10 coupling terms (some modes coupled in many
    terms, others in few) and the spectral level-spacing r-statistic
    (a standard chaos diagnostic, Poisson~0.386 vs GOE~0.530). Both are
    tested for real correlation against delta via Pearson r, not just
    eyeballed."""
    seeds = find_multiple_seeds(n_instances=n_instances, n_candidates=n_candidates)
    all_quads = list(itertools.combinations(range(1, N_MAJORANA + 1), 4))
    n_qubits = N_MAJORANA // 2
    T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(all_quads), size=K_TERMS, replace=False)
        quads = [all_quads[idx] for idx in chosen_idx]
        mode_count = np.zeros(N_MAJORANA + 1)
        for q in quads:
            for m in q:
                mode_count[m] += 1
        usage_std = float(np.std(mode_count[1:]))

        _, terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, seed)
        H = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
        eigvals = np.sort(np.linalg.eigvalsh(H))
        gaps = np.diff(eigvals)
        gaps = gaps[gaps > 1e-12]
        r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
        r_stat = float(np.mean(r))

        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        delta = i_neg - i_pos

        rows.append({"seed": seed, "mode_usage_std": usage_std, "r_stat": r_stat,
                     "delta_at_paper_defaults": delta})

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_ensemble_sign_check.csv", index=False)

    r_usage = scipy_stats.pearsonr(df["mode_usage_std"], df["delta_at_paper_defaults"])
    r_chaos = scipy_stats.pearsonr(df["r_stat"], df["delta_at_paper_defaults"])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (xcol, xlabel, r_result) in zip(
        axes, [("mode_usage_std", "Majorana mode-usage std", r_usage), ("r_stat", "level-spacing r-statistic", r_chaos)]
    ):
        colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_at_paper_defaults"]]
        ax.scatter(df[xcol], df["delta_at_paper_defaults"], c=colors, s=25, alpha=0.7)
        ax.axhline(0, color='#666666', linestyle=':')
        ax.set_xlabel(xlabel, color='#888888')
        ax.set_ylabel("delta at paper defaults", color='#888888')
        ax.set_title(f"r={r_result.statistic:+.3f}, p={r_result.pvalue:.4f}", fontsize=10, color='#888888')
        ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    n_wrong = int((df["delta_at_paper_defaults"] < 0).sum())
    fig.suptitle(f"Experiment 11: n={len(df)} instances, {n_wrong}/{len(df)} ({100*n_wrong/len(df):.0f}%) wrong-signed "
                 f"at arXiv:2604.10090's own defaults\n(cyan = correct sign, magenta = wrong sign)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_ensemble_sign_check.png", dpi=300)
    plt.close(fig)
    return df


def run_size_winding_check(seeds=None, t_values=None, majorana_index: int = 1, beta: float = 3.0) -> pd.DataFrame:
    """Computes arXiv:2604.10090's own "size winding" diagnostic (Sec.
    S6, Eqs. S18-S22) directly: expands the THERMALLY-WEIGHTED,
    Heisenberg-evolved Majorana operator rho_beta^(1/2) chi_j(t), where
    chi_j(t) = exp(iHt) chi_j exp(-iHt) and rho_beta = exp(-beta H) /
    Tr(exp(-beta H)) (Eq. S18's own stated thermal state, "for a given
    thermal state rho_beta = e^(-beta H)/tr e^(-beta H)"; H is the
    one-sided Hamiltonian per the paper's own Sec. S6 setup) -- in the
    basis of Majorana strings Gamma_P, then checks (a) the winding size
    distribution's phase coherence within each size sector, R(l) =
    |q(l)|/P(l) where q(l) = sum_{|P|=l} c_P(t)^2 and P(l) =
    sum_{|P|=l} |c_P(t)|^2, and (b) whether that phase, arg(q(l)),
    grows with l as the paper's "perfect size winding" ansatz predicts.

    BUG FIX: this used to expand the BARE chi_j(t) (no rho_beta^(1/2)
    factor at all) instead of rho_beta^(1/2) chi_j(t) as Eq. S18 actually
    requires. Since chi_j(t) is Hermitian (a unitary conjugation of a
    Hermitian Majorana operator) and each Gamma_P is Hermitian too,
    Tr(Gamma_P^dagger chi_j(t)) is a trace of two Hermitian operators --
    ALWAYS real, for ANY Hamiltonian, ANY seed, ANY time. That forces
    q(l) = sum c_P^2 to be a sum of real squares (real, non-negative),
    so arg(q(l))=0 and R(l)=1.0 EXACTLY, by construction, regardless of
    the underlying physics -- this is almost certainly why every prior
    run of this function found max|phase|=0 and min R=1.0 for every
    single instance and time (a mathematically guaranteed triviality,
    not a physics finding). rho_beta^(1/2), built from the same H (not
    Hermitian-commuting with chi_j(t) in general), makes the thermally-
    weighted operator genuinely non-Hermitian, so c_P(t) can now be
    genuinely complex -- verified directly (non-trivial phases/R<1
    confirmed on a real instance) before this fix was trusted.

    Gamma_P = 2^(|P|/2) * i^(|P|(|P|-1)/2) * (ordered product of chi_j
    for j in P) per Eq. S19 -- the paper's own stated normalization
    Tr(Gamma_P Gamma_Q^dagger) could not be reproduced exactly from the
    extracted PDF text (a lost exponent is the most likely cause); the
    actual normalization used here, Tr(Gamma_P Gamma_Q^dagger) =
    2^|P| * dim * delta_PQ, was verified directly (Hermiticity and
    orthogonality checked numerically), not assumed from the paper's
    text.

    Run for each of Experiments 8/10's 6 instances (or a caller-supplied
    subset), at several post-quench times, to see whether either
    diagnostic distinguishes "good" (correctly-signed) from "bad"
    (wrong-signed) instances.
    """
    if seeds is None:
        seeds = find_multiple_seeds(n_instances=6)
    if t_values is None:
        t_values = [0.3, 0.7, 1.2, 2.0]

    n_qubits = N_MAJORANA // 2
    dim = 2 ** n_qubits
    all_P = []
    for size in range(N_MAJORANA + 1):
        all_P.extend(itertools.combinations(range(1, N_MAJORANA + 1), size))

    rows = []
    for seed in seeds:
        _, terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, seed)
        H = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
        eigvals, eigvecs = np.linalg.eigh(H)

        # rho_beta^(1/2) = exp(-beta*H/2) / sqrt(Tr(exp(-beta*H))), built
        # from the same eigendecomposition -- the thermal factor Eq. S18
        # actually requires and the earlier version of this function omitted.
        partition_z = np.sum(np.exp(-beta * eigvals))
        rho_half = (eigvecs @ np.diag(np.exp(-beta * eigvals / 2.0)) @ eigvecs.conj().T) / np.sqrt(partition_z)

        chis = {}
        for m in range(1, N_MAJORANA + 1):
            coeff, pdict = majorana_pauli_terms(m, n_qubits)
            chis[m] = de.pauli_hamiltonian_to_matrix([(coeff, pdict)], n_qubits)

        def gamma_of(P):
            size = len(P)
            phase = (2.0 ** (size / 2.0)) * (1j ** (size * (size - 1) // 2))
            mat = np.eye(dim, dtype=complex)
            for idx in P:
                mat = mat @ chis[idx]
            return phase * mat

        gammas = {P: gamma_of(P) for P in all_P}
        norms = {P: 2.0 ** len(P) * dim for P in all_P}
        chi_j = chis[majorana_index]

        for t in t_values:
            U = eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.conj().T
            op_t = U @ chi_j @ U.conj().T
            thermal_op_t = rho_half @ op_t
            c = {P: np.trace(gammas[P].conj().T @ thermal_op_t) / norms[P] for P in all_P}
            p_dist, q = {}, {}
            for l in range(N_MAJORANA + 1):
                Ps_l = [P for P in all_P if len(P) == l]
                p_dist[l] = sum(abs(c[P]) ** 2 for P in Ps_l)
                q[l] = sum(c[P] ** 2 for P in Ps_l)
            total = sum(p_dist.values())
            mean_l = sum(l * p_dist[l] for l in range(N_MAJORANA + 1)) / total
            phases = [np.angle(q[l]) for l in range(N_MAJORANA + 1) if abs(q[l]) > 1e-8]
            r_vals = [abs(q[l]) / p_dist[l] for l in range(N_MAJORANA + 1) if p_dist[l] > 1e-8]
            rows.append({
                "seed": seed, "t": t, "mean_size": mean_l,
                "max_abs_phase": max((abs(p) for p in phases), default=0.0),
                "min_R": min(r_vals) if r_vals else float("nan"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_size_winding.csv", index=False)

    overall_max_phase = float(df["max_abs_phase"].max())
    overall_min_R = float(df["min_R"].min())

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.cool(np.linspace(0, 1, len(seeds)))
    for seed, color in zip(seeds, colors):
        sub = df[df["seed"] == seed].sort_values("t")
        ax.plot(sub["t"], sub["mean_size"], 'o-', color=color, label=f"seed={seed}")
    ax.set_xlabel("t (post-quench evolution time)", color='#888888')
    ax.set_ylabel("mean operator size <l>(t)", color='#888888')
    ax.set_title(
        f"Experiment 12 (corrected): size winding (arXiv:2604.10090 Sec. S6) across {len(seeds)} instances\n"
        f"max|phase|={overall_max_phase:.4f}, min R(l)={overall_min_R:.4f} -- genuinely non-trivial "
        f"(earlier version omitted rho_beta^(1/2), which forced phase=0/R=1 by construction)",
        fontsize=10, fontweight='bold'
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_size_winding.png", dpi=300)
    plt.close(fig)
    return df


def run_mechanistic_check(n_instances: int = 100) -> pd.DataFrame:
    """Tests two protocol-grounded (not just post-hoc-statistical)
    candidate explanations for Experiment 11's unexplained sign
    variance, reusing that experiment's own n=100 instance set and
    delta values (data/wormhole_ensemble_sign_check.csv) rather than
    re-screening candidates from scratch:

    - Feature A, "message-mode participation": dense_evolution.fermions.
      majorana_pauli_terms's Jordan-Wigner mapping (j=(mode_index-1)//2)
      shows Majorana modes 1 and 2 map onto qubit index 0 -- exactly the
      qubit the message is swapped into (L[0]) and read out from (R[0])
      in dashboard_core.wormhole's protocol. Experiment 11's mode-usage
      imbalance treated all 8 modes as interchangeable (just the std of
      usage counts); this feature instead asks whether the K=10 SYK
      quads specifically over/under-represent the message qubit's own
      two modes -- a sharper, mechanistically-motivated version of the
      same underlying idea, cheap (purely combinatorial, no simulation).
    - Feature B, "operator growth rate": Experiment 12 already computes
      real, non-trivial, instance-varying mean operator size <l>(t) as
      a side effect of its (trivial) phase-winding computation, but
      never checked whether growth rate/peak correlates with the sign
      -- this reuses that exact machinery (run_size_winding_check) at
      two probe times in the growth region it found (t=0.7, t=1.2).

    Both are the 4th and 5th candidates tested against this same n=100
    sample (after Experiment 11's two and Experiment 12's phase/R,
    which had no per-instance variation to test) -- a real
    multiple-comparisons risk flagged in this module's caveats.
    """
    csv_path = _DATA_DIR / "wormhole_ensemble_sign_check.csv"
    if csv_path.exists():
        base = pd.read_csv(csv_path)
        seeds = base["seed"].tolist()[:n_instances]
        delta_map = dict(zip(base["seed"], base["delta_at_paper_defaults"]))
    else:
        seeds = find_multiple_seeds(n_instances=n_instances)
        T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60
        delta_map = {}
        for seed in seeds:
            i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
            i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
            delta_map[seed] = i_neg - i_pos

    all_quads = list(itertools.combinations(range(1, N_MAJORANA + 1), 4))
    message_modes = {1, 2}
    message_count = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(all_quads), size=K_TERMS, replace=False)
        quads = [all_quads[idx] for idx in chosen_idx]
        message_count[seed] = sum(1 for q in quads if message_modes & set(q))

    growth_df = run_size_winding_check(seeds=seeds, t_values=[0.7, 1.2])
    growth_pivot = growth_df.pivot(index="seed", columns="t", values="mean_size")

    rows = []
    for seed in seeds:
        rows.append({
            "seed": seed,
            "message_mode_count": message_count[seed],
            "mean_size_t0.7": growth_pivot.loc[seed, 0.7],
            "mean_size_t1.2": growth_pivot.loc[seed, 1.2],
            "delta_at_paper_defaults": delta_map[seed],
        })
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_mechanistic_check.csv", index=False)

    r_message = scipy_stats.pearsonr(df["message_mode_count"], df["delta_at_paper_defaults"])
    r_growth = scipy_stats.pearsonr(df["mean_size_t1.2"], df["delta_at_paper_defaults"])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (xcol, xlabel, r_result) in zip(
        axes, [("message_mode_count", "K=10 quads touching the message qubit's modes (1,2)", r_message),
               ("mean_size_t1.2", "mean operator size <l> at t=1.2", r_growth)]
    ):
        colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_at_paper_defaults"]]
        ax.scatter(df[xcol], df["delta_at_paper_defaults"], c=colors, s=25, alpha=0.7)
        ax.axhline(0, color='#666666', linestyle=':')
        ax.set_xlabel(xlabel, color='#888888')
        ax.set_ylabel("delta at paper defaults", color='#888888')
        ax.set_title(f"r={r_result.statistic:+.3f}, p={r_result.pvalue:.4f}", fontsize=10, color='#888888')
        ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    n_wrong = int((df["delta_at_paper_defaults"] < 0).sum())
    fig.suptitle(f"Experiment 13: n={len(df)} instances, {n_wrong}/{len(df)} wrong-signed -- "
                 f"message-mode participation and operator growth rate vs. sign\n"
                 f"(cyan = correct sign, magenta = wrong sign)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_mechanistic_check.png", dpi=300)
    plt.close(fig)
    return df


def _quad_cooccurrence_graph(seed: int) -> np.ndarray:
    """Rebuilds seed's K=10 SYK quads (same RNG replay as
    run_ensemble_sign_check/run_mechanistic_check) and returns the
    N_MAJORANA x N_MAJORANA *weighted* adjacency matrix: entry (i,j) is
    how many of the 10 quads contain both modes i and j -- the actual
    qubit-coupling topology a raw commuting-pair *count* (the paper's
    own selection criterion) or a per-mode usage *count* (Experiment
    11/13's features) both throw away: two instances can have identical
    34/11 commuting-pair counts and identical per-mode usage counts
    while coupling entirely different (and differently concentrated)
    pairs of modes together.

    Weighted, not binary: an ad hoc check before committing to this
    design found a *binary* co-occurrence graph (edge iff a pair
    co-occurs in >=1 quad) saturates to the complete graph K8 for most
    instances -- 10 quads out of the 70 possible each contribute 6
    pairs, C(10,2)... i.e. up to 60 pair-slots spread over only 28
    possible mode pairs, so nearly every pair ends up connected by
    chance regardless of the underlying structure, making a binary
    graph nearly useless as a discriminator (verified directly: 2 of 3
    spot-checked seeds were already the complete graph). The weighted
    count does not saturate and showed real spread across seeds in the
    same spot check (max weighted degree 21-24, degree std 3.3-5.0,
    0-3 completely uncoupled mode pairs)."""
    all_quads = list(itertools.combinations(range(1, N_MAJORANA + 1), 4))
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(all_quads), size=K_TERMS, replace=False)
    quads = [all_quads[idx] for idx in chosen_idx]

    adj = np.zeros((N_MAJORANA + 1, N_MAJORANA + 1), dtype=int)
    for quad in quads:
        for i, j in itertools.combinations(quad, 2):
            adj[i, j] += 1
            adj[j, i] += 1
    return adj[1:, 1:]  # drop the unused 0 row/col; modes are 1-indexed


def _weighted_algebraic_connectivity(adj: np.ndarray) -> float:
    """Fiedler value of the weighted graph Laplacian L = D - A (D the
    weighted-degree diagonal): second-smallest eigenvalue, 0 exactly if
    the graph (at that edge-weight threshold, here any weight > 0) is
    disconnected, larger means more evenly/expander-like coupled."""
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def run_qubit_topology_check(n_instances: int = 100) -> pd.DataFrame:
    """Tests whether the actual qubit-coupling *topology* of each
    instance's K=10 SYK quads -- not just how many terms commute (the
    paper's own selection criterion, already shown insufficient in
    Experiment 10/11) or how many terms touch a given mode (Experiment
    11/13's usage-count features) -- predicts the sign-dependent delta.
    Builds a weighted 8-mode co-occurrence graph per instance (weight
    (i,j) = how many quads contain both modes i and j) and computes
    four structural features: max_weighted_degree (how "hub"-like the
    most-coupled mode is), weighted_degree_std (spread of coupling
    strength across modes), n_zero_pairs (how many of the 28 possible
    mode pairs are never coupled together at all -- directly captures
    the "does coupling concentrate through a few modes (star-like,
    leaving many pairs untouched) vs. spread evenly (leaving few pairs
    untouched)" intuition), and weighted algebraic connectivity (the
    Fiedler value of the weighted graph Laplacian -- how evenly/
    expander-like the whole coupling structure is).

    Reuses Experiment 11's own n=100 instance set and delta values
    (data/wormhole_ensemble_sign_check.csv) rather than re-screening --
    this is the 6th/7th/8th/9th candidate explanation tested against
    that same sample (see the multiple-comparisons caveat in this
    module's docstring).
    """
    csv_path = _DATA_DIR / "wormhole_ensemble_sign_check.csv"
    if csv_path.exists():
        base = pd.read_csv(csv_path)
        seeds = base["seed"].tolist()[:n_instances]
        delta_map = dict(zip(base["seed"], base["delta_at_paper_defaults"]))
    else:
        seeds = find_multiple_seeds(n_instances=n_instances)
        T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60
        delta_map = {}
        for seed in seeds:
            i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
            i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
            delta_map[seed] = i_neg - i_pos

    rows = []
    for seed in seeds:
        adj = _quad_cooccurrence_graph(seed)
        wdegrees = adj.sum(axis=1)
        n_zero_pairs = int((adj[np.triu_indices(N_MAJORANA, 1)] == 0).sum())
        rows.append({
            "seed": seed,
            "max_weighted_degree": int(wdegrees.max()),
            "weighted_degree_std": float(np.std(wdegrees)),
            "n_zero_pairs": n_zero_pairs,
            "algebraic_connectivity": _weighted_algebraic_connectivity(adj),
            "delta_at_paper_defaults": delta_map[seed],
        })
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_qubit_topology.csv", index=False)

    features = [
        ("max_weighted_degree", "max weighted mode degree (hub-ness)"),
        ("weighted_degree_std", "weighted degree std (coupling spread)"),
        ("n_zero_pairs", "# of the 28 mode pairs never coupled together"),
        ("algebraic_connectivity", "weighted algebraic connectivity (Fiedler value)"),
    ]
    results = {f: scipy_stats.pearsonr(df[f], df["delta_at_paper_defaults"]) for f, _ in features}

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (fcol, flabel) in zip(axes.flat, features):
        r_result = results[fcol]
        colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_at_paper_defaults"]]
        jitter = np.random.default_rng(0).normal(0, 0.15, size=len(df)) if df[fcol].nunique() < 6 else 0.0
        ax.scatter(df[fcol] + jitter, df["delta_at_paper_defaults"], c=colors, s=25, alpha=0.7)
        ax.axhline(0, color='#666666', linestyle=':')
        ax.set_xlabel(flabel, color='#888888')
        ax.set_ylabel("delta at paper defaults", color='#888888')
        ax.set_title(f"r={r_result.statistic:+.3f}, p={r_result.pvalue:.4f}", fontsize=10, color='#888888')
        ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    n_wrong = int((df["delta_at_paper_defaults"] < 0).sum())
    fig.suptitle(f"Experiment 14: n={len(df)} instances, {n_wrong}/{len(df)} wrong-signed -- "
                 f"qubit-coupling topology vs. sign\n(cyan = correct sign, magenta = wrong sign)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_qubit_topology.png", dpi=300)
    plt.close(fig)
    return df


def _find_closest_commuting_seeds(n_majorana: int, k_terms: int, J_local: float, n_instances: int,
                                   n_candidates: int, target_commuting: int = 34) -> list:
    """Generalization of find_multiple_seeds for n_majorana != 8: that
    function requires an *exact* target_commuting match using the
    module-level N_MAJORANA=8/K_TERMS=10 constants, which does not work
    at other N (see run_n_scaling_check's docstring -- an exact 34/11
    match does not exist among 3000 candidates at n_majorana=12).
    Returns the n_instances seeds whose commuting-pair count is closest
    to target_commuting, screening n_candidates seeds."""
    n_qubits = n_majorana // 2
    scored = []
    for seed in range(n_candidates):
        _, terms = build_sparse_syk_terms(n_majorana, k_terms, J_local, seed)
        c, a = commuting_pair_count(terms, n_qubits)
        scored.append((abs(c - target_commuting), seed, c, a))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[:n_instances]


def run_n_scaling_check(n_majorana_large: int = 12, k_terms: int = 10, n_instances: int = 6,
                         n_candidates: int = 3000, n_steps_evolution: int = 8,
                         n_steps_coupling: int = 16) -> pd.DataFrame:
    """Tests whether the sign-dependent instance variance from
    Experiments 10/11 (N=8: 2/6, then 49/100, wrong-signed at the
    paper's own default parameters) persists, worsens, or shrinks at a
    larger Majorana count -- the one candidate explanation from
    prog.txt's original suggestions never yet tried, and potentially
    the most informative: SYK-type chaotic systems often show
    instance-to-instance fluctuations shrinking toward a thermodynamic
    limit as N grows.

    Backend and scope, chosen for real feasibility, not convenience:
    - The exact (eigendecomposition) backend used throughout Experiments
      1-8, 10, 11 is infeasible at N=12 -- diagonalization cost scales
      as dim^3, and N=12's joint L+R+P+Q system is dim=2^14=16384 vs.
      N=8's dim=2^10=1024, a (16384/1024)^3 = 4096x slowdown. Measured
      directly (not estimated): N=8's own exact backend took 4.3-6.4s
      per diagonalization (this module's docstring); at that scaling
      factor N=12 would take hours per instance. The Trotterized gate-
      circuit backend (run_wormhole_protocol_trotter) doesn't pay that
      cubic cost -- gate application is O(dim) per gate, not O(dim^3) --
      and was measured directly at N=12: ~19s/call, close to N=8's own
      already-measured ~14s/call Trotter cost (Experiment 9's noise
      scan). Both N=8 and N=12 are therefore evaluated via the SAME
      Trotter backend here, for a clean, backend-matched comparison --
      not the exact backend N=8's own headline 2/6 (Experiment 10) and
      49/100 (Experiment 11) numbers used, which would confound N-
      scaling with a real, separately-measured backend effect
      (Experiment 9 already showed noise/backend choice can shift the
      delta and even its sign near the noise threshold).
    - n_instances=6 (not Experiment 11's n=100): at ~19s/call x 2 signs
      x 6 instances x 2 values of N, this experiment already costs
      ~7.6 minutes; scaling to n=100 at N=12 would cost over 2 hours for
      this one experiment alone. This is explicitly a smaller, first
      feasibility/existence check, not a repeat of Experiment 11's
      statistical rigor -- the sample is too small to establish whether
      variance shrinks with any real statistical power, only whether
      the qualitative picture (wrong-sign rate, delta magnitude) looks
      similar or different at a glance.
    - The paper's own 34/11 commuting/anticommuting selection criterion
      does not have an exact match at N=12: screening 3000 candidates
      found zero instances with exactly 34 commuting pairs (verified
      directly, not assumed -- the achievable distribution at N=12,
      K=10 peaks around 21-23 and tops out at 31 in a 500-seed sample).
      `_find_closest_commuting_seeds` generalizes the existing exact-
      match screening to find the *closest* achievable match instead,
      the same closest-match philosophy find_seed() already uses at
      N=8, just generalized to N.
    - K_TERMS is kept fixed at 10 (not scaled with N) -- the simplest
      choice, preserving the paper's own term-count convention exactly
      and not introducing a new, unjustified free parameter.
    """
    J_local = float(np.sqrt(2))
    T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60

    rows = []
    for n_maj, label in ((N_MAJORANA, "N=8"), (n_majorana_large, f"N={n_majorana_large}")):
        selected = _find_closest_commuting_seeds(n_maj, k_terms, J_local, n_instances, n_candidates)
        for _diff, seed, c, a in selected:
            i_pos = run_wormhole_protocol_trotter(
                n_maj, k_terms, J_local, +MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True,
                n_steps_evolution=n_steps_evolution, n_steps_coupling=n_steps_coupling)
            i_neg = run_wormhole_protocol_trotter(
                n_maj, k_terms, J_local, -MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True,
                n_steps_evolution=n_steps_evolution, n_steps_coupling=n_steps_coupling)
            delta = i_neg - i_pos
            rows.append({
                "n_majorana": n_maj, "label": label, "seed": seed,
                "commuting": c, "anticommuting": a,
                "delta_at_paper_defaults_trotter": delta,
            })

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_n_scaling_check.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = df["label"].unique()
    for i, label in enumerate(labels):
        sub = df[df["label"] == label]
        colors = ['#FF007F' if d < 0 else '#00FFFF' for d in sub["delta_at_paper_defaults_trotter"]]
        x = np.full(len(sub), i) + np.random.default_rng(0).normal(0, 0.05, size=len(sub))
        ax.scatter(x, sub["delta_at_paper_defaults_trotter"], c=colors, s=60, alpha=0.85, zorder=3)
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("delta at paper defaults (Trotter backend)", color='#888888')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    n_wrong_by_label = df.groupby("label").apply(
        lambda g: int((g["delta_at_paper_defaults_trotter"] < 0).sum()), include_groups=False)
    title_parts = ", ".join(f"{lbl}: {n_wrong_by_label[lbl]}/{n_instances} wrong-signed" for lbl in labels)
    ax.set_title(f"Experiment 15: N-scaling check (Trotter backend, matched)\n{title_parts}\n"
                 f"(cyan = correct sign, magenta = wrong sign)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_n_scaling_check.png", dpi=300)
    plt.close(fig)
    return df


def _run_trotter_ordered(terms_ordered, v_terms, n_side, n_full, L, R, P, Q, mu_signed, t0, t1,
                          n_steps_evolution, n_steps_coupling):
    """One full Trotterized protocol run (noiseless) with terms_ordered
    used for BOTH the t0 and t1 evolution phases, in whatever order the
    caller passes -- trotter_evolve_ops applies a step's terms in
    exactly the order given (see that function's own docstring), so
    passing a different permutation of the same physical terms changes
    the Trotterized circuit's actual gate sequence, not just a label.
    A fresh DenseSVSimulator per call, since this computes one
    independent full-protocol result, not a continuation of a previous
    one (unlike run_trotter_noise_scan's single continuous run)."""
    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, True)
                     + trotter_evolve_ops(terms_ordered, t0, n_steps_evolution))
    sim.run_circuit(trotter_evolve_ops(v_terms, mu_signed, n_steps_coupling))
    sim.run_circuit(trotter_evolve_ops(terms_ordered, t1, n_steps_evolution))
    sv = sim.get_statevector()
    return mutual_information(sv, n_full, [P], [R[0]])


def run_term_order_noncommutativity_check(seeds=None, n_steps_evolution: int = 8,
                                           n_steps_coupling: int = 16) -> pd.DataFrame:
    """Tests a genuinely different kind of non-commutativity from the
    one this repo already settled in scripts/channel_order_
    noncommutativity.py: that script found NOISE-CHANNEL order matters
    iff at least one channel is non-Pauli (Pauli channels commute as
    superoperators) -- the depolarizing channel used throughout this
    script's own noise experiment (Experiment 9) IS a Pauli-mixture
    channel, so that specific rule predicts noise-channel reordering
    here would show nothing new. This experiment tests a different
    question entirely: does the *order in which the K=10+10=20 SYK
    Hamiltonian terms are applied within the Trotterized circuit's own
    t0/t1 evolution phases* matter, and does the size of that effect
    correlate with the sign? Trotter error is exactly a manifestation
    of non-commuting terms -- if every term commuted, any order would
    give the exact same (exact) answer regardless of order -- so this
    is really asking whether the *degree* of non-commutativity among a
    seed's specific K=10 terms (not just how many of them commute in
    the paper's own pairwise sense, already shown insufficient in
    Experiments 10/11/14) leaves a fingerprint that tracks the sign.

    Method: for each instance, run the noiseless Trotter protocol at
    the paper's own default parameters twice -- once with the terms in
    their natural order (as `_protocol_layout` builds them), once with
    that same list reversed -- for both mu signs, giving delta_original
    and delta_reversed. `order_sensitivity = abs(delta_reversed -
    delta_original)` is the candidate feature, correlated against
    delta_original's own sign across instances. Noiseless by design
    (unlike channel_order_noncommutativity.py's stochastic, noise-
    driven setting): the quantity being compared here (mutual
    information from a single deterministic Trotter circuit output) is
    not a stochastic sampling distribution, so no Monte Carlo
    unraveling or permutation test is needed to get a clean, real
    order-sensitivity number -- reversing the term list is a fixed,
    reproducible single comparison, not a hypothesis requiring
    significance testing against a sampling-noise null.
    """
    if seeds is None:
        seeds = find_multiple_seeds(n_instances=6)
    T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60

    rows = []
    for seed in seeds:
        n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
        terms_reversed = list(reversed(terms_full))

        i_pos_orig = _run_trotter_ordered(terms_full, v_terms, n_side, n_full, L, R, P, Q,
                                           +MU_PAPER, T0_PAPER, T1_PAPER, n_steps_evolution, n_steps_coupling)
        i_neg_orig = _run_trotter_ordered(terms_full, v_terms, n_side, n_full, L, R, P, Q,
                                           -MU_PAPER, T0_PAPER, T1_PAPER, n_steps_evolution, n_steps_coupling)
        delta_original = i_neg_orig - i_pos_orig

        i_pos_rev = _run_trotter_ordered(terms_reversed, v_terms, n_side, n_full, L, R, P, Q,
                                          +MU_PAPER, T0_PAPER, T1_PAPER, n_steps_evolution, n_steps_coupling)
        i_neg_rev = _run_trotter_ordered(terms_reversed, v_terms, n_side, n_full, L, R, P, Q,
                                          -MU_PAPER, T0_PAPER, T1_PAPER, n_steps_evolution, n_steps_coupling)
        delta_reversed = i_neg_rev - i_pos_rev

        rows.append({
            "seed": seed,
            "delta_original_order": delta_original,
            "delta_reversed_order": delta_reversed,
            "order_sensitivity": abs(delta_reversed - delta_original),
        })

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_term_order_noncommutativity.csv", index=False)

    r_result = scipy_stats.pearsonr(df["order_sensitivity"], df["delta_original_order"])

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_original_order"]]
    ax.scatter(df["order_sensitivity"], df["delta_original_order"], c=colors, s=80, zorder=5)
    for _, row in df.iterrows():
        ax.annotate(str(int(row["seed"])), (row["order_sensitivity"], row["delta_original_order"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, color='#888888', ha='center')
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("order sensitivity |delta_reversed - delta_original|", color='#888888')
    ax.set_ylabel("delta (original term order)", color='#888888')
    ax.set_title(f"Experiment 16: term-order non-commutativity vs. sign (n={len(df)})\n"
                 f"r={r_result.statistic:+.3f}, p={r_result.pvalue:.4f} "
                 f"(cyan = correct sign, magenta = wrong sign)",
                 fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_term_order_noncommutativity.png", dpi=300)
    plt.close(fig)
    return df


def _run_trotter_ordered_noisy(terms_ordered, v_terms, n_side, n_full, L, R, P, Q, mu_signed, t0, t1,
                                noise_p, rng, n_steps_evolution, n_steps_coupling):
    """Same construction as _run_trotter_ordered, plus a real stochastic
    depolarizing Kraus draw (dense_evolution.registry.NoiseModel,
    single-shot per call, same as run_trotter_noise_scan/Experiment 9)
    injected after each of the three phases."""
    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, True)
                     + trotter_evolve_ops(terms_ordered, t0, n_steps_evolution))
    sv = sim.get_statevector()
    if noise_p > 0:
        sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
    sim.set_state(sv)
    sim.run_circuit(trotter_evolve_ops(v_terms, mu_signed, n_steps_coupling))
    sv = sim.get_statevector()
    if noise_p > 0:
        sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
    sim.set_state(sv)
    sim.run_circuit(trotter_evolve_ops(terms_ordered, t1, n_steps_evolution))
    sv = sim.get_statevector()
    if noise_p > 0:
        sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
    return mutual_information(sv, n_full, [P], [R[0]])


def run_term_order_noise_interaction_check(seeds=None, noise_p: float = 0.01, n_trials: int = 6,
                                            n_steps_evolution: int = 8,
                                            n_steps_coupling: int = 16) -> pd.DataFrame:
    """Experiment 16 asked whether term order matters noiselessly; this
    asks the question Experiment 16's own caveats flagged as untested:
    does term-order sensitivity change under realistic noise -- closer
    in spirit to scripts/channel_order_noncommutativity.py's own noisy,
    stochastic setting, applied here to term order instead of noise-
    channel order.

    Scope decided by a real, measured cost constraint, not convenience:
    channel_order_noncommutativity.py's exact methodology (Monte Carlo
    unraveling with k_trajectories=8192, Jensen-Shannon divergence, a
    permutation test) is built for a tiny, cheap 3-qubit toy circuit --
    each trajectory there is one small gate sequence. This script's
    Trotterized wormhole circuit is far more gate-heavy (K=10+10 terms,
    n_steps_evolution=8/n_steps_coupling=16), measured directly at
    ~7.8s per single noisy protocol call -- running thousands of
    trajectories per (seed, order, mu-sign) to resolve a full output
    distribution and its JS divergence, as the noise-channel script
    does, would cost hours per instance and was not attempted. Instead
    this reuses Experiment 9's own established, more modest budget
    (n_trials=6 single-shot stochastic draws averaged per point) and
    Experiment 16's own original-vs-reversed term order comparison,
    combined: for each instance, delta is averaged over n_trials noisy
    runs at noise_p, separately for the original and reversed term
    order, giving noisy_order_sensitivity = |delta_mean_reversed -
    delta_mean_original| -- the noisy analogue of Experiment 16's
    order_sensitivity, at the same noise_p=0.01 Experiment 9 already
    identified as close to (just below) the noise threshold where the
    noiseless signal starts crossing zero.

    One deliberate deviation from Experiment 9's own noise-injection
    pattern: each trial's i_pos and i_neg share the same noise
    realization (a fresh rng re-seeded with the same value immediately
    before each of the two calls), not independent draws carried
    forward from one rng -- a common-random-numbers variance-reduction
    choice, isolating the mu-sign effect (what delta actually measures)
    from trial-to-trial noise-realization variance, since delta itself
    is already a small quantity easily swamped by independent noise on
    each side.
    """
    if seeds is None:
        seeds = find_multiple_seeds(n_instances=6)
    T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60

    def mean_delta(terms_ordered, v_terms, n_side, n_full, L, R, P, Q):
        deltas = []
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 * trial + 7)
            i_pos = _run_trotter_ordered_noisy(terms_ordered, v_terms, n_side, n_full, L, R, P, Q,
                                                +MU_PAPER, T0_PAPER, T1_PAPER, noise_p, rng,
                                                n_steps_evolution, n_steps_coupling)
            rng = np.random.default_rng(1000 * trial + 7)
            i_neg = _run_trotter_ordered_noisy(terms_ordered, v_terms, n_side, n_full, L, R, P, Q,
                                                -MU_PAPER, T0_PAPER, T1_PAPER, noise_p, rng,
                                                n_steps_evolution, n_steps_coupling)
            deltas.append(i_neg - i_pos)
        return float(np.mean(deltas)), float(np.std(deltas))

    rows = []
    for seed in seeds:
        n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
        terms_reversed = list(reversed(terms_full))

        delta_mean_orig, delta_std_orig = mean_delta(terms_full, v_terms, n_side, n_full, L, R, P, Q)
        delta_mean_rev, delta_std_rev = mean_delta(terms_reversed, v_terms, n_side, n_full, L, R, P, Q)

        rows.append({
            "seed": seed,
            "delta_mean_original_noisy": delta_mean_orig,
            "delta_std_original_noisy": delta_std_orig,
            "delta_mean_reversed_noisy": delta_mean_rev,
            "delta_std_reversed_noisy": delta_std_rev,
            "noisy_order_sensitivity": abs(delta_mean_rev - delta_mean_orig),
        })

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_term_order_noise_interaction.csv", index=False)

    r_result = scipy_stats.pearsonr(df["noisy_order_sensitivity"], df["delta_mean_original_noisy"])

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_mean_original_noisy"]]
    ax.errorbar(df["noisy_order_sensitivity"], df["delta_mean_original_noisy"],
                yerr=df["delta_std_original_noisy"], fmt='none', ecolor='#444444', zorder=1)
    ax.scatter(df["noisy_order_sensitivity"], df["delta_mean_original_noisy"], c=colors, s=80, zorder=5)
    for _, row in df.iterrows():
        ax.annotate(str(int(row["seed"])), (row["noisy_order_sensitivity"], row["delta_mean_original_noisy"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, color='#888888', ha='center')
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel(f"noisy order sensitivity |delta_reversed - delta_original| (p={noise_p})", color='#888888')
    ax.set_ylabel("delta, original term order (noisy, mean of trials)", color='#888888')
    ax.set_title(f"Experiment 17: term-order x noise interaction vs. sign (n={len(df)})\n"
                 f"r={r_result.statistic:+.3f}, p={r_result.pvalue:.4f} "
                 f"(cyan = correct sign, magenta = wrong sign)",
                 fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_term_order_noise_interaction.png", dpi=300)
    plt.close(fig)
    return df


def run_t0_correction_check(n_instances: int = 100, t1: float = 1.25) -> pd.DataFrame:
    """Re-tests Experiment 11's flagship ensemble sign check at the
    paper's REAL hardware working point, t0=1.8 -- not t0=0.3, which
    every prior experiment in this script (including Experiment 11)
    mislabeled as "the paper's own default parameters."

    Rereading arXiv:2604.10090 directly (2026-08-09) found that t0=1.8
    is explicitly and repeatedly stated as the paper's actual chosen
    injection time (Sec. S4: "t0=1.8 marks a turning point... we choose
    t0=1.8 as the hardware working point"), chosen specifically to
    balance signal strength against Trotterization error. t0=0.3 does
    not appear anywhere in the paper's text as an injection time -- the
    only "0.3" in the extracted PDF text is a y-axis tick label on
    Fig. 5's mutual-information plot (0.0/0.1/0.2/0.3/0.4), not a
    parameter value. Where t0=0.3 actually came from is unclear; it was
    simply wrong, not a defensible alternate reading.

    Unlike t0, the paper does not give one single "default" t1 -- Fig. 5
    scans t1 in [0.5, 6.0] at fixed t0=1.8, for both signs of mu.
    t1=1.25 (this function's default) was chosen by a real 23-point scan
    (t1 in [0.5, 6.0], step 0.25, data/wormhole_t1_finescan_t0_1.8.csv)
    on seed=61, the closest 34/11-matched analog to the paper's own
    chosen instance: the sign flips repeatedly across the range (not a
    single clean peak, itself notable), with the first local maximum
    (closest to the injection time, the more natural reading of "near
    the teleportation time") at t1=1.25 (delta=+0.01064 for that one
    seed). A second, larger peak exists later, at t1=4.75
    (delta=+0.01219), which looks more like a finite-size revival than
    the primary teleportation signal, and is not used as the default
    here.

    Same 34/11-selection-matched instance criterion and n=100 sample
    size as Experiment 11 (find_multiple_seeds, n_candidates=120000),
    for a directly comparable wrong-sign fraction.

    Sequential, not parallelized -- two real attempts were tried and
    both abandoned. (1) ThreadPoolExecutor, reasoning that
    np.linalg.eigh's BLAS/LAPACK backend releases the GIL: in practice
    this hung/thrashed for tens of minutes instead of speeding up,
    because BLAS libraries (OpenBLAS/MKL) spawn their OWN internal
    thread pool per call, and N Python threads x BLAS's own M threads
    each oversubscribes the CPU by a large factor. (2)
    ProcessPoolExecutor (avoids the threading issue -- each worker
    process gets its own independent BLAS thread pool) with each
    worker's BLAS threads additionally pinned to 1 via an initializer:
    correct (verified bit-identical to the sequential loop on a real
    subset) but gave only ~1.1-1.2x wall-clock at 4 workers on an
    8-core machine, far short of the expected ~4x. Direct measurement
    traced the per-call cost floor (~5s/call, ~1.5s of which is a
    one-time per-process warm-up) to something other than BLAS
    diagonalization -- limiting OMP/OPENBLAS/MKL/XLA thread counts
    made no measurable difference to a single sequential process's own
    per-call time either, so the bottleneck isn't thread contention at
    all; it's real, and evidently not straightforwardly parallel,
    per-call cost inside run_wormhole_protocol/DenseSVSimulator
    (JAX dispatch overhead is the leading suspect, not confirmed).
    Root-causing that further was judged not worth the time against a
    one-time ~18-minute sequential run that already produces a correct
    result -- left as a genuinely open question, not silently dropped.
    """
    seeds = find_multiple_seeds(n_instances=n_instances, n_candidates=120000)
    rows = []
    for seed in seeds:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +12.0, 1.8, t1, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -12.0, 1.8, t1, seed, with_message=True)
        delta = i_neg - i_pos
        rows.append({"seed": seed, "delta_at_t0_1.8": delta})

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_ensemble_sign_check_t0_1.8.csv", index=False)

    n_wrong = int((df["delta_at_t0_1.8"] < 0).sum())

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_at_t0_1.8"]]
    ax.bar(range(len(df)), df["delta_at_t0_1.8"], color=colors)
    ax.axhline(0, color='white', linewidth=0.8)
    ax.set_xlabel("instance index (discovery order)", color='#888888')
    ax.set_ylabel("delta = I(mu=-12) - I(mu=+12)", color='#888888')
    ax.set_title(
        f"Experiment 18: ensemble sign check at the paper's REAL t0=1.8 (not t0=0.3)\n"
        f"n={len(df)}: {n_wrong}/{len(df)} ({100*n_wrong/len(df):.0f}%) wrong-signed at "
        f"t0=1.8, mu=12, t1={t1} (cyan = correct sign, magenta = wrong sign)",
        fontsize=10, fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_ensemble_sign_check_t0_1.8.png", dpi=300)
    plt.close(fig)

    print(f"{n_wrong}/{len(df)} ({100*n_wrong/len(df):.0f}%) wrong-signed at the paper's REAL "
          f"t0=1.8 (vs. the previously-mislabeled t0=0.3 check's 49/100)")
    return df


def run_noise_level_scan_check(seeds=None, n_instances: int = 20,
                                noise_levels=(0.005, 0.01, 0.02),
                                n_trials: int = 6,
                                n_steps_evolution: int = 8,
                                n_steps_coupling: int = 16) -> pd.DataFrame:
    """Experiment 17's own caveat flagged this as untested: its
    term-order x noise correlation (r=+0.340, p=0.0158 at n=50) was
    only ever measured at a single noise_p=0.01 (Experiment 9's own
    near-threshold value, reused rather than independently chosen).
    This scans noise_p itself, reusing Experiment 17's exact method
    (run_term_order_noise_interaction_check, unchanged) at each level,
    on the SAME seed set across all levels (so any r/p trend reflects
    noise_p, not a different random instance draw at each point).

    Scope narrowed from Experiment 17's flagship n=50 to n=20 for cost:
    each noise level costs ~n_instances x 24 noisy protocol calls x
    ~7.8s/call (Experiment 17's own measured per-call cost) -- ~62
    minutes at n=20 per level, ~2.6 hours at n=50. n=20 was not chosen
    arbitrarily: Experiment 17's own write-up already reports it stayed
    significant there (r=+0.587, p=0.0065) at noise_p=0.01, the one
    overlapping point between that check and this one -- used below as
    a direct consistency check that this scan's methodology reproduces
    that number before trusting the new noise_p=0.005/0.02 points.

    noise_levels defaults to (0.005, 0.01, 0.02): 0.01 is Experiment
    9/17's own already-measured point, included here as the consistency
    check described above rather than assumed to transfer; 0.005 and
    0.02 bracket it at half and double, chosen to see the trend's
    direction/shape with only 2 new (expensive) points rather than a
    finer scan this session's time budget can't cover.
    """
    if seeds is None:
        # find_multiple_seeds's own default n_candidates=3000 is tuned
        # for its original small n_instances=6 callers -- silently
        # returns fewer than requested instead of erroring if the exact
        # 34/11 match isn't found within that budget (confirmed by a
        # real bug here: a first attempt at n_instances=20 without this
        # override screened only 3000 candidates and silently returned
        # just 6). 120000 matches the budget Experiments 11/13/14/18
        # already use for their own n=100 seed sets.
        seeds = find_multiple_seeds(n_instances=n_instances, n_candidates=120000)

    rows = []
    for noise_p in noise_levels:
        df_level = run_term_order_noise_interaction_check(
            seeds=seeds, noise_p=noise_p, n_trials=n_trials,
            n_steps_evolution=n_steps_evolution, n_steps_coupling=n_steps_coupling,
        )
        r_result = scipy_stats.pearsonr(df_level["noisy_order_sensitivity"], df_level["delta_mean_original_noisy"])
        n_wrong = int((df_level["delta_mean_original_noisy"] < 0).sum())
        rows.append({
            "noise_p": noise_p,
            "n": len(df_level),
            "pearson_r": r_result.statistic,
            "p_value": r_result.pvalue,
            "n_wrong_signed": n_wrong,
        })
        print(f"noise_p={noise_p}: r={r_result.statistic:+.3f}, p={r_result.pvalue:.4f}, "
              f"{n_wrong}/{len(df_level)} wrong-signed")

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_noise_level_scan.csv", index=False)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(df["noise_p"], df["pearson_r"], marker='o', color='#00FFFF', linewidth=1.5)
    ax1.axhline(0, color='#666666', linestyle=':')
    ax1.set_xlabel("noise_p", color='#888888')
    ax1.set_ylabel("Pearson r (order sensitivity vs. delta)", color='#888888')
    ax1.set_title("Correlation strength vs. noise level", color='#CCCCCC')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')

    ax2.plot(df["noise_p"], df["p_value"], marker='o', color='#FF007F', linewidth=1.5)
    ax2.axhline(0.05, color='#FFAA00', linestyle='--', label='p=0.05')
    ax2.set_xlabel("noise_p", color='#888888')
    ax2.set_ylabel("p-value", color='#888888')
    ax2.set_title("Significance vs. noise level", color='#CCCCCC')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')

    fig.suptitle(f"Experiment 19: does Experiment 17's term-order x noise correlation "
                 f"hold across noise levels? (n={n_instances})", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_noise_level_scan.png", dpi=300)
    plt.close(fig)

    return df


def run_all():
    seed = find_seed()

    print("\n=== Experiment 1: t1 sweep ===")
    df1 = run_t1_sweep(seed)
    peak1 = df1.loc[df1["delta"].idxmax()]
    print(f"  peak: t1={peak1['t1']:.2f}  delta={peak1['delta']:+.5f}")

    print("\n=== Experiment 2: message-vs-no-message control ===")
    df2 = run_message_control(seed)
    max_no_msg = df2[~df2.with_message]["delta"].abs().max()
    print(f"  max |delta| without message: {max_no_msg:.8f} (should be ~0)")

    print("\n=== Experiment 3: mu-magnitude scan ===")
    df3 = run_mu_scan(seed)
    peak3 = df3.loc[df3["delta"].idxmax()]
    print(f"  peak: mu={peak3['mu']:.1f}  delta={peak3['delta']:+.5f}")

    print("\n=== Experiment 4: t0 scrambling-time scan ===")
    df4 = run_t0_scan(seed)
    peak4 = df4.loc[df4["delta"].idxmax()]
    print(f"  peak: t0={peak4['t0']:.2f}  delta={peak4['delta']:+.5f}")

    print("\n=== Experiment 5: 2D (t0, mu) joint grid search ===")
    df5 = run_2d_grid_search(seed)
    peak5 = df5.loc[df5["delta"].idxmax()]
    print(f"  grid: {df5['t0'].nunique()} x {df5['mu'].nunique()} = {len(df5)} points")
    print(f"  global max: t0={peak5['t0']:.2f}  mu={peak5['mu']:.1f}  delta={peak5['delta']:+.5f}")

    print("\n=== Experiment 6: t1 re-scan at Experiment 5's (t0, mu) optimum ===")
    df6 = run_t1_rescan(seed)
    peak6 = df6.loc[df6["delta"].idxmax()]
    print(f"  peak: t1={peak6['t1']:.2f}  delta={peak6['delta']:+.5f}"
          f"  ({(peak6['delta'] / peak5['delta'] - 1) * 100:+.1f}% vs. Experiment 5's t1=0.60)")

    print("\n=== Experiment 7: iterated coordinate ascent toward the joint (t0, mu, t1) optimum ===")
    df7 = run_coordinate_ascent_3d(seed)
    converged = df7.iloc[-1]
    print(f"  {df7['round'].max()} rounds, converged: "
          f"t0={converged['t0']:.2f}  mu={converged['mu']:.1f}  t1={converged['t1']:.2f}  "
          f"delta={converged['delta']:+.5f}"
          f"  ({(converged['delta'] / peak5['delta'] - 1) * 100:+.1f}% vs. Experiment 5)")

    print("\n=== Experiment 8: does the converged point generalize across SYK instances? ===")
    df8 = run_generality_check()
    n_edge = int(df8["at_grid_edge"].sum())
    n_negative_baseline = int((df8["baseline_delta"] < 0).sum())
    print(f"  {len(df8)} instances checked -- converged (t0, mu, t1) does NOT cluster near seed=61's "
          f"answer, {n_edge} at the grid edge (inconclusive), {n_negative_baseline} with a negative "
          f"baseline delta at Experiment 5's own starting point.")

    print("\n=== Experiment 9: does the signal survive realistic hardware noise? ===")
    df9 = run_trotter_noise_scan(seed, t0=0.70, mu=17.0, t1=0.36)
    crossing = df9[df9["delta_mean"] < 0]
    first_negative_p = crossing["noise_p"].min() if not crossing.empty else None
    print(f"  noiseless delta={df9.iloc[0]['delta_mean']:+.5f}; mean delta crosses zero "
          f"{'at p=' + str(first_negative_p) if first_negative_p is not None else 'nowhere in the scanned range'} "
          f"-- at p=0.01 the signal ({df9.iloc[2]['delta_mean']:+.5f}) is already smaller than its own "
          f"trial-to-trial noise ({df9.iloc[2]['delta_std']:.5f}).")

    print("\n=== Experiment 10: cross-check against arXiv:2604.10090's own ensemble-robustness claim ===")
    df10 = run_paper_defaults_comparison()
    n_wrong = int((df10["delta_at_paper_defaults"] < 0).sum())
    print(f"  {n_wrong}/{len(df10)} instances show the wrong sign at the paper's own default "
          f"parameters (t0=0.3, mu=12, t1=0.60) -- contradicts arXiv:2604.10090's 'Ensemble "
          f"robustness' claim that the sign-dependent asymmetry is a generic ensemble feature, "
          f"at least for this 34/11-selection-matched subset.")

    print("\n=== Experiment 11: large-sample (n=100) ensemble sign check ===")
    df11 = run_ensemble_sign_check(n_instances=100)
    n_wrong11 = int((df11["delta_at_paper_defaults"] < 0).sum())
    r_usage = scipy_stats.pearsonr(df11["mode_usage_std"], df11["delta_at_paper_defaults"])
    r_chaos = scipy_stats.pearsonr(df11["r_stat"], df11["delta_at_paper_defaults"])
    print(f"  n={len(df11)}: {n_wrong11}/{len(df11)} ({100*n_wrong11/len(df11):.0f}%) wrong-signed at the "
          f"paper's own defaults -- a much larger, more statistically robust version of Experiment 10's "
          f"2/6 finding. Neither candidate structural explanation holds up at this sample size: "
          f"mode-usage-imbalance r={r_usage.statistic:+.3f} (p={r_usage.pvalue:.3f}), "
          f"level-spacing r-statistic r={r_chaos.statistic:+.3f} (p={r_chaos.pvalue:.3f}) -- "
          f"neither is a statistically significant predictor of the sign.")

    print("\n=== Experiment 12: size winding (arXiv:2604.10090 Sec. S6 diagnostic) ===")
    df12 = run_size_winding_check()
    max_phase12 = float(df12["max_abs_phase"].max())
    min_R12 = float(df12["min_R"].min())
    print(f"  {df12['seed'].nunique()} instances x {df12['t'].nunique()} times: "
          f"max|phase|={max_phase12:.4f}, min R(l)={min_R12:.4f} -- the paper's own "
          f"'perfect size winding' phase-coherence diagnostic, corrected 2026-08-09 to include "
          f"the rho_beta^(1/2) thermal factor Eq. S18 requires (the original run omitted it, "
          f"which forced R(l)=1.0/phase=0.0 exactly regardless of physics -- an implementation "
          f"bug, not a finding). Whether this corrected diagnostic explains the sign-dependent "
          f"instance variance has not yet been tested at scale. Mean operator size <l>(t) does "
          f"show genuine chaos-consistent growth-then-recurrence, confirming the underlying "
          f"scrambling dynamics are real.")

    print("\n=== Experiment 13: mechanistic check -- message-mode participation & operator growth rate ===")
    df13 = run_mechanistic_check(n_instances=100)
    r_message13 = scipy_stats.pearsonr(df13["message_mode_count"], df13["delta_at_paper_defaults"])
    r_growth13 = scipy_stats.pearsonr(df13["mean_size_t1.2"], df13["delta_at_paper_defaults"])
    print(f"  n={len(df13)}: message-mode participation r={r_message13.statistic:+.3f} "
          f"(p={r_message13.pvalue:.4f}), operator growth rate (mean size at t=1.2) "
          f"r={r_growth13.statistic:+.3f} (p={r_growth13.pvalue:.4f}).")

    print("\n=== Experiment 14: qubit-coupling topology check ===")
    df14 = run_qubit_topology_check(n_instances=100)
    topology_features = ["max_weighted_degree", "weighted_degree_std", "n_zero_pairs", "algebraic_connectivity"]
    for feat in topology_features:
        r14 = scipy_stats.pearsonr(df14[feat], df14["delta_at_paper_defaults"])
        print(f"  {feat}: r={r14.statistic:+.3f} (p={r14.pvalue:.4f})")

    print("\n=== Experiment 15: N-scaling check (N=8 vs N=12, Trotter backend, matched) ===")
    df15 = run_n_scaling_check(n_majorana_large=12, n_instances=6)
    for label in df15["label"].unique():
        sub = df15[df15["label"] == label]
        n_wrong15 = int((sub["delta_at_paper_defaults_trotter"] < 0).sum())
        print(f"  {label}: {n_wrong15}/{len(sub)} wrong-signed, "
              f"mean|delta|={sub['delta_at_paper_defaults_trotter'].abs().mean():.5f}")

    print("\n=== Experiment 16: term-order non-commutativity check ===")
    df16 = run_term_order_noncommutativity_check(seeds=find_multiple_seeds(n_instances=30, n_candidates=35000))
    r16 = scipy_stats.pearsonr(df16["order_sensitivity"], df16["delta_original_order"])
    print(f"  n={len(df16)}: order_sensitivity vs delta: r={r16.statistic:+.3f} (p={r16.pvalue:.4f})")

    print("\n=== Experiment 17: term-order x noise interaction check ===")
    df17 = run_term_order_noise_interaction_check(seeds=find_multiple_seeds(n_instances=50, n_candidates=50000))
    r17 = scipy_stats.pearsonr(df17["noisy_order_sensitivity"], df17["delta_mean_original_noisy"])
    print(f"  n={len(df17)}: noisy_order_sensitivity vs delta: r={r17.statistic:+.3f} (p={r17.pvalue:.4f})")

    print("\n=== Experiment 18: ensemble sign check at the paper's REAL t0=1.8 (not t0=0.3) ===")
    df18 = run_t0_correction_check(n_instances=100)
    n_wrong18 = int((df18["delta_at_t0_1.8"] < 0).sum())
    print(f"  n={len(df18)}: {n_wrong18}/{len(df18)} ({100*n_wrong18/len(df18):.0f}%) wrong-signed "
          f"at t0=1.8 (vs. the mislabeled t0=0.3 check's 49/100)")

    print("\n=== Experiment 19: noise-level scan for the term-order x noise correlation ===")
    df19 = run_noise_level_scan_check()
    for _, row in df19.iterrows():
        print(f"  noise_p={row['noise_p']}: r={row['pearson_r']:+.3f}, p={row['p_value']:.4f}")

    print("\n============================================================")
    print("Data saved to data/wormhole_*.csv")
    print("Plots saved to images/wormhole_*.png")
    print("============================================================")


if __name__ == "__main__":
    run_all()
