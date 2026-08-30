"""
Does dense_evolution.zero_noise_extrapolation's "healing-adapted" branch
(sigma_at_base_noise, docs/api/healing.md) actually help once given a real,
oracle-free sigma -- instead of remaining an undesigned placeholder?

Background: the healing branch perturbs the 3 Richardson coefficients via
calculate_delta_preemp(sigma_at_base_noise, target_sigma_ideal). The
library's own calculate_advanced_sigma (kappa*H*Psi*Omega_sync*tau_K) was
meant to produce sigma_at_base_noise, but its 5 inputs have no defined
provenance in a ZNE context (dashboard_core's run_zne_mitigation only ever
has scalar Pauli expectation values, never a density matrix -- so
entropy/purity-style inputs have no real data to come from at this
integration point). This script bypasses that unresolved question and
tests the mechanism directly with the most literal, oracle-free candidate
for "sigma at the base noise level": the real empirical standard deviation
of the n_trials stochastic Kraus-draw ensemble at noise_factor=1 -- no
density matrix, no ideal-state comparison (uhlmann_fidelity's own
docstring bans that: "never to feed into" a correction).

target_sigma_ideal is measured per (n_qubits, pauli_string, n_trials, seed)
config as the empirical std at the smallest real noise_p in the sweep
(p=NOISE_PS[0]), with an independently-seeded RNG -- a genuine best-case
reference the channel noise is compared against, not a guessed constant.

Reproduces run_zne_mitigation's own scalar-expectation pipeline directly
(not calling it) because it needs per-trial values, which that function
computes and discards after averaging.

INCLUDES A NEGATIVE CONTROL (permutation test): re-runs the healing branch
on each row's own (means, ideal) but with base_std SHUFFLED across rows.
If the win rate survives shuffling as well as the real pairing, the
mechanism isn't using real information in base_std -- it's a structural
bias in the coefficient perturbation, the same failure mode Discovery
Section 25 found for a different JSD-ZNE nudge (an apparent win that had
nothing to do with the signal being tested).

Produces `data/zne_healing_sigma_provenance.csv`.

    python scripts/zne_healing_sigma_provenance.py
"""
import csv
import pathlib

import numpy as np
import dense_evolution as de

OUT_CSV = pathlib.Path(__file__).resolve().parent.parent / "data" / "zne_healing_sigma_provenance.csv"

N_QUBITS = 2
PAULI_STRING = "ZZ"
NOISE_MODELS = ["depolarizing", "bitflip", "amplitude_damping"]
NOISE_PS = [0.02, 0.05, 0.10]
SEEDS = list(range(5))
NOISE_FACTORS = (1.0, 2.0, 3.0)
N_TRIALS = 300


def build_bell_ideal():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[0];
    cx q[0], q[1];
    """
    parsed = de.QASMParser().parse(qasm)
    sim = de.DenseSVSimulator(N_QUBITS, use_float32=False)
    sim.run_circuit(parsed.to_tuples())
    return np.asarray(sim.sv)


def trial_values_at_scale(sv_ideal, noise_model, scaled_p, rng, n_trials):
    vals = np.empty(n_trials)
    for i in range(n_trials):
        sv_noisy = de.NoiseModel.apply_to_sv(sv_ideal.copy(), N_QUBITS, noise_model, scaled_p, rng=rng)
        vals[i] = np.real(de.pauli_expectation(sv_noisy, PAULI_STRING))
    return vals


def run_one(sv_ideal, ideal_expectation, noise_model, noise_p, seed, sigma_ideal_ref):
    rng = np.random.default_rng(seed)
    means = []
    base_std = None
    for k, factor in enumerate(NOISE_FACTORS):
        scaled_p = min(noise_p * factor, 1.0)
        vals = trial_values_at_scale(sv_ideal, noise_model, scaled_p, rng, N_TRIALS)
        means.append(float(vals.mean()))
        if k == 0:
            base_std = float(vals.std())

    plain = float(de.zero_noise_extrapolation(means, list(NOISE_FACTORS)))
    healing = float(de.zero_noise_extrapolation(
        means, list(NOISE_FACTORS),
        sigma_at_base_noise=base_std, target_sigma_ideal=sigma_ideal_ref,
    ))

    err_plain = abs(plain - ideal_expectation)
    err_healing = abs(healing - ideal_expectation)
    return means, base_std, err_plain, err_healing


def negative_control_permutation(rows, ideal_expectation):
    """Re-runs the healing branch on each row's own (means, ideal) but with
    base_std SHUFFLED across rows (a real permutation test)."""
    rng = np.random.default_rng(12345)
    shuffled_std = [r["base_std"] for r in rows]
    rng.shuffle(shuffled_std)
    deltas = []
    for r, fake_std in zip(rows, shuffled_std):
        healing_fake = float(de.zero_noise_extrapolation(
            r["means"], list(NOISE_FACTORS),
            sigma_at_base_noise=fake_std, target_sigma_ideal=r["sigma_ideal_ref"],
        ))
        err_healing_fake = abs(healing_fake - ideal_expectation)
        deltas.append(r["err_plain"] - err_healing_fake)
    deltas = np.array(deltas)
    return float(deltas.mean()), float((deltas > 0).mean())


def measure_sigma_ideal_ref(sv_ideal, noise_model, seed):
    rng = np.random.default_rng(1000 + seed)
    vals = trial_values_at_scale(sv_ideal, noise_model, NOISE_PS[0], rng, N_TRIALS)
    return float(vals.std())


def main():
    sv_ideal = build_bell_ideal()
    ideal_expectation = float(np.real(de.pauli_expectation(sv_ideal, PAULI_STRING)))
    print(f"ideal <{PAULI_STRING}> = {ideal_expectation:.6f}")

    rows = []
    for noise_model in NOISE_MODELS:
        for noise_p in NOISE_PS:
            deltas = []
            for seed in SEEDS:
                sigma_ideal_ref = measure_sigma_ideal_ref(sv_ideal, noise_model, seed)
                means, base_std, err_plain, err_healing = run_one(
                    sv_ideal, ideal_expectation, noise_model, noise_p, seed, sigma_ideal_ref)
                delta = err_plain - err_healing  # positive = healing helped
                deltas.append(delta)
                rows.append(dict(
                    noise_model=noise_model, noise_p=noise_p, seed=seed,
                    sigma_ideal_ref=sigma_ideal_ref, base_std=base_std,
                    means=means, err_plain=err_plain, err_healing=err_healing, delta=delta,
                ))
            deltas = np.array(deltas)
            win_rate = float((deltas > 0).mean())
            print(f"{noise_model:16s} p={noise_p:.2f}  mean_delta={deltas.mean():+.5f}  "
                  f"win_rate={win_rate:.2f}  (n={len(deltas)})")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = [k for k in rows[0].keys() if k != "means"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in r.items() if k != "means"})
    print(f"\nwrote {OUT_CSV}")

    all_deltas = np.array([r["delta"] for r in rows])
    print(f"\nREAL PAIRING: mean_delta={all_deltas.mean():+.6f}  "
          f"win_rate={float((all_deltas > 0).mean()):.3f}  n={len(all_deltas)}")

    ctrl_mean, ctrl_win = negative_control_permutation(rows, ideal_expectation)
    print(f"NEGATIVE CONTROL (shuffled sigma): mean_delta={ctrl_mean:+.6f}  "
          f"win_rate={ctrl_win:.3f}  n={len(rows)}")

    if ctrl_win >= float((all_deltas > 0).mean()) - 0.05 and ctrl_mean > 0:
        print("\nVERDICT: CONFOUND -- shuffled sigma performs comparably to the real signal, the win is a structural bias in the coefficient perturbation, not real information in base_std.")
    else:
        print("\nVERDICT: real signal survives the permutation test -- the improvement is genuinely coming from base_std, not a structural artifact.")


if __name__ == "__main__":
    main()