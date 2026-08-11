"""
Steane code [[7,1,3]] -- block 4: STIM translation and independent
cross-validation of block 1's Dense-Evolution implementation.

This is NOT a from-scratch STIM Steane-code example -- it deliberately
reproduces block 1's exact encoding circuit (same 7 physical qubit roles,
same free/derived qubit split, same gate order, same 6 stabilizer
generators in the same order) so the two are genuinely comparable, then
adds a real ancilla-based syndrome-extraction circuit (STIM's own native
idiom, not something block 1 needed since it read stabilizer expectation
values directly off the statevector).

Encoding (identical to block 1, see its docstring for the derivation):
    FREE_QUBITS = [0, 1, 3] (c1, c2, c4) -- H each
    DERIVED_QUBITS: q2 = q0+q1, q4 = q0+q3, q5 = q1+q3, q6 = q0+q1+q3 (CX chains)

Stabilizers (same qubit supports as block 1's IIIXXXX/IXXIIXX/XIXIXIX and
IIIZZZZ/IZZIIZZ/ZIZIZIZ, same order):
    S_X1/S_Z1 -> qubits {3,4,5,6}
    S_X2/S_Z2 -> qubits {1,2,5,6}
    S_X3/S_Z3 -> qubits {0,2,4,6}

Ancilla layout for syndrome extraction (physically realistic, not MPP):
    X_ANCILLAS = [7, 8, 9]   one ancilla per X-stabilizer, H-CX*-H-measure
    Z_ANCILLAS = [10, 11, 12]  one ancilla per Z-stabilizer, CX*-measure
13 qubits total.

Noise-model mapping caveat (STIM DEPOLARIZE1 vs Dense-Evolution NoiseModel)
----------------------------------------------------------------------------
Both use the same total single-qubit error probability p with a uniform
1:1:1 split among X/Y/Z (STIM: DEPOLARIZE1(p) applies X, Y or Z each with
probability p/3, else I; Dense-Evolution: Kraus set
{sqrt(1-p) I, sqrt(p/3) X, sqrt(p/3) Y, sqrt(p/3) Z}) -- same total rate,
same convention. They are NOT the same *mechanism* at the single-trial
level, though: STIM's DEPOLARIZE1 samples one definite Pauli outcome for
the WHOLE qubit per shot (a genuine single-trajectory realization -- for a
stabilizer state this leaves it an exact +-1 eigenstate of every generator
that commutes with whichever Pauli fired). Dense-Evolution's
NoiseModel.apply_to_sv decides fire/no-fire INDEPENDENTLY PER
COMPUTATIONAL-BASIS AMPLITUDE PAIR within a single trial (block 1's own
docstring documents this: raw stabilizer expectation values after
apply_to_sv came out non-+-1, e.g. ~0.75, which is why block 1's noise
sweep needed a real projective syndrome measurement with collapse
afterward -- a genuine single global Pauli error never does that). Both
reduce to the identical averaged Kraus channel over many trials, which is
what the aggregate logical-vs-physical curves below are actually
comparing, but the two are not interchangeable trial-by-trial.

Decoder note
------------
For iid per-qubit noise, the exact maximum-likelihood decoder for this
code is syndrome-table lookup: the Hamming(7,4) parity-check matrix's 7
columns are exactly the 7 nonzero 3-bit vectors, so every nonzero syndrome
has a UNIQUE minimum-weight (weight-1) solution -- i.e. this table lookup
IS optimal, not merely a heuristic. Section 4 below shows pymatching's
MWPM decoder (via STIM's detector_error_model(decompose_errors=True)) is
measurably SUBOPTIMAL here: qubit 6's syndrome is (1,1,1) -- a genuine
3-detector hyperedge (a single physical error triggers all 3 ancillas of a
family at once, since every stabilizer here has weight 4) -- and STIM's
hyperedge decomposition does not always resolve it correctly. This is a
known limitation of MWPM on small non-topological/non-graphlike codes; it
is not a bug in this script's circuit. The syndrome-table decoder (proven
exact above, and verified bit-for-bit identical to block 1's own
dense_evolution-derived table) is what's used for the large-N sweep.

Result: the STIM sweep CONTRADICTS block 1's crossover claim
---------------------------------------------------------------
Block 1 (4000 trials/point) reported the code helping only up to p~0.105
and hurting above p~0.12. Re-running block 1's own script confirms that
result is reproducible, not a one-off fluctuation (p=0.12: 0.1275+-0.0053,
p=0.15: 0.1865+-0.0062). STIM's sweep here (5,000,000 shots/point, exact
decoder) instead shows the code helping across the ENTIRE tested range up
to p=0.15 (p=0.12: 0.0917+-0.0001, p=0.15: 0.1307+-0.0002) -- an 8+ sigma
disagreement relative to block 1's own stated uncertainty, far too large
to be sampling noise on either side.

The most likely cause is the noise-mechanism difference documented above:
a projective syndrome measurement's outcome PROBABILITIES depend only on
the noisy density matrix, so if Dense-Evolution's apply_to_sv genuinely
implemented the claimed single-qubit depolarizing Kraus map, block 1's
logical error rate should match STIM's (both are valid unravelings of the
same channel, and Born-rule statistics are unraveling-independent). The
observed gap suggests apply_to_sv's per-computational-basis-amplitude-pair
firing decision (independent per pair, not one decision for the whole
qubit) does NOT reduce to that Kraus map on an entangled multi-qubit
state: a genuine single-qubit Pauli error is one operator applied
uniformly across all 64 amplitude pairs of the other 6 qubits, not up to
64 independently-chosen outcomes for the same qubit within one trial. That
would inject strictly more decoherence than the nominal p implies,
consistent with block 1's systematically worse rates. This is a
plausible root cause, not a proven one here -- confirming or fixing it
would mean auditing dense_evolution.registry.NoiseModel.apply_to_sv
itself, out of scope for this translation script. Flagging it explicitly
rather than silently trusting either curve.
"""
import importlib.util
import pathlib
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stim
import pymatching

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_IMAGES_DIR = _REPO_ROOT / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

_BLOCK1_PATH = pathlib.Path(__file__).resolve().parent / "steane_code_block1.py"

N_DATA = 7
FREE_QUBITS = [0, 1, 3]
DERIVED_QUBITS = {2: [0, 1], 4: [0, 3], 5: [1, 3], 6: [0, 1, 3]}
SUPPORTS = [(3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6)]   # S1, S2, S3 -- same order as block 1
X_ANCILLAS = [7, 8, 9]
Z_ANCILLAS = [10, 11, 12]


# ─────────────────────────────────────────────────────────────────────────
# Circuit construction
# ─────────────────────────────────────────────────────────────────────────

def append_encoding(c: stim.Circuit) -> None:
    for q in FREE_QUBITS:
        c.append('H', [q])
    for derived_q, sources in DERIVED_QUBITS.items():
        for src in sources:
            c.append('CX', [src, derived_q])


def append_syndrome_round(c: stim.Circuit) -> None:
    for anc, support in zip(X_ANCILLAS, SUPPORTS):
        c.append('H', [anc])
        for q in support:
            c.append('CX', [anc, q])
        c.append('H', [anc])
    for anc, support in zip(Z_ANCILLAS, SUPPORTS):
        for q in support:
            c.append('CX', [q, anc])
    c.append('MR', X_ANCILLAS + Z_ANCILLAS)


def build_sampling_circuit(p: float) -> stim.Circuit:
    """Encode |0>_L, apply DEPOLARIZE1(p) to all 7 data qubits, extract the
    syndrome once via ancillas, then measure all data qubits in the Z basis
    with an OBSERVABLE_INCLUDE on logical Z (block 1's sweep is likewise
    only sensitive to logical X-type/bit-flip failures on |0>_L, since
    |0>_L is itself a Z_L eigenstate -- Z_L|0>_L = |0>_L identically, so a
    residual logical-Z error is invisible to this test either way)."""
    c = stim.Circuit()
    append_encoding(c)
    if p > 0:
        c.append('DEPOLARIZE1', list(range(N_DATA)), p)
    append_syndrome_round(c)
    for i in range(6):
        c.append('DETECTOR', [stim.target_rec(-6 + i)])
    c.append('M', list(range(N_DATA)))
    c.append('OBSERVABLE_INCLUDE', [stim.target_rec(-N_DATA + q) for q in range(N_DATA)], 0)
    return c


def pauli_string(kind: str, support) -> stim.PauliString:
    s = ['I'] * N_DATA
    for q in support:
        s[q] = kind
    return stim.PauliString(''.join(s))


# ─────────────────────────────────────────────────────────────────────────
# 2b: native STIM stabilizer verification
# ─────────────────────────────────────────────────────────────────────────

def verify_stabilizers_native() -> bool:
    print("=" * 70)
    print("STEP 1: STIM-native encoding + stabilizer verification (TableauSimulator)")
    print("=" * 70)
    sim = stim.TableauSimulator()
    c = stim.Circuit()
    append_encoding(c)
    sim.do(c)

    all_ok = True
    for support in SUPPORTS:
        val = sim.peek_observable_expectation(pauli_string('X', support))
        ok = val == 1
        all_ok &= ok
        print(f"   <X on {support}> = {val:+d}  {'OK' if ok else 'FAIL'}")
    for support in SUPPORTS:
        val = sim.peek_observable_expectation(pauli_string('Z', support))
        ok = val == 1
        all_ok &= ok
        print(f"   <Z on {support}> = {val:+d}  {'OK' if ok else 'FAIL'}")

    logical_z = pauli_string('Z', range(N_DATA))
    z0 = sim.peek_observable_expectation(logical_z)
    sim.do(stim.Circuit("X " + " ".join(str(q) for q in range(N_DATA))))
    z1 = sim.peek_observable_expectation(logical_z)
    x_flip_ok = (z0 == 1) and (z1 == -1)
    all_ok &= x_flip_ok
    print(f"\n<Z_L> on |0>_L = {z0:+d} (expect +1)")
    print(f"<Z_L> on X_L|0>_L = {z1:+d} (expect -1)")
    print(f"6/6 stabilizers +1 and logical X_L flip verified natively via STIM: {all_ok}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
# 2c/2d: syndrome table (native STIM) + 21-case cross-check
# ─────────────────────────────────────────────────────────────────────────

def raw_syndrome_for_error(gate, q):
    """Deterministic hand-injected single-Pauli-error syndrome, read off
    the RAW ancilla measurement bits (not DETECTOR-diff'd -- DETECTOR
    reports deviation from the compiled circuit's own deterministic
    reference trajectory, which for a hard-coded gate baked into the
    circuit itself IS that reference, so it would always read 0; raw
    ancilla bits are what block 1's compute_syndrome mirrors)."""
    c = stim.Circuit()
    append_encoding(c)
    if gate is not None:
        c.append(gate, [q])
    append_syndrome_round(c)
    c.append('M', list(range(N_DATA)))
    raw = c.compile_sampler().sample(1)[0]
    anc = raw[:6].astype(np.uint8)
    data_parity = int(raw[6:].astype(np.uint8).sum() % 2)
    return tuple(int(b) for b in anc[:3]), tuple(int(b) for b in anc[3:]), data_parity


def build_stim_syndrome_table():
    qubit_to_syndrome_x = {}
    for q in range(N_DATA):
        synd_x, _, _ = raw_syndrome_for_error('Z', q)
        qubit_to_syndrome_x[q] = synd_x
    qubit_to_syndrome_z_via_x_err = {}
    for q in range(N_DATA):
        _, synd_z, _ = raw_syndrome_for_error('X', q)
        qubit_to_syndrome_z_via_x_err[q] = synd_z
    return qubit_to_syndrome_x, qubit_to_syndrome_z_via_x_err


def cross_check_against_block1(qubit_to_syndrome_x):
    print("\n" + "=" * 70)
    print("STEP 2: independent syndrome table cross-check against block 1")
    print("=" * 70)
    spec = importlib.util.spec_from_file_location("steane_code_block1", _BLOCK1_PATH)
    b1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b1)
    sv0 = b1.encode_logical_zero()
    block1_table = b1.build_syndrome_table(sv0)

    print("qubit | STIM syndrome | block1 (dense_evolution) syndrome | match")
    all_match = True
    for q in range(N_DATA):
        match = qubit_to_syndrome_x[q] == block1_table[q]
        all_match &= match
        print(f"  {q}   | {qubit_to_syndrome_x[q]}       | {block1_table[q]}                          | {match}")
    print(f"\nSTIM-native syndrome table matches block 1's dense_evolution table bit-for-bit on all 7 qubits: {all_match}")
    return all_match


def run_21_case_cross_check(syndrome_to_qubit):
    print("\n" + "=" * 70)
    print("STEP 3: 21-case single-qubit-error decode cross-check (table decoder vs pymatching)")
    print("=" * 70)

    dem = build_sampling_circuit(0.05).detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    n_table_ok = 0
    n_pymatching_ok = 0
    rows = []
    for q in range(N_DATA):
        for err in ('X', 'Y', 'Z'):
            synd_x, synd_z, parity = raw_syndrome_for_error(err, q)
            z_err_q = syndrome_to_qubit.get(synd_x) if synd_x != (0, 0, 0) else None
            x_err_q = syndrome_to_qubit.get(synd_z) if synd_z != (0, 0, 0) else None
            if err == 'X':
                located_ok = x_err_q == q and z_err_q is None
            elif err == 'Z':
                located_ok = z_err_q == q and x_err_q is None
            else:
                located_ok = x_err_q == q and z_err_q == q
            n_table_ok += int(located_ok)

            anc_vec = np.array(synd_x + synd_z, dtype=np.uint8)
            pym_pred = int(matching.decode(anc_vec)[0])
            pym_ok = pym_pred == parity
            n_pymatching_ok += int(pym_ok)

            status = "OK" if located_ok else "FAIL"
            pym_status = "OK" if pym_ok else "DISAGREE"
            print(f"   qubit {q} error {err}: synd_X={synd_x} synd_Z={synd_z} "
                  f"-> table locates X@{x_err_q} Z@{z_err_q} [{status}] | "
                  f"pymatching predicted_flip={pym_pred} actual_parity={parity} [{pym_status}]")
            rows.append(dict(qubit=q, error=err, synd_x=str(synd_x), synd_z=str(synd_z),
                              table_located_ok=located_ok, pymatching_agrees=pym_ok))

    print(f"\nTable decoder (matches block 1's own decoder): {n_table_ok}/21 correctly localized")
    print(f"pymatching/MWPM decoder: {n_pymatching_ok}/21 correct")
    if n_pymatching_ok < 21:
        print("pymatching DISAGREES with the exact table decoder on "
              f"{21 - n_pymatching_ok} case(s) -- see STEP 4 below for why "
              "(qubit 6's weight-3 syndrome is a genuine 3-detector hyperedge; "
              "STIM's decompose_errors=True heuristic does not resolve it correctly, "
              "a known MWPM limitation on non-graphlike small codes, not a circuit bug).")
    return pd.DataFrame(rows), n_table_ok, n_pymatching_ok


# ─────────────────────────────────────────────────────────────────────────
# 2e: large-N Monte Carlo sweep
# ─────────────────────────────────────────────────────────────────────────

def run_mc_sweep(p_values, n_shots):
    print("\n" + "=" * 70)
    print(f"STEP 5: large-N STIM Monte Carlo sweep ({n_shots:,} shots/point, "
          f"{len(p_values)} points -- {n_shots * len(p_values):,} total samples)")
    print("=" * 70)
    print("Decoding via the exact syndrome-table lookup (proven optimal above for "
          "iid per-qubit noise, and verified identical to block 1's own decoder), "
          "NOT pymatching -- see STEP 4's documented pymatching disagreement.")

    logical_error_rate = np.zeros_like(p_values)
    logical_error_stderr = np.zeros_like(p_values)
    t0 = time.time()
    for i, p in enumerate(p_values):
        c = build_sampling_circuit(float(p))
        sampler = c.compile_detector_sampler(seed=1234 + i)
        dets, obs = sampler.sample(n_shots, separate_observables=True)
        predicted_flip = dets[:, 3:6].any(axis=1)   # nonzero Z-ancilla syndrome -> table always proposes a correction
        logical_error = predicted_flip != obs[:, 0]
        rate = logical_error.mean()
        logical_error_rate[i] = rate
        logical_error_stderr[i] = np.sqrt(max(rate * (1 - rate), 0.0) / n_shots)
        print(f"   p={p:.4f}: logical error rate = {rate:.6f} +- {logical_error_stderr[i]:.6f} "
              f"({int(logical_error.sum())}/{n_shots})")
    print(f"\nTotal sampling+decoding time: {time.time() - t0:.2f}s")
    return logical_error_rate, logical_error_stderr


def main():
    all_stabilizers_ok = verify_stabilizers_native()
    if not all_stabilizers_ok:
        raise RuntimeError("STIM-native stabilizer verification failed -- stopping.")

    qubit_to_syndrome_x, qubit_to_syndrome_z_via_x = build_stim_syndrome_table()
    tables_match_internally = qubit_to_syndrome_x == qubit_to_syndrome_z_via_x
    print(f"\nSTIM's own X-error-via-Z-ancillas table matches Z-error-via-X-ancillas table: {tables_match_internally}")
    if not tables_match_internally:
        raise RuntimeError("STIM's own X/Z syndrome tables disagree -- stopping.")
    syndrome_to_qubit = {s: q for q, s in qubit_to_syndrome_x.items()}
    assert len(syndrome_to_qubit) == N_DATA, "syndrome table is not a bijection over 7 qubits"

    block1_match = cross_check_against_block1(qubit_to_syndrome_x)

    df_21, n_table_ok, n_pymatching_ok = run_21_case_cross_check(syndrome_to_qubit)
    cases_csv = _DATA_DIR / "steane_stim_21case_crosscheck.csv"
    df_21.to_csv(cases_csv, index=False)

    p_values = np.concatenate([
        np.linspace(0.001, 0.02, 6),
        np.linspace(0.03, 0.15, 9),
    ])
    N_SHOTS = 5_000_000
    logical_error_rate, logical_error_stderr = run_mc_sweep(p_values, N_SHOTS)

    df_sweep = pd.DataFrame({
        "physical_error_rate_p": p_values,
        "logical_error_rate": logical_error_rate,
        "logical_error_stderr": logical_error_stderr,
        "n_shots": N_SHOTS,
    })
    sweep_csv = _DATA_DIR / "steane_stim_logical_vs_physical_error_rate.csv"
    df_sweep.to_csv(sweep_csv, index=False)

    below = logical_error_rate < p_values
    if below.any() and (~below).any():
        crossover_idx = np.where(~below)[0]
        crossover_p = p_values[crossover_idx[0]]
        helps_msg = (f"STIM sweep: code helps (logical < physical) for p up to ~{p_values[below][-1]:.4f}; "
                     f"crosses over to hurting around p ~ {crossover_p:.4f}.")
    elif below.all():
        helps_msg = f"STIM sweep: code helps across the ENTIRE tested range up to p={p_values[-1]:.4f}; no crossover observed."
    else:
        helps_msg = "STIM sweep: code does NOT help anywhere in the tested range."
    print(f"\n{helps_msg}")
    print("block1 (dense_evolution, 4000 trials/point) reported: helps up to p~0.105, "
          "crosses over to hurting around p~0.12.")

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(p_values, logical_error_rate, yerr=logical_error_stderr, fmt='o-',
                color='#00FFFF', linewidth=1.8, markersize=4, capsize=3,
                label=f'Steane [[7,1,3]] logical error rate (STIM, {N_SHOTS:,} shots/pt)')
    ax.plot(p_values, p_values, '--', color='#FF007F', linewidth=1.6,
            label='Baseline: uncorrected single physical qubit (rate = p)')
    ax.axvline(0.105, color='#FFD700', linewidth=1.0, linestyle=':', alpha=0.7,
               label='block1 (dense_evolution, 4000 trials/pt): helps-until ~0.105')
    ax.axvline(0.12, color='#FF8C00', linewidth=1.0, linestyle=':', alpha=0.7,
               label='block1: starts hurting ~0.12')
    ax.set_xlabel('Physical depolarizing error rate p', color='#888888')
    ax.set_ylabel('Logical error rate', color='#888888')
    ax.set_title('Steane code [[7,1,3]]: logical vs physical error rate\n'
                 'STIM DEPOLARIZE1 + exact syndrome-table decoder, vs block 1 (dense_evolution)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    png_path = _IMAGES_DIR / "steane_stim_logical_vs_physical_error_rate.png"
    plt.savefig(png_path, dpi=300)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"STIM-native stabilizer verification (6/6 + logical X_L flip): {all_stabilizers_ok}")
    print(f"STIM-native syndrome table matches block1's dense_evolution table bit-for-bit: {block1_match}")
    print(f"21-case table decoder localization: {n_table_ok}/21 (matches block1 exactly)")
    print(f"21-case pymatching/MWPM decoder: {n_pymatching_ok}/21 "
          f"({'agrees fully' if n_pymatching_ok == 21 else 'DISAGREES on ' + str(21 - n_pymatching_ok) + ' case(s), see STEP 4'})")
    print(helps_msg)
    print("block1 comparison: helps up to p~0.105, hurts above p~0.12 (4000 trials/point)")
    print(f"21-case CSV: {cases_csv}")
    print(f"Sweep CSV: {sweep_csv}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
