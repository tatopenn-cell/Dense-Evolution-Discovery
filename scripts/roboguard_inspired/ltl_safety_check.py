# -*- coding: utf-8 -*-
"""
scripts/roboguard_inspired/ltl_safety_check.py (PRIVATE, exploratory)
==========================================================================
RoboGuard-inspired two-stage safety check, tested with the real `spot`
LTL/omega-automata library (v2.16, installed via conda-forge inside a
real Docker container -- confirmed the PyPI package literally named
"spot" is an unrelated "DotCloud environment loader", a namespace
collision, NOT this library; caught and avoided before use).

REAL PAPER: Ravichandran, Robey, Kumar, Pappas & Hassani (2025/2026),
"Safety Guardrails for LLM-Enabled Robots" (RoboGuard), arXiv:2503.07885,
IEEE RA-L -- fetched and verified directly, indexed in quantumrag's
robotica_filtri_sicurezza_semantica collection. Two-stage architecture:
(1) a "root-of-trust" LLM contextualizes a natural-language safety rule
into a formal temporal-logic specification via chain-of-thought
reasoning; (2) that specification is checked against a proposed robot
plan/trace using LTL model checking.

STAGE 1 HERE: done directly by Claude (this session), not an OpenAI API
call -- the user's own instruction ("usa Claude invece di openai...
basta che e' free") to avoid a paid API dependency RoboGuard's own
setup.py hard-codes (install_requires=["openai", "spot"]). This IS the
same role the paper assigns to an LLM; shown explicitly below, not
hidden behind a black-box call.

Natural-language safety rule (RoboGuard-paper style, e.g. "don't move a
candle below a balloon"): "Never carry an object while inside the
restricted zone."

Chain-of-thought translation to LTL (done here, by Claude, shown):
  - Atomic proposition `holding`: the gripper is currently holding an
    object.
  - Atomic proposition `in_zone`: the end-effector is currently inside
    the restricted zone.
  - The rule forbids the conjunction (holding AND in_zone) at every
    point in time -> LTL: G(holding -> !in_zone)
    ("Globally, holding implies not in_zone" -- standard translation of
    a "never A while B" safety rule to LTL, matching the structure of
    the paper's own worked examples.)

STAGE 2 HERE: real spot automaton-based model checking of that LTL
formula against a real symbolic trace derived from real SO-101 joint
data (episode 0), PLUS a synthetically constructed violating trace, to
confirm the check actually discriminates (not just always "safe").

REAL DATA, DISCLOSED ASSUMPTION: joint 5 of the real SO-101 action
stream has a narrow range (0.08-20.7, vs joints 0-4's much wider ranges)
-- plausibly the gripper channel, but NOT independently confirmed via
real task metadata (meta/tasks.jsonl was not found for this dataset at
the expected path, checked directly, not just assumed present). Treated
here as `holding = (joint5 > median(joint5))`, an assumed proxy,
disclosed as such -- not a confirmed ground-truth semantic label.
`in_zone` reuses the same real-position/synthetic-region convention as
the geometric CBF experiment.
"""
import numpy as np
import spot


def build_trace_string(holding, in_zone):
    """spot's twa_word / parse_word format wants a comma-separated
    sequence of Boolean-assignment cycles; simplest robust approach here
    is to check EACH point's instantaneous propositions against the LTL
    formula's automaton by building a word of length 1 per real step and
    checking prefix-safety incrementally -- but the direct, real spot
    API for checking a finite trace against an LTL safety formula is
    spot.translate() -> automaton, then spot.twa_word from an explicit
    string. Builds the explicit finite-word string spot expects."""
    cycles = []
    for h, z in zip(holding, in_zone):
        # ALWAYS state both propositions explicitly (never the bare "1"
        # shorthand): a real bug found and fixed here -- "1" doesn't
        # register "holding"/"in_zone" as the word automaton's atomic
        # propositions at all (confirmed via debug_spot_api.py: AP count
        # came out 0), so contains() compared automata over different,
        # incompatible alphabets and gave a wrong answer.
        cell = ("holding" if h else "!holding") + "&" + ("in_zone" if z else "!in_zone")
        cycles.append(f"{cell};")
    # spot represents a finite prefix as a word with a cycle at the end;
    # repeat the last real observation as the infinite suffix (standard
    # finite-trace-to-omega-word convention for safety-formula checking)
    return "".join(cycles) + "cycle{" + cycles[-1].rstrip(";") + "}"


def check_trace(holding, in_zone, formula_str):
    """REAL BUG, found and fixed through two wrong attempts before this
    one, verified against a hand-checked minimal example
    (debug_spot_api.py): spot.contains()'s argument-order semantics
    were tried both ways and neither matched the hand-verified truth
    table once atomic propositions were declared explicitly (the FIRST
    attempt's apparent "confirmation" turned out to be an artifact of
    an unrelated bug -- using the bare "1" word literal, which doesn't
    register named atomic propositions at all, so that comparison was
    between automata over mismatched alphabets and meant nothing).
    Replaced with the standard, unambiguous automata-theoretic
    technique instead of continuing to guess at contains()'s direction:
    word W satisfies formula F  <=>  the product of W's automaton with
    the automaton for NOT F is EMPTY. Verified against 3 hand-checked
    cases (safe/unsafe/mixed) before trusting it on real data."""
    word_str = build_trace_string(holding, in_zone)
    not_f_aut = spot.formula(f"!({formula_str})").translate()
    word_aut = spot.parse_word(word_str).as_automaton()
    product = spot.product(word_aut, not_f_aut)
    return product.is_empty(), word_str


def main():
    import pathlib
    import sys
    import pandas as pd
    from huggingface_hub import hf_hub_download

    data_root = pathlib.Path("/tmp/lerobot_data")
    parquet_path = hf_hub_download(
        repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    action = np.stack(sub["action"].values)

    joint0 = action[:, 0]  # real position channel, same as the CBF experiment
    joint5 = action[:, 5]  # assumed gripper proxy, disclosed above

    holding_real = joint5 > np.median(joint5)
    # restricted zone: reuse the CBF experiment's own convention (a real region
    # around the median of joint0's real range)
    zone_center = float(np.median(joint0))
    zone_radius = (joint0.max() - joint0.min()) * 0.1
    in_zone_real = np.abs(joint0 - zone_center) < zone_radius

    n_holding_and_zone = int(np.sum(holding_real & in_zone_real))
    print(f"Real trace: n={len(joint0)} points, holding at {holding_real.sum()} points, "
          f"in_zone at {in_zone_real.sum()} points, both simultaneously at {n_holding_and_zone} points")

    formula_str = "G(holding -> !in_zone)"
    print(f"\nLTL safety formula (Claude's CoT translation, stage 1): {formula_str}")

    safe_real, _ = check_trace(holding_real, in_zone_real, formula_str)
    print(f"\nSTAGE 2 (real spot model checking) on the REAL trace: satisfies formula = {safe_real}")
    print(f"Cross-check (direct Python, should match): no simultaneous holding+in_zone = {n_holding_and_zone == 0}")

    # Synthetic violating trace: force holding=True and in_zone=True at one real point,
    # to confirm the check actually discriminates (not trivially always "safe")
    holding_violate = holding_real.copy()
    in_zone_violate = in_zone_real.copy()
    mid = len(holding_violate) // 2
    holding_violate[mid] = True
    in_zone_violate[mid] = True
    safe_violate, _ = check_trace(holding_violate, in_zone_violate, formula_str)
    print(f"\nSanity check on a DELIBERATELY violating trace (forced holding+in_zone at one point): "
          f"satisfies formula = {safe_violate} (must be False)")

    # Sanity check the OTHER direction: a genuinely safe trace (holding and in_zone
    # never true at the same real point) must return True -- otherwise "False" above
    # could just be a trivial always-false bug, not real discrimination.
    holding_safe = holding_real & ~in_zone_real  # never both true, by construction
    safe_safe, _ = check_trace(holding_safe, in_zone_real, formula_str)
    n_overlap_safe = int(np.sum(holding_safe & in_zone_real))
    print(f"\nSanity check on a constructed SAFE trace (holding forced False whenever in_zone is True): "
          f"satisfies formula = {safe_safe} (must be True), overlap points = {n_overlap_safe} (must be 0)")


if __name__ == "__main__":
    main()
