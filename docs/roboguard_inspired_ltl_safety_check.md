# A RoboGuard-Inspired Two-Stage LTL Safety Check, With Claude Instead of OpenAI

`prog.txt`'s roadmap named the "contextual/semantic" safety layer -- checking whether a
robot's *plan* violates a natural-language safety rule, not just whether a single command
violates a kinematic or spatial bound -- as unexplored territory beyond `rate_limiter`
and `cbf_filter`. This experiment builds a real, working instance of that layer, inspired
by RoboGuard (Ravichandran, Robey, Kumar, Pappas & Hassani, "Safety Guardrails for
LLM-Enabled Robots", arXiv:2503.07885, IEEE RA-L -- fetched and verified directly,
indexed in quantumrag), with two real deviations from the paper's own reference
implementation, both disclosed up front.

## Step 0. Two real blockers found and worked around, honestly

RoboGuard's own `setup.py` (`KumarRobotics/RoboGuard`, fetched directly) declares
`install_requires=["openai", "spot"]`. Two real problems, found by checking rather than
assuming:

1. **`pip install spot` installs the WRONG package.** PyPI's `spot` is an unrelated
   "DotCloud environment loader" (a real namespace collision) -- confirmed by inspecting
   the installed package directly (`pip show spot` -> `Home-page: http://github.com/3kwa/spot`,
   nothing to do with LTL). Uninstalled immediately. The real Spot (LRE EPITA's LTL/
   omega-automata library) is distributed via conda-forge, not PyPI under that name.
2. **`openai` is a hard dependency in the reference implementation**, requiring a paid
   API key for the "root-of-trust LLM" chain-of-thought step. Per explicit instruction,
   that step is done directly by Claude (this session) instead -- the exact same role
   the paper assigns to an LLM, shown explicitly below rather than hidden behind an API
   call, and genuinely free.

## Step 1. Real environment: Spot via conda-forge, inside Docker

No NVIDIA GPU on this machine (already established for the CBF/SAFER-Splat experiment),
and `spot`'s Windows support is unconfirmed via conda-forge -- reused the same Docker
infrastructure already proven for the ROS2 experiment. Conda's newer non-interactive
Terms-of-Service gate on the default channels was avoided by installing from
`conda-forge` exclusively (`--override-channels -c conda-forge`), not by accepting terms
on channels not actually needed. Real Spot 2.16 installed and importable inside an
`ubuntu:22.04` container.

## Step 2. Stage 1 -- Claude's own chain-of-thought translation (shown, not hidden)

Natural-language safety rule, in RoboGuard's own paper style (e.g. "don't move a candle
below a balloon"): **"Never carry an object while inside the restricted zone."**

Translation to LTL, done directly here:
- Atomic proposition `holding`: the gripper is currently holding an object.
- Atomic proposition `in_zone`: the end-effector is currently inside the restricted zone.
- The rule forbids `holding AND in_zone` at every point in time.
- LTL: **`G(holding -> !in_zone)`** ("Globally, holding implies not in_zone").

## Step 3. Stage 2 -- real Spot model checking, three real bugs found and fixed

The standard automata-theoretic recipe for "does word W satisfy formula F": build the
automaton for `!F`, take the product with W's automaton, check emptiness. Getting there
took three real, sequential mistakes, each caught by checking against a hand-verified
truth table rather than trusting the first plausible-looking result:

1. First attempt used `spot.contains(formula_automaton, word_automaton)` and a bare `"1"`
   literal for "no propositions true" in the word string. Appeared to give a directionally
   sensible answer on one case, but a *constructed, guaranteed-safe* trace (holding and
   in_zone never simultaneously true, verified directly in plain Python) was reported as
   violating the formula -- a contradiction that would have gone unnoticed without
   explicitly testing the "should be safe" direction, not just the "should violate" one.
2. Investigated: `spot.parse_word("1;cycle{1}")` produces a word automaton with **zero**
   declared atomic propositions (confirmed via its own HOA dump) -- the bare `"1"` never
   names `holding`/`in_zone`, so the word and formula automata were being compared over
   mismatched alphabets, making any `contains()` result meaningless. Fixed by always
   stating both propositions explicitly (`!holding&!in_zone` instead of `1`).
3. Still wrong after that fix. Rather than guess a third time at `contains()`'s argument
   order, replaced it with the standard, unambiguous emptiness-check technique above --
   verified against three hand-checked minimal cases (a safe word, an always-violating
   word, a word that's safe then violates) before trusting it on any real data.

## Step 4. Real result, on real SO-101 joint data

`holding` is derived from joint 5 of the real SO-101 action stream (a narrow real range,
0.08-20.7, plausibly the gripper channel) via `holding = joint5 > median(joint5)` -- an
**assumed proxy, disclosed as such**, not confirmed via real task metadata (`meta/tasks.jsonl`
was checked directly and does not exist at the expected path for this dataset). `in_zone`
reuses the same real-position/synthetic-region convention as the geometric CBF experiment.

```
Real trace: n=303 points, holding at 100 points, in_zone at 231 points, both simultaneously at 86 points

LTL safety formula (Claude's CoT translation, stage 1): G(holding -> !in_zone)

STAGE 2 (real spot model checking) on the REAL trace: satisfies formula = False
Cross-check (direct Python, should match): no simultaneous holding+in_zone = False

Sanity check on a DELIBERATELY violating trace (forced holding+in_zone at one point): satisfies formula = False (must be False)

Sanity check on a constructed SAFE trace (holding forced False whenever in_zone is True): satisfies formula = True (must be True), overlap points = 0 (must be 0)
```

The real trace violates the safety rule at 86 real points -- real Spot model checking
agrees exactly with a direct Python cross-check. Both directional sanity checks pass:
a deliberately violating trace is correctly rejected, a constructed safe trace is
correctly accepted -- confirming the check genuinely discriminates, not a trivial
always-False (or always-True) bug.

## Result

A real, working two-stage safety check -- natural-language rule to LTL (Claude, free, no
API key), LTL to a real model-checking verdict on a real robot trace (Spot, verified
correct against hand-built cases in both directions). Complements `rate_limiter.py`
(kinematic) and `cbf_filter.py` (spatial) with a semantic/task-level layer: does the
overall behavior violate a rule expressed in plain language, not just a single
instantaneous command.

---

## Details

**Why joint 5, not a confirmed gripper label**: disclosed directly in Step 4 -- this is
the best available real signal given the dataset's own metadata gap, not a verified
ground-truth semantic. A real deployment would need this confirmed, not assumed.

**Not RoboGuard's full pipeline**: this experiment implements the paper's core two-stage
*mechanism* (LLM contextualization -> LTL -> model checking) on one real rule and one real
trace, not RoboGuard's own benchmark suite, SPINE integration, or its own reported
92%->2.5% unsafe-plan reduction (that number belongs to their own paper's full system and
evaluation, not reproduced here).

**Reproducing this**: requires Docker (see the ROS2 experiment for the setup that first
established this infrastructure on this machine). `docker run --rm -v
"<path-to-this-folder>:/ws:ro" ubuntu:22.04 bash /ws/run_ltl_check.sh` installs Miniconda,
installs real Spot from conda-forge (`--override-channels` to avoid the ToS gate on
unneeded default channels), and runs `ltl_safety_check.py` -- which downloads the
already-familiar real LeRobot parquet (cached from prior experiments where available) and
prints the exact output above. Not part of this repo's normal pytest CI (same as the
ROS2 Docker experiment) -- a real, manually-reproducible artifact, not a claim of
continuous automated verification.

**Paper indexed**: RoboGuard (arXiv:2503.07885) was already indexed in quantumrag's
`robotica_filtri_sicurezza_semantica` collection from the earlier DeepSeek-analysis
verification pass; this experiment is the first real code built against it.
