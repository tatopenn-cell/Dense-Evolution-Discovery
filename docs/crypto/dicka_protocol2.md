# DICKA Protocol 2: the full multi-round structure

The third protocol in the `crypto-q` section (issue #189): the complete multi-round DICKA procedure from Ribeiro, Murta & Wehner 2018 (arXiv:1708.00798), Appendix Protocol 2 -- "a more detailed version of Protocol 1". Built on `di_qkd_ghz.py`'s already-validated single-round primitives (GHZ(3) state, measurements, depolarizing noise, the Parity-CHSH game, the all-Z key round); `dicka_protocol2.py` adds the round-selection and parameter-estimation layer the paper's Protocol 2 specifies around them.

## What a round does

Each of `n` rounds: prepare a fresh GHZ(3) state, then Alice draws `T_i ~ Bernoulli(gamma)` and announces it.

- `T_i=0` (probability `1-gamma`): a **key round** -- everyone measures Z.
- `T_i=1` (probability `gamma`): a **test round** -- the Parity-CHSH game (Definition 8), with the other Bob's question fixed at 1 as established in `di_qkd_ghz.py`.

After all `n` rounds, the protocol computes the observed test-round winning frequency `p_hat` and **aborts** if `p_hat` falls below a threshold `beta` chosen in `]3/4, 1/2+1/(2 sqrt2)[` (between the classical bound and the quantum maximum).

## What this does not reproduce, and why

Theorem 4's exact secure key length `l` depends on `f-tilde(beta)`, a bound on the single-round von Neumann entropy that the paper's own Lemma 3 defines as the unique tangent line to a convex function at a numerically-optimized point `p_opt` -- not a closed-form expression, but the output of a separate optimization the paper carries out on its own. Reporting a number for it here would mean inventing something the paper itself doesn't give in closed form, which goes against how every other result in this RFC has been grounded (see the two bugs documented in `docs/crypto/di_qkd_ghz.md`, both resolved by going back to the paper's literal text rather than assuming).

What **is** reproduced, against the real simulator: the round selection, the honest-implementation winning frequency (parameter estimation), the abort decision, and the raw key's QBER before error correction -- everything the protocol specifies as a physical procedure, stopping short of the privacy-amplification key-rate number.

## Results (N=2000 rounds, gamma=0.7)

Threshold `beta=0.8018` (halfway between classical bound 0.75 and quantum max 0.8536):

| p_dep | p_hat | Decision | QBER(A,B1) | QBER(A,B2) |
|---|---|---|---|---|
| 0.00 | 0.8470 | proceed | 0.0000 | 0.0000 |
| 0.05 | 0.7974 | **abort** | 0.0754 | 0.0578 |
| 0.10 | 0.7603 | abort | 0.1525 | 0.1284 |
| 0.15 | 0.7175 | abort | 0.1958 | 0.1766 |

The abort at `p_dep=0.05` lands right where it should: `expected_win_rate(0.05) = 0.7977` (from `di_qkd_ghz.py`, itself validated to `|z|<0.5` against the simulator) is already just under this `beta`. Raising `beta` closer to the quantum max (0.8436) makes the protocol strictly more sensitive to noise, aborting at the same noise levels with a larger margin.

## Reproduce it

```bash
python scripts/crypto/dicka_protocol2.py
```

`tests/test_dicka_protocol2.py` covers the same scenarios (proceed on an ideal channel, abort under enough noise, QBER rising with noise) at smaller `N` with fixed seeds.
