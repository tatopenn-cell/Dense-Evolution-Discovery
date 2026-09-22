# DI-QKD via GHZ(3): Protocol 1 of the crypto-q RFC

The second protocol in the `crypto-q` section (issue #189): device-independent conference key agreement (DICKA) for three parties -- Alice, Bob1, and a second Bob -- sharing a GHZ(3) state, following Ribeiro, Murta & Wehner 2018, "Fully device-independent conference key agreement" (arXiv:1708.00798). Built entirely from primitives Dense-Evolution already has: `de.ghz_state(3)`, `DenseSVSimulator`, `GATES["h"]`, `sim.measure`, and `NoiseModel`'s depolarizing channel -- the only new piece is `_diagonalizing_unitary`, which numerically derives (via `eigh`) the change-of-basis unitary for Bob1's two rotated `(Z+X)/sqrt2` and `(Z-X)/sqrt2` measurements, rather than hand-picking a rotation angle.

## The game

The paper's own "Parity-CHSH" inequality (Definition 8) extends CHSH from 2 to N parties. Alice and Bob1 get uniformly random binary questions `x, y`; every other Bob is asked a **fixed** question. Everyone answers a bit; the parity `b̄` of the other Bobs' answers folds into the winning condition:

```
a + b1 = x * (y XOR b̄)  (mod 2)
```

Classical strategies are bounded at `P_win <= 3/4`; the quantum maximum, `~0.8536 = 1/2 + 1/(2 sqrt2)`, is the same bound as ordinary CHSH (conditioning on `b̄` reduces the game to standard CHSH, up to relabelling -- the paper's own Remark 2).

## Two bugs found while implementing this

Both were found the same way: run the real simulator, compare against the paper's own closed-form numbers, and if they disagree, go back to the primary source rather than adjust the code to fit.

**1. The other Bob's fixed test-round question.** Definition 8's text fixes it at "always equal to 1" (X-basis). A first implementation used 0 (Z-basis) and got `P_win=0.6768 = 1/2 + 1/(4 sqrt2)` -- exactly half the expected quantum boost. Switching to question=1 reproduced `0.85355...` to machine precision.

**2. The depolarizing-noise comparison formula.** The paper's Eq. A.65 closed form was first coded assuming a channel that shrinks Pauli expectation values by `(1-p)` per noisy qubit. dense_evolution's own `'depolarizing'` model is the isotropic-Pauli-error Kraus map (`K0=sqrt(1-p)I, K1..3=sqrt(p/3)*Pauli`), which shrinks by `1-4p/3` instead -- the same convention already behind `bb84.py`'s validated `QBER=2p/3` result. Assuming `(1-p)` matched the simulator only at `p=0` and grew to `z=-3.65` by `p=0.15`; rewriting the formula in terms of the correct shrink factor brought every point in the sweep under `|z|=0.5`.

## Results (N=3000 rounds per point)

| Test | Observed | Expected | z |
|---|---|---|---|
| Ideal channel | 0.8527 | 0.8536 | -0.10 |
| Depolarizing p=0.02 | 0.8347 | 0.8305 | +0.46 |
| Depolarizing p=0.05 | 0.8013 | 0.7977 | +0.40 |
| Depolarizing p=0.10 | 0.7470 | 0.7479 | -0.09 |
| Depolarizing p=0.15 | 0.7030 | 0.7036 | -0.07 |

Key-generation rounds (all parties measure Z, `Y=(0,2,0)` per Protocol 2 step (c) -- unaffected by either bug) give QBER 0 at `p=0`, rising with `p` as expected for a noisier channel.

## Reproduce it

```bash
python scripts/crypto/di_qkd_ghz.py
```

`tests/test_di_qkd_ghz.py` covers the same scenarios at a smaller `N` with fixed seeds, checking the win rate clears the classical bound and stays close to the closed form, without pinning exact numbers.
