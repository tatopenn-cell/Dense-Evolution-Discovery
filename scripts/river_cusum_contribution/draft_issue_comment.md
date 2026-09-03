Hi @MaxHalford, I'd like to take on **CUSUM (Page 1954)** from Family 1 -- the fixed-reference classical control chart specifically, not a duplicate of `PageHinkley`.

I checked `PageHinkley`'s own docstring first ("implements the CUSUM control chart") before assuming this was a real gap: it standardizes against a **fading, exponentially-forgetting mean** (`alpha`, updated every step), which is a genuinely different scheme from Page's original -- a **fixed** reference mean/std, estimated once from an in-control baseline and never updated, so a sustained shift keeps accumulating against the original target indefinitely instead of fading as the window catches up. river doesn't have that variant yet.

A prototype and an honest evaluation (following @jevwithwind's harness discussion in this issue and #1963):

- The commonly-cited "textbook" tuning (k=0.5, h=5.0) turned out to be a real trap for stream lengths typical here: ~88% stream-level false-alarm rate on a purely stable 1000-sample N(0,1) stream (200 trials), because its average run length under no-change is only ~19-38 samples. Empirically recalibrated to h=20.0 (3% stream-level false-alarm rate, 0% missed-detection for shift>=1.0). Cross-checked this wasn't a one-off: I maintain a separate library (Dense-Armor, unrelated to this PR) with the same algorithm in batch form, also shipping h=5.0 as its "textbook" default -- same test, same result (100% stream-level false-alarm rate), now fixed there too. Two independent implementations reaching for the same uncritical citation is a decent signal this is worth flagging generally, not just here.
- Stress-tested the evaluation harness itself against `AlwaysFire`/`NeverFire`/`DummyDriftDetector(t_0=100)` dummy baselines, prompted by @mateenali66's comment above about a real detector landing below a data-ignoring one -- an early version of my own harness had exactly that bug (false alarms counted per-STREAM instead of per-sample, so a detector firing on every point scored the same as one firing once). Fixed; CUSUM now clears every dummy baseline by a wide margin at every shift size tested (F1 0.42-0.84 vs 0.0-0.12 for the dummies).
- Honest comparison against ADWIN/KSWIN/PageHinkley at their own library defaults (no tuning in anyone's favor), F1 by shift size:

  | shift | CUSUM | ADWIN | PageHinkley | KSWIN |
  |---|---|---|---|---|
  | 0.5σ | 0.424 | **0.847** | 0.797 | 0.359 |
  | 1.0σ | 0.835 | **1.000** | 0.799 | 0.527 |
  | 1.5σ | 0.842 | **1.000** | 0.802 | 0.539 |
  | 2.0σ | 0.842 | **1.000** | 0.802 | 0.539 |

  ADWIN is strictly better everywhere. CUSUM beats PageHinkley on F1 for medium/large shifts (mainly detection **speed**: roughly 2-3x faster than ADWIN at a 2-sigma shift, 12.9 vs 34.5 samples), but is clearly the weakest of the three real detectors at a small 0.5-sigma shift -- h=20.0 trades away small-shift sensitivity for a low false-alarm rate, and I don't have a fix for that yet (tried a second, small-k parallel channel; it recovers some small-shift sensitivity but adds false alarms everywhere, net negative). Not proposing this as an ADWIN replacement -- it's a genuinely different point on the speed/precision tradeoff for medium/large shifts specifically, which is presumably why the paper's authors wanted it as a separate baseline in the first place.

Happy to post the full numbers/code for review before opening a PR, and to size it like recent merged PRs (this alone, no bundled benchmark harness -- that's already in flight in #1963). Let me know if that's the right scope, or if you'd rather I wait for #1963 to land first so I can validate against the "official" harness API instead of my own throwaway one.
