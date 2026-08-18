# Deferred model experiments

Three candidate changes to the scoring model, none of them implemented. They are recorded here with the
measurements that motivated them so the work can be resumed cold.

Run every comparison through `evaluation/compare_variants.py` — a variant with more parameters will fit the training
corpus better whether or not it generalises, so single-split numbers cannot referee between these.

## Measurements on the shipped models

Taken from `data/en_quadgrams.json` and `data/he_quadgrams.json` before pruning was introduced.

| | EN | HE |
|---|---|---|
| distinct quadgrams | 585,066 | 505,371 |
| quadgram tokens | 670,204,747 | 407,978,444 |
| word tokens | 173,439,345 | 111,150,260 |
| mean word length | 4.86 | 4.67 |
| token-weighted median quadgram count | 131,901 | 31,815 |
| quadgram tokens starting with space (pinned to position 0) | 24.9% | 27.0% |
| quadgram tokens ending with space (pinned to word-final) | 24.9% | 27.0% |
| pure-interior quadgram tokens (position ambiguous) | 54.1% | 49.5% |

Derivation of the word counts: `scripts/build_quadgrams.py` pads every word as `' ' + word + ' '`, so exactly
one bigram per word token starts with a space. Summing those gives the word count; `total_bigrams - W` gives
the summed length. Both cross-check exactly against the trigram and quadgram totals.

## Technical mode — measured

`_score_text_en` returns `max(score_en, score_so)`, so technical mode can only *raise* the English score.
That fixes the sign per corpus before any measurement: on English the raised score is the correct one
(active in the FP test, shadow in the FN test) so both improve, and on Hebrew it is the wrong one in both
directions so both degrade. The open question was only the magnitude, and whether the gain on technical text
justifies the loss.

Measured by `evaluation/technical_mode.py` at delta 4.0, K=5, fold 0. The `so` and `he` models were built
without their test lines; `en` reuses the shipped model and is a floor check, not a clean measurement.

| arm | words | FP/1k std → tech | FN/1k std → tech |
|---|---|---|---|
| so (held out) | 2,137,289 | 0.363 → 0.082 (−77%) | 2.112 → 0.771 (−64%) |
| en | 2,845,199 | 0.295 → 0.215 (−27%) | 3.483 → 1.876 (−46%) |
| he (held out) | 1,636,374 | 0.196 → 0.362 (**+85%**) | 5.349 → 10.106 (**+89%**) |

The mode works: on held-out Stack Overflow text it removes three quarters of the false positives. It also
helps *plain* English by a quarter, which is not what the name suggests — the SO model is not merely a
technical-vocabulary supplement, it is a second English model whose rare tail covers strings CulturaX does
not.

That same permissiveness is the Hebrew cost. Hebrew shadow strings are Latin mojibake, exactly the sort of
unusual character sequence SO's identifier-and-code tail makes plausible, so the inflated English score both
pulls the user out of Hebrew (FP) and holds them on the wrong layout (FN). Roughly doubling both is severe
enough that the mode must stay opt-in and per-context; leaving it on globally is a bad default for anyone who
types Hebrew.

Worth noting for Arm C: this is the same failure shape — an English score that runs hot relative to Hebrew —
reached by a different route.

## Combining the two English models — measured

`max()` is an upper envelope, so the asymmetry above is structural rather than incidental: it can only raise
the English side. The SO model is ~18x smaller under the same add-1 smoothing, so it is flatter and its
penalty for unseen n-grams is milder. Measured directly: over 3,000 random junk strings the SO model wins
61% of the time and inflates the English score by **1.71 nats on average**, against a delta of 4.0.
`evaluation/calibrate_models.py` confirms it on real text — SO runs at −1.29 nats/char with σ 0.57 against
EN's −1.48 with σ 0.75, i.e. both more generous and flatter.

Three alternatives, measured by `evaluation/combination_bench.py` at delta 4.0, K=5, fold 0, 75,000 lines
per arm. `mixture:W` is `log((1−W)·P_en + W·P_so)`, of which `max` is the degenerate limit; `merged:S` sums
the count tables with SO scaled by S (`scripts/merge_en_so_counts.py`), moving interpolation down to the
quadgram context and halving runtime scoring; `calibrated` rescales the SO score onto EN's nats/char scale
before the max.

| variant | so FP/1k | so FN/1k | en FP/1k | en FN/1k | he FP/1k | he FN/1k |
|---|---|---|---|---|---|---|
| standard | 0.363 | 2.112 | 0.295 | 3.483 | 0.196 | 5.349 |
| **max** (shipped) | **0.082** | **0.771** | **0.215** | **1.876** | **0.362** | **10.106** |
| mixture:0.1 | 0.219 | 1.301 | 0.252 | 2.184 | 0.224 | 8.173 |
| mixture:0.2 | 0.183 | 1.227 | 0.258 | 2.192 | 0.240 | 8.321 |
| mixture:0.35 | 0.128 | 1.124 | 0.274 | 2.227 | 0.241 | 8.227 |
| mixture:0.5 | 0.101 | 0.976 | 0.302 | 2.366 | 0.256 | 8.398 |
| mixture:0.65 | 0.109 | 0.980 | 0.343 | 2.548 | 0.258 | 7.705 |
| mixture:0.8 | 0.117 | 0.923 | 0.425 | 2.844 | 0.266 | 6.948 |
| mixture:0.9 | 0.133 | 0.936 | 0.538 | 3.237 | 0.266 | 6.508 |
| merged:1 | 0.265 | 1.574 | 0.297 | 3.516 | 0.198 | 5.345 |
| merged:5 | 0.244 | 1.508 | 0.337 | 3.622 | 0.200 | 5.136 |
| merged:18 | 0.163 | 1.393 | 0.418 | 4.004 | 0.197 | 4.377 |
| calibrated | 0.289 | 1.587 | 0.292 | 3.462 | 0.207 | 5.760 |

**No variant dominates.** Paired block-bootstrap 95% CIs against `max` (`evaluation/sample_size.py`) put
every trade firmly outside zero in *both* directions:

| variant | so FP/1k vs max | so FN/1k vs max | he FP/1k vs max | he FN/1k vs max |
|---|---|---|---|---|
| mixture:0.35 | +55% [+41, +73] | +47% [+32, +67] | −34% [−38, −29] | −19% [−23, −15] |
| mixture:0.5 | +23% [+15, +33] | +27% [+16, +41] | −29% [−33, −26] | −17% [−20, −14] |
| merged:18 | +98% [+77, +122] | +82% [+63, +107] | −46% [−51, −41] | −57% [−61, −53] |

So the choice is a genuine product judgement about how Hebrew damage trades against technical accuracy, not
a measurement question. **Decided: `max()` stays**, on the grounds that its Hebrew cost is already mitigated
by the mode being opt-in and per-context, which makes the best technical accuracy worth keeping. The losing
rules live in `CombinationEngine` in `evaluation/combination_bench.py`, not in `core/` — the production
scoring path should not carry branches for an answered question. Three things the numbers do settle:

- **The mixture weight is not a free parameter above 0.5.** Higher weights keep buying Hebrew (he FN falls to
  +22% at W=0.9) but wreck plain English, where FP/1k runs 0.302 → 0.343 → 0.425 → 0.538 across W = 0.5 →
  0.9, ending up **+83% worse than standard mode**. The usable range is W ≤ 0.5.
- **`merged` buys the most Hebrew and costs the most English.** At S=18 it is the only variant that leaves
  Hebrew *better* than standard mode on FN (4.377 vs 5.349) while still halving SO false positives, but its
  en arm degrades +42% FP — the scaled SO counts swamp a model trained on 18x more text. Its one structural
  advantage is cost: one `score()` call at runtime instead of two.
- **`calibrated` is nearly a no-op.** It removes the flatness bias as intended, and what remains is small in
  both directions (so −20% FP, he +5% FP). This closes Arm C for the EN/SO pair: the inter-model offset is
  real but is *not* the main mechanism — most of `max`'s effect is the envelope itself, not the miscalibration.

### On sample size

`evaluation/sample_size.py` block-bootstraps these arms. The binding constraint is the FP *count*, not the
line count: the so arm at 75,000 lines yields 176 FPs under `max`, carrying ~1/sqrt(176) = 7.5% relative
standard error however many words they are divided by. Precision therefore improves as 1/sqrt(N) with no
knee, and the sample size has to be chosen against a decision:

- **Variant vs standard mode** is resolvable at **3,500 lines** — every such comparison already excludes zero
  there, and the point estimates are converged (standard reads 0.362 FP/1k at 3,500 against 0.363 at 75,000).
- **Variant vs variant** needs the full corpus. Separating `max` from `mixture:0.5` means resolving ~5 points
  of a −77% vs −72% difference; only the *paired* bootstrap does it, because pairing on blocks cancels the
  shared line-to-line difficulty that dominates the unpaired intervals.

Screen at 7,500 lines, confirm finalists at 75,000. The mixture-weight sweep above was run entirely at full
size, which is backwards — the differences among W = 0.5, 0.65, 0.8 on the so arm (0.101 / 0.109 / 0.117)
sit inside the ±14% interval and should not be read as a ranking.

## Arm A — positional n-grams

Condition each estimate on the position of the n-gram within the word, so the model asks "how likely is this
quadgram *at index i*".

**The case against, which should be tested rather than assumed.** The space padding already pins half the
quadgram tokens to a position: `" abc"` can only occur at position 0, `"xyz "` only at word end. With a mean
word length of 4.86, a word carries only **2.09 interior quadgrams** — the sole slots where position adds
anything. In Hebrew the redundancy is doubled: the final forms `ם ן ץ ף ך` encode word-final position in the
orthography itself, and the `ו ה ב כ ל מ ש` prefixes sit at position 0.

The sparsity cost is narrow but lands in the wrong place. Typical typing is unaffected — the token-weighted
median quadgram count is 131,901, so even a 7-way split leaves ~19k observations. But false positives happen
on names, abbreviations and rare morphology, which is exactly the low-count tail where splitting turns a
count of 40 into six counts of ~7.

**If built:**

- Schema: nested, `{"abcd": {"0": 5000, "2": 120}}`, with a `"model_type": "positional"` marker. Nested
  rather than flat `"0:abcd"` keys so global counts marginalize with one `sum()` and interpolation reads both
  estimates from a single lookup.
- Position = index of the n-gram's start in the padded word. Cap at 5 and bucket the remainder.
- **The denominator must also be positional.** `P(c4|c1c2c3, i)` requires `count(c1c2c3 @ i)`. A positional
  numerator over a global denominator is not a probability, and the resulting score shift would be silently
  absorbed into delta tuning — it would look like it works while meaning nothing. Assert
  `sum_c4 quad[c1c2c3c4][i] == tri[c1c2c3][i]` (modulo the word-end edge) in a test.
- **Back off to the global estimate**, `P = λ·P_pos + (1−λ)·P_global` with `λ = n_pos/(n_pos+k)`. λ=0
  reproduces the current model exactly, giving a no-regression floor and the knob that neutralises thin-bin
  variance in the rare-word tail.
- No API change: `core/engine.py` always prepends `' '`, so `score()` receives a word-initial string and the
  position is just the loop index.
- Don't bother making `core/quadgram.py`'s `len == 2` and `len == 3` branches positional — they are
  unreachable in production. `engine.py` prepends a space, `hooks.py` gates mid-word evaluation at ≥3 chars,
  and delimiter evaluation appends a trailing space, so `score()` never sees fewer than 4 characters.
- Size: growth is ~2–3×, not 7× — most quadgrams occur at only a few distinct positions.

## Arm B — Jelinek–Mercer interpolation

Scoring-time only. Same count files, zero size cost. **The most promising of the three.**

Today's Laplace add-1 penalises an unseen continuation in proportion to how *common its context* is:

```
trigram 'the' (12,006,103 occurrences) + unseen 4th char  ->  -16.30 nats
trigram 'ing' ( 6,343,165 occurrences) + unseen 4th char  ->  -15.66 nats
seen quadgram 'the ' (8,970,831)                          ->   -0.29 nats
```

A gap of 16 nats means the model treats the unseen continuation as ~9 million times less likely. A delta of
4.0 is a threshold of e⁴ ≈ 55×. So a single unfamiliar character sequence — a name, a loanword, a typo —
overwhelms the decision by five orders of magnitude. That is a false-positive generator.

Replace with:

```
P(d|abc) = λ₃·P_ML(d|abc) + λ₂·P_ML(d|bc) + λ₁·P_ML(d|c) + λ₀·P_ML(d)
```

where `P_ML(d|abc) = count(abcd)/count(abc)` and the λ's sum to 1. Prefer Witten–Bell context weights,
`λ₃ = c(abc) / (c(abc) + N₁₊(abc))`, where `N₁₊(abc)` is the number of distinct characters ever seen after
`abc` — trust the quadgram when its context is well-attested and diverse, lean on lower orders otherwise.
Precompute `N₁₊` at load time following the `_bigram_first_totals` pattern in `core/quadgram.py`.

**Not stupid backoff.** Its scores are unnormalized, and this application *differences two separate models*,
so unnormalized scores would inject exactly the inter-model bias described in Arm C.

## Arm C — per-model calibration

The mean per-character log-probability each model accumulates differs:

```
EN model: -1.372 nats/char
HE model: -1.649 nats/char
        -> 0.277 nats/char offset
```

Because `score_diff` subtracts one model's score from the other's, that offset compounds **linearly with word
length**: 1.39 nats over a 5-char word, 2.50 nats over a 9-char word. The HE model runs colder, so real
Hebrew text gets a depressed `score_active`, inflating `score_diff` toward spurious switching *out of Hebrew,
more so the longer the word*.

**Diagnostic before fix.** Plot FP rate against word length, split by direction, and confirm the predicted
asymmetry. Part of this gap is genuine — unvocalized Hebrew really does carry more information per character,
and a true likelihood ratio should include that — so this is a hypothesis, not an established bug.

If confirmed, the fix is one scalar per model, subtracted as `H_model × chars_scored` before differencing in
`EvaluationEngine.evaluate`. Zero size cost.

**Measured for the EN/SO pair** (see "Combining the two English models"), where the same offset exists —
SO −1.29 nats/char against EN −1.48 — and calibrating it away changes little. That is evidence about the
EN/SO pair only; the EN/HE offset above is larger, applies to every switch decision rather than to technical
mode alone, and remains untested.

## Note on evidence accumulation

`score()` sums log-probabilities across the word without length normalization. This is correct and should not
be "fixed": `score_diff` is a log-likelihood ratio, summing is the proper accumulation of evidence, and a
fixed threshold on an LLR is the right decision rule. Normalizing by length would discard evidence. Arm C is
the legitimate length-dependent concern in this area.

## Note on mixed-script evaluation

`load_corpus_lines` in `benchmark.py` deliberately excludes mixed EN/HE lines. A manual layout change fires a
CRE (`hooks.py`, `_fire_cre('manual_layout_change')`) which clears both buffers and history and resets delta
to baseline, so each script run in a mixed line is an independent segment. Modelled correctly, a mixed-script
test collapses into the pure-script test run per segment; modelled incorrectly, it feeds English words while
`current_layout='he'` and scores the engine's correct switch as a false positive.
