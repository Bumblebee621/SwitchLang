# Deferred model experiments

Three candidate changes to the scoring model, none of them implemented. They are recorded here with the
measurements that motivated them so the work can be resumed cold. Arm C has since been measured and did not
hold up; it is kept here with its result so nobody re-derives it.

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

## Arm C — per-model calibration — measured, not confirmed

The mean per-character log-probability each model accumulates differs:

```
EN model: -1.372 nats/char
HE model: -1.649 nats/char
        -> 0.277 nats/char offset
```

The hypothesis was that because `score_diff` subtracts one model's score from the other's, that offset
compounds **linearly with word length** — 1.39 nats over a 5-char word, 2.50 over a 9-char word — depressing
`score_active` on real Hebrew and inflating `score_diff` toward spurious switching *out of Hebrew, more so the
longer the word*.

**It does not.** Measured by `evaluation/calibration.py` on held-out fold models (K=5, fold 0, `prune2`),
20,000 lines per language, 775k EN and 441k HE words:

| | slope of `score_diff` vs word length | intercept |
|---|---|---|
| EN layout | −3.198 nats/char | +0.21 |
| HE layout | −3.176 nats/char | +3.10 |
| **HE − EN** | **+0.022 nats/char** | **+2.89** |

The predicted asymmetry is +0.277 nats/char. The measured one is +0.022 — an order of magnitude smaller.

It is not zero. `--batches` refits the slope on disjoint samples, and the effect is consistently positive once
the sample is large enough to see it:

| batch size | batches | mean HE−EN slope | sd | range | negative |
|---|---|---|---|---|---|
| 400 lines | 20 | +0.0278 | 0.0948 | −0.165 to +0.260 | 7/20 |
| 6,000 lines | 4 | +0.0254 | 0.0159 | +0.010 to +0.047 | 0/4 |

So Arm C's mechanism exists and points the way it predicted. It is simply **~11× smaller than claimed**, and
it sits on top of a −3.2 nats/char trend running the other way. Note the first row: at 400 lines the slope
wanders across zero and can reach +0.26 by luck, which is close enough to the predicted +0.277 to look like a
confirmation. Any run of this arm needs thousands of lines before its slope means anything.

The same script reproduces the −1.3696 / −1.6413 table means exactly, so the 0.277 figure is correct; it
simply **does not transfer to `score_diff`** at anything like full strength. It is a
token-weighted mean over each model's own quadgram table, not the coefficient of length in the difference of
two scores on real text. The shadow string's per-character cost differs by direction in a way that very nearly
cancels the models' intrinsic difference.

### The intercept is not a fallback fix

The two slopes are near-identical, so the two fits are parallel lines and the **intercept gap of +2.89 nats**
is the vertical distance between them: Hebrew's `score_diff` sits ~2.9 nats closer to the threshold at every
length. That is 72% of delta, which looks alarming until you divide by the spread.

Hebrew's `score_diff` is 15–30% *less scattered* than English's, and that cancels the offset:

| word length | EN mean | EN sd | EN σ to Δ | HE mean | HE sd | HE σ to Δ | HE FP/1k |
|---|---|---|---|---|---|---|---|
| 2 | −8.09 | 5.72 | 2.11 | −2.54 | 2.49 | **2.63** | 1.70 |
| 3 | −8.84 | 3.56 | 3.61 | −6.28 | 3.51 | **2.93** | 1.20 |
| 4 | −12.78 | 5.31 | 3.16 | −9.79 | 4.17 | **3.31** | 1.05 |
| 5 | −15.25 | 5.87 | 3.28 | −13.05 | 4.57 | **3.73** | 0.64 |
| 6 | −18.04 | 6.14 | 3.59 | −16.38 | 5.05 | **4.04** | 0.40 |
| 7 | −21.92 | 6.53 | 3.97 | −19.51 | 5.75 | **4.09** | 0.22 |

In standard deviations — the units that decide whether a word crosses — Hebrew is *further* from the threshold
than English at every common length except 3. The observed crossing rates agree: 0.64/1k against English's
1.37/1k at length 5.

So a flat per-model constant is not a lesser version of Arm C worth shipping. Subtracting 2.89 from the Hebrew
side would move length-5 Hebrew from 3.73σ to 4.36σ, making the engine still more reluctant to leave Hebrew
and correspondingly slower to rescue Hebrew typed on an English layout, in exchange for false positives that
were never elevated. **Both shapes of Arm C are dead, for the same reason: the mean-level asymmetry does not
survive contact with the spread.**

Length 3 is the one exception — Hebrew genuinely sits closer there (2.93σ vs 3.61σ), and it is exactly where
mid-word evaluation begins.

Two further reasons not to build the fix:

- **The length trend runs the safe way, hard.** `score_diff` falls ~3.2 nats per added character in both
  directions, because each extra character is more evidence that the shadow reading is gibberish. Long words
  are far *safer*. The 0.277 the arm is about would have been a 9% correction on that.
- **Hebrew is not the exposed direction.** Switch rate per 1,000 evaluations falls with length in both
  directions, and Hebrew's is lower than English's at nearly every length (0.22 vs 0.30 per 1k words overall).
  The mean-level asymmetry does not reach the tail where decisions are made.

One caveat on the linear fit: HE − EN is not actually flat across the range, it is U-shaped — 5.55 nats at
length 2, 1.66 at length 6, 11.53 at length 12. The long-word climb is Arm C's predicted direction, but it
starts around length 9, where Hebrew has under 1% of its word mass and the switch rate is already zero.

**Where Hebrew's real exposure is: three-character words.** It is the only length where Hebrew sits closer to
the threshold than English once spread is accounted for (2.93σ vs 3.61σ), and it is where mid-word evaluation
starts — `hooks.py` gates on `len(buffer_active) >= 3`, so a 3-char prefix is the first thing the engine ever
judges. The collision list (342 entries, all 2–5 chars) and the delta schedule (`sensitivity.py:45-48`, delta
8.5 by the third word of a line) are the machinery already covering that range. That is the area worth a
future arm, not per-model calibration.

## Note on evidence accumulation

`score()` sums log-probabilities across the word without length normalization. This is correct and should not
be "fixed": `score_diff` is a log-likelihood ratio, summing is the proper accumulation of evidence, and a
fixed threshold on an LLR is the right decision rule. Normalizing by length would discard evidence. Arm C was
the candidate length-dependent concern in this area, and it measured out at +0.025 nats/char against a
−3.2 nats/char trend in the other direction — real, but too small to be worth a correction.

## Note on mixed-script evaluation

`load_corpus_lines` in `benchmark.py` deliberately excludes mixed EN/HE lines. A manual layout change fires a
CRE (`hooks.py`, `_fire_cre('manual_layout_change')`) which clears both buffers and history and resets delta
to baseline, so each script run in a mixed line is an independent segment. Modelled correctly, a mixed-script
test collapses into the pure-script test run per segment; modelled incorrectly, it feeds English words while
`current_layout='he'` and scores the engine's correct switch as a false positive.
