**Verdict: keep v1. Neither v1c nor v2 met the decision rule; Task 7 (swap) was not run.**

# Corpus v1 vs v1c vs v2 — A/B results (2026-10-02)

Plan: `docs/superpowers/plans/2026-10-02-corpus-v2.md`.

## Corpora

| Version | What it is |
|---|---|
| v1 | Current `data/{en,he}_corpus.txt`: the first 5M lines of CulturaX, raw |
| v1c | v1 run through `scripts/clean_corpus.py` (keyboard normalization + dedup) |
| v2 | `scripts/download_corpora.py` at commit `7c237ce` (see below) |

**v2 settings:**
- 6 seeded shards per language
- each document kept with probability 0.12
- normalized and deduplicated
- at most 25,000 lines per website
- 5,000,000 lines per language

**v2 shards:**
- EN: `en_part_00420`, `00487`, `01308`, `02058`, `02096`, `02650`
- HE: all six shards, `he_part_00000`–`00005`

## Method

**Test sets** (per language):
- `v1`: last 300k lines of v1c
- `v2`: last 300k lines of v2
- `chat`: OpenSubtitles, normalized, reversed Hebrew lines re-oriented, 6 subtitles per test line

**Models:**
- One quadgram model per version × language (`evaluation/corpus_ab.py`).
- Training lines whose dedup key appears in *any* of that language's three test sets are dropped.
- Each version's EN and HE models are scored together as a pair.

**Benchmark:** `evaluation/benchmark.py` at the default Δ=6.0, K=2, α=0.1, with up to 1M lines. The code was unchanged across all 18 runs.

| Model | Training lines | Quadgrams |
|---|---|---|
| en_v1 | 4,692,575 | 323,766 |
| en_v1c | 4,333,142 | 328,610 |
| en_v2 | 4,698,145 | 335,081 |
| he_v1 | 4,676,656 | 297,407 |
| he_v1c | 2,787,208 | 302,040 |
| he_v2 | 4,653,816 | 354,589 |

Test sizes are 7.9M–11.6M words each.

## Results (per 1k words; lower is better)

| Lang | Test | v1 FP | v1c FP | v2 FP | v1 FN | v1c FN | v2 FN |
|---|---|---|---|---|---|---|---|
| en | v1 | 0.134 | 0.132 | **0.125** | **0.601** | 0.622 | 0.603 |
| en | v2 | 0.133 | 0.134 | **0.128** | 0.611 | 0.623 | **0.603** |
| en | chat | 0.045 | 0.045 | **0.040** | **0.400** | 0.408 | 0.569 |
| he | v1 | 0.058 | **0.047** | 0.055 | 0.951 | **0.940** | 1.082 |
| he | v2 | 0.030 | **0.028** | 0.032 | 0.669 | **0.665** | 0.757 |
| he | chat | 0.205 | **0.191** | 0.215 | 1.385 | **1.341** | 1.389 |

## Decision

The rule, set before the runs: a candidate beats v1 only if both of these hold:
1. On `chat`, FP and FN are both lower in both languages.
2. On `v1`, no metric gets worse by more than 5% relative.

**v1c fails.**
- On EN `chat`, FP is unchanged (0.045) and FN is 2% worse.
- Hebrew improves everywhere: FP is −7% on `chat` and −19% on `v1`; FN is −1% to −3%.

**v2 fails, and clearly.**
- EN `chat` FN is **+42%** (0.400 → 0.569).
- HE FN is **+14%** on `v1` and **+13%** on its own `v2` test.
- EN FP improves (−7% to −11%) but doesn't make up for it.

At 5–11M words per test, a ±2% difference is within noise. The v2 FN regressions are far outside it.

## Interpretation (cause unknown)

**v2 shifts the trade-off from false switches to missed switches** (FN up, EN FP down). Its models have 4–19% more distinct quadgrams. A broader model may find more letter sequences plausible, but that link is unmeasured.

**Two explanations are ruled out:**
- **Less training data.** v2 trained on more words than v1: EN 171M vs 162M, HE 175M vs 101M. The quadgram learning curve (`data/learning_curve_*.json`) is flat beyond about 2M words.
- **A noisier corpus.** On a 300k-line sample, the share of model-eligible words unknown to `wordfreq` is the same for EN (0.94% v1, 0.95% v2) and lower for HE v2 (1.25% vs 1.39%). This doesn't rule out machine-translated or spun text, which uses real words.

**Next step:** compare the specific missed switches (words v2 misses and v1 catches, especially EN `chat`) and the quadgrams that make them score as plausible.

**v1c's Hebrew gain looks real but small.** Its English cost is within noise. If cleanup is wanted on Hebrew alone, it would need a rule exception: the plan requires symmetric changes.

## Artifacts (gitignored, under `data/corpus_ab/`)

- `v2/{en,he}_corpus.txt` + `.provenance.json`
- `{en,he}_v1c.txt`
- `test/`, `models/`
- `bench_*.log`
- `ab_results.tsv`
