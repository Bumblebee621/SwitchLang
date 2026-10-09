# Char-GRU Tuning by FP/FN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not use worktrees** (see Global Constraints), even where those skills say to.

**Goal:** Make `scripts/tune_char_gru.py` rank Char-GRU configurations by switching quality (FP/FN), not validation loss. Add the two new inference knobs (unigram floor, fast-switch margin) to the search. Run it, then compare the tuned setting against three ablations: no floor with fast switch, floor with no fast switch, and neither.

**Architecture:**
- **One source of training hyperparameters.** `GRU_HPARAMS` in `scripts/train_char_gru.py` is the only definition. `train_model()`, the training CLI, `kfold_cv`, `corpus_ab` and the tuner's baseline all read it.
- **Harness knobs.** `evaluation/benchmark.py` takes the floor as a model-loading argument. The fast-switch margin is set per chunk, so sweeping it (or delta) reuses each worker's cached GRU scores.
- **Objective A (FN at an FP budget).** Sweep delta upward. At the first delta where *both* languages' FP/1k are at or below budget, the score is FN/1k(EN) + FN/1k(HE). The budget is the quadgram's FP/1k at Δ6.0 with no fast switch, measured on the same held-out lines. A configuration that never meets the budget is infeasible and ranks last.
- **`--lang`** picks whose GRU hyperparameters are searched (the other language stays at the baseline); both languages are always scored.
- **Two phases.**
  1. Phase 1 searches the training hyperparameters (random search, or `--coarse-to-fine`), with the floor and margin fixed at their current constants.
  2. Phase 2 sweeps floor × margin on the phase-1 winner's models. The tuned row and the three ablations are each the best phase-2 setting allowed by that row's restriction.

**Tech Stack:** Python `.venv` (uv-managed), PyTorch (ROCm) for training, NumPy GRU inference, `marisa_trie` quadgrams, pytest.

## Global Constraints

- `$PY` = `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH .venv/bin/python`, written as a per-command prefix. The Claude Desktop session runs inside an FHS environment whose `/lib64` loader bypasses nix-ld, so plain `.venv/bin/python` fails with `libz.so.1: cannot open shared object file`. Never `export` it. Never create a venv with `python3 -m venv`.
- Tests: `$PY -m pytest <files> -q`. Run only the files a task names, plus `tests/` in full at the end of Tasks 2 and 4.
- Never run `scripts/download_corpora.py` or any other download script. The corpora are already in `data/`.
- Work on branch `feature/neural-model` in place. No worktrees.
- Do not overwrite `data/*_char_gru.*`. The tuner writes models to a temp dir only.
- Do not change `GRU_HPARAMS`, `UNIGRAM_FLOOR` or `FAST_SWITCH_MARGIN` to tuned values. Shipping is the user's decision after reading the results.
- One delta for both languages: the app has a single `baseline_delta` (`core/sensitivity.py`).

---

### Task 0: Commit the in-progress work this plan builds on

The working tree has uncommitted changes from the 2026-10-02 session: the floor, the margin, typeable-text filtering and delta sweeps. Every later task edits the same files, so commit them first. Otherwise this plan's commits would absorb them.

**Files:** as listed per commit. Leave untracked `evaluation/*.log`, `evaluation/kfold_gru_vs_quad_v2.json` and `.claude/` alone.

- [ ] **Step 1: Run the suites the changes touch**

Run: `$PY -m pytest tests/test_evidence.py tests/test_neural_model.py tests/test_train_char_gru.py tests/test_benchmark_fn.py -q`
Expected: all pass. If anything fails, stop and report. Do not fix inside this task.

- [ ] **Step 2: Commit the model-side changes**

```bash
git add core/engine.py core/hooks.py core/neural_model.py scripts/train_char_gru.py
git commit -m "feat(neural): unigram floor, shared fast-switch margin, seeded GRU training

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Commit the evaluation-side changes**

```bash
git add evaluation/benchmark.py evaluation/kfold_cv.py evaluation/corpus_ab.py scripts/build_quadgrams.py tests/test_evidence.py
git commit -m "feat(evaluation): typeable-text filtering, hook word split, k-fold delta sweeps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Commit the plan-doc fix**

```bash
git add docs/superpowers/plans/2026-10-02-corpus-v2.md docs/superpowers/plans/2026-10-09-gru-tuning-fp-fn.md
git commit -m "docs(plans): drop LD_LIBRARY_PATH step; add GRU FP/FN tuning plan

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: One definition of the GRU training hyperparameters

Today five places each define "the shipped hyperparameters", and they disagree:

| Where | emb | hidden | dropout | lr | batch | epochs |
|---|---|---|---|---|---|---|
| `data/*_char_gru.meta.json` + train CLI | 32 | 96 | 0.05 | 0.004 | (512) | (4) |
| `train_model()` defaults | 32 | 128 | 0.1 | 0.008 | 512 | 5 |
| tuner `BASELINE_PARAMS` | 64 | 128 | 0.1 | 0.01 | 512 | — |
| `kfold_cv` / `corpus_ab` | meta.json | | | | 512 | 3 |
| `CharGRU.__init__` defaults | 32 | 96 | 0.05 | — | — | — |

`GRU_HPARAMS` takes the values the shipped model was trained with: the meta.json values, plus the train CLI's batch and epochs.

**Files:**
- Modify: `scripts/train_char_gru.py` (constant, `CharGRU.__init__`, `train_model`, `main`)
- Modify: `evaluation/kfold_cv.py` (delete `shipped_hparams`, epochs default)
- Modify: `evaluation/corpus_ab.py` (GRU branch, docstring)
- Test: `tests/test_train_char_gru.py`

**Interfaces:**
- Produces: `scripts.train_char_gru.GRU_HPARAMS: dict` with keys `emb_dim, hidden_dim, dropout, lr, weight_decay, batch_size, epochs`.
- Produces: `train_model(words, lang='en', val_words=None, device=None, seed=0, **hparams) -> (model, vocab_info)`. Any `hparams` key not in `GRU_HPARAMS` raises `TypeError`. `vocab_info` records all 7 hyperparameters.

- [ ] **Step 1: Write the failing tests** (append to `TestTrainValLoss` in `tests/test_train_char_gru.py`, and add `GRU_HPARAMS` to the import)

```python
from scripts.train_char_gru import GRU_HPARAMS, train_model
```

```python
    def test_defaults_come_from_gru_hparams(self):
        _, info = train_model(WORDS, epochs=1, device='cpu')
        for key, value in GRU_HPARAMS.items():
            self.assertEqual(info[key], 1 if key == 'epochs' else value, key)

    def test_unknown_hyperparameter_rejected(self):
        with self.assertRaises(TypeError):
            train_model(WORDS, hiden_dim=8, epochs=1, device='cpu')
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_train_char_gru.py -q`
Expected: FAIL with `ImportError: cannot import name 'GRU_HPARAMS'`

- [ ] **Step 3: Implement in `scripts/train_char_gru.py`**

Below `HE_CHARS = sorted(ALLOWED_HE)` add:

```python
# Training hyperparameters of the shipped model: the single definition every trainer,
# k-fold run and tuner baseline reads. Change it here (after a tuning run) and nowhere else.
GRU_HPARAMS = {
    'emb_dim': 32,
    'hidden_dim': 96,
    'dropout': 0.05,
    'lr': 0.004,
    'weight_decay': 1e-4,
    'batch_size': 512,
    'epochs': 4,
}
```

Change `CharGRU.__init__` to `def __init__(self, vocab_size, emb_dim, hidden_dim, dropout):`. This drops a second copy of the defaults; its only caller passes all four.

Replace the `train_model` signature and add the merge as its first lines:

```python
def train_model(words, lang='en', val_words=None, device=None, seed=0, **hparams):
    """Train the CharGRU model and return (model, vocab_info).

    *hparams* override GRU_HPARAMS key by key; anything else is a typo and raises.
    """
    unknown = hparams.keys() - GRU_HPARAMS.keys()
    if unknown:
        raise TypeError(f"unknown hyperparameters: {sorted(unknown)}")
    hp = {**GRU_HPARAMS, **hparams}
    emb_dim, hidden_dim, dropout = hp['emb_dim'], hp['hidden_dim'], hp['dropout']
    lr, weight_decay, batch_size, epochs = hp['lr'], hp['weight_decay'], hp['batch_size'], hp['epochs']
```

The `torch.manual_seed(seed)` line and everything after it stays. In `vocab_info`, replace the five keys `'emb_dim'`, `'hidden_dim'`, `'dropout'`, `'lr'`, `'weight_decay'` with `**hp,`.

In `main()`, replace the five arguments `--epochs`, `--emb-dim`, `--hidden-dim`, `--lr`, `--dropout` and `--weight-decay` with:

```python
    for key, value in GRU_HPARAMS.items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value), default=value,
                            help=f"default: {value} (GRU_HPARAMS)")
```

Then replace the six hyperparameter keyword arguments in its `train_model(...)` call with `**{key: getattr(args, key) for key in GRU_HPARAMS},`.

- [ ] **Step 4: Point `kfold_cv` and `corpus_ab` at the constant**

`evaluation/kfold_cv.py`:
- Import `GRU_HPARAMS` from `scripts.train_char_gru`.
- Delete `shipped_hparams()`. Drop `**shipped_hparams(lang, data_dir)` and `**shipped_hparams(other_lang, data_dir)` from the two `train_model` calls, keeping `epochs=epochs`.
- Change `run_kfold_evaluation`'s `epochs=3` default to `epochs=GRU_HPARAMS['epochs']`.
- Change the `--epochs` argument to `default=GRU_HPARAMS['epochs'], help="Neural training epochs per fold (default: GRU_HPARAMS)"`.

`evaluation/corpus_ab.py` GRU branch:
- Import `extract_words_from_lines, CHARS` only (no `shipped_hparams`).
- Call `train_model(words[:-n_val], lang=args.lang, val_words=words[-n_val:], seed=seed)`.
- In the module docstring, change "3 epochs, shipped hyperparameters" to "GRU_HPARAMS".

- [ ] **Step 5: Run the tests and check no reference is left behind**

Run: `$PY -m pytest tests/test_train_char_gru.py -q`
Expected: 4 passed.

Run: `grep -rn "shipped_hparams" --include=*.py . | grep -v .venv`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_char_gru.py evaluation/kfold_cv.py evaluation/corpus_ab.py tests/test_train_char_gru.py
git commit -m "refactor(neural): single GRU_HPARAMS definition for every trainer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Floor and fast-switch margin as harness knobs

**Files:**
- Modify: `evaluation/benchmark.py` (`EvaluationHarness.__init__`, `_worker_models` → `_worker_harnesses`, `_run_chunk`, `run_test`)
- Create: `tests/test_benchmark_knobs.py`

**Interfaces:**
- Consumes: `CharNeuralModel(model_path, floor=UNIGRAM_FLOOR)` (exists).
- Produces: `EvaluationHarness(data_dir, en_model_path=None, he_model_path=None, mode='standard', req_confirmations=2, model_type='quadgram', accumulate=True, alpha=0.1, floor=UNIGRAM_FLOOR, fast_margin=FAST_SWITCH_MARGIN)`. `floor` is new and goes *before* `fast_margin`, because workers build harnesses positionally from the key.
- Produces: `run_test(..., floor=UNIGRAM_FLOOR, fast_margin=FAST_SWITCH_MARGIN, **kw)`. `math.inf` as `fast_margin` means fast switch off.
- Produces: `_run_chunk(job)`, where `job = (test, offset, lines, lang, delta, key, fast_margin, kw)`. `key` no longer contains `fast_margin`.

- [ ] **Step 1: Write the failing tests** in `tests/test_benchmark_knobs.py`

```python
"""
test_benchmark_knobs.py — the unigram floor reaches the neural models, and changing the
fast-switch margin reuses a worker's harness (and so its GRU score cache).
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation.benchmark as bm
from scripts.train_char_gru import export_numpy_weights, train_model

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class TestHarnessKnobs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = {}
        for lang, words in (('en', ['hello', 'world'] * 20), ('he', ['שלום', 'עולם'] * 20)):
            model, info = train_model(words, lang=lang, emb_dim=8, hidden_dim=8, epochs=1, device='cpu')
            base = os.path.join(cls.tmp.name, lang)
            export_numpy_weights(model, info, base)
            cls.paths[f'{lang}_model_path'] = f'{base}.npz'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_floor_reaches_models(self):
        off = bm.EvaluationHarness(DATA_DIR, model_type='neural', floor=0.0, **self.paths)
        on = bm.EvaluationHarness(DATA_DIR, model_type='neural', floor=0.05, **self.paths)
        self.assertIsNone(off.engine.en_model._floor)
        self.assertIsNotNone(on.engine.en_model._floor)
        self.assertNotEqual(off.engine.en_model.score(' hello'), on.engine.en_model.score(' hello'))

    def test_margin_change_reuses_harness(self):
        key = (DATA_DIR, self.paths['en_model_path'], self.paths['he_model_path'],
               'standard', 2, 'neural', True, 0.1, 0.05)
        bm._worker_harnesses.clear()
        bm._run_chunk(('fp', 0, ['hello world'], 'en', 6.0, key, 2.5, {}))
        first = bm._worker_harnesses[key]
        bm._run_chunk(('fp', 0, ['hello world'], 'en', 6.0, key, math.inf, {}))
        self.assertIs(bm._worker_harnesses[key], first)
        self.assertEqual(first.fast_margin, math.inf)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_benchmark_knobs.py -q`
Expected: FAIL. `test_floor_reaches_models` fails with `TypeError: ... unexpected keyword argument 'floor'`; `test_margin_change_reuses_harness` fails with `AttributeError: ... '_worker_harnesses'`.

- [ ] **Step 3: Implement in `evaluation/benchmark.py`**

Add `UNIGRAM_FLOOR` to the `core.neural_model` import.

In `EvaluationHarness.__init__`:
- Change the signature to end `accumulate=True, alpha=0.1, floor=UNIGRAM_FLOOR, fast_margin=FAST_SWITCH_MARGIN):`.
- In the neural branch, replace the `load_neural_models(...)` call and the two overrides with:

```python
            # Explicit paths skip the shipped files entirely; the floor applies to both models.
            paths = {'en': en_model_path or os.path.join(data_dir, 'en_char_gru.npz'),
                     'he': he_model_path or os.path.join(data_dir, 'he_char_gru.npz')}
            models = {lang: CharNeuralModel(path, floor=floor) for lang, path in paths.items()}
```

If `load_neural_models` is now unused in this file, drop it from the import.

Replace `_worker_models = (None, None)` and `_run_chunk` with:

```python
_worker_harnesses = {}


def _run_chunk(job):
    test, offset, lines, lang, delta, key, fast_margin, kw = job
    harness = _worker_harnesses.get(key)
    if harness is None:
        if len(_worker_harnesses) >= 4:
            # ponytail: drop all, bounds memory (each harness caches up to ~150 MB of GRU scores); LRU if seeds > 4
            _worker_harnesses.clear()
        harness = _worker_harnesses[key] = EvaluationHarness(*key)
    # The margin only gates the decision, not the cached scores, so sweeping it keeps the cache.
    harness.fast_margin = fast_margin
    return _test_method(harness, test)(lines, lang, delta, line_offset=offset, progress=False, **kw)
```

In `run_test`:
- Add `floor=UNIGRAM_FLOOR,` before `fast_margin=FAST_SWITCH_MARGIN` in the signature.
- Build the key as `(data_dir, en_model_path, he_model_path, mode, req_confirmations, model_type, accumulate, alpha, floor)`.
- In the `jobs <= 1` branch, use:

```python
        harness = EvaluationHarness(*key)
        harness.fast_margin = fast_margin
        return _test_method(harness, test)(lines, lang, delta, **kw)
```

- Build the chunk tuples as `(test, i, lines[i:i + size], lang, delta, key, fast_margin, kw)`.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_benchmark_knobs.py -q`
Expected: 2 passed.

Run: `$PY -m pytest tests -q`
Expected: no new failures. `tests/test_tune_char_gru.py` still passes here; Task 4 replaces it.

- [ ] **Step 5: Commit**

```bash
git add evaluation/benchmark.py tests/test_benchmark_knobs.py
git commit -m "feat(evaluation): unigram floor and fast-switch margin as harness knobs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Objective A and ablation selection (pure functions)

These are the parts of the tuner whose logic can go wrong silently, so they get unit tests with fake inputs. Everything else in the tuner is wiring.

**Files:**
- Modify: `scripts/tune_char_gru.py`. Add the functions below the imports. Task 4 rewrites the rest of the file.
- Replace: `tests/test_tune_char_gru.py`

**Interfaces:**
- Produces: `fn_at_budget(fp_at, fn_at, budget, deltas=DELTAS) -> dict | None`. Inputs:
  - `fp_at(delta) -> {lang: float}`
  - `fn_at(delta) -> {lang: {'fn': float, 'med': float}}`
  
  Returns `{'delta', 'fp': {lang}, 'fn': {lang}, 'med': {lang}, 'score': float}`, or `None` when no delta passes.
- Produces: `rank_key(row) -> tuple`, which puts infeasible rows (where `row['result'] is None`) last.
- Produces: `best(rows, keep=lambda r: True) -> row | None`.
- Produces: `ablations(rows) -> {'tuned', 'no floor, fast switch', 'floor, no fast switch', 'neither': row | None}`. Each row has `floor`, `margin` and `result`.
- Produces: the constants `LANGS`, `DELTAS`, `FLOORS`, `MARGINS`, `BUDGET_DELTA`.

- [ ] **Step 1: Write the failing tests**: replace `tests/test_tune_char_gru.py` entirely

```python
"""
test_tune_char_gru.py — objective A (FN at an FP budget) and the ablation pick.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tune_char_gru import ablations, best, fn_at_budget

BUDGET = {'en': 0.10, 'he': 0.02}


def _fn(delta):
    return {'en': {'fn': delta / 10, 'med': 5.0}, 'he': {'fn': delta / 20, 'med': 4.0}}


class TestObjective(unittest.TestCase):

    def test_first_delta_within_budget_in_both_languages(self):
        fps = {4.5: {'en': 0.3, 'he': 0.01}, 6.0: {'en': 0.08, 'he': 0.03}, 7.5: {'en': 0.05, 'he': 0.02}}
        fn_calls = []

        def fn_at(d):
            fn_calls.append(d)
            return _fn(d)

        res = fn_at_budget(fps.__getitem__, fn_at, BUDGET, deltas=(7.5, 4.5, 6.0))
        self.assertEqual(res['delta'], 7.5)          # 6.0 fails HE, 4.5 fails EN
        self.assertEqual(fn_calls, [7.5])            # FN only measured at the chosen delta
        self.assertAlmostEqual(res['score'], 0.75 + 0.375)
        self.assertEqual(res['med'], {'en': 5.0, 'he': 4.0})

    def test_none_when_budget_never_met(self):
        self.assertIsNone(fn_at_budget(lambda d: {'en': 1.0, 'he': 1.0}, _fn, BUDGET, deltas=(6.0, 9.0)))


class TestAblations(unittest.TestCase):

    def _row(self, floor, margin, score):
        return {'floor': floor, 'margin': margin, 'result': None if score is None else {'score': score}}

    def test_each_restriction_gets_its_own_best(self):
        rows = [self._row(0.05, 3.5, 2.0), self._row(0.0, 3.5, 2.5), self._row(0.0, 5.0, 2.2),
                self._row(0.05, math.inf, 2.4), self._row(0.0, math.inf, None), self._row(0.1, 2.5, None)]
        picked = ablations(rows)
        self.assertEqual(picked['tuned'], rows[0])
        self.assertEqual(picked['no floor, fast switch'], rows[2])
        self.assertEqual(picked['floor, no fast switch'], rows[3])
        self.assertEqual(picked['neither'], rows[4])   # infeasible, still reported

    def test_best_ranks_infeasible_last(self):
        rows = [self._row(0.0, 2.5, None), self._row(0.0, 3.5, 9.0)]
        self.assertEqual(best(rows), rows[1])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_tune_char_gru.py -q`
Expected: FAIL with `ImportError: cannot import name 'ablations'`

- [ ] **Step 3: Add to `scripts/tune_char_gru.py`** (below the existing imports; `import math` is already there)

```python
LANGS = ('en', 'he')
BUDGET_DELTA = 6.0                                   # shipped baseline_delta
DELTAS = (4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0)
FLOORS = (0.0, 0.02, 0.05, 0.1)
MARGINS = (math.inf, 2.5, 3.5, 5.0)                  # inf = fast switch off


def fn_at_budget(fp_at, fn_at, budget, deltas=DELTAS):
    """Objective A: summed FN/1k at the smallest delta whose FP/1k is within *budget* in every language.

    fp_at(delta) -> {lang: FP/1k}; fn_at(delta) -> {lang: {'fn': FN/1k, 'med': median latency}}.
    Returns None when no delta qualifies (infeasible).
    """
    # ponytail: stops at the first passing delta, assuming FP falls and FN rises with delta;
    # sweep every delta and take the min FN if a run ever shows otherwise.
    for d in sorted(deltas):
        fp = fp_at(d)
        if all(fp[lang] <= budget[lang] for lang in budget):
            fn = fn_at(d)
            return {'delta': d, 'fp': fp,
                    'fn': {lang: v['fn'] for lang, v in fn.items()},
                    'med': {lang: v['med'] for lang, v in fn.items()},
                    'score': sum(v['fn'] for v in fn.values())}
    return None


def rank_key(row):
    """Feasible rows by score, infeasible rows last."""
    return (row['result'] is None, row['result']['score'] if row['result'] else 0.0)


def best(rows, keep=lambda r: True):
    """Best-ranked row passing *keep*, or None."""
    return min((r for r in rows if keep(r)), key=rank_key, default=None)


def ablations(rows):
    """The tuned floor/margin and the three ablations, each at its own best phase-2 setting."""
    return {
        'tuned': best(rows),
        'no floor, fast switch': best(rows, lambda r: r['floor'] == 0 and r['margin'] != math.inf),
        'floor, no fast switch': best(rows, lambda r: r['floor'] > 0 and r['margin'] == math.inf),
        'neither': best(rows, lambda r: r['floor'] == 0 and r['margin'] == math.inf),
    }
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_tune_char_gru.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/tune_char_gru.py tests/test_tune_char_gru.py
git commit -m "feat(tuning): objective A (FN at FP budget) and ablation selection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Rewrite the tuner around objective A

The old tuner splits a word list, ranks by validation loss, and trains in CPU worker processes. Replace it with the version below:
- One held-out split per language (every 5th line, as fold 1 of `kfold_cv`).
- Quadgram budget measured on that split.
- GPU training run serially and memoized.
- Evaluation spread over the harness pool.
- A tqdm progress bar with an ETA per phase or stage.

**`--lang {en,he,both}`** names whose GRU takes the candidate's hyperparameters. Any other language trains with `BASELINE_PARAMS` once per seed and is reused. Both languages are *always scored*, because the app uses one delta for both.

**`--coarse-to-fine`** stays, ranked by objective A:
- **Stage 1:** model shape × learning rate, everything else from the baseline.
- **Stage 2:** regularization × batch size around the stage-1 winner.

Random search (`--trials`) stays the default mode.

**Removed:**
- `--grid`
- `--test`: unit tests cover it.
- `--export-best`: to ship, set `GRU_HPARAMS` and run `train_char_gru.py`.
- `--max-words`: replaced by `--train-words`.

`DEFAULT_SEARCH_SPACE` and `compute_param_count` stay unchanged.

**Files:**
- Modify: `scripts/tune_char_gru.py` (everything except Task 3's block)
- Modify: `tests/test_tune_char_gru.py` (append one test class)

**Interfaces:**
- Consumes:
  - From Task 1: `GRU_HPARAMS`, `train_model`, `export_numpy_weights`.
  - From Task 2: `run_test(..., floor=, fast_margin=)`.
  - From Task 3: `fn_at_budget`, `rank_key`, `best`, `ablations`, and the constants.
  - From `evaluation.kfold_cv`: `load_pool`, `extract_words_from_lines`, `ALLOWED`, `CHARS`.
  - From `scripts.build_quadgrams`: `build_quadgrams_from_lines`, `save_model_data_to_trie`.
  - tqdm (installed; same pattern as `evaluation/learning_curve.py:121`).
- Produces:
  - CLI `scripts/tune_char_gru.py [--lang en|he|both] [--coarse-to-fine] [--data-dir data] [--max-lines 150000] [--train-words 1000000] [--trials 12] [--seeds 0 1] [--sample-seed 42] [--jobs N] [--out PATH]`.
  - The JSON output has keys `lang, mode, budget, seeds, phase1, phase2, comparison`.
  - `coarse_candidates() -> list[dict]`: 27 configs, baseline first.
  - `fine_candidates(arch) -> list[dict]`: 18 configs.

- [ ] **Step 1: Write the failing grid test** (append to `tests/test_tune_char_gru.py`, and extend the import to `from scripts.tune_char_gru import BASELINE_PARAMS, ablations, best, coarse_candidates, fine_candidates, fn_at_budget`)

```python
class TestCoarseToFine(unittest.TestCase):

    def test_stage1_starts_at_baseline_and_holds_27_unique(self):
        cands = coarse_candidates()
        self.assertEqual(cands[0], BASELINE_PARAMS)
        self.assertEqual(len({tuple(sorted(c.items())) for c in cands}), 27)
        self.assertEqual(len(cands), 27)

    def test_stage2_varies_only_regularization_and_batch(self):
        cands = fine_candidates(BASELINE_PARAMS)
        self.assertEqual(len({tuple(sorted(c.items())) for c in cands}), 18)
        for c in cands:
            self.assertEqual((c['emb_dim'], c['hidden_dim'], c['lr']),
                             (BASELINE_PARAMS['emb_dim'], BASELINE_PARAMS['hidden_dim'], BASELINE_PARAMS['lr']))
```

Run: `$PY -m pytest tests/test_tune_char_gru.py -q`
Expected: FAIL with `ImportError: cannot import name 'coarse_candidates'`

- [ ] **Step 2: Replace the module docstring, imports, `BASELINE_PARAMS` and the candidate generators**

```python
"""
tune_char_gru.py — Char-GRU hyperparameter search ranked by switching quality.

Objective A: an EN+HE GRU pair is scored by FN at an FP budget. Delta is swept upward; at
the first delta where every language's FP/1k is within budget, the score is the sum of the
two languages' FN/1k there. The budget is the quadgram's FP/1k at Δ6.0 with no fast switch
(the shipped quadgram behaviour) on the same held-out lines. A pair that never meets the
budget is infeasible and ranks last. Validation loss is reported, not ranked on.

--lang picks whose GRU takes the candidate hyperparameters; the other language trains with
BASELINE_PARAMS. Both languages are always scored: the app has one delta for both.

Phase 1 searches the training hyperparameters at the current UNIGRAM_FLOOR / FAST_SWITCH_MARGIN,
by random search (--trials) or --coarse-to-fine (shape and lr, then regularization).
Phase 2 sweeps floor x margin on the phase-1 winner and reports the tuned setting against
three ablations (no floor, no fast switch, neither), each at its own best setting.
FP and FN are means over --seeds training seeds.

Usage:
    .venv/bin/python scripts/tune_char_gru.py --lang en --coarse-to-fine --seeds 0 1 --out evaluation/tune_char_gru_en.json
"""

import argparse
import itertools
import json
import logging
import math
import os
import random
import shutil
import statistics
import sys
import tempfile

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import FAST_SWITCH_MARGIN
from core.neural_model import UNIGRAM_FLOOR
from evaluation.benchmark import run_test
from evaluation.kfold_cv import ALLOWED, CHARS, extract_words_from_lines, load_pool
from scripts.build_quadgrams import build_quadgrams_from_lines, save_model_data_to_trie
from scripts.train_char_gru import GRU_HPARAMS, export_numpy_weights, train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('tune_char_gru')

K = 5   # held-out test = every K-th line, as fold 1 of kfold_cv

# Candidate #0: the shipped training setup. Epochs are not searched.
BASELINE_PARAMS = {k: v for k, v in GRU_HPARAMS.items() if k != 'epochs'}
```

Then keep Task 3's block, `DEFAULT_SEARCH_SPACE` and `compute_param_count` unchanged. Replace `generate_candidates`, `generate_coarse_candidates` and `generate_fine_candidates` with:

```python
def generate_candidates(trials, seed=42, search_space=None):
    """Random search: baseline first, then unique random draws from the search space."""
    space = search_space or DEFAULT_SEARCH_SPACE
    rng = random.Random(seed)
    candidates = [dict(BASELINE_PARAMS)]
    seen = {tuple(sorted(BASELINE_PARAMS.items()))}
    for _ in range(trials * 50):
        if len(candidates) >= trials:
            break
        cand = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(cand.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(cand)
    return candidates


def coarse_candidates():
    """Stage 1: shape and learning rate, regularization held at the baseline (27 configs, baseline first)."""
    grid = [{**BASELINE_PARAMS, 'emb_dim': e, 'hidden_dim': h, 'lr': lr}
            for e, h, lr in itertools.product([32, 48, 64], [64, 80, 96], [0.004, 0.006, 0.008])]
    return [dict(BASELINE_PARAMS)] + [c for c in grid if c != BASELINE_PARAMS]


def fine_candidates(arch):
    """Stage 2: regularization and batch size around the stage-1 winner *arch* (18 configs)."""
    return [{**arch, 'dropout': d, 'weight_decay': wd, 'batch_size': bs}
            for d, wd, bs in itertools.product([0.0, 0.05, 0.10], [0.0, 1e-5, 1e-4], [128, 256])]
```

Delete `_eval_candidate`, `_run_batch_eval`, `tune_hyperparameters`, `print_scoreboard` and the old `main`.

Run: `$PY -m pytest tests/test_tune_char_gru.py -q`
Expected: 6 passed.

- [ ] **Step 3: Add the split, training and scoring functions**

```python
def prepare_split(data_dir, max_lines, train_words, jobs, work_dir):
    """Hold out every K-th line per language, build quadgrams on the rest, measure the FP budget."""
    split = {'test': {}, 'words': {}, 'quad': {}}
    for lang in LANGS:
        lines = load_pool(lang, data_dir, K, max_lines, train_words)
        train = [l for i, l in enumerate(lines) if i % K]
        split['test'][lang] = [l for i, l in enumerate(lines) if not i % K]
        split['words'][lang] = extract_words_from_lines(train, CHARS[lang], max_words=None)
        base = os.path.join(work_dir, f'quad_{lang}')
        counts = build_quadgrams_from_lines(train, ALLOWED[lang], min_count=2, num_workers=jobs)
        save_model_data_to_trie(counts, f'{base}.marisa', f'{base}.meta.json')
        split['quad'][f'{lang}_model_path'] = f'{base}.marisa'
    split['budget'] = {
        lang: run_test('fp', split['test'][lang], lang, BUDGET_DELTA, data_dir, jobs=jobs,
                       model_type='quadgram', fast_margin=math.inf, **split['quad']).fp_per_1k
        for lang in LANGS
    }
    logger.info("FP budget (quadgram, Δ%g, no fast switch): %s", BUDGET_DELTA, split['budget'])
    return split


def train_one(lang, hp, seed, split, work_dir, memo):
    """Train one GRU on the split's words, memoized on (lang, hp, seed); return (npz path, val loss)."""
    key = (lang, tuple(sorted(hp.items())), seed)
    if key not in memo:
        words = split['words'][lang]
        n_val = len(words) // 20   # corpus-tail 5% held out, as train_char_gru does
        model, info = train_model(words[:-n_val], lang=lang, val_words=words[-n_val:], seed=seed, **hp)
        base = os.path.join(work_dir, f'gru{len(memo)}_{lang}')
        export_numpy_weights(model, info, base)
        memo[key] = (f'{base}.npz', info['val_loss'])
    return memo[key]


def train_pair(hp, searched, split, seed, work_dir, memo):
    """EN+HE GRUs: languages in *searched* take *hp*, the others BASELINE_PARAMS.

    Returns (run_test model-path kwargs, {lang: val loss}).
    """
    paths, val = {}, {}
    for lang in LANGS:
        paths[f'{lang}_model_path'], val[lang] = train_one(
            lang, hp if lang in searched else BASELINE_PARAMS, seed, split, work_dir, memo)
    return paths, val


def score(pairs, split, floor, margin, data_dir, jobs):
    """Objective A for one floor/margin, FP and FN averaged over the seed model *pairs*."""
    def run(test, lang, delta):
        return [run_test(test, split['test'][lang], lang, delta, data_dir, jobs=jobs,
                         model_type='neural', floor=floor, fast_margin=margin, **p) for p in pairs]

    def fp_at(delta):
        return {lang: statistics.mean(r.fp_per_1k for r in run('fp', lang, delta)) for lang in LANGS}

    def fn_at(delta):
        out = {}
        for lang in LANGS:
            reports = run('fn', lang, delta)
            out[lang] = {'fn': statistics.mean(r.fn_per_1k for r in reports),
                         'med': statistics.mean(statistics.median(r.latency_values) if r.latency_values else 0.0
                                                for r in reports)}
        return out

    return fn_at_budget(fp_at, fn_at, split['budget'])
```

- [ ] **Step 4: Add the phases, printing, and `main`**

```python
def phase1(candidates, desc, searched, split, seeds, data_dir, jobs, work_dir, memo):
    """Score training-hyperparameter *candidates* at the current floor and margin, best first."""
    rows = []
    for i, hp in enumerate(tqdm(candidates, desc=desc, unit='config')):
        trained = [train_pair(hp, searched, split, s, work_dir, memo) for s in seeds]
        pairs = [p for p, _ in trained]
        rows.append({'params': hp, 'is_baseline': hp == BASELINE_PARAMS, 'pairs': pairs,
                     'val_loss': {lang: statistics.mean(v[lang] for _, v in trained) for lang in searched},
                     'result': score(pairs, split, UNIGRAM_FLOOR, FAST_SWITCH_MARGIN, data_dir, jobs)})
        logger.info("%s %d/%d %s -> %s", desc, i + 1, len(candidates), hp, _fmt(rows[-1]['result']))
    return sorted(rows, key=rank_key)


def phase2(pairs, split, data_dir, jobs):
    """Every floor x margin on fixed models. Floor is the outer loop: margins reuse the score cache."""
    settings = list(itertools.product(FLOORS, MARGINS))
    return [{'floor': f, 'margin': m, 'result': score(pairs, split, f, m, data_dir, jobs)}
            for f, m in tqdm(settings, desc='phase 2', unit='setting')]


def _fmt(res):
    if res is None:
        return "infeasible (no delta meets the FP budget)"
    return (f"Δ{res['delta']:<5g} FP en {res['fp']['en']:.3f} he {res['fp']['he']:.3f} | "
            f"FN en {res['fn']['en']:.3f} he {res['fn']['he']:.3f} sum {res['score']:.3f} | "
            f"med lat en {res['med']['en']:.1f} he {res['med']['he']:.1f}")


def print_report(budget, searched, p1, comparison):
    print(f"\nFP budget per 1k words (quadgram, Δ{BUDGET_DELTA:g}, no fast switch): "
          f"en {budget['en']:.3f}  he {budget['he']:.3f}")
    print(f"\nPHASE 1 — training hyperparameters of {'+'.join(searched)} "
          f"(floor {UNIGRAM_FLOOR:g}, margin {FAST_SWITCH_MARGIN:g}), best first")
    for r in p1:
        p = r['params']
        val = ' '.join(f"{lang} {v:.4f}" for lang, v in r['val_loss'].items())
        tag = ' (baseline)' if r['is_baseline'] else ''
        print(f"  emb {p['emb_dim']:>3} hid {p['hidden_dim']:>3} drop {p['dropout']:.2f} lr {p['lr']:.4f} "
              f"wd {p['weight_decay']:.0e} bs {p['batch_size']:>3} | val {val} | {_fmt(r['result'])}{tag}")
    print("\nPHASE 2 — floor / fast-switch on the phase-1 winner")
    for name, r in comparison.items():
        knobs = '' if r is None else f"floor {r['floor']:g}, margin {r['margin']:g}"
        print(f"  {name:<22} {knobs:<24} {_fmt(r and r['result'])}")


def main():
    parser = argparse.ArgumentParser(description="Tune Char-GRU hyperparameters by FN at an FP budget")
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='both',
                        help="Whose GRU takes the candidate hyperparameters; the other uses the baseline. "
                             "Both languages are always scored (default: both)")
    parser.add_argument('--coarse-to-fine', action='store_true',
                        help="Stage 1 shape x lr (27 configs), stage 2 regularization x batch (18) "
                             "instead of random search")
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--max-lines', type=int, default=150_000, help="Corpus lines read per language (default: 150000)")
    parser.add_argument('--train-words', type=int, default=1_000_000, help="Training words per language (default: 1000000)")
    parser.add_argument('--trials', type=int, default=12, help="Random-search candidates, baseline included (default: 12)")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1], help="Training seeds per candidate (default: 0 1)")
    parser.add_argument('--sample-seed', type=int, default=42, help="Seed for drawing random candidates (default: 42)")
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 4, help="Evaluation worker processes")
    parser.add_argument('--out', default=None, help="Write budget, both phases and the comparison to this JSON file")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    searched = LANGS if args.lang == 'both' else (args.lang,)
    work_dir = tempfile.mkdtemp(prefix='switchlang_tune_')
    memo = {}
    try:
        with logging_redirect_tqdm():
            split = prepare_split(data_dir, args.max_lines, args.train_words, args.jobs, work_dir)

            def run(candidates, desc):
                return phase1(candidates, desc, searched, split, args.seeds, data_dir, args.jobs, work_dir, memo)

            if args.coarse_to_fine:
                stage1 = run(coarse_candidates(), 'stage 1')
                p1 = sorted(stage1 + run(fine_candidates(stage1[0]['params']), 'stage 2'), key=rank_key)
            else:
                p1 = run(generate_candidates(args.trials, args.sample_seed), 'phase 1')
            p2 = phase2(p1[0]['pairs'], split, data_dir, args.jobs)
        comparison = ablations(p2)
        print_report(split['budget'], searched, p1, comparison)
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as fh:
                json.dump({'lang': args.lang, 'mode': 'coarse-to-fine' if args.coarse_to_fine else 'random',
                           'budget': split['budget'], 'seeds': args.seeds,
                           'phase1': [{k: v for k, v in r.items() if k != 'pairs'} for r in p1],
                           'phase2': p2, 'comparison': comparison}, fh, indent=2)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Unit tests still pass**

Run: `$PY -m pytest tests -q`
Expected: no failures.

- [ ] **Step 6: Smoke run on a tiny split**

This exercises the real wiring (GPU training, the memo, the pool, the budget, both phases and the tqdm bars) in a few minutes. Coarse-to-fine is too long for a smoke run; Step 1's tests cover its grids.

Run: `$PY scripts/tune_char_gru.py --lang en --max-lines 3000 --train-words 40000 --trials 2 --seeds 0 --out "$SCRATCH/tune_smoke.json"`. `$SCRATCH` is the session scratchpad directory, given in your dispatch.

Expected: it exits 0 and shows tqdm bars `phase 1` and `phase 2` with an ETA. It prints:
- a budget line
- a phase-1 table with 2 rows, one tagged `(baseline)`, with validation loss for `en` only
- a phase-2 block with 4 rows

The JSON file has keys `lang, mode, budget, seeds, phase1, phase2, comparison`. The log shows the HE GRU trained once, not once per candidate. Infeasible rows are acceptable at this size. A crash is not.

- [ ] **Step 7: Commit**

```bash
git add scripts/tune_char_gru.py tests/test_tune_char_gru.py
git commit -m "feat(tuning): rank Char-GRU configs by FN at a quadgram FP budget

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Run the tuner and record the comparison

**Files:**
- Create: `evaluation/tune_char_gru_en.json`, `evaluation/tune_char_gru_en.log` (outputs)
- Modify: `evaluation/EXPERIMENTS.md` (new section at the end)

- [ ] **Step 1: Launch the full run in the background**

Run in the background. Expected wall time is 2.5–3 h: 45 configs × 2 seeds, only EN retrained per config, plus phase 2.
`$PY scripts/tune_char_gru.py --lang en --coarse-to-fine --seeds 0 1 --out evaluation/tune_char_gru_en.json > evaluation/tune_char_gru_en.log 2>&1`

Wait for the completion notification. Do not poll.

- [ ] **Step 2: Check the run is usable**

Confirm all of the following. If any fails, stop and report rather than re-running with changed settings:
- The log ends with the PHASE 2 block.
- The budget values are non-zero.
- At least one phase-1 row is feasible.

- [ ] **Step 3: Write the results section in `evaluation/EXPERIMENTS.md`**

Add a section `## Char-GRU tuning by FN at an FP budget — measured (2026-10-09)` containing:
- **Setup:**
  - `--lang en --coarse-to-fine`: EN hyperparameters searched, HE GRU at the baseline.
  - About 1M training words per language, every 5th line held out.
  - 2 seeds, the budget values, the delta grid.
- **Phase-1 winner vs the baseline row:** hyperparameters, chosen Δ, FP, FN, latency.
- **Comparison table:** the 4 rows from `comparison` with floor, margin, Δ, FP/FN per language, FN sum, median latency.
- **Validation-loss check:** one sentence on whether phase-1 order by EN validation loss matched order by objective A. Compute it from the JSON.
- **Caveats:**
  - Phase 2 tuned the knobs on the phase-1 winner only.
  - One split, not k-fold.
  - Infeasible rows, if any.

- [ ] **Step 4: Commit**

```bash
git add evaluation/tune_char_gru_en.json evaluation/tune_char_gru_en.log evaluation/EXPERIMENTS.md
git commit -m "docs(evaluation): EN Char-GRU tuning by FN at FP budget, floor/fast-switch ablations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Report to the user**

Report:
- The comparison table.
- The winning hyperparameters, floor and margin, as a proposal for `GRU_HPARAMS`, `UNIGRAM_FLOOR` and `FAST_SWITCH_MARGIN`. Do not apply them.
- If the EN winner differs from `BASELINE_PARAMS`, `GRU_HPARAMS` needs splitting per language before shipping. Flag it; don't build it.
- That shipping needs a retrain with `scripts/train_char_gru.py`, because the shipped GRU has no unigram table, so the floor is a no-op in the app today.
