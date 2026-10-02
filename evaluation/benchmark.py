"""
benchmark.py — SwitchLang evaluation test harness.

Measures false positive rate, false negative rate, and detection latency
by replaying text files through the evaluation engine — no OS hooks needed.

The simulation mirrors hooks.py behaviour: mid-word evaluation (≥3 chars),
delimiter evaluation, sensitivity decay, collision blacklist, and retroactive
lookback via history deque / correction blocks.

Usage:
    python benchmark.py                                   # en_corpus.txt, all lines
    python benchmark.py --max-lines 1000                  # first 1000 lines
    python benchmark.py data/he_corpus.txt                # Hebrew corpus
    python benchmark.py myfile.txt --lang en              # explicit language
    python benchmark.py --test fp                         # only false-positive test
    python benchmark.py --holdout-frac 0.1                # evaluate on held-out tail
    python benchmark.py --jobs 1                          # serial (default: all cores)
"""

import argparse
import atexit
import collections
import io
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields

# Force UTF-8 output on Windows (cp1252 can't encode Hebrew characters)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import EvaluationEngine
from core.quadgram import load_models, QuadgramModel
from core.neural_model import load_neural_models, CharNeuralModel
from core.sensitivity import SensitivityManager
from core.keymap import shadow

# Re-use the same namedtuple that hooks.py uses for lookback history
_WordEntry = collections.namedtuple(
    '_WordEntry', ['active', 'shadow', 'delimiter', 'is_colliding', 'is_ambiguous', 'evidence']
)


# ═══════════════════════════════════════════════════════════════════════════
# RESULT DATA-CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WordResult:
    """Outcome of evaluating a single word through the engine."""
    word: str
    buffer_active: str
    buffer_shadow: str
    switched: bool
    switch_char_idx: int = -1   # ≥0 -> mid-word index;  -1 -> delimiter
    score_diff: float = 0.0
    is_colliding: bool = False
    is_ambiguous: bool = False
    score_active: float = 0.0   # log-prob of the current word on the active layout
    score_shadow: float = 0.0   # ... and on the other layout, at the decision point


@dataclass
class FPReport:
    """False-positive test results."""
    lang: str
    lines_tested: int = 0
    words_tested: int = 0
    fp_count: int = 0
    recovery_count: int = 0
    lines_with_fp: int = 0
    flagged_lines: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fp_per_1k(self):
        return (self.fp_count / self.words_tested * 1000) if self.words_tested else 0.0


@dataclass
class FNReport:
    """False-negative test results."""
    lang: str
    lines_tested: int = 0
    words_tested: int = 0
    lines_switched: int = 0
    words_not_switched: int = 0
    total_latency_chars: int = 0
    latency_values: list = field(default_factory=list)
    flagged_lines: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fn_per_1k(self):
        return (self.words_not_switched / self.words_tested * 1000) if self.words_tested else 0.0


@dataclass
class MixedReport:
    """'Forgot to switch' results: correct prefix words, then wrong-layout words."""
    lang: str
    cases: int = 0
    prefix_fp: int = 0          # cases that switched during the correct prefix (excluded below)
    suffix_words: int = 0
    uncorrected: int = 0        # suffix words left on the wrong layout
    over_corrected: int = 0     # prefix words the correction block wrongly flipped
    latency_values: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fn_per_1k(self):
        return (self.uncorrected / self.suffix_words * 1000) if self.suffix_words else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION HARNESS
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationHarness:
    """Replays text through the SwitchLang engine to measure accuracy."""

    accumulate = True   # carry switch evidence across words (CUSUM), as hooks.py does
    alpha = 0.1         # sensitivity decay; 0.1 is what main.py ships

    def __init__(self, data_dir, en_model_path=None, he_model_path=None,
                 mode='standard', req_confirmations=2, model_type='quadgram',
                 accumulate=True, alpha=0.1):
        self.model_type = model_type
        self.accumulate = accumulate
        self.alpha = alpha
        if model_type == 'neural':
            models = load_neural_models(data_dir)
            if en_model_path:
                models['en'] = CharNeuralModel(en_model_path)
            if he_model_path:
                models['he'] = CharNeuralModel(he_model_path)
            collisions_path = os.path.join(data_dir, 'collisions.json')
            self.engine = EvaluationEngine(
                models['en'], models['he'],
                collisions_path=collisions_path,
                enable_logging=False,
                model_mode=mode,
                model_type='neural'
            )
        else:
            models = load_models(data_dir, load_so=(mode == 'technical'))
            # Allow overriding individual model files
            if en_model_path:
                models['en'] = QuadgramModel(en_model_path)
            if he_model_path:
                models['he'] = QuadgramModel(he_model_path)
            collisions_path = os.path.join(data_dir, 'collisions.json')
            self.engine = EvaluationEngine(
                models['en'], models['he'],
                collisions_path=collisions_path,
                enable_logging=False,
                en_so_model=models.get('so'),
                model_mode=mode,
                model_type='quadgram'
            )
        self.req_confirmations = req_confirmations

    def _score_text_detailed(self, text, layout):
        """Score text and return (total_score, state_dict) for incremental tracking."""
        mode = self.engine.model_mode
        if layout == 'en':
            score_std = self.engine.en_model.score(text)
            state = {'std': score_std}
            if mode == 'technical' and self.engine.en_so_model:
                score_so = self.engine.en_so_model.score(text)
                state['so'] = score_so
                return max(score_std, score_so), state
            return score_std, state
        else:
            score = self.engine.he_model.score(text)
            return score, {'score': score}

    @staticmethod
    def _score_incremental_char(model, prev3, new_char):
        """Score a single new character given the previous three using model's trie."""
        if len(prev3) < 3:
            return 0.0
        quadgram = (prev3[-3:] + new_char).lower()
        trigram = prev3[-3:].lower()
        quad_count = model.count(quadgram)
        tri_count = model.count(trigram)
        return math.log((quad_count + 1) / (tri_count + model.vocab_size))

    def _score_incremental_detailed(self, state, prev3, new_char, layout):
        """Score a single new character given previous state and return (total_score, new_state)."""
        mode = self.engine.model_mode
        if layout == 'en':
            inc_std = self._score_incremental_char(self.engine.en_model, prev3, new_char)
            new_std = state['std'] + inc_std
            new_state = {'std': new_std}
            if 'so' in state and self.engine.en_so_model and mode == 'technical':
                inc_so = self._score_incremental_char(self.engine.en_so_model, prev3, new_char)
                new_so = state['so'] + inc_so
                new_state['so'] = new_so
                return max(new_std, new_so), new_state
            return new_std, new_state
        else:
            inc = self._score_incremental_char(self.engine.he_model, prev3, new_char)
            new_score = state['score'] + inc
            return new_score, {'score': new_score}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _other(layout):
        return 'he' if layout == 'en' else 'en'

    @staticmethod
    def _shadow_dir(text_lang):
        """Return the shadow direction string for keymap.shadow()."""
        return 'en_to_he' if text_lang == 'en' else 'he_to_en'

    def _get_buffers(self, word, text_lang, current_layout):
        """Map *word* (in *text_lang*) to (buffer_active, buffer_shadow).

        The physical keys the user presses correspond to characters in
        *text_lang*.  Whether those keys produce *text_lang* chars or the
        other layout's chars depends on *current_layout*.
        """
        other_text = shadow(word, self._shadow_dir(text_lang))
        if current_layout == text_lang:
            return word, other_text          # correct layout
        else:
            return other_text, word          # wrong layout

    @staticmethod
    def _neural_start(model, prefix):
        """State (score, next-char log-probs, hidden) after reading ' ' + prefix."""
        log_probs, h = model.step(' ', model.init_state())
        state = (0.0, log_probs, h)
        for ch in prefix:
            state = EvaluationHarness._neural_advance(model, state, ch)
        return state

    @staticmethod
    def _neural_advance(model, state, ch):
        """Append *ch*: add log P(ch | text so far), then step the GRU over it."""
        total, log_probs, h = state
        ch = ch.lower()
        total += float(log_probs[model.char_to_idx.get(ch, model.unk_idx)])
        log_probs, h = model.step(ch, h)
        return total, log_probs, h

    def _prior(self, history):
        """Evidence carried into the next word: the last entry's, 0 after a clear."""
        return history[-1].evidence if (self.accumulate and history) else 0.0

    def _simulate_word(self, word, text_lang, current_layout, sensitivity, evidence=0.0):
        """Evaluate one word character-by-character using incremental O(1) scoring, then on delimiter."""
        buf_active, buf_shadow = self._get_buffers(word, text_lang, current_layout)

        target_layout = 'he' if current_layout == 'en' else 'en'
        consecutive_hits = 0
        if self.model_type == 'neural':
            if current_layout == 'en':
                model_a, model_s = self.engine.en_model, self.engine.he_model
            else:
                model_a, model_s = self.engine.he_model, self.engine.en_model
            # Incremental equivalent of model.score(' ' + prefix): one GRU step per
            # keystroke instead of re-running the whole prefix (same floats, same order).
            state_a = self._neural_start(model_a, buf_active[:2])
            state_s = self._neural_start(model_s, buf_shadow[:2])
            consecutive_hits = 0
            for i in range(2, len(buf_active)):
                partial_a = buf_active[:i + 1]
                partial_s = buf_shadow[:i + 1]

                state_a = self._neural_advance(model_a, state_a, buf_active[i])
                state_s = self._neural_advance(model_s, state_s, buf_shadow[i])
                score_a, score_s = state_a[0], state_s[0]

                diff = score_s - score_a
                coll = self.engine.check_collision(partial_a, partial_s)
                should = (diff + evidence > sensitivity.delta) if not coll else False
                amb = (not should and diff + evidence > 0 and not coll)

                if should:
                    consecutive_hits += 1
                    # Adaptive K: decisive score (> delta + 2.5) switches at K=1 (character 3!)
                    effective_k = 1 if (diff + evidence > sensitivity.delta + 2.5) else self.req_confirmations
                    if consecutive_hits >= effective_k:
                        return WordResult(
                            word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
                            switched=True, switch_char_idx=i,
                            score_diff=diff, is_colliding=coll, is_ambiguous=amb,
                            score_active=score_a, score_shadow=score_s,
                        )
                else:
                    consecutive_hits = 0

            # Delimiter evaluation
            score_a_del = state_a[0] + float(state_a[1][model_a.char_to_idx[' ']])
            score_s_del = state_s[0] + float(state_s[1][model_s.char_to_idx[' ']])

            diff = score_s_del - score_a_del
            coll = self.engine.check_collision(buf_active, buf_shadow)
            should = (diff + evidence > sensitivity.delta) if not coll else False
            amb = (not should and diff + evidence > 0 and not coll)
            return WordResult(
                word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
                switched=should, switch_char_idx=-1,
                score_diff=diff, is_colliding=coll, is_ambiguous=amb,
                score_active=score_a_del, score_shadow=score_s_del,
            )

        act_state = None
        shd_state = None
        diff = 0.0
        coll = False
        amb = False

        for i in range(2, len(buf_active)):
            partial_a = buf_active[:i + 1]
            partial_s = buf_shadow[:i + 1]

            if i == 2:
                score_a, act_state = self._score_text_detailed(' ' + partial_a, current_layout)
                score_s, shd_state = self._score_text_detailed(' ' + partial_s, target_layout)
            else:
                prev3_a = (' ' + buf_active[:i])[-3:]
                score_a, act_state = self._score_incremental_detailed(
                    act_state, prev3_a, partial_a[-1], current_layout
                )
                prev3_s = (' ' + buf_shadow[:i])[-3:]
                score_s, shd_state = self._score_incremental_detailed(
                    shd_state, prev3_s, partial_s[-1], target_layout
                )

            diff = score_s - score_a
            coll = self.engine.check_collision(partial_a, partial_s)
            should = (diff + evidence > sensitivity.delta) if not coll else False
            amb = (not should and diff + evidence > 0 and not coll)

            if should:
                consecutive_hits += 1
                if consecutive_hits >= self.req_confirmations:
                    return WordResult(
                        word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
                        switched=True, switch_char_idx=i,
                        score_diff=diff, is_colliding=coll, is_ambiguous=amb,
                        score_active=score_a, score_shadow=score_s,
                    )
            else:
                consecutive_hits = 0

        # delimiter evaluation
        if act_state is not None:
            prev3_a = (' ' + buf_active)[-3:]
            score_a_del, _ = self._score_incremental_detailed(
                act_state, prev3_a, ' ', current_layout
            )
            prev3_s = (' ' + buf_shadow)[-3:]
            score_s_del, _ = self._score_incremental_detailed(
                shd_state, prev3_s, ' ', target_layout
            )
            diff = score_s_del - score_a_del
            coll = self.engine.check_collision(buf_active, buf_shadow)
            should = (diff + evidence > sensitivity.delta) if not coll else False
            amb = (not should and diff + evidence > 0 and not coll)
        else:
            should, diff, coll, amb = self.engine.evaluate(
                buf_active, buf_shadow, sensitivity.delta,
                current_layout=current_layout, on_delimiter=True, evidence=evidence,
            )
            score_a_del = score_s_del = 0.0
            if should:  # scores only matter for reporting a switch
                score_a_del = self._score_text_detailed(' ' + buf_active + ' ', current_layout)[0]
                score_s_del = self._score_text_detailed(' ' + buf_shadow + ' ', target_layout)[0]

        return WordResult(
            word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
            switched=should, switch_char_idx=-1,
            score_diff=diff, is_colliding=coll, is_ambiguous=amb,
            score_active=score_a_del, score_shadow=score_s_del,
        )

    @staticmethod
    def _build_correction_block(history):
        """Contiguous correctable tail — same logic as hooks.py."""
        block = []
        for entry in reversed(history):
            if entry.is_colliding or entry.is_ambiguous:
                block.append(entry)
            else:
                break
        block.reverse()
        return block

    @staticmethod
    def _switch_event(words, w_idx, res, layout, delta, kind):
        """Snapshot of a switch: where it fired, per-language word scores, and the text up to it."""
        word = words[w_idx]
        typed = word[:res.switch_char_idx + 1] if res.switch_char_idx >= 0 else word
        other = 'he' if layout == 'en' else 'en'
        scores = {layout: res.score_active, other: res.score_shadow}
        return dict(kind=kind, word_idx=w_idx, word=word, char_idx=res.switch_char_idx,
                    context=' '.join(words[:w_idx] + [typed]),
                    score_en=scores['en'], score_he=scores['he'],
                    diff=res.score_diff, delta=delta)

    # ------------------------------------------------------------------
    # FALSE-POSITIVE TEST
    # ------------------------------------------------------------------

    def test_false_positives(self, lines, text_lang, baseline_delta=6.0,
                             line_offset=0, progress=True):
        """Feed *valid* text on the *correct* layout.  Any switch = FP."""
        report = FPReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang

        for line_num, line in enumerate(lines, line_offset + 1):
            words = line.strip().split()
            if not words:
                continue
            report.lines_tested += 1

            current = correct
            sensitivity = SensitivityManager(baseline_delta=baseline_delta, alpha=self.alpha)
            history = collections.deque(maxlen=50)
            line_fps = 0
            events = []

            for w_idx, word in enumerate(words):
                if not word:
                    continue
                report.words_tested += 1
                res = self._simulate_word(word, text_lang, current, sensitivity, self._prior(history))

                if res.switched:
                    if current == correct:
                        # Switching AWAY from correct layout -> false positive
                        line_fps += 1
                        report.fp_count += 1
                        kind = 'first_fp' if line_fps == 1 else 'later_fp'
                    else:
                        # Switching BACK to correct layout -> recovery (not punished)
                        report.recovery_count += 1
                        kind = 'recovery'
                    events.append(self._switch_event(
                        words, w_idx, res, current, sensitivity.delta, kind))

                    current = self._other(current)
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                else:
                    buf_a, buf_s = self._get_buffers(word, text_lang, current)
                    history.append(_WordEntry(
                        active=buf_a, shadow=buf_s, delimiter=' ',
                        is_colliding=res.is_colliding, is_ambiguous=res.is_ambiguous,
                        evidence=EvaluationEngine.accumulate(
                            self._prior(history), res.score_diff, res.is_colliding),
                    ))
                    sensitivity.on_word_complete()

            if line_fps > 0:
                report.lines_with_fp += 1
                report.flagged_lines.append(dict(
                    line_num=line_num, n_fp=line_fps, events=events,
                    # severity: how far past the threshold the first FP was
                    margin=events[0]['diff'] - events[0]['delta'],
                ))

            if progress and line_num % 5000 == 0:
                print(f'  [FP] {line_num}/{len(lines)} lines …', flush=True)

        report.elapsed_sec = time.time() - t0
        return report

    # ------------------------------------------------------------------
    # FALSE-NEGATIVE TEST
    # ------------------------------------------------------------------

    def test_false_negatives(self, lines, text_lang, baseline_delta=6.0,
                             line_offset=0, progress=True):
        """Feed *inverted* text (wrong layout).  Failure to switch = FN.

        Latency = total chars typed on the wrong layout from the start of
        the line until the first switch fires (includes ALL wrong chars,
        not just those eventually corrected).
        """
        report = FNReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang
        wrong = self._other(text_lang)

        for line_num, line in enumerate(lines, line_offset + 1):
            words = line.strip().split()
            if not words:
                continue
            report.lines_tested += 1

            current = wrong                # start on WRONG layout
            wrong_since = 0                # first word of the current wrong-layout stretch
            sensitivity = SensitivityManager(baseline_delta=baseline_delta, alpha=self.alpha)
            history = collections.deque(maxlen=50)

            latency_chars = 0
            first_switch_word = None
            switch_event = None
            best_diff = None            # closest the wrong-layout words came to switching
            line_words_not_switched = 0
            line_switched = False

            for w_idx, word in enumerate(words):
                if not word:
                    continue
                report.words_tested += 1

                res = self._simulate_word(word, text_lang, current, sensitivity, self._prior(history))

                if current != correct and (best_diff is None or res.score_diff > best_diff):
                    best_diff = res.score_diff

                if res.switched and current != correct:
                    # ── correct switch detected ──
                    line_switched = True
                    if first_switch_word is None:
                        first_switch_word = w_idx
                        switch_event = self._switch_event(
                            words, w_idx, res, current, sensitivity.delta, 'switch')

                    # Latency: chars typed so far in wrong layout
                    if res.switch_char_idx >= 0:
                        # mid-word switch: count chars up to and including trigger
                        latency_chars += res.switch_char_idx + 1
                    else:
                        # delimiter switch: full word + delimiter
                        latency_chars += len(word) + 1

                    # Determine uncorrected words
                    block = self._build_correction_block(history)
                    corrected = len(block) + 1          # block + trigger word
                    total_wrong = w_idx + 1 - wrong_since  # words in this wrong-layout stretch
                    uncorrected = max(0, total_wrong - corrected)
                    line_words_not_switched += uncorrected

                    current = self._other(current)
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                elif res.switched:
                    # switched while already on correct layout (FP within FN test)
                    # — just track the layout flip
                    current = self._other(current)
                    wrong_since = w_idx + 1
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                else:
                    # no switch — accumulate into history
                    if current != correct:
                        # still on wrong layout: this word contributes to latency
                        latency_chars += len(word) + 1  # word + space
                    buf_a, buf_s = self._get_buffers(word, text_lang, current)
                    history.append(_WordEntry(
                        active=buf_a, shadow=buf_s, delimiter=' ',
                        is_colliding=res.is_colliding, is_ambiguous=res.is_ambiguous,
                        evidence=EvaluationEngine.accumulate(
                            self._prior(history), res.score_diff, res.is_colliding),
                    ))
                    sensitivity.on_word_complete()

            # If engine never switched, entire line is FN
            if not line_switched:
                line_words_not_switched = len(words)
                latency_chars = sum(len(w) for w in words) + len(words)

            report.words_not_switched += line_words_not_switched
            report.latency_values.append(latency_chars)
            report.total_latency_chars += latency_chars

            if line_switched:
                report.lines_switched += 1

            # Flag lines with issues
            if not line_switched or line_words_not_switched > 0:
                report.flagged_lines.append(dict(
                    line_num=line_num, text=line.strip(), n_words=len(words),
                    uncorrected=line_words_not_switched, latency=latency_chars,
                    switch=switch_event, best_diff=best_diff, delta=baseline_delta,
                ))

            if progress and line_num % 5000 == 0:
                print(f'  [FN] {line_num}/{len(lines)} lines …', flush=True)

        report.elapsed_sec = time.time() - t0
        return report

    # ------------------------------------------------------------------
    # MIXED ("FORGOT TO SWITCH") TEST
    # ------------------------------------------------------------------

    def test_mixed(self, pairs, text_lang, baseline_delta=6.0, n_prefix=3,
                   line_offset=0, progress=True):
        """Each case: *n_prefix* words of the other language typed correctly, then
        a *text_lang* line typed on that same (now wrong) layout.

        Latency counts wrong-layout chars from the start of the suffix.
        """
        report = MixedReport(lang=text_lang)
        t0 = time.time()
        other = self._other(text_lang)

        for line, prefix_line in pairs:
            prefix = prefix_line.split()[:n_prefix]
            suffix = line.split()
            if len(prefix) < n_prefix or not suffix:
                continue
            report.cases += 1

            seq = [(w, other) for w in prefix] + [(w, text_lang) for w in suffix]
            sensitivity = SensitivityManager(baseline_delta=baseline_delta, alpha=self.alpha)
            history = collections.deque(maxlen=50)
            latency = 0

            for i, (word, lang) in enumerate(seq):
                res = self._simulate_word(word, lang, other, sensitivity, self._prior(history))
                if res.switched:
                    if i < n_prefix:
                        report.prefix_fp += 1
                    else:
                        # Correction block is the history tail: words first..i-1.
                        first = i - len(self._build_correction_block(history))
                        report.over_corrected += max(0, n_prefix - first)
                        report.uncorrected += max(0, first - n_prefix)
                        latency += (res.switch_char_idx + 1 if res.switch_char_idx >= 0
                                    else len(word) + 1)
                    break
                if i >= n_prefix:
                    latency += len(word) + 1
                buf_a, buf_s = self._get_buffers(word, lang, other)
                history.append(_WordEntry(
                    active=buf_a, shadow=buf_s, delimiter=' ',
                    is_colliding=res.is_colliding, is_ambiguous=res.is_ambiguous,
                    evidence=EvaluationEngine.accumulate(
                        self._prior(history), res.score_diff, res.is_colliding),
                ))
                sensitivity.on_word_complete()
            else:
                report.uncorrected += len(suffix)   # never switched

            if not (res.switched and i < n_prefix):
                report.suffix_words += len(suffix)
                report.latency_values.append(latency)

        report.elapsed_sec = time.time() - t0
        return report


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
#
# Every line resets layout, sensitivity and history, so lines are independent
# and splitting them across processes reproduces the serial result exactly.

# Size 1: a variant is scored FP then FN, so the second call reuses the models
# while memory stays at one model set per worker rather than one per variant.
_worker_models = (None, None)


def _test_method(harness, test):
    return {'fp': harness.test_false_positives, 'fn': harness.test_false_negatives,
            'mixed': harness.test_mixed}[test]


def _run_chunk(job):
    global _worker_models
    test, offset, lines, lang, delta, key, kw = job
    if _worker_models[0] != key:
        _worker_models = (key, EvaluationHarness(*key))
    return _test_method(_worker_models[1], test)(
        lines, lang, delta, line_offset=offset, progress=False, **kw)


def _merge_reports(reports):
    """Sum counters and concatenate lists across chunk reports."""
    merged = reports[0]
    for report in reports[1:]:
        for f in fields(merged):
            value = getattr(report, f.name)
            if isinstance(value, bool) or f.name == 'lang':
                continue
            if isinstance(value, (int, float)):
                setattr(merged, f.name, getattr(merged, f.name) + value)
            elif isinstance(value, list):
                getattr(merged, f.name).extend(value)
    return merged


_pool = None
_pool_jobs = None


def _get_pool(jobs):
    """One pool for the whole process — respawning it per call costs seconds."""
    global _pool, _pool_jobs
    if _pool is None or _pool_jobs != jobs:
        shutdown_pool()
        _pool = ProcessPoolExecutor(max_workers=jobs)
        _pool_jobs = jobs
    return _pool


def shutdown_pool():
    global _pool, _pool_jobs
    if _pool is not None:
        _pool.shutdown()
        _pool, _pool_jobs = None, None


atexit.register(shutdown_pool)


def run_test(test, lines, lang, delta, data_dir, en_model_path=None,
             he_model_path=None, mode='standard', jobs=11,
             req_confirmations=2, model_type='quadgram', accumulate=True, alpha=0.1,
             **kw):
    """Run the 'fp', 'fn' or 'mixed' test over *lines*, optionally across processes.

    For 'mixed', *lines* are (line, prefix_line) pairs and *kw* carries n_prefix.
    """
    key = (data_dir, en_model_path, he_model_path, mode, req_confirmations, model_type,
           accumulate, alpha)

    if jobs <= 1:
        return _test_method(EvaluationHarness(*key), test)(lines, lang, delta, **kw)

    # Several chunks per worker keeps them busy when line lengths vary.
    n_chunks = jobs * 4
    size = max(1, -(-len(lines) // n_chunks))
    chunks = [(test, i, lines[i:i + size], lang, delta, key, kw)
              for i in range(0, len(lines), size)]

    t0 = time.time()
    reports = list(_get_pool(jobs).map(_run_chunk, chunks))
    merged = _merge_reports(reports)
    merged.elapsed_sec = time.time() - t0
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# REPORT PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def load_corpus_lines(path, lang, cap=None):
    """Read *path*, keeping non-empty lines written purely in *lang*'s script.

    Mixed-script lines are excluded: a real layout switch fires a CRE, so each
    script run is an independent segment rather than one continuous context.
    """
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if not l:
                continue

            has_hebrew = bool(re.search(r'[\u0590-\u05FF]', l))
            has_english = bool(re.search(r'[a-zA-Z]', l))
            has_nikud = bool(re.search(r'[\u0591-\u05C7]', l))

            if lang == 'he' and (not has_hebrew or has_english or has_nikud):
                continue
            if lang == 'en' and (not has_english or has_hebrew):
                continue

            lines.append(l)
            if cap is not None and len(lines) >= cap:
                break
    return lines


def _pct(num, denom):
    return (num / denom * 100) if denom else 0.0


def print_fp_report(report, corpus_path, model_path, provenance='', max_flagged=25,
                    delta=None, req_confirmations=None, mode=None, jobs=None):
    print(f'\n{"=" * 65}')
    print(f' FALSE POSITIVE TEST  (valid {report.lang.upper()}, layout={report.lang})')
    print(f'{"=" * 65}')
    print(f'Corpus:           {corpus_path}')
    print(f'Split:            {provenance}')
    print(f'Model:            {model_path}')
    if delta is not None:
        print(f'Delta (Δ):        {delta}')
    if req_confirmations is not None:
        print(f'Confirmations (K):{req_confirmations}')
    if mode is not None:
        print(f'Mode:             {mode}')
    if jobs is not None:
        print(f'Jobs:             {jobs}')
    print(f'Lines tested:     {report.lines_tested}')
    print(f'Words tested:     {report.words_tested}')
    fpr = _pct(report.fp_count, report.words_tested)
    print(f'False positives:  {report.fp_count}/{report.words_tested}  ({fpr:.3f}%)')
    print(f'FP per 1k words:  {report.fp_per_1k:.3f}')
    print(f'  first in line:  {report.lines_with_fp}   later in line: {report.fp_count - report.lines_with_fp}')
    print(f'Lines with FP:    {report.lines_with_fp}/{report.lines_tested}')
    print(f'Recoveries:       {report.recovery_count}')
    print(f'Time:             {report.elapsed_sec:.1f}s')

    if report.flagged_lines:
        worst = sorted(report.flagged_lines, key=lambda r: (r['n_fp'], r['margin']), reverse=True)
        print(f'\nWorst {min(max_flagged, len(worst))} of {len(worst)} lines with FPs '
              f'(most FPs, then furthest past Δ; scores are log-probs of the word being typed):')
        for rank, r in enumerate(worst[:max_flagged], 1):
            n_rec = sum(e['kind'] == 'recovery' for e in r['events'])
            print(f'\n #{rank:<3} line {r["line_num"]} · {r["n_fp"]} FP · {n_rec} recovery · '
                  f'margin {r["margin"]:+.2f}')
            for e in r['events']:
                print(_format_event(e))
        _print_remaining(len(worst), max_flagged)


def print_fn_report(report, corpus_path, model_path, provenance='', max_flagged=25,
                    delta=None, req_confirmations=None, mode=None, jobs=None):
    other_lang = 'he' if report.lang == 'en' else 'en'
    print(f'\n{"=" * 65}')
    print(f' FALSE NEGATIVE TEST  (inverted {report.lang.upper()}, layout={other_lang})')
    print(f'{"=" * 65}')
    print(f'Corpus:             {corpus_path}')
    print(f'Split:              {provenance}')
    print(f'Model:              {model_path}')
    if delta is not None:
        print(f'Delta (Δ):          {delta}')
    if req_confirmations is not None:
        print(f'Confirmations (K):  {req_confirmations}')
    if mode is not None:
        print(f'Mode:               {mode}')
    if jobs is not None:
        print(f'Jobs:               {jobs}')
    print(f'Lines tested:       {report.lines_tested}')
    print(f'Words tested:       {report.words_tested}')
    sr = _pct(report.lines_switched, report.lines_tested)
    print(f'Lines switched:     {report.lines_switched}/{report.lines_tested}  ({sr:.3f}%)')
    fnr = _pct(report.words_not_switched, report.words_tested)
    print(f'Words not switched: {report.words_not_switched}  ({fnr:.3f}%)')
    print(f'FN per 1k words:    {report.fn_per_1k:.3f}')

    if report.latency_values:
        vals = report.latency_values
        mean_l = statistics.mean(vals)
        median_l = statistics.median(vals)
        s = sorted(vals)
        p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
        print(f'Mean latency:       {mean_l:.1f} chars')
        print(f'Median latency:     {median_l:.0f} chars')
        print(f'P95 latency:        {p95} chars')

    print(f'Time:               {report.elapsed_sec:.1f}s')

    if report.flagged_lines:
        worst = sorted(report.flagged_lines, key=lambda r: (r['uncorrected'], r['latency']), reverse=True)
        print(f'\nWorst {min(max_flagged, len(worst))} of {len(worst)} flagged lines '
              f'(most uncorrected words, then latency; text shown is what the user meant to type):')
        for rank, r in enumerate(worst[:max_flagged], 1):
            print(f'\n #{rank:<3} line {r["line_num"]} · {r["uncorrected"]}/{r["n_words"]} words uncorrected · '
                  f'latency {r["latency"]} chars')
            if r['switch']:
                print(_format_event(r['switch']))
            else:
                best = f'{r["best_diff"]:+.2f}' if r['best_diff'] is not None else 'n/a'
                print(f'     NO SWITCH  best diff {best} vs Δ={r["delta"]:.2f}')
                print(f'       {_tail(r["text"])}')
        _print_remaining(len(worst), max_flagged)


def _tail(text, width=160):
    return text if len(text) <= width else '…' + text[-width:]


def _format_event(e):
    """Two lines: what fired with per-language scores, then the text up to the switch."""
    label = {'first_fp': 'FIRST FP', 'later_fp': 'LATER FP',
             'recovery': 'recovery', 'switch': 'SWITCH'}[e['kind']]
    where = f'mid-word @char {e["char_idx"] + 1}' if e['char_idx'] >= 0 else 'on delimiter'
    return (f'     {label:<9} word {e["word_idx"] + 1} "{e["word"]}" ({where})  '
            f'EN={e["score_en"]:.2f}  HE={e["score_he"]:.2f}  '
            f'diff={e["diff"]:+.2f} > Δ={e["delta"]:.2f}\n'
            f'       {_tail(e["context"])}')


def _print_remaining(total, shown):
    if total > shown:
        print(f'\n  … and {total - shown} more')


def explain_word(word, data_dir, baseline_delta=6.0, mode='standard', req_confirmations=2,
                 en_model_path=None, he_model_path=None):
    """Provide step-by-step diagnostic scoring for a single word."""
    models = load_models(data_dir, load_so=(mode == 'technical'))
    if en_model_path:
        models['en'] = QuadgramModel(en_model_path)
    if he_model_path:
        models['he'] = QuadgramModel(he_model_path)
    collisions_path = os.path.join(data_dir, 'collisions.json')
    engine = EvaluationEngine(
        models['en'], models['he'],
        collisions_path=collisions_path,
        enable_logging=False,
        en_so_model=models.get('so'),
        model_mode=mode,
    )

    def _format_breakdown(model, text, label):
        text = text.lower()
        print(f"\n--- {label}: [{text!r}] ---")
        if len(text) < 2:
            print("  Text too short to score.")
            return 0.0

        v = model.vocab_size

        if len(text) == 2:
            count = model.count(text)
            total = model._bigram_first_totals.get(text[0], 0)
            prob = (count + 1) / (total + v)
            step_log = math.log(prob)
            print(f"  [BIGRAM ONLY] {text!r} -> count={count:,}, total_first={total:,}, prob={prob:.4e}, log_prob={step_log:.4f}")
            print(f"  Total log-prob: {step_log:.4f}")
            return step_log

        if len(text) == 3:
            tri_count = model.count(text)
            bi_count = model.count(text[:2])
            prob = (tri_count + 1) / (bi_count + v)
            step_log = math.log(prob)
            print(f"  [TRIGRAM ONLY] {text!r} (bi={text[:2]!r}) -> tri={tri_count:,}, bi={bi_count:,}, prob={prob:.4e}, log_prob={step_log:.4f}")
            print(f"  Total log-prob: {step_log:.4f}")
            return step_log

        first_bi = text[:2]
        bi_cnt = model.count(first_bi)
        denom = model.total_bigrams + (v ** 2)
        step_prob = (bi_cnt + 1) / denom
        step_log = math.log(step_prob)
        total_log_prob = step_log
        print(f"  [START BIGRAM] {first_bi!r} -> count={bi_cnt:,}, denom={denom:,}, log_prob={step_log:.4f}")

        for i in range(len(text) - 3):
            quad = text[i:i + 4]
            tri = text[i:i + 3]
            q_cnt = model.count(quad)
            t_cnt = model.count(tri)
            s_prob = (q_cnt + 1) / (t_cnt + v)
            s_log = math.log(s_prob)
            total_log_prob += s_log
            print(f"  [QUADGRAM] {quad!r} (tri={tri!r}) -> quad={q_cnt:,}, tri={t_cnt:,}, step_prob={s_prob:.4e}, log_prob={s_log:.4f} (cum={total_log_prob:.4f})")

        print(f"  Total log-prob: {total_log_prob:.4f}")
        return total_log_prob

    has_hebrew = bool(re.search(r'[\u0590-\u05FF]', word))
    word_clean = word.strip()
    word_en = word_clean if not has_hebrew else shadow(word_clean, 'he_to_en')
    word_he = shadow(word_en, 'en_to_he')

    print("=" * 80)
    print(f" EXPLAIN SCORING: EN='{word_en}' <-> HE='{word_he}' (delta={baseline_delta}, confirmations={req_confirmations}, mode={mode})")
    print("=" * 80)

    for layout, text_active, text_shadow, target_layout in [
        ('en', word_en, word_he, 'he'),
        ('he', word_he, word_en, 'en'),
    ]:
        print(f"\n{'#' * 80}")
        print(f" SIMULATING TYPING IN {layout.upper()} LAYOUT (Active: '{text_active}', Shadow: '{text_shadow}')")
        print(f"{'#' * 80}")

        _format_breakdown(models[layout], ' ' + text_active + ' ', f"ACTIVE MODEL ({layout.upper()})")
        if layout == 'en' and mode == 'technical' and models.get('so'):
            so_log = _format_breakdown(models['so'], ' ' + text_active + ' ', "ACTIVE SO MODEL (Stack Overflow)")
            en_log = models['en'].score(' ' + text_active + ' ')
            print(f"  [TECHNICAL MODE ACTIVE] max(EN={en_log:.4f}, SO={so_log:.4f}) = {max(en_log, so_log):.4f}")

        _format_breakdown(models[target_layout], ' ' + text_shadow + ' ', f"SHADOW MODEL ({target_layout.upper()})")
        if target_layout == 'en' and mode == 'technical' and models.get('so'):
            so_log = _format_breakdown(models['so'], ' ' + text_shadow + ' ', "SHADOW SO MODEL (Stack Overflow)")
            en_log = models['en'].score(' ' + text_shadow + ' ')
            print(f"  [TECHNICAL MODE SHADOW] max(EN={en_log:.4f}, SO={so_log:.4f}) = {max(en_log, so_log):.4f}")

        print(f"\nKeystroke-by-keystroke progression in {layout.upper()}:")
        print(f"{'char':<6} {'partial_active':<16} {'partial_shadow':<16} {'diff':>8} {'consec':>7} {'collision':>10} {'switch?':>10}")
        print("-" * 77)

        consec = 0
        switched = False
        for i in range(len(text_active)):
            if i < 2:
                print(f"{text_active[i]:<6} {text_active[:i+1]:<16} {text_shadow[:i+1]:<16} {'(len<3)':>8} {consec:>7} {'-':>10} {'NO':>10}")
                continue
            pa = text_active[:i+1]
            ps = text_shadow[:i+1]
            should, diff, coll, amb = engine.evaluate(pa, ps, baseline_delta, current_layout=layout)
            if should:
                consec += 1
                did_fire = (consec >= req_confirmations)
            else:
                consec = 0
                did_fire = False

            switch_str = "YES" if did_fire else "NO"
            if did_fire and not switched:
                switch_str = "**SWITCH**"
                switched = True

            coll_str = "YES" if coll else "NO"
            print(f"{text_active[i]:<6} {pa:<16} {ps:<16} {diff:>+8.3f} {consec:>7} {coll_str:>10} {switch_str:>10}")

        # Delimiter evaluation
        should_del, diff_del, coll_del, amb_del = engine.evaluate(text_active, text_shadow, baseline_delta, current_layout=layout, on_delimiter=True)
        coll_del_str = "YES" if coll_del else "NO"
        switch_del_str = "**SWITCH**" if should_del else "NO"
        print(f"{'<del>':<6} {text_active:<16} {text_shadow:<16} {diff_del:>+8.3f} {'-':>7} {coll_del_str:>10} {switch_del_str:>10}")

    print("\n" + "=" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SwitchLang evaluation test harness — measures FP, FN, and latency.',
    )
    parser.add_argument(
        'text_file', nargs='?', default=None,
        help='Path to text file (default: data/en_corpus.txt)',
    )
    parser.add_argument(
        '--lang', choices=['en', 'he'], default=None,
        help='Language of the text file.  Auto-detected from filename if omitted.',
    )
    parser.add_argument(
        '--max-lines', type=int, default=50_000,
        help='Max non-empty lines to process (default: entire file).',
    )
    parser.add_argument(
        '--baseline-delta', type=float, default=6.0,
        help='Initial score delta threshold (default: 6.0).',
    )
    parser.add_argument(
        '--test', choices=['fp', 'fn', 'both'], default='both',
        help='Which test(s) to run (default: both).',
    )
    parser.add_argument(
        '--data-dir', default=None,
        help='Path to data/ directory (default: auto-detect).',
    )
    parser.add_argument(
        '--en-model', default=None, metavar='PATH',
        help='Override path to English quadgram MARISA model (e.g. data/en_quadgrams.marisa).',
    )
    parser.add_argument(
        '--he-model', default=None, metavar='PATH',
        help='Override path to Hebrew quadgram MARISA model (e.g. data/he_quadgrams.marisa).',
    )
    parser.add_argument(
        '--holdout-frac', type=float, default=None, metavar='F',
        help='Evaluate only on the last F fraction of eligible lines (e.g. 0.1). '
             'The model must have been built without them, or this proves nothing.',
    )
    parser.add_argument(
        '--mode', choices=['standard', 'technical'], default='standard',
        help='Model mode (default: standard).  technical also scores against so_quadgrams.marisa.',
    )
    parser.add_argument(
        '-j', '--jobs', type=int, default=os.cpu_count() or 1, metavar='N',
        help='Worker processes for scoring (default: all cores).  1 disables.',
    )
    parser.add_argument(
        '--confirmations', type=int, default=2,
        help='Number of consecutive hits required for mid-word switch (default: 2).',
    )
    parser.add_argument(
        '--variants', default=None, metavar='VARS',
        help='Comma-separated variants to sweep as K:Δ[:α][:noacc], e.g. "k2:6:0.1,k2:3.5:0.3:noacc" '
             '(runs FP+FN head-to-head comparison; noacc stops carrying evidence across words).',
    )
    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='Sensitivity decay alpha (default: 0.1, the shipped value).',
    )
    parser.add_argument(
        '--mixed', default=None, metavar='N[,N]',
        help='With --variants: also run the "forgot to switch" test after N correct '
             'words of the other language, e.g. "3,6".',
    )
    parser.add_argument(
        '--explain', default=None, metavar='WORD',
        help='Explain step-by-step model scoring and switch evaluation for a single word.',
    )
    parser.add_argument(
        '--model-type', choices=['quadgram', 'neural'], default='quadgram',
        help='Model architecture to evaluate: quadgram (MARISA trie) or neural (Char-GRU).',
    )
    args = parser.parse_args()

    if args.confirmations < 1:
        parser.error('--confirmations must be at least 1')

    # ── resolve data dir ──
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data')

    # ── explain single word ──
    if args.explain:
        explain_word(args.explain, data_dir=args.data_dir, baseline_delta=args.baseline_delta,
                     mode=args.mode, req_confirmations=args.confirmations,
                     en_model_path=args.en_model, he_model_path=args.he_model)
        return

    # ── resolve text file ──
    if args.text_file is None:
        args.text_file = os.path.join(args.data_dir, 'en_corpus.txt')

    # ── auto-detect language ──
    if args.lang is None:
        bn = os.path.basename(args.text_file).lower()
        if bn.startswith('en') or 'en_' in bn:
            args.lang = 'en'
        elif bn.startswith('he') or 'he_' in bn:
            args.lang = 'he'
        else:
            parser.error(
                'Cannot auto-detect language from filename.  Use --lang en/he.'
            )

    # ── load text ──
    print(f'Loading text from {args.text_file} …')
    # A holdout needs the whole file scanned before the tail can be sliced off.
    cap = None if args.holdout_frac else args.max_lines
    lines = load_corpus_lines(args.text_file, args.lang, cap)

    eligible = len(lines)
    if args.holdout_frac:
        split_at = int(eligible * (1.0 - args.holdout_frac))
        lines = lines[split_at:]
        if args.max_lines is not None:
            lines = lines[:args.max_lines]
        provenance = (f'held out last {args.holdout_frac:.0%} of {eligible:,} '
                      f'eligible lines (from index {split_at:,})')
    else:
        provenance = 'NO HOLDOUT — results may be inflated by train/test overlap'

    print(f'Loaded {len(lines)} non-empty pure lines (lang={args.lang})')
    print(f'Split: {provenance}')

    if args.en_model:
        print(f'  EN model override: {args.en_model}')
    if args.he_model:
        print(f'  HE model override: {args.he_model}')

    # ── variant comparison sweep ──
    if args.variants:
        var_specs = [v.strip() for v in args.variants.split(',') if v.strip()]
        mixed_ns = [int(n) for n in args.mixed.split(',')] if args.mixed else []
        if mixed_ns:
            other = 'he' if args.lang == 'en' else 'en'
            prefix_lines = [l for l in load_corpus_lines(
                                os.path.join(args.data_dir, f'{other}_corpus.txt'), other,
                                cap=len(lines) * 2)
                            if len(l.split()) >= max(mixed_ns)]
            pairs = [(l, prefix_lines[i % len(prefix_lines)]) for i, l in enumerate(lines)]
            mixed_results = {n: [] for n in mixed_ns}
        print("\n" + "=" * 108)
        print(f" HEAD-TO-HEAD VARIANT COMPARISON: {args.lang.upper()} ({len(lines):,} lines, mode={args.mode}, jobs={args.jobs})")
        print(f" Variants: {', '.join(var_specs)}")
        print("=" * 108)

        results = []
        baseline_fp, baseline_fn = None, None

        for spec in var_specs:
            parts = spec.lower().split(':')
            acc = parts[-1] != 'noacc'
            if not acc:
                parts.pop()
            if len(parts) == 1:
                k, d, a = args.confirmations, float(parts[0]), args.alpha
            else:
                k, d = int(parts[0].lstrip('k')), float(parts[1])
                a = float(parts[2]) if len(parts) > 2 else args.alpha

            common = dict(data_dir=args.data_dir, en_model_path=args.en_model,
                          he_model_path=args.he_model, mode=args.mode, jobs=args.jobs,
                          req_confirmations=k, model_type=args.model_type,
                          accumulate=acc, alpha=a)
            t0 = time.time()
            fp = run_test('fp', lines, args.lang, d, **common)
            fn = run_test('fn', lines, args.lang, d, **common)
            for n in mixed_ns:
                mixed_results[n].append(
                    (spec, run_test('mixed', pairs, args.lang, d, n_prefix=n, **common)))
            elapsed = time.time() - t0

            mean_l = statistics.mean(fn.latency_values) if fn.latency_values else 0.0
            med_l = statistics.median(fn.latency_values) if fn.latency_values else 0.0

            if baseline_fp is None:
                baseline_fp, baseline_fn = fp.fp_per_1k, fn.fn_per_1k
                d_fp_str, d_fn_str = "—", "—"
            else:
                d_fp = ((fp.fp_per_1k - baseline_fp) / baseline_fp * 100) if baseline_fp else 0.0
                d_fn = ((fn.fn_per_1k - baseline_fn) / baseline_fn * 100) if baseline_fn else 0.0
                d_fp_str = f"{d_fp:+.1f}%"
                d_fn_str = f"{d_fn:+.1f}%"

            results.append((spec, k, d, fp.words_tested, fp.fp_count, fp.fp_per_1k, d_fp_str,
                            fn.words_not_switched, fn.fn_per_1k, d_fn_str, med_l, mean_l, elapsed))

        print(f"\n{'variant':<16} {'K':>3} {'Δ':>5} {'Words':>10} {'FP':>6} {'FP/1k':>8} {'ΔFP%':>8} {'FN':>6} {'FN/1k':>8} {'ΔFN%':>8} {'Med':>6} {'Mean':>7} {'Time':>6}")
        print("-" * 108)
        for r in results:
            print(f"{r[0]:<16} {r[1]:>3} {r[2]:>5.1f} {r[3]:>10,} {r[4]:>6,} {r[5]:>8.3f} {r[6]:>8} {r[7]:>6,} {r[8]:>8.3f} {r[9]:>8} {r[10]:>5.1f}c {r[11]:>6.2f}c {r[12]:>5.1f}s")
        print("-" * 108)

        for n in mixed_ns:
            print(f"\nFORGOT-TO-SWITCH after {n} correct words ({other.upper()} prefix, {args.lang.upper()} suffix)")
            print(f"{'variant':<16} {'Cases':>7} {'PrefixFP':>9} {'SuffixW':>9} {'Uncorr':>7} {'FN/1k':>8} {'OverCorr':>9} {'Med':>6} {'Mean':>7}")
            print("-" * 86)
            for spec, m in mixed_results[n]:
                med = statistics.median(m.latency_values) if m.latency_values else 0.0
                mean = statistics.mean(m.latency_values) if m.latency_values else 0.0
                print(f"{spec:<16} {m.cases:>7,} {m.prefix_fp:>9,} {m.suffix_words:>9,} {m.uncorrected:>7,} "
                      f"{m.fn_per_1k:>8.3f} {m.over_corrected:>9,} {med:>5.1f}c {mean:>6.2f}c")
            print("-" * 86)
        return

    # ── run tests ──
    print(f"\nConfiguration: Δ={args.baseline_delta}, K={args.confirmations}, model={args.model_type}, mode={args.mode}, jobs={args.jobs}")
    if args.model_type == 'neural':
        model_override = args.en_model if args.lang == 'en' else args.he_model
        model_path = model_override if model_override else os.path.join(args.data_dir, f'{args.lang}_char_gru.npz')
    else:
        model_override = args.en_model if args.lang == 'en' else args.he_model
        model_path = model_override if model_override else os.path.join(args.data_dir, f'{args.lang}_quadgrams.marisa')
    corpus_path = args.text_file

    common = dict(data_dir=args.data_dir, en_model_path=args.en_model,
                  he_model_path=args.he_model, mode=args.mode, jobs=args.jobs,
                  req_confirmations=args.confirmations, model_type=args.model_type,
                  alpha=args.alpha)

    report_kwargs = dict(delta=args.baseline_delta, req_confirmations=args.confirmations,
                          mode=args.mode, jobs=args.jobs)

    if args.test in ('fp', 'both'):
        fp = run_test('fp', lines, args.lang, args.baseline_delta, **common)
        print_fp_report(fp, corpus_path, model_path, provenance, **report_kwargs)

    if args.test in ('fn', 'both'):
        fn = run_test('fn', lines, args.lang, args.baseline_delta, **common)
        print_fn_report(fn, corpus_path, model_path, provenance, **report_kwargs)


if __name__ == '__main__':
    main()
