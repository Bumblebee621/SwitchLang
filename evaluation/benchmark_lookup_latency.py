"""
benchmark_lookup_latency.py — Rigorous latency evaluation of Python Dict vs RecordTrie.

Measures:
1. Pure lookup latency (ns/op) for hit keys (keys existing in model)
2. Pure lookup latency (ns/op) for miss keys (unseen ngrams)
3. Mixed lookup latency (realistic ~85% hit rate)
4. Keystroke latency: score_incremental() end-to-end (ns/op)
5. Word latency: score() end-to-end (µs/word)
Subtracts loop overhead for true hardware/Python instruction measurement.
"""

import os
import sys
import time
import random
import json
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quadgram import QuadgramModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def measure_loop_overhead(n_iters, keys):
    # Baseline loop overhead with no operation inside
    t0 = time.perf_counter_ns()
    dummy = None
    for k in keys:
        dummy = k
    t1 = time.perf_counter_ns()
    return (t1 - t0) / n_iters


def run_trials(func, trials=7):
    times = []
    for _ in range(trials):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    # Return median and min
    return statistics.median(times), min(times), statistics.stdev(times) if len(times) > 1 else 0


def main():
    print("=" * 70)
    print(" RIGOROUS LATENCY BENCHMARK: PYTHON DICT vs RECORDTRIE")
    print("=" * 70)

    json_path = os.path.join(DATA_DIR, 'en_quadgrams.json')
    marisa_path = os.path.join(DATA_DIR, 'en_quadgrams.marisa')

    print("Loading models...")
    with open(json_path, 'r') as f:
        raw_json = json.load(f)

    dict_quads = raw_json['quadgram_counts']
    dict_tris = raw_json['trigram_counts']

    # Load via QuadgramModel
    trie_model = QuadgramModel(marisa_path)
    trie = trie_model._trie

    # Also build a dict-based QuadgramModel for end-to-end comparison
    class DictQuadgramModel(QuadgramModel):
        def __init__(self, data):
            self.quadgram_counts = data.get('quadgram_counts', {})
            self.trigram_counts = data.get('trigram_counts', {})
            self.bigram_counts = data.get('bigram_counts', {})
            self.vocab_size = data.get('vocab_size', 30)
            self.total_bigrams = sum(self.bigram_counts.values())
            self._bigram_first_totals = {}
            for k, c in self.bigram_counts.items():
                if k:
                    self._bigram_first_totals[k[0]] = self._bigram_first_totals.get(k[0], 0) + c
            self._is_trie = False

    dict_model = DictQuadgramModel(raw_json)

    # Prepare datasets of keys:
    # 1. 100% Hits: sample 100,000 real quadgram keys
    all_quad_keys = list(dict_quads.keys())
    random.seed(42)
    sample_hits = random.sample(all_quad_keys, 100000)

    # 2. 100% Misses: generate 100,000 random strings that don't exist
    sample_misses = []
    while len(sample_misses) < 100000:
        candidate = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789!@#", k=4))
        if candidate not in dict_quads:
            sample_misses.append(candidate)

    # 3. Realistic Mixed: 85% hits, 15% misses
    sample_mixed = sample_hits[:85000] + sample_misses[:15000]
    random.shuffle(sample_mixed)

    n = len(sample_hits)

    # Measure loop overhead
    loop_ns_per_op = measure_loop_overhead(n, sample_hits)
    print(f"Loop overhead baseline: {loop_ns_per_op:.2f} ns/iteration\n")

    def bench_suite(name, keys):
        print(f"--- Scenario: {name} ({len(keys):,} lookups) ---")

        # Bench Dict
        def test_dict():
            dummy = 0
            for k in keys:
                res = dict_quads.get(k, 0)
                dummy += res
            return dummy

        med_d, min_d, std_d = run_trials(test_dict)
        dict_raw_ns = med_d / n
        dict_net_ns = dict_raw_ns - loop_ns_per_op

        # Bench Trie
        def test_trie():
            dummy = 0
            for k in keys:
                res = trie.get(k)
                val = res[0][0] if res else 0
                dummy += val
            return dummy

        med_t, min_t, std_t = run_trials(test_trie)
        trie_raw_ns = med_t / n
        trie_net_ns = trie_raw_ns - loop_ns_per_op

        ratio = trie_net_ns / dict_net_ns if dict_net_ns > 0 else trie_raw_ns / dict_raw_ns
        print(f"  Python dict : {dict_net_ns:6.1f} ns/op (raw: {dict_raw_ns:6.1f} ns, min: {min_d/n - loop_ns_per_op:6.1f} ns)")
        print(f"  RecordTrie  : {trie_net_ns:6.1f} ns/op (raw: {trie_raw_ns:6.1f} ns, min: {min_t/n - loop_ns_per_op:6.1f} ns)")
        print(f"  Speed Ratio : RecordTrie is {ratio:4.1f}× dict latency (+{trie_net_ns - dict_net_ns:.1f} ns difference)")
        print()

    bench_suite("100% Hit Rate (all keys present)", sample_hits)
    bench_suite("100% Miss Rate (all keys missing)", sample_misses)
    bench_suite("Realistic Mixed (85% Hit / 15% Miss)", sample_mixed)

    # 4. End-to-End Keystroke Evaluation (score_incremental)
    print("--- End-to-End Per-Keystroke Latency (score_incremental) ---")
    keystrokes = [
        ("the", " "), ("ing", " "), ("tion", "s"), ("and", " "),
        ("int", "e"), ("com", "p"), ("abc", "d"), ("xyz", "q")
    ] * 25000  # 200,000 keystrokes
    n_keys = len(keystrokes)

    def test_inc_dict():
        s = 0.0
        for p, c in keystrokes:
            s += dict_model.score_incremental(p, c)
        return s

    def test_inc_trie():
        s = 0.0
        for p, c in keystrokes:
            s += trie_model.score_incremental(p, c)
        return s

    med_kd, min_kd, _ = run_trials(test_inc_dict)
    med_kt, min_kt, _ = run_trials(test_inc_trie)

    kd_ns = med_kd / n_keys
    kt_ns = med_kt / n_keys

    print(f"  Dict score_incremental : {kd_ns:6.1f} ns / keystroke ({kd_ns/1000:6.3f} µs)")
    print(f"  Trie score_incremental : {kt_ns:6.1f} ns / keystroke ({kt_ns/1000:6.3f} µs)")
    print(f"  Absolute Difference    : {abs(kt_ns - kd_ns)/1000:6.3f} µs per keystroke ({abs(kt_ns - kd_ns):.0f} ns)")
    print(f"  Typing Impact at 100WPM: {(abs(kt_ns - kd_ns)/1e9) / 0.100 * 100:.5f}% of human keystroke budget\n")

    # 5. Full Word Scoring
    print("--- End-to-End Word Scoring (score() on realistic sentences) ---")
    sentences = [
        "the international conference on software engineering",
        "quick brown fox jumps over the lazy dog",
        "real time keyboard layout auto switcher for linux and windows",
        "def evaluate_quadgram_probabilities(text, model):",
    ] * 5000  # 20,000 sentences
    n_sents = len(sentences)

    def test_score_dict():
        s = 0.0
        for sent in sentences:
            s += dict_model.score(sent)
        return s

    def test_score_trie():
        s = 0.0
        for sent in sentences:
            s += trie_model.score(sent)
        return s

    med_sd, min_sd, _ = run_trials(test_score_dict)
    med_st, min_st, _ = run_trials(test_score_trie)

    sd_us = (med_sd / n_sents) / 1000
    st_us = (med_st / n_sents) / 1000

    print(f"  Dict score()           : {sd_us:6.2f} µs / sentence (~50 chars)")
    print(f"  Trie score()           : {st_us:6.2f} µs / sentence (~50 chars)")
    print(f"  Sentences per second   : Dict = {1e6/sd_us:,.0f}/sec | Trie = {1e6/st_us:,.0f}/sec")
    print("=" * 70)


if __name__ == '__main__':
    main()
