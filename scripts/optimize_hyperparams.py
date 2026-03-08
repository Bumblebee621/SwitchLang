import os
import sys
import random
import itertools

# Add project root to sys.path so we can import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import EvaluationEngine
from core.sensitivity import SensitivityManager
from core.trigram import load_models
from core.keymap import EN_TO_HE_FULL, HE_TO_EN_FULL

def en_to_he(c):
    return EN_TO_HE_FULL.get(c, c)

def he_to_en(c):
    return HE_TO_EN_FULL.get(c, c)

def simulate_typing(text, intended_lang, actual_layout, engine, sensitivity):
    """
    Simulate typing a string character by character.
    Returns (switched: bool, latency_chars: int)
    """
    sensitivity.reset()
    buffer_active = ''
    buffer_shadow = ''
    
    chars_typed = 0
    
    for c in text:
        chars_typed += 1
        
        if c == ' ':
            if buffer_active:
                switched, _ = engine.evaluate(
                    buffer_active, buffer_shadow, sensitivity.delta,
                    current_layout=actual_layout, on_delimiter=True
                )
                if switched:
                    return True, chars_typed
                sensitivity.on_word_complete()
            buffer_active = ''
            buffer_shadow = ''
        else:
            if intended_lang == 'en':
                if actual_layout == 'en':
                    buf_act = c
                    buf_shd = en_to_he(c)
                else: # layout is he
                    buf_act = en_to_he(c)
                    buf_shd = c
            else: # intended_lang == 'he'
                if actual_layout == 'he':
                    buf_act = c
                    buf_shd = he_to_en(c)
                else: # layout is en
                    buf_act = he_to_en(c)
                    buf_shd = c
                    
            buffer_active += buf_act
            buffer_shadow += buf_shd
            
            if len(buffer_active) >= 3:
                switched, _ = engine.evaluate(
                    buffer_active, buffer_shadow, sensitivity.delta,
                    current_layout=actual_layout, on_delimiter=False
                )
                if switched:
                    return True, chars_typed

    return False, chars_typed

def load_samples(filepath, num_samples, min_len=20, max_len=100):
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if min_len <= len(line) <= max_len and ' ' in line:
                samples.append(line)
    random.shuffle(samples)
    return samples[:num_samples]

def evaluate_params(en_model, he_model, data_en, data_he, baseline_delta, alpha, p):
    engine = EvaluationEngine(
        en_model, he_model, 
        collisions_path=os.path.join('data', 'collisions.json')
    )
    sensitivity = SensitivityManager(baseline_delta=baseline_delta, alpha=alpha, p=p)
    
    # 1. Valid English typing (Intended EN, Layout EN) -> SHOULD NOT SWITCH
    fp_en = 0
    for text in data_en:
        switched, _ = simulate_typing(text, 'en', 'en', engine, sensitivity)
        if switched: fp_en += 1
            
    # 2. Valid Hebrew typing (Intended HE, Layout HE) -> SHOULD NOT SWITCH
    fp_he = 0
    for text in data_he:
        switched, _ = simulate_typing(text, 'he', 'he', engine, sensitivity)
        if switched: fp_he += 1
            
    fpr = (fp_en + fp_he) / (len(data_en) + len(data_he))
    
    # 3. Mistyped English (Intended EN, Layout HE) -> SHOULD SWITCH QUICKLY
    tp_en = 0
    latency_en = []
    for text in data_en:
        switched, latency = simulate_typing(text, 'en', 'he', engine, sensitivity)
        if switched:
            tp_en += 1
            latency_en.append(latency)
            
    # 4. Mistyped Hebrew (Intended HE, Layout EN) -> SHOULD SWITCH QUICKLY
    tp_he = 0
    latency_he = []
    for text in data_he:
        switched, latency = simulate_typing(text, 'he', 'en', engine, sensitivity)
        if switched:
            tp_he += 1
            latency_he.append(latency)
            
    recall = (tp_en + tp_he) / (len(data_en) + len(data_he))
    avg_latency = 0
    if latency_en or latency_he:
        avg_latency = sum(latency_en + latency_he) / len(latency_en + latency_he)
        
    return fpr, recall, avg_latency

def main():
    print("Loading models...")
    en_model, he_model = load_models('data')
    
    print("Loading datasets...")
    # Sample 200 sentences from each corpus for fast feedback
    num_samples = 200
    data_en = load_samples('data/en_corpus.txt', num_samples)
    data_he = load_samples('data/he_corpus.txt', num_samples)
    
    print(f"Loaded {len(data_en)} EN samples and {len(data_he)} HE samples.")
    
    # Grid search space
    # delta = baseline_delta + alpha * (word_count ** p)
    deltas = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    alphas = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    ps = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    results = []
    
    print(f"{'delta':<6} | {'alpha':<6} | {'p':<4} | {'FPR':<6} | {'Recall':<6} | {'AvgLat'}")
    print("-" * 50)
    
    for d, a, p in itertools.product(deltas, alphas, ps):
        fpr, recall, lat = evaluate_params(en_model, he_model, data_en, data_he, d, a, p)
        results.append({
            'd': d, 'a': a, 'p': p,
            'fpr': fpr, 'recall': recall, 'lat': lat
        })
        print(f"{d:<6.1f} | {a:<6.1f} | {p:<4.1f} | {fpr:<6.2%} | {recall:<6.2%} | {lat:.1f}")

    # Top combination minimizes FPR, then maximizes Recall, then minimizes latency
    best = sorted(results, key=lambda x: (x['fpr'], -x['recall'], x['lat']))
    print("\n--- TOP 5 COMBINATIONS ---")
    for r in best[:5]:
         print(f"d={r['d']}, a={r['a']}, p={r['p']} -> FPR: {r['fpr']:.2%}, Recall: {r['recall']:.2%}, Avg Latency: {r['lat']:.1f} chars")

if __name__ == '__main__':
    main()
