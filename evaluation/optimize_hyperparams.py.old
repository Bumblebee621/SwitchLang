import os
import sys
import random
import itertools
import concurrent.futures
from multiprocessing import cpu_count
from typing import List, Tuple

# Add project root to sys.path so we can import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import EvaluationEngine
from core.sensitivity import SensitivityManager
from core.quadgram import load_models
from core.keymap import EN_TO_HE_FULL, HE_TO_EN_FULL

def en_to_he(c):
    return EN_TO_HE_FULL.get(c, c)

def he_to_en(c):
    return HE_TO_EN_FULL.get(c, c)

def chunk_text(text: str) -> List[Tuple[str, str]]:
    """
    Parse a sentence and segment it by language using Unicode character ranges.
    Returns: List of (language_code, string_chunk)
             e.g. [("he", "התקנתי "), ("en", "python "), ("he", "על המחשב")]
    """
    chunks = []
    current_lang = 'en' # Default starting assumption
    current_chunk = ''
    
    for char in text:
        # Check Unicode block
        if '\u0590' <= char <= '\u05FF':
            target_lang = 'he'
        elif char.isalpha() and char.isascii():
            target_lang = 'en'
        else:
            # Punctuation/Spaces keep the current language context
            target_lang = current_lang
            
        if current_lang != target_lang and current_chunk:
            chunks.append((current_lang, current_chunk))
            current_chunk = ''
            current_lang = target_lang
            
        current_chunk += char
        
    if current_chunk:
        chunks.append((current_lang, current_chunk))
        
    return chunks

def simulate_mixed_typing(chunks: List[Tuple[str, str]], 
                          engine: EvaluationEngine, 
                          sensitivity: SensitivityManager, 
                          trap_index: int = -1):
    """
    Simulates typing a chunked sequence with Context Resumption Events.
    
    Args:
        chunks: The parsed (language, text) tuples.
        engine: The core evaluation engine.
        sensitivity: The SensitivityManager.
        trap_index: If set to a valid index, the simulator will NOT trigger an 
                    Alt+Shift (CRE) before this chunk, creating the mid-sentence trap.
                    Set to -1 for perfect valid typing.
                    
    Returns:
        Tuple of (status: str, total_chars_typed: int, chars_since_trap: int).
        status is one of:
          - 'caught_trap': Engine correctly detected the trap and switched.
          - 'fp_early':    Engine switched BEFORE reaching the trap (false positive).
          - 'no_switch':   Engine never triggered a switch.
    """
    sensitivity.reset()
    buffer_active = ''
    buffer_shadow = ''
    
    total_chars = 0
    chars_since_trap = 0
    in_trap = False
    
    if not chunks:
        return 'no_switch', 0, 0
        
    # Standard: Start OS layout in the language of the first chunk
    os_layout = chunks[0][0]
    
    for i, (intended_lang, chunk_str) in enumerate(chunks):
        
        # Determine if a language context switch occurred between intended words
        if i > 0 and intended_lang != chunks[i-1][0]:
            if i == trap_index:
                # THE TRAP: User intended to switch but OS layout didn't update.
                # No CRE fired.
                in_trap = True
            else:
                # VALID SWITCH: OS layout matches intended.
                # Fire the CRE to reset delta.
                os_layout = intended_lang
                sensitivity.reset(reason='manual_layout_change')
                # Buffers clear on manual layout change
                buffer_active = ''
                buffer_shadow = ''
        
        for c in chunk_str:
            total_chars += 1
            if in_trap:
                chars_since_trap += 1
                
            if c in (' ', '\n', '\t'):
                if buffer_active:
                    switched, _, _, _ = engine.evaluate(
                        buffer_active, buffer_shadow, sensitivity.delta,
                        current_layout=os_layout, on_delimiter=True
                    )
                    if switched:
                        status = 'caught_trap' if in_trap else 'fp_early'
                        return status, total_chars, chars_since_trap
                    sensitivity.on_word_complete()
                buffer_active = ''
                buffer_shadow = ''
            else:
                # Map physical keystrokes based on intended vs OS layout
                if intended_lang == 'en':
                    if os_layout == 'en':
                        buf_act, buf_shd = c, en_to_he(c)
                    else: # OS is he
                        buf_act, buf_shd = en_to_he(c), c
                else: # intended_lang == 'he'
                    if os_layout == 'he':
                        buf_act, buf_shd = c, he_to_en(c)
                    else: # OS is en
                        buf_act, buf_shd = he_to_en(c), c
                        
                buffer_active += buf_act
                buffer_shadow += buf_shd
                
                if len(buffer_active) >= 3:
                    switched, _, _, _ = engine.evaluate(
                        buffer_active, buffer_shadow, sensitivity.delta,
                        current_layout=os_layout, on_delimiter=False
                    )
                    if switched:
                        status = 'caught_trap' if in_trap else 'fp_early'
                        return status, total_chars, chars_since_trap

    return 'no_switch', total_chars, chars_since_trap


def evaluate_mixed_params(en_model, he_model, data_mixed, baseline_delta, alpha, p):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    engine = EvaluationEngine(
        en_model, he_model, 
        collisions_path=os.path.join(data_dir, 'collisions.json'),
        enable_logging=False
    )
    sensitivity = SensitivityManager(baseline_delta=baseline_delta, alpha=alpha, p=p)
    
    fp_count = 0
    tp_count = 0
    latency_sum = 0
    total_traps = 0
    
    # PASS A: Valid Typing (FPR Test)
    # trap_index=-1 means perfect typing with valid CREs — any switch is a FP.
    for text in data_mixed:
        chunks = chunk_text(text)
        if len(chunks) > 1:
            status, _, _ = simulate_mixed_typing(chunks, engine, sensitivity, trap_index=-1)
            if status != 'no_switch':
                fp_count += 1
                
    # PASS B: The Mid-Sentence Trap (Recall/Latency Test)
    # Only count 'caught_trap' as a True Positive. 'fp_early' means the engine
    # switched before the trap was even reached — that's a False Negative for
    # the trap (the engine caught something else, not the intended error).
    for text in data_mixed:
        chunks = chunk_text(text)
        if len(chunks) > 1:
            total_traps += 1
            trap_idx = len(chunks) - 1
            status, _, chars_since_trap = simulate_mixed_typing(chunks, engine, sensitivity, trap_index=trap_idx)
            
            if status == 'caught_trap':
                tp_count += 1
                latency_sum += chars_since_trap

    valid_cases = len([c for c in data_mixed if len(chunk_text(c)) > 1])
    
    fpr = fp_count / valid_cases if valid_cases > 0 else 0
    recall = tp_count / total_traps if total_traps > 0 else 0
    avg_lat = latency_sum / tp_count if tp_count > 0 else 0
    
    return fpr, recall, avg_lat

_en_model = None
_he_model = None

def init_worker(data_dir):
    global _en_model, _he_model
    from core.quadgram import load_models
    _en_model, _he_model = load_models(data_dir)

def worker(args):
    """Worker function for multiprocessing."""
    data_mixed, d, a, p = args
    fpr, recall, lat = evaluate_mixed_params(_en_model, _he_model, data_mixed, d, a, p)
    return {
        'd': d, 'a': a, 'p': p,
        'fpr': fpr, 'recall': recall, 'lat': lat
    }

def main():
    import json
    
    print("Loading models...")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    en_model, he_model = load_models(data_dir)
    
    print("Loading datasets...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mixed_reddit_test_set.json'))
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please run 'python scripts/extract_test_data.py' first to generate the local test set.")
        sys.exit(1)
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data_mixed = json.load(f)
    
    print(f"Loaded {len(data_mixed)} mixed EN/HE Reddit samples from local cache.")
    
    # Extended search ranges
    deltas = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
    ps = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    
    results = []
    
    print(f"{'delta':<6} | {'alpha':<6} | {'p':<4} | {'FPR':<6} | {'Recall':<6} | {'TrapLat'}")
    print("-" * 50)
    
    # Prepare arguments for the worker pool (exclude models)
    tasks = [
        (data_mixed, d, a, p)
        for d, a, p in itertools.product(deltas, alphas, ps)
    ]
    
    num_workers = max(1, cpu_count() - 1)
    print(f"Starting ProcessPoolExecutor with {num_workers} workers to evaluate {len(tasks)} combinations...")
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(data_dir,)
    ) as executor:
        for result in executor.map(worker, tasks):
            results.append(result)
            try:
                print(f"{result['d']:<6.1f} | {result['a']:<6.1f} | {result['p']:<4.1f} | "
                      f"{result['fpr']:<6.2%} | {result['recall']:<6.2%} | {result['lat']:.1f}")
            except UnicodeEncodeError:
                # Fallback for terminals that can't handle percent sign or other chars? 
                # Unlikely for ASCII/HE, but being safe.
                print(f"d={result['d']}, a={result['a']}, p={result['p']} -> FPR: {result['fpr']}, Recall: {result['recall']}")

    best = sorted(results, key=lambda x: (x['fpr'], -x['recall'], x['lat']))
    print("\n--- TOP 10 MIXED DATSET COMBINATIONS ---")
    for r in best[:10]:
         print(f"d={r['d']}, a={r['a']}, p={r['p']} -> FPR: {r['fpr']:.2%}, Recall: {r['recall']:.2%}, Trap Latency: {r['lat']:.1f} chars")

if __name__ == '__main__':
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
