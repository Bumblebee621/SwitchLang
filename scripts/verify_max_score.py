# Add project root to path
import os
import sys
PROJECT_ROOT = r"C:\Users\Ariel\Documents\Antigravity\SwitchLang"
sys.path.append(PROJECT_ROOT)

from core.quadgram import QuadgramModel, load_models
from core.engine import EvaluationEngine

def test_3_modes():
    data_dir = r"C:\Users\Ariel\Documents\Antigravity\SwitchLang\data"
    
    print("Loading models...")
    models = load_models(data_dir, load_so=True)
    en_model = models['en']
    so_model = models.get('so')
    he_model = models['he']
    
    engine = EvaluationEngine(
        en_model, he_model, None, 
        enable_logging=False, 
        en_so_model=so_model
    )

    test_str = "stdout"
    eval_str = " " + test_str
    
    score_std = en_model.score(eval_str)
    score_so = so_model.score(eval_str) if so_model else score_std
    score_max = max(score_std, score_so)

    print(f"\nTest string: '{test_str}'")
    print(f"Standard EN score: {score_std:.4f}")
    print(f"Stack Overflow score: {score_so:.4f}")

    # Mode 1: Always Standard
    res_std = engine.evaluate(eval_str, eval_str, 2.0, mode='standard')[1]
    print(f"Mode=Standard: Result={res_std:.4f} (Expected {0:.4f} if same as standard)")
    # (res_std will be score_shadow - score_active. In our script he_model.score - en_model.score)
    
    # Mode 2: Smart (simulated by HookManager logic)
    # Case A: In Editor
    is_editor = True
    eff_mode = 'technical' if is_editor else 'standard'
    score_active_editor = engine.evaluate(eval_str, eval_str, 2.0, mode=eff_mode, current_layout='en')[0] # just checking logic
    
    # Let's check internal scoring directly
    res_tech = engine._score_text_en(eval_str, mode='technical')
    res_smart_editor = engine._score_text_en(eval_str, mode='technical') # Smart in editor
    res_smart_other = engine._score_text_en(eval_str, mode='standard') # Smart in browser
    
    print(f"Smart (In Editor) score: {res_smart_editor:.4f}")
    print(f"Smart (In Browser) score: {res_smart_other:.4f}")
    
    assert res_smart_editor == score_max, "Smart Mode (Editor) logic failure!"
    assert res_smart_other == score_std, "Smart Mode (Browser) logic failure!"
    print("SUCCESS: 3-mode logic verified.")

if __name__ == "__main__":
    try:
        test_3_modes()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Verification FAILED: {e}")
        sys.exit(1)
