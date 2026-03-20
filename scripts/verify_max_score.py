# Add project root to path
import os
import sys
PROJECT_ROOT = r"C:\Users\Ariel\Documents\Antigravity\SwitchLang"
sys.path.append(PROJECT_ROOT)

from core.quadgram import QuadgramModel, load_models
from core.engine import EvaluationEngine

def test_max_score():
    data_dir = r"C:\Users\Ariel\Documents\Antigravity\SwitchLang\data"
    
    print("Loading models...")
    models = load_models(data_dir, load_so=True)
    
    en_model = models['en']
    so_model = models.get('so')
    he_model = models['he']
    
    if not so_model:
        print("FAILED: so_quadgrams.json not found or not loaded.")
        return

    engine = EvaluationEngine(
        en_model, he_model, None, 
        enable_logging=False, 
        en_so_model=so_model,
        model_mode='technical'
    )

    # Test case 1: A common technical term 'stdout'
    # In OpenSubtitles it might be rare, in SO it should be common.
    test_str = "stdout"
    eval_str = " " + test_str
    
    score_std = en_model.score(eval_str)
    score_so = so_model.score(eval_str)
    
    print(f"\nTest string: '{test_str}'")
    print(f"Standard EN score: {score_std:.4f}")
    print(f"Stack Overflow score: {score_so:.4f}")
    
    # Engine evaluation in technical mode
    engine.set_model_mode('technical')
    score_tech = engine._score_text_en(eval_str)
    print(f"Engine (Technical) score: {score_tech:.4f}")
    
    assert score_tech == max(score_std, score_so), "Max score logic failure!"
    print("SUCCESS: Max score logic verified.")

    # Test case 2: Standard mode
    engine.set_model_mode('standard')
    score_standard = engine._score_text_en(eval_str)
    print(f"Engine (Standard) score: {score_standard:.4f}")
    
    assert score_standard == score_std, "Standard mode logic failure!"
    print("SUCCESS: Standard mode logic verified.")

if __name__ == "__main__":
    try:
        test_max_score()
    except Exception as e:
        print(f"Verification FAILED: {e}")
        sys.exit(1)
