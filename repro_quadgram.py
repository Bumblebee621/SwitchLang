import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trigram import load_models

def test_quadgram():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    en_model, he_model = load_models(data_dir)
    
    # Simulate typing "ght" as the start of a word
    # The engine prepends a space to strings to indicate word boundaries
    test_str = " ght"
    
    en_score = en_model.score(test_str)
    
    print(f"Testing sequence: '{test_str}'")
    print(f"English Score: {en_score:.4f}")
    
    # Compare with a valid start
    valid_str = " the"
    valid_score = en_model.score(valid_str)
    print(f"Valid sequence: '{valid_str}'")
    print(f"English Score: {valid_score:.4f}")
    
    if en_score < valid_score:
        print("SUCCESS: 'ght' at start of word is correctly penalized.")
    else:
        print("FAILURE: 'ght' still has a high score.")

if __name__ == "__main__":
    test_quadgram()
