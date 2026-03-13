"""
extract_test_data.py — One-off ETL script for fetching mixed EN/HE sentences.

Connects to Hugging Face, streams Pushshift Reddit comments, filters for
sentences containing both English and Hebrew characters, and saves exactly
1,000 samples to data/mixed_reddit_test_set.json.

This decoupling prevents the hyperparameter optimization script from relying
on a slow, unreliable internet stream.
"""

import json
import os
import sys
from dotenv import load_dotenv

# Load variables from .env if it exists
load_dotenv()

def load_pushshift_reddit_comments(num_samples: int = 100):
    """
    Loads samples from HuggingFace dataset 'fddemarco/pushshift-reddit-comments'.
    NOTE: Requires `datasets` library: pip install datasets
    Filters for relevant Israeli subreddits or just pulls raw and relies on 
    the chunker to find mixed comments.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please 'pip install datasets' to load Pushshift Reddit dumps.")
        sys.exit(1)

    print("Streaming fddemarco/pushshift-reddit-comments... (This may take a moment)")
    
    # Use an environment variable for the secret token to avoid git push blocks
    hf_token = os.environ.get("HF_TOKEN")
    
    dataset = load_dataset(
        "fddemarco/pushshift-reddit-comments", 
        split='train', 
        streaming=True,
        token=hf_token
    )
    
    samples = []
    # Identify subreddits likely to have mixed EN/HE
    target_subs = {'ani_bm', 'Israel', 'Judaism', 'Hebrew', 'TelAviv'}
    
    iterator = iter(dataset)
    while len(samples) < num_samples:
        try:
            row = next(iterator)
        except StopIteration:
            print("Dataset ended unexpectedly.")
            break
        except Exception as e:
            print(f"Error during streaming: {e}. Attempting to continue...")
            # Re-initialize iterator if it completely fails, or just break and save what we have
            try:
                iterator = iter(dataset)
                continue
            except:
                break

        body = row.get('body', "")
        sub = row.get('subreddit', "")
        
        # Skip empty or deleted comments
        if not body or body == '[deleted]' or body == '[removed]':
            continue
            
        # Filter by target subreddits or simply look for hebrew characters
        # Optimization: Just check if there's any hebrew char in the body
        has_hebrew = any('\u0590' <= c <= '\u05FF' for c in body)
        has_english = any('\u0020' <= c <= '\u007F' and c.isalpha() for c in body)
        
        # We want mixed sentences for our trap
        if has_hebrew and has_english and len(body) > 20:
            # Clean newlines for single-line testing
            body = body.replace('\n', ' ').strip()
            # Avoid duplicates if resuming
            if body not in samples:
                samples.append(body)
                if len(samples) % 100 == 0:
                    print(f"Gathered {len(samples)}/{num_samples} mixed samples...")
            
    return samples

def main():
    # Ensure data directory exists
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    
    out_path = os.path.join(data_dir, 'mixed_reddit_test_set.json')
    
    existing_samples = []
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                existing_samples = json.load(f)
            print(f"Found {len(existing_samples)} existing samples.")
        except:
            pass

    if len(existing_samples) >= 1000:
        print(f"File {out_path} already has 1,000+ samples.")
        return

    print(f"Starting extraction of 1,000 mixed EN/HE samples (need {1000 - len(existing_samples)} more)...")
    new_samples = load_pushshift_reddit_comments(num_samples=1000)
    
    # Combine and save (using set logic in load_pushshift_reddit_comments handles dedup if needed, 
    # but let's be safe here too)
    all_samples = list(dict.fromkeys(existing_samples + new_samples))
    all_samples = all_samples[:1000]
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccess! Total {len(all_samples)} samples in {out_path}")

if __name__ == '__main__':
    main()
