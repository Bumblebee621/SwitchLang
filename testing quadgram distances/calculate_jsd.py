"""
Utility script for calculating the Jensen-Shannon Divergence (JSD) between 
two trained n-gram models. Used to mathematically measure the similarity/distance 
between corpora or languages.
"""
import sys
import json
import math

def compute_jsd_from_dicts(dict_a: dict, dict_b: dict) -> float:
    """
    Computes the Jensen-Shannon Divergence between two frequency dictionaries.
    
    Normalizes the raw count frequencies into probability distributions (P and Q)
    and then calculates the symmetric JSD based on the Kullback-Leibler divergence.
    
    Args:
        dict_a (dict): The first frequency distribution (e.g. n-gram counts).
        dict_b (dict): The second frequency distribution.
        
    Returns:
        float: The calculated Jensen-Shannon Divergence (>= 0.0), where 0.0 implies 
               the distributions are identical.
    """
    # Calculate total frequencies for normalization to probabilities
    sum_a = sum(dict_a.values())
    sum_b = sum(dict_b.values())

    if sum_a == 0 or sum_b == 0:
        raise ValueError("One or both frequency dictionaries cannot be completely empty.")

    div_p = 0.0
    div_q = 0.0

    # Pass 1: Iterate over keys in dictionary A
    for x, count_a in dict_a.items():
        p_x = count_a / sum_a
        count_b = dict_b.get(x, 0)
        q_x = count_b / sum_b if count_b > 0 else 0.0
        
        m_x = 0.5 * p_x + 0.5 * q_x
        
        if p_x > 0:
            div_p += p_x * math.log2(p_x / m_x)
            
        if q_x > 0:
            div_q += q_x * math.log2(q_x / m_x)

    # Pass 2: Iterate over keys in dictionary B that were NOT in dictionary A
    for x, count_b in dict_b.items():
        if x not in dict_a:
            q_x = count_b / sum_b
            m_x = 0.5 * q_x
            
            if q_x > 0:
                div_q += q_x * math.log2(q_x / m_x)
                
    jsd = 0.5 * div_p + 0.5 * div_q
    
    return max(0.0, jsd)

def calculate_jsds(file_a: str, file_b: str) -> dict:
    """
    Calculates JSD metrics for all n-gram sizes found within two JSON model files.
    
    Parses the JSON model data, extracts bigram, trigram, and quadgram dictionaries
    if available, and computes their individual JSD scores.
    
    Args:
        file_a (str): Path to the first JSON model file.
        file_b (str): Path to the second JSON model file.
        
    Returns:
        dict: A dictionary mapping n-gram sizes (e.g., 'bigram', 'all') to their JSD float value.
    """
    # Load dictionaries into memory
    with open(file_a, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    
    with open(file_b, 'r', encoding='utf-8') as f:
        data_b = json.load(f)

    results = {}
    
    ngram_keys = ["bigram_counts", "trigram_counts", "quadgram_counts"]
    
    # Check if files contain nested dictionaries for each n-gram size
    for key in ngram_keys:
        dict_a = data_a.get(key)
        dict_b = data_b.get(key)
        
        if dict_a and dict_b:
            label = key.replace("_counts", "")
            try:
                results[label] = compute_jsd_from_dicts(dict_a, dict_b)
            except ValueError:
                pass
                
    if not results:
        # Fallback for plain counts mapping directly in JSON or backward compatibility
        dict_a = data_a.get("quadgram_counts", data_a)
        dict_b = data_b.get("quadgram_counts", data_b)
        try:
            results["all"] = compute_jsd_from_dicts(dict_a, dict_b)
        except ValueError:
            pass

    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_jsd.py <file_a_path> <file_b_path>")
        sys.exit(1)
        
    file_a_path = sys.argv[1]
    file_b_path = sys.argv[2]
    
    try:
        results = calculate_jsds(file_a_path, file_b_path)
        for label, val in results.items():
            print(f"{label} JSD: {val}")
    except Exception as e:
        print(f"Error calculating JSD: {e}", file=sys.stderr)
        sys.exit(1)
