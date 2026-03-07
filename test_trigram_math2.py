import json
import math

with open('data/he_trigrams.json', 'r', encoding='utf-8') as f:
    he = json.load(f)
with open('data/en_trigrams.json', 'r', encoding='utf-8') as f:
    en = json.load(f)

# Compute total bigrams
he_total_bigrams = sum(he['bigram_counts'].values())
en_total_bigrams = sum(en['bigram_counts'].values())

def score_new(word, model, total_bigrams):
    text = word.lower()
    if len(text) < 2: return 0.0

    v = model['vocab_size']
    bi_dict = model['bigram_counts']
    tri_dict = model['trigram_counts']

    # Initial bigram
    first_bigram = text[:2]
    bi_c = bi_dict.get(first_bigram, 0)
    score = math.log((bi_c + 1) / (total_bigrams + v))

    # Trigram increments
    for i in range(2, len(text)):
        prev2 = text[i-2:i]
        char = text[i]
        tri = prev2 + char

        c_tri = tri_dict.get(tri, 0)
        c_bi = bi_dict.get(prev2, 0)
        
        score += math.log((c_tri + 1) / (c_bi + v))

    return score

w_en = "new"
w_he = "מק'"

print("\n--- EN MODEL ---")
print(f"new -> EN: {score_new(w_en, en, en_total_bigrams):.3f}")
print(f"מק' -> EN: {score_new(w_he, en, en_total_bigrams):.3f}")

print("\n--- HE MODEL ---")
print(f"new -> HE: {score_new(w_en, he, he_total_bigrams):.3f}")
print(f"מק' -> HE: {score_new(w_he, he, he_total_bigrams):.3f}")

print("\n--- ACTUAL DECISION (Typed in HE) ---")
diff = score_new(w_en, en, en_total_bigrams) - score_new(w_he, he, he_total_bigrams)
print(f"Diff: {diff:.3f} (Threshold is ~3.0)")

