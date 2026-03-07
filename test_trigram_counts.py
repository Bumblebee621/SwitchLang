import json

with open('data/he_trigrams.json', 'r', encoding='utf-8') as f:
    he = json.load(f)

print("HE bigram 'מק':", he['bigram_counts'].get('מק', 0))
print("HE trigram 'מק'':", he['trigram_counts'].get("מק'", 0))
print("HE char ''':", sum(1 for k in he['trigram_counts'] if "'" in k))

print("\nEN bigram 'ne':", he['bigram_counts'].get('ne', 0))
print("EN trigram 'new':", he['trigram_counts'].get("new", 0))

with open('data/en_trigrams.json', 'r', encoding='utf-8') as f:
    en = json.load(f)

print("\n--- EN MODEL ---")
print("EN bigram 'ne':", en['bigram_counts'].get('ne', 0))
print("EN trigram 'new':", en['trigram_counts'].get("new", 0))

print("\nHE bigram 'מק':", en['bigram_counts'].get('מק', 0))
print("HE trigram 'מק'':", en['trigram_counts'].get("מק'", 0))

print("\nVocab Sizes:")
print("HE:", he['vocab_size'])
print("EN:", en['vocab_size'])
