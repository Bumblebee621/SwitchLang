from core.trigram import load_models
import sys

en, he = load_models('data')

with open("test_trigram_seq_output.txt", "w", encoding="utf-8") as f:
    def log(msg):
        f.write(msg + "\n")

    def analyze(word):
        log(f"\n--- Analyzing '{word}' ---")
        
        score_en = en.score(word)
        score_he = he.score(word)
        
        log(f"Total EN score: {score_en:.3f}")
        log(f"Total HE score: {score_he:.3f}")
        
        log("EN Trigrams:")
        for i in range(len(word)-2):
            tri = word[i:i+3]
            bi = word[i:i+2]
            c_tri = en.trigram_counts.get(tri.lower(), 0)
            c_bi = en.bigram_counts.get(bi.lower(), 0)
            s = en.score_incremental(bi, word[i+2])
            log(f"  '{tri}' : tri={c_tri}, bi={c_bi} -> score={s:.3f}")

        log("HE Trigrams:")
        for i in range(len(word)-2):
            tri = word[i:i+3]
            bi = word[i:i+2]
            c_tri = he.trigram_counts.get(tri.lower(), 0)
            c_bi = he.bigram_counts.get(bi.lower(), 0)
            s = he.score_incremental(bi, word[i+2])
            log(f"  '{tri}' : tri={c_tri}, bi={c_bi} -> score={s:.3f}")

    analyze("new")
    analyze("new ")
    analyze("מק'")
    analyze("מק' ")
