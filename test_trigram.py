from core.trigram import load_models

en, he = load_models('data')

w_en = "new"
w_he = "מק'"

print(f"{w_en} (en model):", en.score(w_en))
print(f"{w_he} (en model):", en.score(w_he))
print(f"{w_en} (he model):", he.score(w_en))
print(f"{w_he} (he model):", he.score(w_he))

print("\n---")
diff = he.score(w_en) - he.score(w_he)  # This is the actual diff (shadow en - active he)
print(f"Diff if typed in HE: {diff}")
