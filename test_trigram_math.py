import math

en_v = 135
he_v = 120

def score(tri_c, bi_c, v):
    return math.log((tri_c + 1) / (bi_c + v))

print("--- HE MODEL ---")
print("מק' -> HE:", score(242, 22227, he_v))
print("new -> HE:", score(19, 402, he_v))

print("\n--- EN MODEL ---")
print("new -> EN:", score(7158, 129394, en_v))
print("מק' -> EN:", score(0, 0, en_v))

