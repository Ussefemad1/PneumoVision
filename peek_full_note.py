import pandas as pd

radiology = pd.read_csv("data/mimic-iv-note/radiology_chest_clean.csv")
discharge = pd.read_csv("data/mimic-iv-note/discharge_clean.csv")

print(f"Radiology total: {len(radiology):,}")
print(f"Discharge total: {len(discharge):,}")

# Sample from beginning, middle, end, and random
def sample_indices(n_total, n_samples=8):
    import random
    random.seed(42)
    indices = [0, 1, n_total // 4, n_total // 2, (3 * n_total) // 4, n_total - 2, n_total - 1]
    indices += random.sample(range(n_total), n_samples)
    return sorted(set(indices))

print("\n=== RADIOLOGY SAMPLES ===")
for i in sample_indices(len(radiology)):
    print(f"\n--- Row {i} ---")
    print(radiology.iloc[i]['text'][:400])
    print("...")

print("\n\n=== DISCHARGE SAMPLES ===")
for i in sample_indices(len(discharge)):
    print(f"\n--- Row {i} ---")
    print(discharge.iloc[i]['text'][:400])
    print("...")