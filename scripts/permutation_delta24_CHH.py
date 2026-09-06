import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    sys.exit("Usage: python3 permutation_delta24_CHH.py INPUT.tsv OUTPUT_PREFIX")

INPUT = sys.argv[1]
PREFIX = sys.argv[2]

N_PERM = 10000
SEED = 12345

df = pd.read_csv(INPUT, sep="\t")

x = df["delta24"].to_numpy(dtype=float)
y = df["deltaCHH"].to_numpy(dtype=float)
chrom = df["chr"].to_numpy()

obs_r = np.corrcoef(x, y)[0, 1]

same_mask = (x != 0) & (y != 0)
obs_conc = np.mean(np.sign(x[same_mask]) == np.sign(y[same_mask]))

rng = np.random.default_rng(SEED)

perm_r = np.empty(N_PERM)
perm_conc = np.empty(N_PERM)

chr_indices = {
    c: np.where(chrom == c)[0]
    for c in np.unique(chrom)
}

for i in range(N_PERM):
    xp = x.copy()

    for c, idx in chr_indices.items():
        xp[idx] = rng.permutation(x[idx])

    perm_r[i] = np.corrcoef(xp, y)[0, 1]

    m = (xp != 0) & (y != 0)
    perm_conc[i] = np.mean(np.sign(xp[m]) == np.sign(y[m]))

p_r_two = (np.sum(np.abs(perm_r) >= abs(obs_r)) + 1) / (N_PERM + 1)

dev_obs = abs(obs_conc - 0.5)
p_conc_two = (
    np.sum(np.abs(perm_conc - 0.5) >= dev_obs) + 1
) / (N_PERM + 1)

print(f"windows = {len(df)}")
print(f"observed Pearson r = {obs_r:.8f}")
print(f"permutation mean r = {perm_r.mean():.8f}")
print(f"permutation SD r = {perm_r.std(ddof=1):.8f}")
print(f"two-sided permutation P for r = {p_r_two:.6f}")
print()
print(f"observed directional concordance = {obs_conc:.8f}")
print(f"permutation mean concordance = {perm_conc.mean():.8f}")
print(f"permutation SD concordance = {perm_conc.std(ddof=1):.8f}")
print(f"two-sided permutation P for concordance = {p_conc_two:.6f}")

np.savetxt(PREFIX + "_r.txt", perm_r, fmt="%.10f")
np.savetxt(PREFIX + "_concordance.txt", perm_conc, fmt="%.10f")
