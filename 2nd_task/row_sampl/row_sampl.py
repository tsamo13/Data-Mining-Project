from pathlib import Path
import pandas as pd
import numpy as np
import math

IN_FILE  = Path("../../merged_reduced.parquet")
OUT_FILE = Path("../../data/stratified.parquet")

FRAC = 0.20  #sampling fraction => the sampled dataset will contain about 20% of the total rows
SEED = 42    # ensures the sampling is reproducible => running the script again yields the same sampled rows when i have the same seed

df = pd.read_parquet(IN_FILE, engine="pyarrow")  # we use pyarrow for reading parquet

# 1) we compute counts per stratum
sizes = df.groupby(["Label", "Traffic Type"], observed=True).size() # a pandas Series where index=(Label,Traffic Type) pairs and value = count of rows in that pair

# 2) target total rows
target_total = int(round(FRAC * len(df))) # compute the number of rows you want in the final sample (the 20% of the dataset => 1731353 rows)

# 3) exact quotas via largest remainder
exact = sizes / sizes.sum() * target_total # computes the ideal (fractional) number of sampled rows per stratum (how many rows must have each stratum (combination) to the new dataset)
base = np.floor(exact).astype(int) # takes the floor (round down) of each fractional quota (the number of rows for each stratum)
remainder = (exact - base).sort_values(ascending=False) # computes the leftover fractional parts and sorts stratum by largest remainder first

need = target_total - base.sum()  # computes how many rows are missing after flooring
base.loc[remainder.index[:need]] += 1 # distributes the need remaining rows by selecting the strata with the largest fractional remainders(the first need rows) and increments their quotas by 1

# 4) sample exactly per stratum
rng = np.random.default_rng(SEED)  # default_rng is a numpy's modern random number generator. this line creates a random generator that behaves randomly but in a reproducible way
parts = []
for (label, ttype), n in base.items():  # loop over each stratum quota
    g = df[(df["Label"] == label) & (df["Traffic Type"] == ttype)]  # the stratum's full data
    if n > 0:
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)  # randomly selects exactly n row indices from the stratum. replace = False means no duplicates
        parts.append(df.loc[idx])  # retrieves the sampled rows from df and appends them to parts

sampled = pd.concat(parts, ignore_index=True) # concactnates all the sampled per-stratum dfs into one df. At this moment, len(sampled) == target_total

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
sampled.to_parquet(OUT_FILE, engine="pyarrow", compression="zstd", index=False)

print("Original rows:", len(df))
print("Sampled rows: ", len(sampled))