import os, json
from collections import Counter
import pandas as pd
import numpy as np
import sys

IN = "../data.csv"
OUT_DIR = "data/chunks"
CHUNK = 4500000

SUMMARY_PATH = os.path.join(OUT_DIR, "_summary.json")


if os.path.isfile(SUMMARY_PATH):
    print('Parquet files existed')
    with open(SUMMARY_PATH, "r") as f:
        print(f.read())
    sys.exit(0)

os.makedirs(OUT_DIR, exist_ok=True)


# Counters for quick sanity
counts_label = Counter()
counts_type = Counter()
counts_subtype = Counter()

def downcast(df: pd.DataFrame) -> pd.DataFrame:  # takes a df and returns a df. Tries to reduce ram by converting large dtypes to smaller representations
    # mutate in-place to save RAM
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    # handle pandas nullable ints 
    for c in df.select_dtypes(include=["Int64"]).columns:  # integers that can represent also missing values 
        try:
            df[c] = df[c].astype("Int32")  # keeps NA support, smaller footprint
        except Exception:
            pass
    for c in df.select_dtypes(include=["float64","Float64"]).columns:
        df[c] = df[c].astype("float32")
        # Convert object columns to pandas Category 
    for c in df.select_dtypes(include=["object"]).columns:  # select object columns (usually strings) and convert them to categorical dtype. This can reduce dramatically memory when values repeat a lot (common in labels/types)
        df[c] = df[c].astype("category")
    return df

wrote_rows = 0  # total numbers of rows written across all chunk files
num_parts = 0   # number of chunk files created

for i, chunk in enumerate(pd.read_csv(IN, chunksize=CHUNK, low_memory=False)):  # loop over chunks. lowmemory=false means that pandas reads more of the file before deciding dtypes => dtype inference is done more carefully. With low_memory=true, pandas readsthe csv in small internal blocks and infers column dtypes per block => reduces peak ram usage but if different blocks suggest different types, pandas must fall back to object.

    # Update global counts once per chunk
    if "Label" in chunk: counts_label.update(chunk["Label"].dropna())
    if "Traffic Type" in chunk: counts_type.update(chunk["Traffic Type"].dropna())
    if "Traffic Subtype" in chunk: counts_subtype.update(chunk["Traffic Subtype"].dropna())

    chunk = downcast(chunk)
    path = f"{OUT_DIR}/part_{i:02d}.parquet"
    chunk.to_parquet(path, index=False, compression="snappy") # fast compression commonly used with Parquet.
    wrote_rows += len(chunk)
    num_parts += 1
    print("wrote", path, len(chunk), "rows")
 

# Save tiny summary
summary = {
    "rows_written": wrote_rows,
    "format": "parquet",
    "num_parts": num_parts,
    "label_counts": dict(counts_label),
    "traffic_type_counts": dict(counts_type),
    "traffic_subtype_counts": dict(counts_subtype),
}
with open(f"{OUT_DIR}/_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("Summary saved to", f"{OUT_DIR}/_summary.json")
