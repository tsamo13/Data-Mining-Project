import pandas as pd
import numpy as np

from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("../../merged_reduced.parquet", engine="pyarrow")

TARGET_COLS = ["Label", "Traffic Type", "Traffic Subtype"]

X = df.drop(columns=TARGET_COLS)
y = df[TARGET_COLS]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

birch = Birch(threshold=0.5, branching_factor=50, n_clusters=None)
birch.fit(X_scaled) # after this command, each row has been assigned to a cluster


labels = birch.labels_ # retrieve the cluster label for each original row

df["cluster"] = labels  # we add the column cluster to the df so we can group by the cluster labels later

numeric_cols = X.columns.tolist()
centroids = df.groupby("cluster")[numeric_cols].mean()  # for each cluster take all rows inside it and compute the mean value per column => compute representatives

label_repr = df.groupby("cluster")[TARGET_COLS].agg(lambda s: s.mode().iloc[0])  # for each cluster, we look at all the label values inside it and pick the most frequent one (mode function does that and iloc[0] returns a scalar value and not a series) (same for traffic type, traffic subtype)

birch_representatives = centroids.join(label_repr)

cluster_sizes = df.groupby("cluster").size().rename("cluster_size")  # counts how many original rows each centroid represents
birch_representatives = birch_representatives.join(cluster_sizes)

birch_representatives.to_parquet("../../data/birch_representatives.parquet", engine="pyarrow", compression="zstd", index=True)

print("Original rows:", len(df))
print("Clusters / representatives:", len(birch_representatives))