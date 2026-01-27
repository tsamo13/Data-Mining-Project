import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("../../merged_reduced.parquet", engine="pyarrow")

TARGET_COLS = ["Label", "Traffic Type", "Traffic Subtype"]

X = df.drop(columns=TARGET_COLS)
y = df[TARGET_COLS]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 10000

kmeans = MiniBatchKMeans(
    n_clusters=k,
    batch_size=50000,
    random_state=42,
    max_iter=500,
    reassignment_ratio=0.01
)

labels = kmeans.fit_predict(X_scaled)

df["cluster"] = labels

numeric_cols = X.columns.tolist()
centroids = df.groupby("cluster")[numeric_cols].mean()
target_repr = df.groupby("cluster")[TARGET_COLS].agg(lambda s: s.mode().iloc[0])
cluster_sizes = df.groupby("cluster").size().rename("cluster_size")

reps = centroids.join(target_repr).join(cluster_sizes)

reps.to_parquet("../../data/minibatch_kmeans_representatives.parquet", engine="pyarrow", compression="zstd",index=True)

print("Original rows:", len(df))
print("Clusters / representatives:", len(reps))