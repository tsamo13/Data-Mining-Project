# Data Mining Project - Network Traffic Analysis

This repository contains the implementation and analysis of a Data Mining project focused on network traffic data, including exploratory data analysis, dataset reduction techniques, and supervised machine learning models for classification.

The project examines how different data reduction strategies affect the performance of classifiers on both binary and multi-class prediction tasks.

## 📊 Dataset

The original dataset is a large-scale network traffic dataset that cannot be loaded entirely into memory.
To handle this, the data is processed in chunks and stored in Parquet format for efficiency.

Key target variables:

Label: Binary classification (Benign / Malicious)

Traffic Type: Multi-class traffic categorization

### 🧪 Task 1 – Exploratory Data Analysis (EDA)

Implemented in **1st_task/**.

Main objectives:

- Efficient loading of large CSV files using chunking

- Feature inspection and statistical summaries

- Global aggregation of statistics (min, max, mean, std)

- Initial exploration of relationships between features and targets

- Parquet files are used to significantly reduce I/O time and memory usage.

### 🧩 Task 2 – Dataset Reduction

Implemented in **2nd_task/**.

Three different dataset reduction strategies are examined:

1️⃣ Stratified Sampling

- Preserves the joint distribution of Label and Traffic Type

- Produces a representative subset of the original dataset

- Output: stratified.parquet

2️⃣ BIRCH Clustering

Hierarchical clustering suitable for large datasets

- Produces representative centroids

- Preserves structure better than random reduction

- Output: birch_representatives.parquet

3️⃣ MiniBatch K-Means

- Highly scalable clustering algorithm

- Generates a fixed number of centroids

- Fast but may lose minority class information

- Output: minibatch_kmeans_representatives.parquet

### Task 3 – Classification Models

Implemented in **main.ipynb**.

All experiments and evaluations are conducted in main.ipynb.

Two classifiers are used:

🔹 Multi-Layer Perceptron (MLP)

- Non-linear model

- Performs well on complex patterns

- Sensitive to class imbalance

🔹 Support Vector Machine (SVM)

- Linear SVM (LinearSVC)

- Strong baseline for high-dimensional tabular data

- Used with class_weight="balanced"

### Note:
The **main.ipynb** notebook also includes visualizations, plots, and representative examples from all project tasks (EDA, sampling, clustering, and classification), providing an interactive overview of the entire workflow and results.

### 📈 Evaluation

Metrics used:

- Accuracy

- F1-score (macro) → main metric for imbalanced & multi-class problems

- F1-score (weighted)

Key findings:

- Stratified dataset provides the most reliable results

- Birch representatives offer a strong trade-off between compression and accuracy

- MiniBatch K-Means struggles to preserve minority classes, especially for Traffic Type

- MLP generally outperforms SVM on complex multi-class tasks

-  SVM performs competitively on the binary Label task

#### All results and interpretations are documented in report.pdf.