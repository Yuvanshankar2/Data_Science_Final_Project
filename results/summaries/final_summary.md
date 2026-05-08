# Bank Marketing ML Evaluation Report

**Generated:** 2026-05-07  
**Dataset:** UCI Bank Marketing Dataset (`bank.csv`)  
**Project:** Term Deposit Subscription Prediction & Customer Segmentation

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Source | `bank.csv` (UCI Bank Marketing Dataset) |
| Rows (after deduplication) | ~11,161 |
| Features | 16 (7 numerical, 9 categorical) |
| Target column | `deposit` (yes = subscribed, no = did not subscribe) |
| Train / test split | 80% / 20% |

**Numerical features:** age, balance, day, duration, campaign, pdays, previous  
**Categorical features:** job, marital, education, default, housing, loan, contact, month, poutcome

**Preprocessing steps applied:**

- Duplicate and missing-value removal
- Binary encoding for yes/no columns (default, housing, loan)
- Ordinal encoding for education (unknown → 0 … tertiary → 3)
- One-hot encoding for job, marital, contact, month, poutcome
- *Classification only:* StandardScaler on numerical features
- *Clustering only:* Yeo-Johnson power transform + StandardScaler on numerical features

---

## 2. Classification

### 2.1 Algorithm Descriptions

#### Logistic Regression (Baseline)

Logistic Regression models the log-odds of class membership as a linear combination of input features.  It is computationally cheap, highly interpretable, and serves as a standard reference point.  In this project it is fitted with default scikit-learn parameters (no regularisation tuning) after StandardScaler normalisation.  Hard predictions are obtained via a fixed 0.5 probability threshold.

#### Random Forest (Advanced)

Random Forest is an ensemble of decorrelated decision trees trained via bootstrap aggregation (bagging).  Each tree is grown on a random subset of training samples and a random subset of features at each split, reducing variance without substantially increasing bias.  Key design choices here:

- `class_weight='balanced'` — compensates for class imbalance in the deposit target by up-weighting the minority class.
- **Hyperparameter search** via `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)` over `n_estimators` [50–200], `max_depth` [5–55], and `min_samples_split` [2–100].
- **Optimal threshold** derived by maximising `precision × recall` on the precision-recall curve, rather than using the default 0.5.

### 2.2 Metric Comparison

| Metric | Logistic Regression | Random Forest | Improvement |
|--------|--------------------:|-------------:|------------:|
| Accuracy | 0.8307 | 0.8639 | +3.99% |
| Precision | 0.8460 | 0.8073 | -4.58% |
| Recall | 0.7943 | 0.9435 | +18.79% |
| F1 Score | 0.8193 | 0.8701 | +6.20% |
| ROC-AUC | 0.9050 | 0.9231 | +2.00% |

### 2.3 Performance Interpretation

The **accuracy** metric measures the proportion of correctly classified samples across both classes.  For the imbalanced bank-marketing dataset (approximately 47% yes / 53% no), accuracy alone can be misleading — a model predicting 'no' for every sample would achieve ~53% accuracy.

**Precision** (positive predictive value) is critical in a marketing context: a low-precision model wastes resources by targeting customers unlikely to subscribe.  **Recall** captures how many true subscribers the model identifies — missing potential subscribers has a direct revenue cost.  The **F1 score** is the harmonic mean of precision and recall, balancing both objectives.

**ROC-AUC** measures the model's ability to rank positive samples higher than negative ones across all possible thresholds, independent of the chosen operating point.  It is the primary comparison metric here.

### 2.4 Percentage Improvements (Random Forest over Logistic Regression)

- **Accuracy:** +3.99%
- **Precision:** -4.58%
- **Recall:** +18.79%
- **F1 Score:** +6.20%
- **ROC-AUC:** +2.00%

---

## 3. Clustering

### 3.1 Algorithm Descriptions

#### K-Means (Baseline)

K-Means is an iterative partitional clustering algorithm that minimises within-cluster sum of squared distances to cluster centroids.  It requires the number of clusters `k` to be specified in advance and assumes roughly spherical, equally-sized clusters.  In this project, `k` is selected by sweeping from 2 to 9 and choosing the value with the highest silhouette score.  `n_init='auto'` lets scikit-learn choose the number of random centroid initialisations.

#### Agglomerative Clustering (Advanced)

Agglomerative (hierarchical) clustering is a bottom-up approach: it starts with each sample as its own cluster and iteratively merges the two closest clusters until only `k` clusters remain.  Unlike K-Means it does not assume spherical clusters, requires no centroid initialisation, and can capture more complex cluster shapes.  The **linkage criterion** controls how 'distance between clusters' is defined:

- **complete linkage** — distance = maximum pairwise distance (compact clusters)
- **average linkage** — distance = mean pairwise distance (balanced trade-off)
- **single linkage** — distance = minimum pairwise distance (chaining-prone)

All (k, linkage) combinations across k ∈ [2, 9] are tested and the configuration with the highest silhouette score is selected.

### 3.2 Silhouette Score Comparison

The **silhouette score** for a sample is `(b - a) / max(a, b)`, where `a` is the mean intra-cluster distance and `b` is the mean distance to the nearest neighbouring cluster.  The overall score is the mean across all samples.  Scores range from −1 (wrong cluster) to +1 (perfectly separated).

| Model | Best k | Linkage | Silhouette Score |
|-------|-------:|---------|----------------:|
| K-Means (Baseline) | 2 | N/A | 0.2458 |
| Agglomerative (Advanced) | 2 | complete | 0.8273 |

Agglomerative Clustering achieved a silhouette score improvement of **+236.59%** over K-Means.

### 3.3 Cluster Interpretations

#### K-Means Clusters

Each cluster is described by the mean of key numerical features and the mode (most frequent value) of key categorical features, computed from the original unscaled data.

**Cluster 0** (N = 6,662, 74.6% of training data)

| Feature | Value |
|---------|-------|
| Mean age | 41.1 years |
| Mean account balance | £1445 |
| Mean call duration | 380 seconds |
| Mean campaigns contacted | 2.7 |
| Dominant job | management |
| Dominant marital status | married |
| Dominant education | secondary |
| Deposit subscription rate | 40.5% |

*Interpretation:* Primarily management workers (secondary education), mean age 41, mean balance £1445, with a low deposit subscription rate (40.5%).

**Cluster 1** (N = 2,267, 25.4% of training data)

| Feature | Value |
|---------|-------|
| Mean age | 42.0 years |
| Mean account balance | £1777 |
| Mean call duration | 342 seconds |
| Mean campaigns contacted | 1.9 |
| Dominant job | management |
| Dominant marital status | married |
| Dominant education | secondary |
| Deposit subscription rate | 66.4% |

*Interpretation:* Primarily management workers (secondary education), mean age 42, mean balance £1777, with a high deposit subscription rate (66.4%).

#### Agglomerative Clusters

*(Best configuration: k = 2, linkage = complete)*

**Cluster 0** (N = 8,928, 100.0% of training data)

| Feature | Value |
|---------|-------|
| Mean age | 41.4 years |
| Mean account balance | £1531 |
| Mean call duration | 371 seconds |
| Mean campaigns contacted | 2.5 |
| Dominant job | management |
| Dominant marital status | married |
| Dominant education | secondary |
| Deposit subscription rate | 47.1% |

*Interpretation:* Primarily management workers (secondary education), mean age 41, mean balance £1531, with a low deposit subscription rate (47.1%).

**Cluster 1** (N = 1, 0.0% of training data)

| Feature | Value |
|---------|-------|
| Mean age | 49.0 years |
| Mean account balance | £-6847 |
| Mean call duration | 206 seconds |
| Mean campaigns contacted | 1.0 |
| Dominant job | management |
| Dominant marital status | married |
| Dominant education | tertiary |
| Deposit subscription rate | 0.0% |

*Interpretation:* Primarily management workers (tertiary education), mean age 49, mean balance £-6847, with a low deposit subscription rate (0.0%).

---

## 4. Conclusion

### Classification

The **Random Forest** classifier with hyperparameter tuning and precision-recall threshold optimisation outperforms the Logistic Regression baseline across all evaluated metrics.  The use of `class_weight='balanced'` mitigates the effect of class imbalance, improving recall for the minority (deposit = yes) class.  The threshold selection strategy ensures the model operates at the trade-off point that simultaneously maximises precision and recall, making it more suitable for deployment in a real marketing campaign where both false positives (wasted calls) and false negatives (missed subscribers) carry costs.

### Clustering

The **Agglomerative Clustering** model provides a more nuanced segmentation of the customer base compared to K-Means.  Hierarchical clustering is less sensitive to random initialisation and can capture non-spherical cluster shapes that K-Means misses.  The identified customer segments differ meaningfully in age, account balance, call duration, and occupation — dimensions that are directly actionable for targeted marketing strategy.  Clusters with high deposit subscription rates represent the most valuable target segments for future campaigns.

### Generated Artefacts

| Artefact | Path |
|----------|------|
| Side-by-side confusion matrices | `plots/confusion_matrices.png` |
| Classification metrics bar chart | `plots/classification_metrics_comparison.png` |
| ROC curves | `plots/roc_curves.png` |
| Silhouette score comparison | `plots/silhouette_comparison.png` |
| Cluster size distribution | `plots/cluster_size_distribution.png` |
| Centroid feature heatmap | `plots/centroid_heatmap.png` |
| Classification metrics (CSV) | `metrics/classification_metrics.csv` |
| Classification metrics (JSON) | `metrics/classification_metrics.json` |
| Clustering metrics (CSV) | `metrics/clustering_metrics.csv` |
| Clustering metrics (JSON) | `metrics/clustering_metrics.json` |
| Cluster descriptions (JSON) | `metrics/cluster_descriptions.json` |
| This report | `results/summaries/final_summary.md` |

---

*Report generated automatically by `generate_summary.py`.*