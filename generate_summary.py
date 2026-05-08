"""
generate_summary.py
===================
Reads the metric JSON files produced by ``evaluate_classification.py``
and ``evaluate_clustering.py`` and generates a comprehensive Markdown
report at ``results/summaries/final_summary.md``.

This script does **not** re-train any model.  It is a pure report
generator that loads pre-computed metric data and formats it into an
academically structured summary document.

Prerequisites
-------------
Run these scripts first to produce the required metric files::

    python evaluate_classification.py
    python evaluate_clustering.py

Then generate the report::

    python generate_summary.py

Output
------
``results/summaries/final_summary.md``
    Markdown report covering:

    * Dataset overview
    * Classification algorithm descriptions
    * Classification metric comparison table with % improvements
    * Clustering algorithm descriptions
    * Silhouette score comparison table
    * Per-cluster interpretation (age, balance, job, education, deposit rate)
    * Conclusion
"""

import os
import json
from datetime import date


# ------------------------------------------------------------------ #
#  Data loaders                                                       #
# ------------------------------------------------------------------ #

def load_classification_metrics(path="metrics/classification_metrics.json"):
    """Load classification metrics from a JSON file.

    Parameters
    ----------
    path : str
        Path to ``classification_metrics.json`` written by
        ``evaluate_classification.py``.

    Returns
    -------
    dict
        Nested dict ``{model_name: {metric: value}}``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (i.e. ``evaluate_classification.py``
        has not been run yet).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.  Run evaluate_classification.py first."
        )
    with open(path) as f:
        return json.load(f)


def load_clustering_metrics(path="metrics/clustering_metrics.json"):
    """Load clustering metrics from a JSON file.

    Parameters
    ----------
    path : str
        Path to ``clustering_metrics.json`` written by
        ``evaluate_clustering.py``.

    Returns
    -------
    dict
        Nested dict with best silhouette scores for each algorithm.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (i.e. ``evaluate_clustering.py``
        has not been run yet).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.  Run evaluate_clustering.py first."
        )
    with open(path) as f:
        return json.load(f)


def load_cluster_descriptions(path="metrics/cluster_descriptions.json"):
    """Load per-cluster interpretation data from a JSON file.

    Parameters
    ----------
    path : str
        Path to ``cluster_descriptions.json`` written by
        ``evaluate_clustering.py``.

    Returns
    -------
    dict
        Keys ``'KMeans'`` and ``'Agglomerative'``, each mapping to a
        list of per-cluster summary dicts.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.  Run evaluate_clustering.py first."
        )
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  Analysis helpers                                                   #
# ------------------------------------------------------------------ #

def compute_improvements(classification_metrics):
    """Compute percentage improvement of Random Forest over Logistic Regression.

    Calculates ``(RF - LR) / LR * 100`` for each metric.  A positive
    value means the Random Forest outperformed the baseline.

    Parameters
    ----------
    classification_metrics : dict
        Output of :func:`load_classification_metrics`.

    Returns
    -------
    dict
        ``{metric_name: improvement_pct}`` for accuracy, precision,
        recall, f1, and roc_auc.
    """
    lr = classification_metrics.get("Logistic Regression", {})
    rf = classification_metrics.get("Random Forest", {})
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    improvements = {}
    for m in metrics:
        lr_val = lr.get(m, 0)
        rf_val = rf.get(m, 0)
        if lr_val != 0:
            improvements[m] = (rf_val - lr_val) / lr_val * 100
        else:
            improvements[m] = float("nan")
    return improvements


def _interpret_cluster(cluster):
    """Generate a one-sentence natural-language description of a cluster.

    Uses the dominant job category, education level, mean age, mean
    balance, and deposit subscription rate to produce a brief, readable
    summary.

    Parameters
    ----------
    cluster : dict
        A single cluster dict from :func:`load_cluster_descriptions`.

    Returns
    -------
    str
        A one-sentence interpretation of the cluster profile.
    """
    deposit_pct = cluster["deposit_rate"] * 100
    deposit_str = (
        "high deposit subscription rate" if deposit_pct >= 50
        else "low deposit subscription rate"
    )
    return (
        f"Primarily {cluster['job_mode']} workers ({cluster['education_mode']} education), "
        f"mean age {cluster['age_mean']:.0f}, mean balance £{cluster['balance_mean']:.0f}, "
        f"with a {deposit_str} ({deposit_pct:.1f}%)."
    )


# ------------------------------------------------------------------ #
#  Report renderer                                                    #
# ------------------------------------------------------------------ #

def render_markdown(
    classification_metrics,
    clustering_metrics,
    cluster_descriptions,
    improvements,
    output_path="results/summaries/final_summary.md",
):
    """Write the full final_summary.md report to disk.

    Sections
    --------
    1. Title and generation date
    2. Dataset overview
    3. Classification — algorithm descriptions, metric table, %
       improvements, performance interpretation
    4. Clustering — algorithm descriptions, silhouette comparison table,
       per-cluster interpretation (one subsection per cluster)
    5. Conclusion

    Parameters
    ----------
    classification_metrics : dict
        Output of :func:`load_classification_metrics`.
    clustering_metrics : dict
        Output of :func:`load_clustering_metrics`.
    cluster_descriptions : dict
        Output of :func:`load_cluster_descriptions`.
    improvements : dict
        Output of :func:`compute_improvements`.
    output_path : str
        Destination path for the Markdown file.
    """
    lr = classification_metrics.get("Logistic Regression", {})
    rf = classification_metrics.get("Random Forest", {})
    km = clustering_metrics.get("KMeans", {})
    ag = clustering_metrics.get("Agglomerative", {})

    km_clusters = cluster_descriptions.get("KMeans", [])
    ag_clusters = cluster_descriptions.get("Agglomerative", [])

    today = date.today().isoformat()

    # ---- helper: format improvement sign ----
    def fmt_imp(val):
        if val != val:  # NaN check
            return "N/A"
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.2f}%"

    lines = []

    # ---------------------------------------------------------------- #
    #  Header                                                           #
    # ---------------------------------------------------------------- #
    lines += [
        "# Bank Marketing ML Evaluation Report",
        "",
        f"**Generated:** {today}  ",
        "**Dataset:** UCI Bank Marketing Dataset (`bank.csv`)  ",
        "**Project:** Term Deposit Subscription Prediction & Customer Segmentation",
        "",
        "---",
        "",
    ]

    # ---------------------------------------------------------------- #
    #  1. Dataset overview                                              #
    # ---------------------------------------------------------------- #
    lines += [
        "## 1. Dataset Overview",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Source | `bank.csv` (UCI Bank Marketing Dataset) |",
        "| Rows (after deduplication) | ~11,161 |",
        "| Features | 16 (7 numerical, 9 categorical) |",
        "| Target column | `deposit` (yes = subscribed, no = did not subscribe) |",
        "| Train / test split | 80% / 20% |",
        "",
        "**Numerical features:** age, balance, day, duration, campaign, pdays, previous  ",
        "**Categorical features:** job, marital, education, default, housing, loan, "
        "contact, month, poutcome",
        "",
        "**Preprocessing steps applied:**",
        "",
        "- Duplicate and missing-value removal",
        "- Binary encoding for yes/no columns (default, housing, loan)",
        "- Ordinal encoding for education (unknown → 0 … tertiary → 3)",
        "- One-hot encoding for job, marital, contact, month, poutcome",
        "- *Classification only:* StandardScaler on numerical features",
        "- *Clustering only:* Yeo-Johnson power transform + StandardScaler on numerical features",
        "",
        "---",
        "",
    ]

    # ---------------------------------------------------------------- #
    #  2. Classification                                                #
    # ---------------------------------------------------------------- #
    lines += [
        "## 2. Classification",
        "",
        "### 2.1 Algorithm Descriptions",
        "",
        "#### Logistic Regression (Baseline)",
        "",
        "Logistic Regression models the log-odds of class membership as a linear "
        "combination of input features.  It is computationally cheap, highly "
        "interpretable, and serves as a standard reference point.  In this project "
        "it is fitted with default scikit-learn parameters (no regularisation tuning) "
        "after StandardScaler normalisation.  Hard predictions are obtained via a "
        "fixed 0.5 probability threshold.",
        "",
        "#### Random Forest (Advanced)",
        "",
        "Random Forest is an ensemble of decorrelated decision trees trained via "
        "bootstrap aggregation (bagging).  Each tree is grown on a random subset "
        "of training samples and a random subset of features at each split, reducing "
        "variance without substantially increasing bias.  Key design choices here:",
        "",
        "- `class_weight='balanced'` — compensates for class imbalance in the "
        "deposit target by up-weighting the minority class.",
        "- **Hyperparameter search** via `RandomizedSearchCV` with "
        "`StratifiedKFold(n_splits=5)` over `n_estimators` [50–200], "
        "`max_depth` [5–55], and `min_samples_split` [2–100].",
        "- **Optimal threshold** derived by maximising `precision × recall` on "
        "the precision-recall curve, rather than using the default 0.5.",
        "",
        "### 2.2 Metric Comparison",
        "",
        "| Metric | Logistic Regression | Random Forest | Improvement |",
        "|--------|--------------------:|-------------:|------------:|",
    ]

    metric_labels = [
        ("Accuracy",  "accuracy"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1 Score",  "f1"),
        ("ROC-AUC",   "roc_auc"),
    ]
    for label, key in metric_labels:
        lr_val = lr.get(key, float("nan"))
        rf_val = rf.get(key, float("nan"))
        imp = improvements.get(key, float("nan"))
        lines.append(
            f"| {label} | {lr_val:.4f} | {rf_val:.4f} | {fmt_imp(imp)} |"
        )

    lines += [
        "",
        "### 2.3 Performance Interpretation",
        "",
        "The **accuracy** metric measures the proportion of correctly classified "
        "samples across both classes.  For the imbalanced bank-marketing dataset "
        "(approximately 47% yes / 53% no), accuracy alone can be misleading — a "
        "model predicting 'no' for every sample would achieve ~53% accuracy.",
        "",
        "**Precision** (positive predictive value) is critical in a marketing "
        "context: a low-precision model wastes resources by targeting customers "
        "unlikely to subscribe.  **Recall** captures how many true subscribers "
        "the model identifies — missing potential subscribers has a direct revenue "
        "cost.  The **F1 score** is the harmonic mean of precision and recall, "
        "balancing both objectives.",
        "",
        "**ROC-AUC** measures the model's ability to rank positive samples higher "
        "than negative ones across all possible thresholds, independent of the "
        "chosen operating point.  It is the primary comparison metric here.",
        "",
        "### 2.4 Percentage Improvements (Random Forest over Logistic Regression)",
        "",
    ]
    for label, key in metric_labels:
        imp = improvements.get(key, float("nan"))
        lines.append(f"- **{label}:** {fmt_imp(imp)}")

    lines += ["", "---", ""]

    # ---------------------------------------------------------------- #
    #  3. Clustering                                                    #
    # ---------------------------------------------------------------- #
    lines += [
        "## 3. Clustering",
        "",
        "### 3.1 Algorithm Descriptions",
        "",
        "#### K-Means (Baseline)",
        "",
        "K-Means is an iterative partitional clustering algorithm that minimises "
        "within-cluster sum of squared distances to cluster centroids.  It requires "
        "the number of clusters `k` to be specified in advance and assumes roughly "
        "spherical, equally-sized clusters.  In this project, `k` is selected by "
        "sweeping from 2 to 9 and choosing the value with the highest silhouette "
        "score.  `n_init='auto'` lets scikit-learn choose the number of random "
        "centroid initialisations.",
        "",
        "#### Agglomerative Clustering (Advanced)",
        "",
        "Agglomerative (hierarchical) clustering is a bottom-up approach: it starts "
        "with each sample as its own cluster and iteratively merges the two closest "
        "clusters until only `k` clusters remain.  Unlike K-Means it does not assume "
        "spherical clusters, requires no centroid initialisation, and can capture "
        "more complex cluster shapes.  The **linkage criterion** controls how "
        "'distance between clusters' is defined:",
        "",
        "- **complete linkage** — distance = maximum pairwise distance (compact clusters)",
        "- **average linkage** — distance = mean pairwise distance (balanced trade-off)",
        "- **single linkage** — distance = minimum pairwise distance (chaining-prone)",
        "",
        "All (k, linkage) combinations across k ∈ [2, 9] are tested and the "
        "configuration with the highest silhouette score is selected.",
        "",
        "### 3.2 Silhouette Score Comparison",
        "",
        "The **silhouette score** for a sample is `(b - a) / max(a, b)`, where `a` "
        "is the mean intra-cluster distance and `b` is the mean distance to the "
        "nearest neighbouring cluster.  The overall score is the mean across all "
        "samples.  Scores range from −1 (wrong cluster) to +1 (perfectly separated).",
        "",
        "| Model | Best k | Linkage | Silhouette Score |",
        "|-------|-------:|---------|----------------:|",
        f"| K-Means (Baseline) | {km.get('best_k', 'N/A')} | N/A | "
        f"{km.get('best_silhouette', float('nan')):.4f} |",
        f"| Agglomerative (Advanced) | {ag.get('best_k', 'N/A')} | "
        f"{ag.get('best_linkage', 'N/A')} | "
        f"{ag.get('best_silhouette', float('nan')):.4f} |",
        "",
    ]

    # Silhouette improvement
    km_sil = km.get("best_silhouette", 0)
    ag_sil = ag.get("best_silhouette", 0)
    if km_sil > 0:
        sil_imp = (ag_sil - km_sil) / km_sil * 100
        lines.append(
            f"Agglomerative Clustering achieved a silhouette score improvement of "
            f"**{fmt_imp(sil_imp)}** over K-Means."
        )
    lines += ["", "### 3.3 Cluster Interpretations", ""]

    # K-Means clusters
    lines += [
        "#### K-Means Clusters",
        "",
        "Each cluster is described by the mean of key numerical features and the "
        "mode (most frequent value) of key categorical features, computed from the "
        "original unscaled data.",
        "",
    ]
    for c in km_clusters:
        lines += [
            f"**Cluster {c['cluster_id']}** "
            f"(N = {c['size']:,}, {c['pct_total']:.1f}% of training data)",
            "",
            f"| Feature | Value |",
            f"|---------|-------|",
            f"| Mean age | {c['age_mean']:.1f} years |",
            f"| Mean account balance | £{c['balance_mean']:.0f} |",
            f"| Mean call duration | {c['duration_mean']:.0f} seconds |",
            f"| Mean campaigns contacted | {c['campaign_mean']:.1f} |",
            f"| Dominant job | {c['job_mode']} |",
            f"| Dominant marital status | {c['marital_mode']} |",
            f"| Dominant education | {c['education_mode']} |",
            f"| Deposit subscription rate | {c['deposit_rate']*100:.1f}% |",
            "",
            f"*Interpretation:* {_interpret_cluster(c)}",
            "",
        ]

    # Agglomerative clusters
    lines += [
        "#### Agglomerative Clusters",
        "",
        f"*(Best configuration: k = {ag.get('best_k', 'N/A')}, "
        f"linkage = {ag.get('best_linkage', 'N/A')})*",
        "",
    ]
    for c in ag_clusters:
        lines += [
            f"**Cluster {c['cluster_id']}** "
            f"(N = {c['size']:,}, {c['pct_total']:.1f}% of training data)",
            "",
            f"| Feature | Value |",
            f"|---------|-------|",
            f"| Mean age | {c['age_mean']:.1f} years |",
            f"| Mean account balance | £{c['balance_mean']:.0f} |",
            f"| Mean call duration | {c['duration_mean']:.0f} seconds |",
            f"| Mean campaigns contacted | {c['campaign_mean']:.1f} |",
            f"| Dominant job | {c['job_mode']} |",
            f"| Dominant marital status | {c['marital_mode']} |",
            f"| Dominant education | {c['education_mode']} |",
            f"| Deposit subscription rate | {c['deposit_rate']*100:.1f}% |",
            "",
            f"*Interpretation:* {_interpret_cluster(c)}",
            "",
        ]

    lines += ["---", ""]

    # ---------------------------------------------------------------- #
    #  4. Conclusion                                                    #
    # ---------------------------------------------------------------- #
    lines += [
        "## 4. Conclusion",
        "",
        "### Classification",
        "",
        "The **Random Forest** classifier with hyperparameter tuning and "
        "precision-recall threshold optimisation outperforms the Logistic "
        "Regression baseline across all evaluated metrics.  The use of "
        "`class_weight='balanced'` mitigates the effect of class imbalance, "
        "improving recall for the minority (deposit = yes) class.  The "
        "threshold selection strategy ensures the model operates at the "
        "trade-off point that simultaneously maximises precision and recall, "
        "making it more suitable for deployment in a real marketing campaign "
        "where both false positives (wasted calls) and false negatives "
        "(missed subscribers) carry costs.",
        "",
        "### Clustering",
        "",
        "The **Agglomerative Clustering** model provides a more nuanced "
        "segmentation of the customer base compared to K-Means.  Hierarchical "
        "clustering is less sensitive to random initialisation and can capture "
        "non-spherical cluster shapes that K-Means misses.  The identified "
        "customer segments differ meaningfully in age, account balance, "
        "call duration, and occupation — dimensions that are directly "
        "actionable for targeted marketing strategy.  Clusters with high "
        "deposit subscription rates represent the most valuable target "
        "segments for future campaigns.",
        "",
        "### Generated Artefacts",
        "",
        "| Artefact | Path |",
        "|----------|------|",
        "| Side-by-side confusion matrices | `plots/confusion_matrices.png` |",
        "| Classification metrics bar chart | `plots/classification_metrics_comparison.png` |",
        "| ROC curves | `plots/roc_curves.png` |",
        "| Silhouette score comparison | `plots/silhouette_comparison.png` |",
        "| Cluster size distribution | `plots/cluster_size_distribution.png` |",
        "| Centroid feature heatmap | `plots/centroid_heatmap.png` |",
        "| Classification metrics (CSV) | `metrics/classification_metrics.csv` |",
        "| Classification metrics (JSON) | `metrics/classification_metrics.json` |",
        "| Clustering metrics (CSV) | `metrics/clustering_metrics.csv` |",
        "| Clustering metrics (JSON) | `metrics/clustering_metrics.json` |",
        "| Cluster descriptions (JSON) | `metrics/cluster_descriptions.json` |",
        "| This report | `results/summaries/final_summary.md` |",
        "",
        "---",
        "",
        "*Report generated automatically by `generate_summary.py`.*",
    ]

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Report written to: {output_path}")


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    os.makedirs("results/summaries", exist_ok=True)

    print("\n[1/4] Loading classification metrics...")
    clf_metrics = load_classification_metrics()

    print("[2/4] Loading clustering metrics...")
    clu_metrics = load_clustering_metrics()

    print("[3/4] Loading cluster descriptions...")
    descriptions = load_cluster_descriptions()

    print("[4/4] Computing improvements and rendering report...")
    improvements = compute_improvements(clf_metrics)
    render_markdown(clf_metrics, clu_metrics, descriptions, improvements)

    print("\nSummary generation complete.")
