"""
evaluate_clustering.py

Standalone evaluation script that trains and compares two clustering
algorithms on the bank-marketing dataset:

* **K-Means** (baseline) — replicates ``Clustering_Baseline.py`` exactly:
  sweeps ``k`` from 2 to 9 and selects the run with the highest
  silhouette score.

* **Agglomerative Clustering** (advanced) — replicates the training logic
  in ``Model_Training_Evaluation.clustering_model``: exhaustively tests
  all combinations of ``n_clusters`` ∈ [2, 9] and linkage methods
  ``{complete, average, single}``.

Generated outputs

plots/
    silhouette_comparison.png
        Bar chart comparing best silhouette scores for both algorithms.
    cluster_size_distribution.png
        Side-by-side bar charts showing how many samples each cluster
        contains for both best models.
    centroid_heatmap.png
        Heatmap of mean standardised feature values per cluster (both
        algorithms stacked vertically).

metrics/
    clustering_metrics.csv
        Best score, k, and linkage for each model.
    clustering_metrics.json
        Nested dict with best results.
    cluster_descriptions.json
        Per-cluster interpretations derived from original unscaled data
        (age, balance, duration, job, education, deposit rate).

Console
    Best silhouette scores and cluster counts for both models.

Run
---
::

    python evaluate_clustering.py

Notes
-----
* ``matplotlib.use("Agg")`` is set at module level for headless operation.
* This script does **not** import ``Clustering_Baseline.py`` or
  ``Model_Training_Evaluation.py`` to avoid triggering their module-level
  training code.  The clustering pipelines are replicated inline.
* Because ``Preprocessing.clustering_processing`` does not set a
  ``random_state`` in ``train_test_split``, the pandas index of
  ``X_train`` is used to align cluster labels back to the original
  unscaled DataFrame for interpretation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — must come before pyplot import
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from Preprocessing import Preprocessing


# ------------------------------------------------------------------ #
#  Model sweeps                                                       #
# ------------------------------------------------------------------ #

def run_kmeans_sweep(X_train, k_range=range(2, 10)):
    """Run K-Means for each k in ``k_range`` and return the best result.

    Replicates ``Clustering_Baseline.py`` exactly.  K-Means is fitted
    with ``n_init='auto'`` (scikit-learn chooses the number of restarts).
    The best model is selected by the highest silhouette score.

    Parameters
    X_train : pd.DataFrame
        Encoded and scaled training feature matrix from
        Preprocessing.clustering_processing.
    k_range : range
        Cluster counts to evaluate (default 2–9 inclusive).

    Returns
    dict
        Keys:

        * ``best_model``  – fitted :class:`~sklearn.cluster.KMeans`
        * ``best_k``      – int, optimal number of clusters
        * ``best_score``  – float, best silhouette score
        * ``all_scores``  – dict ``{k: silhouette_score}``
        * ``best_labels`` – np.ndarray, cluster labels for ``X_train``
    """
    best_score = 0.0
    best_model = None
    best_labels = None
    all_scores = {}

    for k in k_range:
        model = KMeans(n_clusters=k, n_init="auto")
        labels = model.fit_predict(X_train)
        score = silhouette_score(X_train, labels)
        all_scores[k] = score

        if score > best_score:
            best_score = score
            best_model = model
            best_labels = labels.copy()

    return {
        "best_model": best_model,
        "best_k": best_model.n_clusters,
        "best_score": best_score,
        "all_scores": all_scores,
        "best_labels": best_labels,
    }


def run_agglomerative_sweep(
    X_train,
    k_range=range(2, 10),
    linkage_methods=None,
):
    """Run Agglomerative Clustering for all (k, linkage) combinations.

    Replicates ``Model_Training_Evaluation.clustering_model`` exactly.
    Tests 8 × 3 = 24 configurations and returns the one with the highest
    silhouette score.

    Linkage method semantics
    
    complete
        Distance between clusters = max pairwise distance (merges compact
        clusters; sensitive to outliers).
    average
        Distance = average of all pairwise distances (balanced trade-off).
    single
        Distance = min pairwise distance (can produce elongated clusters;
        prone to chaining).

    Parameters
    X_train : pd.DataFrame
        Encoded and scaled training feature matrix.
    k_range : range
        Cluster counts to evaluate (default 2–9 inclusive).
    linkage_methods : list of str, optional
        Linkage methods to test.  Defaults to
        ``['complete', 'average', 'single']``.

    Returns
    dict
        Keys:

        * ``best_model``    – fitted AgglomerativeClustering
        * ``best_k``        – int
        * ``best_linkage``  – str
        * ``best_score``    – float
        * ``all_scores``    – dict ``{(k, linkage): silhouette_score}``
        * ``best_labels``   – np.ndarray, cluster labels for ``X_train``
    """
    if linkage_methods is None:
        linkage_methods = ["complete", "average", "single"]

    best_score = -1.0
    best_model = None
    best_labels = None
    best_k = -1
    best_linkage = None
    all_scores = {}

    for k in k_range:
        for linkage in linkage_methods:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(X_train)
            score = silhouette_score(X_train, labels)
            all_scores[(k, linkage)] = score

            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels.copy()
                best_k = k
                best_linkage = linkage

    return {
        "best_model": best_model,
        "best_k": best_k,
        "best_linkage": best_linkage,
        "best_score": best_score,
        "all_scores": {str(k): v for k, v in all_scores.items()},  # JSON-serialisable keys
        "best_labels": best_labels,
    }


#  Cluster interpretation                                             

def describe_clusters(labels, X_train_raw, y_train_raw, model_name):
    """Describe each cluster using original (unscaled) feature values.

    Groups the pre-encoded, pre-scaled rows of ``X_train_raw`` by cluster
    label, then computes summary statistics per cluster.  This gives
    human-interpretable descriptions of what each cluster represents in
    the context of the bank marketing campaign.

    Parameters
    labels : np.ndarray
        Cluster label array aligned row-by-row with ``X_train_raw``.
    X_train_raw : pd.DataFrame
        Original unencoded feature rows (subset of ``bank.csv``).
    y_train_raw : pd.Series
        Original deposit target values (``'yes'`` / ``'no'``) aligned
        with ``X_train_raw`` (used to compute deposit subscription rate
        per cluster).
    model_name : str
        ``'K-Means'`` or ``'Agglomerative'`` — used for labelling.

    Returns
    list of dict
        One dict per cluster with keys: ``model``, ``cluster_id``,
        ``size``, ``pct_total``, ``age_mean``, ``balance_mean``,
        ``duration_mean``, ``campaign_mean``, ``job_mode``,
        ``marital_mode``, ``education_mode``, ``deposit_rate``.
    """
    descriptions = []
    n_total = len(labels)

    for cluster_id in sorted(np.unique(labels)):
        # Boolean mask selecting rows belonging to this cluster
        mask = labels == cluster_id
        cluster_rows = X_train_raw.iloc[mask] if not X_train_raw.index.equals(
            pd.RangeIndex(len(X_train_raw))
        ) else X_train_raw.loc[np.where(mask)[0]]

        # Use positional indexing to stay robust to any index type
        cluster_rows = X_train_raw.iloc[mask]
        cluster_y = y_train_raw.iloc[mask]

        size = mask.sum()

        # Mean of numerical features — describes the 'average' cluster member
        age_mean = cluster_rows["age"].mean() if "age" in cluster_rows else float("nan")
        balance_mean = cluster_rows["balance"].mean() if "balance" in cluster_rows else float("nan")
        duration_mean = cluster_rows["duration"].mean() if "duration" in cluster_rows else float("nan")
        campaign_mean = cluster_rows["campaign"].mean() if "campaign" in cluster_rows else float("nan")

        # Mode of categorical features — most common category per cluster
        job_mode = cluster_rows["job"].mode().iloc[0] if "job" in cluster_rows and len(cluster_rows) > 0 else "unknown"
        marital_mode = cluster_rows["marital"].mode().iloc[0] if "marital" in cluster_rows and len(cluster_rows) > 0 else "unknown"
        education_mode = cluster_rows["education"].mode().iloc[0] if "education" in cluster_rows and len(cluster_rows) > 0 else "unknown"

        # Fraction of cluster that subscribed to a term deposit
        deposit_rate = (cluster_y == "yes").sum() / size if size > 0 else 0.0

        descriptions.append({
            "model": model_name,
            "cluster_id": int(cluster_id),
            "size": int(size),
            "pct_total": round(float(size / n_total * 100), 2),
            "age_mean": round(float(age_mean), 2),
            "balance_mean": round(float(balance_mean), 2),
            "duration_mean": round(float(duration_mean), 2),
            "campaign_mean": round(float(campaign_mean), 2),
            "job_mode": str(job_mode),
            "marital_mode": str(marital_mode),
            "education_mode": str(education_mode),
            "deposit_rate": round(float(deposit_rate), 4),
        })

    return descriptions


# ------------------------------------------------------------------ #
#  Plot generators                                                    #
# ------------------------------------------------------------------ #

def plot_silhouette_comparison(kmeans_result, agg_result, output_dir="plots"):
    """Save a bar chart comparing best silhouette scores for both models.

    Each bar is annotated with its exact score value.  Higher silhouette
    scores (closer to 1.0) indicate tighter, better-separated clusters.

    Parameters
    kmeans_result : dict
        Output of :func:`run_kmeans_sweep`.
    agg_result : dict
        Output of :func:`run_agglomerative_sweep`.
    output_dir : str
        Directory to save the figure.

    Output
    ``{output_dir}/silhouette_comparison.png``
    """
    models = ["K-Means\n(Baseline)", "Agglomerative\n(Advanced)"]
    scores = [kmeans_result["best_score"], agg_result["best_score"]]
    colors = ["steelblue", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, scores, color=colors, width=0.4, edgecolor="black", linewidth=0.7)

    # Annotate each bar with its score value above the bar
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Silhouette Score (higher = better)")
    ax.set_title("Best Silhouette Score Comparison\nK-Means vs Agglomerative Clustering")
    ax.set_ylim(0, max(scores) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "silhouette_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cluster_size_distribution(kmeans_result, agg_result, output_dir="plots"):
    """Save side-by-side bar charts of cluster size distributions.

    Shows how many training samples are assigned to each cluster for both
    the best K-Means and Agglomerative models.  Unequal cluster sizes can
    indicate imbalanced segmentation.

    Parameters
    kmeans_result : dict
        Output of :func:`run_kmeans_sweep`.
    agg_result : dict
        Output of :func:`run_agglomerative_sweep`.
    output_dir : str
        Directory to save the figure.

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cluster Size Distribution", fontsize=14)

    for ax, result, title, color in [
        (axes[0], kmeans_result, f"K-Means (k={kmeans_result['best_k']})", "steelblue"),
        (axes[1], agg_result,   f"Agglomerative (k={agg_result['best_k']}, "
                                 f"linkage={agg_result['best_linkage']})", "darkorange"),
    ]:
        labels = result["best_labels"]
        unique_ids, counts = np.unique(labels, return_counts=True)

        bars = ax.bar(unique_ids, counts, color=color, edgecolor="black", linewidth=0.7)

        # Annotate each bar with its count
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(count),
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Number of Samples")
        ax.set_title(title)
        ax.set_xticks(unique_ids)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_size_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_centroid_heatmap(kmeans_result, agg_result, X_train, output_dir="plots"):
    """Save a feature-mean heatmap per cluster for both models.

    Visualises the average standardised feature value for each cluster,
    giving insight into which features drive cluster separation.  For
    Agglomerative Clustering (which has no ``cluster_centers_`` attribute),
    centroids are computed as the column-wise mean of all samples in each
    cluster.

    If the feature space has more than 30 columns after one-hot encoding,
    only numerical features are plotted to keep the chart readable.

    Parameters
    ----------
    kmeans_result : dict
        Output of :func:`run_kmeans_sweep`.
    agg_result : dict
        Output of :func:`run_agglomerative_sweep`.
    X_train : pd.DataFrame
        The same encoded/scaled feature matrix used during training.
    output_dir : str
        Directory to save the figure.

    Output
    ------
    ``{output_dir}/centroid_heatmap.png``
    """
    # Limit to numerical columns if OHE expanded the feature space too much
    if X_train.shape[1] > 30:
        plot_cols = X_train.select_dtypes(include="number").columns
    else:
        plot_cols = X_train.columns
    X_plot = X_train[plot_cols]

    def compute_centroids(labels, X):
        """Compute mean feature vector for each cluster label."""
        cluster_ids = sorted(np.unique(labels))
        rows = []
        for cid in cluster_ids:
            rows.append(X.iloc[labels == cid].mean().values)
        return np.array(rows), cluster_ids

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Feature Means by Cluster (Standardised Values)", fontsize=14)

    for ax, result, title in [
        (axes[0], kmeans_result, f"K-Means (k={kmeans_result['best_k']})"),
        (axes[1], agg_result,    f"Agglomerative (k={agg_result['best_k']}, "
                                  f"linkage={agg_result['best_linkage']})"),
    ]:
        centroids, cluster_ids = compute_centroids(result["best_labels"], X_plot)

        im = ax.imshow(centroids, aspect="auto", cmap="coolwarm", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Standardised Feature Value")

        ax.set_yticks(range(len(cluster_ids)))
        ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids])
        ax.set_xticks(range(len(plot_cols)))
        ax.set_xticklabels(plot_cols, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(output_dir, "centroid_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
#  Metric persistence                                                 #
# ------------------------------------------------------------------ #

def save_metrics(kmeans_result, agg_result, kmeans_desc, agg_desc, output_dir="metrics"):
    """Save clustering metrics and cluster descriptions to CSV and JSON.

    Parameters
    ----------
    kmeans_result : dict
        Output of :func:`run_kmeans_sweep`.
    agg_result : dict
        Output of :func:`run_agglomerative_sweep`.
    kmeans_desc : list of dict
        Output of :func:`describe_clusters` for K-Means.
    agg_desc : list of dict
        Output of :func:`describe_clusters` for Agglomerative.
    output_dir : str
        Directory to save metric files.

    Outputs
    -------
    ``{output_dir}/clustering_metrics.csv``
    ``{output_dir}/clustering_metrics.json``
    ``{output_dir}/cluster_descriptions.json``
    """
    # Summary table: one row per model with best score details
    rows = [
        {
            "model": "K-Means (Baseline)",
            "best_k": kmeans_result["best_k"],
            "best_linkage": "N/A",
            "best_silhouette": kmeans_result["best_score"],
        },
        {
            "model": "Agglomerative (Advanced)",
            "best_k": agg_result["best_k"],
            "best_linkage": agg_result["best_linkage"],
            "best_silhouette": agg_result["best_score"],
        },
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "clustering_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # JSON with best results (easy to load in generate_summary.py)
    json_data = {
        "KMeans": {
            "best_k": kmeans_result["best_k"],
            "best_silhouette": kmeans_result["best_score"],
        },
        "Agglomerative": {
            "best_k": agg_result["best_k"],
            "best_linkage": agg_result["best_linkage"],
            "best_silhouette": agg_result["best_score"],
        },
    }
    json_path = os.path.join(output_dir, "clustering_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Per-cluster descriptions for both models
    descriptions_data = {
        "KMeans": kmeans_desc,
        "Agglomerative": agg_desc,
    }
    desc_path = os.path.join(output_dir, "cluster_descriptions.json")
    with open(desc_path, "w") as f:
        json.dump(descriptions_data, f, indent=2)
    print(f"  Saved: {desc_path}")


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Create output directories if they do not exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    DATA_FILE = "bank.csv"

    print("\n[1/5] Loading raw data for cluster interpretation...")
    # Load the original unscaled data before any encoding/transformation.
    # This is used later in describe_clusters() to produce human-readable
    # cluster summaries with original feature values.
    df_raw = pd.read_csv(DATA_FILE).drop_duplicates().dropna()
    X_raw = df_raw.drop(columns=["deposit"])
    y_raw = df_raw["deposit"]  # original yes/no labels for deposit rate calculation

    print("\n[2/5] Preprocessing for clustering...")
    prep = Preprocessing()
    X_train, X_test = prep.pre_processing("clustering", DATA_FILE)

    # Align raw rows to the training split using the preserved pandas index.
    # train_test_split keeps the original DataFrame index, so we can use .iloc
    # with a positional approach; using iloc with a boolean mask is safer here.
    X_train_raw = X_raw.iloc[X_train.index]
    y_train_raw = y_raw.iloc[X_train.index]

    print("\n[3/5] Running K-Means sweep (baseline)...")
    kmeans_result = run_kmeans_sweep(X_train)
    print(f"       Best K-Means silhouette: {kmeans_result['best_score']:.4f} "
          f"(k={kmeans_result['best_k']})")

    print("\n[4/5] Running Agglomerative Clustering sweep (advanced)...")
    agg_result = run_agglomerative_sweep(X_train)
    print(f"       Best Agglomerative silhouette: {agg_result['best_score']:.4f} "
          f"(k={agg_result['best_k']}, linkage={agg_result['best_linkage']})")

    print("\n[5/5] Generating plots and saving metrics...")
    # Describe clusters using original feature values
    kmeans_desc = describe_clusters(
        kmeans_result["best_labels"], X_train_raw, y_train_raw, "K-Means"
    )
    agg_desc = describe_clusters(
        agg_result["best_labels"], X_train_raw, y_train_raw, "Agglomerative"
    )

    plot_silhouette_comparison(kmeans_result, agg_result)
    plot_cluster_size_distribution(kmeans_result, agg_result)
    plot_centroid_heatmap(kmeans_result, agg_result, X_train)
    save_metrics(kmeans_result, agg_result, kmeans_desc, agg_desc)

    print("\nClustering evaluation complete.")
