"""
customer_targeting_system.py
============================
Inference-based customer targeting and recommendation system.

Architecture
------------
This module is structured in two clearly separated phases:

**Setup phase** (runs once on startup)
    Calls :meth:`Model_Training_Evaluation.classification_model` and
    :meth:`Model_Training_Evaluation.clustering_model` — the existing
    training functions — to obtain the fitted model objects.  No new
    training logic is written here.

**Inference phase** (runs per customer)
    Takes a raw customer record, retrieves its correctly preprocessed
    representation from the pipeline that was fitted during setup, runs
    both models, and produces a marketing recommendation.

Why this separation matters
---------------------------
The previous approach ran predictions on the entire bank.csv dataset,
including records that the models were trained on.  This module instead
samples ``N`` customer records from the **held-out test set** (the 20%
of bank.csv that the classifier never saw during training).  Those
records serve as the "arriving customers" for the targeting system.
Their preprocessed forms (``X_test_clf``, and the clustering equivalent)
are the output of transformers that were *fitted only on the training
data* — so inference is correct and leak-free.

Inference pipeline per customer
--------------------------------
::

    Raw record (from bank.csv)
         |
         | [displayed as customer input]
         |
    X_test_clf[idx]        →  clf_model.predict_proba() →  deposit_prob, pred_class
    all_clust[idx]         →  nearest_centroid(agg_model) →  cluster_id
         |
    cluster_profiles[id]   →  characterise_clusters()   →  cluster_label
         |
    (deposit_prob, cluster_label) → lookup_recommendation() → action, reasoning

Clustering inference note
-------------------------
:class:`~sklearn.cluster.AgglomerativeClustering` is transductive — it
has no ``predict()`` method.  For customers whose preprocessed form lives
in ``X_test_clust``, cluster assignment is performed by nearest-centroid
lookup: cluster centroids are the column-wise means of the training rows
in each cluster (``agg_model.labels_``), and the test customer is
assigned to the closest centroid by Euclidean distance.

Run
---
::

    python customer_targeting_system.py

Outputs (saved to ``results/``)
--------------------------------
targeting_report.csv
    customer_id, raw input features, deposit_probability, predicted_class,
    cluster_id, cluster_label, recommended_action, reasoning.
deposit_probability_by_cluster.png
    Box plot of deposit probability distributions per segment.
cluster_distribution.png
    Bar chart of customer counts per segment.
recommendation_distribution.png
    Horizontal bar chart of recommended action frequencies.

Reused components
-----------------
``Preprocessing``               from Preprocessing.py
``Model_Training_Evaluation``   from Model_Training_Evaluation.py
    - classification_model()    → RandomForest + t_best
    - clustering_model()        → AgglomerativeClustering
"""

import os
import json
import numpy as np
import pandas as pd

# Set non-interactive backend BEFORE any pyplot import.
# Model_Training_Evaluation.py imports pyplot at module level, so this
# line must appear before that import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances

# ── Reuse the existing project modules ────────────────────────────────────────
from Preprocessing import Preprocessing
from Model_Training_Evaluation import Model_Training_Evaluation


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_CUSTOMERS    = 20       # number of customer records to process
RANDOM_SEED    = 42       # reproducible sampling from the test set
DEPOSIT_HIGH   = 0.70     # probability ≥ this → "high" tier
DEPOSIT_LOW    = 0.40     # probability < this → "low" tier

# (deposit_tier, cluster_label) → (action, reasoning)
RECOMMENDATION_TABLE = {
    ("high",   "High-Value Engaged"):
        ("Premium Savings Offer",
         "High conversion likelihood + above-average balance + engaged call "
         "history. Offer a tailored premium savings product with personal "
         "advisor support."),

    ("high",   "High-Value Passive"):
        ("Advisor Outreach",
         "High conversion probability but shorter engagement suggests the "
         "customer prefers a direct conversation. Assign a relationship "
         "manager for a personal call."),

    ("high",   "Budget Engaged"):
        ("Digital Savings Promotion",
         "High predicted conversion with strong call engagement but a "
         "below-average balance. A low-minimum digital savings product "
         "via mobile or email is the right fit."),

    ("high",   "Low-Engagement"):
        ("Fast-Track Deposit Offer",
         "High model confidence despite limited historical engagement. "
         "Act quickly with a simple, frictionless deposit offer."),

    ("medium", "High-Value Engaged"):
        ("Personalised Savings Consultation",
         "Affluent and engaged but undecided. A one-to-one consultation "
         "on long-term financial goals is likely to convert."),

    ("medium", "High-Value Passive"):
        ("Relationship Manager Follow-Up",
         "Above-average balance but sitting on the fence. A warm "
         "follow-up from a relationship manager should tip the balance."),

    ("medium", "Budget Engaged"):
        ("Loyalty Incentive",
         "Engaged but price-sensitive. A bonus interest rate for the "
         "first six months directly addresses the likely cost barrier."),

    ("medium", "Low-Engagement"):
        ("Re-Engagement Campaign",
         "Uncertain probability and low engagement. Run a multi-touch "
         "digital campaign (email + SMS) to rebuild interest before "
         "making a direct offer."),

    ("low",    "High-Value Engaged"):
        ("Nurture — Premium Content",
         "Affluent and engaged but not ready now. Maintain the "
         "relationship through tailored financial education content; "
         "re-score in 90 days."),

    ("low",    "High-Value Passive"):
        ("Nurture — Periodic Check-In",
         "High-balance customer, currently unlikely to subscribe. "
         "Preserve the relationship with a light-touch quarterly "
         "check-in."),

    ("low",    "Budget Engaged"):
        ("Nurture — Digital Newsletter",
         "Engaged profile but low current probability. Keep the "
         "customer warm with regular digital newsletter updates."),

    ("low",    "Low-Engagement"):
        ("Deprioritise — Monitor",
         "Low probability and low engagement. Minimal resource "
         "investment recommended; monitor for behavioural changes."),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_models(data_file):
    """Run both preprocessing pipelines and train both models.

    This function is the single entry point for the setup phase.  It
    calls the existing :class:`Preprocessing` and
    :class:`Model_Training_Evaluation` classes without duplicating any
    of their logic.

    Parameters
    ----------
    data_file : str
        Path to ``bank.csv``.

    Returns
    -------
    clf_model : RandomForestClassifier
        Fitted classifier returned by
        :meth:`Model_Training_Evaluation.classification_model`.
    t_best : float
        Optimal probability threshold from the precision-recall curve.
    agg_model : AgglomerativeClustering
        Fitted clustering model returned by
        :meth:`Model_Training_Evaluation.clustering_model`.
    X_train_clf, X_test_clf : pd.DataFrame
        Classification feature matrices (encoded, unscaled).
    y_train_clf, y_test_clf : pd.Series
        Binary classification targets.
    X_train_clust, X_test_clust : pd.DataFrame
        Clustering feature matrices (encoded + power-transformed + scaled).
    """
    prep    = Preprocessing()
    trainer = Model_Training_Evaluation()

    print("  [setup] Running classification preprocessing pipeline...")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = prep.pre_processing(
        "classification", data_file
    )

    print("  [setup] Running clustering preprocessing pipeline...")
    X_train_clust, X_test_clust = prep.pre_processing(
        "clustering", data_file
    )

    # classification_model() side-effects that are expected and harmless:
    #   · overwrites metrics.png   (confusion matrix at default 0.5 threshold)
    #   · calls plt.show()         (no-op under Agg backend)
    #   · prints y_pred array and metric values to stdout
    print("  [setup] Training Random Forest classifier "
          "(RandomizedSearchCV — may take a few minutes)...")
    clf_model, t_best = trainer.classification_model(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf
    )

    print("  [setup] Training Agglomerative Clustering model "
          "(24-configuration grid search)...")
    agg_model = trainer.clustering_model(X_train_clust, X_test_clust)

    return (clf_model, t_best, agg_model,
            X_train_clf, X_test_clf, y_train_clf, y_test_clf,
            X_train_clust, X_test_clust)


# ─────────────────────────────────────────────────────────────────────────────
#  Customer input sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_customer_inputs(X_test_clf, df_raw, n, seed):
    """Sample ``n`` customer records from the held-out classifier test set.

    The test set (``X_test_clf``) is the 20% of bank.csv that the Random
    Forest classifier was **never trained on**.  Sampling from it gives us
    realistic customer records that represent genuinely unseen inputs to
    the targeting system.

    Parameters
    ----------
    X_test_clf : pd.DataFrame
        Held-out encoded feature matrix from the classification pipeline.
        Its index maps back to ``df_raw``.
    df_raw : pd.DataFrame
        The original (unscaled, unencoded) bank.csv DataFrame — used to
        display readable customer attributes alongside predictions.
    n : int
        Number of customer records to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sampled_idx : pd.Index
        Original DataFrame indices of the sampled customers.
    customer_inputs_raw : pd.DataFrame
        Raw (human-readable) attributes for the sampled customers.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(X_test_clf.index, size=min(n, len(X_test_clf)),
                        replace=False)
    sampled_idx          = pd.Index(chosen)
    customer_inputs_raw  = df_raw.loc[sampled_idx].copy()
    return sampled_idx, customer_inputs_raw


# ─────────────────────────────────────────────────────────────────────────────
#  Clustering inference
# ─────────────────────────────────────────────────────────────────────────────

def assign_cluster(agg_model, X_train_clust, customer_clust_row):
    """Assign a single customer to a cluster via nearest-centroid lookup.

    :class:`~sklearn.cluster.AgglomerativeClustering` stores
    ``labels_`` (the training-data assignments) but has no ``predict()``
    method.  To assign a new customer, we compute the Euclidean distance
    from the customer's feature vector to each cluster centroid (the
    column-wise mean of training rows in that cluster) and return the
    closest cluster ID.

    Parameters
    ----------
    agg_model : AgglomerativeClustering
        Fitted model with ``labels_`` set.
    X_train_clust : pd.DataFrame
        The training feature matrix the model was fitted on.
    customer_clust_row : pd.Series or np.ndarray
        Single customer's scaled+encoded feature vector.

    Returns
    -------
    int
        Nearest cluster ID.
    """
    # Compute centroid for each cluster from training data
    train_df = X_train_clust.copy()
    train_df["_label"] = agg_model.labels_
    centroids = train_df.groupby("_label").mean()   # (n_clusters, n_features)

    vec = np.array(customer_clust_row).reshape(1, -1)
    dists = euclidean_distances(vec, centroids.values)
    return int(dists.argmin())


# ─────────────────────────────────────────────────────────────────────────────
#  Cluster characterisation
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_profiles(agg_model, X_train_clust, df_raw):
    """Build a human-interpretable profile for every cluster.

    Uses ``agg_model.labels_`` (training-data assignments) to group the
    original unscaled rows of ``df_raw`` and computes per-cluster summary
    statistics.

    Parameters
    ----------
    agg_model : AgglomerativeClustering
        Fitted model with ``labels_``.
    X_train_clust : pd.DataFrame
        Training feature matrix (same index as the training subset of
        ``df_raw``).
    df_raw : pd.DataFrame
        Full original dataset (unscaled, unencoded), indexed consistently
        with the preprocessing output.

    Returns
    -------
    profiles : dict
        ``{cluster_id (int): profile_dict}`` with keys
        ``size``, ``pct_total``, ``age_mean``, ``balance_mean``,
        ``duration_mean``, ``campaign_mean``, ``job_mode``,
        ``education_mode``, ``deposit_rate``, ``label``.
    global_stats : dict
        Population medians: ``median_balance``, ``median_duration``,
        ``median_deposit_rate``.
    """
    # Only use training rows (model was fitted on X_train_clust)
    train_raw = df_raw.loc[X_train_clust.index]
    labels    = agg_model.labels_
    n_total   = len(labels)

    global_stats = {
        "median_balance":      float(train_raw["balance"].median()),
        "median_duration":     float(train_raw["duration"].median()),
        "median_deposit_rate": float((train_raw["deposit"] == "yes").mean()),
    }

    profiles = {}
    for cid in sorted(np.unique(labels)):
        mask  = labels == cid
        rows  = train_raw.iloc[mask]
        size  = int(mask.sum())

        profile = {
            "cluster_id":     int(cid),
            "size":           size,
            "pct_total":      round(size / n_total * 100, 2),
            "age_mean":       round(float(rows["age"].mean()),      1),
            "balance_mean":   round(float(rows["balance"].mean()),  0),
            "duration_mean":  round(float(rows["duration"].mean()), 0),
            "campaign_mean":  round(float(rows["campaign"].mean()), 1),
            "job_mode":       str(rows["job"].mode().iloc[0])       if size > 0 else "unknown",
            "education_mode": str(rows["education"].mode().iloc[0]) if size > 0 else "unknown",
            "deposit_rate":   round(float((rows["deposit"] == "yes").mean()), 4),
        }
        profile["label"] = _assign_cluster_label(profile, global_stats)
        profiles[cid] = profile

    return profiles, global_stats


def _assign_cluster_label(profile, global_stats):
    """Derive a human-readable segment name from a cluster's statistics.

    Three binary dimensions relative to population medians determine the label:

    * balance  > median → "High-Value" else "Budget"
    * duration > median → "Engaged"
    * deposit_rate > median deposit rate → contributes to "Engaged"

    Parameters
    ----------
    profile : dict
        Single cluster profile from :func:`build_cluster_profiles`.
    global_stats : dict
        Population medians from :func:`build_cluster_profiles`.

    Returns
    -------
    str
        One of ``'High-Value Engaged'``, ``'High-Value Passive'``,
        ``'Budget Engaged'``, ``'Low-Engagement'``.
    """
    affluent   = profile["balance_mean"]  > global_stats["median_balance"]
    engaged    = profile["duration_mean"] > global_stats["median_duration"]
    responsive = profile["deposit_rate"]  > global_stats["median_deposit_rate"]

    if affluent and engaged:
        return "High-Value Engaged"
    if affluent:
        return "High-Value Passive"
    if engaged or responsive:
        return "Budget Engaged"
    return "Low-Engagement"


# ─────────────────────────────────────────────────────────────────────────────
#  Recommendation logic
# ─────────────────────────────────────────────────────────────────────────────

def _deposit_tier(probability):
    """Convert a probability to a named tier string."""
    if probability >= DEPOSIT_HIGH:
        return "high"
    if probability >= DEPOSIT_LOW:
        return "medium"
    return "low"


def lookup_recommendation(deposit_prob, cluster_label):
    """Return the recommended marketing action for a customer.

    Parameters
    ----------
    deposit_prob : float
        Predicted deposit subscription probability (0–1).
    cluster_label : str
        Segment label from :func:`_assign_cluster_label`.

    Returns
    -------
    action : str
        Short name for the recommended action.
    reasoning : str
        Explanation tied to the customer's probability tier and segment.
    """
    tier = _deposit_tier(deposit_prob)
    return RECOMMENDATION_TABLE.get(
        (tier, cluster_label),
        ("General Savings Offer",
         f"Standard savings offer based on {tier} deposit probability.")
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-customer inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(customer_idx, raw_record,
                  clf_model, t_best,
                  agg_model, X_train_clust,
                  X_clf_lookup, X_clust_lookup,
                  cluster_profiles):
    """Run the full inference pipeline for a single customer.

    This is the core function of the targeting system.  It takes one
    customer, retrieves their correctly preprocessed feature vectors, runs
    both models, and assembles a recommendation.

    Parameters
    ----------
    customer_idx : int
        Original DataFrame index (customer ID).
    raw_record : pd.Series
        Unscaled, unencoded customer attributes for display.
    clf_model : RandomForestClassifier
        Fitted classification model from :func:`setup_models`.
    t_best : float
        Optimal probability threshold from :func:`setup_models`.
    agg_model : AgglomerativeClustering
        Fitted clustering model from :func:`setup_models`.
    X_train_clust : pd.DataFrame
        Training feature matrix used to fit ``agg_model`` (needed for
        centroid computation).
    X_clf_lookup : pd.DataFrame
        Full encoded classification feature matrix (train + test) indexed
        by original DataFrame index.
    X_clust_lookup : pd.DataFrame
        Full encoded/scaled clustering feature matrix (train + test).
    cluster_profiles : dict
        ``{cluster_id: profile_dict}`` from :func:`build_cluster_profiles`.

    Returns
    -------
    dict
        Keys: ``customer_id``, ``deposit_probability``, ``predicted_class``,
        ``deposit_tier``, ``cluster_id``, ``cluster_label``,
        ``recommended_action``, ``reasoning``,
        plus all raw feature values (age, job, balance, …).
    """
    # ── Classification inference ─────────────────────────────────────────────
    # Retrieve the customer's encoded feature vector (fitted on training data)
    x_clf  = X_clf_lookup.loc[[customer_idx]]          # keep 2-D for predict_proba
    prob   = float(clf_model.predict_proba(x_clf)[0, 1])
    pred   = int(prob >= t_best)

    # ── Clustering inference ──────────────────────────────────────────────────
    # Retrieve the customer's scaled+encoded feature vector
    x_clust    = X_clust_lookup.loc[customer_idx]       # pd.Series (1-D)
    cluster_id = assign_cluster(agg_model, X_train_clust, x_clust)

    # ── Recommendation ───────────────────────────────────────────────────────
    profile = cluster_profiles.get(cluster_id, {})
    label   = profile.get("label", "Unknown")
    action, reasoning = lookup_recommendation(prob, label)

    result = {
        "customer_id":        int(customer_idx),
        # Raw input fields (for display / report)
        "age":                int(raw_record.get("age",       0)),
        "job":                str(raw_record.get("job",       "")),
        "marital":            str(raw_record.get("marital",   "")),
        "education":          str(raw_record.get("education", "")),
        "balance":            int(raw_record.get("balance",   0)),
        "housing_loan":       str(raw_record.get("housing",   "")),
        "personal_loan":      str(raw_record.get("loan",      "")),
        "last_contact_month": str(raw_record.get("month",     "")),
        "call_duration_sec":  int(raw_record.get("duration",  0)),
        "num_campaigns":      int(raw_record.get("campaign",  0)),
        "prev_outcome":       str(raw_record.get("poutcome",  "")),
        # Model outputs
        "deposit_probability": round(prob, 4),
        "predicted_class":     pred,
        "deposit_tier":        _deposit_tier(prob),
        "cluster_id":          cluster_id,
        "cluster_label":       label,
        # Recommendation
        "recommended_action":  action,
        "reasoning":           reasoning,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_deposit_prob_by_cluster(report_df, cluster_profiles, output_dir):
    """Box plot: deposit probability distribution per customer segment.

    Dashed lines mark the high/low probability tier thresholds.

    Output: ``{output_dir}/deposit_probability_by_cluster.png``
    """
    cids   = sorted(report_df["cluster_id"].unique())
    groups = [
        report_df.loc[report_df["cluster_id"] == cid,
                      "deposit_probability"].values
        for cid in cids
    ]
    x_labels = [
        f"Cluster {cid}\n{cluster_profiles[cid]['label']}\n"
        f"(n={report_df['cluster_id'].eq(cid).sum()})"
        for cid in cids
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(cids) * 2.5), 6))
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})

    palette = plt.cm.tab10.colors
    for patch, colour in zip(bp["boxes"], palette):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.axhline(DEPOSIT_HIGH, color="green",  linestyle="--", linewidth=1.2,
               label=f"High tier  ≥ {DEPOSIT_HIGH:.0%}")
    ax.axhline(DEPOSIT_LOW,  color="orange", linestyle="--", linewidth=1.2,
               label=f"Low tier   < {DEPOSIT_LOW:.0%}")

    ax.set_xticks(range(1, len(cids) + 1))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Predicted Deposit Probability")
    ax.set_xlabel("Customer Segment")
    ax.set_title("Deposit Probability by Customer Segment")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, "deposit_probability_by_cluster.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cluster_distribution(report_df, cluster_profiles, output_dir):
    """Bar chart: number of sampled customers in each segment.

    Output: ``{output_dir}/cluster_distribution.png``
    """
    cids   = sorted(report_df["cluster_id"].unique())
    counts = [report_df["cluster_id"].eq(cid).sum() for cid in cids]
    labels = [
        f"Cluster {cid}\n{cluster_profiles[cid]['label']}"
        for cid in cids
    ]
    total  = sum(counts)

    fig, ax = plt.subplots(figsize=(max(7, len(cids) * 2), 5))
    palette = plt.cm.tab10.colors
    bars = ax.bar(
        labels, counts,
        color=[palette[i % len(palette)] for i in range(len(cids))],
        edgecolor="black", linewidth=0.7, alpha=0.85,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{count}  ({count / total * 100:.0f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Number of Customers")
    ax.set_title(f"Customer Distribution Across Segments  (N = {total})")
    ax.set_ylim(0, max(counts) * 1.20)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_recommendation_distribution(report_df, output_dir):
    """Horizontal bar chart: frequency of each recommended action.

    Output: ``{output_dir}/recommendation_distribution.png``
    """
    counts = (
        report_df["recommended_action"]
        .value_counts()
        .sort_values(ascending=True)
    )
    total  = counts.sum()
    colors = plt.cm.Paired.colors

    fig, ax = plt.subplots(figsize=(11, max(4, len(counts) * 0.8)))
    bars = ax.barh(
        counts.index, counts.values,
        color=[colors[i % len(colors)] for i in range(len(counts))],
        edgecolor="black", linewidth=0.6, alpha=0.85,
    )
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count}  ({count / total * 100:.0f}%)",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Number of Customers")
    ax.set_title(f"Marketing Recommendation Distribution  (N = {total})")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, "recommendation_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Display helper
# ─────────────────────────────────────────────────────────────────────────────

def print_customer_result(result, idx, total):
    """Pretty-print a single customer's inference result to stdout."""
    bar = "─" * 72
    print(f"\n{bar}")
    print(f"  Customer {idx + 1}/{total}   (ID: {result['customer_id']})")
    print(bar)
    print(f"  INPUT  │ Age: {result['age']}  |  Job: {result['job']}"
          f"  |  Education: {result['education']}")
    print(f"         │ Marital: {result['marital']}"
          f"  |  Balance: £{result['balance']:,}"
          f"  |  Housing loan: {result['housing_loan']}")
    print(f"         │ Duration: {result['call_duration_sec']} s"
          f"  |  Campaigns: {result['num_campaigns']}"
          f"  |  Prev. outcome: {result['prev_outcome']}")
    print(bar)
    tier_icon = {"high": "●", "medium": "◑", "low": "○"}
    print(f"  RESULT │ Deposit probability : "
          f"{result['deposit_probability']:.4f}  "
          f"[{result['deposit_tier'].upper()}]  "
          f"{tier_icon.get(result['deposit_tier'], '')}")
    print(f"         │ Predicted class     : "
          f"{'WILL SUBSCRIBE' if result['predicted_class'] else 'WILL NOT SUBSCRIBE'}")
    print(f"         │ Cluster             : "
          f"{result['cluster_id']} — {result['cluster_label']}")
    print(bar)
    print(f"  ACTION │ {result['recommended_action']}")
    print(f"  WHY    │ {result['reasoning']}")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    DATA_FILE = "bank.csv"

    # ── Phase 1: Setup ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CUSTOMER TARGETING SYSTEM — SETUP PHASE")
    print("=" * 72)
    print("  Loading bank.csv and training models via Model_Training_Evaluation...")

    (clf_model, t_best, agg_model,
     X_train_clf, X_test_clf, y_train_clf, y_test_clf,
     X_train_clust, X_test_clust) = setup_models(DATA_FILE)

    # Load the full raw dataset for raw-record display and cluster profiling
    df_raw = pd.read_csv(DATA_FILE).drop_duplicates().dropna().reset_index(drop=True)

    # Reindex df_raw to match the original integer index used by the preprocessing
    # pipeline (train_test_split preserves the source DataFrame's integer index)
    df_raw.index = df_raw.index   # already 0-based after reset_index

    # Build cluster profiles from training data
    print("\n  Building cluster profiles from training data...")
    cluster_profiles, global_stats = build_cluster_profiles(
        agg_model, X_train_clust, df_raw
    )
    print(f"  Clusters found: {len(cluster_profiles)}")
    for cid, p in sorted(cluster_profiles.items()):
        print(f"    Cluster {cid}: {p['label']}"
              f"  (n={p['size']:,}, deposit rate={p['deposit_rate']:.1%},"
              f" avg balance=£{p['balance_mean']:,.0f})")

    # Pre-build lookup tables that align original index to preprocessed vectors
    X_clf_lookup   = pd.concat([X_train_clf,   X_test_clf])
    X_clust_lookup = pd.concat([X_train_clust, X_test_clust])

    # ── Phase 2: Sample customer inputs ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CUSTOMER TARGETING SYSTEM — SAMPLING CUSTOMER INPUTS")
    print("=" * 72)
    print(f"  Sampling {N_CUSTOMERS} customers from the held-out test set")
    print("  (these customers were NOT used to train the classifier)")

    sampled_idx, customer_inputs_raw = sample_customer_inputs(
        X_test_clf, df_raw, N_CUSTOMERS, RANDOM_SEED
    )
    print(f"  Sampled {len(sampled_idx)} customers  "
          f"(indices: {list(sampled_idx[:5])} ...)")

    # ── Phase 3: Inference ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CUSTOMER TARGETING SYSTEM — INFERENCE PHASE")
    print("=" * 72)
    print(f"  Running inference pipeline for {len(sampled_idx)} customers...\n")

    results = []
    for i, customer_idx in enumerate(sampled_idx):
        raw_record = df_raw.loc[customer_idx]
        result = run_inference(
            customer_idx    = customer_idx,
            raw_record      = raw_record,
            clf_model       = clf_model,
            t_best          = t_best,
            agg_model       = agg_model,
            X_train_clust   = X_train_clust,
            X_clf_lookup    = X_clf_lookup,
            X_clust_lookup  = X_clust_lookup,
            cluster_profiles= cluster_profiles,
        )
        results.append(result)
        print_customer_result(result, i, len(sampled_idx))

    # ── Phase 4: Outputs ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CUSTOMER TARGETING SYSTEM — SAVING OUTPUTS")
    print("=" * 72)

    # Targeting report CSV
    report_df   = pd.DataFrame(results)
    report_path = os.path.join("results", "targeting_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"  Saved: {report_path}  ({len(report_df)} rows)")

    # Cluster profiles JSON
    profiles_path = os.path.join("results", "cluster_profiles.json")
    with open(profiles_path, "w") as fh:
        json.dump(
            {str(k): {ky: v for ky, v in p.items() if ky != "cluster_id"}
             for k, p in cluster_profiles.items()},
            fh, indent=2,
        )
    print(f"  Saved: {profiles_path}")

    # Plots
    print("\n  Generating plots...")
    plot_deposit_prob_by_cluster(report_df, cluster_profiles, "results")
    plot_cluster_distribution(report_df, cluster_profiles, "results")
    plot_recommendation_distribution(report_df, "results")

    # Summary
    print("\n" + "=" * 72)
    print("  RECOMMENDATION SUMMARY")
    print("=" * 72)
    total = len(report_df)
    for action, count in report_df["recommended_action"].value_counts().items():
        print(f"  {action:<44} {count:>3}  ({count / total * 100:.0f}%)")
    print("=" * 72)
    print(f"\n  {total} customers processed.  "
          f"Predicted subscribers: "
          f"{report_df['predicted_class'].sum()} "
          f"({report_df['predicted_class'].mean():.0%})")
    print(f"  Outputs saved to: results/\n")
