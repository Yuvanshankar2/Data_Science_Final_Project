"""
evaluate_classification.py

Standalone evaluation script that trains and compares two classifiers on
the bank-marketing dataset:

* **Logistic Regression** (baseline) — replicates ``Classification_Baseline.py``
  exactly: StandardScaler + default LogisticRegression, evaluated with
  hard ``predict()`` labels.

* **Random Forest** (advanced) — replicates the training logic in
  ``Model_Training_Evaluation.classification_model``: RandomForest with
  ``class_weight='balanced'``, RandomizedSearchCV over n_estimators /
  max_depth / min_samples_split, StratifiedKFold(5), and an optimal
  probability threshold derived from the precision-recall curve.

Generated outputs

plots/
    confusion_matrices.png
        Side-by-side confusion matrices for both classifiers.
    classification_metrics_comparison.png
        Grouped bar chart comparing Accuracy, Precision, Recall, F1.
    roc_curves.png
        ROC curves for both classifiers on the same axes with AUC annotations.

metrics/
    classification_metrics.csv
        Wide-format table, one row per model, with all metric values.
    classification_metrics.json
        Nested dict ``{model_name: {metric: value}}``.

Console
    Formatted comparison table showing metric values and % improvements.

Run

    python evaluate_classification.py

Notes

* ``matplotlib.use("Agg")`` is set at the top of this module so plots
  can be generated in headless (no-display) environments.
* This script does **not** import ``Classification_Baseline.py`` or
  ``Model_Training_Evaluation.py`` to avoid triggering their module-level
  training code.  The model pipelines are replicated inline.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — must come before pyplot import
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

from Preprocessing import Preprocessing


#  Model builders                                                     

def build_logistic_regression_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a Logistic Regression baseline classifier.

    Replicates ``Classification_Baseline.py`` exactly: applies
    :class:`~sklearn.preprocessing.StandardScaler` to numerical features
    and fits a default :class:`~sklearn.linear_model.LogisticRegression`
    with no hyperparameter tuning.

    Parameters
    
    X_train : pd.DataFrame
        Encoded training feature matrix.
    X_test : pd.DataFrame
        Encoded test feature matrix.
    y_train : pd.Series
        Binary training labels.
    y_test : pd.Series
        Binary test labels.

    Returns
    
    dict
        Keys: ``model``, ``scaler``, ``y_pred``, ``y_prob``,
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
    """
    # Scale numerical features — required for Logistic Regression
    scaler = StandardScaler()
    numerical_columns = X_train.select_dtypes(include="number").columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Fit default Logistic Regression — no hyperparameter tuning (baseline)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    # predict_proba[:, 1] gives the probability of the positive class
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def build_random_forest_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a Random Forest classifier with hyperparameter search.

    Replicates ``Model_Training_Evaluation.classification_model`` exactly:
    uses RandomizedSearchCV over ``n_estimators``, ``max_depth``, and ``min_samples_split`` with
    5-fold StratifiedKFold
    cross-validation.  The optimal probability threshold is derived by
    maximising ``precision × recall`` on the precision-recall curve.

    Parameters
    
    X_train : pd.DataFrame
        Encoded training feature matrix (no scaling needed for RF).
    X_test : pd.DataFrame
        Encoded test feature matrix.
    y_train : pd.Series
        Binary training labels.
    y_test : pd.Series
        Binary test labels.

    Returns
    
    dict
        Keys: ``model``, ``t_best``, ``y_pred``, ``y_prob``,
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
    """
    # class_weight="balanced" compensates for class imbalance in deposit target
    base_model = RandomForestClassifier(class_weight="balanced")

    hyperparameter_tune = RandomizedSearchCV(
        base_model,
        param_distributions={
            "n_estimators": range(50, 200),
            "max_depth": range(5, 55),
            "min_samples_split": range(2, 100),
        },
        cv=StratifiedKFold(n_splits=5),
    )
    hyperparameter_tune.fit(X_train, y_train)
    final_model = hyperparameter_tune.best_estimator_

    # Get class probabilities for the positive class (deposit = yes)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    # Select threshold that maximises precision × recall simultaneously
    precision_vals, recall_vals, threshold_list = precision_recall_curve(y_test, y_prob)
    idx = np.argmax(precision_vals * recall_vals)
    t_best = threshold_list[idx]

    # Convert probabilities to hard labels using the optimal threshold
    y_pred = (y_prob >= t_best).astype(int)

    return {
        "model": final_model,
        "t_best": t_best,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


# ------------------------------------------------------------------ #
#  Plot generators                                                    #
# ------------------------------------------------------------------ #

def plot_confusion_matrices(lr_result, rf_result, y_test, output_dir="plots"):
    """Save side-by-side confusion matrices for both classifiers.

    Uses :class:`~sklearn.metrics.ConfusionMatrixDisplay.from_predictions`
    for both models so the same scaled test set is not required to be
    re-passed through each model.

    Parameters
    
    lr_result : dict
        Output of :func:`build_logistic_regression_model`.
    rf_result : dict
        Output of :func:`build_random_forest_model`.
    y_test : pd.Series
        Ground-truth binary test labels.
    output_dir : str
        Directory to save the figure.  Created if it does not exist.

    Output
    ``{output_dir}/confusion_matrices.png``
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices — Baseline vs Advanced Classifier", fontsize=14)

    # Logistic Regression confusion matrix (left subplot)
    ConfusionMatrixDisplay.from_predictions(
        y_test, lr_result["y_pred"],
        display_labels=["No Deposit", "Deposit"],
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Logistic Regression (Baseline)")

    # Random Forest confusion matrix (right subplot)
    ConfusionMatrixDisplay.from_predictions(
        y_test, rf_result["y_pred"],
        display_labels=["No Deposit", "Deposit"],
        colorbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Random Forest (Advanced)")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_bar_chart(lr_result, rf_result, output_dir="plots"):
    """Save a grouped bar chart comparing Accuracy, Precision, Recall, and F1.

    Parameters
    
    lr_result : dict
        Output of :func:`build_logistic_regression_model`.
    rf_result : dict
        Output of :func:`build_random_forest_model`.
    output_dir : str
        Directory to save the figure.

    Output
    ``{output_dir}/classification_metrics_comparison.png``
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    lr_scores = [lr_result[m] for m in metrics]
    rf_scores = [rf_result[m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35   # width of each bar in the grouped pair

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_lr = ax.bar(x - width / 2, lr_scores, width, label="Logistic Regression (Baseline)", color="steelblue")
    bars_rf = ax.bar(x + width / 2, rf_scores, width, label="Random Forest (Advanced)", color="darkorange")

    # Annotate each bar with its exact value
    for bar in bars_lr:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    for bar in bars_rf:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (0 – 1)")
    ax.set_title("Classification Metrics — Baseline vs Advanced")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "classification_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(lr_result, rf_result, y_test, output_dir="plots"):
    """Save ROC curves for both classifiers on the same axes.

    Both curves use raw predicted probabilities (not threshold-adjusted
    labels) for a fair AUC comparison.  A diagonal dashed line shows the
    performance of a random classifier.

    Parameters
    
    lr_result : dict
        Output of :func:`build_logistic_regression_model`.
    rf_result : dict
        Output of :func:`build_random_forest_model`.
    y_test : pd.Series
        Ground-truth binary test labels.
    output_dir : str
        Directory to save the figure.

    Output
    ``{output_dir}/roc_curves.png``
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute ROC curve points for Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_result["y_prob"])
    ax.plot(fpr_lr, tpr_lr, color="steelblue", lw=2,
            label=f"Logistic Regression (AUC = {lr_result['roc_auc']:.3f})")

    # Compute ROC curve points for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_result["y_prob"])
    ax.plot(fpr_rf, tpr_rf, color="darkorange", lw=2,
            label=f"Random Forest (AUC = {rf_result['roc_auc']:.3f})")

    # Diagonal reference line represents a random classifier (AUC = 0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Baseline vs Advanced Classifier")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
#  Metric persistence                                                 #
# ------------------------------------------------------------------ #

def save_metrics(lr_result, rf_result, output_dir="metrics"):
    """Save classification metrics to CSV and JSON.

    Parameters
    
    lr_result : dict
        Output of :func:`build_logistic_regression_model`.
    rf_result : dict
        Output of :func:`build_random_forest_model`.
    output_dir : str
        Directory to save metric files.  Created if it does not exist.

    Outputs
    ``{output_dir}/classification_metrics.csv``
        Wide-format table, one row per model.
    ``{output_dir}/classification_metrics.json``
        Nested dict ``{model_name: {metric: value}}``.
    """
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Build a row for each model
    rows = [
        {"model": "Logistic Regression (Baseline)", **{k: lr_result[k] for k in metric_keys}},
        {"model": "Random Forest (Advanced)",       **{k: rf_result[k] for k in metric_keys}},
    ]
    df_metrics = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "classification_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Also save as JSON for easy programmatic consumption by generate_summary.py
    json_data = {
        "Logistic Regression": {k: lr_result[k] for k in metric_keys},
        "Random Forest":       {k: rf_result[k] for k in metric_keys},
    }
    json_path = os.path.join(output_dir, "classification_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")



def print_comparison_table(lr_result, rf_result):
    """Print a formatted metric comparison table to stdout.

    Displays Accuracy, Precision, Recall, F1, and ROC-AUC side-by-side
    for both models, along with the percentage improvement of the
    Random Forest over the Logistic Regression baseline.

    Parameters
    
    lr_result : dict
        Output of :func:`build_logistic_regression_model`.
    rf_result : dict
        Output of :func:`build_random_forest_model`.
    """
    metrics = [
        ("Accuracy",  "accuracy"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1 Score",  "f1"),
        ("ROC-AUC",   "roc_auc"),
    ]

    header = f"{'Metric':<12} {'Logistic Regression':>22} {'Random Forest':>16} {'Improvement':>14}"
    print("\n" + "=" * len(header))
    print("  Classification Comparison: Baseline vs Advanced")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, key in metrics:
        lr_val = lr_result[key]
        rf_val = rf_result[key]
        # Percentage improvement relative to the baseline value
        improvement = ((rf_val - lr_val) / lr_val) * 100 if lr_val != 0 else float("nan")
        sign = "+" if improvement >= 0 else ""
        print(
            f"{label:<12} {lr_val:>22.4f} {rf_val:>16.4f} "
            f"{sign}{improvement:>12.2f}%"
        )

    print("=" * len(header) + "\n")


#Main
if __name__ == "__main__":
    # Create output directories if they do not exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    print("\n[1/5] Loading and preprocessing data...")
    prep = Preprocessing()
    X_train, X_test, y_train, y_test = prep.pre_processing("classification", "bank.csv")

    print("\n[2/5] Training Logistic Regression (baseline)...")
    lr_result = build_logistic_regression_model(X_train, X_test, y_train, y_test)
    print(f"       LR accuracy: {lr_result['accuracy']:.4f}")

    print("\n[3/5] Training Random Forest (advanced — this may take a few minutes)...")
    rf_result = build_random_forest_model(X_train, X_test, y_train, y_test)
    print(f"       RF accuracy: {rf_result['accuracy']:.4f}")

    print("\n[4/5] Generating plots...")
    plot_confusion_matrices(lr_result, rf_result, y_test)
    plot_metrics_bar_chart(lr_result, rf_result)
    plot_roc_curves(lr_result, rf_result, y_test)

    print("\n[5/5] Saving metrics...")
    save_metrics(lr_result, rf_result)

    print_comparison_table(lr_result, rf_result)
    print("Classification evaluation complete.")
