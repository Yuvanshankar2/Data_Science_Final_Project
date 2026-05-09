"""
Model Training and Evaluation

Defines class:`Model_Training_Evaluation`, which trains and evaluates
two advanced ML models on the bank-marketing dataset:

Classification:
    This class trains a 
    RandomForestClassifier with RandomizedSearchCV over
    ``n_estimators``, ``max_depth``, and ``min_samples_split``.  The
    optimal probability threshold is derived from the precision-recall
    curve.

Clustering:
    This class exhaustively searches over cluster counts (2–9) and linkage methods
    (complete, average, single) for Hierarchical Clustering, selecting the
    combination with the highest silhouette score.

Run standalone
::

    python Model_Training_Evaluation.py

Output

* Confusion matrix saved to ``metrics.png`` (classification).
* Optimal silhouette score printed to stdout (clustering).

"""

from Preprocessing import Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering


class Model_Training_Evaluation():
    """Train and evaluate advanced classification and clustering models.

    This class acts as the primary model-training interface for the
    bank-marketing project.  It wraps hyperparameter search, threshold
    optimisation, and evaluation into two high-level methods.
    """

    def __init__(self):
        """Initialise the Model_Training_Evaluation helper.

        Prints a confirmation message to stdout on instantiation.
        """
        print("Welcome to model")

    #  Classification

    def classification_model(self, X_train, X_test, y_train, y_test):
        """Train a Random Forest classifier with hyperparameter search.

        Uses RandomizedSearchCV over three hyperparameters with 5-fold
        StratifiedKFold cross-validation.  After training, the probability threshold is
        tuned on the test set by finding the point on the precision-recall
        curve that maximises ``precision × recall``.

        Parameters

        X_train : pd.DataFrame
            Encoded training feature matrix from
            Preprocessing.classification_processing.
        X_test : pd.DataFrame
            Encoded test feature matrix.
        y_train : pd.Series
            Binary training labels (0 = no deposit, 1 = deposit).
        y_test : pd.Series
            Binary test labels.

        Returns

        final_model : RandomForestClassifier
            The best estimator selected by RandomizedSearchCV.
        t_best : float
            Optimal probability threshold derived from the
            precision-recall curve (maximises precision × recall).

        Side effects

        Saves a confusion matrix plot to ``metrics.png`` at 300 DPI.
        """

        model = RandomForestClassifier(class_weight="balanced")
        hyperparameter_tune = RandomizedSearchCV(
            model,
            param_distributions={
                "n_estimators": range(50, 200),
                "max_depth": range(5, 55),
                "min_samples_split": range(2, 100),
            },
            cv=StratifiedKFold(n_splits=5),
        )
        hyperparameter_tune.fit(X_train, y_train)
        final_model = hyperparameter_tune.best_estimator_

        # Evaluate the model. Display the precision recall curve.
        y_pred = final_model.predict_proba(X_test)[:, 1]
        precision, recall, threshold_list = precision_recall_curve(y_test, y_pred)

        # Find the best possible threshold to use.

        idx = np.argmax(precision * recall)
        t_best = threshold_list[idx]

        # Apply the optimal threshold to convert probabilities to hard labels
        y_pred = (y_pred >= t_best).astype(int)
        print(y_pred)

        # Compute the metrics
        ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test)
        plt.savefig("metrics.png", dpi=300, bbox_inches="tight")
        plt.show()

        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(precision * 100)
        print(recall * 100)
        print(accuracy_score(y_test, y_pred))

        # Compare the results to the baseline classification model metrics

        # Use SHAP values to explain the predictions. Display a swarm plot.

        # Return the model, shap explainer, and threshold
        return final_model, t_best

    #  Clustering

    def clustering_model(self, X_train, X_test):
        """Fit Agglomerative Clustering with exhaustive linkage and cluster search.

        This function tests all combinations of ``n_clusters`` ∈ [2, 9] and linkage
        methods ``{'complete', 'average', 'single'}`` — 24 combinations
        in total.  Selects the combination with the highest silhouette
        score.

        Agglomerative Clustering is a hierarchical bottom-up approach:
        it starts with each sample as its own cluster and iteratively
        merges the closest pair.  Unlike K-Means it does not assume
        spherical clusters and requires no centroid management.

        Parameters

        X_train : pd.DataFrame
            Encoded and scaled training feature matrix from
            Preprocessing.clustering_processing.
        X_test : pd.DataFrame
            Encoded and scaled test feature matrix (unused in fitting,
            kept for API consistency).

        Returns

        optimal_model : AgglomerativeClustering
            Best fitted clustering model by silhouette score.

        Prints

        Optimal silhouette score and optimal number of clusters to stdout.
        """
        # Define the clustering model. Use Hierarchical Clustering.
        methods = ["complete", "average", "single"]

        # Find the best parameters for this model iteratively
        optimal_model = None
        num_clusters = -1
        optimal_score = -1

        for i in range(2, 10):
            for j in methods:
                cluster_model = AgglomerativeClustering(n_clusters=i, linkage=j)
                result = cluster_model.fit_predict(X_train)

                # Get the silhouette score — higher means better-defined clusters
                model_score = silhouette_score(X_train, result)
                if model_score > optimal_score:
                    optimal_score = model_score
                    optimal_model = cluster_model
                    num_clusters = i

        # Compare the results to the baseline clustering model metrics
        print("Optimal score: ", optimal_score)
        print("Optimal number of clusters: ", num_clusters)

        # Return the model
        return optimal_model


# Main                                               

if __name__ == "__main__":
    processor = Preprocessing()
    X_train, X_test = processor.pre_processing("clustering", "bank.csv")

    trainer = Model_Training_Evaluation()
    model = trainer.clustering_model(X_train, X_test)
