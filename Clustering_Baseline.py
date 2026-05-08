"""
Clustering Baseline
===================
Runs K-Means clustering on the bank-marketing dataset across a range of
cluster counts (k = 2 … 9) to establish a silhouette-score baseline for
comparison with more advanced clustering algorithms
(e.g. Agglomerative Clustering in ``Model_Training_Evaluation.py``).

Algorithm: K-Means
------------------
K-Means partitions data by iteratively assigning each sample to its
nearest centroid and recomputing centroids.  It is fast and scalable but
assumes spherical, equally-sized clusters and is sensitive to
initialisation.

Hyperparameter search
---------------------
The optimal number of clusters ``k`` is selected by maximising the
silhouette score over the range [2, 9].  A higher silhouette score
(closer to 1.0) indicates well-separated, compact clusters.

Run standalone
--------------
::

    python Clustering_Baseline.py

Output
------
Best silhouette score across all tested cluster counts printed to stdout.
"""

import numpy as np
from sklearn.cluster import KMeans
from Preprocessing import Preprocessing
from sklearn.metrics import silhouette_score


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Data loading & preprocessing                                       #
    # ------------------------------------------------------------------ #
    best_kmeans_score = 0   # Track the highest silhouette score found
    best_kmeans_model = None  # Keep the best fitted model for downstream use

    preprocessor = Preprocessing()
    X_train, X_test = preprocessor.pre_processing("clustering", "bank.csv")

    # ------------------------------------------------------------------ #
    #  Hyperparameter sweep: find optimal k                               #
    # ------------------------------------------------------------------ #
    for i in range(2, 10):
        # Fit K-Means with k = i clusters; n_init="auto" lets sklearn
        # choose an appropriate number of random initialisations
        cluster_model = KMeans(n_clusters=i, n_init="auto")
        results = cluster_model.fit_predict(X_train)

        # Silhouette score measures cluster cohesion vs separation
        k_means_score = silhouette_score(X_train, results)

        # Retain the configuration with the highest silhouette score
        if k_means_score > best_kmeans_score:
            best_kmeans_score = k_means_score
            best_kmeans_model = cluster_model

    print("Baseline Score: ", best_kmeans_score)
