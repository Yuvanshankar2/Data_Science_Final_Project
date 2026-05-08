import numpy as np
from sklearn.cluster import KMeans
from Preprocessing import Preprocessing
from sklearn.metrics import silhouette_score

best_kmeans_score = 0

best_kmeans_model = None


preprocessor = Preprocessing()
X_train,X_test = preprocessor.pre_processing("clustering","bank.csv")

for i in range(2,10):
    cluster_model = KMeans(n_clusters=i,n_init="auto")
    results = cluster_model.fit_predict(X_train)
    k_means_score = silhouette_score(X_train,results)
    if(k_means_score>best_kmeans_score):
        best_kmeans_score =k_means_score
        best_kmeans_model = cluster_model


print("Baseline Score: ",best_kmeans_score)

