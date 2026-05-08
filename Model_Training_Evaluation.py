from Preprocessing import Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve,recall_score,precision_score,accuracy_score,f1_score,ConfusionMatrixDisplay,silhouette_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering

class Model_Training_Evaluation():
    def __init__(self):
        print("Welcome to model")
    

    def classification_model(self,X_train,X_test,y_train,y_test):
        # Apply hyperparameter tuning to the model. Use the RandomSearch method
        model = RandomForestClassifier(class_weight="balanced")
        hyperparameter_tune = RandomizedSearchCV(model,param_distributions={"n_estimators": range(50,200),"max_depth": range(5,55),"min_samples_split": range(2,100)},cv=StratifiedKFold(n_splits=5))
        hyperparameter_tune.fit(X_train,y_train)
        final_model  = hyperparameter_tune.best_estimator_
        
        # Evaluate the model. Display the precision recall curve
        y_pred = final_model.predict_proba(X_test)[:,1]
        precision,recall,threshold_list = precision_recall_curve(y_test,y_pred)
        # Find the best possible threshold to use.
        idx = np.argmax(precision*recall)
        t_best = threshold_list[idx]

        y_pred = (y_pred>=t_best).astype(int)
        print(y_pred)
        
        # Compute the metrics
        ConfusionMatrixDisplay.from_estimator(final_model,X_test,y_test)
        plt.savefig("metrics.png",dpi=300,bbox_inches="tight")
        plt.show()
        precision = precision_score(y_test,y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)

        print(precision*100)
        print(recall*100)
        print(accuracy_score(y_test,y_pred))

        # Compare the results to the baseline classification model metrics

        # Use SHAP values to explain the predictions. Display a swarm plot.

        # Return the model, shap explainer, and threshold
        return final_model,t_best

    def clustering_model(self,X_train,X_test):
        # Define the clustering model. Use Hierarchical Clustering.
        methods = ["complete", "average", "single"]
        # Find the best parameters for this model iteratively
        optimal_model = None
        num_clusters = -1
        optimal_score=-1
        for i in range(2,10):
            for j in methods:
                cluster_model = AgglomerativeClustering(n_clusters=i,linkage=j)
                result = cluster_model.fit_predict(X_train)
                
                # Get the silhoutte score
                model_score = silhouette_score(X_train,result)
                if(model_score>optimal_score):
                    optimal_score = model_score
                    optimal_model = cluster_model
                    num_clusters = i

        # Compare the results to the baseline clustering model metrics
        print("Optimal score: ",optimal_score)
        print("Optimal number of clusters: ",num_clusters)
        # Return the model
        return optimal_model
    

processor = Preprocessing()
X_train,X_test = processor.pre_processing("clustering","bank.csv")
    
trainer = Model_Training_Evaluation()
model = trainer.clustering_model(X_train,X_test)