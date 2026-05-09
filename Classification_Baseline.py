"""
Classification Baseline

Trains a Logistic Regression classifier on the bank-marketing dataset
as a performance baseline for comparison with more advanced models
(e.g. Random Forest in ``Model_Training_Evaluation.py``).

Pipeline
--------
1. Load and preprocess data via :`Preprocessing.classification_processing`.
2. Apply :`StandardScaler` to all numerical
   features (required by Logistic Regression for convergence).
3. Fit a default :`LogisticRegression` with
   no hyperparameter tuning — this establishes a simple, unoptimised
   reference point.
4. Evaluate on the held-out test set and print precision, recall, and
   accuracy to stdout.

Run standalone

::

    python Classification_Baseline.py

Output
------
Precision, recall, and accuracy scores printed to the console.
"""

import numpy as np
import pandas as pd
from Preprocessing import Preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score


if __name__ == "__main__":
    # Data loading & preprocessing
    preprocessor = Preprocessing()
    X_train, X_test, y_train, y_test = preprocessor.pre_processing(
        "classification", "bank.csv"
    )

    # Scale numerical features — Logistic Regression is sensitive to
    # feature magnitude; StandardScaler brings all features to zero mean
    # and unit variance.
    standardize = StandardScaler()
    numerical_columns = X_train.select_dtypes(include="number").columns
    X_train[numerical_columns] = standardize.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = standardize.transform(X_test[numerical_columns])

    # Model training
    # Default Logistic Regression — no tuning, serves as a simple baseline
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(precision)
    print(recall)
    print(accuracy_score(y_test, y_pred))
