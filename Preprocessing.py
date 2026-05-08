"""
Preprocessing
=============
Provides the ``Preprocessing`` class for loading and preparing the bank
marketing dataset (``bank.csv``) for downstream classification and
clustering models.

Supported pipeline modes
------------------------
classification
    Encodes features and splits into labelled train/test sets
    (X_train, X_test, y_train, y_test).  Suitable for supervised learning.

clustering
    Additionally applies a Yeo-Johnson power transform and
    StandardScaler normalisation on numerical features before splitting
    into train/test sets (X_train, X_test — no labels).  Suitable for
    unsupervised learning.

Dataset
-------
The expected input CSV (``bank.csv``) is the UCI Bank Marketing dataset:
11,161 rows × 17 columns.  The binary target column is ``deposit``
(yes = 1, no = 0 after encoding).

Dependencies
------------
pandas, numpy, scikit-learn, category_encoders
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import power_transform, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder


class Preprocessing():
    """Utility class for loading and preprocessing the bank marketing dataset.

    Provides two preprocessing pipelines — one for supervised
    classification tasks and one for unsupervised clustering tasks —
    accessible through a single dispatcher method :meth:`pre_processing`.
    """

    def __init__(self):
        """Initialise the Preprocessing helper.

        Prints a welcome message to confirm instantiation.
        """
        print("Welcome")

    # ------------------------------------------------------------------ #
    #  Classification pipeline                                            #
    # ------------------------------------------------------------------ #

    def classification_processing(self, df):
        """Prepare a bank-marketing DataFrame for binary classification.

        Processing steps
        ----------------
        1. Drop duplicate rows.
        2. Drop rows with missing values.
        3. Separate features (``X``) from the target column ``deposit``.
        4. Perform an 80/20 train/test split.
        5. Validate that ``age`` and ``day`` columns contain no negative
           values (prints a warning if they do).
        6. Binary-encode ordinal columns:
           ``default``, ``housing``, ``loan`` (yes → 1, no → 0).
        7. Ordinal-encode ``education``
           (unknown → 0, primary → 1, secondary → 2, tertiary → 3).
        8. One-hot-encode nominal columns:
           ``job``, ``marital``, ``contact``, ``month``, ``poutcome``.
        9. Binary-encode the target: yes → 1, no → 0.

        Parameters
        ----------
        df : pd.DataFrame
            Raw bank-marketing dataset loaded from ``bank.csv``.

        Returns
        -------
        X_train : pd.DataFrame
            Encoded training feature matrix.
        X_test : pd.DataFrame
            Encoded test feature matrix.
        y_train : pd.Series
            Binary training target vector (0 = no deposit, 1 = deposit).
        y_test : pd.Series
            Binary test target vector (0 = no deposit, 1 = deposit).
        """
        # Remove duplicate entries
        df = df.drop_duplicates()

        # Remove missing values
        df = df.dropna()

        # Split the dataset into a training and testing set.
        X = df.drop(columns=["deposit"])
        y = df["deposit"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Check for negative values in columns that should always be positive
        if (df['age'] < 0).any():
            print("Negative values detected in age")
        if (df['day'] < 0).any():
            print("Negative values detected in Days")

        # Binary-encode binary categorical columns (yes/no → 1/0)
        for col in ["default", "housing", "loan"]:
            X_train[col] = X_train[col].map({"yes": 1, "no": 0})
            X_test[col] = X_test[col].map({"yes": 1, "no": 0})

        # Ordinal-encode education: preserves natural ordering of levels
        edu_map = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
        X_train["education"] = X_train["education"].map(edu_map)
        X_test["education"] = X_test["education"].map(edu_map)

        # One-hot-encode remaining nominal columns (no ordinal relationship)
        nominal_tree_encoder = OneHotEncoder(
            cols=["job", "marital", "contact", "month", "poutcome"]
        )
        X_train = nominal_tree_encoder.fit_transform(X_train)
        X_test = nominal_tree_encoder.transform(X_test)

        # Encode target: yes → 1 (subscribed), no → 0 (did not subscribe)
        y_train = y_train.map({"yes": 1, "no": 0})
        y_test = y_test.map({"yes": 1, "no": 0})

        # The dataset is ready for model training.
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    #  Clustering pipeline                                                #
    # ------------------------------------------------------------------ #

    def clustering_processing(self, df):
        """Prepare a bank-marketing DataFrame for unsupervised clustering.

        Processing steps
        ----------------
        1. Drop duplicate rows.
        2. Drop rows with missing values.
        3. Separate features (``X``) from the target column ``deposit``
           (the target is dropped from features).
        4. Perform an 80/20 train/test split.
        5. Apply a Yeo-Johnson power transform to reduce skew in
           numerical features.
        6. Standardise numerical columns with :class:`StandardScaler`
           (zero mean, unit variance).
        7. Binary-encode ordinal columns:
           ``default``, ``housing``, ``loan``.
        8. Ordinal-encode ``education``.
        9. One-hot-encode nominal columns:
           ``job``, ``marital``, ``contact``, ``month``, ``poutcome``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw bank-marketing dataset loaded from ``bank.csv``.

        Returns
        -------
        X_train : pd.DataFrame
            Fully encoded and scaled training feature matrix (no labels).
        X_test : pd.DataFrame
            Fully encoded and scaled test feature matrix (no labels).
        """
        # Remove duplicates
        df = df.drop_duplicates()

        # Remove missing values
        df = df.dropna()

        # Print categorical column names for inspection during development
        print(df.select_dtypes(include="object").columns)

        # Separate features from the deposit target (not used in clustering)
        X = df.drop(columns=["deposit"])
        y = df["deposit"]

        # Split before any transforms to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        numerical_columns = X_train.select_dtypes(include="number").columns

        # Apply Yeo-Johnson to reduce the skew of numerical features;
        # fitted on train set only to avoid leakage
        transformer = PowerTransformer(method="yeo-johnson")
        X_train[numerical_columns] = transformer.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = transformer.transform(X_test[numerical_columns])

        # Standardize the numerical features (zero mean, unit variance)
        standardize = StandardScaler()
        X_train[numerical_columns] = standardize.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = standardize.transform(X_test[numerical_columns])

        # Binary-encode binary categorical columns (yes/no → 1/0)
        for col in ["default", "housing", "loan"]:
            X_train[col] = X_train[col].map({"yes": 1, "no": 0})
            X_test[col] = X_test[col].map({"yes": 1, "no": 0})

        # Ordinal-encode education
        edu_map = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
        X_train["education"] = X_train["education"].map(edu_map)
        X_test["education"] = X_test["education"].map(edu_map)

        # One-hot-encode remaining nominal columns
        nominal = OneHotEncoder(cols=["job", "marital", "contact", "month", "poutcome"])
        X_train = nominal.fit_transform(X_train)
        X_test = nominal.transform(X_test)

        # The set is ready for model training
        return X_train, X_test

    # ------------------------------------------------------------------ #
    #  Dispatcher                                                         #
    # ------------------------------------------------------------------ #

    def pre_processing(self, model_mode, file: str):
        """Load ``bank.csv`` and dispatch to the correct preprocessing pipeline.

        Parameters
        ----------
        model_mode : str
            ``'classification'`` to run the supervised preprocessing
            pipeline; any other value runs the clustering (unsupervised)
            pipeline.
        file : str
            Path to the CSV data file (e.g. ``'bank.csv'``).

        Returns
        -------
        classification mode
            X_train, X_test, y_train, y_test
        clustering mode
            X_train, X_test
        """
        df = pd.read_csv(file)
        # Run df.describe() and report what you see
        if model_mode == 'classification':
            X_train, X_test, y_train, y_test = self.classification_processing(df)
            print("Classification works")
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = self.clustering_processing(df)
            print("Clustering works")
            return X_train, X_test


# ------------------------------------------------------------------ #
#  Standalone smoke-test                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    testClass = Preprocessing()
    testClass.pre_processing("clustering", "bank.csv")
