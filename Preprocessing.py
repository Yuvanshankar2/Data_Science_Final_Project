import numpy as np
import pandas as pd
from sklearn.preprocessing import power_transform,StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
# [EXPLAIN THIS CLASS]
class Preprocessing():
    def __init__(self):
        print("Welcome")

    # [EXPLAIN THIS FUNCTION]
    def classification_processing(self,df):
        # Remove duplicate entries
        df = df.drop_duplicates()

        # Remove missing values
        df = df.dropna()

        # Split the dataset into a training and testing set.
        X = df.drop(columns=["deposit"])
        y = df["deposit"]
        X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)
        # Check for negative values
        if((df['age']<0).any()):
            print("Negative values detected in age")
        if((df['day']<0).any()):
            print("Negative values detected in Days")

        # Perform encoding
        X_train["default"] = X_train["default"].map({"yes":1,"no":0})
        X_test["default"] = X_test["default"].map({"yes":1,"no":0})
        X_train["housing"] = X_train["housing"].map({"yes":1,"no":0})
        X_test["housing"] = X_test["housing"].map({"yes":1,"no":0})
        X_train["loan"] = X_train["loan"].map({"yes":1,"no":0})
        X_test["loan"] = X_test["loan"].map({"yes":1,"no":0})
        X_train["education"] = X_train["education"].map({"unknown":0,"primary":1,"secondary":2,"tertiary":3})
        X_test["education"] = X_test["education"].map({"unknown":0,"primary":1,"secondary":2,"tertiary":3})
        nominal_tree_encoder  = OneHotEncoder(cols=["job","marital","contact","month","poutcome"])
        X_train = nominal_tree_encoder.fit_transform(X_train)
        X_test = nominal_tree_encoder.transform(X_test)
        y_train = y_train.map({"yes":1,"no":0})
        y_test = y_test.map({"yes":1,"no":0})

        # The dataset is ready for model training.
        return X_train, X_test, y_train, y_test

    # [EXPLAIN THIS FUNCTION] 
    def clustering_processing(self,df):
        # Remove duplicates
        df = df.drop_duplicates()

        # Remove missing values
        df = df.dropna()

        print(df.select_dtypes(include="object").columns)
        # Check for negative values
        X = df.drop(columns=["deposit"])
        y = df["deposit"]

        # Apply Yeo-Johnson to reduce the skew of numerical features
        X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2)
        numerical_columns = X_train.select_dtypes(include="number").columns
        transformer = PowerTransformer(method="yeo-johnson")
        X_train[numerical_columns] = transformer.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = transformer.transform(X_test[numerical_columns])
        
        # Standardize the numerical features
        standardize = StandardScaler()
        X_train[numerical_columns] = standardize.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = standardize.transform(X_test[numerical_columns])
        
        # Apply encoding on the training set
        X_train["default"] = X_train["default"].map({"yes":1, "no":0})
        X_test["default"] = X_test["default"].map({"yes":1, "no":0})
        X_train["housing"] = X_train["housing"].map({"yes":1, "no":0})
        X_test["housing"] = X_test["housing"].map({"yes":1, "no":0})
        X_train["loan"] = X_train["loan"].map({"yes":1,"no":0})
        X_test["loan"] = X_test["loan"].map({"yes":1, "no":0})
        X_train["education"] = X_train["education"].map({"unknown":0,"primary":1,"secondary":2,"tertiary":3})
        X_test["education"] = X_test["education"].map({"unknown":0,"primary":1,"secondary":2,"tertiary":3})
        nominal = OneHotEncoder(cols=["job","marital","contact","month","poutcome"])
        X_train = nominal.fit_transform(X_train)
        X_test = nominal.transform(X_test)

        # The set is ready for model training
        return X_train, X_test

    # [EXPLAIN THIS FUNCTION] 
    def pre_processing(self,model_mode,file:str):
        df = pd.read_csv(file)
        # Run df.describe() and report what you see
        if(model_mode == 'classification'):
             X_train, X_test,y_train, y_test = self.classification_processing(df)
             print("Classification works")
             return X_train,X_test,y_train,y_test
        else:
             X_train, X_test = self.clustering_processing(df)
             print("Clustering works")
             return X_train,X_test
        


        





testClass = Preprocessing()
testClass.pre_processing("clustering","bank.csv")


