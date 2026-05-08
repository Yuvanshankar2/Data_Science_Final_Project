import numpy as np
import pandas as pd
from Preprocessing import Preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score
preprocessor = Preprocessing()

X_train,X_test,y_train, y_test = preprocessor.pre_processing("classification","bank.csv")
standardize = StandardScaler()
numerical_columns  = X_train.select_dtypes(include="number").columns
X_train[numerical_columns] = standardize.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = standardize.transform(X_test[numerical_columns])
model  = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)

print(precision)
print(recall)
print(accuracy_score(y_test,y_pred))
