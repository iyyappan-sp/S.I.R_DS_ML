"""
Multiclass Classification using Binary classification Algorithm
Logistic Regression--- heuristic methods are : one-vs-one & one-vs-rest

"""

import pandas as pd
data = pd.read_csv('iris_dataset.csv')
data.info()
print(data)
print(data.describe())

X = data.iloc[:, 0:4]
y = data.iloc[:, 4]
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape)
print(y_train.shape)