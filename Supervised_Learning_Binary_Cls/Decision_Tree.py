import pandas as pd

dataset = pd.read_csv("diabetes.csv")
print(dataset)
print(dataset.info())
print(dataset.describe())
counts = dataset['Outcome'].value_counts()
print(counts)

X = dataset.iloc[:, 0:7]
y = dataset.iloc[:,8]
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)
diabetes_counts = y_train.value_counts()
print(diabetes_counts)