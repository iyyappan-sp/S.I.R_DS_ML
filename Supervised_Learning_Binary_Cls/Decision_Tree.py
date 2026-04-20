import pandas as pd

dataset = pd.read_csv("diabetes.csv")
print(dataset)
print(dataset.info())
print(dataset.describe())
counts = dataset['Outcome'].value_counts()
print(counts)