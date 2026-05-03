import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Income.csv")
print(df)

X = df.drop('Name', axis=1)
print(X)

plt.scatter(X.Age, X.Income)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()