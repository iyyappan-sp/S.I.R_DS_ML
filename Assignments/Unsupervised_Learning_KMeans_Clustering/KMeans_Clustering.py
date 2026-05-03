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

# KMeans Cluster
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
y_pred = model.predict(X)
print(y_pred)

X['cluster'] = y_pred
print(X)
