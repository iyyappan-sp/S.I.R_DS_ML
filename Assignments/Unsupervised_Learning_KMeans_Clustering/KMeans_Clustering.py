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

df1 = X[X['cluster'] == 0]
df2 = X[X['cluster'] == 1]
df3 = X[X['cluster'] == 2]
print(df1)
print(df2)
print(df3)

plt.scatter(df1.Age, df1.Income, color='red')
plt.scatter(df2.Age, df2.Income, color='green')
plt.scatter(df3.Age, df3.Income, color='blue')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

print(model.cluster_centers_)
center_x = model.cluster_centers_[:,0]
center_y = model.cluster_centers_[:,1]
plt.scatter(center_x, center_y, color='purple', marker='*')
plt.show()

plt.scatter(df1.Age, df1.Income, color='red')
plt.scatter(df2.Age, df2.Income, color='green')
plt.scatter(df3.Age, df3.Income, color='blue')
plt.scatter(center_x, center_y, color='purple', marker='*')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
