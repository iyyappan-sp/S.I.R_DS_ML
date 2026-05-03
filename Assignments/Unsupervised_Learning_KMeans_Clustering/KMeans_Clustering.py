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

# Minmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X[['Income']])
X.Income = scaler.transform(X[['Income']])
print(X.head())
scaler.fit(X[['Age']])
X.Age = scaler.transform(X[['Age']])
print(X.head())

X = X.drop('cluster', axis=1)
print(X)

plt.scatter(X.Age, X.Income)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

model_2 = KMeans(n_clusters=3)
model_2.fit(X)
y_pred = model_2.predict(X)
print(y_pred)
print(X.head())

X['cluster'] = y_pred
print(X.head())

print(model_2.cluster_centers_)
center_x = model_2.cluster_centers_[:,0]
center_y = model_2.cluster_centers_[:,1]
df1 = X[X['cluster'] == 0]
df2 = X[X['cluster'] == 1]
df3 = X[X['cluster'] == 2]
plt.scatter(df1.Age, df1.Income, color='red')
plt.scatter(df2.Age, df2.Income, color='green')
plt.scatter(df3.Age, df3.Income, color='blue')
plt.scatter(center_x, center_y, color='purple', marker='*')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

sse = []
for k in range(1,10):
    model = KMeans(n_clusters=k)
    model.fit(X[['Age', 'Income']])
    sse.append(model.inertia_)

x = range(1,10)
y = sse
plt.xlabel('k')
plt.ylabel('Sum of Square Error')
plt.plot(x, y)
plt.show()