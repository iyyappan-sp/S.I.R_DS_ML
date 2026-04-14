import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Fish.csv")
data = df.copy()    # copy of data
print(df.head())
print(df.sample(5))    # want to see random use this
print(df.shape)

df.rename(columns = {'Length1':'Vertical','Length2':'Diagonal','Length3':'Cross'},inplace = True)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

sp = df.Species.value_counts()
print(sp)

sp = pd.DataFrame(sp)
print(sp)

import seaborn as sns
sns.barplot(x="Species", y="count", data=sp)
plt.show()

corr_matrix = df.select_dtypes(include="number").corr()
print(corr_matrix)

sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()

sns.pairplot(df, kind='scatter', hue='Species')
plt.show()

sns.boxplot(x=df.Weight)
plt.show()

df = df[df.Weight <=1500]
sns.boxplot(x=df.Weight)
plt.show()
print(df.shape)
print(df.head())

y = df.Weight
print(y)

X = df.iloc[:,2:7]  #row,column(Species and Weight columns are not include)
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)

import joblib
joblib.dump(model, 'weight_predict.pkl')
