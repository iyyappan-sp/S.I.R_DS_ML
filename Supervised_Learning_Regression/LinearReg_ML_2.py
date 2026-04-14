import warnings
warnings.filterwarnings('ignore')
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

corr = df.select_dtypes(include="number").corr()
print(corr)

sns.heatmap(corr, annot=True, cmap='YlGnBu')
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

y_pred = model.predict(X_test)
print(y_pred)
y_pred = pd.DataFrame(y_pred, columns = ['Predicted_Weight'])
print(y_pred.head())

y_test = pd.DataFrame(y_test)
print(y_test.head())

y_test = y_test.reset_index(drop = True)
print(y_test.head())

y_output = pd.concat([y_test,y_pred], axis = 1)
print(y_output)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

print(X_train.head(2))

import numpy as np
new_data = np.array([[23,25,30,11,14]])
prediction = model.predict(new_data)
print(prediction)