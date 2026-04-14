import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('Fish.csv')
print(df)
print(df.isnull().sum())
print(df.describe())
df.rename(columns={'Length1':'Vertical','Length2':'Diagonal','Length3':'Cross'}, inplace=True)
print(df)
sp = df.Species.value_counts()
print(sp)

sp = pd.DataFrame(sp)
print(sp)

import seaborn as sns
sns.barplot(x='Species', y='count', data=sp)
plt.show()

corr = df.select_dtypes(include='number').corr()
print(corr)

sns.heatmap(corr, annot=True)
plt.show()

sns.boxplot(x=df.Width)
plt.show()

# train the model

y = df['Width']
print(y)

X = df.drop(columns=["Width", "Species"])
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state = 42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
print(model.coef_)
print(model.intercept_)

import joblib
joblib.dump(model, 'width_prediction.pkl')

y_pred = model.predict(X_test)
print(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Width_Prediction'])
print(y_pred.head())

y_test = pd.DataFrame(y_test)
print(y_test.head())
y_test = y_test.reset_index(drop=True)
print(y_test.head())

y_output = pd.concat([y_test,y_pred], axis=1)
print(y_output)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

import numpy as np
new_data = np.array([[300,25,27,30,8]])
prediction = model.predict(new_data)
print(prediction)