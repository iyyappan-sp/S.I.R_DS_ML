import pandas as pd

df = pd.read_csv('hiring_salaries.csv')
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.head())

df.experience = df.experience.fillna('zero')
print(df)

df = df.fillna(df.median(numeric_only=True))
print(df)
print(df.isnull().sum(axis = 0))

# working a new python library word2number

from word2number import w2n
df.experience = df.experience.apply(w2n.word_to_num)
print(df)

#train the model

from sklearn.linear_model import LinearRegression

X = df.drop('salary', axis = 1)
Y = df['salary']
print(X)
print(Y)

# initialize the model
model = LinearRegression()
model.fit(X,Y)    #input, output
print(model.coef_)
print(model.intercept_)

# save the model and load the model a new file

import joblib
joblib.dump(model, 'salary_model.pkl')
