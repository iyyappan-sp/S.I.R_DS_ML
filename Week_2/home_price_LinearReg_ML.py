import pandas as pd

df = pd.read_csv('home_prices.csv')
print(df.info)
print(df.describe())
print(df.isnull().sum())

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
print(df)

from sklearn.linear_model import LinearRegression

X = df.drop('price', axis=1)
Y = df['price']

#model initialize
model = LinearRegression()
model.fit(X,Y)
print(model.coef_)
print(model.intercept_)

