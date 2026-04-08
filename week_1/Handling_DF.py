import pandas as pd

df = pd.read_csv('House_Prices.csv')
print(df.head())
print(df.shape)    #rows x columns
print(df.info())
print(df.isnull().sum())

# basic statistical

print(df.describe())