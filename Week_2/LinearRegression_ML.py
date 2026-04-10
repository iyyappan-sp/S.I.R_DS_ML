import pandas as pd

df = pd.read_csv('hiring_salaries.csv')
print(df.info())
print(df.isnull().sum())
