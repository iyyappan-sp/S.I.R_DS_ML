import pandas as pd

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