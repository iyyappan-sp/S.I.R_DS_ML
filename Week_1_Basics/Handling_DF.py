import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('House_Prices.csv')
print(df.head())
print(df.shape)    #rows x columns
print(df.info())
print(df.isnull().sum())

# basic statistical

print(df.describe())
#
# # EDA Process
#
# df.estimated_value.hist()
# plt.show()
#
# sns.boxplot(df['estimated_value'])
# plt.show()
# #reduce the out layers
# df = df[df.estimated_value <=1000000]
# print(df.shape)
#
# #after reduce out layers
# sns.boxplot(df['estimated_value'])
# plt.show()
#
# #after reduce out layers hist
#
# df.estimated_value.hist()
# plt.show()
#
# df.bedrooms.hist()
# plt.show()
df = df[['bedrooms','bathrooms','rooms','squareFootage','lotSize','yearBuilt','priorSaleAmount','estimated_value']]
print(df.shape)
print(df.head())
print(df.isna().sum())
df = df.fillna(df.median())
print(df.head())
print(df.isna().sum())
print(df.isin([0]).sum())