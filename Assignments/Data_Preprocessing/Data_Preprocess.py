"""
Steps of preprocessing of data

step:1---- Import necessary libraries
step:2---- Read Dataset
step:3---- Sanity check of data
step:4---- Exploratory Data Analysis
step:5---- Missing Values treatments
step:6---- Outliers treatments
step:7---- Duplicates & Garbage value treatments
step:8---- Normalization
step:9---- Encoding of data

"""

# step:1: Import Necessary Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# step:2: Read Dataset

df = pd.read_csv("Life_Expectancy_Data.csv")
print(df)    # it shows full overview of the dataset
print(df.head())     # it shows head of  5(0-4) the dataset
print(df.tail())     # it shows tail of 5 in the dataset

# step:3: Sanity check of data

print(df.shape)    # it will show how many rows x columns
print(df.info())   # it gives overall information like datatypes and columns names
print(df.isnull().sum())     # it shows how many null/missing values
print((df.isnull().sum() / df.shape[0]) * 100)    # it shows percentage of overall missing data
print(df.duplicated().sum())    # it shows how many duplicated values on the dataset

# identifiying garbage values
for i in df.select_dtypes(include="object").columns:
    print(df[i].value_counts())
    print("***"*10)

#step:4---- Exploratory Data Analysis(EDA)
print(df.describe().T)     # 'T' means transpose
print(df.describe(include="object"))

# histogram for understand the distribution
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i)
    plt.show()

# boxplot-to-identify outliers
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df, x=i)
    plt.show()

# scatter plot for understand the relationship
print(df.select_dtypes(include="number").columns)
for i in ['Year', 'Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'Schooling']:
    sns.scatterplot(data=df, x=i, y='Life expectancy ')
    plt.show()

# correlation with heatmap for interpret the relation and multicolliniarity
correl =df.select_dtypes(include="number").corr()
print(correl)
plt.figure(figsize=(15,15))
sns.heatmap(correl, annot=True)
plt.show()

# step:5 - Missing Values treatments

# choose the method of imputing missing values
#like you choose numeric columns use mean,median and KNNImputer
# but you choose categorical columns use mode

""" mean --- only for numerical data
 median --- only for numerical data
 mode --- Both numerical & categorical data
 KNN Imputer --- Mostly numeric data (after encoding) """

print(df.info())
print(df.isnull().sum())

for i in [' BMI ','Polio','Income composition of resources']:
    df[i].fillna(df[i].median(),inplace=True)
print(df.isnull().sum())  #BMI, Polio, Income composition of resources---0
