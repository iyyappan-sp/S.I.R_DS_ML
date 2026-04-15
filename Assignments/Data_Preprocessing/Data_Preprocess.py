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


