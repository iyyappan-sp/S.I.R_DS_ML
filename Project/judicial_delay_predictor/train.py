import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("cases_2010_2018.csv")
print("Rows loaded:", len(df))
print(df.head())
print(df.info())

# Class distribution
names = {0: "Fast", 1: "Moderate", 2: "High", 3: "Severe"}
print("\nDelay class breakdown:")
for cls, count in df["delay_class"].value_counts().sort_index().items():
    print(" ", names[cls], ":", count)