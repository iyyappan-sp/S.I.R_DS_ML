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
delay_names = {0: "Fast", 1: "Moderate", 2: "High", 3: "Severe"}
print("\nDelay class breakdown:")
for cls, count in df["delay_class"].value_counts().sort_index().items():
    print(" ", delay_names[cls], ":", count)

# EDA
# Plot 1 — delay distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["delay_days"].clip(upper=4000), bins=60, color="#3498db", edgecolor="white")
ax.set_xlabel("Delay (days)")
ax.set_ylabel("Number of cases")
ax.set_title("How long do Indian district court cases take?")
ax.axvline(365,  color="red",    linestyle="--", label="1 year")
ax.axvline(1825, color="orange", linestyle="--", label="5 years")
ax.legend()
plt.tight_layout()
plt.show()

# Plot 2 — cases per delay class
class_counts = df["delay_class"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([delay_names[i] for i in class_counts.index], class_counts.values,
       color=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"], edgecolor="white")
ax.set_xlabel("Delay class")
ax.set_ylabel("Number of cases")
ax.set_title("How many cases fall in each delay category?")
plt.tight_layout()
plt.show()

# Plot 3 — median delay by year
yearly_delay = df.groupby("year")["delay_days"].median().reset_index()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(yearly_delay["year"], yearly_delay["delay_days"], marker="o", color="#2980b9")
ax.set_xlabel("Year filed")
ax.set_ylabel("Median delay (days)")
ax.set_title("Has delay improved over the years?")
plt.tight_layout()
plt.show()

# Plot 4 — which case types take the longest (using type_name_key for readable names)
type_keys = pd.read_csv("type_name_key.csv")[["type_name", "type_name_s"]].drop_duplicates("type_name")
df_named  = df.merge(type_keys, on="type_name", how="left")
case_delay = (
    df_named.groupby("type_name_s")
    .agg(median_delay=("delay_days", "median"), count=("delay_days", "count"))
    .query("count > 500")
    .sort_values("median_delay", ascending=True)
    .tail(10)
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(case_delay["type_name_s"], case_delay["median_delay"], color="#e74c3c", edgecolor="white")
ax.set_xlabel("Median delay (days)")
ax.set_title("Which case types take the longest to resolve?")
plt.tight_layout()
plt.show()