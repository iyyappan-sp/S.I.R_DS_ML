import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("upi_transactions_2024.csv")
print("Rows loaded:", len(df))

print("\nFraud distribution:")
print(" Genuine:", (df["fraud_flag"] == 0).sum())
print(" Fraud:  ", (df["fraud_flag"] == 1).sum())