"""
            ============================================
                TN GOVT SCHEME RECOMMENDATION SYSTEM
            ============================================
                A Machine Learning Project for Recommending Government Schemes to Citizens

                step:1: ---- Import necessary libraries
                step:2: ---- Read the Datasets
                step:3: ---- Preprocessing Data
                step:4: ---- EDA Processing (Visualization)
                step:5: ---- Outliers treatments
                step:6: ---- Eligibility Rules
                step:7: ---- Building Training Data
                step:8: ---- Encoding & Splitting Data
                step:9: ---- SMOTE
                step:10: ---- Decision Tree
                step:11: ---- Random Forest
                step:12: ---- ROC Curve
                step:13: ---- Feature Engineering
                step:14: ---- Recommendation Engine
                step:15: ---- Save For API

"""

# step:1: ---- Import necessary libraries
import warnings, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# step:2: ---- Read the Datasets
citizens = pd.read_csv("tn_citizen_dataset.csv")
schemes  = pd.read_csv("tn_schemes.csv")
citizens.info()
schemes.info()
