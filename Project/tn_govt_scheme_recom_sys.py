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

# step:3: ---- Preprocessing Data
# these the columns are unnecessary
citizens = citizens.drop(columns=['name', 'ration_card_type', 'bank_account', 'land_ownership', 'land_holding_acres'])
# fill missing values
citizens['religion'] = citizens['religion'].fillna('Unknown')
citizens['education'] = citizens['education'].fillna(citizens['education'].mode()[0])
citizens['annual_income'] = citizens['annual_income'].fillna(citizens['annual_income'].median())
citizens['marital_status'] = citizens['marital_status'].fillna(citizens['marital_status'].mode()[0])
citizens['district'] = citizens['district'].fillna(citizens['district'].mode()[0])
citizens['family_size'] = citizens['family_size'].fillna(citizens['family_size'].median())
citizens['disability_status'] = citizens['disability_status'].fillna('No')

# fill missing values on education column and use mapping (higher education → higher number)

edu_levels = {
    'No Formal Education': 0, '5th Pass': 1, '8th Pass': 2,
    '10th Pass': 3, '12th Pass': 4, 'Graduate': 5, 'Post Graduate': 6
}
citizens['edu_rank'] = citizens['education'].map(edu_levels).fillna(0).astype(int)

# Some schemes cover coastal districts, others target major urban areas
coastal_districts = {'Chennai', 'Tiruvallur', 'Chengalpattu', 'Villupuram',
                     'Cuddalore', 'Mayiladuthurai', 'Nagapattinam', 'Tiruvarur', 'Thanjavur',
                     'Pudukkottai', 'Ramanathapuram', 'Thoothukudi', 'Tirunelveli', 'Kanyakumari'}

urban_districts = {'Chennai', 'Chengalpattu', 'Tiruvallur', 'Coimbatore', 'Tiruppur',
                   'Madurai', 'Salem', 'Tiruchirappalli', 'Erode', 'Kancheepuram'}

# to categorized the districts
def get_area_type(district):
    if district in coastal_districts:
        return 'Coastal'
    elif district in urban_districts:
        return 'Urban'
    return 'Rural'

citizens['district_type'] = citizens['district'].apply(get_area_type)
citizens.info()
