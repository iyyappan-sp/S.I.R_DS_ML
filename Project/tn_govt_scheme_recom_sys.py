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

# step:4: ---- EDA Processing (Visualization)
# age distribution – line are show mean(continuous data)
plt.hist(citizens['age'], bins=25, color='steelblue', edgecolor='white')
plt.axvline(citizens['age'].mean(), color='red', linestyle='--', label=f"Mean: {citizens['age'].mean():.1f}")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

# gender split – pie plot are show which gender is high(categorical data)
gender_counts = citizens['gender'].value_counts()
gender_mode = gender_counts.index[0]
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#2980B9', '#C0392B', '#8E44AD'])
plt.title(f"Gender Split  (Mode: {gender_mode})")
plt.show()

# annual income – mean is affected by outliers & median is more stable. lines added (skewed, both matter)
plt.hist(citizens['annual_income'], bins=30, color='#F39C12', edgecolor='white')
plt.axvline(citizens['annual_income'].mean(),   color='red',  linestyle='--', label=f"Mean:   Rs.{citizens['annual_income'].mean()/1000:.0f}K")
plt.axvline(citizens['annual_income'].median(), color='blue', linestyle='--', label=f"Median: Rs.{citizens['annual_income'].median()/1000:.0f}K")
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.legend()
plt.show()

# Distribution of occupations in dataset using Horizontal Bar Graph(barh)
plt.barh(citizens['occupation'].value_counts().index, citizens['occupation'].value_counts().values, color='#16A085')
plt.title(f"Occupation Breakdown  (Mode: {citizens['occupation'].mode()[0]})")
plt.xlabel("Count")
plt.show()

# schemes by category – mode shown in title
sns.barplot(x=schemes['category'].value_counts().values, y=schemes['category'].value_counts().index, palette='tab10')
plt.title(f"Schemes by Category  (Mode: {schemes['category'].mode()[0]})")
plt.xlabel("Count")
plt.show()

# step:5: ---- Outliers treatments
# boxplot to identify outliers – only for numerical columns
for col in citizens.select_dtypes(include='number').columns:
    sns.boxplot(data=citizens, x=col)
    plt.title(f"Outlier Check – {col}")
    plt.show()

# IQR method to find and cap outliers
# only applicable for continuous numerical columns (age, annual_income, family_size)
# edu_rank is ordinal (ranked categories) so we skip it
for col in ['age', 'annual_income', 'family_size']:
    q1, q3 = np.percentile(citizens[col], [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    print(f"  {col} → outliers capped to ({lower:.0f} – {upper:.0f})")
    citizens[col] = citizens[col].clip(lower=lower, upper=upper)

sns.boxplot(data=citizens, x='annual_income')
plt.title("Outlier Check After – annual_income")
plt.show()

# step:6: ---- Eligibility Rules
def check_eligibility(person, scheme):
    # disability match check
    if scheme['category'] == 'Disability':
        if scheme['scheme_name'] == 'Transgender Welfare Scheme':
            if person['gender'] != 'Transgender':
                return False
        elif person.get('disability_status', 'No') != 'Yes':
            return False

    # caste match check
    if scheme['caste'] != 'Any':
        if scheme['caste'] == 'BC/MBC' and person['caste'] not in ('BC', 'MBC'):
            return False
        if scheme['caste'] == 'SC/ST' and person['caste'] not in ('SC', 'ST'):
            return False
        if scheme['caste'] not in ('BC/MBC', 'SC/ST') and person['caste'] != scheme['caste']:
            return False

    # occupation match check
    if scheme['occupation'] != 'Any':
        if person['occupation'] not in [o.strip() for o in scheme['occupation'].split('/')]:
            return False

    # education match check
    if scheme['education'] != 'Any':
        if edu_levels.get(person['education'], 0) < edu_levels.get(scheme['education'], 0):
            return False

    return (
        (scheme['gender'] == 'Any' or person['gender'] == scheme['gender']) and
        scheme['min_age'] <= person['age'] <= scheme['max_age'] and
        person['annual_income'] <= scheme['income_limit'] and
        (scheme['marital_status'] == 'Any' or person['marital_status'] == scheme['marital_status']) and
        (scheme['district'] == 'Any' or person['district_type'] == scheme['district'])
    )

# step:7: ---- Building Training Data
print("\nBuilding training data...")
start = time.time()

# Sample 100 citizens per occupation to keep it balanced
sample = pd.concat([
    citizens[citizens['occupation'] == occ].sample(n=min(100, (citizens['occupation'] == occ).sum()), random_state=42)
    for occ in citizens['occupation'].unique()
]).reset_index(drop=True)

# Stores final training rows
training_rows = []
for _, row in sample.iterrows():
    person_dict = {**row.to_dict(), 'district_type': row['district_type']}
    for _, scheme in schemes.iterrows():
        training_rows.append({
            'age'                  : row['age'],
            'annual_income'        : row['annual_income'],
            'family_size'          : row['family_size'],
            'edu_rank'             : row['edu_rank'],
            'gender'               : row['gender'],
            'caste'                : row['caste'],
            'occupation'           : row['occupation'],
            'education'            : row['education'],
            'marital_status'       : row['marital_status'],
            'disability_status'    : row['disability_status'],
            'district_type'        : row['district_type'],
            'scheme_min_age'       : scheme['min_age'],
            'scheme_max_age'       : scheme['max_age'],
            'scheme_income_limit'  : scheme['income_limit'],
            'scheme_benefit_amount': scheme['benefit_amount'],
            'age_in_range'         : int(scheme['min_age'] <= row['age'] <= scheme['max_age']),
            'income_ok'            : int(row['annual_income'] <= scheme['income_limit']),
            'income_ratio'         : round(row['annual_income'] / scheme['income_limit'], 4),
            'age_ratio'            : round(row['age'] / scheme['max_age'], 4),
            'eligible'             : int(check_eligibility(person_dict, scheme))
        })

training_data = pd.DataFrame(training_rows)
print(f"Done. {training_data.shape} | Eligible: {training_data['eligible'].mean()*100:.1f}% | {time.time()-start:.1f}s")

# step:8: ---- Encoding & Splitting Data
# Encode categorical columns
cat_cols = ['gender', 'caste', 'occupation', 'education', 'marital_status', 'disability_status', 'district_type']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    training_data[col] = le.fit_transform(training_data[col].astype(str))
    encoders[col] = le

feature_cols = [
    'age', 'gender', 'caste', 'occupation', 'education', 'annual_income',
    'marital_status', 'disability_status', 'district_type', 'family_size',
    'edu_rank', 'scheme_min_age', 'scheme_max_age', 'scheme_income_limit',
    'scheme_benefit_amount', 'age_in_range', 'income_ok', 'income_ratio', 'age_ratio'
]

X = training_data[feature_cols]
y = training_data['eligible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# class distribution before SMOTE (Synthetic Minority Over-sampling Technique)
plt.bar(['Not Eligible', 'Eligible'], training_data['eligible'].value_counts().values, color=['#2980B9', '#27AE60'])
plt.title("Class Distribution (Before SMOTE)")
plt.ylabel("Count")
plt.show()

# step:9: ---- SMOTE to fix class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"After SMOTE → Train: {len(X_train):,} | Eligible: {y_train.sum():,} | Not Eligible: {(y_train==0).sum():,}")

# class distribution after SMOTE
plt.bar(['Not Eligible', 'Eligible'], [(y_train==0).sum(), y_train.sum()], color=['#2980B9', '#27AE60'])
plt.title("Class Distribution (After SMOTE)")
plt.ylabel("Count")
plt.show()
