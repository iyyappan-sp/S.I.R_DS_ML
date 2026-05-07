import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# load dataset
data = pd.read_csv("diabetes.csv")

print(data.head())
print(data.columns)

# split features and target
x = data.iloc[:, 0:8]
y = data.iloc[:, 8]

# Filter Method (Chi-Square)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi2_best = SelectKBest(score_func=chi2, k=4)
k_best = chi2_best.fit(x, y)

print("Chi-Square Scores")
print(k_best.scores_)

k_features = k_best.transform(x)

print("Selected features first 5 rows")
print(k_features[0:5, :])

# Plot Chi-Square scores
plt.figure()
plt.bar(x.columns, k_best.scores_)
plt.xticks(rotation=45)
plt.title("Chi-Square Feature Scores")
plt.show()

# Wrapper Method (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

rfe = RFE(estimator=model, n_features_to_select=3)
fit = rfe.fit(x, y)

print("Number of features")
print(fit.n_features_)

print("Selected features")
print(fit.support_)

print("Feature ranking")
print(fit.ranking_)
