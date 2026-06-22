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
df.info()
print(df['fraud_flag'])

print("\nFraud distribution:")
print(" Genuine:", (df["fraud_flag"] == 0).sum())
print(" Fraud:  ", (df["fraud_flag"] == 1).sum())

# EDA
# Plot 1 — how many transactions are fraud vs genuine
fraud_counts = df["fraud_flag"].value_counts()
fraud_vs_genuine, fraud_bar = plt.subplots(figsize=(6, 4))
fraud_bar.bar(["Genuine", "Fraud"], fraud_counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="white")
fraud_bar.set_title("How many UPI transactions are fraudulent?")
fraud_bar.set_ylabel("Number of transactions")
plt.tight_layout()
plt.show()

# Plot 2 — which transaction type has the highest fraud rate
fraud_rate_by_transaction = df.groupby("transaction type")["fraud_flag"].mean().sort_values(ascending=False)
fraud_by_transaction_type, transaction_bar = plt.subplots(figsize=(7, 4))
transaction_bar.bar(fraud_rate_by_transaction.index, fraud_rate_by_transaction.values * 100, color="#e74c3c", edgecolor="white")
transaction_bar.set_title("Which transaction type has the highest fraud rate?")
transaction_bar.set_ylabel("Fraud rate (%)")
plt.tight_layout()
plt.show()

# Plot 3 — at what hour of the day does fraud peak
fraud_rate_by_hour = df.groupby("hour_of_day")["fraud_flag"].mean()
fraud_by_hour_of_day, hourly_fraud_line = plt.subplots(figsize=(10, 4))
hourly_fraud_line.plot(fraud_rate_by_hour.index, fraud_rate_by_hour.values * 100, marker="o", color="#e74c3c")
hourly_fraud_line.set_title("At what hour of the day does fraud peak?")
hourly_fraud_line.set_xlabel("Hour of day")
hourly_fraud_line.set_ylabel("Fraud rate (%)")
plt.tight_layout()
plt.show()

# Plot 4 — which merchant category is most targeted by fraud
fraud_rate_by_merchant = df.groupby("merchant_category")["fraud_flag"].mean().sort_values(ascending=True)
fraud_by_merchant_category, merchant_bar = plt.subplots(figsize=(9, 5))
merchant_bar.barh(fraud_rate_by_merchant.index, fraud_rate_by_merchant.values * 100, color="#e67e22", edgecolor="white")
merchant_bar.set_title("Which merchant category is most targeted by fraud?")
merchant_bar.set_xlabel("Fraud rate (%)")
plt.tight_layout()
plt.show()

# Feature engineering
features = [
    "transaction type", "merchant_category", "amount (INR)",
    "transaction_status", "sender_age_group", "receiver_age_group",
    "sender_state", "sender_bank", "receiver_bank",
    "device_type", "network_type",
    "hour_of_day", "is_weekend"
]

encoders = {}
for col in df[features].select_dtypes("object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[features]
y = df["fraud_flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
print("\nAfter SMOTE — Training samples:", len(X_train))

# Train
print("\nTraining model...")
model = XGBClassifier( n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluation:")
print(classification_report(y_test, model.predict(X_test), target_names=["Genuine", "Fraud"]))

confusion, confusion_plot = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)), display_labels=["Genuine", "Fraud"]).plot(ax=confusion_plot, colorbar=False, cmap="Reds")
confusion_plot.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

# SHAP
print("\nComputing SHAP values...")
model.get_booster().set_param({'base_score': 0.5})
explainer = shap.TreeExplainer(model)
sample = X_test.sample(300, random_state=42)
shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False, plot_size=(10, 5))
plt.title("SHAP — which features drive fraud predictions?")
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(model,     "model.pkl")
joblib.dump(encoders,  "encoders.pkl")
joblib.dump(features,  "features.pkl")
joblib.dump(explainer, "explainer.pkl")
print("\nModel saved.")