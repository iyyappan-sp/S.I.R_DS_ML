import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os

base     = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(base, "model.pkl"))
encoders = joblib.load(os.path.join(base, "encoders.pkl"))
features = joblib.load(os.path.join(base, "features.pkl"))
explainer= joblib.load(os.path.join(base, "explainer.pkl"))

app = FastAPI(title="UPI Fraud Detector", version="1.0.0")


class Transaction(BaseModel):
    transaction_type:   str
    merchant_category:  str
    amount:             float
    transaction_status: str
    sender_age_group:   str
    receiver_age_group: str
    sender_state:       str
    sender_bank:        str
    receiver_bank:      str
    device_type:        str
    network_type:       str
    hour_of_day:        int
    is_weekend:         int = 0


def prepare(txn: Transaction):
    row = {
        "transaction type":    txn.transaction_type,
        "merchant_category":   txn.merchant_category,
        "amount (INR)":        txn.amount,
        "transaction_status":  txn.transaction_status,
        "sender_age_group":    txn.sender_age_group,
        "receiver_age_group":  txn.receiver_age_group,
        "sender_state":        txn.sender_state,
        "sender_bank":         txn.sender_bank,
        "receiver_bank":       txn.receiver_bank,
        "device_type":         txn.device_type,
        "network_type":        txn.network_type,
        "hour_of_day":         txn.hour_of_day,
        "is_weekend":          txn.is_weekend,
    }
    for col, le in encoders.items():
        val = str(row.get(col, "unknown"))
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

    df = pd.DataFrame([[row.get(f, 0) for f in features]], columns=features)
    return df.astype(float)


def get_shap_reasons(X_row, top_n=3):
    shap_vals = explainer.shap_values(X_row)
    # For binary classification XGBoost returns a single array
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]
    else:
        vals = shap_vals[0]

    pairs = sorted(zip(features, vals), key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature":   name,
            "impact":    round(float(val), 3),
            "direction": "increases fraud risk" if val > 0 else "decreases fraud risk",
        }
        for name, val in pairs[:top_n]
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost"}


@app.post("/predict")
def predict(txn: Transaction):
    X    = prepare(txn)
    prob = model.predict_proba(X)[0]
    fraud_prob = round(float(prob[1]), 3)
    is_fraud   = bool(fraud_prob >= 0.5)
    reasons    = get_shap_reasons(X)

    return {
        "fraud":       is_fraud,
        "probability": fraud_prob,
        "verdict":     "FRAUD" if is_fraud else "GENUINE",
        "reasons":     reasons,
    }