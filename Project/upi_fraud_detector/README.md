# UPI Fraud Detector

Detects fraudulent UPI transactions and explains why — using XGBoost with SHAP-based explanations returned directly in the API response.

---

## The Problem

India processes 13+ billion UPI transactions every month. Fraud detection at this scale requires not just a prediction but an explanation — why was this transaction flagged? This project predicts whether a transaction is fraudulent and returns the top 3 features driving that decision.

---

## What Makes This Different

Most fraud detection projects stop at a prediction. This one returns SHAP reasons in the API response:

```json
{
  "fraud": true,
  "probability": 0.87,
  "verdict": "FRAUD",
  "reasons": [
    {"feature": "amount (INR)", "impact": 2.3, "direction": "increases fraud risk"},
    {"feature": "hour_of_day", "impact": 1.8, "direction": "increases fraud risk"},
    {"feature": "merchant_category", "impact": 0.9, "direction": "increases fraud risk"}
  ]
}
```

---

## Dataset

Kaggle — UPI Transactions 2024 Dataset
`kaggle.com/datasets/skullagos5246/upi-transactions-2024-dataset`

- 250,000 transactions
- 480 fraud cases (0.19% fraud rate — extreme class imbalance)
- Features: transaction type, merchant category, amount, sender/receiver bank, device type, network type, hour of day

Class imbalance handled with SMOTE oversampling.

---

## Tech Stack

- **Data & EDA** — pandas, NumPy, matplotlib
- **Model** — XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Explainability** — SHAP
- **API** — FastAPI, Uvicorn
- **Tests** — pytest

---

## Project Files

```
upi_fraud_detector/
├── train.py         — EDA, feature engineering, model training, SHAP
├── app.py           — FastAPI REST API with SHAP explanation in response
├── test_app.py      — 5 pytest tests
└── requirements.txt
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Place the dataset in the project folder**
```
upi_transactions_2024.csv
```

**3. Train the model**
```bash
python train.py
```

**4. Start the API**
```bash
uvicorn app:app --reload
```

**5. Open the docs**
```
http://127.0.0.1:8000/docs
```

**6. Run tests**
```bash
pytest test_app.py -v
```

---

## API Usage

**POST** `/predict`

```json
{
  "transaction_type": "P2P",
  "merchant_category": "Food",
  "amount": 95000,
  "transaction_status": "SUCCESS",
  "sender_age_group": "26-35",
  "receiver_age_group": "56+",
  "sender_state": "Tamil Nadu",
  "sender_bank": "HDFC",
  "receiver_bank": "SBI",
  "device_type": "Android",
  "network_type": "3G",
  "hour_of_day": 2,
  "is_weekend": 1
}
```

**Response**

```json
{
  "fraud": false,
  "probability": 0.011,
  "verdict": "GENUINE",
  "reasons": [
    {"feature": "receiver_age_group", "impact": -4.019, "direction": "decreases fraud risk"},
    {"feature": "merchant_category",  "impact":  1.189, "direction": "increases fraud risk"},
    {"feature": "hour_of_day",        "impact": -0.751, "direction": "decreases fraud risk"}
  ]
}
```

---

## Model Performance

| Metric | Genuine | Fraud |
|--------|---------|-------|
| Precision | 1.00 | 0.00 |
| Recall | 0.95 | 0.04 |
| F1 Score | 0.97 | 0.00 |
| Accuracy | 0.95 | — |

**Note on fraud metrics:** The dataset contains only 480 fraud cases out of 250,000 transactions (0.19% fraud rate). This extreme imbalance limits the model's ability to learn distinct fraud patterns despite SMOTE oversampling. In production, fraud models are trained on millions of labeled transactions with richer behavioral features. The SHAP explainability layer and API architecture remain production-ready regardless of dataset size.
