# TN Government Scheme Recommendation System

Recommends eligible Tamil Nadu government schemes to citizens based on their personal profile — using rule-based eligibility filtering combined with a Random Forest classifier.

---

## The Problem

Tamil Nadu has hundreds of government welfare schemes across agriculture, education, disability, housing, and more. Most citizens don't know which schemes they qualify for. This system takes a citizen's profile and returns the top 5 matching schemes ranked by relevance.

---

## How It Works

Two-stage approach:

**Stage 1 — Rule-based eligibility filter**
Hard rules check age, income, caste, gender, occupation, education, marital status, disability status, and district type. A citizen either passes or fails each scheme's criteria.

**Stage 2 — ML ranking**
A Random Forest classifier scores each eligible scheme by probability of relevance. The top 5 are returned ranked by match score.

---

## Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Decision Tree | 0.9672 | 0.9278 | 0.9953 |
| Random Forest | 0.9817 | 0.9576 | 0.9980 |

Random Forest was chosen for the final API. Class imbalance handled with SMOTE.

---

## Tech Stack

- **Data & EDA** — pandas, NumPy, matplotlib, seaborn
- **Model** — scikit-learn (Decision Tree, Random Forest), imbalanced-learn (SMOTE)
- **API** — Flask
- **Persistence** — joblib

---

## Project Files

```
tn_govt_scheme_recom_sys/
├── tn_govt_scheme_recom_sys.py   — EDA, training, model saving
├── app.py                        — Flask REST API
└── requirements.txt
```

---

## Dataset

Two CSV files used for training (not included in repo):

- `tn_citizen_dataset.csv` — 10,400 citizen profiles with age, income, caste, occupation, education, district, disability status
- `tn_schemes.csv` — 52 Tamil Nadu government schemes with eligibility rules and benefit details

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the model**
```bash
python tn_govt_scheme_recom_sys.py
```
This saves `scheme_project.pkl` in the same folder.

**3. Start the API**
```bash
python app.py
```

API runs at `http://localhost:5000`

---

## API Usage

**POST** `/recommend`

```json
{
  "age": 35,
  "gender": "Female",
  "caste": "BC",
  "occupation": "Farmer",
  "education": "10th Pass",
  "annual_income": 80000,
  "marital_status": "Married",
  "disability_status": "No",
  "district": "Madurai",
  "family_size": 4
}
```

**Response**

```json
{
  "status": "success",
  "total_found": 8,
  "recommendations": [
    {
      "scheme_name": "Uzhavar Suraksha Thittam",
      "category": "Agriculture",
      "benefit_amount": 50000,
      "benefit_type": "Insurance",
      "match_percentage": 87.4
    }
  ]
}
```

---

## Eligibility Rules Covered

- Age range (min and max)
- Annual income limit
- Caste (Any / BC / MBC / SC / ST / BC/MBC / SC/ST)
- Gender (Any / Male / Female / Transgender)
- Occupation (Any / Farmer / Fisher / Weaver etc.)
- Education minimum level
- Marital status
- Disability status
- District type (Rural / Urban / Coastal)
