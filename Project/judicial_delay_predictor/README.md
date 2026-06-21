# Judicial Delay Predictor

Predicts how long an Indian district court case will take to resolve — using 9 years of real court data.

**Live prediction via REST API. Built for portfolio.**

---

## The Problem

India has 45 million pending court cases. A civil suit filed in 2010 might still be unresolved in 2020. The question is — can we predict this at the time of filing?

This project answers that using the DevDataLab eCourts dataset: 81 million real cases from India's district courts (2010–2018).

---

## What It Predicts

Given basic case details at the time of filing, the model predicts one of 4 delay classes:

| Class | Label | Range |
|-------|-------|-------|
| 0 | Fast | < 1 year |
| 1 | Moderate | 1–3 years |
| 2 | High | 3–5 years |
| 3 | Severe | > 5 years |

---

## Key Findings

- Cases filed in earlier years show significantly higher median delays — the backlog compounds over time
- Court type is a strong predictor — district and sessions courts have the highest median delays
- Class imbalance is severe — most cases resolve fast, but severe delays are the real problem

---

## Tech Stack

- **Data** — pandas, NumPy
- **Model** — XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **API** — FastAPI, Uvicorn
- **Tests** — pytest

---

## Dataset

DevDataLab eCourts Dataset — 81M cases, 2010–2018
https://www.devdatalab.org/judicial-data

Licensed under Open Database License (ODbL). No personally identifiable information.

We sample 100,000 rows per year × 9 years = 900,000 rows for training.

---

## Project Files

```
judicial_delay_predictor/
├── merge_all_years.py   — samples and merges all 9 year files
├── train.py             — EDA, feature engineering, model training
├── app.py               — FastAPI REST API
├── test_app.py          — pytest tests
└── requirements.txt
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Place your data files in the project folder**
```
cases_2010.csv
cases_2011.csv
...
cases_2018.csv
```

**3. Merge all years**
```bash
python merge_all_years.py
```

**4. Train the model**
```bash
python train.py
```

**5. Start the API**
```bash
uvicorn app:app --reload
```

**6. Open the docs**
```
http://127.0.0.1:8000/docs
```

**7. Run tests**
```bash
pytest test_app.py -v
```

---

## API Usage

**POST** `/predict`

```json
{
  "state_code": "21",
  "dist_code": "1",
  "type_name": "682.0",
  "year": 2015,
  "female_petitioner": 0,
  "judge_position": "civil court"
}
```

**Response**

```json
{
  "delay_class": 0,
  "label": "Fast",
  "range": "< 1 year",
  "probability": 0.94
}
```

---

## Model Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Fast | 0.84 | 0.65 | 0.73 |
| Moderate | 0.44 | 0.39 | 0.42 |
| High | 0.26 | 0.31 | 0.28 |
| Severe | 0.25 | 0.73 | 0.38 |

Class imbalance (Fast >> Severe) is handled with SMOTE oversampling. The model prioritises recall on Severe cases — catching long delays matters more than false positives.
