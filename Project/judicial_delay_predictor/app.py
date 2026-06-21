import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Judicial Delay Predictor", version="1.0.0")

model    = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")

LABELS = {0: "Fast", 1: "Moderate", 2: "High", 3: "Severe"}
RANGES = {0: "< 1 year", 1: "1–3 years", 2: "3–5 years", 3: "> 5 years"}


class Case(BaseModel):
    state_code:        str
    dist_code:         str
    type_name:         str
    year:              int
    female_petitioner: Optional[int] = 0
    judge_position:    Optional[str] = "filled"


def prepare(case: Case):
    row = case.model_dump()
    for col, le in encoders.items():
        val = str(row.get(col, "unknown"))
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    return [[row.get(f, 0) for f in features]]


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost"}


@app.post("/predict")
def predict(case: Case):
    X    = prepare(case)
    prob = model.predict_proba(X)[0]
    cls  = int(np.argmax(prob))
    return {
        "delay_class": cls,
        "label":       LABELS[cls],
        "range":       RANGES[cls],
        "probability": round(float(prob[cls]), 2),
    }