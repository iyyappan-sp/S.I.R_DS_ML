import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

SAMPLE = {
    "transaction_type":   "P2P",
    "merchant_category":  "Food",
    "amount":             5000,
    "transaction_status": "SUCCESS",
    "sender_age_group":   "26-35",
    "receiver_age_group": "26-35",
    "sender_state":       "Tamil Nadu",
    "sender_bank":        "HDFC",
    "receiver_bank":      "SBI",
    "device_type":        "Android",
    "network_type":       "4G",
    "hour_of_day":        14,
    "is_weekend":         0,
}


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_returns_verdict():
    res = client.post("/predict", json=SAMPLE)
    assert res.status_code == 200
    assert res.json()["verdict"] in ["FRAUD", "GENUINE"]


def test_predict_probability_range():
    res = client.post("/predict", json=SAMPLE)
    prob = res.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_returns_reasons():
    res = client.post("/predict", json=SAMPLE)
    reasons = res.json()["reasons"]
    assert len(reasons) == 3
    assert "feature" in reasons[0]
    assert "direction" in reasons[0]


def test_missing_field_returns_422():
    res = client.post("/predict", json={"amount": 5000})
    assert res.status_code == 422