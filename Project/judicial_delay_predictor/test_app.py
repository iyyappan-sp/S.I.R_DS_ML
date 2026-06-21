import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_returns_label():
    res = client.post("/predict", json={
        "state_code": "21",
        "dist_code":  "1",
        "type_name":  "682.0",
        "year":       2015,
    })
    assert res.status_code == 200
    assert res.json()["label"] in ["Fast", "Moderate", "High", "Severe"]


def test_predict_probability_between_0_and_1():
    res = client.post("/predict", json={
        "state_code": "7",
        "dist_code":  "2",
        "type_name":  "929.0",
        "year":       2013,
    })
    assert res.status_code == 200
    prob = res.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_missing_required_field_returns_422():
    res = client.post("/predict", json={"year": 2015})
    assert res.status_code == 422