from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_ml.config import Settings
from fraud_ml.features.build import FEATURE_COLUMNS
from fraud_ml.serve import create_app

ROOT = Path(__file__).resolve().parents[1]

LEGIT_PAYLOAD = {
    "transaction_id": "tx-legit",
    "amount": 42.50,
    "txn_count_1h": 0,
    "txn_count_24h": 3,
    "amount_zscore": 0.2,
    "seconds_since_last_txn": 21600,
    "is_new_merchant": 0,
    "geo_distance_km": 1.2,
    "is_night": 0,
}

CARD_TESTING_PAYLOAD = {
    "transaction_id": "tx-burst",
    "amount": 1.75,
    "txn_count_1h": 14,
    "txn_count_24h": 16,
    "amount_zscore": -0.9,
    "seconds_since_last_txn": 11,
    "is_new_merchant": 1,
    "geo_distance_km": 0.3,
    "is_night": 1,
}


@pytest.fixture(scope="module")
def client():
    settings = Settings(
        model_path=ROOT / "models" / "fraud_xgb.json",
        feature_spec_path=ROOT / "models" / "feature_spec.json",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == len(FEATURE_COLUMNS)


def test_score_response_shape(client: TestClient) -> None:
    response = client.post("/score", json=LEGIT_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "tx-legit"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"allow", "review"}
    assert len(body["top_features"]) == 3
    for item in body["top_features"]:
        assert item["feature"] in FEATURE_COLUMNS


def test_burst_scores_higher_than_legit(client: TestClient) -> None:
    legit = client.post("/score", json=LEGIT_PAYLOAD).json()
    burst = client.post("/score", json=CARD_TESTING_PAYLOAD).json()
    assert burst["fraud_probability"] > legit["fraud_probability"]


def test_rejects_invalid_amount(client: TestClient) -> None:
    response = client.post("/score", json={**LEGIT_PAYLOAD, "amount": -5.0})
    assert response.status_code == 422
