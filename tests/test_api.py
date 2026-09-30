import os
import tempfile

os.environ["FLAGGED_DB"] = os.path.join(tempfile.mkdtemp(), "test_flagged.db")

from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["users_known"] > 0

def test_dashboard():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "UPI Fraud Monitor" in response.text

def test_score_transaction():
    with TestClient(app) as client:
        response = client.post("/score", json={
            "sender_id": "U0001",
            "receiver_id": "M0001",
            "amount": 5000,
            "location": "Mumbai",
            "device_type": "Android",
            "transaction_type": "P2P",
            "upi_channel": "GPay"
        })
        assert response.status_code == 200
        data = response.json()
        assert "fraud_score" in data
        assert "risk_level" in data
        assert "methods_flagged" in data
        assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

def test_alerts_endpoint():
    with TestClient(app) as client:
        response = client.get("/alerts")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

def test_invalid_amount():
    with TestClient(app) as client:
        response = client.post("/score", json={
            "sender_id": "U0001",
            "receiver_id": "M0001",
            "amount": -10
        })
        assert response.status_code == 422
