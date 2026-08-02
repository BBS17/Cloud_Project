"""
Tests for Fact Checker API endpoints.

conftest.py mocks app.final_model so torch/transformers are never loaded.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fact Checker API running"}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@patch("app.main.check_db")
def test_health_ok(mock_check_db):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.is_model_loaded", return_value=False)
def test_health_model_not_loaded_returns_503(mock_loaded):
    response = client.get("/health")
    assert response.status_code == 503
    assert "Model" in response.json()["detail"]


@patch("app.main.check_db", side_effect=Exception("connection refused"))
def test_health_db_down_returns_503(mock_check_db):
    response = client.get("/health")
    assert response.status_code == 503
    assert "Database" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@patch("app.main.predict_text")
@patch("app.main.log_inference")
def test_predict(mock_log, mock_predict):
    mock_predict.return_value = {"label": "Truth", "confidence": 95.0}

    response = client.post("/predict", json={"text": "The sky is blue."})

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Truth"
    assert data["confidence"] == 95.0
    mock_predict.assert_called_once_with("The sky is blue.")


@patch("app.main.predict_text")
@patch("app.main.log_inference")
def test_predict_misinformation(mock_log, mock_predict):
    mock_predict.return_value = {"label": "Misinformation", "confidence": 88.5}

    response = client.post("/predict", json={"text": "The earth is flat."})

    assert response.status_code == 200
    assert response.json()["label"] == "Misinformation"


@patch("app.main.predict_text")
def test_predict_empty_text_rejected(mock_predict):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    mock_predict.assert_not_called()


@patch("app.main.predict_text")
def test_predict_too_short_text_rejected(mock_predict):
    response = client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422
    mock_predict.assert_not_called()


@patch("app.main.predict_text")
def test_predict_oversized_text_rejected(mock_predict):
    response = client.post("/predict", json={"text": "a" * 5001})
    assert response.status_code == 422
    mock_predict.assert_not_called()


@patch("app.main.predict_text", side_effect=RuntimeError("model error"))
@patch("app.main.log_inference")
def test_predict_inference_error_returns_500(mock_log, mock_predict):
    response = client.post("/predict", json={"text": "Valid text here."})
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@patch("app.main.get_metrics_summary")
def test_metrics(mock_metrics):
    mock_metrics.return_value = {
        "total_requests": 10,
        "avg_latency_ms": 42.5,
        "min_latency_ms": 10.0,
        "max_latency_ms": 100.0,
    }
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["total_requests"] == 10


@patch("app.main.get_metrics_summary")
def test_metrics_db_unavailable_returns_503(mock_metrics):
    mock_metrics.return_value = None
    response = client.get("/metrics")
    assert response.status_code == 503
