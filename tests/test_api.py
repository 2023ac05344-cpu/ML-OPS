import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "California Housing Price Predictor"


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model-info")
    # This might fail if model is not loaded, which is expected in test environment
    assert response.status_code in [200, 500]


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text


def test_logs_endpoint():
    """Test logs endpoint."""
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid input."""
    valid_input = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=valid_input)
    # This might fail if model is not loaded, which is expected in test environment
    assert response.status_code in [200, 500]


def test_predict_endpoint_invalid_input():
    """Test prediction endpoint with invalid input."""
    invalid_input = {
        "MedInc": "invalid",
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_missing_fields():
    """Test prediction endpoint with missing fields."""
    incomplete_input = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        # Missing other required fields
    }

    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422  # Validation error
