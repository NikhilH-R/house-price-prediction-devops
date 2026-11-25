"""
Automated API Tests - House Price Prediction

Tests endpoint availability, request validation,
prediction responses, model availability, CORS,
and exception handling.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app, FEATURE_COLUMNS


# ==================================================
# 1. TEST DATA
# ==================================================

VALID_INPUT = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322.0,
    "AveOccup": 2.55,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


# ==================================================
# 2. TEST FIXTURES
# ==================================================

@pytest.fixture
def mock_model():
    """Create a deterministic mock prediction model."""

    model = Mock()

    model.predict.return_value = np.array(
        [452350.75]
    )

    return model


@pytest.fixture
def client(mock_model, monkeypatch):
    """Create an API test client with a mock model."""

    monkeypatch.setattr(
        app.state,
        "model",
        mock_model,
        raising=False,
    )

    with TestClient(app) as test_client:
        app.state.model = mock_model
        yield test_client


# ==================================================
# 3. GENERAL ENDPOINT TESTS
# ==================================================

def test_root_endpoint(client):
    """Verify that the root endpoint is available."""

    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()


def test_health_endpoint(client):
    """Verify model availability through the health API."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_swagger_documentation(client):
    """Verify that Swagger documentation is accessible."""

    response = client.get("/docs")

    assert response.status_code == 200


def test_openapi_schema(client):
    """Verify that the OpenAPI schema is available."""

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


# ==================================================
# 4. PREDICTION TESTS
# ==================================================

def test_valid_prediction(client):
    """Verify successful house price prediction."""

    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_price_usd"] == 452350.75
    assert "message" in result


def test_prediction_feature_order(client, mock_model):
    """Verify feature ordering passed to the model."""

    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 200

    mock_model.predict.assert_called_once()

    model_input = mock_model.predict.call_args.args[0]

    assert list(model_input.columns) == FEATURE_COLUMNS
    assert model_input.shape == (1, 8)


def test_prediction_response_type(client):
    """Verify the prediction response data type."""

    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 200

    assert isinstance(
        response.json()["predicted_price_usd"],
        (int, float),
    )


# ==================================================
# 5. INPUT VALIDATION TESTS
# ==================================================

@pytest.mark.parametrize(
    "field",
    FEATURE_COLUMNS,
)
def test_missing_required_field(client, field):
    """Reject requests with missing input features."""

    invalid_input = VALID_INPUT.copy()
    invalid_input.pop(field)

    response = client.post(
        "/predict",
        json=invalid_input,
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [
        ("MedInc", -1),
        ("HouseAge", -5),
        ("AveRooms", 0),
        ("AveBedrms", -1),
        ("Population", -100),
        ("AveOccup", 0),
        ("Latitude", 100),
        ("Latitude", -100),
        ("Longitude", 200),
        ("Longitude", -200),
        ("MedInc", "invalid"),
        ("HouseAge", None),
    ],
)
def test_invalid_input_values(client, field, value):
    """Reject invalid numerical and geographical values."""

    invalid_input = VALID_INPUT.copy()
    invalid_input[field] = value

    response = client.post(
        "/predict",
        json=invalid_input,
    )

    assert response.status_code == 422


def test_empty_request(client):
    """Reject an empty prediction request."""

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


# ==================================================
# 6. MODEL AVAILABILITY TESTS
# ==================================================

def test_health_when_model_unavailable(client):
    """Return HTTP 503 when the model is unavailable."""

    app.state.model = None

    response = client.get("/health")

    assert response.status_code == 503


def test_prediction_when_model_unavailable(client):
    """Reject predictions when the model is unavailable."""

    app.state.model = None

    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 503


# ==================================================
# 7. ERROR HANDLING TESTS
# ==================================================

def test_prediction_exception(client, mock_model):
    """Verify graceful handling of prediction failures."""

    mock_model.predict.side_effect = RuntimeError(
        "Simulated prediction failure"
    )

    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "Unable to complete the prediction"
    )


@pytest.mark.parametrize(
    "prediction",
    [
        -100.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_model_output(
    client,
    mock_model,
    prediction,
):
    """Reject invalid machine learning predictions."""

    mock_model.predict.return_value = np.array(
        [prediction]
    )

    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 500


# ==================================================
# 8. CORS CONFIGURATION TEST
# ==================================================

def test_allowed_cors_origin(client):
    """Verify requests from the local React frontend."""

    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 200

    assert response.headers.get(
        "access-control-allow-origin"
    ) == "http://localhost:5173"


def test_disallowed_cors_origin(client):
    """Verify that unapproved browser origins are rejected."""

    response = client.options(
        "/predict",
        headers={
            "Origin": "https://unapproved.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400


# ==================================================
# 9. HTTP METHOD VALIDATION
# ==================================================

def test_invalid_http_method(client):
    """Reject unsupported prediction HTTP methods."""

    response = client.get("/predict")

    assert response.status_code == 405


def test_unknown_endpoint(client):
    """Return HTTP 404 for an unknown API route."""

    response = client.get("/unknown-endpoint")

    assert response.status_code == 404
