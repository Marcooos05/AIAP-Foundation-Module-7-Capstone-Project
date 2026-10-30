import pytest
from fastapi.testclient import TestClient

from src.api import MAX_BATCH_SIZE, NOTICE, app, get_predictors
from src.predict import RiskPredictor


@pytest.fixture
def predictors(base_config, saved_artifacts):
    models_dir, reports_dir = saved_artifacts
    return {
        variant: RiskPredictor.load(variant, config=base_config, models_dir=models_dir, reports_dir=reports_dir)
        for variant in ("with_glucose", "without_glucose")
    }


@pytest.fixture
def client_with(predictors):
    """Builds a client with only the given variants loaded."""
    def _client(*variants):
        app.dependency_overrides[get_predictors] = lambda: {v: predictors[v] for v in variants}
        return TestClient(app)
    yield _client
    app.dependency_overrides.clear()


@pytest.fixture
def client(client_with):
    return client_with("with_glucose", "without_glucose")


class TestHealth:
    def test_ok_when_all_models_are_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "models": {"with_glucose": True, "without_glucose": True}}

    @pytest.mark.parametrize("loaded", [(), ("with_glucose",)])
    def test_503_when_a_model_is_missing(self, client_with, loaded):
        response = client_with(*loaded).get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"


class TestPredict:
    def test_defaults_to_with_glucose(self, client, valid_record):
        response = client.post("/predict", json={"records": [valid_record]})
        assert response.status_code == 200
        body = response.json()
        assert body["variant"] == "with_glucose"
        assert body["threshold"] == 0.4
        assert body["notice"] == NOTICE
        assert set(body["predictions"][0]) == {"risk_score", "flagged", "warnings"}

    def test_without_glucose_variant_accepts_no_glucose(self, client, valid_record):
        del valid_record["fasting_glucose_level"]
        response = client.post("/predict", json={"variant": "without_glucose", "records": [valid_record]})
        assert response.status_code == 200
        assert response.json()["model"] == "logistic_regression__without_glucose"

    def test_batch_keeps_one_result_per_record(self, client, valid_record):
        response = client.post("/predict", json={"records": [valid_record] * 3})
        assert len(response.json()["predictions"]) == 3

    def test_503_when_the_variant_is_not_loaded(self, client_with, valid_record):
        response = client_with("without_glucose").post("/predict", json={"records": [valid_record]})
        assert response.status_code == 503


class TestInvalidRequests:
    @pytest.mark.parametrize("change", [
        {"bmi": "high"},
        {"family_history_of_diabetes": "Unknown"},
        {"unexpected_field": 1},
        {"age": 16},
        {"fasting_glucose_level": None},
    ])
    def test_bad_record_returns_422(self, client, valid_record, change):
        response = client.post("/predict", json={"records": [{**valid_record, **change}]})
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client, valid_record):
        del valid_record["bmi"]
        assert client.post("/predict", json={"records": [valid_record]}).status_code == 422

    def test_nan_is_rejected(self, client, valid_record):
        body = '{"records": [{"age": NaN}]}'
        response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    def test_errors_do_not_echo_submitted_values(self, client, valid_record):
        response = client.post("/predict", json={"records": [{**valid_record, "bmi": "secret-value"}]})
        assert response.status_code == 422
        assert "secret-value" not in response.text

    def test_unknown_variant_returns_422(self, client, valid_record):
        assert client.post("/predict", json={"variant": "x", "records": [valid_record]}).status_code == 422

    @pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
    def test_batch_size_is_limited(self, client, valid_record, size):
        assert client.post("/predict", json={"records": [valid_record] * size}).status_code == 422
