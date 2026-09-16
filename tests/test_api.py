import pytest
import json
from unittest.mock import patch, MagicMock
from src.api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_predictor():
    with patch("src.api.predictor") as p:
        p.models = {"xgboost": MagicMock(), "lightgbm": MagicMock()}
        p.feature_columns = ["col1", "col2"]
        p.comparison_data = {
            "xgboost": {"test": {"r2": 0.95, "mae": 10, "rmse": 15, "mape": 5}},
            "lightgbm": {"test": {"r2": 0.93, "mae": 12, "rmse": 18, "mape": 6}},
        }
        p.model_scores = {"xgboost": 0.96, "lightgbm": 0.94}
        p.city_rates = {"__default__": {"median": 0.10}}
        p.validate_input.return_value = []
        p.predict.return_value = {
            "predictions": {"xgboost": 100.0, "lightgbm": 105.0},
            "ensemble": {
                "weighted_ensemble": 102.0,
                "price_range": {"lower": 90.0, "upper": 114.0, "confidence": "High"},
            },
        }
        yield p


class TestHealthEndpoint:
    def test_health_returns_ok(self, client, mock_predictor):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"

    def test_health_shows_models(self, client, mock_predictor):
        res = client.get("/health")
        data = res.get_json()
        assert "xgboost" in data["models_loaded"]


class TestPredictEndpoint:
    def test_predict_missing_json(self, client, mock_predictor):
        res = client.post("/predict", content_type="application/json")
        assert res.status_code == 400

    def test_predict_empty_features(self, client, mock_predictor):
        mock_predictor.validate_input.return_value = ["Missing required field: State"]
        res = client.post(
            "/predict",
            data=json.dumps({"features": {}}),
            content_type="application/json",
        )
        assert res.status_code == 400

    def test_predict_validation_error(self, client, mock_predictor):
        mock_predictor.validate_input.return_value = ["Missing required field: State"]
        res = client.post(
            "/predict",
            data=json.dumps({"features": {"City": "Mumbai"}}),
            content_type="application/json",
        )
        assert res.status_code == 400
        data = res.get_json()
        assert "details" in data

    def test_predict_success(self, client, mock_predictor):
        res = client.post(
            "/predict",
            data=json.dumps({
                "features": {
                    "State": "Maharashtra",
                    "City": "Mumbai",
                    "Property_Type": "Apartment",
                    "BHK": 3,
                    "Size_in_SqFt": 1200,
                    "Furnished_Status": "Semi-furnished",
                    "Floor_No": 5,
                    "Total_Floors": 15,
                    "Age_of_Property": 10,
                    "Year_Built": 2015,
                    "Nearby_Schools": 5,
                    "Nearby_Hospitals": 3,
                    "Public_Transport_Accessibility": "High",
                    "Parking_Space": "Yes",
                    "Security": "Yes",
                    "Facing": "East",
                    "Owner_Type": "Owner",
                    "Availability_Status": "Ready_to_Move",
                }
            }),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "predictions" in data
        assert "ensemble" in data
        assert "latency_ms" in data

    def test_predict_no_models(self, client):
        with patch("src.api.predictor", None):
            res = client.post(
                "/predict",
                data=json.dumps({"features": {"State": "Test"}}),
                content_type="application/json",
            )
            assert res.status_code == 503

    def test_predict_non_json_body(self, client, mock_predictor):
        res = client.post(
            "/predict",
            data="not json",
            content_type="text/plain",
        )
        assert res.status_code == 400

    def test_predict_features_not_dict(self, client, mock_predictor):
        res = client.post(
            "/predict",
            data=json.dumps({"features": ["not", "a", "dict"]}),
            content_type="application/json",
        )
        assert res.status_code == 400


class TestComparisonEndpoint:
    def test_comparison_returns_data(self, client, mock_predictor):
        res = client.get("/api/comparison")
        assert res.status_code == 200
        data = res.get_json()
        assert "xgboost" in data

    def test_comparison_no_models(self, client):
        with patch("src.api.predictor", None):
            res = client.get("/api/comparison")
            assert res.status_code == 503


class TestScoresEndpoint:
    def test_scores_returns_data(self, client, mock_predictor):
        res = client.get("/api/scores")
        assert res.status_code == 200

    def test_scores_no_models(self, client):
        with patch("src.api.predictor", None):
            res = client.get("/api/scores")
            assert res.status_code == 503


class TestRateEndpoint:
    def test_rate_requires_params(self, client, mock_predictor):
        res = client.get("/api/rate")
        assert res.status_code == 400

    def test_rate_requires_city(self, client, mock_predictor):
        res = client.get("/api/rate?state=Maharashtra")
        assert res.status_code == 400

    def test_rate_returns_value(self, client, mock_predictor):
        res = client.get("/api/rate?state=Maharashtra&city=Mumbai")
        assert res.status_code == 200
        data = res.get_json()
        assert "Price_per_SqFt" in data


class TestModelsEndpoint:
    def test_models_returns_list(self, client, mock_predictor):
        res = client.get("/models")
        assert res.status_code == 200
        data = res.get_json()
        assert data["count"] == 2

    def test_models_no_models(self, client):
        with patch("src.api.predictor", None):
            res = client.get("/models")
            assert res.status_code == 503


class TestErrorHandlers:
    def test_404(self, client):
        res = client.get("/nonexistent")
        assert res.status_code == 404

    def test_405(self, client, mock_predictor):
        res = client.delete("/predict")
        assert res.status_code == 405


class TestPageEndpoints:
    def test_index_page(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_predict_page(self, client):
        res = client.get("/predict", follow_redirects=True)
        assert res.status_code == 200
