import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from src.predict import HousePricePredictor, REQUIRED_FIELDS, NUMERIC_FIELDS, CATEGORICAL_FIELDS


@pytest.fixture
def valid_features():
    return {
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
        "Price_per_SqFt": 0.15,
    }


class TestValidateInput:
    def test_valid_input_no_errors(self, valid_features):
        p = HousePricePredictor()
        errors = p.validate_input(valid_features)
        assert errors == []

    def test_missing_required_field(self, valid_features):
        p = HousePricePredictor()
        for field in REQUIRED_FIELDS:
            features = valid_features.copy()
            del features[field]
            errors = p.validate_input(features)
            assert any(field in e for e in errors), f"Should fail for missing {field}"

    def test_none_required_field(self, valid_features):
        p = HousePricePredictor()
        features = valid_features.copy()
        features["State"] = None
        errors = p.validate_input(features)
        assert any("State" in e for e in errors)

    def test_empty_string_required_field(self, valid_features):
        p = HousePricePredictor()
        features = valid_features.copy()
        features["City"] = ""
        errors = p.validate_input(features)
        assert any("City" in e for e in errors)

    def test_numeric_out_of_range(self, valid_features):
        p = HousePricePredictor()
        features = valid_features.copy()
        features["BHK"] = 100
        errors = p.validate_input(features)
        assert any("BHK" in e for e in errors)

    def test_numeric_below_range(self, valid_features):
        p = HousePricePredictor()
        features = valid_features.copy()
        features["BHK"] = 0
        errors = p.validate_input(features)
        assert any("BHK" in e for e in errors)

    def test_non_numeric_value(self, valid_features):
        p = HousePricePredictor()
        features = valid_features.copy()
        features["Size_in_SqFt"] = "big"
        errors = p.validate_input(features)
        assert any("Size_in_SqFt" in e for e in errors)

    def test_invalid_categorical_value(self, valid_features):
        p = HousePricePredictor()
        features = valid_features.copy()
        features["Property_Type"] = "Skyscraper"
        errors = p.validate_input(features)
        assert any("Property_Type" in e for e in errors)

    def test_floor_exceeds_total(self, valid_features):
        p = HousePricePredictor()
        features = valid_features.copy()
        features["Floor_No"] = 20
        features["Total_Floors"] = 10
        errors = p.validate_input(features)
        assert any("Floor_No" in e for e in errors)

    def test_all_categorical_fields_valid(self, valid_features):
        p = HousePricePredictor()
        errors = p.validate_input(valid_features)
        for field in CATEGORICAL_FIELDS:
            assert not any(field in e for e in errors), f"{field} should be valid"


class TestPredictorInit:
    def test_default_model_dir(self):
        p = HousePricePredictor()
        assert p.model_dir.exists() or not p.model_dir.exists()

    def test_custom_model_dir(self, tmp_path):
        p = HousePricePredictor(model_dir=str(tmp_path))
        assert p.model_dir == tmp_path

    def test_initial_state(self):
        p = HousePricePredictor()
        assert p.models == {}
        assert p.scaler is None
        assert p.feature_columns == []


class TestPredictorLoad:
    def test_load_no_comparison(self, tmp_path):
        p = HousePricePredictor(model_dir=str(tmp_path))
        assert p.load() is False

    def test_load_with_comparison(self, tmp_path, valid_features):
        import json as json_mod
        from sklearn.preprocessing import MinMaxScaler
        comparison = {
            "feature_columns": ["col1", "col2"],
            "onehot_categories": {},
            "model_scores": {"test_model": 0.9},
            "models": {"test_model": {"test": {"r2": 0.9}}},
        }
        (tmp_path / "comparison.json").write_text(json_mod.dumps(comparison))

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        X_dummy = np.random.rand(10, 2)
        y_dummy = np.random.rand(10)
        model.fit(X_dummy, y_dummy)
        import pickle
        (tmp_path / "test_model_20260101.pkl").write_bytes(pickle.dumps(model))

        scaler = MinMaxScaler()
        scaler.fit(X_dummy)
        (tmp_path / "scaler.pkl").write_bytes(pickle.dumps(scaler))

        rates = {"__default__": {"median": 0.10}}
        (tmp_path / "city_rates.json").write_text(json_mod.dumps(rates))

        p = HousePricePredictor(model_dir=str(tmp_path))
        result = p.load()
        assert result is True
        assert "test_model" in p.models
        assert p.scaler is not None
