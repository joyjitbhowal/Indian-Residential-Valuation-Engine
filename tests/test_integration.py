import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.scoring import compute_scores, multi_modal_ensemble


@pytest.fixture
def housing_df():
    np.random.seed(42)
    n = 200
    states = ["Maharashtra", "Karnataka", "Delhi"]
    cities = {"Maharashtra": ["Mumbai", "Pune"], "Karnataka": ["Bangalore"], "Delhi": ["New Delhi"]}
    state_choices = np.random.choice(states, n)
    city_choices = [np.random.choice(cities[s]) for s in state_choices]

    return pd.DataFrame({
        "State": state_choices,
        "City": city_choices,
        "Property_Type": np.random.choice(["Apartment", "Villa", "Independent House"], n),
        "BHK": np.random.randint(1, 6, n),
        "Size_in_SqFt": np.random.randint(300, 5000, n),
        "Price_in_Lakhs": np.random.randint(20, 500, n),
        "Price_per_SqFt": np.random.uniform(0.05, 0.5, n),
        "Year_Built": np.random.randint(1990, 2024, n),
        "Furnished_Status": np.random.choice(["Furnished", "Semi-furnished", "Unfurnished"], n),
        "Floor_No": np.random.randint(1, 20, n),
        "Total_Floors": np.random.randint(5, 30, n),
        "Age_of_Property": np.random.randint(0, 30, n),
        "Nearby_Schools": np.random.randint(0, 10, n),
        "Nearby_Hospitals": np.random.randint(0, 8, n),
        "Public_Transport_Accessibility": np.random.choice(["High", "Medium", "Low"], n),
        "Parking_Space": np.random.choice(["Yes", "No"], n),
        "Security": np.random.choice(["Yes", "No"], n),
        "Facing": np.random.choice(["North", "South", "East", "West"], n),
        "Owner_Type": np.random.choice(["Owner", "Builder", "Broker"], n),
        "Availability_Status": np.random.choice(["Ready_to_Move", "Under_Construction"], n),
    })


class TestEndToEndPreprocessing:
    def test_clean_encode_engineer_split(self, housing_df):
        dp = DataPreprocessor()
        cleaned = dp.clean(housing_df)
        assert len(cleaned) <= len(housing_df)
        assert not cleaned.isnull().any().any()

        dp.fit_encode_categories(cleaned)
        encoded = dp.encode_categoricals(cleaned)
        assert encoded.select_dtypes(include=["object"]).empty

        fe = FeatureEngineer()
        engineered = fe.fit_transform(encoded)
        assert engineered.shape[1] > housing_df.shape[1]

        target = "Price_in_Lakhs"
        X_train, X_val, X_test, y_train, y_val, y_test = dp.split(engineered, target=target)
        assert target not in X_train.columns
        assert len(X_train) > len(X_val) > 0
        assert len(X_test) > 0

        scaled = dp.scale(X_train, X_val, X_test)
        assert scaled["X_train"].shape == X_train.shape


class TestScoringIntegration:
    def test_compute_and_ensemble(self):
        metrics = {
            "xgboost": {"r2": 0.95, "mae": 8.5, "mape": 4.2},
            "lightgbm": {"r2": 0.94, "mae": 9.0, "mape": 4.5},
            "random_forest": {"r2": 0.93, "mae": 9.5, "mape": 4.8},
        }
        scores = compute_scores(metrics)
        assert len(scores) == 3
        assert all(0 < s < 1 for s in scores.values())

        preds = {"xgboost": 150.0, "lightgbm": 155.0, "random_forest": 148.0}
        ensemble = multi_modal_ensemble(preds, scores)
        assert ensemble["weighted_ensemble"] > 0
        assert ensemble["price_range"]["lower"] < ensemble["weighted_ensemble"]
        assert ensemble["price_range"]["upper"] > ensemble["weighted_ensemble"]
        assert ensemble["price_range"]["confidence"] in ("High", "Medium", "Low")
        assert abs(sum(ensemble["model_weights"].values()) - 1.0) < 0.01


class TestFeatureConsistency:
    def test_engineer_deterministic(self, housing_df):
        fe1 = FeatureEngineer()
        fe2 = FeatureEngineer()
        r1 = fe1.fit_transform(housing_df)
        r2 = fe2.fit_transform(housing_df)
        assert list(r1.columns) == list(r2.columns)
        assert r1.shape == r2.shape

    def test_encode_deterministic(self, housing_df):
        dp1 = DataPreprocessor()
        dp2 = DataPreprocessor()
        dp1.fit_encode_categories(housing_df)
        dp2.fit_encode_categories(housing_df)
        e1 = dp1.encode_categoricals(housing_df)
        e2 = dp2.encode_categoricals(housing_df)
        assert list(e1.columns) == list(e2.columns)
