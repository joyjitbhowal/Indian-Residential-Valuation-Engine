import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.feature_engineering import FeatureEngineer
from src.data_preprocessing import DataPreprocessor
from src.scoring import compute_scores, multi_modal_ensemble
from src.city_rates import get_rate
from src.utils import logger

REQUIRED_FIELDS = [
    "State", "City", "Property_Type", "BHK", "Size_in_SqFt",
    "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property",
    "Year_Built", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Facing", "Owner_Type", "Availability_Status",
]

NUMERIC_FIELDS = {
    "BHK": (1, 10),
    "Size_in_SqFt": (50, 100000),
    "Floor_No": (1, 200),
    "Total_Floors": (1, 200),
    "Age_of_Property": (0, 200),
    "Year_Built": (1900, 2030),
    "Nearby_Schools": (0, 50),
    "Nearby_Hospitals": (0, 50),
    "Price_per_SqFt": (0.001, 10.0),
}

CATEGORICAL_FIELDS = {
    "Property_Type": ["Apartment", "Villa", "Independent House"],
    "Furnished_Status": ["Furnished", "Semi-furnished", "Unfurnished"],
    "Public_Transport_Accessibility": ["High", "Medium", "Low"],
    "Parking_Space": ["Yes", "No"],
    "Security": ["Yes", "No"],
    "Facing": ["North", "South", "East", "West"],
    "Owner_Type": ["Owner", "Builder", "Broker"],
    "Availability_Status": ["Ready_to_Move", "Under_Construction"],
}


class HousePricePredictor:
    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models"
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.comparison_data = {}
        self.model_scores = {}
        self.city_rates = {}
        self._onehot_categories = {}
        self.preprocessor = DataPreprocessor()
        self.fe = FeatureEngineer()

    def load(self) -> bool:
        comparison_path = self.model_dir / "comparison.json"
        if not comparison_path.exists():
            logger.error("No comparison.json found. Run compare.py first.")
            return False

        with open(comparison_path, "r") as f:
            cd = json.load(f)
            self.comparison_data = cd.get("models", {})
            self.feature_columns = cd.get("feature_columns", [])
            self.model_scores = cd.get("model_scores", {})
            self._onehot_categories = cd.get("onehot_categories", {})

        self.preprocessor._onehot_categories = self._onehot_categories

        for model_name in self.comparison_data:
            model_files = sorted(self.model_dir.glob(f"{model_name}_*.joblib"))
            if not model_files:
                model_files = sorted(self.model_dir.glob(f"{model_name}_*.pkl"))
            if model_files:
                self.models[model_name] = joblib.load(model_files[-1])
                logger.info(f"Loaded model: {model_name}")

        scaler_path = self.model_dir / "scaler.joblib"
        if not scaler_path.exists():
            scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        rates_path = self.model_dir / "city_rates.json"
        if rates_path.exists():
            with open(rates_path) as f:
                self.city_rates = json.load(f)
            logger.info(f"Loaded {len(self.city_rates)} city rates")

        logger.info(f"Loaded {len(self.models)} models, {len(self.feature_columns)} features")
        return len(self.models) > 0

    def validate_input(self, features: dict) -> list:
        errors = []
        for field in REQUIRED_FIELDS:
            if field not in features or features[field] is None or features[field] == "":
                errors.append(f"Missing required field: {field}")

        for field, (lo, hi) in NUMERIC_FIELDS.items():
            if field in features and features[field] is not None:
                try:
                    val = float(features[field])
                    if val < lo or val > hi:
                        errors.append(f"{field} must be between {lo} and {hi}, got {val}")
                except (TypeError, ValueError):
                    errors.append(f"{field} must be a number, got {type(features[field]).__name__}")

        for field, valid_values in CATEGORICAL_FIELDS.items():
            if field in features and features[field] is not None:
                if features[field] not in valid_values:
                    errors.append(f"{field} must be one of {valid_values}, got '{features[field]}'")

        if "Floor_No" in features and "Total_Floors" in features:
            try:
                if int(features["Floor_No"]) > int(features["Total_Floors"]):
                    errors.append("Floor_No cannot exceed Total_Floors")
            except (TypeError, ValueError):
                pass

        return errors

    def prepare_features(self, raw: dict) -> np.ndarray:
        df = pd.DataFrame([raw])

        if "Price_per_SqFt" not in raw or raw.get("Price_per_SqFt") is None:
            state = raw.get("State", "")
            city = raw.get("City", "")
            rate = get_rate(state, city, self.city_rates)
            df["Price_per_SqFt"] = rate

        df = self.preprocessor.encode_categoricals(df)
        df = self.fe._engineer(df)

        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]

        X = self.scaler.transform(df) if self.scaler else df.values
        return X

    def predict(self, features: dict) -> dict:
        X = self.prepare_features(features)
        preds = {}
        for name, model in self.models.items():
            preds[name] = round(float(model.predict(X)[0]), 2)

        test_metrics = {
            n: self.comparison_data.get(n, {}).get("test", {})
            for n in self.models
        }
        scores = compute_scores(
            {n: test_metrics[n] for n in self.models if test_metrics[n]}
        )
        ensemble = multi_modal_ensemble(preds, scores)
        return {"predictions": preds, "ensemble": ensemble}

    def predict_batch(self, records: list) -> list:
        results = []
        for r in records:
            try:
                results.append(self.predict(r))
            except Exception as e:
                logger.exception("Batch prediction failed")
                results.append({"error": "Prediction failed for one or more records"})
        return results
