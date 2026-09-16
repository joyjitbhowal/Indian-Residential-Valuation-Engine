import numpy as np
import pandas as pd
from src.utils import logger


class FeatureEngineer:
    def __init__(self):
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit() first.")
        return self._engineer(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def _get_feature_names(df: pd.DataFrame) -> list:
        exclude = {"Price_in_Lakhs", "price", "ID", "id", "date", "Locality"}
        return [c for c in df.columns if c not in exclude]

    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Floor_No" in df.columns and "Total_Floors" in df.columns:
            total = df["Total_Floors"].replace(0, 1)
            df["floor_ratio"] = (df["Floor_No"] / total).round(3)
            df["is_top_floor"] = (df["Floor_No"] == df["Total_Floors"]).astype(int)
            df["is_ground_floor"] = (df["Floor_No"] == 1).astype(int)
            df["is_mid_floor"] = ((df["Floor_No"] > 1) & (df["Floor_No"] < df["Total_Floors"])).astype(int)

        if "BHK" in df.columns and "Size_in_SqFt" in df.columns:
            bhk_safe = df["BHK"].replace(0, 1)
            df["sqft_per_bhk"] = (df["Size_in_SqFt"] / bhk_safe).round(2)
            df["bhk_x_size"] = (df["BHK"] * df["Size_in_SqFt"]) / 1000

        if "Nearby_Schools" in df.columns and "Nearby_Hospitals" in df.columns:
            df["nearby_amenities_total"] = df["Nearby_Schools"] + df["Nearby_Hospitals"]
            df["school_hospital_ratio"] = (
                (df["Nearby_Schools"] + 1) / (df["Nearby_Hospitals"] + 1)
            ).round(2)

        if "Age_of_Property" in df.columns:
            df["is_new_property"] = (df["Age_of_Property"] <= 5).astype(int)
            df["is_old_property"] = (df["Age_of_Property"] > 20).astype(int)
            df["age_squared"] = (df["Age_of_Property"] ** 2) / 100

        if "Year_Built" in df.columns:
            df["decade_built"] = (df["Year_Built"] // 10) * 10

        if "Size_in_SqFt" in df.columns:
            df["log_size"] = np.log1p(df["Size_in_SqFt"])

        if "Price_per_SqFt" in df.columns:
            df["log_price_per_sqft"] = np.log1p(df["Price_per_SqFt"] * 100)
            df["size_x_pps"] = (df["Size_in_SqFt"] * df["Price_per_SqFt"]) / 1000

        if "BHK" in df.columns and "bathrooms" not in df.columns:
            df["bhk_squared"] = df["BHK"] ** 2

        drop_cols = {"Floor_No", "Total_Floors"}
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

        logger.info(f"Feature engineering: {df.shape[1]} features")
        return df
