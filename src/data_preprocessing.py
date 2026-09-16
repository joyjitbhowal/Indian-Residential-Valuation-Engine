import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from src.utils import logger


class DataPreprocessor:
    BINARY_MAP = {"yes": 1, "no": 0}
    TARGET = "Price_in_Lakhs"

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        self.scaler = MinMaxScaler()
        self._fitted = False
        self._encoded_cols = []
        self._onehot_categories = {}

    def load(self, path: str = None) -> pd.DataFrame:
        p = path or self.data_path
        if not p:
            p = str(Path(__file__).parent.parent / "india_housing_prices.csv")
        self.df = pd.read_csv(p)
        logger.info(f"Loaded {len(self.df)} rows from {p}")
        return self.df

    @staticmethod
    def is_binary(obj_col: pd.Series) -> bool:
        return set(obj_col.dropna().str.lower().unique()).issubset({"yes", "no"})

    def fit_encode_categories(self, df: pd.DataFrame):
        self._onehot_categories = {}
        for col in df.select_dtypes(include=["object"]).columns:
            if not self.is_binary(df[col]):
                self._onehot_categories[col] = sorted(df[col].unique())
        logger.info(f"Fitted encoding for {len(self._onehot_categories)} multi-class columns")
        return self

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self._encoded_cols = []

        for col in df.select_dtypes(include=["object"]).columns:
            if self.is_binary(df[col]):
                df[col] = df[col].str.lower().map(self.BINARY_MAP)
                self._encoded_cols.append((col, "binary"))
            else:
                all_cats = self._onehot_categories.get(col, sorted(df[col].unique()))
                for cat in all_cats:
                    col_name = f"{col}_{cat}"
                    df[col_name] = (df[col].astype(str).str.lower() == str(cat).lower()).astype(int)
                    self._encoded_cols.append((col_name, "one-hot"))
                df = df.drop(columns=[col])
                if len(all_cats) > 2:
                    logger.info(f"Encoded '{col}' -> {len(all_cats)} one-hot columns")

        return df

    def clean(self, df: pd.DataFrame = None) -> pd.DataFrame:
        df = df if df is not None else self.df
        if df is None:
            raise ValueError("No data loaded")

        df = df.copy()
        df = df.drop_duplicates()

        if "Amenities" in df.columns:
            df["amenity_count"] = df["Amenities"].astype(str).str.split(",").str.len()

        drop_cols = {"ID", "Locality", "Amenities"}
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

        for col in df.select_dtypes(include=[np.number]).columns:
            if col == self.TARGET:
                continue
            q_low = df[col].quantile(0.002)
            q_high = df[col].quantile(0.998)
            df = df[(df[col] >= q_low) & (df[col] <= q_high)]

        df = df.dropna()
        logger.info(f"Clean: {len(df)} rows remaining")
        self.df = df
        return df

    def split(self, df: pd.DataFrame = None, target: str = None,
              test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        df = df if df is not None else self.df
        if df is None:
            raise ValueError("No data available")
        target_col = target or self.TARGET

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )

        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None,
              X_test: pd.DataFrame = None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self._fitted = True
        result = {"X_train": X_train_scaled}

        if X_val is not None:
            result["X_val"] = self.scaler.transform(X_val)
        if X_test is not None:
            result["X_test"] = self.scaler.transform(X_test)

        logger.info("Features scaled using MinMaxScaler")
        return result

    def transform(self, X: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Scaler not fitted yet. Call scale() first.")
        return self.scaler.transform(X)
