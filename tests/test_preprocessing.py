import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def housing_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "State": np.random.choice(["Maharashtra", "Karnataka"], n),
        "City": np.random.choice(["Mumbai", "Pune", "Bangalore"], n),
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


class TestDataPreprocessor:
    def test_clean_removes_duplicates(self, housing_df):
        dp = DataPreprocessor()
        duped = pd.concat([housing_df, housing_df.iloc[:5]], ignore_index=True)
        cleaned = dp.clean(duped)
        assert len(cleaned) <= len(duped)

    def test_clean_handles_missing_values(self, housing_df):
        dp = DataPreprocessor()
        df = housing_df.copy()
        original_len = len(df)
        df.loc[0, "BHK"] = np.nan
        cleaned = dp.clean(df)
        assert len(cleaned) < original_len

    def test_split_returns_correct_shapes(self, housing_df):
        dp = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = dp.split(housing_df)
        total = len(housing_df)
        assert len(X_train) > len(X_val)
        assert len(X_test) > 0
        assert len(X_train) + len(X_val) + len(X_test) == total

    def test_split_custom_sizes(self, housing_df):
        dp = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = dp.split(
            housing_df, test_size=0.3, val_size=0.1
        )
        assert len(X_test) > len(X_val)

    def test_scale_maintains_shape(self, housing_df):
        dp = DataPreprocessor()
        dp.fit_encode_categories(housing_df)
        encoded = dp.encode_categoricals(housing_df)
        X_train, X_val, X_test, _, _, _ = dp.split(encoded)
        scaled = dp.scale(X_train, X_val, X_test)
        assert scaled["X_train"].shape[1] == X_train.shape[1]
        assert scaled["X_val"].shape[1] == X_val.shape[1]
        assert scaled["X_test"].shape[1] == X_test.shape[1]

    def test_encode_binary_columns(self, housing_df):
        dp = DataPreprocessor()
        dp.fit_encode_categories(housing_df)
        encoded = dp.encode_categoricals(housing_df)
        for col in ["Parking_Space", "Security"]:
            assert col in encoded.columns
            assert set(encoded[col].unique()).issubset({0, 1})

    def test_encode_onehot_columns(self, housing_df):
        dp = DataPreprocessor()
        dp.fit_encode_categories(housing_df)
        encoded = dp.encode_categoricals(housing_df)
        assert "Property_Type" not in encoded.columns
        assert any("Property_Type" in c for c in encoded.columns)

    def test_fit_encode_categories_stores_categories(self, housing_df):
        dp = DataPreprocessor()
        dp.fit_encode_categories(housing_df)
        assert "Property_Type" in dp._onehot_categories
        assert len(dp._onehot_categories["Property_Type"]) == 3

    def test_encode_with_consistent_categories(self, housing_df):
        dp = DataPreprocessor()
        dp.fit_encode_categories(housing_df)
        e1 = dp.encode_categoricals(housing_df.iloc[:10])
        e2 = dp.encode_categoricals(housing_df.iloc[10:20])
        assert list(e1.columns) == list(e2.columns)

    def test_split_preserves_target(self, housing_df):
        dp = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = dp.split(housing_df)
        assert y_train.name == "Price_in_Lakhs"
        assert y_val.name == "Price_in_Lakhs"
        assert y_test.name == "Price_in_Lakhs"


class TestFeatureEngineer:
    def test_fit_transform_returns_dataframe(self, housing_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(housing_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == housing_df.shape[0]

    def test_floor_features_created(self, housing_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(housing_df)
        assert "floor_ratio" in result.columns
        assert "is_top_floor" in result.columns
        assert "is_ground_floor" in result.columns

    def test_bhk_features_created(self, housing_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(housing_df)
        assert "sqft_per_bhk" in result.columns
        assert "bhk_x_size" in result.columns

    def test_amenity_features_created(self, housing_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(housing_df)
        assert "nearby_amenities_total" in result.columns
        assert "school_hospital_ratio" in result.columns

    def test_age_features_created(self, housing_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(housing_df)
        assert "is_new_property" in result.columns
        assert "is_old_property" in result.columns
        assert "age_squared" in result.columns

    def test_log_features_created(self, housing_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(housing_df)
        assert "log_size" in result.columns
        assert "log_price_per_sqft" in result.columns

    def test_target_excluded_from_features(self, housing_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(housing_df)
        feature_names = FeatureEngineer._get_feature_names(housing_df)
        assert "Price_in_Lakhs" not in feature_names

    def test_get_feature_names_excludes_target(self, housing_df):
        names = FeatureEngineer._get_feature_names(housing_df)
        assert "Price_in_Lakhs" not in names
        assert "BHK" in names

    def test_engineer_deterministic(self, housing_df):
        fe1 = FeatureEngineer()
        fe2 = FeatureEngineer()
        r1 = fe1.fit_transform(housing_df)
        r2 = fe2.fit_transform(housing_df)
        assert list(r1.columns) == list(r2.columns)
