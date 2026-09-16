import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_housing_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "State": np.random.choice(["Maharashtra", "Karnataka", "Delhi"], n),
        "City": np.random.choice(["Mumbai", "Pune", "Bangalore", "New Delhi"], n),
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


@pytest.fixture
def valid_prediction_input():
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
